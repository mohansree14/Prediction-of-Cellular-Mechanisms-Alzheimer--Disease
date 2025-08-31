import sys
from pathlib import Path
import traceback
import argparse
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=== DIG-Ensemble Training Test (Verbose) ===")

try:
    print("1. Importing libraries...")
    import torch
    import torch_geometric
    from src.models.deep_learning.dig_ensemble import create_dig_ensemble_model, DIGEnsembleTrainer, evaluate_dig_ensemble
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from torch_geometric.data import Data, DataLoader
    from sklearn.utils.class_weight import compute_class_weight
    print("✓ All imports successful")

    print("\n2. Loading data...")
    parser = argparse.ArgumentParser(description="DIG-Ensemble Training Test (Verbose)")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config['data_path']
    data = pd.read_csv(data_path, low_memory=False)
    print(f"✓ Data loaded: {data.shape}")

    print("\n3. Processing data...")
    # Strip whitespace from columns
    data.columns = data.columns.str.strip()
    
    # Identify feature columns
    metadata_cols = set([
        'orig.ident', 'RNA_snn_res.0.8', 'seurat_clusters', 'marker_cell_type', 'scina_cell_type',
        'predicted.id', 'barcode', 'group', 'RNA_snn_res.4', 'cluster_group', 'healhy_cells_percent',
        'combine_group_pathlogy', 'associate_cells', 'data_id', 'RNA_snn_res.1.5'
    ])
    
    # Extract gene expression features
    gene_columns = [col for col in data.columns 
                   if col not in metadata_cols and 
                   str(data[col].dtype) in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"✓ Found {len(gene_columns)} gene expression features")
    
    # Extract features and labels
    features = data[gene_columns].values
    features = np.nan_to_num(features, nan=0.0)
    
    # Identify target column
    target_col = 'seurat_clusters'  # Use known column
    labels = data[target_col].astype(str)
    labels = labels.replace(['nan', 'None', 'unknown', 'Unknown'], 'Unlabeled')
    
    # Class filtering - simplified
    class_counts = labels.value_counts()
    valid_classes = class_counts[class_counts >= 10].head(10).index  # Only top 10 classes
    mask = labels.isin(valid_classes)
    features = features[mask]
    labels = labels[mask]
    
    print(f"✓ Filtered to {features.shape[0]} samples, {len(valid_classes)} classes")
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Sample size management - increased for better training
    max_samples = 2000
    if len(features) > max_samples:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(features)-max_samples, random_state=42)
        for train_idx, _ in sss.split(features, encoded_labels):
            features = features[train_idx]
            encoded_labels = encoded_labels[train_idx]
            break
        print(f"✓ Limited to {max_samples} samples")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Simple data splitting
    train_idx, temp_idx = train_test_split(range(len(features_scaled)), test_size=0.3, random_state=42, stratify=encoded_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=encoded_labels[temp_idx])
    
    print(f"✓ Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    print("\n4. Creating graph structures...")
    def create_simple_graph(features_subset, split_name):
        print(f"  Creating {split_name} graph...")
        n_neighbors = min(10, len(features_subset) - 1)  # Increased neighbors
        
        if n_neighbors < 1:
            edge_index = torch.tensor([[i, i] for i in range(len(features_subset))], dtype=torch.long).t().contiguous()
        else:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            nbrs.fit(features_subset)
            distances, indices = nbrs.kneighbors(features_subset)
            
            edge_index = []
            for i, neighbors in enumerate(indices):
                for j in neighbors[1:]:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        print(f"  ✓ {split_name} graph: {edge_index.shape[1]} edges")
        return edge_index

    # Create PyTorch Geometric Data objects
    train_features = torch.tensor(features_scaled[train_idx], dtype=torch.float)
    train_labels = torch.tensor(encoded_labels[train_idx], dtype=torch.long)
    train_edge_index = create_simple_graph(features_scaled[train_idx], "training")
    train_data = Data(x=train_features, edge_index=train_edge_index, y=train_labels)
    
    val_features = torch.tensor(features_scaled[val_idx], dtype=torch.float)
    val_labels = torch.tensor(encoded_labels[val_idx], dtype=torch.long)
    val_edge_index = create_simple_graph(features_scaled[val_idx], "validation")
    val_data = Data(x=val_features, edge_index=val_edge_index, y=val_labels)
    
    test_features = torch.tensor(features_scaled[test_idx], dtype=torch.float)
    test_labels = torch.tensor(encoded_labels[test_idx], dtype=torch.long)
    test_edge_index = create_simple_graph(features_scaled[test_idx], "test")
    test_data = Data(x=test_features, edge_index=test_edge_index, y=test_labels)
    
    train_dataset = [train_data]
    val_dataset = [val_data]
    test_dataset = [test_data]
    
    print("✓ Graph structures created")

    print("\n5. Creating model...")
    input_dim = train_dataset[0].x.shape[1]
    num_classes = int(train_dataset[0].y.max().item() + 1)
    
    print(f"  Input dim: {input_dim}, Num classes: {num_classes}")
    
    model = create_dig_ensemble_model(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=64,  # Increased for better capacity
        gnn_layers=3,
        num_trees=5,
        tree_depth=4,
        gnn_type='gat'
    )
    print("✓ Model created")

    print("\n6. Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    print("✓ Data loaders created")

    print("\n7. Setting up trainer...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_dataset[0].y.numpy()),
        y=train_dataset[0].y.numpy()
    )
    class_weights = torch.FloatTensor(class_weights)
    
    trainer = DIGEnsembleTrainer(
        model, 
        learning_rate=0.001,
        weight_decay=1e-4,
        class_weights=class_weights
    )
    print("✓ Trainer created")

    print("\n8. Starting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=50,  # Increased epochs to see improvement
        early_stopping_patience=10
    )
    print("✓ Training completed!")

    print("\n9. Evaluating model...")
    metrics = evaluate_dig_ensemble(model, test_loader)
    print("✓ Evaluation completed!")
    print(f"Final metrics: {metrics}")

    print("\n=== Test completed successfully! ===")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIG-Ensemble Training Test (Verbose)")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Use config['data_path'] as needed 