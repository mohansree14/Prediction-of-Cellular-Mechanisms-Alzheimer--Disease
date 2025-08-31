"""
DIG-Ensemble Training Script

This script trains the Deeply Integrated Graph Ensemble (DIG-Ensemble) model
and saves all results to the results folder with proper organization.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml

# Add project root to path (same as other training scripts)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import json
import logging
from datetime import datetime
from torch_geometric.data import DataLoader
from src.models.deep_learning.dig_ensemble import create_dig_ensemble_model, DIGEnsembleTrainer, evaluate_dig_ensemble, DIGEnsembleVisualizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# =====================
# 1. Setup Results Directory
# =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path(f"results/11_train_dig_ensemble_{timestamp}")
results_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(results_dir / "training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Results will be saved to: {results_dir}")

# =====================
# 2. Load Your Data
# =====================
# Replace these with your actual data loading logic
# Each dataset should be a list of torch_geometric.data.Data objects
# Each Data object must have .x (features), .edge_index (graph), and .y (labels)

def load_datasets():
    """
    Load datasets for DIG-Ensemble training using the same pattern as existing scripts.
    Creates graph structures from single-cell RNA data.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from torch_geometric.data import Data
    import torch
    
    logger.info("Loading datasets for DIG-Ensemble training...")
    
    # Load data using the same pattern as existing scripts
    data_path = 'data/processed/combined_data_final_features.csv'
    logger.info(f"Loading data from: {data_path}")
    
    # Load main data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Strip whitespace from columns
    data.columns = data.columns.str.strip()
    
    # Identify feature columns (same as in existing scripts)
    metadata_cols = set([
        'orig.ident', 'RNA_snn_res.0.8', 'seurat_clusters', 'marker_cell_type', 'scina_cell_type',
        'predicted.id', 'barcode', 'group', 'RNA_snn_res.4', 'cluster_group', 'healhy_cells_percent',
        'combine_group_pathlogy', 'associate_cells', 'data_id', 'RNA_snn_res.1.5'
    ])
    
    # Extract gene expression features
    gene_columns = [col for col in data.columns 
                   if col not in metadata_cols and 
                   str(data[col].dtype) in ['float64', 'int64', 'float32', 'int32']]
    
    logger.info(f"Using {len(gene_columns)} gene expression features")
    
    # Extract features and labels
    features = data[gene_columns].values
    logger.info(f"Feature matrix shape: {features.shape}")
    
    # Handle missing values
    features = np.nan_to_num(features, nan=0.0)
    
    # Identify target column (cell type or cluster)
    target_col = None
    for col in ['cell_type', 'seurat_clusters', 'CellType', 'cluster', 'marker_cell_type']:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError('No cell type or cluster column found in data!')
    
    # Prepare labels
    labels = data[target_col].astype(str)
    labels = labels.replace(['nan', 'None', 'unknown', 'Unknown'], 'Unlabeled')

    # === CLASS BALANCING: Remove rare classes ===
    class_counts = labels.value_counts()
    print("Class distribution before filtering:", class_counts.to_dict())
    min_samples_per_class = 20  # You can adjust this threshold
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    mask = labels.isin(valid_classes)
    features = features[mask]
    labels = labels[mask]
    class_counts = labels.value_counts()
    print("Class distribution after filtering:", class_counts.to_dict())

    # === FEATURE SELECTION: Use Random Forest importance ===
    from sklearn.ensemble import RandomForestClassifier
    # Use the full dataset for feature selection
    features_subset = features
    labels_subset = labels

    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)  # Fewer trees for speed
    rf.fit(features_subset, labels_subset)
    importances = rf.feature_importances_
    N = 30  # Number of top features to select
    indices = np.argsort(importances)[::-1][:N]
    features_selected = features[:, indices]
    print(f"Selected top {N} features based on importance.")
    features = features_selected
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    logger.info(f"Final class distribution: {dict(zip(le.classes_, np.bincount(encoded_labels)))}")
    
    # IMPROVEMENT 3: Better feature selection
    if features.shape[1] < 50:
        # If we have few features, create additional statistical features
        logger.info("Creating additional statistical features...")
        
        # Add statistical features
        mean_features = np.mean(features, axis=1, keepdims=True)
        std_features = np.std(features, axis=1, keepdims=True)
        max_features = np.max(features, axis=1, keepdims=True)
        min_features = np.min(features, axis=1, keepdims=True)
        median_features = np.median(features, axis=1, keepdims=True)
        
        # Add polynomial features for important genes
        if features.shape[1] > 0:
            # Square of top features
            top_feature_idx = np.argmax(np.var(features, axis=0))
            squared_features = (features[:, top_feature_idx:top_feature_idx+1]) ** 2
            
            # Interaction features
            if features.shape[1] > 1:
                interaction_features = features[:, 0:1] * features[:, 1:2]
            else:
                interaction_features = np.zeros((features.shape[0], 1))
            
            # Combine all features
            features = np.hstack([
                features, mean_features, std_features, max_features, 
                min_features, median_features, squared_features, interaction_features
            ])
        
        logger.info(f"Enhanced feature matrix shape: {features.shape}")
    
    # IMPROVEMENT 4: Better sample size management
    # max_samples = 5000  # Increased from 2000
    # if len(features) > max_samples:
    #     # Stratified sampling to maintain class balance
    #     from sklearn.model_selection import StratifiedShuffleSplit
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=len(features)-max_samples, random_state=42)
    #     for train_idx, _ in sss.split(features, encoded_labels):
    #         features = features[train_idx]
    #         encoded_labels = encoded_labels[train_idx]
    #         break
    #     logger.info(f"Limited to {max_samples} samples using stratified sampling")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # IMPROVEMENT 5: Better data splitting with stratification
    logger.info("Splitting data into train/validation/test sets...")
    
    # First split: train vs temp (70% train, 30% temp)
    train_idx, temp_idx = train_test_split(
        range(len(features_scaled)), 
        test_size=0.3, 
        random_state=42, 
        stratify=encoded_labels
    )
    
    # Second split: validation vs test (15% val, 15% test)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        random_state=42, 
        stratify=encoded_labels[temp_idx]
    )
    
    def create_graph_for_split(features_subset, split_name):
        """Create graph structure for a specific data split using faiss for fast k-NN search."""
        logger.info(f"Creating graph structure for {split_name} split (using faiss)...")
        n_neighbors = min(5, len(features_subset) - 1)  # Fewer neighbors for speed
        if n_neighbors < 1:
            logger.warning(f"{split_name} split has too few samples for k-NN graph. Creating self-loops only.")
            edge_index = torch.arange(len(features_subset)).repeat(2, 1)
        else:
            try:
                import faiss
            except ImportError:
                raise ImportError("faiss is required for fast k-NN search. Please install with 'pip install faiss-cpu'.")
            features_np = features_subset.astype(np.float32)
            index = faiss.IndexFlatL2(features_np.shape[1])
            index.add(features_np)
            distances, indices = index.search(features_np, n_neighbors)
            # Exclude self-connections (first neighbor is self)
            sources = np.repeat(np.arange(indices.shape[0]), n_neighbors - 1)
            targets = indices[:, 1:].reshape(-1)
            edge_index_np = np.vstack([
                np.concatenate([sources, targets]),
                np.concatenate([targets, sources])
            ])
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        logger.info(f"Created {split_name} graph with {edge_index.shape[1]} edges")
        return edge_index
    
    # Create PyTorch Geometric Data objects with separate graph structures
    logger.info("Creating PyTorch Geometric Data objects...")
    
    # Training data
    train_features = torch.tensor(features_scaled[train_idx], dtype=torch.float)
    train_labels = torch.tensor(encoded_labels[train_idx], dtype=torch.long)
    train_edge_index = create_graph_for_split(features_scaled[train_idx], "training")
    train_data = Data(x=train_features, edge_index=train_edge_index, y=train_labels)
    
    # Validation data
    val_features = torch.tensor(features_scaled[val_idx], dtype=torch.float)
    val_labels = torch.tensor(encoded_labels[val_idx], dtype=torch.long)
    val_edge_index = create_graph_for_split(features_scaled[val_idx], "validation")
    val_data = Data(x=val_features, edge_index=val_edge_index, y=val_labels)
    
    # Test data
    test_features = torch.tensor(features_scaled[test_idx], dtype=torch.float)
    test_labels = torch.tensor(encoded_labels[test_idx], dtype=torch.long)
    test_edge_index = create_graph_for_split(features_scaled[test_idx], "test")
    test_data = Data(x=test_features, edge_index=test_edge_index, y=test_labels)
    
    # Create datasets (lists of Data objects)
    train_dataset = [train_data]
    val_dataset = [val_data]
    test_dataset = [test_data]
    
    logger.info(f"Dataset sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    logger.info(f"Number of classes: {len(le.classes_)}")
    logger.info(f"Feature dimension: {features_scaled.shape[1]}")
    
    return train_dataset, val_dataset, test_dataset

logger.info("Loading datasets...")
train_dataset, val_dataset, test_dataset = load_datasets()

if len(train_dataset) == 0:
    raise ValueError("train_dataset is empty. Please load your data in load_datasets().")

logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# =====================
# 3. Model Parameters
# =====================
input_dim = train_dataset[0].x.shape[1]
num_classes = int(train_dataset[0].y.max().item() + 1)

# Training parameters (debug settings)
training_params = {
    'input_dim': input_dim,
    'num_classes': num_classes,
    'hidden_dim': 256,
    'gnn_layers': 4,
    'num_trees': 15,
    'tree_depth': 6,
    'gnn_type': 'gat',
    'dropout': 0.2,
    'batch_size': 64,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'epochs': 10,  # Fewer epochs for debugging
    'early_stopping_patience': 3  # Short patience for debugging
}

# Save parameters
with open(results_dir / "training_parameters.json", 'w') as f:
    json.dump(training_params, f, indent=2)

logger.info(f"Model parameters: {training_params}")

# =====================
# 4. Instantiate Model
# =====================
logger.info("Creating DIG-Ensemble model...")
model = create_dig_ensemble_model(
    input_dim=training_params['input_dim'],
    num_classes=training_params['num_classes'],
    hidden_dim=training_params['hidden_dim'],
    gnn_layers=training_params['gnn_layers'],
    num_trees=training_params['num_trees'],
    tree_depth=training_params['tree_depth'],
    gnn_type=training_params['gnn_type']
)

# =====================
# 5. DataLoaders
# =====================
logger.info("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'])

# =====================
# 6. Train the Model
# =====================
logger.info("Starting single debug training run...")

lr = training_params['learning_rate']
bs = training_params['batch_size']
hd = training_params['hidden_dim']

model = create_dig_ensemble_model(
    input_dim=training_params['input_dim'],
    num_classes=training_params['num_classes'],
    hidden_dim=hd,
    gnn_layers=training_params['gnn_layers'],
    num_trees=training_params['num_trees'],
    tree_depth=training_params['tree_depth'],
    gnn_type=training_params['gnn_type']
)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs)
test_loader = DataLoader(test_dataset, batch_size=bs)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_dataset[0].y.numpy()),
    y=train_dataset[0].y.numpy()
)
class_weights = torch.FloatTensor(class_weights)

trainer = DIGEnsembleTrainer(
    model, 
    learning_rate=lr,
    weight_decay=training_params['weight_decay'],
    class_weights=class_weights  # Pass class weights for imbalanced data
)

history = trainer.train(
    train_loader,
    val_loader,
    epochs=training_params['epochs'],
    early_stopping_patience=training_params['early_stopping_patience']
)

metrics = evaluate_dig_ensemble(model, test_loader)
acc = metrics['accuracy']
logger.info(f"Test accuracy for lr={lr}, batch_size={bs}, hidden_dim={hd}: {acc}")

best_acc = acc
best_params = {'learning_rate': lr, 'batch_size': bs, 'hidden_dim': hd}
best_metrics = metrics
best_model_state = model.state_dict()
with open(results_dir / "training_history.json", 'w') as f:
    json.dump(history, f, indent=2)

# Save the best model
model_path = results_dir / "dig_ensemble_model.pth"
torch.save(best_model_state, model_path)
logger.info(f"Best model saved to: {model_path}")

# Save best evaluation metrics
with open(results_dir / "evaluation_results.json", 'w') as f:
    json.dump(best_metrics, f, indent=2)

logger.info(f"Best Test Metrics: {best_metrics}")
logger.info(f"Best Hyperparameters: {best_params}")

# =====================
# 9. Generate Summary Report
# =====================
summary_report = f"""
DIG-Ensemble Training Summary
============================

Best Hyperparameters:
- Learning Rate: {best_params['learning_rate']}
- Batch Size: {best_params['batch_size']}
- Hidden Dimension: {best_params['hidden_dim']}

Training Parameters:
- Input Dimension: {training_params['input_dim']}
- Number of Classes: {training_params['num_classes']}
- Hidden Dimension: {best_params['hidden_dim']}
- GNN Layers: {training_params['gnn_layers']}
- Number of Trees: {training_params['num_trees']}
- Tree Depth: {training_params['tree_depth']}
- GNN Type: {training_params['gnn_type']}
- Batch Size: {best_params['batch_size']}
- Learning Rate: {best_params['learning_rate']}
- Epochs: {training_params['epochs']}

Dataset Sizes:
- Training: {train_dataset[0].x.shape[0]}
- Validation: {val_dataset[0].x.shape[0]}
- Test: {test_dataset[0].x.shape[0]}

Final Test Metrics:
- Overall Accuracy: {best_metrics['accuracy']:.4f}
- F1 Macro: {best_metrics['f1_macro']:.4f}
- F1 Weighted: {best_metrics['f1_weighted']:.4f}
- GNN Component Accuracy: {best_metrics['gnn_accuracy']:.4f}
- Forest Component Accuracy: {best_metrics['forest_accuracy']:.4f}
- Alpha Value: {best_metrics['alpha_value']:.4f}
- Attention Gate Mean: {best_metrics['attention_gate_mean']:.4f}
- Attention Gate Std: {best_metrics['attention_gate_std']:.4f}

Model saved to: {model_path}
Results directory: {results_dir}
"""

with open(results_dir / "training_summary.txt", 'w') as f:
    f.write(summary_report)

logger.info("Training summary saved!")

# =====================
# 10. Save Data Info
# =====================
data_info = {
    'train_samples': train_dataset[0].x.shape[0],
    'val_samples': val_dataset[0].x.shape[0],
    'test_samples': test_dataset[0].x.shape[0],
    'input_dim': input_dim,
    'num_classes': num_classes,
    'feature_names': [f'feature_{i}' for i in range(input_dim)] if input_dim <= 100 else None
}

with open(results_dir / "data_info.json", 'w') as f:
    json.dump(data_info, f, indent=2)

logger.info("DIG-Ensemble training completed successfully!")
logger.info(f"All results saved to: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIG Ensemble Model Training")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Use config['data_path'] and config['deep_learning_dir'] as needed 