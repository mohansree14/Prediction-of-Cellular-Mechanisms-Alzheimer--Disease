#!/usr/bin/env python3
"""
Hybrid Modeling Pipeline for Single-Cell RNA Analysis

This script implements a comprehensive hybrid modeling approach combining:
1. Graph Neural Networks (GNN) for spatial transcriptomics analysis
2. Conditional Variational Autoencoders (CVAE) for generative modeling

The pipeline includes:
- Data preparation and preprocessing
- Model training and validation
- Performance evaluation and comparison
- Result visualization and interpretation
- Synthetic data generation

Author: Dissertation Research
Date: 2024
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import warnings
import logging
import yaml
from datetime import datetime
import json
import argparse

# Ensure logs directory exists
Path('results/logs').mkdir(parents=True, exist_ok=True)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our custom models
from models.deep_learning.gnn_model import (
    SpatialGraphConstructor, 
    SpatialGNN, 
    SpatialGNNTrainer,
    create_spatial_gnn_model
)
from models.deep_learning.generative_model import (
    ConditionalVAE,
    CVAETrainer,
    SingleCellDataProcessor,
    create_cvae_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/hybrid_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class HybridModelingPipeline:
    """
    Comprehensive pipeline for hybrid modeling of single-cell RNA data.
    
    This class orchestrates the entire modeling process, from data loading
    to model training, evaluation, and result generation.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "results/04_hybrid_modeling",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the hybrid modeling pipeline.
        
        Args:
            data_path: Path to the processed data file
            output_dir: Directory to save results
            device: Device to run models on
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Initialize data containers
        self.data = None
        self.gene_columns = None
        self.condition_columns = None
        self.cell_type_encoder = None
        
        # Model containers
        self.gnn_model = None
        self.cvae_model = None
        self.gnn_trainer = None
        self.cvae_trainer = None
        
        # Results containers
        self.results = {
            'gnn_performance': {},
            'cvae_performance': {},
            'hybrid_performance': {},
            'training_history': {},
            'generated_samples': {}
        }
        
        logger.info(f"Hybrid Modeling Pipeline initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_and_prepare_data(self, sample_size: int = 10000):
        """
        Load and prepare data for modeling.
        
        Args:
            sample_size: Number of cells to sample for modeling
        """
        logger.info("Loading and preparing data...")
        
        # Load data in chunks to handle large file
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= sample_size:
                break
        
        self.data = pd.concat(chunks, ignore_index=True)
        
        # Sample if needed
        if len(self.data) > sample_size:
            self.data = self.data.sample(n=sample_size, random_state=42)
        
        logger.info(f"Loaded {len(self.data)} cells with {len(self.data.columns)} features")
        
        # Identify feature columns
        self._identify_feature_columns()
        
        # Prepare data for modeling
        self._prepare_modeling_data()
        
        logger.info("Data preparation complete")
    
    def _identify_feature_columns(self):
        """Identify different types of feature columns."""
        # Gene expression features (numeric, non-categorical)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove known non-gene columns
        non_gene_cols = [
            'nCount_RNA', 'nFeature_RNA', 'RNA_snn_res.0.8', 'seurat_clusters',
            'group', 'RNA_snn_res.4', 'cluster_group', 'healhy_cells_percent',
            'combine_group_pathlogy', 'associate_cells', 'data_id'
        ]
        
        # Prediction score columns
        prediction_cols = [col for col in numeric_cols if 'prediction.score' in col]
        
        # Gene expression columns (remaining numeric)
        self.gene_columns = [col for col in numeric_cols 
                           if col not in non_gene_cols and col not in prediction_cols]
        
        # Condition columns (categorical and key features)
        self.condition_columns = [
            'marker_cell_type', 'scina_cell_type', 'predicted.id',
            'nCount_RNA', 'nFeature_RNA', 'seurat_clusters'
        ]
        
        logger.info(f"Identified {len(self.gene_columns)} gene expression features")
        logger.info(f"Identified {len(self.condition_columns)} condition features")
    
    def _prepare_modeling_data(self):
        """Prepare data for modeling by handling missing values and encoding."""
        # Handle missing values
        for col in self.gene_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(0)
        
        # Encode cell types for GNN
        if 'marker_cell_type' in self.data.columns:
            self.cell_type_encoder = LabelEncoder()
            self.data['cell_type_encoded'] = self.cell_type_encoder.fit_transform(
                self.data['marker_cell_type'].fillna('Unknown')
            )
        
        # Create synthetic spatial coordinates if not available
        if 'spatial_x' not in self.data.columns or 'spatial_y' not in self.data.columns:
            logger.info("Creating synthetic spatial coordinates...")
            np.random.seed(42)
            self.data['spatial_x'] = np.random.rand(len(self.data)) * 1000
            self.data['spatial_y'] = np.random.rand(len(self.data)) * 1000
        
        logger.info("Data preparation complete")
    
    def train_gnn_model(self, 
                       hidden_dim: int = 128,
                       num_layers: int = 3,
                       epochs: int = 50,
                       batch_size: int = 32):
        """
        Train the Graph Neural Network model.
        
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info("Training GNN model...")
        
        # Prepare data for GNN
        gene_data = self.data[self.gene_columns].values
        spatial_coords = self.data[['spatial_x', 'spatial_y']].values
        cell_types = self.data['cell_type_encoded'].values
        
        # Split data
        train_idx, test_idx = train_test_split(
            range(len(gene_data)), test_size=0.2, random_state=42, stratify=cell_types
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, 
            stratify=cell_types[train_idx]
        )
        
        # Create graphs
        graph_constructor = SpatialGraphConstructor(n_neighbors=10, radius=100.0)
        
        # Train graph
        train_graph = graph_constructor.construct_graph(
            gene_data[train_idx], spatial_coords[train_idx], cell_types[train_idx]
        )
        
        # Validation graph
        val_graph = graph_constructor.construct_graph(
            gene_data[val_idx], spatial_coords[val_idx], cell_types[val_idx]
        )
        
        # Create data loaders
        train_loader = DataLoader([train_graph], batch_size=1, shuffle=True)
        val_loader = DataLoader([val_graph], batch_size=1, shuffle=False)
        
        # Create model
        num_classes = len(np.unique(cell_types))
        self.gnn_model = create_spatial_gnn_model(
            input_dim=len(self.gene_columns),
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # Create trainer
        self.gnn_trainer = SpatialGNNTrainer(self.gnn_model, device=self.device)
        
        # Train model
        history = self.gnn_trainer.train(
            train_loader, val_loader, epochs=epochs, early_stopping_patience=10
        )
        
        # Save results
        self.results['gnn_performance'] = {
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': max(history['val_accuracy']),
            'num_classes': num_classes,
            'model_parameters': sum(p.numel() for p in self.gnn_model.parameters())
        }
        
        # Save model
        torch.save(self.gnn_model.state_dict(), 
                  self.output_dir / "models" / "gnn_model.pth")
        
        logger.info(f"GNN training complete. Best accuracy: {self.results['gnn_performance']['best_val_accuracy']:.4f}")
    
    def train_cvae_model(self,
                        latent_dim: int = 32,
                        hidden_dims: list = [256, 128, 64],
                        beta: float = 1.0,
                        epochs: int = 50,
                        batch_size: int = 64):
        """
        Train the Conditional Variational Autoencoder model.
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            beta: Beta weight for KL divergence
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info("Training CVAE model...")
        
        # Prepare data for CVAE
        gene_data = self.data[self.gene_columns].values
        condition_data = self.data[self.condition_columns].values
        
        # Split data
        train_idx, test_idx = train_test_split(
            range(len(gene_data)), test_size=0.2, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42
        )
        
        # Create data processor
        data_processor = SingleCellDataProcessor(
            gene_columns=self.gene_columns,
            condition_columns=self.condition_columns,
            batch_size=batch_size
        )
        
        # Preprocess data
        gene_train, condition_train = data_processor.preprocess_data(
            self.data.iloc[train_idx]
        )
        gene_val, condition_val = data_processor.preprocess_data(
            self.data.iloc[val_idx]
        )
        
        # Create data loaders
        train_loader = data_processor.create_data_loader(gene_train, condition_train)
        val_loader = data_processor.create_data_loader(gene_val, condition_val, shuffle=False)
        
        # Create model
        self.cvae_model = create_cvae_model(
            input_dim=len(self.gene_columns),
            condition_dim=len(self.condition_columns),
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            beta=beta
        )
        
        # Create trainer
        self.cvae_trainer = CVAETrainer(self.cvae_model, device=self.device)
        
        # Train model
        history = self.cvae_trainer.train(
            train_loader, val_loader, epochs=epochs, early_stopping_patience=10
        )
        
        # Save results
        self.results['cvae_performance'] = {
            'final_val_loss': history['val_total_loss'][-1],
            'best_val_loss': min(history['val_total_loss']),
            'model_parameters': sum(p.numel() for p in self.cvae_model.parameters())
        }
        
        # Save model
        torch.save(self.cvae_model.state_dict(), 
                  self.output_dir / "models" / "cvae_model.pth")
        
        logger.info(f"CVAE training complete. Best loss: {self.results['cvae_performance']['best_val_loss']:.4f}")
    
    def generate_synthetic_data(self, num_samples: int = 1000):
        """
        Generate synthetic single-cell data using the trained CVAE.
        
        Args:
            num_samples: Number of synthetic samples to generate
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        if self.cvae_model is None:
            logger.error("CVAE model not trained. Please train the model first.")
            return
        
        # Generate synthetic conditions
        synthetic_conditions = []
        for _ in range(num_samples):
            # Randomly sample from existing conditions
            idx = np.random.randint(0, len(self.data))
            condition = self.data[self.condition_columns].iloc[idx].values
            synthetic_conditions.append(condition)
        
        synthetic_conditions = np.array(synthetic_conditions)
        
        # Preprocess conditions
        data_processor = SingleCellDataProcessor(
            gene_columns=self.gene_columns,
            condition_columns=self.condition_columns
        )
        synthetic_conditions_scaled = data_processor.condition_scaler.transform(synthetic_conditions)
        
        # Generate synthetic gene expression
        self.cvae_model.eval()
        with torch.no_grad():
            condition_tensor = torch.FloatTensor(synthetic_conditions_scaled).to(self.device)
            synthetic_expression = self.cvae_model.generate(condition_tensor, num_samples)
            synthetic_expression = synthetic_expression.cpu().numpy()
        
        # Inverse transform to original scale
        synthetic_expression_original = data_processor.gene_scaler.inverse_transform(synthetic_expression)
        
        # Create synthetic dataset
        synthetic_data = pd.DataFrame(
            synthetic_expression_original,
            columns=self.gene_columns
        )
        
        # Add condition columns
        for i, col in enumerate(self.condition_columns):
            synthetic_data[col] = synthetic_conditions[:, i]
        
        # Save synthetic data
        synthetic_data.to_csv(self.output_dir / "results" / "synthetic_cells.csv", index=False)
        
        self.results['generated_samples'] = {
            'num_samples': num_samples,
            'file_path': str(self.output_dir / "results" / "synthetic_cells.csv")
        }
        
        logger.info(f"Synthetic data saved to {self.results['generated_samples']['file_path']}")
    
    def evaluate_models(self):
        """Evaluate and compare model performance."""
        logger.info("Evaluating models...")
        
        # Create evaluation report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_cells': len(self.data),
                'gene_features': len(self.gene_columns),
                'condition_features': len(self.condition_columns)
            },
            'model_performance': {
                'gnn': self.results['gnn_performance'],
                'cvae': self.results['cvae_performance']
            },
            'model_comparison': {
                'gnn_parameters': self.results['gnn_performance']['model_parameters'],
                'cvae_parameters': self.results['cvae_performance']['model_parameters'],
                'total_parameters': (
                    self.results['gnn_performance']['model_parameters'] +
                    self.results['cvae_performance']['model_parameters']
                )
            }
        }
        
        # Save evaluation report
        with open(self.output_dir / "results" / "evaluation_report.json", 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        # Create performance summary
        self._create_performance_summary()
        
        logger.info("Model evaluation complete")
    
    def _create_performance_summary(self):
        """Create a human-readable performance summary."""
        summary = f"""
HYBRID MODELING PERFORMANCE SUMMARY
====================================

Data Summary:
- Total cells: {len(self.data):,}
- Gene features: {len(self.gene_columns)}
- Condition features: {len(self.condition_columns)}

GNN Model Performance:
- Best validation accuracy: {self.results['gnn_performance']['best_val_accuracy']:.4f}
- Number of classes: {self.results['gnn_performance']['num_classes']}
- Model parameters: {self.results['gnn_performance']['model_parameters']:,}

CVAE Model Performance:
- Best validation loss: {self.results['cvae_performance']['best_val_loss']:.4f}
- Model parameters: {self.results['cvae_performance']['model_parameters']:,}

Total Model Parameters: {self.results['gnn_performance']['model_parameters'] + self.results['cvae_performance']['model_parameters']:,}

Generated Data:
- Synthetic samples: {self.results['generated_samples']['num_samples']:,}
- File location: {self.results['generated_samples']['file_path']}

Results saved to: {self.output_dir}
        """
        
        with open(self.output_dir / "results" / "performance_summary.txt", 'w') as f:
            f.write(summary)
        
        print(summary)
    
    def create_visualizations(self):
        """Create comprehensive visualizations of model results."""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # GNN Performance
        ax1.bar(['GNN'], [self.results['gnn_performance']['best_val_accuracy']], 
                color='skyblue', alpha=0.7)
        ax1.set_title('GNN Model Performance')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_ylim(0, 1)
        
        # CVAE Performance
        ax2.bar(['CVAE'], [self.results['cvae_performance']['best_val_loss']], 
                color='lightcoral', alpha=0.7)
        ax2.set_title('CVAE Model Performance')
        ax2.set_ylabel('Validation Loss')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "model_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Complexity Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['GNN', 'CVAE']
        parameters = [
            self.results['gnn_performance']['model_parameters'],
            self.results['cvae_performance']['model_parameters']
        ]
        
        bars = ax.bar(models, parameters, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax.set_title('Model Complexity Comparison')
        ax.set_ylabel('Number of Parameters')
        
        # Add value labels on bars
        for bar, param in zip(bars, parameters):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{param:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "model_complexity.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Data Distribution Comparison (if synthetic data generated)
        if 'generated_samples' in self.results:
            try:
                synthetic_data = pd.read_csv(self.results['generated_samples']['file_path'])
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Compare distributions of key features
                key_features = self.gene_columns[:4] if len(self.gene_columns) >= 4 else self.gene_columns
                
                for i, feature in enumerate(key_features):
                    row, col = i // 2, i % 2
                    
                    # Original data
                    axes[row, col].hist(self.data[feature].dropna(), bins=50, alpha=0.7, 
                                      label='Original', color='skyblue')
                    # Synthetic data
                    axes[row, col].hist(synthetic_data[feature].dropna(), bins=50, alpha=0.7, 
                                      label='Synthetic', color='lightcoral')
                    axes[row, col].set_title(f'Distribution: {feature}')
                    axes[row, col].legend()
                    axes[row, col].set_xlabel('Expression Level')
                    axes[row, col].set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / "figures" / "data_distribution_comparison.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create distribution comparison: {e}")
        
        logger.info("Visualizations created and saved")
    
    def run_complete_pipeline(self, 
                            sample_size: int = 10000,
                            gnn_epochs: int = 50,
                            cvae_epochs: int = 50,
                            generate_synthetic: bool = True):
        """
        Run the complete hybrid modeling pipeline.
        
        Args:
            sample_size: Number of cells to use for modeling
            gnn_epochs: Number of epochs for GNN training
            cvae_epochs: Number of epochs for CVAE training
            generate_synthetic: Whether to generate synthetic data
        """
        logger.info("Starting complete hybrid modeling pipeline...")
        
        try:
            # 1. Load and prepare data
            self.load_and_prepare_data(sample_size)
            
            # 2. Train GNN model
            self.train_gnn_model(epochs=gnn_epochs)
            
            # 3. Train CVAE model
            self.train_cvae_model(epochs=cvae_epochs)
            
            # 4. Generate synthetic data
            if generate_synthetic:
                self.generate_synthetic_data()
            
            # 5. Evaluate models
            self.evaluate_models()
            
            # 6. Create visualizations
            self.create_visualizations()
            
            logger.info("Hybrid modeling pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    print("=" * 80)
    print("HYBRID MODELING PIPELINE FOR SINGLE-CELL RNA ANALYSIS")
    print("=" * 80)
    
    # Configuration
    parser = argparse.ArgumentParser(description="Hybrid Modeling Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config['data_path']
    output_dir = config.get('hybrid_modeling_dir', 'results/04_hybrid_modeling')
    
    # Check if data file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data preprocessing pipeline has been run first.")
        return
    
    # Create and run pipeline
    pipeline = HybridModelingPipeline(data_path, output_dir)
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(
        sample_size=10000,  # Adjust based on available memory
        gnn_epochs=30,      # Reduced for faster execution
        cvae_epochs=30,     # Reduced for faster execution
        generate_synthetic=True
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main() 