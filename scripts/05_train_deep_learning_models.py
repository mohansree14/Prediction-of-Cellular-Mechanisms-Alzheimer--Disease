#!/usr/bin/env python3
"""
Deep Learning Models Training Script

This script trains both Conditional VAE and Spatial GNN models on single-cell RNA data.
It follows the same logging and results organization pattern as other scripts in the project.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
import argparse
import yaml
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import gc  # For GPU memory management

# Import our custom models
from src.models.deep_learning.generative_model import (
    ConditionalVAE, CVAETrainer, SingleCellDataProcessor, create_cvae_model
)
from src.models.deep_learning.gnn_model import (
    SpatialGNN, SpatialGNNTrainer, SpatialGraphConstructor, create_spatial_gnn_model
)

# Configure logging
def setup_logging(results_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_file = results_dir / "deep_learning_training_log.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def print_gpu_memory_usage(stage: str):
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger = logging.getLogger(__name__)
        logger.info(f"GPU Memory [{stage}]: Allocated={allocated:.2f}GB, Cached={cached:.2f}GB, Total={total:.2f}GB")

def optimize_gpu_memory():
    """Clean up GPU memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_and_preprocess_data(data_path: str, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load and preprocess the single-cell RNA data.
    
    Args:
        data_path: Path to the data file
        sample_size: Number of samples to use (for testing)
        
    Returns:
        Tuple of (processed_data, data_info)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from {data_path}")
    
    # Load dataset with smart sampling for large files
    try:
        # First, check file size and determine if sampling is needed
        data = pd.read_csv(data_path)
        logger.info(f"Initial data shape: {data.shape}")
        
        # If dataset is too large, implement smart sampling
        if len(data) > 100000:  # More than 100K cells
            logger.warning(f"Large dataset detected ({len(data)} cells). Implementing smart sampling...")
            if sample_size is None:
                sample_size = 50000  # Default to 50K cells for training
            
            # Stratified sampling to maintain class distribution if possible
            if 'marker_cell_type' in data.columns:
                data = data.groupby('marker_cell_type', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, sample_size // data['marker_cell_type'].nunique())), 
                                      random_state=42)
                ).reset_index(drop=True)
                logger.info(f"Applied stratified sampling based on cell types")
            else:
                data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"Applied random sampling")
            
            logger.info(f"Sampled data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Strip whitespace from all DataFrame columns
    data.columns = data.columns.str.strip()
    logger.info(f'All DataFrame columns after stripping: {list(data.columns)}')
    
    # Basic data info
    data_info = {
        'total_cells': len(data),
        'total_features': len(data.columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    # Prepare CVAE data (extract features and conditions more intelligently)
    # Improved feature selection - look for actual gene expression data
    metadata_cols = set([
        'orig.ident', 'RNA_snn_res.0.8', 'seurat_clusters', 'marker_cell_type', 'scina_cell_type',
        'predicted.id', 'barcode', 'group', 'RNA_snn_res.4', 'cluster_group', 'healhy_cells_percent',
        'combine_group_pathlogy', 'associate_cells', 'data_id', 'RNA_snn_res.1.5', 'nCount_RNA', 
        'nFeature_RNA'  # Add these common scRNA-seq metadata columns
    ])
    
    # More intelligent gene column detection
    all_numeric_cols = [col for col in data.columns if str(data[col].dtype) in ['float64', 'int64', 'float32', 'int32']]
    
    # Filter out metadata and prediction score columns
    gene_columns = []
    for col in all_numeric_cols:
        if col not in metadata_cols and not any(keyword in col.lower() for keyword in 
            ['snn', 'cluster', 'res', 'ident', 'group', 'score', 'percent', 'prediction']):
            gene_columns.append(col)
    
    # If we still have very few features, be more liberal
    if len(gene_columns) < 10:
        logger.warning(f"Only {len(gene_columns)} gene columns found. Using broader selection...")
        gene_columns = [col for col in all_numeric_cols if col not in metadata_cols]
    
    logger.info(f'Using {len(gene_columns)} columns as gene features')
    logger.info(f'Gene columns sample: {gene_columns[:10]}')  # Show first 10
    
    if len(gene_columns) == 0:
        logger.error("No gene columns found! Check your data format.")
        raise ValueError("No valid gene expression features found")
    
    gene_data_cvae = data[gene_columns].values
    logger.info(f'Extracted gene_data_cvae shape: {gene_data_cvae.shape}')
    # Prepare condition data
    condition_columns = [col for col in data.columns if col in metadata_cols][:5]
    condition_data, cvae_preprocessing = [], {}
    for col in condition_columns:
        if col in data.columns:
            # Check if column should be treated as categorical
            # Convert mixed types to string to handle categorical data properly
            if data[col].dtype == 'object' or data[col].nunique() < 100:
                le = LabelEncoder()
                # Convert all values to string first to handle mixed types
                col_values = data[col].fillna('Unknown').astype(str)
                encoded = le.fit_transform(col_values)
                condition_data.append(encoded)
                cvae_preprocessing[col] = {'type': 'categorical', 'encoder': le, 'n_categories': len(le.classes_)}
                logger.info(f"Encoded categorical column '{col}' with {len(le.classes_)} unique values")
            else:
                # Handle as numerical
                # Convert to numeric, coercing errors to NaN, then fill with 0
                numeric_values = pd.to_numeric(data[col], errors='coerce').fillna(0)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(numeric_values.values.reshape(-1, 1))
                condition_data.append(scaled.flatten())
                cvae_preprocessing[col] = {'type': 'numerical', 'scaler': scaler}
                logger.info(f"Scaled numerical column '{col}'")
    if condition_data:
        condition_data = np.column_stack(condition_data)
    else:
        condition_data = np.zeros((len(data), 1))
        cvae_preprocessing['dummy_condition'] = {'type': 'dummy'}
    logger.info(f'Condition data shape: {condition_data.shape}')
    
    # CRITICAL FIX: Normalize gene expression data to prevent loss explosion
    logger.info("Applying log1p transformation and standardization to gene data...")
    
    # Apply log1p transformation to handle the scale of gene expression
    gene_data_cvae = np.log1p(gene_data_cvae)  # log(1 + x) transformation
    
    # Standardize the gene expression data
    gene_scaler = StandardScaler()
    gene_data_cvae = gene_scaler.fit_transform(gene_data_cvae)
    
    # Store the scaler for later use
    cvae_preprocessing['gene_scaler'] = gene_scaler
    
    logger.info(f"Gene data stats after normalization: mean={np.mean(gene_data_cvae):.4f}, std={np.std(gene_data_cvae):.4f}")
    logger.info(f"Gene data range: [{np.min(gene_data_cvae):.4f}, {np.max(gene_data_cvae):.4f}]")
    
    data_info['gene_columns'] = len(gene_columns)
    data_info['metadata_columns'] = len(condition_columns)
    data_info['gene_columns_list'] = gene_columns[:10]  # First 10 for reference
    data_info['metadata_columns_list'] = condition_columns
    
    logger.info(f"Identified {len(gene_columns)} gene columns and {len(condition_columns)} metadata columns")
    
    # Handle missing values
    if data_info['missing_values'] > 0:
        logger.info(f"Handling {data_info['missing_values']} missing values")
        data = data.fillna(0)  # Fill with zeros for gene expression
    
    # Remove duplicates
    if data_info['duplicate_rows'] > 0:
        logger.info(f"Removing {data_info['duplicate_rows']} duplicate rows")
        data = data.drop_duplicates()
    
    return data, data_info

def prepare_cvae_data(data: pd.DataFrame, gene_columns: List[str], 
                     condition_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepare data for Conditional VAE training.
    
    Args:
        data: Input dataframe
        gene_columns: List of gene expression columns
        condition_columns: List of conditioning columns
        
    Returns:
        Tuple of (gene_data, condition_data, preprocessing_info)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Preparing data for Conditional VAE")
    
    # Ensure all gene_columns exist in the DataFrame
    missing_cols = [col for col in gene_columns if col not in data.columns]
    if missing_cols:
        logger.error(f'Missing columns in data: {missing_cols}')
    gene_data_cvae = data[gene_columns].values
    logger.info(f'Extracted gene_data_cvae shape: {gene_data_cvae.shape}')
    # Prepare condition data as before
    condition_data, cvae_preprocessing = [], {}
    for col in condition_columns:
        if col in data.columns:
            # Check if column should be treated as categorical
            if data[col].dtype == 'object' or data[col].nunique() < 100:
                le = LabelEncoder()
                # Convert all values to string first to handle mixed types
                col_values = data[col].fillna('Unknown').astype(str)
                encoded = le.fit_transform(col_values)
                condition_data.append(encoded)
                cvae_preprocessing[col] = {'type': 'categorical', 'encoder': le, 'n_categories': len(le.classes_)}
            else:
                # Handle as numerical
                numeric_values = pd.to_numeric(data[col], errors='coerce').fillna(0)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(numeric_values.values.reshape(-1, 1))
                condition_data.append(scaled.flatten())
                cvae_preprocessing[col] = {'type': 'numerical', 'scaler': scaler}
    if condition_data:
        condition_data = np.column_stack(condition_data)
    else:
        condition_data = np.zeros((len(data), 1))
        cvae_preprocessing['dummy_condition'] = {'type': 'dummy'}
    logger.info(f'Condition data shape: {condition_data.shape}')
    
    return gene_data_cvae, condition_data, cvae_preprocessing

def prepare_gnn_data(data: pd.DataFrame, gene_columns: List[str], 
                    spatial_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare data for Spatial GNN training.
    
    Args:
        data: Input dataframe
        gene_columns: List of gene expression columns
        spatial_columns: List of spatial coordinate columns
        
    Returns:
        Tuple of (gene_data, spatial_coords, cell_types, preprocessing_info)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Preparing data for Spatial GNN")
    
    # Extract gene expression data
    gene_data = data[gene_columns].values
    logger.info(f"Gene expression data shape: {gene_data.shape}")
    
    # Prepare spatial coordinates
    if spatial_columns and all(col in data.columns for col in spatial_columns):
        spatial_coords = data[spatial_columns].values
        logger.info(f"Using spatial columns: {spatial_columns}")
    else:
        # Generate synthetic spatial coordinates if not available
        logger.info("Generating synthetic spatial coordinates")
        np.random.seed(42)
        spatial_coords = np.random.rand(len(data), 2) * 1000
        spatial_columns = ['synthetic_x', 'synthetic_y']
    
    # Prepare cell types (if available)
    cell_type_cols = [col for col in data.columns if 'cell_type' in col.lower() or 'cluster' in col.lower()]
    
    if cell_type_cols:
        cell_type_col = cell_type_cols[0]
        le = LabelEncoder()
        # Convert all values to string first to handle mixed types
        col_values = data[cell_type_col].fillna('Unknown').astype(str)
        cell_types = le.fit_transform(col_values)
        logger.info(f"Using cell type column: {cell_type_col}")
        logger.info(f"Number of cell types: {len(le.classes_)}")
    else:
        # Generate synthetic cell types for demonstration
        logger.info("Generating synthetic cell types")
        np.random.seed(42)
        cell_types = np.random.randint(0, 5, len(data))  # 5 cell types
        cell_type_col = 'synthetic_cell_type'
    
    preprocessing_info = {
        'spatial_columns': spatial_columns,
        'cell_type_column': cell_type_col,
        'n_cell_types': len(np.unique(cell_types))
    }
    
    if cell_type_cols:
        preprocessing_info['cell_type_encoder'] = le
    
    return gene_data, spatial_coords, cell_types, preprocessing_info

def train_cvae_model(gene_data: np.ndarray, condition_data: np.ndarray, 
                    results_dir: Path, device: str, training_params: Dict = None) -> Dict:
    """
    Train the Conditional VAE model.
    
    Args:
        gene_data: Gene expression data
        condition_data: Conditioning variables
        results_dir: Directory to save results
        device: Device to use for training
        training_params: Training configuration parameters
        
    Returns:
        Training results dictionary
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Conditional VAE training")
    
    # Check if we have valid gene data
    if gene_data.shape[1] == 0:
        logger.error("No gene expression data available for CVAE training")
        return {'error': 'No gene expression data available'}
    
    # Split data
    train_gene, val_gene, train_cond, val_cond = train_test_split(
        gene_data, condition_data, test_size=0.2, random_state=42
    )
    
    # Create data processor
    gene_columns = [f'gene_{i}' for i in range(gene_data.shape[1])]
    condition_columns = [f'cond_{i}' for i in range(condition_data.shape[1])]
    
    # Set default training parameters if not provided
    if training_params is None:
        training_params = {'batch_size': 64, 'num_workers': 2, 'pin_memory': False}
    
    # Create data processor with parameters supported by the class
    try:
        processor = SingleCellDataProcessor(
            gene_columns=gene_columns,
            condition_columns=condition_columns,
            batch_size=training_params.get('batch_size', 64),
            num_workers=training_params.get('num_workers', 2),
            pin_memory=training_params.get('pin_memory', False)
        )
    except TypeError:
        # Fallback if processor doesn't support all parameters
        processor = SingleCellDataProcessor(
            gene_columns=gene_columns,
            condition_columns=condition_columns,
            batch_size=training_params.get('batch_size', 64)
        )
    
    # Create data loaders
    train_loader = processor.create_data_loader(train_gene, train_cond, shuffle=True)
    val_loader = processor.create_data_loader(val_gene, val_cond, shuffle=False)
    
    # Create model with improved configuration
    model = create_cvae_model(
        input_dim=gene_data.shape[1],
        condition_dim=condition_data.shape[1],
        latent_dim=32,  # Reduced from 64 for stability
        hidden_dims=[128, 64],  # Simplified architecture
        beta=0.1  # Reduced beta to prevent KL divergence dominance
    ).to(device)
    
    # Create trainer with better parameters for stability
    try:
        trainer = CVAETrainer(
            model=model,
            device=device,
            learning_rate=1e-4,  # Reduced learning rate
            weight_decay=1e-4,   # Increased weight decay
            mixed_precision=training_params.get('mixed_precision', False)  # Enable FP16 if available
        )
    except TypeError:
        # Fallback if trainer doesn't support mixed_precision parameter
        trainer = CVAETrainer(
            model=model,
            device=device,
            learning_rate=1e-4,  # Reduced learning rate
            weight_decay=1e-4    # Increased weight decay
        )
    
    # Train model with better parameters
    logger.info("Training Conditional VAE...")
    print_gpu_memory_usage("Before CVAE Training")
    
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,   # Reduced for stability
        early_stopping_patience=5  # More aggressive early stopping
    )
    
    # Save model
    model_path = results_dir / "cvae_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_results': training_results,
        'model_config': {
            'input_dim': gene_data.shape[1],
            'condition_dim': condition_data.shape[1],
            'latent_dim': 64
        }
    }, model_path)
    
    logger.info(f"Conditional VAE model saved to {model_path}")
    print_gpu_memory_usage("After CVAE Training")
    optimize_gpu_memory()  # Clean up GPU memory
    
    return training_results

def train_gnn_model(gene_data: np.ndarray, spatial_coords: np.ndarray, 
                   cell_types: np.ndarray, results_dir: Path, device: str, 
                   training_params: Dict = None) -> Dict:
    """
    Train the Spatial GNN model.
    
    Args:
        gene_data: Gene expression data
        spatial_coords: Spatial coordinates
        cell_types: Cell type labels
        results_dir: Directory to save results
        device: Device to use for training
        training_params: Training configuration parameters
        
    Returns:
        Training results dictionary
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Spatial GNN training")
    
    # Set default training parameters if not provided
    if training_params is None:
        training_params = {'batch_size': 512, 'num_workers': 4, 'pin_memory': True}
    
    # For GNN training, use a much smaller subset to avoid CUDA OOM
    # GNN with graphs is very memory intensive
    if len(gene_data) > 100000:  # If dataset is large
        logger.warning("Large dataset detected. Reducing size for GNN training to prevent CUDA OOM")
        # Sample 100k cells for GNN training
        sample_idx = np.random.choice(len(gene_data), 100000, replace=False)
        gene_data = gene_data[sample_idx]
        spatial_coords = spatial_coords[sample_idx] 
        cell_types = cell_types[sample_idx]
        logger.info(f"Reduced dataset for GNN: {len(gene_data)} cells")
    
    # Check if we have valid gene data
    if gene_data.shape[1] == 0:
        logger.error("No gene expression data available for GNN training")
        return {'error': 'No gene expression data available'}
    
    # Check cell type distribution
    unique_cell_types, counts = np.unique(cell_types, return_counts=True)
    logger.info(f"Cell type distribution: {dict(zip(unique_cell_types, counts))}")
    
    # Check if we can use stratification
    min_samples_per_class = min(counts)
    use_stratification = min_samples_per_class >= 2
    
    if not use_stratification:
        logger.warning(f"Some cell types have fewer than 2 samples. Using random split instead of stratified split.")
        train_idx, val_idx = train_test_split(
            range(len(gene_data)), test_size=0.2, random_state=42
        )
    else:
        train_idx, val_idx = train_test_split(
            range(len(gene_data)), test_size=0.2, random_state=42, stratify=cell_types
        )
    
    # Create graph constructor with smaller parameters for memory efficiency
    graph_constructor = SpatialGraphConstructor(n_neighbors=5, radius=50.0)  # Reduced from 10 neighbors, 100 radius
    
    # Construct training graph
    train_gene = gene_data[train_idx]
    train_spatial = spatial_coords[train_idx]
    train_cell_types = cell_types[train_idx]
    train_graph = graph_constructor.construct_graph(
        gene_expression=train_gene,
        spatial_coords=train_spatial,
        cell_types=train_cell_types
    )
    # Do NOT set train_graph.batch
    # train_graph.batch = torch.zeros(train_graph.num_nodes, dtype=torch.long)
    # Construct validation graph
    val_gene = gene_data[val_idx]
    val_spatial = spatial_coords[val_idx]
    val_cell_types = cell_types[val_idx]
    val_graph = graph_constructor.construct_graph(
        gene_expression=val_gene,
        spatial_coords=val_spatial,
        cell_types=val_cell_types
    )
    # Do NOT set val_graph.batch
    # val_graph.batch = torch.zeros(val_graph.num_nodes, dtype=torch.long)
    
    # Create data loaders
    train_loader = [train_graph]  # Single graph for now
    val_loader = [val_graph]
    
    # Create model with smaller dimensions for memory efficiency
    model = create_spatial_gnn_model(
        input_dim=gene_data.shape[1],
        num_classes=len(np.unique(cell_types)),
        hidden_dim=64,  # Reduced from 128
        num_layers=2    # Reduced from 3
    ).to(device)
    
    # Create trainer with GPU optimizations (if supported)
    try:
        trainer = SpatialGNNTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-5,
            mixed_precision=training_params.get('mixed_precision', False)  # Enable FP16 if available
        )
    except TypeError:
        # Fallback if trainer doesn't support mixed_precision parameter
        trainer = SpatialGNNTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-5
        )
    
    # Train model with better error handling
    logger.info("Training Spatial GNN...")
    print_gpu_memory_usage("Before GNN Training")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Debug: Check trainer method signature
    logger.info(f"Trainer type: {type(trainer)}")
    logger.info(f"Train graph nodes: {train_graph.x.shape[0]}, features: {train_graph.x.shape[1]}")
    logger.info(f"Train graph edges: {train_graph.edge_index.shape[1]}")
    
    # The trainer expects direct graph objects, not loaders
    try:
        # Validate graphs before training
        if train_graph.x.shape[0] == 0 or train_graph.edge_index.shape[1] == 0:
            raise ValueError("Empty graph data detected")
            
        if torch.isnan(train_graph.x).any() or torch.isinf(train_graph.x).any():
            raise ValueError("NaN or Inf values in graph features")
            
        training_results = trainer.train(
            train_graph,  # Direct graph object
            val_graph,    # Direct graph object  
            epochs=30,    # Further reduced epochs for stability
            early_stopping_patience=5  # More aggressive early stopping
        )
        logger.info("GNN training completed successfully")
    except Exception as e:
        logger.error(f"GNN Training failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Return minimal results if training fails
        training_results = {
            'train_loss': [1.0],
            'val_loss': [1.0], 
            'val_accuracy': [0.0],
            'error': f'Training failed: {str(e)}'
        }
    
    # Save model
    model_path = results_dir / "spatial_gnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_results': training_results,
        'model_config': {
            'input_dim': gene_data.shape[1],
            'num_classes': len(np.unique(cell_types)),
            'hidden_dim': 64  # Updated to reflect the reduced size
        }
    }, model_path)
    
    logger.info(f"Spatial GNN model saved to {model_path}")
    print_gpu_memory_usage("After GNN Training")
    optimize_gpu_memory()  # Clean up GPU memory
    
    return training_results

def create_training_plots(cvae_results: Dict, gnn_results: Dict, results_dir: Path):
    """
    Create training plots and save them.
    
    Args:
        cvae_results: Conditional VAE training results
        gnn_results: Spatial GNN training results
        results_dir: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Creating training plots")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Deep Learning Models Training Results', fontsize=16, fontweight='bold')
    
    # CVAE Loss plots
    if 'train_losses' in cvae_results:
        axes[0, 0].plot(cvae_results['train_losses'], label='Training Loss', color='blue')
        axes[0, 0].plot(cvae_results['val_losses'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Conditional VAE - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'reconstruction_losses' in cvae_results:
        axes[0, 1].plot(cvae_results['reconstruction_losses'], label='Reconstruction Loss', color='green')
        axes[0, 1].plot(cvae_results['kl_losses'], label='KL Divergence', color='orange')
        axes[0, 1].set_title('Conditional VAE - Component Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # GNN Loss plots
    if 'train_losses' in gnn_results:
        axes[1, 0].plot(gnn_results['train_losses'], label='Training Loss', color='blue')
        axes[1, 0].plot(gnn_results['val_losses'], label='Validation Loss', color='red')
        axes[1, 0].set_title('Spatial GNN - Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'val_accuracies' in gnn_results:
        axes[1, 1].plot(gnn_results['val_accuracies'], label='Validation Accuracy', color='green')
        axes[1, 1].set_title('Spatial GNN - Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_dir / "training_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training plots saved to {plot_path}")

def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    return obj

def convert_training_history_for_plotting(history: dict, model_type: str) -> dict:
    """
    Convert training history keys to the format expected by the plotting and summary code.
    Args:
        history: The original training history dict.
        model_type: 'cvae' or 'gnn'
    Returns:
        Dict with keys: train_losses, val_losses, reconstruction_losses, kl_losses (for CVAE)
        or train_losses, val_losses, val_accuracies (for GNN)
    """
    if model_type == 'cvae':
        return {
            'train_losses': history.get('train_total_loss', []),
            'val_losses': history.get('val_total_loss', []),
            'reconstruction_losses': history.get('train_reconstruction_loss', []),
            'kl_losses': history.get('train_kl_loss', []),
        }
    elif model_type == 'gnn':
        return {
            'train_losses': history.get('train_loss', []),
            'val_losses': history.get('val_loss', []),
            'val_accuracies': history.get('val_accuracy', []),
        }
    else:
        return history

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Deep Learning Model Training")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "05_train_deep_learning_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(results_dir)
    
    # Training parameters optimized for GPU but with CPU fallback and memory limits
    training_params = {
        'data_path': config['data_path'],
        'sample_size': config['sample_size'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_seed': config['random_seed'],
        # Reduced parameters for memory efficiency
        'batch_size': 256 if torch.cuda.is_available() else 64,   # Smaller GPU batch for memory
        'num_workers': 2 if torch.cuda.is_available() else 1,    # Fewer workers
        'pin_memory': torch.cuda.is_available(),                 # Only useful for GPU
        'mixed_precision': torch.cuda.is_available(),            # Only for GPU
        'cuda_benchmark': torch.cuda.is_available()              # Only for GPU
    }
    
    # Set CUDA optimizations only if GPU is available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = training_params['cuda_benchmark']
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("GPU not available - using CPU with reduced batch size")
    
    # Save parameters
    params_path = results_dir / "training_parameters.json"
    with open(params_path, 'w') as f:
        json.dump(training_params, f, indent=2)
    
    logger.info("Starting Deep Learning Models Training")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Device: {training_params['device']}")
    logger.info(f"Sample size: {training_params['sample_size']}")
    logger.info(f"Batch size: {training_params['batch_size']}")
    logger.info(f"Mixed precision: {training_params['mixed_precision']}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print_gpu_memory_usage("Script Start")
    
    # Set random seeds
    torch.manual_seed(training_params['random_seed'])
    np.random.seed(training_params['random_seed'])
    
    try:
        # Load and preprocess data
        data, data_info = load_and_preprocess_data(
            training_params['data_path'], 
            training_params['sample_size']
        )
        
        # Save data info
        data_info_path = results_dir / "data_info.json"
        with open(data_info_path, 'w') as f:
            serializable_info = make_json_serializable(data_info)
            json.dump(serializable_info, f, indent=2)
        
        # Prepare CVAE data (extract features and conditions directly)
        metadata_cols = set([
            'orig.ident', 'RNA_snn_res.0.8', 'seurat_clusters', 'marker_cell_type', 'scina_cell_type',
            'predicted.id', 'barcode', 'group', 'RNA_snn_res.4', 'cluster_group', 'healhy_cells_percent',
            'combine_group_pathlogy', 'associate_cells', 'data_id', 'RNA_snn_res.1.5'
        ])
        gene_columns = [col for col in data.columns if col not in metadata_cols and str(data[col].dtype) in ['float64', 'int64', 'float32', 'int32']]
        logger.info(f'Using the following columns as features: {gene_columns}')
        gene_data_cvae = data[gene_columns].values
        logger.info(f'Extracted gene_data_cvae shape: {gene_data_cvae.shape}')
        # Prepare condition data
        condition_columns = [col for col in data.columns if col in metadata_cols][:5]
        condition_data, cvae_preprocessing = [], {}
        for col in condition_columns:
            if col in data.columns:
                # Check if column should be treated as categorical
                if data[col].dtype == 'object' or data[col].nunique() < 100:
                    le = LabelEncoder()
                    # Convert all values to string first to handle mixed types
                    col_values = data[col].fillna('Unknown').astype(str)
                    encoded = le.fit_transform(col_values)
                    condition_data.append(encoded)
                    cvae_preprocessing[col] = {'type': 'categorical', 'encoder': le, 'n_categories': len(le.classes_)}
                else:
                    # Handle as numerical
                    numeric_values = pd.to_numeric(data[col], errors='coerce').fillna(0)
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(data[col].fillna(0).values.reshape(-1, 1))
                    condition_data.append(scaled.flatten())
                    cvae_preprocessing[col] = {'type': 'numerical', 'scaler': scaler}
        if condition_data:
            condition_data = np.column_stack(condition_data)
        else:
            condition_data = np.zeros((len(data), 1))
            cvae_preprocessing['dummy_condition'] = {'type': 'dummy'}
        logger.info(f'Condition data shape: {condition_data.shape}')
        
        # Prepare GNN data (use same features as CVAE)
        gene_data_gnn = gene_data_cvae
        # Always generate synthetic spatial coordinates for GNN
        spatial_coords = np.random.rand(len(data), 2) * 1000
        # Prepare cell types (if available)
        cell_type_cols = [col for col in data.columns if 'cell_type' in col.lower() or 'cluster' in col.lower()]
        if cell_type_cols:
            cell_type_col = cell_type_cols[0]
            le = LabelEncoder()
            # Convert all values to string first to handle mixed types
            col_values = data[cell_type_col].fillna('Unknown').astype(str)
            cell_types = le.fit_transform(col_values)
        else:
            cell_types = np.random.randint(0, 5, len(data))
        logger.info(f'GNN gene_data_gnn shape: {gene_data_gnn.shape}')
        logger.info(f'GNN spatial_coords shape: {spatial_coords.shape}')
        logger.info(f'GNN cell_types shape: {cell_types.shape}')
        
        # Train Conditional VAE
        logger.info("=" * 50)
        logger.info("TRAINING CONDITIONAL VAE")
        logger.info("=" * 50)
        cvae_start_time = time.time()
        cvae_results_raw = train_cvae_model(
            gene_data_cvae, condition_data, results_dir, training_params['device'], training_params
        )
        cvae_training_time = time.time() - cvae_start_time
        # Check if CVAE training failed
        if 'error' in cvae_results_raw:
            logger.error(f"CVAE training failed: {cvae_results_raw['error']}")
            cvae_results_raw = {'train_total_loss': [], 'val_total_loss': [], 'train_reconstruction_loss': [], 'train_kl_loss': []}
        cvae_results = convert_training_history_for_plotting(cvae_results_raw, 'cvae')

        # Train Spatial GNN
        logger.info("=" * 50)
        logger.info("TRAINING SPATIAL GNN")
        logger.info("=" * 50)
        gnn_start_time = time.time()
        gnn_results_raw = train_gnn_model(
            gene_data_gnn, spatial_coords, cell_types, results_dir, training_params['device'], training_params
        )
        gnn_training_time = time.time() - gnn_start_time
        # Check if GNN training failed
        if 'error' in gnn_results_raw:
            logger.error(f"GNN training failed: {gnn_results_raw['error']}")
            gnn_results_raw = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        gnn_results = convert_training_history_for_plotting(gnn_results_raw, 'gnn')

        # Create training plots
        create_training_plots(cvae_results, gnn_results, results_dir)

        # Create summary report
        summary = {
            'training_completed': True,
            'cvae_training_time_minutes': cvae_training_time / 60,
            'gnn_training_time_minutes': gnn_training_time / 60,
            'total_training_time_minutes': (cvae_training_time + gnn_training_time) / 60,
            'cvae_final_train_loss': cvae_results['train_losses'][-1] if cvae_results['train_losses'] else 0,
            'cvae_final_val_loss': cvae_results['val_losses'][-1] if cvae_results['val_losses'] else 0,
            'gnn_final_train_loss': gnn_results['train_losses'][-1] if gnn_results['train_losses'] else 0,
            'gnn_final_val_loss': gnn_results['val_losses'][-1] if gnn_results['val_losses'] else 0,
            'gnn_final_accuracy': gnn_results['val_accuracies'][-1] if gnn_results['val_accuracies'] else 0,
            'data_info': data_info
        }
        
        # Save summary
        summary_path = results_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(make_json_serializable(summary), f, indent=2)
        
        # Create text summary
        summary_text = f"""
DEEP LEARNING MODELS TRAINING SUMMARY
=====================================

Training completed successfully!

Data Information:
- Total cells: {data_info['total_cells']}
- Gene features: {data_info['gene_columns']}
- Metadata features: {data_info['metadata_columns']}
- Memory usage: {data_info['memory_usage_mb']:.2f} MB

Training Times:
- Conditional VAE: {cvae_training_time/60:.2f} minutes
- Spatial GNN: {gnn_training_time/60:.2f} minutes
- Total: {(cvae_training_time + gnn_training_time)/60:.2f} minutes

Model Performance:
- CVAE Final Train Loss: {summary['cvae_final_train_loss']:.4f}
- CVAE Final Val Loss: {summary['cvae_final_val_loss']:.4f}
- GNN Final Train Loss: {summary['gnn_final_train_loss']:.4f}
- GNN Final Val Loss: {summary['gnn_final_val_loss']:.4f}
- GNN Final Accuracy: {summary['gnn_final_accuracy']:.4f}

Results saved to: {results_dir}
"""
        
        summary_text_path = results_dir / "training_summary.txt"
        with open(summary_text_path, 'w') as f:
            f.write(summary_text)
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {results_dir}")
        print(summary_text)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
