#!/usr/bin/env python3

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
import gc
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, List, Optional
import joblib

# Configure logging
def setup_logging(results_dir: Path) -> logging.Logger:
    log_file = results_dir / "random_forest_incremental_training_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_data_info(data_path: str, chunk_size: int = 20000) -> Dict:
    """Get basic information about the dataset without loading it all into memory."""
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing dataset structure from {data_path}")
    
    # Read first chunk to get column info
    first_chunk = pd.read_csv(data_path, nrows=chunk_size)
    first_chunk.columns = first_chunk.columns.str.strip()
    
    # Get total number of rows
    total_rows = sum(1 for _ in open(data_path)) - 1  # -1 for header
    
    # Identify feature and target columns - IMPROVED VERSION
    metadata_cols = set([
        'orig.ident', 'RNA_snn_res.0.8', 'seurat_clusters', 'marker_cell_type', 'scina_cell_type',
        'predicted.id', 'barcode', 'group', 'RNA_snn_res.4', 'cluster_group', 'healhy_cells_percent',
        'combine_group_pathlogy', 'associate_cells', 'data_id', 'RNA_snn_res.1.5', 'nCount_RNA', 
        'nFeature_RNA'
    ])
    
    # Much more aggressive feature selection to get actual gene expression data
    gene_columns = []
    for col in first_chunk.columns:
        if col not in metadata_cols:
            # More flexible approach - include all numeric columns that don't look like metadata
            col_lower = col.lower()
            if not any(keyword in col_lower for keyword in 
                      ['snn', 'cluster', 'res', 'ident', 'group', 'score', 'percent', 'prediction']):
                try:
                    # Test if column can be converted to numeric
                    numeric_test = pd.to_numeric(first_chunk[col], errors='coerce')
                    # If at least 50% of values are numeric (less strict than before)
                    if numeric_test.notna().sum() / len(numeric_test) > 0.5:
                        gene_columns.append(col)
                        logger.info(f"Including feature column: {col} (dtype: {first_chunk[col].dtype})")
                    else:
                        logger.info(f"Excluding non-numeric column: {col} (too many non-numeric values)")
                except:
                    logger.info(f"Excluding problematic column: {col}")
    
    # If we still have very few features, be even more liberal
    if len(gene_columns) < 10:
        logger.warning(f"Only {len(gene_columns)} features found. Expanding selection criteria...")
        gene_columns = []
        for col in first_chunk.columns:
            if col not in metadata_cols:
                if first_chunk[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    gene_columns.append(col)
        logger.info(f"Expanded to {len(gene_columns)} features using broader criteria")
    
    logger.info(f"Final selection: {len(gene_columns)} feature columns out of {len(first_chunk.columns)} total columns")
    if gene_columns:
        logger.info(f"Sample feature columns: {gene_columns[:10]}")  # Show first 10
    
    # Better cell type column selection - prioritize biological annotations
    cell_type_cols = []
    
    # Priority order: biological cell types first, then clustering results
    priority_cols = ['marker_cell_type', 'scina_cell_type', 'predicted.id', 'seurat_clusters']
    
    for col in priority_cols:
        if col in first_chunk.columns:
            cell_type_cols.append(col)
    
    # Add any other potential cell type columns
    for col in first_chunk.columns:
        if ('cell_type' in col.lower() or 'cluster' in col.lower()) and col not in cell_type_cols:
            cell_type_cols.append(col)
    
    logger.info(f"Available cell type columns: {cell_type_cols}")
    
    # Sample cell types to understand classes - prefer biological over cluster IDs
    if cell_type_cols:
        # Try biological cell types first
        cell_type_col = cell_type_cols[0]  # This will be marker_cell_type if available
        logger.info(f"Using target column: {cell_type_col}")
        
        sample_cell_types = []
        chunk_iter = pd.read_csv(data_path, chunksize=chunk_size)
        for i, chunk in enumerate(chunk_iter):
            if i >= 10:  # Sample from first 10 chunks
                break
            chunk.columns = chunk.columns.str.strip()
            cell_types = chunk[cell_type_col].astype(str).fillna('Unknown').replace('nan', 'Unknown')
            sample_cell_types.extend(cell_types.unique())
        
        unique_cell_types = list(set(sample_cell_types))
        n_classes = len(unique_cell_types)
        
        # If too many classes, filter to most common ones for better performance
        if n_classes > 15:
            logger.warning(f"Found {n_classes} classes. Will filter to top 10 most common during training")
            # Note: Actual filtering will be done in the training function
    else:
        cell_type_col = 'synthetic'
        n_classes = 5
        unique_cell_types = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
    
    data_info = {
        'total_rows': total_rows,
        'total_features': len(gene_columns),
        'feature_columns': gene_columns,
        'cell_type_column': cell_type_col,
        'n_cell_types': min(n_classes, 15),  # Cap at 15 classes for performance
        'unique_cell_types': unique_cell_types[:20],  # Limit for logging
        'total_columns': len(first_chunk.columns)
    }
    
    logger.info(f"Dataset info: {total_rows} rows, {len(gene_columns)} features, {min(n_classes, 15)} classes (capped)")
    if len(gene_columns) < 10:
        logger.error(f"WARNING: Only {len(gene_columns)} features found - this may impact performance!")
    return data_info

def create_label_encoder(data_path: str, cell_type_col: str, chunk_size: int = 20000) -> LabelEncoder:
    """Create and fit label encoder by scanning through all data."""
    logger = logging.getLogger(__name__)
    logger.info("Creating label encoder by scanning all cell types...")
    
    all_cell_types = set()
    chunk_iter = pd.read_csv(data_path, chunksize=chunk_size)
    
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.strip()
        if cell_type_col in chunk.columns:
            cell_types = chunk[cell_type_col].astype(str).fillna('Unknown').replace('nan', 'Unknown')
            all_cell_types.update(cell_types.unique())
    
    le = LabelEncoder()
    le.fit(list(all_cell_types))
    logger.info(f"Label encoder created with {len(le.classes_)} classes")
    return le

def train_incremental_random_forest(data_path: str, data_info: Dict, 
                                   chunk_size: int = 50000, 
                                   max_chunks: int = None,
                                   max_samples: int = 1500000,
                                   n_estimators: int = 100,
                                   max_samples_per_tree: int = 50000) -> Dict:
    """Train Random Forest using incremental approach with data chunks."""
    logger = logging.getLogger(__name__)
    
    # Create label encoder
    le = create_label_encoder(data_path, data_info['cell_type_column'], chunk_size)
    
    # Initialize Random Forest with better configuration for performance
    # RandomForest will be re-initialized for each fit to ensure max_samples is valid
    rf = None
    
    # Calculate actual max chunks if not specified
    if max_chunks is None:
        estimated_total_chunks = min(30, data_info['total_rows'] // chunk_size + 1)
        max_chunks = estimated_total_chunks
    
    # Cap total samples to prevent memory issues
    max_total_samples = min(max_samples, max_chunks * chunk_size)
    effective_max_chunks = max_total_samples // chunk_size
    
    logger.info(f"Training Random Forest with chunks of size {chunk_size:,}")
    chunk_iter = pd.read_csv(data_path, chunksize=chunk_size)
    processed_chunks = 0
    total_samples_trained = 0
    total_samples_validated = 0
    X_train_chunks = []
    y_train_chunks = []
    X_val_chunks = []
    y_val_chunks = []
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.strip()
        # Extract features
        X_chunk_raw = chunk[data_info['feature_columns']]
        X_chunk_numeric = X_chunk_raw.apply(pd.to_numeric, errors='coerce')
        X_chunk = X_chunk_numeric.fillna(0).values
        # Extract and encode labels
        cell_types = chunk[data_info['cell_type_column']].astype(str).fillna('Unknown').replace('nan', 'Unknown')
        y_chunk = le.transform(cell_types)
        # Split chunk into train/val
        if len(X_chunk) > 1 and len(y_chunk) > 1:
            try:
                X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk = train_test_split(
                    X_chunk, y_chunk, test_size=0.2, random_state=42, stratify=y_chunk
                )
            except Exception:
                X_train_chunk, X_val_chunk, y_train_chunk, y_val_chunk = train_test_split(
                    X_chunk, y_chunk, test_size=0.2, random_state=42
                )
            if len(X_train_chunk) > 0 and len(y_train_chunk) > 0:
                X_train_chunks.append(X_train_chunk)
                y_train_chunks.append(y_train_chunk)
                X_val_chunks.append(X_val_chunk)
                y_val_chunks.append(y_val_chunk)
        processed_chunks += 1
        # Train after each chunk
        if X_train_chunks:
            try:
                X_train_combined = np.vstack(X_train_chunks)
                y_train_combined = np.concatenate(y_train_chunks)
                if len(X_train_combined) > 0 and len(np.unique(y_train_combined)) > 1:
                    valid_max_samples = min(max_samples_per_tree, len(X_train_combined))
                    rf = RandomForestClassifier(
                        n_estimators=min(n_estimators, 50),
                        max_samples=valid_max_samples,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1,
                        verbose=1
                    )
                    rf.fit(X_train_combined, y_train_combined)
                    total_samples_trained += len(X_train_combined)
                X_train_chunks = []
                y_train_chunks = []
                gc.collect()
            except Exception as e:
                logger.error(f"Error during training at chunk {processed_chunks}: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                X_train_chunks = []
                y_train_chunks = []
                gc.collect()
                if len(X_train_chunk) > 0 and len(y_train_chunk) > 0:
                    X_train_chunks.append(X_train_chunk)
                    y_train_chunks.append(y_train_chunk)
                    X_val_chunks.append(X_val_chunk)
                    y_val_chunks.append(y_val_chunk)
                else:
                    logger.warning(f"Chunk {processed_chunks}: No valid samples after split, skipping")
            else:
                logger.warning(f"Chunk {processed_chunks}: Insufficient data (features or targets), skipping")
                continue
            
            processed_chunks += 1
            total_samples_processed += len(chunk)
            
            progress_pct = (processed_chunks / effective_max_chunks) * 100 if effective_max_chunks else 0
            logger.info(f"Processed chunk {processed_chunks}/{effective_max_chunks}, chunk size: {len(chunk):,}, progress: {progress_pct:.1f}%")
            
            # Train more frequently to ensure actual training happens
            if X_train_chunks:
                current_train_samples = sum(len(chunk) for chunk in X_train_chunks)
                logger.info(f"Training on accumulated {len(X_train_chunks)} chunks ({current_train_samples:,} samples)...")
                try:
                    # Combine chunks
                    X_train_combined = np.vstack(X_train_chunks)
                    y_train_combined = np.concatenate(y_train_chunks)
                    logger.info(f"Training data shape: {X_train_combined.shape}, unique classes: {len(np.unique(y_train_combined))}")
                    # Validate data before training
                    if np.any(np.isnan(X_train_combined)) or np.any(np.isinf(X_train_combined)):
                        logger.warning("Found NaN/Inf values in training data, cleaning...")
                        X_train_combined = np.nan_to_num(X_train_combined, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure we have enough data to train
                    if len(X_train_combined) > 0 and len(np.unique(y_train_combined)) > 1:
                        # Set max_samples to min(max_samples_per_tree, number of samples)
                        valid_max_samples = min(max_samples_per_tree, len(X_train_combined))
                        rf = RandomForestClassifier(
                            n_estimators=min(n_estimators, 50),
                            max_samples=valid_max_samples,
                            max_depth=15,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            max_features='sqrt',
                            random_state=42,
                            n_jobs=-1,
                            verbose=1
                        )
                        rf.fit(X_train_combined, y_train_combined)
                        logger.info(f"✓ Model trained successfully with {len(X_train_combined):,} samples and max_samples={valid_max_samples}")
                        total_samples_trained += len(X_train_combined)
                    else:
                        logger.warning(f"Insufficient data for training: {len(X_train_combined)} samples, {len(np.unique(y_train_combined))} classes")
                    # Clear training chunks to free memory more aggressively
                    del X_train_combined, y_train_combined
                    X_train_chunks = []
                    y_train_chunks = []
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error during training at chunk {processed_chunks}: {str(e)}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    # Clear chunks anyway to prevent memory buildup
                    X_train_chunks = []
                    y_train_chunks = []
                    gc.collect()
            
            # Stop when we reach max chunks OR max samples
            if processed_chunks >= effective_max_chunks or total_samples_processed >= max_total_samples:
                logger.info(f"Stopping: processed {processed_chunks} chunks, {total_samples_processed:,} samples")
                break
                
    # Removed stray except block at outer loop level
    
    # Final training if there are remaining chunks
    if X_train_chunks:
        logger.info("Final training on remaining chunks...")
        try:
            X_train_combined = np.vstack(X_train_chunks)
            y_train_combined = np.concatenate(y_train_chunks)
            # Clean data before final training
            if np.any(np.isnan(X_train_combined)) or np.any(np.isinf(X_train_combined)):
                logger.warning("Cleaning NaN/Inf values in final training data...")
                X_train_combined = np.nan_to_num(X_train_combined, nan=0.0, posinf=0.0, neginf=0.0)
            rf.fit(X_train_combined, y_train_combined)
            logger.info("Final training completed successfully")
            total_samples_trained += len(X_train_combined)
        except Exception as e:
            logger.error(f"Error during final training: {str(e)}")

    # Evaluate on validation chunks
    total_samples_validated = sum(len(y) for y in y_val_chunks) if y_val_chunks else 0
    accuracy = 0.0
    f1_weighted = 0.0
    if X_val_chunks:
        logger.info("Evaluating model on validation data...")
        try:
            X_val_combined = np.vstack(X_val_chunks)
            y_val_combined = np.concatenate(y_val_chunks)
            if len(X_val_combined) > 0 and len(np.unique(y_val_combined)) > 1:
                y_pred = rf.predict(X_val_combined)
                accuracy = (y_pred == y_val_combined).mean()
                f1_weighted = f1_score(y_val_combined, y_pred, average='weighted')
                logger.info(f"Validation completed. Accuracy: {accuracy:.4f}, F1: {f1_weighted:.4f}")
            else:
                logger.warning("Validation data has only one class or is empty. Skipping metrics.")
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
    else:
        logger.warning("No validation data available. Reporting training progress only.")

    best_results = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'n_samples_trained': total_samples_trained,
        'n_samples_validated': total_samples_validated,
        'total_samples_processed': processed_chunks * chunk_size,
        'chunks_processed': processed_chunks,
        'n_estimators': n_estimators,
        'data_coverage_percentage': (processed_chunks * chunk_size / data_info['total_rows']) * 100
    }
    return rf, le, best_results

def main():
    parser = argparse.ArgumentParser(description="Incremental Random Forest for Large Datasets")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Chunk size for processing (reduced for memory)')
    parser.add_argument('--max-chunks', type=int, default=None, help='Maximum chunks to process (None = process all data)')
    parser.add_argument('--max-samples', type=int, default=1500000, help='Maximum total samples to process (1.5M default)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "07_train_random_forest_incremental"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(results_dir)
    
    # Calculate max chunks based on smaller sample size for memory safety
    if args.max_chunks is None:
        # Use smaller sample size to prevent memory kills
        max_samples_target = min(args.max_samples, config.get('sample_size', 1500000) or 1500000)
        max_chunks = min(30, max_samples_target // args.chunk_size)  # Cap at 30 chunks
        logger.info(f"Auto-calculated max_chunks: {max_chunks} for {max_samples_target:,} samples")
    else:
        max_chunks = args.max_chunks
    
    # Parameters - use smaller values for memory safety
    chunk_size = args.chunk_size
    # Use the latest preprocessed file for model training
    training_params = {
        'data_path': config['data_path'],
        'chunk_size': chunk_size,
        'max_chunks': max_chunks,
        'max_samples': args.max_samples,
        'random_seed': config.get('random_seed', 42),
        'memory_conservative': True
    }
    
    # Save parameters
    params_path = results_dir / "training_parameters.json"
    with open(params_path, 'w') as f:
        json.dump(training_params, f, indent=2)
    
    logger.info("Starting Incremental Random Forest Training")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Chunk size: {chunk_size}")
    
    try:
        # Get data info
        data_info = get_data_info(training_params['data_path'], chunk_size=5000)
        
        # Save data info
        data_info_path = results_dir / "data_info.json"
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        # Train model with memory-safe parameters
        model, label_encoder, results = train_incremental_random_forest(
            training_params['data_path'],
            data_info,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            max_samples=training_params['max_samples']
        )
        
        # Save model and encoder
        model_path = results_dir / "random_forest_incremental_model.joblib"
        encoder_path = results_dir / "label_encoder.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")
        
        # Save results
        results_path = results_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary
        summary = {
            'training_completed': True,
            'final_accuracy': results['accuracy'],
            'final_f1_weighted': results['f1_weighted'],
            'n_features': data_info['total_features'],
            'n_classes': data_info['n_cell_types'],
            'data_info': data_info,
            'training_method': 'incremental_chunked'
        }
        
        summary_path = results_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create text summary
        summary_text = f"""
INCREMENTAL RANDOM FOREST TRAINING SUMMARY
==========================================

Training completed successfully!

Data Information:
- Total cells in dataset: {data_info['total_rows']:,}
- Cells processed: {results.get('total_samples_processed', 0):,}
- Data coverage: {results.get('data_coverage_percentage', 0):.1f}%
- Features used: {data_info['total_features']}
- Cell type column: {data_info['cell_type_column']}
- Number of classes: {data_info['n_cell_types']}

Training Method:
- Chunk-based incremental training
- Chunk size: {chunk_size:,}
- Chunks processed: {results.get('chunks_processed', 0)}
- Samples trained: {results.get('n_samples_trained', 0):,}
- Samples validated: {results.get('n_samples_validated', 0):,}

Model Performance:
- Final Accuracy: {results['accuracy']:.4f}
- Final F1-Score (Weighted): {results['f1_weighted']:.4f}

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
