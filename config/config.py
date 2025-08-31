"""
Centralized configuration for the single-cell RNA analysis project.
"""
import os
from pathlib import Path

# --- DIRECTORY SETUP ---
# Get the project root directory (assuming this file is in 'config/')
ROOT_DIR = Path(__file__).parent.parent

# --- PATHS ---
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = ROOT_DIR / 'results'
LOG_DIR = RESULTS_DIR / 'logs'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
MODELS_DIR = RESULTS_DIR / 'models'

# --- RAW DATA ---
RAW_DATA_FILENAME = 'combined_data.csv'
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

# --- PROCESSED DATA ---
PROCESSED_DATA_FILENAME = 'combined_data_fixed.csv'
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME

# --- DATA CHARACTERISTICS ---
# Based on initial analysis from 'results/analyze_csv/analysis_log.txt'
# Note: Many of these had 100% missing values in the sample.
# We will keep them for now and handle them during preprocessing.
NUMERIC_COLS = [
    'nCount_RNA', 
    'nFeature_RNA', 
    'RNA_snn_res.0.8', 
    'seurat_clusters', 
    'group', 
    'RNA_snn_res.4', 
    'cluster_group', 
    'healhy_cells_percent', 
    'combine_group_pathlogy', 
    'associate_cells', 
    'prediction.score.Excitatory.neurons', 
    'prediction.score.Oligodendrocytes', 
    'prediction.score.Oligodendrocyte.precursor.cells', 
    'prediction.score.Astrocytes', 
    'prediction.score.Endothelial.cells', 
    'prediction.score.Inhibitory.neurons', 
    'prediction.score.Microglia', 
    'prediction.score.max', 
    'RNA_snn_res.1.5', 
    'prediction.score.Pericytes', 
    'prediction.score.unknown', 
    'RNA_snn_res.1.2', 
    'RNA_snn_res.0.5', 
    'prediction.score.NK.cells', 
    'data_id'
]

CATEGORICAL_COLS = [
    'orig.ident', 
    'marker_cell_type', 
    'scina_cell_type', 
    'predicted.id'
    # 'barcode' is excluded as it is a high-cardinality identifier
]

# Ensure all necessary directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True) 