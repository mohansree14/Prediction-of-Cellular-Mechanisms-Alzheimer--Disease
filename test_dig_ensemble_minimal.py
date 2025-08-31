import sys
from pathlib import Path
import argparse
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("Testing imports...")

try:
    import torch
    print("✓ PyTorch imported")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torch_geometric
    print("✓ PyTorch Geometric imported")
except Exception as e:
    print(f"✗ PyTorch Geometric import failed: {e}")

try:
    from src.models.deep_learning.dig_ensemble import create_dig_ensemble_model
    print("✓ DIG-Ensemble model imported")
except Exception as e:
    print(f"✗ DIG-Ensemble import failed: {e}")
    import traceback
    traceback.print_exc()

print("Testing data loading...")

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from torch_geometric.data import Data
    print("✓ All data processing imports successful")
except Exception as e:
    print(f"✗ Data processing import failed: {e}")

print("Testing data file access...")

try:
    data_path = 'data/processed/combined_data_final_features.csv'
    data = pd.read_csv(data_path, nrows=100, low_memory=False)
    print(f"✓ Data file loaded: {data.shape}")
except Exception as e:
    print(f"✗ Data file loading failed: {e}")

print("Test completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIG-Ensemble Minimal Test")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Use config['data_path'] as needed 