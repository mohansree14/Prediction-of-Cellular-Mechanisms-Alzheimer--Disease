"""
Script to run the Feature Engineering workflow.

This script imports the FeatureEngineer class and uses paths and column lists
from the central config file to generate new features, saving the final,
augmented dataset ready for modeling.
"""
import sys
import os
import pandas as pd
import numpy as np
import argparse
import yaml

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_engineering.feature_generator import FeatureEngineer
from config.config import (
    PROCESSED_DATA_PATH,
    RESULTS_DIR,
    NUMERIC_COLS  # We will use the original numeric columns for feature creation
)

def run_feature_engineering():
    """
    Initializes and runs the feature engineering workflow.
    """
    print("=" * 80)
    print("STARTING FEATURE ENGINEERING WORKFLOW")
    print("=" * 80)

    # --- Setup ---
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_results_dir = RESULTS_DIR / script_name
    script_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs for this script will be saved to: {script_results_dir}")

    # Define input and output paths
    # The input is the *preprocessed* data from the previous step
    input_data_path = PROCESSED_DATA_PATH.parent / f"{PROCESSED_DATA_PATH.stem}_preprocessed.csv"
    output_data_path = PROCESSED_DATA_PATH.parent / "combined_data_final_features.csv"

    try:
        # We need to determine the actual numeric columns present in the preprocessed file
        print(f"Reading data from {input_data_path} to identify numeric features...")
        df = pd.read_csv(input_data_path, low_memory=False)
        
        # Dynamically identify purely numeric columns, excluding identifiers
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # It's good practice to remove ID columns from feature engineering
        if 'data_id' in numeric_cols:
            numeric_cols.remove('data_id')
            
        print(f"Found {len(numeric_cols)} purely numeric columns available for feature engineering.")
        
        # --- Initialize and run the feature engineer ---
        engineer = FeatureEngineer(
            input_path=str(input_data_path),
            output_path=str(output_data_path),
            report_dir=str(script_results_dir),
            numeric_cols=numeric_cols,
            dataframe=df  # Pass the loaded dataframe to avoid reading it again
        )
        engineer.run()

    except FileNotFoundError as e:
        print(f"\nERROR: Input file not found.")
        print(f"Details: {e}")
        print("Please ensure you have run the '02_preprocess_data.py' script first.")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING WORKFLOW COMPLETE")
    print(f"A detailed report is available at: {script_results_dir / 'feature_engineering_report.yaml'}")
    print(f"The final, feature-rich data for modeling is saved at: {output_data_path}")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature Engineering and Extraction")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Use config['data_path'] and config['feature_engineering_dir'] as needed
    run_feature_engineering() 