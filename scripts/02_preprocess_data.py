"""
Script to run the Data Preprocessing workflow.

This script imports the DataPreprocessor class and uses the paths from the
central config file to clean the data, handle missing values, and save the
final, analysis-ready dataset.
"""
import sys
import os
import argparse
import yaml

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing.preprocessing import DataPreprocessor
from config.config import (
    PROCESSED_DATA_PATH,  # Input for this script
    RESULTS_DIR
)

def run_preprocessing():
    """
    Initializes and runs the data preprocessing workflow.
    """
    print("=" * 80)
    print("STARTING DATA PREPROCESSING WORKFLOW")
    print("=" * 80)

    # --- Setup ---
    # Define the dedicated results directory for this script
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_results_dir = RESULTS_DIR / script_name
    script_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs for this script will be saved to: {script_results_dir}")
    
    # Define the final output path for the preprocessed data
    output_data_path = PROCESSED_DATA_PATH.parent / f"{PROCESSED_DATA_PATH.stem}_preprocessed.csv"

    try:
        # --- Initialize and run the preprocessor ---
        preprocessor = DataPreprocessor(
            input_path=str(PROCESSED_DATA_PATH),
            output_path=str(output_data_path),
            report_dir=str(script_results_dir)
        )
        preprocessor.run()

    except FileNotFoundError as e:
        print(f"\nERROR: Input file not found.")
        print(f"Details: {e}")
        print("Please ensure you have run the '01_run_csv_inspection.py' script first.")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return

    print("\n" + "=" * 80)
    print("DATA PREPROCESSING WORKFLOW COMPLETE")
    print(f"A detailed report is available at: {script_results_dir / 'preprocessing_report.yaml'}")
    print(f"The final, analysis-ready data is saved at: {output_data_path}")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Preprocessing and Cleaning")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Use config['data_path'] and config['preprocessing_dir'] as needed
    run_preprocessing() 