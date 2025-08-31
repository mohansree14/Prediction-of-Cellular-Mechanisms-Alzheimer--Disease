"""
Script to run the CSV Inspector and Fixer utility.

This script imports the CSVInspectorFixer class and uses the paths from the
central config file to inspect and, if necessary, fix the raw CSV data.
"""
import sys
import os
import argparse
import yaml

# Add the project root to the Python path to allow for absolute imports
# This is necessary because we are running a script in a subdirectory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing.csv_inspector_fixer import CSVInspectorFixer
from config.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, RESULTS_DIR

def run_inspection():
    """
    Initializes and runs the CSV inspection and fixing process.
    """
    print("=" * 80)
    print("STARTING CSV DATA QUALITY WORKFLOW")
    print("=" * 80)

    # Define the dedicated results directory for this script, following the user's desired pattern
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_results_dir = RESULTS_DIR / script_name
    script_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs for this script will be saved to: {script_results_dir}")

    # Ensure the raw data file exists before proceeding
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: Raw data file not found at {RAW_DATA_PATH}")
        print("Please ensure your data is in the 'data/raw' directory.")
        return

    # Initialize the inspector with separate paths for data and logs
    inspector = CSVInspectorFixer(
        input_file_path=str(RAW_DATA_PATH),
        processed_data_dir=str(PROCESSED_DATA_DIR),
        log_dir=str(script_results_dir)
    )

    # 1. Inspect the file (a full scan is recommended for data integrity)
    print("\nSTEP 1: Scanning the entire file for structural issues...")
    inspection_report = inspector.inspect_csv(sample_lines=-1) # -1 for full scan

    if inspection_report is None:
        print("\nWorkflow halted due to an error during inspection.")
        return

    # 2. If problems are found, fix the file
    if inspection_report['problematic_lines']:
        print("\nSTEP 2: Problems found. Proceeding to fix the file...")
        fixed_file_path = inspector.fix_csv()

        if fixed_file_path:
            # 3. Verify the fixed file
            print("\nSTEP 3: Verifying the integrity of the fixed file...")
            inspector.test_fixed_csv(fixed_file_path)
    else:
        print("\nSTEP 2: No structural problems found in the CSV file. No fixing is needed.")

    print("\n" + "=" * 80)
    print("CSV DATA QUALITY WORKFLOW COMPLETE")
    print(f"A detailed log is available at: {inspector.log_file}")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSV Inspection and Data Validation")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Use config['data_path'] and config['results_dir'] as needed
    run_inspection() 