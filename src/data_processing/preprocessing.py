"""
This module contains the DataPreprocessor class for cleaning and preparing
the single-cell RNA sequencing data.
"""
import pandas as pd
from pathlib import Path

class DataPreprocessor:
    """
    A class to handle the preprocessing of the cleaned scRNA-seq dataset.

    This involves analyzing and handling missing values, converting data types,
    and generating a report of all actions taken.
    """

    def __init__(self, input_path: str, output_path: str, report_dir: str):
        """
        Initializes the DataPreprocessor.

        Args:
            input_path (str): The path to the cleaned CSV file.
            output_path (str): The path to save the final preprocessed CSV file.
            report_dir (str): The directory to save the preprocessing report.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.report = {
            'input_file': str(self.input_path),
            'output_file': str(self.output_path),
            'initial_shape': None,
            'actions': [],
            'final_shape': None,
            'missing_value_summary': {}
        }

    def load_data(self):
        """Loads the dataset from the input path."""
        print(f"Loading data from {self.input_path}...")
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Using low_memory=False is often necessary for mixed-type columns
        self.data = pd.read_csv(self.input_path, low_memory=False)
        self.report['initial_shape'] = self.data.shape
        print(f"Successfully loaded data with shape: {self.data.shape}")

    def analyze_missing_values(self):
        """Analyzes the extent of missing values in the dataset."""
        print("\nAnalyzing missing values...")
        missing_summary = self.data.isnull().sum()
        missing_percent = (missing_summary / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_summary,
            'missing_percent': missing_percent
        })
        
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
            by='missing_percent', ascending=False
        )
        
        self.report['missing_value_summary'] = missing_df.to_dict('index')
        print("Missing value analysis complete. Summary:")
        print(missing_df)

    def handle_missing_values(self, drop_threshold: float = 90.0):
        """
        Handles missing values by dropping columns above a threshold and
        imputing others.

        Args:
            drop_threshold (float): Percentage of missing values above which
                                    a column will be dropped.
        """
        print(f"\nHandling missing values (drop threshold = {drop_threshold}%)...")
        
        # --- Drop columns with high percentage of missing values ---
        cols_to_drop = [
            col for col, summary in self.report['missing_value_summary'].items()
            if summary['missing_percent'] > drop_threshold
        ]
        
        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)
            action = f"Dropped {len(cols_to_drop)} columns with >{drop_threshold}% missing values: {', '.join(cols_to_drop)}"
            self.report['actions'].append(action)
            print(action)
        else:
            print("No columns exceeded the drop threshold.")

        # --- Impute remaining missing values ---
        # For this dataset, we'll impute with 0 for numeric and 'Unknown' for object types
        # This is a common strategy in scRNA-seq where missing can be meaningful (e.g., no prediction score)
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col].fillna(0, inplace=True)
                    action = f"Imputed {self.data[col].isnull().sum()} missing values in numeric column '{col}' with 0."
                else: # Categorical/Object
                    self.data[col].fillna('Unknown', inplace=True)
                    action = f"Imputed {self.data[col].isnull().sum()} missing values in categorical column '{col}' with 'Unknown'."
                
                # We log this action only if imputation happened
                if "Imputed" in action:
                    self.report['actions'].append(action)
                    print(f"  - Imputed missing values in column '{col}'.")
        
        print("Missing value handling complete.")

    def save_preprocessed_data(self):
        """Saves the preprocessed DataFrame to the output path."""
        print(f"\nSaving preprocessed data to {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.output_path, index=False)
        self.report['final_shape'] = self.data.shape
        print("Preprocessed data saved successfully.")
    
    def generate_report(self):
        """Generates and saves a text report of the preprocessing steps."""
        report_path = self.report_dir / "preprocessing_report.txt"
        print(f"Generating preprocessing report at {report_path}...")
        
        # Convert Path objects to strings for report
        self.report['input_file'] = str(self.input_path)
        self.report['output_file'] = str(self.output_path)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PREPROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Input File: {self.report['input_file']}\n")
            f.write(f"Output File: {self.report['output_file']}\n")
            f.write(f"Initial Shape: {self.report['initial_shape']}\n")
            f.write(f"Final Shape: {self.report['final_shape']}\n\n")
            
            f.write("ACTIONS PERFORMED:\n")
            f.write("-" * 20 + "\n")
            for action in self.report['actions']:
                f.write(f"• {action}\n")
            f.write("\n")
            
            f.write("MISSING VALUE SUMMARY:\n")
            f.write("-" * 25 + "\n")
            for col, summary in self.report['missing_value_summary'].items():
                f.write(f"{col}:\n")
                f.write(f"  Missing Count: {summary['missing_count']}\n")
                f.write(f"  Missing Percent: {summary['missing_percent']:.2f}%\n\n")
        
        print("Report generated successfully.")
        
    def run(self):
        """Executes the full preprocessing pipeline."""
        self.load_data()
        self.analyze_missing_values()
        self.handle_missing_values()
        self.save_preprocessed_data()
        self.generate_report() 