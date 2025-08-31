import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import logging
import os
import argparse
import yaml
warnings.filterwarnings('ignore')

def analyze_csv_file(file_path, log_file=None, sample_csv_file=None):
    """Analyze a CSV file and provide comprehensive insights. Optionally log output and save sample."""
    
    # Setup logging if log_file is provided
    if log_file:
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ])
        log = logging.info
    else:
        log = print
    
    log("=" * 60)
    log("CSV FILE ANALYSIS")
    log("=" * 60)
    
    # Check file size
    file_size = Path(file_path).stat().st_size
    log(f"File size: {file_size / (1024*1024):.2f} MB")
    
    # Read the first few rows to understand structure
    log("\n1. READING FILE STRUCTURE...")
    try:
        # Read first 1000 rows to get a sample
        df_sample = pd.read_csv(file_path, nrows=1000)
        log(f"Sample data shape: {df_sample.shape}")
        
        # Get column information
        log(f"\nColumns ({len(df_sample.columns)}):")
        for i, col in enumerate(df_sample.columns, 1):
            log(f"  {i}. {col}")
        
        # Data types
        log(f"\nData types:")
        log(df_sample.dtypes)
        
        # Check for missing values in sample
        log(f"\nMissing values in sample:")
        missing_counts = df_sample.isnull().sum()
        if missing_counts.sum() > 0:
            log(missing_counts[missing_counts > 0])
        else:
            log("No missing values found in sample")
        
        # Basic statistics for numeric columns
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            log(f"\nNumeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
            log("\nBasic statistics for numeric columns:")
            log(df_sample[numeric_cols].describe())
        
        # Categorical columns analysis
        categorical_cols = df_sample.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            log(f"\nCategorical columns ({len(categorical_cols)}): {list(categorical_cols)}")
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                unique_count = df_sample[col].nunique()
                log(f"\n{col}:")
                log(f"  Unique values: {unique_count}")
                if unique_count <= 10:
                    log(f"  Value counts:")
                    log(df_sample[col].value_counts().head())
                else:
                    log(f"  Top 5 values:")
                    log(df_sample[col].value_counts().head())
        
        # Now read the full file for total row count
        log("\n2. COUNTING TOTAL ROWS...")
        row_count = sum(1 for line in open(file_path)) - 1  # Subtract header
        log(f"Total rows: {row_count:,}")
        
        # Memory estimation
        estimated_memory = (file_size / 1000) * (row_count / 1000) / 1024  # Rough estimate
        log(f"Estimated memory usage if loaded: {estimated_memory:.2f} MB")
        
        # Check for duplicates in sample
        log(f"\n3. DUPLICATE ANALYSIS (sample):")
        duplicates = df_sample.duplicated().sum()
        log(f"Duplicate rows in sample: {duplicates}")
        if duplicates > 0:
            log(f"Duplicate percentage: {(duplicates/len(df_sample)*100):.2f}%")
        
        # Data quality insights
        log(f"\n4. DATA QUALITY INSIGHTS:")
        
        # Check for empty strings
        empty_strings = {}
        for col in df_sample.columns:
            if df_sample[col].dtype == 'object':
                empty_count = (df_sample[col] == '').sum()
                if empty_count > 0:
                    empty_strings[col] = empty_count
        
        if empty_strings:
            log("Columns with empty strings:")
            for col, count in empty_strings.items():
                log(f"  {col}: {count}")
        else:
            log("No empty strings found")
        
        # Check for whitespace-only strings
        whitespace_only = {}
        for col in df_sample.columns:
            if df_sample[col].dtype == 'object':
                whitespace_count = df_sample[col].str.strip().eq('').sum()
                if whitespace_count > 0:
                    whitespace_only[col] = whitespace_count
        
        if whitespace_only:
            log("Columns with whitespace-only strings:")
            for col, count in whitespace_only.items():
                log(f"  {col}: {count}")
        
        # Recommendations
        log(f"\n5. RECOMMENDATIONS:")
        log(f"- File is {'large' if row_count > 100000 else 'moderate' if row_count > 10000 else 'small'} sized")
        
        if estimated_memory > 1000:  # More than 1GB
            log("- Consider chunked processing for memory efficiency")
            log("- Use dask or vaex for very large datasets")
        
        if len(numeric_cols) > 0:
            log("- Numeric columns available for statistical analysis")
        
        if len(categorical_cols) > 0:
            log("- Categorical columns available for grouping and analysis")
        
        if missing_counts.sum() > 0:
            log("- Missing data detected - consider data cleaning strategies")
        
        log(f"\n6. SAMPLE DATA PREVIEW:")
        log(df_sample.head())
        
        # Save sample DataFrame if requested
        if sample_csv_file:
            df_sample.to_csv(sample_csv_file, index=False)
            log(f"Sample data saved to: {sample_csv_file}")
        
        return df_sample
        
    except Exception as e:
        log(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV File Analysis")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config.get('data_path', 'data/raw/combined_data_fixed.csv')
    results_dir = config.get('results_dir', 'results/')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = Path(results_dir) / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "analysis_log.txt"
    sample_csv_file = output_dir / "analysis_sample.csv"
    # Analyze the CSV file (use fixed version from csv_inspector_fixer)
    df = analyze_csv_file(data_path, log_file=log_file, sample_csv_file=sample_csv_file)
    
    # If analysis was successful, offer to create visualizations
    if df is not None:
        print(f"\n" + "=" * 60)
        print("Would you like to create visualizations or perform specific analysis?")
        print("Available options:")
        print("1. Create correlation heatmap (for numeric columns)")
        print("2. Create distribution plots")
        print("3. Create value count plots for categorical columns")
        print("4. Export summary statistics")
        print("=" * 60) 