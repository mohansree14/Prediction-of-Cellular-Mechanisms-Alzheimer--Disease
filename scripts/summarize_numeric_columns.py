import pandas as pd
import numpy as np

# Change this path if needed
csv_path = 'data/processed/combined_data_fixed.csv'

# Read a sample for speed (change nrows for more rows)
df = pd.read_csv(csv_path, nrows=100000)

print(f"Shape: {df.shape}")
print("\nSummary statistics for numeric columns:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
summary = df[numeric_cols].describe().T
summary['num_nonzero'] = (df[numeric_cols] != 0).sum()
summary['num_non_nan'] = df[numeric_cols].count()
print(summary)

print("\nColumns with only zeros or NaNs:")
for col in numeric_cols:
    if summary.loc[col, 'num_nonzero'] == 0 or summary.loc[col, 'num_non_nan'] == 0:
        print(col)
