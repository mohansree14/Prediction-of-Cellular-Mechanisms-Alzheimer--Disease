"""
This module contains the CSVInspectorFixer class for comprehensive CSV
inspection and fixing.
"""
import csv
import os
from pathlib import Path
import pandas as pd

class CSVInspectorFixer:
    """
    A class to inspect and fix structural issues in CSV files.

    It identifies rows with incorrect column counts, generates a detailed report,
    and can create a corrected version of the file by padding or truncating
    problematic rows.
    """

    def __init__(self, input_file_path: str, processed_data_dir: str, log_dir: str):
        """
        Initializes the CSVInspectorFixer.

        Args:
            input_file_path (str): The path to the CSV file to be processed.
            processed_data_dir (str): The directory where the fixed data file will be saved.
            log_dir (str): The directory where the log file will be saved.
        """
        self.input_path = Path(input_file_path)
        self.processed_data_dir = Path(processed_data_dir)
        self.log_dir = Path(log_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        log_filename = f"inspection_log_{self.input_path.stem}.txt"
        self.log_file = self.log_dir / log_filename
        self._clear_log_file()

    def _clear_log_file(self):
        """Clears the log file at the start of a run."""
        if self.log_file.exists():
            self.log_file.unlink()

    def log(self, message: str):
        """Logs a message to both the console and the log file."""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def inspect_csv(self, sample_lines: int = 10000) -> dict:
        """
        Performs a comprehensive inspection of the CSV file.

        Args:
            sample_lines (int): The number of lines to sample for the inspection.
                                Set to -1 to scan the entire file.

        Returns:
            dict: A dictionary containing the inspection report.
        """
        self.log("=" * 80)
        self.log("CSV INSPECTION REPORT")
        self.log("=" * 80)

        if not self.input_path.exists():
            self.log(f"ERROR: File not found: {self.input_path}")
            return None

        file_size_mb = self.input_path.stat().st_size / (1024 * 1024)
        self.log(f"File: {self.input_path}")
        self.log(f"Size: {file_size_mb:.2f} MB")

        self.log("\nCounting total lines...")
        with open(self.input_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)
        self.log(f"Total lines: {total_lines:,}")
        
        scan_all = (sample_lines == -1) or (sample_lines >= total_lines)
        scan_limit = total_lines if scan_all else sample_lines
        
        if scan_all:
             self.log(f"\nAnalyzing entire file ({total_lines:,} lines)...")
        else:
            self.log(f"\nAnalyzing first {sample_lines:,} lines...")

        column_counts = {}
        problematic_lines = []
        header = []

        with open(self.input_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                expected_cols = len(header)
                self.log(f"Header found with {expected_cols} columns.")
                column_counts[expected_cols] = 1

                for i, row in enumerate(reader, 2):
                    col_count = len(row)
                    column_counts[col_count] = column_counts.get(col_count, 0) + 1
                    if col_count != expected_cols:
                        problematic_lines.append((i, col_count, expected_cols))
                    
                    if not scan_all and i >= scan_limit:
                        break
            except StopIteration:
                self.log("Warning: CSV file is empty or contains only a header.")
                expected_cols = 0
            except Exception as e:
                self.log(f"An error occurred while reading CSV: {e}")
                return None

        self.log("\nColumn count distribution:")
        for count, num_lines in sorted(column_counts.items()):
            percentage = (num_lines / scan_limit) * 100
            self.log(f"  {count} columns: {num_lines:,} lines ({percentage:.2f}%)")
        
        self.log(f"\nExpected columns based on header: {expected_cols}")
        if problematic_lines:
            self.log(f"Found {len(problematic_lines)} problematic lines in the sample.")
            for line, actual, expected in problematic_lines[:10]:
                self.log(f"  - Line {line:,}: has {actual} columns, expected {expected}")
        else:
            self.log("✓ No problematic lines found in the scanned sample.")

        return {
            'total_lines': total_lines,
            'expected_cols': expected_cols,
            'header': header,
            'column_counts': column_counts,
            'problematic_lines': problematic_lines,
        }

    def fix_csv(self) -> str:
        """
        Fixes the CSV by padding or truncating rows to match the header's column count.

        A new file with the suffix '_fixed.csv' is created in the output directory.

        Returns:
            str: The path to the fixed CSV file, or None if fixing failed.
        """
        self.log("\n" + "=" * 80)
        self.log("FIXING CSV FILE")
        self.log("=" * 80)

        inspection_report = self.inspect_csv(sample_lines=100) # Quick inspection for header
        if not inspection_report or not inspection_report['header']:
            self.log("ERROR: Cannot fix file without a valid header. Aborting.")
            return None
        
        expected_cols = inspection_report['expected_cols']
        total_lines = inspection_report['total_lines']
        
        fixed_file_path = self.processed_data_dir / f"{self.input_path.stem}_fixed.csv"
        self.log(f"Expected columns: {expected_cols}")
        self.log(f"Output file: {fixed_file_path}")

        lines_processed = 0
        lines_fixed = 0

        try:
            with open(self.input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
                 open(fixed_file_path, 'w', encoding='utf-8', newline='') as outfile:
                
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                for i, row in enumerate(reader, 1):
                    lines_processed += 1
                    if i % 500000 == 0:
                        self.log(f"  ...processed {lines_processed:,} / {total_lines:,} lines")

                    if len(row) != expected_cols:
                        lines_fixed += 1
                        if len(row) > expected_cols:
                            writer.writerow(row[:expected_cols]) # Truncate
                        else:
                            padded_row = row + [''] * (expected_cols - len(row))
                            writer.writerow(padded_row) # Pad
                    else:
                        writer.writerow(row)
            
            self.log(f"  ...processed {lines_processed:,} / {total_lines:,} lines (complete).")

        except Exception as e:
            self.log(f"FATAL ERROR during fixing process: {e}")
            return None

        self.log("\n" + "=" * 80)
        self.log("FIXING COMPLETE")
        self.log(f"Total lines processed: {lines_processed:,}")
        self.log(f"Total lines fixed: {lines_fixed:,}")
        self.log(f"Output saved to: {fixed_file_path}")
        
        return str(fixed_file_path)

    def test_fixed_csv(self, fixed_file_path: str) -> bool:
        """
        Tests the integrity of the fixed CSV file by trying to load it with pandas.

        Args:
            fixed_file_path (str): The path to the fixed CSV file.

        Returns:
            bool: True if the file loads successfully, False otherwise.
        """
        self.log("\n" + "=" * 80)
        self.log(f"VERIFYING FIXED CSV: {fixed_file_path}")
        try:
            df = pd.read_csv(fixed_file_path, low_memory=False)
            self.log("✓ Verification successful!")
            self.log(f"  - Shape: {df.shape}")
            self.log(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return True
        except Exception as e:
            self.log(f"✗ Verification FAILED. Error loading with pandas: {e}")
            return False 