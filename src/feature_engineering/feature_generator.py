"""
This module contains the FeatureEngineer class for creating new features
from the preprocessed single-cell RNA sequencing data.
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """
    A class to engineer new features for the scRNA-seq dataset.

    This involves creating interaction terms, polynomial features, and adding
    principal components to enrich the dataset for modeling.
    """

    def __init__(self, input_path: str, output_path: str, report_dir: str, numeric_cols: list, dataframe: pd.DataFrame = None):
        """
        Initializes the FeatureEngineer.

        Args:
            input_path (str): The path to the preprocessed CSV file.
            output_path (str): The path to save the final feature-engineered CSV file.
            report_dir (str): The directory to save the feature engineering report.
            numeric_cols (list): List of numeric column names to use for feature creation.
            dataframe (pd.DataFrame, optional): Pre-loaded DataFrame to use. Defaults to None.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.numeric_cols = numeric_cols
        self.data = dataframe
        self.report = {
            'input_file': str(self.input_path),
            'output_file': str(self.output_path),
            'initial_shape': None,
            'actions': [],
            'final_shape': None
        }

    def load_data(self):
        """Loads the preprocessed dataset from the input path."""
        if self.data is not None:
            print("Data already loaded into the FeatureEngineer.")
        else:
            print(f"Loading preprocessed data from {self.input_path}...")
            if not self.input_path.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
            self.data = pd.read_csv(self.input_path, low_memory=False)

        self.report['initial_shape'] = self.data.shape if self.data is not None else (0,0)
        print(f"Successfully loaded data with shape: {self.data.shape}")

    def create_interaction_features(self):
        """Creates interaction features between the top numeric columns."""
        print("\nCreating interaction features...")
        # We'll create an interaction between the two most fundamental metrics
        if 'nCount_RNA' in self.data.columns and 'nFeature_RNA' in self.data.columns:
            self.data['rna_x_feature_count'] = self.data['nCount_RNA'] * self.data['nFeature_RNA']
            action = "Created interaction feature: 'rna_x_feature_count'."
            self.report['actions'].append(action)
            print(f"  - {action}")
        else:
            print("  - Skipping interaction features: required columns not found.")

    def create_polynomial_features(self, degree: int = 2):
        """
        Creates polynomial features for all numeric columns.
        
        Args:
            degree (int): The degree of the polynomial features.
        """
        print(f"\nCreating polynomial features (degree={degree})...")
        
        numeric_data = self.data[self.numeric_cols].copy()
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(numeric_data)
        
        # Create new column names for the polynomial features
        poly_feature_names = poly.get_feature_names_out(self.numeric_cols)
        
        # Create a DataFrame with the new features
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=self.data.index)
        
        # Drop original columns from poly_df to avoid duplication
        poly_df.drop(columns=self.numeric_cols, inplace=True)
        
        # Concatenate the new polynomial features with the original data
        self.data = pd.concat([self.data, poly_df], axis=1)
        
        action = f"Created {poly_df.shape[1]} polynomial features of degree {degree}."
        self.report['actions'].append(action)
        print(f"  - {action}")

    def add_pca_components(self, n_components: int = 5):
        """
        Performs PCA on numeric data and adds the top components as new features.

        Args:
            n_components (int): The number of principal components to add.
        """
        print(f"\nAdding {n_components} principal components as features...")
        numeric_data = self.data[self.numeric_cols].copy()
        
        # --- Robustness Check ---
        # Ensure n_components is not greater than the number of features available.
        max_components = min(n_components, len(self.numeric_cols))
        if max_components < n_components:
            print(f"  - Warning: Requested {n_components} PCA components, but only {len(self.numeric_cols)} numeric features are available.")
            print(f"  - Adjusting to {max_components} components.")
            n_components = max_components

        if n_components == 0:
            print("  - Skipping PCA: No numeric features available.")
            return

        # It's crucial to scale the data before PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(scaled_data)
        
        # Create a DataFrame for the PCA components
        pca_cols = [f'PC_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_components, columns=pca_cols, index=self.data.index)
        
        # Add the PCA features to the main DataFrame
        self.data = pd.concat([self.data, pca_df], axis=1)
        
        explained_variance = sum(pca.explained_variance_ratio_)
        action = f"Added {n_components} PCA components, explaining {explained_variance:.2%} of the variance."
        self.report['actions'].append(action)
        print(f"  - {action}")

    def save_feature_engineered_data(self):
        """Saves the final feature-engineered DataFrame to the output path."""
        print(f"\nSaving feature-engineered data to {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.output_path, index=False)
        self.report['final_shape'] = self.data.shape
        print("Feature-engineered data saved successfully.")
    
    def generate_report(self):
        """Generates and saves a text report of the feature engineering steps."""
        report_path = self.report_dir / "feature_engineering_report.txt"
        print(f"Generating feature engineering report at {report_path}...")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("FEATURE ENGINEERING REPORT\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Input File: {self.report['input_file']}\n")
            f.write(f"Output File: {self.report['output_file']}\n")
            f.write(f"Initial Shape: {self.report['initial_shape']}\n")
            f.write(f"Final Shape: {self.report['final_shape']}\n\n")
            
            f.write("FEATURES CREATED:\n")
            f.write("-" * 18 + "\n")
            for action in self.report['actions']:
                f.write(f"• {action}\n")
            f.write("\n")
            
            f.write("FEATURE SUMMARY:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Original Features: {self.report['initial_shape'][1]}\n")
            f.write(f"New Features Added: {self.report['final_shape'][1] - self.report['initial_shape'][1]}\n")
            f.write(f"Total Features: {self.report['final_shape'][1]}\n")
        
        print("Report generated successfully.")
        
    def run(self):
        """Executes the full feature engineering pipeline."""
        self.load_data()
        self.create_interaction_features()
        self.create_polynomial_features()
        self.add_pca_components()
        self.save_feature_engineered_data()
        self.generate_report() 