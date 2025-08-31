import argparse
import yaml
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

# ========== CONFIGURATION ========== #
RANDOM_STATE = 42
MAX_POINTS = 10000
OUTLIER_STD = 5
DEFAULT_DATA_PATH = "data/raw/combined_data_fixed.csv"

# ========== SIMPLE FEATURE EXTRACTION FUNCTIONS ========== #
def simple_extract_basic_features(data):
    features = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        features[f'{col}_mean'] = data[col].mean()
        features[f'{col}_std'] = data[col].std()
        features[f'{col}_median'] = data[col].median()
        features[f'{col}_min'] = data[col].min()
        features[f'{col}_max'] = data[col].max()
    return features

def simple_load_and_preprocess(data_path, log, output_dir):
    data = pd.read_csv(data_path)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    X = data[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mask = np.all(np.abs(X_scaled) < OUTLIER_STD, axis=1)
    X = X[mask]
    data = data.iloc[mask].reset_index(drop=True)
    if X.shape[0] > MAX_POINTS:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(X.shape[0], MAX_POINTS, replace=False)
        X_sample = X[idx].astype(np.float32)
        labels_sample = data['marker_cell_type'].iloc[idx].reset_index(drop=True) if 'marker_cell_type' in data.columns else None
        log(f"Subsampled {MAX_POINTS} rows for feature extraction (original: {X.shape[0]})")
    else:
        X_sample = X.astype(np.float32)
        labels_sample = data['marker_cell_type'] if 'marker_cell_type' in data.columns else None
    return X_sample, labels_sample, data, numeric_cols

def simple_plot_distributions(data, numeric_cols, log, output_dir):
    log('Plotting feature distributions...')
    for col in tqdm(numeric_cols, desc='Plotting Distributions'):
        plt.figure(figsize=(8, 6))
        data[col].hist(bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
        plt.title(f'{col} Distribution', fontweight='bold', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir / f'{col}_distribution.png', dpi=300)
        plt.close()

def simple_save_summary_report(params, features, output_dir):
    report = []
    report.append('='*60)
    report.append('SIMPLE FEATURE EXTRACTION SUMMARY REPORT')
    report.append('='*60)
    report.append('Parameters:')
    for k, v in params.items():
        report.append(f'- {k}: {v}')
    report.append('\nBasic Features:')
    for k, v in features.items():
        report.append(f'- {k}: {v}')
    with open(output_dir / 'feature_extraction_simple_summary.txt', 'w') as f:
        f.write('\n'.join(report))

def run_simple_feature_extraction(data_path, output_dir, log):
    X_sample, labels_sample, data, numeric_cols = simple_load_and_preprocess(data_path, log, output_dir)
    params = {
        'random_state': RANDOM_STATE,
        'max_points': MAX_POINTS,
        'outlier_std': OUTLIER_STD,
        'numeric_cols': numeric_cols
    }
    with open(output_dir / 'feature_extraction_simple_parameters.json', 'w') as f:
        json.dump(params, f, indent=2)
    features = simple_extract_basic_features(data)
    simple_plot_distributions(data, numeric_cols, log, output_dir)
    simple_save_summary_report(params, features, output_dir)
    log(f'Simple feature extraction complete. See results in {output_dir}/')

# ========== COMPREHENSIVE FEATURE EXTRACTION CLASS ========== #
class FeatureExtractor:
    """Comprehensive feature extraction for single-cell RNA sequencing data"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.features = None
        self.labels = None
        self.scaled_features = None
    def load_and_preprocess_data(self, chunk_size=10000, max_chunks=50):
        print("=" * 60)
        print("FEATURE EXTRACTION FOR SINGLE-CELL RNA DATA")
        print("=" * 60)
        print(f"Loading data from {self.data_path}...")
        all_chunks = []
        chunk_count = 0
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"Processed {chunk_count} chunks...")
            feature_cols = ['nCount_RNA', 'nFeature_RNA', 'RNA_snn_res.0.8', 'seurat_clusters']
            label_cols = ['marker_cell_type', 'scina_cell_type', 'predicted.id']
            chunk_features = chunk[feature_cols].copy()
            chunk_labels = chunk[label_cols].copy()
            valid_mask = ~(chunk_features.isnull().any(axis=1) | chunk_labels.isnull().any(axis=1))
            chunk_features = chunk_features[valid_mask]
            chunk_labels = chunk_labels[valid_mask]
            if len(chunk_features) > 0:
                all_chunks.append(pd.concat([chunk_features, chunk_labels], axis=1))
            if chunk_count >= max_chunks:
                break
        self.data = pd.concat(all_chunks, ignore_index=True)
        print(f"Loaded {len(self.data):,} cells with {len(self.data.columns)} features")
        self.features = self.data[['nCount_RNA', 'nFeature_RNA', 'RNA_snn_res.0.8', 'seurat_clusters']]
        self.labels = self.data[['marker_cell_type', 'scina_cell_type', 'predicted.id']]
        print(f"Feature shape: {self.features.shape}")
        print(f"Label shape: {self.labels.shape}")
        return self.data
    def basic_statistical_features(self):
        features = {}
        for col in self.features.columns:
            features[f'{col}_mean'] = self.features[col].mean()
            features[f'{col}_std'] = self.features[col].std()
            features[f'{col}_median'] = self.features[col].median()
            features[f'{col}_min'] = self.features[col].min()
            features[f'{col}_max'] = self.features[col].max()
        return features
    def _calculate_entropy(self, series):
        counts = series.value_counts()
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy
    def scale_features(self, method='standard'):
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        self.scaled_features = pd.DataFrame(scaler.fit_transform(self.features), columns=self.features.columns)
        return self.scaled_features
    def dimensionality_reduction(self, method='pca', n_components=2):
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=RANDOM_STATE)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        reduced = reducer.fit_transform(self.scaled_features)
        return reduced, reducer
    def feature_selection(self, method='mutual_info', n_features=2):
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
        elif method == 'anova':
            selector = SelectKBest(f_classif, k=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        selected = selector.fit_transform(self.scaled_features, self.labels['marker_cell_type'])
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        return selected, scores, selected_indices
    def create_visualizations(self, output_dir):
        pca_features, _ = self.dimensionality_reduction('pca', 2)
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_features[:, 0], pca_features[:, 1], s=10, alpha=0.7)
        plt.title('PCA Scatter Plot', fontsize=16, fontweight='bold')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_scatter.png', dpi=300)
        plt.close()
    def save_extracted_features(self, output_dir):
        self.scaled_features.to_csv(output_dir / 'scaled_features.csv', index=False)
        basic_features = self.basic_statistical_features()
        basic_features_df = pd.DataFrame([basic_features])
        basic_features_df.to_csv(output_dir / 'statistical_features.csv', index=False)
        pca_features, pca_reducer = self.dimensionality_reduction('pca', 2)
        pca_df = pd.DataFrame(pca_features, columns=['PC1', 'PC2'])
        pca_df.to_csv(output_dir / 'pca_features.csv', index=False)
        tsne_features, _ = self.dimensionality_reduction('tsne', 2)
        tsne_df = pd.DataFrame(tsne_features, columns=['tSNE1', 'tSNE2'])
        tsne_df.to_csv(output_dir / 'tsne_features.csv', index=False)
        umap_features, _ = self.dimensionality_reduction('umap', 2)
        umap_df = pd.DataFrame(umap_features, columns=['UMAP1', 'UMAP2'])
        umap_df.to_csv(output_dir / 'umap_features.csv', index=False)
        selected_features, feature_scores, selected_indices = self.feature_selection('mutual_info', 2)
        feature_selection_df = pd.DataFrame({
            'feature': self.features.columns,
            'score': feature_scores,
            'selected': [i in selected_indices for i in range(len(self.features.columns))]
        })
        feature_selection_df.to_csv(output_dir / 'feature_selection_results.csv', index=False)
        plt.savefig(output_dir / 'feature_extraction_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        report = []
        report.append("="*60)
        report.append("FEATURE EXTRACTION SUMMARY REPORT")
        report.append("="*60)
        report.append("OUTPUT FILES:")
        report.append("- scaled_features.csv: Standardized feature values")
        report.append("- statistical_features.csv: Statistical summary features")
        report.append("- pca_features.csv: PCA reduced features")
        report.append("- tsne_features.csv: t-SNE reduced features")
        report.append("- umap_features.csv: UMAP reduced features")
        report.append("- feature_selection_results.csv: Feature importance scores")
        report.append("- feature_extraction_dashboard.png: Comprehensive visualization")
        with open(output_dir / 'feature_extraction_report.txt', 'w') as f:
            f.write('\n'.join(report))
        print("\u2713 Summary report saved")

def run_comprehensive_feature_extraction(data_path, output_dir, log):
    extractor = FeatureExtractor(data_path)
    extractor.load_and_preprocess_data()
    extractor.basic_statistical_features()
    extractor.scale_features('standard')
    extractor.dimensionality_reduction('pca', 2)
    extractor.feature_selection('mutual_info', 2)
    extractor.create_visualizations(output_dir)
    extractor.save_extracted_features(output_dir)
    log(f'Comprehensive feature extraction complete. See results in {output_dir}/')

# ========== MAIN ========== #
def main():
    parser = argparse.ArgumentParser(description="Feature Extraction (Simple or Comprehensive)")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    parser.add_argument('--mode', type=str, choices=['comprehensive', 'simple'], default='comprehensive', help='Extraction mode: comprehensive (default) or simple')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config.get('data_path', DEFAULT_DATA_PATH)
    results_dir = config.get('results_dir', 'results/')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = Path(results_dir) / script_name
    output_dir.mkdir(parents=True, exist_ok=True)
    def log(msg):
        print(msg)
        with open(output_dir / 'feature_extraction_log.txt', 'a') as f:
            f.write(msg + '\n')
    with open(output_dir / 'feature_extraction_log.txt', 'w') as f:
        f.write('')
    if args.mode == 'simple':
        run_simple_feature_extraction(data_path, output_dir, log)
        print("\n[INFO] Ran SIMPLE feature extraction mode.")
    else:
        run_comprehensive_feature_extraction(data_path, output_dir, log)
        print("\n[INFO] Ran COMPREHENSIVE feature extraction mode.")
    print(f"[INFO] All outputs saved in: {output_dir}")

if __name__ == '__main__':
    main() 