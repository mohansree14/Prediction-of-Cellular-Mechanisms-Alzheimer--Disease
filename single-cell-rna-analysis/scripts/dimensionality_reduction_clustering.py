import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import os
from tqdm import tqdm
import json
import argparse
import yaml
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette('tab20')  # Colorblind-friendly palette

# ========== CONFIGURATION ========== #
RANDOM_STATE = 42
MAX_POINTS = 10000
UMAP_POINTS = 2000
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
N_CLUSTERS = 8
OUTLIER_STD = 5  # Remove points >5 std from mean

# ========== UTILITY FUNCTIONS ========== #
def save_params(params, outdir):
    with open(outdir / 'analysis_parameters.json', 'w') as f:
        json.dump(params, f, indent=2)

# ========== DATA LOADING & PREPROCESSING ========== #
def load_and_preprocess(data_path):
    """Load and preprocess the data"""
    log('Loading data...')
    data = pd.read_csv(data_path)  # Use config-based path
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    X = data[numeric_cols].values
    # Remove outliers
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mask = np.all(np.abs(X_scaled) < OUTLIER_STD, axis=1)
    X = X[mask]
    data = data.iloc[mask].reset_index(drop=True)
    # Subsample for efficiency
    if X.shape[0] > MAX_POINTS:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(X.shape[0], MAX_POINTS, replace=False)
        X_sample = X[idx].astype(np.float32)
        labels_sample = data['marker_cell_type'].iloc[idx].reset_index(drop=True) if 'marker_cell_type' in data.columns else None
        log(f"Subsampled {MAX_POINTS} rows for analysis (original: {X.shape[0]})")
    else:
        X_sample = X.astype(np.float32)
        labels_sample = data['marker_cell_type'] if 'marker_cell_type' in data.columns else None
    return X_sample, labels_sample, data, numeric_cols

# ========== PCA ========== #
def run_pca(X, output_dir, labels=None):
    log('Performing PCA...')
    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    # Scatter plot (first 2 components)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab20', s=10, alpha=0.7, legend='full')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.7)
    plt.title('PCA of Extracted Features', fontsize=16, fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_scatter.png', dpi=300)
    plt.close()
    # Explained variance plot
    explained_var = pca.explained_variance_ratio_
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(explained_var)+1), np.cumsum(explained_var), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_explained_variance.png', dpi=300)
    plt.close()
    return X_pca, pca

# ========== t-SNE ========== #
def run_tsne(X, output_dir, labels=None, perplexity=TSNE_PERPLEXITY):
    log('Performing t-SNE...')
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab20', s=10, alpha=0.7, legend='full')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, alpha=0.7)
    plt.title(f't-SNE (perplexity={perplexity})', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_scatter.png', dpi=300)
    plt.close()
    return X_tsne

# ========== UMAP ========== #
def run_umap(X, output_dir, labels=None, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST):
    try:
        import umap
        log('Performing UMAP...')
        # Use a smaller sample for UMAP if needed
        umap_points = min(UMAP_POINTS, X.shape[0])
        if X.shape[0] > umap_points:
            np.random.seed(RANDOM_STATE)
            umap_idx = np.random.choice(X.shape[0], umap_points, replace=False)
            X_umap_input = X[umap_idx]
            labels_umap = labels.iloc[umap_idx].reset_index(drop=True) if labels is not None else None
            log(f"Subsampled {umap_points} rows for UMAP (original: {X.shape[0]})")
        else:
            X_umap_input = X
            labels_umap = labels
        umap_model = umap.UMAP(n_components=2, random_state=RANDOM_STATE, metric='euclidean', n_neighbors=n_neighbors, min_dist=min_dist)
        X_umap = umap_model.fit_transform(X_umap_input)
        plt.figure(figsize=(8, 6))
        if labels_umap is not None:
            sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels_umap, palette='tab20', s=10, alpha=0.7, legend='full')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            plt.scatter(X_umap[:, 0], X_umap[:, 1], s=10, alpha=0.7)
        plt.title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.tight_layout()
        plt.savefig(output_dir / 'umap_scatter.png', dpi=300)
        plt.close()
        return X_umap
    except ImportError:
        log('UMAP not installed, skipping UMAP step.')
        return None

# ========== KMEANS CLUSTERING ========== #
def run_kmeans(X, output_dir, labels=None, max_clusters=15):
    log('Performing k-means clustering...')
    # Elbow method
    inertias = []
    silhouettes = []
    K_range = range(2, max_clusters+1)
    for k in tqdm(K_range, desc='KMeans Elbow/Silhouette'):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        clusters = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, clusters))
    # Plot elbow
    plt.figure(figsize=(6,4))
    plt.plot(list(K_range), inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('KMeans Elbow Method')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'kmeans_elbow.png', dpi=300)
    plt.close()
    # Plot silhouette
    plt.figure(figsize=(6,4))
    plt.plot(list(K_range), silhouettes, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('KMeans Silhouette Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'kmeans_silhouette.png', dpi=300)
    plt.close()
    # Use optimal k
    best_k = K_range[np.argmax(silhouettes)]
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clusters, palette='tab20', s=10, alpha=0.7, legend='full')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title='Cluster')
    plt.title(f'K-means Clusters (k={best_k})', fontsize=16, fontweight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(output_dir / 'kmeans_pca_scatter.png', dpi=300)
    plt.close()
    # Save cluster assignments
    cluster_df = pd.DataFrame({'PC1': X[:, 0], 'PC2': X[:, 1], 'Cluster': clusters})
    if labels is not None:
        cluster_df['CellType'] = labels.values
    cluster_df.to_csv(output_dir / 'kmeans_clusters.csv', index=False)
    return clusters, best_k, max(silhouettes)

# ========== SUMMARY REPORT ========== #
def save_summary_report(output_dir, params, best_k, silhouette):
    report = []
    report.append('='*60)
    report.append('DIMENSIONALITY REDUCTION & CLUSTERING SUMMARY REPORT')
    report.append('='*60)
    report.append(f'Parameters:')
    for k, v in params.items():
        report.append(f'- {k}: {v}')
    report.append(f'\nKMeans:')
    report.append(f'- Optimal clusters: {best_k}')
    report.append(f'- Best silhouette score: {silhouette:.3f}')
    report.append(f'\nSee plots and CSVs in {output_dir}')
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(report))

# ========== MAIN ========== #
def main():
    parser = argparse.ArgumentParser(description="Dimensionality Reduction & Clustering")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config['data_path']
    results_dir = config.get('results_dir', 'results/')
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = Path(results_dir) / script_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redefine log and save functions to use this directory
    def log(msg):
        print(msg)
        with open(output_dir / 'analysis_log.txt', 'a') as f:
            f.write(msg + '\n')
    def save_params(params):
        with open(output_dir / 'analysis_parameters.json', 'w') as f:
            json.dump(params, f, indent=2)
    def save_summary_report(params, best_k, silhouette):
        report = []
        report.append('='*60)
        report.append('DIMENSIONALITY REDUCTION & CLUSTERING SUMMARY REPORT')
        report.append('='*60)
        report.append(f'Parameters:')
        for k, v in params.items():
            report.append(f'- {k}: {v}')
        report.append(f'\nKMeans:')
        report.append(f'- Optimal clusters: {best_k}')
        report.append(f'- Best silhouette score: {silhouette:.3f}')
        report.append(f'\nSee plots and CSVs in {output_dir}')
        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write('\n'.join(report))

    # Clear log
    with open(output_dir / 'analysis_log.txt', 'w') as f:
        f.write('')
    # Load and preprocess
    X_sample, labels_sample, data, numeric_cols = load_and_preprocess(data_path)
    params = {
        'random_state': RANDOM_STATE,
        'max_points': MAX_POINTS,
        'umap_points': UMAP_POINTS,
        'tsne_perplexity': TSNE_PERPLEXITY,
        'umap_n_neighbors': UMAP_N_NEIGHBORS,
        'umap_min_dist': UMAP_MIN_DIST,
        'n_clusters': N_CLUSTERS,
        'outlier_std': OUTLIER_STD,
        'numeric_cols': numeric_cols
    }
    save_params(params)
    # PCA
    X_pca, pca_model = run_pca(X_sample, output_dir, labels_sample)
    # t-SNE
    X_tsne = run_tsne(X_sample, output_dir, labels_sample, perplexity=TSNE_PERPLEXITY)
    # UMAP
    X_umap = run_umap(X_sample, output_dir, labels_sample, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST)
    # KMeans (on PCA)
    clusters, best_k, silhouette = run_kmeans(X_pca, output_dir, labels_sample)
    # Save summary
    save_summary_report(params, best_k, silhouette)
    log(f'All dimensionality reduction and clustering results saved to {output_dir}')

if __name__ == '__main__':
    main()