import argparse
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("tab20")

# ========== CONFIGURATION ========== #
RANDOM_STATE = 42
MAX_POINTS = 50000
OUTLIER_STD = 5
DEFAULT_DATA_PATH = "data/raw/combined_data_fixed.csv"

# ========== GENERAL VISUALIZATION FUNCTIONS ========== #
def general_load_and_preprocess(data_path, log):
    data = pd.read_csv(data_path)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
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
        log(f"Subsampled {MAX_POINTS} rows for visualization (original: {X.shape[0]})")
    else:
        X_sample = X.astype(np.float32)
        labels_sample = data['marker_cell_type'] if 'marker_cell_type' in data.columns else None
    return X_sample, labels_sample, data, numeric_cols, categorical_cols

def general_plot_distributions(data, numeric_cols, log, output_dir):
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

def general_plot_correlation_heatmap(data, numeric_cols, log, output_dir):
    log('Plotting correlation heatmap...')
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def general_save_summary_report(output_dir):
    with open(output_dir / 'visualization_insights.txt', 'w') as f:
        f.write('')

def general_save_params(params, output_dir):
    with open(output_dir / 'visualization_parameters.json', 'w') as f:
        json.dump(params, f, indent=2)

def run_general_visualization(data_path, output_dir, log):
    X_sample, labels_sample, data, numeric_cols, categorical_cols = general_load_and_preprocess(data_path, log)
    params = {
        'random_state': RANDOM_STATE,
        'max_points': MAX_POINTS,
        'outlier_std': OUTLIER_STD,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    general_save_params(params, output_dir)
    general_plot_distributions(data, numeric_cols, log, output_dir)
    general_plot_correlation_heatmap(data, numeric_cols, log, output_dir)
    general_save_summary_report(output_dir)
    log(f'Visualization complete. See results in {output_dir}/')

# ========== DETAILED ANALYSIS FUNCTIONS ========== #
def run_detailed_analysis(data_path, output_dir, log):
    from src.visualization.plotting_utils import ProfessionalPlottingUtils
    if not Path(data_path).exists():
        print(f"[ERROR] Data file not found: {data_path}\nPlease ensure the file exists at the specified location.")
        sys.exit(1)
    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path, nrows=100000)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)
    print(f"Data loaded: {data.shape[0]:,} rows, {data.shape[1]} columns.")
    plotter = ProfessionalPlottingUtils()
    plotter.create_quality_control_dashboard(data, save_path=output_dir / "quality_control/")
    plotter.create_cell_type_comparison_plots(data, save_path=output_dir / "cell_types/")
    plotter.create_clustering_analysis_plots(data, save_path=output_dir / "clustering/")
    plotter.create_correlation_heatmap(data, save_path=output_dir)
    plotter.create_publication_figure(data, save_path=output_dir / "publication/")
    print("\nAll figures saved to the appropriate results subdirectories:")
    print(f"{output_dir}")
    log('Detailed analysis complete.')

# ========== ADVANCED ANALYSIS FUNCTIONS ========== #
def advanced_clustering_analysis(data, log, output_dir):
    log('Creating cluster size distribution plot...')
    cluster_col = data['seurat_clusters'].astype(str)
    cluster_counts = cluster_col.value_counts().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Cluster ID', fontsize=14)
    ax.set_ylabel('Number of Cells', fontsize=14)
    ax.set_title('Top 10 Clusters by Cell Count', fontsize=16, fontweight='bold')
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_size_distribution.png', dpi=300)
    plt.close()
    log('Creating cell type composition by cluster plot...')
    cell_type_col = data['marker_cell_type'].astype(str)
    top_clusters = cluster_counts.index[:5]
    top_cell_types = cell_type_col.value_counts().head(5).index
    heatmap_data = []
    for cluster in top_clusters:
        cluster_data = data[cluster_col == cluster]
        cluster_cell_types = cluster_data['marker_cell_type'].astype(str).value_counts()
        row = [cluster_cell_types.get(cell_type, 0) for cell_type in top_cell_types]
        heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data).T
    # Robustness: Only plot if heatmap_data is not empty
    if heatmap_data.size == 0 or heatmap_data.shape[0] == 0 or heatmap_data.shape[1] == 0:
        log('[WARNING] Skipping cell type composition heatmap: no data for top clusters/cell types.')
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Cluster ID', fontsize=14)
        ax.set_ylabel('Cell Type', fontsize=14)
        ax.set_title('Cell Type Composition (Top 5 Clusters × Top 5 Cell Types)', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(top_clusters)))
        ax.set_xticklabels(top_clusters)
        ax.set_yticks(range(len(top_cell_types)))
        ax.set_yticklabels(top_cell_types)
        for i in range(len(top_cell_types)):
            for j in range(len(top_clusters)):
                ax.text(j, i, str(heatmap_data[i, j]), ha='center', va='center', color='black', fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Cell Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'cell_type_composition_by_cluster.png', dpi=300)
        plt.close()
        log('Cluster analysis plots saved as cluster_size_distribution.png and cell_type_composition_by_cluster.png')

def run_advanced_analysis(data_path, output_dir, log):
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
        log(f"Subsampled {MAX_POINTS} rows for advanced analysis (original: {X.shape[0]})")
    else:
        X_sample = X.astype(np.float32)
        labels_sample = data['marker_cell_type'] if 'marker_cell_type' in data.columns else None
    params = {
        'random_state': RANDOM_STATE,
        'max_points': MAX_POINTS,
        'outlier_std': OUTLIER_STD,
        'numeric_cols': numeric_cols
    }
    with open(output_dir / 'advanced_analysis_parameters.json', 'w') as f:
        json.dump(params, f, indent=2)
    advanced_clustering_analysis(data, log, output_dir)
    report = []
    report.append('='*60)
    report.append('ADVANCED ANALYSIS SUMMARY REPORT')
    report.append('='*60)
    report.append('Parameters:')
    for k, v in params.items():
        report.append(f'- {k}: {v}')
    with open(output_dir / 'advanced_analysis_summary.txt', 'w') as f:
        f.write('\n'.join(report))
    log(f'Advanced analysis complete. See results in {output_dir}/')

# ========== MAIN ========== #
def main():
    parser = argparse.ArgumentParser(description="Analysis and Visualization (General, Detailed, or Advanced)")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
    parser.add_argument('--type', type=str, choices=['general', 'detailed', 'advanced'], default='general', help='Type of analysis/visualization to run')
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
        with open(output_dir / f'{args.type}_analysis_log.txt', 'a') as f:
            f.write(msg + '\n')
    with open(output_dir / f'{args.type}_analysis_log.txt', 'w') as f:
        f.write('')
    if args.type == 'general':
        run_general_visualization(data_path, output_dir, log)
        print("\n[INFO] Ran GENERAL visualization mode.")
    elif args.type == 'detailed':
        run_detailed_analysis(data_path, output_dir, log)
        print("\n[INFO] Ran DETAILED analysis mode.")
    else:
        run_advanced_analysis(data_path, output_dir, log)
        print("\n[INFO] Ran ADVANCED analysis mode.")
    print(f"[INFO] All outputs saved in: {output_dir}")

if __name__ == '__main__':
    main() 