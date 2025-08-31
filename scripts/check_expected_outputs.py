import os
import glob
from tabulate import tabulate

# Define expected outputs and their descriptions
EXPECTED_OUTPUTS = [
    {
        'Type': 'Cell Clustering',
        'Description': 'UMAP/t-SNE/cluster plots and cluster assignment files',
        'Patterns': [
            'results/*/umap_scatter.png',
            'results/*/tsne_scatter.png',
            'results/*/kmeans_clusters.csv',
            'results/*/pca_scatter.png',
            'single-cell-rna-analysis/results/*/umap_scatter.png',
            'single-cell-rna-analysis/results/*/tsne_scatter.png',
            'single-cell-rna-analysis/results/*/kmeans_clusters.csv',
            'single-cell-rna-analysis/results/*/pca_scatter.png',
        ]
    },
    {
        'Type': 'Cellular Interaction Networks',
        'Description': 'GNN graph visualizations or edge lists',
        'Patterns': [
            'results/*/gnn_graph*.png',
            'results/*/interaction*.png',
            'results/*/network*.png',
            'results/*/edge*.csv',
            'single-cell-rna-analysis/results/*/gnn_graph*.png',
            'single-cell-rna-analysis/results/*/interaction*.png',
            'single-cell-rna-analysis/results/*/network*.png',
            'single-cell-rna-analysis/results/*/edge*.csv',
        ]
    },
    {
        'Type': 'Gene Importance',
        'Description': 'Feature importance CSVs or bar plots',
        'Patterns': [
            'results/*/feature_importance*.png',
            'results/*/feature_importance*.csv',
            'single-cell-rna-analysis/results/*/feature_importance*.png',
            'single-cell-rna-analysis/results/*/feature_importance*.csv',
        ]
    },
    {
        'Type': 'Prediction Scores',
        'Description': 'ROC, confusion matrix, metrics, probability maps',
        'Patterns': [
            'results/*/evaluation_*roc*.png',
            'results/*/evaluation_*confusion*.png',
            'results/*/evaluation_*metrics*.png',
            'results/*/evaluation_results.png',
            'results/*/evaluation_summary.txt',
            'results/*/training_summary.txt',
            'single-cell-rna-analysis/results/*/roc*.png',
            'single-cell-rna-analysis/results/*/confusion*.png',
            'single-cell-rna-analysis/results/*/metrics*.png',
            'single-cell-rna-analysis/results/*/evaluation_results.png',
            'single-cell-rna-analysis/results/*/evaluation_summary.txt',
            'single-cell-rna-analysis/results/*/training_summary.txt',
        ]
    },
    {
        'Type': 'Pathway Enrichment',
        'Description': 'Pathway enrichment plots or reports',
        'Patterns': [
            'results/*/pathway*.png',
            'results/*/enrichment*.png',
            'results/*/enrichment*.txt',
            'single-cell-rna-analysis/results/*/pathway*.png',
            'single-cell-rna-analysis/results/*/enrichment*.png',
            'single-cell-rna-analysis/results/*/enrichment*.txt',
        ]
    },
    {
        'Type': 'Evaluation Metrics',
        'Description': 'Accuracy, AUC, Precision, Recall, F1-score, ablation',
        'Patterns': [
            'results/*/evaluation_summary.txt',
            'results/*/training_summary.txt',
            'results/*/evaluation_report.json',
            'results/*/training_results.json',
            'single-cell-rna-analysis/results/*/evaluation_summary.txt',
            'single-cell-rna-analysis/results/*/training_summary.txt',
            'single-cell-rna-analysis/results/*/evaluation_report.json',
            'single-cell-rna-analysis/results/*/training_results.json',
        ]
    },
]


def find_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return files


def main():
    summary = []
    for output in EXPECTED_OUTPUTS:
        found_files = find_files(output['Patterns'])
        summary.append([
            output['Type'],
            output['Description'],
            'YES' if found_files else 'NO',
            '\n'.join(found_files) if found_files else '—'
        ])

    headers = ["Output Type", "Description", "Found?", "Files Found"]
    table_txt = tabulate(summary, headers=headers, tablefmt="github")
    table_tsv = tabulate(summary, headers=headers, tablefmt="tsv")

    print("\n===== SUMMARY OF EXPECTED OUTPUTS =====\n")
    print(table_txt)
    print("\nIf any output is missing, please check your scripts or pipeline to generate it.")

    # Save to a dedicated summary folder in results
    summary_dir = os.path.join('results', 'expected_outputs_summary')
    os.makedirs(summary_dir, exist_ok=True)
    txt_path = os.path.join(summary_dir, 'expected_outputs_summary.txt')
    tsv_path = os.path.join(summary_dir, 'expected_outputs_summary.tsv')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(table_txt)
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write(table_tsv)
    print(f"\n[INFO] Output summary saved in both text and table format:")
    print(f"  - Text table:      @{txt_path}")
    print(f"  - Spreadsheet/TSV: @{tsv_path}")
    print("\nYou can include these files as supplementary material or for reporting in your dissertation.")

if __name__ == "__main__":
    main() 