import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file')
parser.add_argument('--sample', action='store_true', help='Run on a sample of 10,000 rows for quick testing')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
data_path = Path(config['data_path'])
results_dir = Path(config.get('results_dir', 'results/')) / '10_train_rf_with_all_features'
results_dir.mkdir(parents=True, exist_ok=True)

# Paths to feature files (relative to results_dir or data/processed)
main_path = data_path
pca_path = results_dir.parent.parent / 'results' / 'feature_extraction' / 'pca_features.csv'
umap_path = results_dir.parent.parent / 'results' / 'feature_extraction' / 'umap_features.csv'
tsne_path = results_dir.parent.parent / 'results' / 'feature_extraction' / 'tsne_features.csv'
stat_path = results_dir.parent.parent / 'results' / 'feature_extraction' / 'statistical_features.csv'

# Load all available features
if args.sample:
    print("Running in SAMPLE MODE: Only loading the first 10,000 rows from each file.")
    main_df = pd.read_csv(main_path, nrows=10000)
else:
    main_df = pd.read_csv(main_path)

# Try to load each feature file, skip if not found
feature_dfs = [main_df]
for path in [pca_path, umap_path, tsne_path, stat_path]:
    if os.path.exists(path):
        if args.sample:
            df = pd.read_csv(path, nrows=10000)
        else:
            df = pd.read_csv(path)
        feature_dfs.append(df)

# Concatenate all features (assume same row order)
all_features = pd.concat(feature_dfs, axis=1)

# Remove duplicate columns
all_features = all_features.loc[:, ~all_features.columns.duplicated()]

# Drop non-feature columns (keep only numeric)
non_feature_cols = [col for col in all_features.columns if all_features[col].dtype == 'object' or 'barcode' in col or 'cell_type' in col or 'seurat_clusters' in col]
X = all_features.drop(columns=non_feature_cols, errors='ignore')

# Use cell type or cluster as target
target_col = None
for col in ['cell_type', 'seurat_clusters', 'CellType', 'cluster']:
    if col in all_features.columns:
        target_col = col
        break
if target_col is None:
    raise ValueError('No cell type or cluster column found in data!')
y = all_features[target_col].astype(str)
y = y.replace(['nan', 'None', 'unknown', 'Unknown'], 'Unlabeled')

# Remove classes with fewer than 2 samples
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
mask = y.isin(valid_classes)
X = X[mask]
y = y[mask]

# Limit to top 200 features by variance
variances = X.var().sort_values(ascending=False)
top_features = variances.head(200).index
X = X[top_features]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Random Forest with even fewer trees
rf = RandomForestClassifier(n_estimators=50, max_depth=None, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Save results
summary_file = results_dir / 'training_summary_all_features.txt'
with open(summary_file, 'w') as f:
    f.write('RANDOM FOREST TRAINING SUMMARY (ALL FEATURES)\n')
    f.write('==========================================\n\n')
    f.write('Training completed successfully!\n\n')
    f.write(f'Total features used: {X.shape[1]}\n')
    f.write(f'Number of classes: {len(np.unique(y))}\n')
    f.write(f'Final Accuracy: {acc:.4f}\n')
    f.write(f'Final F1-Score (Weighted): {f1:.4f}\n\n')
    f.write('Classification Report:\n')
    f.write(classification_report(y_test, y_pred))
    f.write(f'\nResults saved to: {results_dir.resolve()}\n')
print(f"Training complete. Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
print(f"Summary saved to {summary_file}")

# Save trained model
model_file = results_dir / 'random_forest_all_features.joblib'
joblib.dump(rf, model_file)
print(f"Model saved to {model_file}")

# Plot and save feature importances (summary plot, remove zero importances)
importances = rf.feature_importances_
feature_names = X.columns

# Filter out features with zero importance
nonzero_mask = importances > 0
importances_nonzero = importances[nonzero_mask]
feature_names_nonzero = feature_names[nonzero_mask]

# Sort by importance descending
sorted_idx = np.argsort(importances_nonzero)[::-1]
importances_sorted = importances_nonzero[sorted_idx]
feature_names_sorted = feature_names_nonzero[sorted_idx]

plt.figure(figsize=(12, max(6, 0.3*len(importances_sorted))))
plt.title('Feature Importances (Random Forest, Nonzero Only)')
plt.barh(range(len(importances_sorted)), importances_sorted, align='center')
plt.yticks(range(len(importances_sorted)), feature_names_sorted, fontsize=10)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Most important at top
plt.tight_layout()
importance_plot_file = results_dir / 'feature_importances_all_features.png'
plt.savefig(importance_plot_file, dpi=200)
plt.close()
print(f"Feature importance summary plot saved to {importance_plot_file}")

# SHAP summary plot (beeswarm)
try:
    import shap
except ImportError:
    print("shap library not found. Please install it with: pip install shap")
    shap = None

if shap is not None:
    # Use a sample of the training data for SHAP (to save time/memory)
    X_shap = X_train.copy()
    if len(X_shap) > 1000:
        X_shap = X_shap.sample(1000, random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_shap)

    # For multiclass, use mean(|shap|) across classes
    if isinstance(shap_values, list):
        # shape: [n_classes][n_samples, n_features]
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_abs = np.abs(shap_values)

    # Remove features with mean(|SHAP|)==0
    mean_shap = shap_abs.mean(axis=0)
    nonzero_shap_mask = mean_shap > 0
    X_shap_nonzero = X_shap.iloc[:, nonzero_shap_mask]
    shap_values_nonzero = shap_values
    if isinstance(shap_values, list):
        shap_values_nonzero = [sv[:, nonzero_shap_mask] for sv in shap_values]
    else:
        shap_values_nonzero = shap_values[:, nonzero_shap_mask]

    # SHAP summary plot (beeswarm)
    plt.figure(figsize=(12, max(6, 0.3*X_shap_nonzero.shape[1])))
    shap.summary_plot(
        shap_values_nonzero,
        X_shap_nonzero,
        plot_type="dot",
        show=False
    )
    plt.tight_layout()
    plt.savefig(importance_plot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {importance_plot_file}")
else:
    print("SHAP plot not generated because shap is not installed.") 