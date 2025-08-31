#!/usr/bin/env python3
"""
Professional Orchestration Script: Full Analysis Pipeline
Runs all major analysis steps in the correct order, covering both the main project and the single-cell subproject.
All paths are relative, and config files are passed as needed. Ready for cluster or local use.
"""
import subprocess
import sys
from pathlib import Path

# Define the steps in professional order
PIPELINE_STEPS = [
    # Data inspection & cleaning
    ("scripts/01_run_csv_inspection.py", "--config config/config.yaml"),
    ("scripts/02_preprocess_data.py", "--config config/config.yaml"),
    # Feature engineering
    ("scripts/03_run_feature_engineering.py", "--config config/config.yaml"),
    # Single-cell subproject feature extraction & clustering
    ("scripts/feature_extraction.py", "--config config/config.yaml"),
    ("scripts/dimensionality_reduction_clustering.py", "--config config/config.yaml"),
    ("scripts/visualize_data.py", "--config config/config.yaml"),
    # Model training
    ("scripts/05_train_deep_learning_models.py", "--config config/config.yaml"),
    ("scripts/04_run_hybrid_modeling.py", "--config config/config.yaml"),
    ("scripts/11_train_dig_ensemble.py", "--config config/config.yaml"),
    ("scripts/07_train_random_forest.py", "--config config/config.yaml"),
    ("scripts/10_train_rf_with_all_features.py", ""),
    # Model evaluation
    ("scripts/06_evaluate_models.py", "--config config/config.yaml"),
    # Pathway enrichment
    ("scripts/08_pathway_enrichment_analysis.py", "--config config/config.yaml"),
    # Output validation & reporting
    ("scripts/check_expected_outputs.py", "--config config/config.yaml"),
    ("scripts/run_complete_analysis.py", "--config config/config.yaml"),
]

def run_step(script_path, args):
    script = Path(script_path)
    if not script.exists():
        print(f"[SKIP] Script not found: {script_path}")
        return True  # Not a hard failure
    cmd = [sys.executable, str(script)]
    if args:
        cmd += args.split()
    print(f"\n{'='*80}\n[RUNNING] {script_path} {args}\n{'='*80}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"[SUCCESS] {script_path}")
        return True
    else:
        print(f"[FAIL] {script_path} (see logs for details)")
        return False

def main():
    print("\n" + "#"*80)
    print("PROFESSIONAL FULL ANALYSIS PIPELINE: STARTING")
    print("#"*80 + "\n")
    all_success = True
    for script_path, args in PIPELINE_STEPS:
        success = run_step(script_path, args)
        if not success:
            all_success = False
    print("\n" + "#"*80)
    if all_success:
        print("✅ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    else:
        print("❌ SOME PIPELINE STEPS FAILED. CHECK LOGS FOR DETAILS.")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main() 