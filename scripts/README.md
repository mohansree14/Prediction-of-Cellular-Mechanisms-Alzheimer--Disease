# Analysis Pipeline Scripts

This folder contains the main scripts for the single-cell RNA-seq analysis pipeline. All scripts are organized, numbered, and designed to be run in sequence for a complete, reproducible workflow.

## Pipeline Order & Script Descriptions

| Step | Script Name                        | Purpose/Functionality                                 |
|------|------------------------------------|------------------------------------------------------|
| 01   | 01_analyze_csv.py                  | Data/CSV inspection and summary                      |
| 02   | 02_feature_extraction.py           | Feature extraction (comprehensive/simple via --mode) |
| 03   | 03_analysis_and_visualization.py   | Visualization, detailed, and advanced analysis       |
| 04   | 04_run_hybrid_modeling.py          | Hybrid GNN+CVAE pipeline                             |
| 05   | 05_train_deep_learning_models.py   | Deep learning (CVAE, GNN) training                   |
| 06   | 07_train_random_forest.py          | Random Forest baseline training                      |
| 07   | 08_pathway_enrichment_analysis.py  | Pathway enrichment analysis                          |
| 08   | 09_generate_synthetic_networks.py  | Synthetic network generation                         |
| 09   | 10_train_rf_with_all_features.py   | Random Forest on all features, SHAP                  |
| 10   | 11_train_dig_ensemble.py           | DIG-Ensemble model                                   |

## How to Run the Full Pipeline

The recommended way to run the entire pipeline is via the main orchestrator:

```sh
python main.py
```

This will execute all steps in order, using the configuration in `config/config.yaml`.

## How to Run Individual Steps

You can also run any script individually. For example:

```sh
python scripts/02_feature_extraction.py --mode comprehensive --config config/config.yaml
python scripts/03_analysis_and_visualization.py --type detailed --config config/config.yaml
```

- Use `--mode` or `--type` arguments as needed for scripts with multiple functionalities.
- All scripts require the `--config` argument to specify the config file.

## Configuration

- All scripts use a central config file: `config/config.yaml`.
- This file defines the input data path and the main results directory.
- Example:

```yaml
data_path: data/raw/combined_data_fixed.csv
results_dir: results/
```

## Output Structure

- Each script creates its own subfolder in the main `results/` directory (e.g., `results/02_feature_extraction/`).
- Logs, plots, and output files are saved in these subfolders for easy tracking and reproducibility.

## Notes

- Only the main, numbered scripts are required for the standard pipeline. Other scripts in this folder are utilities or legacy and can be ignored or archived.
- For troubleshooting, check the log files in each script's results subfolder.

---

For questions or to extend the pipeline, see the code comments or contact the project maintainer. 