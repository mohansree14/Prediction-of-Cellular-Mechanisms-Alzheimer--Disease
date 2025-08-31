# Single-Cell RNA Sequencing Analysis with Hybrid Machine Learning Models

## 🎯 Project Overview

**Title:** Comprehensive Analysis of Single-Cell RNA Sequencing Data Using Hybrid Machine Learning Models  
**Domain:** Bioinformatics, Computational Biology, Machine Learning  
**Data:** 14.7M cells, 30 features, Brain tissue scRNA-seq data  

This dissertation project implements a comprehensive analysis pipeline for single-cell RNA sequencing data using a hybrid approach combining traditional machine learning, deep learning, and ensemble methods.

## 📊 Dataset Information

- **File:** `data/raw/combined_data.csv`
- **Size:** 2.0 GB
- **Cells:** 14,738,154
- **Features:** 30
- **Tissue:** Brain
- **Cell Types:** Oligodendrocytes, Astrocytes, Neurons, etc.

## 🏗️ Project Structure

```
single-cell-rna-analysis/
├── 📁 data/                    # Data storage
│   ├── raw/                   # Original data
│   ├── processed/             # Processed data
│   ├── external/              # Reference data
│   └── intermediate/          # Intermediate files
├── 📁 src/                    # Source code
│   ├── data_processing/       # Data processing pipeline
│   ├── models/                # Machine learning models
│   ├── analysis/              # Analysis modules
│   ├── visualization/         # Visualization modules
│   └── utils/                 # Utility functions
├── 📁 notebooks/              # Jupyter notebooks
├── 📁 results/                # Analysis results
├── 📁 docs/                   # Documentation
├── 📁 scripts/                # Standalone scripts
├── 📁 tests/                  # Unit tests
└── 📁 config/                 # Configuration files
```

## 🎯 Research Objectives

### Primary Objectives:
1. **Cell Type Classification:** Accurate prediction of cell types using hybrid models
2. **Quality Assessment:** Identify high-quality cells and remove low-quality ones
3. **Feature Selection:** Identify important genes/features for classification
4. **Model Comparison:** Compare traditional ML, deep learning, and hybrid approaches

### Secondary Objectives:
1. **Interpretability:** Understand model decisions and feature importance
2. **Scalability:** Handle large-scale data efficiently
3. **Reproducibility:** Ensure reproducible results
4. **Novel Insights:** Discover new biological patterns

## 🧬 Hybrid Model Strategy

### Model 1: Traditional Machine Learning
- **Random Forest:** Robust, interpretable, handles high-dimensional data
- **Support Vector Machine:** Good for binary classification, kernel methods
- **Gradient Boosting:** High performance, feature importance
- **Ensemble Methods:** Combine multiple traditional models

### Model 2: Deep Learning
- **Feed-forward Neural Network:** Baseline deep learning approach
- **Autoencoder:** Dimensionality reduction, feature learning
- **Variational Autoencoder:** Probabilistic modeling, latent space
- **Attention Mechanisms:** Focus on important features

### Model 3: Hybrid Approach
- **Stacked Generalization:** Combine traditional and deep learning
- **Voting Classifiers:** Majority voting across models
- **Weighted Ensembles:** Optimize weights for best performance
- **Pipeline Integration:** Sequential model combination

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- 16GB+ RAM (for large dataset processing)
- Git

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd single-cell-rna-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup the project
python scripts/setup_environment.py
```

### Dependencies
Key packages include:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Traditional ML models
- `tensorflow`, `pytorch` - Deep learning
- `matplotlib`, `seaborn` - Visualization
- `scanpy`, `anndata` - Single-cell analysis
- `jupyter` - Interactive notebooks

## 🚀 Quick Start

### 1. Data Exploration
```bash
python scripts/run_complete_analysis.py --mode exploration
```

### 2. Model Training
```bash
python scripts/run_model_training.py --models all
```

### 3. Hybrid Analysis
```bash
python scripts/run_hybrid_analysis.py
```

### 4. Generate Reports
```bash
python scripts/generate_report.py
```

## 📈 Expected Outcomes

### Technical Outcomes:
- Comprehensive model comparison
- Best-performing hybrid approach
- Feature importance analysis
- Model interpretability insights

### Biological Outcomes:
- Improved cell type classification
- Quality control metrics
- Novel cell type markers
- Biological pathway insights

### Dissertation Contributions:
- Novel hybrid model approach
- Comprehensive scRNA-seq analysis pipeline
- Benchmarking of different methods
- Practical implementation guidelines

## 📊 Results Structure

### Generated Files:
- **Figures:** Quality control plots, cell type distributions, model performance
- **Tables:** Summary statistics, model metrics, feature importance
- **Models:** Trained model files for each approach
- **Reports:** Comprehensive analysis reports

### Key Visualizations:
- Quality metrics plots
- Cell type comparison charts
- Clustering analysis
- Model performance comparisons
- Feature importance rankings

## 🔬 Methodology

### Phase 1: Data Processing (Week 1-2)
- Data loading and validation
- Quality control implementation
- Feature engineering
- Data preprocessing

### Phase 2: Model Development (Week 3-6)
- Traditional ML models
- Deep learning models
- Hybrid model development
- Model training and validation

### Phase 3: Analysis & Evaluation (Week 7-8)
- Model comparison
- Performance evaluation
- Statistical analysis
- Results interpretation

### Phase 4: Documentation (Week 9-10)
- Report generation
- Documentation
- Code cleanup
- Final presentation

## 📚 Documentation

- **`docs/methodology.md`** - Detailed methodology
- **`docs/model_architecture.md`** - Model architecture details
- **`docs/results_summary.md`** - Results summary
- **`docs/technical_notes.md`** - Technical implementation notes

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📝 Contributing

This is a dissertation project. For academic collaboration:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for academic research purposes.

## 👨‍🎓 Author

**Student Name**  Mohansree Vijayakumar
**Institution**  university of surrey
**Supervisor:** Dr . Samaneh Kouchaki   
**Year:** 2025

## 📞 Contact

For questions about this dissertation project, please contact:
- **Email:** mv00582@surrey.ac.uk
- **Institution:** university of surrey

---

**Last Updated:** Sep 2025  
**Version:** 1.0.0  
**Status:** completed

## Running the Complete Pipeline

1. **Edit the config file**
   - Open `config/config.yaml` and set `data_path` to the full path of your input data file (especially important for cluster use).
   - Adjust output directories as needed.

2. **Run the pipeline**
   ```bash
   python scripts/run_complete_analysis.py --config config/config.yaml
   ```
   This will execute all steps in order, passing the config to each script.

3. **Cluster Usage**
   - Submit the above command as a job on your cluster (e.g., using SLURM, PBS, etc.).
   - Make sure all paths in `config.yaml` are accessible from the cluster compute nodes.

4. **Customizing Steps**
   - You can run individual scripts with the same `--config` argument, e.g.:
     ```bash
     python scripts/05_train_deep_learning_models.py --config config/config.yaml
     ```

5. **Outputs**
   - All results, logs, and reports will be saved in the directories specified in `config.yaml`.

# Machine Learning-Based Prediction of Cellular Mechanisms in Alzheimer’s Disease

## Project Overview
This project analyzes single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics data to predict cellular mechanisms involved in Alzheimer’s Disease. It includes data inspection, preprocessing, feature extraction, visualization, and machine learning modeling.

## Folder Structure
- `data/`
  - `raw/`: Original data files (CSV format)
  - `processed/`: Cleaned and processed data
- `results/`: All output files, plots, logs, and model artifacts
- `scripts/`: Python scripts for each analysis step
- `config/`: Configuration files (YAML, Python)
- `notebooks/`: Jupyter notebooks for exploration

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```powershell
   & .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. **Configure paths**
   - Edit `config/config.yaml` to set your data and results directories.

## Data Source
- Download scRNA-seq and spatial datasets from:
  [BMBLx Data Portal](https://bmblx.bmi.osumc.edu/ssread/downloads/#scrna-seq-%26-snrnaseq-and-spatial-datasets)
- After downloading, convert the data to CSV format using your preferred tool (e.g., Excel, Python pandas):
   ```python
   import pandas as pd
   df = pd.read_excel('your_downloaded_file.xlsx')
   df.to_csv('data/raw/your_data.csv', index=False)
   ```

## Typical Workflow
1. **Inspect and fix raw CSV:**
   ```powershell
   python scripts/01_run_csv_inspection.py --config config/config.yaml
   ```
2. **Feature extraction:**
   ```powershell
   python scripts/02_feature_extraction.py --config config/config.yaml
   ```
3. **Preprocessing:**
   ```powershell
   python scripts/02_preprocess_data.py --config config/config.yaml
   ```
4. **Analysis and visualization:**
   ```powershell
   python scripts/03_analysis_and_visualization.py --type general --config config/config.yaml
   ```
5. **Model training and evaluation:**
   - Run subsequent scripts in `scripts/` as needed for modeling and analysis.

## Output
- All results, plots, and logs are saved in the `results/` folder, organized by analysis step.

## Troubleshooting
- If plots are empty, check your data for missing or constant values.
- Ensure all dependencies are installed in your virtual environment.
- Review logs in `results/logs/` for error details.

## Contact

For questions or issues, please contact the repository owner.
