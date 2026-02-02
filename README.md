# Spaceflight-Induced Retinal Gene Expression Analysis

A machine learning pipeline for identifying genes associated with spaceflight-induced oxidative stress (HNE) and apoptosis (TUNEL) in mouse retinal tissue using NASA GeneLab transcriptomics data.

## Overview

This project analyzes RNA-seq data from NASA's Rodent Research 9 (RR-9) mission to identify gene expression signatures predictive of retinal damage markers. We use an ensemble of regression models to discover genes whose expression correlates with:

- **HNE (4-Hydroxynonenal)**: A marker of lipid peroxidation and oxidative stress
- **TUNEL**: A marker of cellular apoptosis (programmed cell death)

### Scientific Background

Spaceflight exposes astronauts to multiple stressors including microgravity, radiation, and altered circadian rhythms. The retina is particularly vulnerable to these stressors, making it critical to understand the molecular pathways involved in spaceflight-induced eye damage.

## Methods

### Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| OSD-255 | Mouse retina RNA-seq (RR-9) | [NASA GeneLab](https://osdr.nasa.gov/bio/repo/data/studies/OSD-255) |
| OSD-557 | HNE immunostaining data | [NASA GeneLab](https://osdr.nasa.gov/bio/repo/data/studies/OSD-557) |
| OSD-568 | TUNEL assay data | [NASA GeneLab](https://osdr.nasa.gov/bio/repo/data/studies/OSD-568) |

### Analysis Pipeline

```
1. Data Download → 2. Preprocessing → 3. Model Training → 4. Enrichment → 5. Aggregation
```

1. **Data Download**: Retrieve RNA-seq counts, metadata, and phenotype data from NASA GeneLab
2. **Preprocessing**: Filter genes (low counts, non-coding), normalize (TPM), transform (log, standardization)
3. **Model Training**: Train 5 regression models with hyperparameter tuning
4. **Enrichment**: Perform GO and KEGG pathway analysis on significant genes
5. **Aggregation**: Combine results across 12 random seeds for robustness

### Machine Learning Models

| Model | Description |
|-------|-------------|
| ElasticNet | L1+L2 regularized linear regression |
| SVR | Support Vector Regression with linear kernel |
| Ridge | L2 regularized linear regression |
| Lasso | L1 regularized linear regression |
| Linear | Ordinary least squares regression |

### Feature Selection

For each model, we extract important genes using:
- **PFI**: Permutation Feature Importance
- **RFE**: Recursive Feature Elimination
- **Coefficients**: Top positive/negative model weights

Consensus genes are identified by intersecting results across methods and models.

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/HNE-TUNEL_RR9_Transcriptomics.git
cd HNE-TUNEL_RR9_Transcriptomics

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run a single experiment (HNE assay, seed 42)
python scripts/run_all.py --assay hne --seed 42

# Run the full experimental suite (both assays, all 12 seeds)
python scripts/run_all.py --run-all
```

### Individual Pipeline Steps

```bash
# Step 1: Download and cache data
python scripts/01_download_data.py

# Step 2: Preprocess data
python scripts/02_preprocess.py --assay hne --seed 42

# Step 3: Train models
python scripts/03_train_models.py --assay hne --seed 42

# Step 4: Run pathway enrichment
python scripts/04_run_enrichment.py --assay hne --seed 42

# Step 5: Aggregate results across seeds
python scripts/05_aggregate_results.py
```

### Configuration

All experiment parameters are centralized in `src/config.py`:

```python
SEEDS = [2, 4, 8, 16, 32, 64, 127, 255, 511, 1023, 2047, 4095]  # Random seeds
N_GENES = 40          # Top genes per model
TEST_SIZE = 0.30      # Train/test split
N_FOLDS = 5           # Cross-validation folds
R2_PERF_THRESH = 0.9  # Minimum R² threshold
```

## Project Structure

```
HNE-TUNEL_RR9_Transcriptomics/
├── src/                      # Source library
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Experiment configuration
│   ├── data_loading.py       # NASA GeneLab API utilities
│   ├── preprocessing.py      # Gene filtering & transformations
│   ├── feature_selection.py  # Feature importance methods
│   ├── models.py             # Regression model implementations
│   ├── deseq2_utils.py       # Differential expression analysis
│   ├── visualization.py      # Plotting functions
│   └── utils.py              # General utilities
├── scripts/                  # Executable pipeline
│   ├── 01_download_data.py   # Download data
│   ├── 02_preprocess.py      # Preprocess data
│   ├── 03_train_models.py    # Train models
│   ├── 04_run_enrichment.py  # Pathway enrichment
│   ├── 05_aggregate_results.py # Aggregate results
│   └── run_all.py            # Main orchestration
├── data/                     # Downloaded data (gitignored)
├── results/                  # Analysis outputs (gitignored)
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment (alternative)
└── README.md                 # This file
```

## Output Files

After running the pipeline, results are saved to `results/`:

```
results/
├── hne_seed42/
│   ├── preprocessed_data.pkl    # Preprocessed data
│   ├── X_filtered.csv           # Filtered expression matrix
│   ├── model_results.pkl        # Full model results
│   ├── performance_metrics.csv  # Model performance
│   ├── consensus_genes.txt      # Significant genes
│   └── enrichment/              # Pathway analysis
│       ├── hne_majority_GO_Biological_Process.csv
│       ├── hne_majority_KEGG.csv
│       └── ...
├── tunel_seed42/
│   └── ...
├── aggregated_results.pkl       # Combined results
└── aggregated_summary.txt       # Human-readable summary
```

## Dependencies

- **numpy, pandas, scipy**: Core data manipulation
- **scikit-learn**: Machine learning models
- **matplotlib, seaborn**: Visualization
- **mygene, pybiomart**: Gene annotation
- **pydeseq2**: Differential expression analysis
- **rnanorm**: TPM normalization
- **gseapy**: Pathway enrichment

## Citation

If you use this pipeline, please cite the NASA GeneLab data repository:

> NASA GeneLab: Open Science for Life in Space. https://genelab.nasa.gov/

## License

MIT License - See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.
