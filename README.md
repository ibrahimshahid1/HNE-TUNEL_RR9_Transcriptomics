# NASA RNA-seq Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

**Machine learning analysis of spaceflight-induced gene expression changes in mouse retinal tissue.**

This repository contains a reproducible pipeline for analyzing NASA GeneLab RNA-seq datasets (OSD-255) combined with phenotype measurements from spaceflight experiments. The analysis utilizes nested cross-validation to identify robust gene signatures associated with cellular stress markers (TUNEL and HNE) in microgravity environments.

---

##  Project Structure

```text
nasa-rnaseq-analysis/
├── src/
│   ├── data_loading.py       # Data download and alignment utilities
│   ├── preprocessing.py      # Gene filtering, transformers, and normalization
│   ├── models.py             # Model configs, training loops, and evaluation
│   └── enrichment.py         # GSEA/PEA functions via gseapy
├── scripts/
│   └── run_experiment.py     # Main CLI entry point for experiments
├── results/                  # Output directory (gitignored)
├── environment.yml           # Conda environment configuration
├── requirements.txt          # Pip dependencies
└── README.md
```

##  Installation

### Option 1: Conda (Recommended)

Isolate dependencies using a Conda environment:

```bash
conda env create -f environment.yml
conda activate nasa-rnaseq
```

### Option 2: Pip

Install directly using pip:

```bash
pip install -r requirements.txt
```

##  Quick Start

### Run a Single Experiment

Run the pipeline for a specific assay using "Mode C" (proper normalization) and a specific random seed:

```bash
python scripts/run_experiment.py --assay TUNEL --mode C --seed 42
```

### Run Full Experimental Suite

To reproduce the full analysis (240 total model training runs), use the run-all flag:

```bash
python scripts/run_experiment.py --run-all
```

This executes:
- 2 Assays: TUNEL, HNE
- 3 Modes: A, B, C
- 8 Seeds: Random state initialization
- 5 Models: Ridge, Lasso, ElasticNet, SVR, RandomForest

##  Experimental Design

### Assays

The pipeline targets two specific physiological markers observed in spaceflown mice:

- **TUNEL**: Marker for cellular apoptosis (Target: Density_EC).
- **HNE**: Marker for lipid peroxidation and oxidative stress (Target: sumEC).

### Transformation Modes

To ensure rigorous validation, we compare three data processing strategies:

- **Mode A (Baseline/Leakage)**: Pre-split normalization. (Comparison baseline).
- **Mode B (Mismatch)**: Normalization based on training data statistics only, applied blindly to test data.
- **Mode C (Correct)**: Proper fit-transform on training splits and transform-only on testing splits within the cross-validation loop.

### Pipeline Steps

1. **Data Loading**: Ingest RNA-seq counts (OSD-255) and phenotype data.
2. **Filtering**: Remove non-coding genes and low-count features.
3. **Preprocessing**: CPM normalization $\rightarrow$ $\log(x+1)$ transformation $\rightarrow$ Standardization.
4. **Feature Selection**: Select top 1000 variable genes per fold.
5. **Modeling**: Train models using Nested Cross-Validation (5x5).
6. **Enrichment**: Extract feature importances and run pathway analysis (GO, KEGG, ARCHS4).

##  Outputs

Results are saved automatically to the results/ directory, organized by experiment configuration:

```text
results/
└── TUNEL_modeC_seed42/
    ├── Ridge_metrics.csv
    ├── Lasso_metrics.csv
    ├── ElasticNet_metrics.csv
    ├── SVR_metrics.csv
    ├── RandomForest_metrics.csv
    ├── consensus_genes.txt                # Genes appearing across multiple folds
    ├── enrichment_summary.csv
    └── enrichment/
        ├── TUNEL_modeC_GO_Biological_Process.csv
        ├── TUNEL_modeC_KEGG.csv
        └── ...
```

### Metrics Tracked

- $R^2$: Variance explained
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- Spearman/Pearson: Correlation coefficients between predicted and actual stress levels.

##  Data Sources

Data is automatically downloaded via the NASA GeneLab Data System APIs or retrieved from local caches.

| Dataset ID | Description |
|-----------|-------------|
| OSD-255 | RNA-seq of Mouse Retinal Tissue (Transcriptomics) |
| OSD-568 / LSDS-5 | Apoptosis assays (Phenotype TUNEL) |
| OSD-557 / LSDS-1 | Oxidative stress assays (Phenotype HNE) |

##  Citation


```

## 🛠 Dependencies

- python >= 3.8
- scikit-learn
- pandas & numpy
- gseapy (Enrichment)
- pydeseq2 (Normalization)
- pybiomart & mygene (Annotation)

**Acknowledgments**: This research utilizes data from NASA's GeneLab platform and Open Science Data Repository.
