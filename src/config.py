"""
Configuration settings for the RNA-seq analysis pipeline.

This module centralizes all hyperparameters and experimental settings
to ensure reproducibility and easy modification.
"""

# =============================================================================
# RANDOM SEEDS
# =============================================================================
# Seeds for reproducible experiments across multiple runs
SEEDS = [2, 4, 8, 16, 32, 64, 127, 255, 511, 1023, 2047, 4095]

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================
# Minimum R² score to consider a model's results
R2_PERF_THRESH = 0.9

# =============================================================================
# DATA TRANSFORMATION
# =============================================================================
# List of transformations to apply in order
# Options: 'tpm', 'log', 'std', 'power', 'boxcox'
X_LIST = ['tpm', 'std']

# =============================================================================
# GENE FILTERING
# =============================================================================
# Coefficient of variation threshold (0 = no filtering)
CVS = 0

# Number of top correlated genes to keep (0 = no filtering)
K_CORR = 1000

# DESeq2 significance threshold (0 = no DGEA filtering)
ALPHA = 0

# Low count filter: (count_threshold, proportion_threshold)
# Removes genes with counts <= 100 in >= 80% of samples
LOWCOUNT = (100, 0.8)

# =============================================================================
# MODEL TRAINING
# =============================================================================
# Number of cross-validation folds for hyperparameter tuning
N_FOLDS = 5

# Number of top genes to extract from each model
N_GENES = 40

# Fraction of data to hold out for testing
TEST_SIZE = 0.30

# Performance metric for optimization
# Options: 'r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'
REG_PERF = 'r2'

# =============================================================================
# RNA-SEQ DATA SOURCE
# =============================================================================
# Options:
#   '255_rna_seq_STAR_Unnormalized_Counts' - STAR aligner raw counts
#   '255_rna_seq_RSEM_Unnormalized_Counts' - RSEM raw counts
#   '255_rna_seq_Normalized_Counts' - Pre-normalized counts
RNA_SEQ_KEY = '255_rna_seq_STAR_Unnormalized_Counts'

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================
EXPERIMENT_HNE = {
    'name': 'hne',
    'pheno_var': 'sumEC',
    'description': '4-Hydroxynonenal - marker of lipid peroxidation/oxidative stress'
}

EXPERIMENT_TUNEL = {
    'name': 'tunel',
    'pheno_var': 'Density_EC',
    'description': 'TUNEL - marker of cellular apoptosis'
}

EXPERIMENTS = [EXPERIMENT_HNE, EXPERIMENT_TUNEL]

# =============================================================================
# FILE PATHS
# =============================================================================
# GTF file for TPM calculation
GTF_FILENAME = 'Mus_musculus.GRCm39.115.gtf.gz'
GTF_URL = 'https://ftp.ensembl.org/pub/release-115/gtf/mus_musculus/Mus_musculus.GRCm39.115.gtf.gz'

# Output directory for results
RESULTS_DIR = 'results'

# Data directory for cached downloads
DATA_DIR = 'data'

# =============================================================================
# NASA GENELAB API
# =============================================================================
OSDR_BASE_URL = 'https://osdr.nasa.gov/geode-py/ws/studies'

# Dataset IDs
OSD_255 = '255'  # RNA-seq transcriptomics
OSD_557 = '557'  # HNE phenotype data
OSD_568 = '568'  # TUNEL phenotype data

# =============================================================================
# MODEL LIST
# =============================================================================
MODELS = ['svm', 'lasso', 'lr', 'elastic', 'ridge']

# =============================================================================
# VISUALIZATION
# =============================================================================
# PCA plot colors
CLASS_COLORS = {
    'class_1': 'navy',
    'class_2': 'red'
}

# Plot DPI for saved figures
PLOT_DPI = 300


def get_experiment_by_name(name):
    """
    Get experiment configuration by name.
    
    Args:
        name (str): Experiment name ('hne' or 'tunel')
        
    Returns:
        dict: Experiment configuration
    """
    for exp in EXPERIMENTS:
        if exp['name'] == name:
            return exp
    raise ValueError(f"Unknown experiment: {name}")


def print_config():
    """Print current configuration settings."""
    print("=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"R² threshold: {R2_PERF_THRESH}")
    print(f"Transformations: {X_LIST}")
    print(f"CV filter: {CVS}")
    print(f"Correlation filter k: {K_CORR}")
    print(f"DESeq2 alpha: {ALPHA}")
    print(f"Low count filter: {LOWCOUNT}")
    print(f"CV folds: {N_FOLDS}")
    print(f"Top genes: {N_GENES}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Metric: {REG_PERF}")
    print(f"RNA-seq key: {RNA_SEQ_KEY}")
    print("=" * 60)
