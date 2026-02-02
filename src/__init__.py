"""
NASA RNA-seq Analysis Pipeline

This package provides tools for analyzing spaceflight-induced gene expression
changes in mouse retinal tissue using machine learning approaches.

Modules:
    - data_loading: Download and load data from NASA GeneLab
    - preprocessing: Gene filtering and data transformations
    - feature_selection: Feature importance and gene selection
    - models: Regression model implementations
    - deseq2_utils: Differential expression analysis
    - visualization: Plotting and visualization
    - utils: General utilities
    - config: Configuration settings
"""

from .data_loading import (
    read_meta_data,
    read_rnaseq_data,
    read_phenotype_data,
    read_data_and_metadata_hne,
    read_data_and_metadata_tunel,
    transpose_df,
    run_create_Y,
)

from .preprocessing import (
    drop_nans,
    filter_cvs,
    filterGenesByPercentLowCount,
    filter_genes,
    filter_data,
    run_filter_data,
    full_transform,
    myboxcox,
)

from .feature_selection import (
    filterNotCorrelated,
    permutation_feature_importance,
    get_symbol_from_id,
    convert_ids_to_names,
)

from .models import (
    run_elasticnet,
    run_svm,
    run_ridge_regression,
    run_lasso_regression,
    run_linear_regression,
)

from .deseq2_utils import (
    run_deseq2,
    get_results,
    get_sig_genes,
    filter_by_dgea,
)

from .visualization import (
    results_dist_plot,
    plot_2d_pca,
    plot_3d_pca,
    plotbox_and_stats,
)

from .utils import (
    store_results,
    load_results,
    run_create_groups,
    intersect_samples,
    aggregate_gene_results,
)

from .config import (
    SEEDS,
    R2_PERF_THRESH,
    X_LIST,
    CVS,
    K_CORR,
    N_FOLDS,
    ALPHA,
    N_GENES,
    TEST_SIZE,
    LOWCOUNT,
    REG_PERF,
    EXPERIMENTS,
    GTF_FILENAME,
    RESULTS_DIR,
    print_config,
)

__version__ = '2.0.0'
__author__ = 'NASA GeneLab Analysis Team'
