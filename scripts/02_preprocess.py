#!/usr/bin/env python3
"""
Script 02: Preprocess RNA-seq data.

This script:
1. Loads RNA-seq and phenotype data
2. Aligns samples across datasets
3. Applies gene filtering (NaN, low-count, non-coding, DESeq2)
4. Creates target vectors (y_vals, y_classes, y_conditions)
5. Saves preprocessed data for model training

Usage:
    python scripts/02_preprocess.py --assay hne --seed 42 [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loading import (
    read_data_and_metadata_hne,
    read_data_and_metadata_tunel,
    run_create_Y,
)
from src.preprocessing import run_filter_data
from src.config import (
    CVS, ALPHA, LOWCOUNT, K_CORR, RESULTS_DIR,
    EXPERIMENT_HNE, EXPERIMENT_TUNEL, RNA_SEQ_KEY,
)


def preprocess_experiment(assay, seed, output_dir):
    """
    Run preprocessing pipeline for a single experiment.
    
    Args:
        assay (str): 'hne' or 'tunel'
        seed (int): Random seed for reproducibility
        output_dir (str): Directory to save preprocessed data
    """
    # Initialize data containers
    data = {}
    metadata = {}
    
    # Get experiment config
    if assay == 'hne':
        experiment = EXPERIMENT_HNE
        X_orig, metadata_orig, pheno_data = read_data_and_metadata_hne(
            data, metadata, RNA_SEQ_KEY
        )
    elif assay == 'tunel':
        experiment = EXPERIMENT_TUNEL
        X_orig, metadata_orig, pheno_data = read_data_and_metadata_tunel(
            data, metadata, RNA_SEQ_KEY
        )
    else:
        raise ValueError(f"Unknown assay: {assay}. Use 'hne' or 'tunel'.")
    
    pheno_var = experiment['pheno_var']
    
    print(f"\n{'='*60}")
    print(f"Preprocessing: {assay.upper()} (seed={seed})")
    print(f"Phenotype variable: {pheno_var}")
    print(f"{'='*60}")
    
    # Create target vectors
    print("\n[1/2] Creating target vectors...")
    y_conditions, y_classes, y_vals = run_create_Y(
        metadata_orig, pheno_data, pheno_var
    )
    
    # Run filtering pipeline
    print("\n[2/2] Running filtering pipeline...")
    X_filtered, X_array = run_filter_data(
        X_orig=X_orig,
        metadata_orig=metadata_orig,
        y_vals=y_vals,
        y_classes=y_classes,
        cvs=CVS,
        alpha=ALPHA,
        lowcount=LOWCOUNT,
        correlation=K_CORR,
        seed=seed,
    )
    
    # Check for empty result
    if X_filtered.shape[0] == 0 or X_filtered.shape[1] == 0:
        raise ValueError("No data remaining after filtering!")
    
    print(f"\n  Final X shape: {X_filtered.shape}")
    print(f"  Final y_vals length: {len(y_vals)}")
    
    # Create output directory
    exp_output_dir = os.path.join(output_dir, f'{assay}_seed{seed}')
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Save preprocessed data
    output_data = {
        'X_orig': X_filtered,
        'X_array': X_array,
        'y_vals': y_vals,
        'y_classes': y_classes,
        'y_conditions': y_conditions,
        'metadata_orig': metadata_orig,
        'pheno_data': pheno_data,
        'pheno_var': pheno_var,
        'experiment': experiment,
        'seed': seed,
    }
    
    output_path = os.path.join(exp_output_dir, 'preprocessed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n  Preprocessed data saved to: {output_path}")
    
    # Also save as CSV for inspection
    X_filtered.to_csv(os.path.join(exp_output_dir, 'X_filtered.csv'), index=False)
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess RNA-seq data')
    parser.add_argument(
        '--assay',
        type=str,
        required=True,
        choices=['hne', 'tunel'],
        help='Assay type: hne or tunel'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=RESULTS_DIR,
        help='Output directory for preprocessed data'
    )
    args = parser.parse_args()
    
    # Run preprocessing
    preprocess_experiment(
        assay=args.assay,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
