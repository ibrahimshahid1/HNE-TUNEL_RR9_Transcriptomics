#!/usr/bin/env python3
"""
Script 03: Train machine learning models.

This script:
1. Loads preprocessed data
2. Trains all 5 regression models (ElasticNet, SVR, Ridge, Lasso, Linear)
3. Extracts feature importances (PFI, RFE, coefficients)
4. Saves model results and gene lists

Usage:
    python scripts/03_train_models.py --assay hne --seed 42 [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    run_elasticnet,
    run_svm,
    run_ridge_regression,
    run_lasso_regression,
    run_linear_regression,
)
from src.config import (
    N_GENES, REG_PERF, X_LIST, N_FOLDS, TEST_SIZE,
    RESULTS_DIR, GTF_FILENAME, DATA_DIR,
)


def load_preprocessed_data(assay, seed, data_dir):
    """Load preprocessed data from pickle file."""
    data_path = os.path.join(data_dir, f'{assay}_seed{seed}', 'preprocessed_data.pkl')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_path}\n"
            f"Run 02_preprocess.py first."
        )
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def train_all_models(X_orig, y_vals, seed, gtf_path=None):
    """
    Train all regression models on the data.
    
    Args:
        X_orig (pd.DataFrame): Feature matrix (samples x genes)
        y_vals (np.ndarray): Target values
        seed (int): Random seed
        gtf_path (str): Path to GTF file for TPM
        
    Returns:
        dict: Results for all models
    """
    results = {
        'genes': {},
        'estimators': {},
        'perfs': {},
        'pos_coefs': {},
        'neg_coefs': {},
    }
    
    # Remove sample column for training
    X_train = X_orig.drop(columns=['sample']) if 'sample' in X_orig.columns else X_orig
    
    models = [
        ('svm', run_svm),
        ('lasso', run_lasso_regression),
        ('lr', run_linear_regression),
        ('elastic', run_elasticnet),
        ('ridge', run_ridge_regression),
    ]
    
    for model_name, model_func in models:
        print(f"\n  Training {model_name.upper()}...")
        
        try:
            genes, estimator, perfs, pos_coefs, neg_coefs = model_func(
                y=y_vals,
                X_array=None,
                X_orig=X_train,
                n_genes=N_GENES,
                score=REG_PERF,
                xform_list=X_LIST,
                cv=N_FOLDS,
                seed=seed,
                test_size=TEST_SIZE,
                gtf_path=gtf_path,
            )
            
            results['genes'][model_name] = genes
            results['estimators'][model_name] = estimator
            results['perfs'][model_name] = perfs
            results['pos_coefs'][model_name] = pos_coefs
            results['neg_coefs'][model_name] = neg_coefs
            
        except Exception as e:
            print(f"    Error training {model_name}: {e}")
            results['genes'][model_name] = {}
            results['perfs'][model_name] = {'error': str(e)}
    
    return results


def compute_gene_consensus(gene_results, r2_threshold=0.9):
    """
    Compute consensus genes across models.
    
    For each model with test_r2 > threshold, compute:
    - Intersection of PFI and RFE
    - Intersection with union of positive and negative coefficient genes
    
    Args:
        gene_results (dict): Gene results from all models
        r2_threshold (float): Minimum R² to include model
        
    Returns:
        dict: Consensus gene sets
    """
    model_genes = {}
    
    for model in gene_results['genes']:
        perfs = gene_results['perfs'].get(model, {})
        test_r2 = perfs.get('test_r2', 0)
        
        if test_r2 >= r2_threshold:
            genes = gene_results['genes'][model]
            
            pfi = set(genes.get('pfi', []))
            rfe = set(genes.get('rfe', []))
            pos = set(genes.get('pos', []))
            neg = set(genes.get('neg', []))
            
            # Intersection of feature importance methods, then intersect with coef genes
            model_genes[model] = pfi.intersection(rfe).intersection(pos.union(neg))
        else:
            model_genes[model] = set()
    
    # Union across all models
    union_genes = set()
    for genes in model_genes.values():
        union_genes.update(genes)
    
    # Majority voting (genes in > 2 models)
    from collections import Counter
    gene_counts = Counter()
    for genes in model_genes.values():
        for g in genes:
            gene_counts[g] += 1
    
    majority_genes = {g for g, c in gene_counts.items() if c > 2}
    
    # Intersection (genes in all models with good R²)
    valid_model_genes = [g for g in model_genes.values() if len(g) > 0]
    if valid_model_genes:
        intersection_genes = set.intersection(*valid_model_genes)
    else:
        intersection_genes = set()
    
    return {
        'by_model': model_genes,
        'union': union_genes,
        'majority': majority_genes,
        'intersection': intersection_genes,
    }


def save_results(results, consensus, output_dir):
    """Save training results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results as pickle
    with open(os.path.join(output_dir, 'model_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save consensus genes
    with open(os.path.join(output_dir, 'consensus_genes.txt'), 'w') as f:
        f.write("UNION GENES:\n")
        f.write(', '.join(sorted(consensus['union'])) + '\n\n')
        
        f.write("MAJORITY GENES (>2 models):\n")
        f.write(', '.join(sorted(consensus['majority'])) + '\n\n')
        
        f.write("INTERSECTION GENES:\n")
        f.write(', '.join(sorted(consensus['intersection'])) + '\n\n')
        
        f.write("\nPER-MODEL GENES:\n")
        for model, genes in consensus['by_model'].items():
            f.write(f"  {model}: {', '.join(sorted(genes))}\n")
    
    # Save performance metrics as CSV
    import pandas as pd
    perf_rows = []
    for model, perfs in results['perfs'].items():
        row = {'model': model}
        row.update(perfs)
        perf_rows.append(row)
    
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
    
    print(f"\n  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train regression models')
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
        help='Random seed'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=RESULTS_DIR,
        help='Directory containing preprocessed data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data-dir/assay_seed/)'
    )
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, f'{args.assay}_seed{args.seed}')
    
    print("=" * 60)
    print(f"Model Training: {args.assay.upper()} (seed={args.seed})")
    print("=" * 60)
    
    # Load preprocessed data
    print("\n[1/3] Loading preprocessed data...")
    data = load_preprocessed_data(args.assay, args.seed, args.data_dir)
    
    X_orig = data['X_orig']
    y_vals = data['y_vals']
    
    print(f"  X shape: {X_orig.shape}")
    print(f"  y length: {len(y_vals)}")
    
    # Get GTF path
    gtf_path = os.path.join(DATA_DIR, GTF_FILENAME)
    if not os.path.exists(gtf_path):
        gtf_path = None
        print("  Warning: GTF file not found, TPM normalization disabled")
    
    # Train models
    print("\n[2/3] Training models...")
    results = train_all_models(X_orig, y_vals, args.seed, gtf_path)
    
    # Compute consensus
    print("\n[3/3] Computing gene consensus...")
    consensus = compute_gene_consensus(results)
    
    print(f"  Union genes: {len(consensus['union'])}")
    print(f"  Majority genes: {len(consensus['majority'])}")
    print(f"  Intersection genes: {len(consensus['intersection'])}")
    
    # Save results
    save_results(results, consensus, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
