#!/usr/bin/env python3
"""
Script 05: Aggregate results across multiple seeds.

This script:
1. Loads results from all seed experiments
2. Aggregates gene sets across seeds
3. Computes final consensus genes
4. Compares HNE vs TUNEL results

Usage:
    python scripts/05_aggregate_results.py [--data-dir DATA_DIR]
"""

import os
import sys
import argparse
import pickle
from collections import Counter

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import SEEDS, RESULTS_DIR, R2_PERF_THRESH


def find_completed_experiments(data_dir, assay):
    """Find all completed experiments for an assay."""
    completed = []
    
    for seed in SEEDS:
        exp_dir = os.path.join(data_dir, f'{assay}_seed{seed}')
        results_file = os.path.join(exp_dir, 'model_results.pkl')
        
        if os.path.exists(results_file):
            completed.append(seed)
    
    return completed


def load_all_results(data_dir, assay, seeds):
    """Load results from all seed experiments."""
    all_results = {}
    
    for seed in seeds:
        results_file = os.path.join(data_dir, f'{assay}_seed{seed}', 'model_results.pkl')
        
        try:
            with open(results_file, 'rb') as f:
                all_results[seed] = pickle.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load seed {seed}: {e}")
    
    return all_results


def aggregate_genes_across_seeds(all_results, r2_threshold=0.9):
    """
    Aggregate gene results across all seeds.
    
    For each model with test_r2 > threshold, compute consensus genes.
    Then aggregate across all seeds.
    
    Returns:
        dict: Aggregated gene sets
    """
    all_genes = Counter()
    seed_counts = {}
    
    for seed, results in all_results.items():
        for model, perfs in results['perfs'].items():
            test_r2 = perfs.get('test_r2', 0)
            
            if test_r2 >= r2_threshold:
                genes = results['genes'].get(model, {})
                
                pfi = set(genes.get('pfi', []))
                rfe = set(genes.get('rfe', []))
                pos = set(genes.get('pos', []))
                neg = set(genes.get('neg', []))
                
                # Consensus for this model/seed
                consensus = pfi.intersection(rfe).intersection(pos.union(neg))
                
                for gene in consensus:
                    all_genes[gene] += 1
                    
                    if gene not in seed_counts:
                        seed_counts[gene] = set()
                    seed_counts[gene].add(seed)
    
    # Categorize genes by frequency
    n_seeds = len(all_results)
    
    return {
        'all_genes': all_genes,
        'seed_counts': seed_counts,
        'n_seeds': n_seeds,
        # Genes appearing in >50% of seeds
        'frequent': {g for g, c in all_genes.items() if len(seed_counts.get(g, set())) > n_seeds // 2},
        # Genes appearing in >75% of seeds
        'robust': {g for g, c in all_genes.items() if len(seed_counts.get(g, set())) > n_seeds * 0.75},
    }


def compare_assays(hne_genes, tunel_genes):
    """Compare gene sets between HNE and TUNEL."""
    hne_set = set(hne_genes.keys()) if isinstance(hne_genes, dict) else set(hne_genes)
    tunel_set = set(tunel_genes.keys()) if isinstance(tunel_genes, dict) else set(tunel_genes)
    
    return {
        'hne_only': hne_set - tunel_set,
        'tunel_only': tunel_set - hne_set,
        'both': hne_set & tunel_set,
        'union': hne_set | tunel_set,
    }


def save_aggregated_results(hne_agg, tunel_agg, comparison, output_dir):
    """Save aggregated results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated results
    with open(os.path.join(output_dir, 'aggregated_results.pkl'), 'wb') as f:
        pickle.dump({
            'hne': hne_agg,
            'tunel': tunel_agg,
            'comparison': comparison,
        }, f)
    
    # Save human-readable summary
    with open(os.path.join(output_dir, 'aggregated_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AGGREGATED GENE ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        if hne_agg:
            f.write("HNE RESULTS:\n")
            f.write(f"  Total unique genes: {len(hne_agg['all_genes'])}\n")
            f.write(f"  Frequent genes (>50% seeds): {len(hne_agg['frequent'])}\n")
            f.write(f"    {', '.join(sorted(hne_agg['frequent']))}\n")
            f.write(f"  Robust genes (>75% seeds): {len(hne_agg['robust'])}\n")
            f.write(f"    {', '.join(sorted(hne_agg['robust']))}\n\n")
        
        if tunel_agg:
            f.write("TUNEL RESULTS:\n")
            f.write(f"  Total unique genes: {len(tunel_agg['all_genes'])}\n")
            f.write(f"  Frequent genes (>50% seeds): {len(tunel_agg['frequent'])}\n")
            f.write(f"    {', '.join(sorted(tunel_agg['frequent']))}\n")
            f.write(f"  Robust genes (>75% seeds): {len(tunel_agg['robust'])}\n")
            f.write(f"    {', '.join(sorted(tunel_agg['robust']))}\n\n")
        
        if comparison:
            f.write("COMPARISON (HNE vs TUNEL):\n")
            f.write(f"  Genes in both: {len(comparison['both'])}\n")
            f.write(f"    {', '.join(sorted(comparison['both']))}\n")
            f.write(f"  HNE-only genes: {len(comparison['hne_only'])}\n")
            f.write(f"    {', '.join(sorted(comparison['hne_only']))}\n")
            f.write(f"  TUNEL-only genes: {len(comparison['tunel_only'])}\n")
            f.write(f"    {', '.join(sorted(comparison['tunel_only']))}\n")
    
    print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate results across seeds')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=RESULTS_DIR,
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--r2-threshold',
        type=float,
        default=R2_PERF_THRESH,
        help='Minimum R² to include model results'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Aggregating Results Across Seeds")
    print("=" * 60)
    
    # Find completed experiments
    print("\n[1/4] Finding completed experiments...")
    hne_seeds = find_completed_experiments(args.data_dir, 'hne')
    tunel_seeds = find_completed_experiments(args.data_dir, 'tunel')
    
    print(f"  HNE experiments: {len(hne_seeds)} seeds")
    print(f"  TUNEL experiments: {len(tunel_seeds)} seeds")
    
    # Load all results
    print("\n[2/4] Loading results...")
    hne_results = load_all_results(args.data_dir, 'hne', hne_seeds) if hne_seeds else {}
    tunel_results = load_all_results(args.data_dir, 'tunel', tunel_seeds) if tunel_seeds else {}
    
    # Aggregate genes
    print("\n[3/4] Aggregating genes across seeds...")
    hne_agg = aggregate_genes_across_seeds(hne_results, args.r2_threshold) if hne_results else {}
    tunel_agg = aggregate_genes_across_seeds(tunel_results, args.r2_threshold) if tunel_results else {}
    
    if hne_agg:
        print(f"  HNE: {len(hne_agg['all_genes'])} unique genes, {len(hne_agg['frequent'])} frequent")
    if tunel_agg:
        print(f"  TUNEL: {len(tunel_agg['all_genes'])} unique genes, {len(tunel_agg['frequent'])} frequent")
    
    # Compare assays
    print("\n[4/4] Comparing HNE vs TUNEL...")
    comparison = {}
    if hne_agg and tunel_agg:
        comparison = compare_assays(hne_agg['frequent'], tunel_agg['frequent'])
        print(f"  Genes in both: {len(comparison['both'])}")
        print(f"  HNE-only: {len(comparison['hne_only'])}")
        print(f"  TUNEL-only: {len(comparison['tunel_only'])}")
    
    # Save results
    save_aggregated_results(hne_agg, tunel_agg, comparison, args.data_dir)
    
    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
