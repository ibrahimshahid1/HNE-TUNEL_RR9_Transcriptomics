#!/usr/bin/env python3
"""
Script 04: Run pathway enrichment analysis.

This script:
1. Loads consensus genes from model training
2. Runs Gene Ontology and KEGG pathway enrichment
3. Saves enrichment results

Usage:
    python scripts/04_run_enrichment.py --assay hne --seed 42 [--output-dir OUTPUT_DIR]
    
Note: Requires gseapy package (pip install gseapy)
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import RESULTS_DIR


def load_consensus_genes(assay, seed, data_dir):
    """Load consensus genes from training results."""
    genes_path = os.path.join(data_dir, f'{assay}_seed{seed}', 'consensus_genes.txt')
    
    if not os.path.exists(genes_path):
        raise FileNotFoundError(
            f"Consensus genes not found: {genes_path}\n"
            f"Run 03_train_models.py first."
        )
    
    genes = {
        'union': set(),
        'majority': set(),
        'intersection': set(),
    }
    
    current_section = None
    
    with open(genes_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if 'UNION GENES:' in line:
                current_section = 'union'
            elif 'MAJORITY GENES' in line:
                current_section = 'majority'
            elif 'INTERSECTION GENES:' in line:
                current_section = 'intersection'
            elif 'PER-MODEL GENES:' in line:
                current_section = None
            elif current_section and line:
                gene_list = [g.strip() for g in line.split(',') if g.strip()]
                genes[current_section].update(gene_list)
    
    return genes


def run_enrichment_gseapy(gene_list, organism='mouse', output_dir=None, prefix=''):
    """
    Run pathway enrichment using gseapy.
    
    Args:
        gene_list (list): List of gene symbols
        organism (str): 'mouse' or 'human'
        output_dir (str): Directory to save results
        prefix (str): Prefix for output files
        
    Returns:
        dict: Enrichment results by database
    """
    try:
        import gseapy as gp
    except ImportError:
        print("Error: gseapy not installed. Install with: pip install gseapy")
        return {}
    
    if len(gene_list) < 3:
        print(f"  Warning: Only {len(gene_list)} genes, skipping enrichment")
        return {}
    
    results = {}
    
    # Set organism for gseapy
    if organism == 'mouse':
        org_db = 'mouse'
    else:
        org_db = 'human'
    
    # Gene Ontology Biological Process
    print("  Running GO Biological Process enrichment...")
    try:
        go_bp = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=['GO_Biological_Process_2021'],
            organism=org_db,
            outdir=None,
        )
        results['GO_BP'] = go_bp.results
        
        if output_dir:
            go_bp.results.to_csv(
                os.path.join(output_dir, f'{prefix}GO_Biological_Process.csv'),
                index=False
            )
    except Exception as e:
        print(f"    GO BP failed: {e}")
    
    # KEGG Pathways
    print("  Running KEGG pathway enrichment...")
    try:
        kegg = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=['KEGG_2021_Mouse'] if organism == 'mouse' else ['KEGG_2021_Human'],
            organism=org_db,
            outdir=None,
        )
        results['KEGG'] = kegg.results
        
        if output_dir:
            kegg.results.to_csv(
                os.path.join(output_dir, f'{prefix}KEGG.csv'),
                index=False
            )
    except Exception as e:
        print(f"    KEGG failed: {e}")
    
    # GO Molecular Function
    print("  Running GO Molecular Function enrichment...")
    try:
        go_mf = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=['GO_Molecular_Function_2021'],
            organism=org_db,
            outdir=None,
        )
        results['GO_MF'] = go_mf.results
        
        if output_dir:
            go_mf.results.to_csv(
                os.path.join(output_dir, f'{prefix}GO_Molecular_Function.csv'),
                index=False
            )
    except Exception as e:
        print(f"    GO MF failed: {e}")
    
    # GO Cellular Component
    print("  Running GO Cellular Component enrichment...")
    try:
        go_cc = gp.enrichr(
            gene_list=list(gene_list),
            gene_sets=['GO_Cellular_Component_2021'],
            organism=org_db,
            outdir=None,
        )
        results['GO_CC'] = go_cc.results
        
        if output_dir:
            go_cc.results.to_csv(
                os.path.join(output_dir, f'{prefix}GO_Cellular_Component.csv'),
                index=False
            )
    except Exception as e:
        print(f"    GO CC failed: {e}")
    
    return results


def summarize_enrichment(results, top_n=10):
    """Print top enriched terms."""
    for db_name, df in results.items():
        if df is None or len(df) == 0:
            continue
            
        print(f"\n  Top {top_n} {db_name} terms:")
        
        # Sort by adjusted p-value
        if 'Adjusted P-value' in df.columns:
            df_sorted = df.sort_values('Adjusted P-value').head(top_n)
            for _, row in df_sorted.iterrows():
                term = row.get('Term', 'Unknown')[:50]
                pval = row.get('Adjusted P-value', 1.0)
                print(f"    {term}: p={pval:.2e}")


def main():
    parser = argparse.ArgumentParser(description='Run pathway enrichment')
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
        help='Directory containing model results'
    )
    parser.add_argument(
        '--gene-set',
        type=str,
        default='majority',
        choices=['union', 'majority', 'intersection'],
        help='Which gene set to use for enrichment'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Pathway Enrichment: {args.assay.upper()} (seed={args.seed})")
    print("=" * 60)
    
    # Load consensus genes
    print("\n[1/2] Loading consensus genes...")
    genes = load_consensus_genes(args.assay, args.seed, args.data_dir)
    
    gene_list = genes[args.gene_set]
    print(f"  Using {args.gene_set} genes: {len(gene_list)} genes")
    
    if len(gene_list) == 0:
        print("  No genes found, exiting.")
        return
    
    print(f"  Genes: {', '.join(sorted(gene_list)[:10])}...")
    
    # Create enrichment output directory
    enrichment_dir = os.path.join(
        args.data_dir, f'{args.assay}_seed{args.seed}', 'enrichment'
    )
    os.makedirs(enrichment_dir, exist_ok=True)
    
    # Run enrichment
    print("\n[2/2] Running enrichment analysis...")
    results = run_enrichment_gseapy(
        gene_list=gene_list,
        organism='mouse',
        output_dir=enrichment_dir,
        prefix=f'{args.assay}_{args.gene_set}_',
    )
    
    # Summarize results
    if results:
        summarize_enrichment(results)
        print(f"\n  Enrichment results saved to: {enrichment_dir}")
    else:
        print("\n  No enrichment results generated.")
    
    print("\n" + "=" * 60)
    print("Enrichment analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
