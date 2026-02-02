"""
Utility functions for the RNA-seq analysis pipeline.

This module provides general-purpose utilities including:
- Results storage and loading
- Sample grouping by conditions
- Display configuration
"""

import os
import numpy as np
import pandas as pd


def set_maxdisplay(n=None):
    """
    Set maximum display rows for pandas and notebook output.
    
    Args:
        n (int): Maximum rows to display (None for unlimited)
    """
    pd.set_option('display.max_rows', n)
    
    try:
        from notebook.services.config import ConfigManager
        cm = ConfigManager().update('notebook', {'limit_output': n})
    except ImportError:
        pass  # Not in notebook environment


def store_results(genes, perfs, X_hne, X_tunel, loc):
    """
    Save analysis results to disk.
    
    Args:
        genes (dict): Gene sets from different analyses
        perfs (dict): Performance metrics
        X_hne (pd.DataFrame): HNE expression data
        X_tunel (pd.DataFrame): TUNEL expression data
        loc (str): Output directory path
    """
    if not os.path.exists(loc):
        os.makedirs(loc)
    
    # Save genes dictionary
    with open(os.path.join(loc, 'genes.txt'), 'w') as f:
        f.write(str(genes))
    
    # Save performance metrics
    with open(os.path.join(loc, 'perfs.txt'), 'w') as f:
        f.write(str(perfs))
    
    # Save expression data
    if X_hne is not None:
        X_hne.to_csv(os.path.join(loc, 'X_hne.csv'), index=False)
    
    if X_tunel is not None:
        X_tunel.to_csv(os.path.join(loc, 'X_tunel.csv'), index=False)
    
    print(f"Results saved to {loc}")


def load_results(loc):
    """
    Load previously saved analysis results.
    
    Args:
        loc (str): Directory containing saved results
        
    Returns:
        dict: Dictionary with 'genes', 'perfs', 'X_hne', 'X_tunel'
    """
    results = {}
    
    genes_path = os.path.join(loc, 'genes.txt')
    if os.path.exists(genes_path):
        with open(genes_path, 'r') as f:
            results['genes'] = eval(f.read())
    
    perfs_path = os.path.join(loc, 'perfs.txt')
    if os.path.exists(perfs_path):
        with open(perfs_path, 'r') as f:
            results['perfs'] = eval(f.read())
    
    hne_path = os.path.join(loc, 'X_hne.csv')
    if os.path.exists(hne_path):
        results['X_hne'] = pd.read_csv(hne_path)
    
    tunel_path = os.path.join(loc, 'X_tunel.csv')
    if os.path.exists(tunel_path):
        results['X_tunel'] = pd.read_csv(tunel_path)
    
    return results


def run_create_groups(metadata_orig, X_orig):
    """
    Split samples into space and ground control groups.
    
    Args:
        metadata_orig (pd.DataFrame): Sample metadata
        X_orig (pd.DataFrame): Expression data (samples x genes with 'sample' column)
        
    Returns:
        tuple: (X_space, X_ground, X_space_array, X_ground_array)
    """
    y_conditions_groups = {'space': [], 'ground': []}
    
    for i in range(len(metadata_orig)):
        sample = metadata_orig.iloc[i]['Sample Name']
        if metadata_orig.iloc[i]['Factor Value[Spaceflight]'] == 'Space Flight':
            y_conditions_groups['space'].append(sample)
        else:
            y_conditions_groups['ground'].append(sample)
    
    print(f"  Space samples: {len(y_conditions_groups['space'])}")
    print(f"  Ground samples: {len(y_conditions_groups['ground'])}")
    
    # Subset expression data
    X_space = X_orig[X_orig['sample'].isin(y_conditions_groups['space'])]
    X_ground = X_orig[X_orig['sample'].isin(y_conditions_groups['ground'])]
    
    # Convert to numpy arrays
    X_space_array = np.array(X_space.drop(columns=['sample']))
    X_ground_array = np.array(X_ground.drop(columns=['sample']))
    
    return X_space, X_ground, X_space_array, X_ground_array


def intersect_samples(A_list, B_list):
    """
    Find intersection of samples between two lists using ID matching.
    
    Handles sample naming conventions where 'G' maps to 'GC' (ground control)
    and 'F' maps to 'F' (flight).
    
    Args:
        A_list (list): First list of sample names
        B_list (list): Second list of sample names
        
    Returns:
        list: Common sample identifiers
    """
    samples_A_dict = {}
    samples_B_list = []
    
    for sample in A_list:
        num = ""
        for c in sample:
            if c.isdigit():
                num += str(c)
        
        if "G" in sample:
            samples_A_dict["GC" + num] = sample
        elif "F" in sample:
            samples_A_dict["F" + num] = sample
    
    for sample in B_list:
        num = ""
        for c in sample:
            if c.isdigit():
                num += str(c)
        
        if "G" in sample:
            samples_B_list.append("GC" + num)
        elif "F" in sample:
            samples_B_list.append("F" + num)
    
    # Find intersection
    samples_both = list(set(samples_A_dict.keys()) & set(samples_B_list))
    
    return samples_both


def exclude_samples_by_prefix(df, prefix="V", colname="Source Name"):
    """
    Get list of samples to exclude based on name prefix.
    
    Args:
        df (pd.DataFrame): DataFrame with sample information
        prefix (str): Prefix to match for exclusion
        colname (str): Column containing sample names
        
    Returns:
        list: Sample names to exclude
    """
    sample_names = list(df[colname].values)
    exclude_names = [sn for sn in sample_names if sn.startswith(prefix)]
    return exclude_names


def aggregate_gene_results(gene_dict, seeds, experiments, r2_threshold=0.9):
    """
    Aggregate gene results across multiple seeds and experiments.
    
    Args:
        gene_dict (dict): Dictionary of gene results per seed/experiment
        seeds (list): List of random seeds used
        experiments (list): List of experiment configs
        r2_threshold (float): Minimum R² to include results
        
    Returns:
        dict: Aggregated results with union, majority, intersection genes
    """
    union_genes = {}
    majority_genes = {}
    intersection_genes = {}
    
    from collections import Counter
    
    for seed in seeds:
        for experiment in experiments:
            exp_name = experiment['name']
            key = f"{exp_name}_{seed}"
            
            if key not in gene_dict:
                continue
            
            # Union of all genes from all models
            all_model_genes = set()
            for model in gene_dict[key]:
                if isinstance(gene_dict[key][model], set):
                    all_model_genes.update(gene_dict[key][model])
            
            union_genes[key] = all_model_genes
            
            # Majority voting (genes appearing in >2 models)
            gene_counts = Counter()
            for model in gene_dict[key]:
                if isinstance(gene_dict[key][model], set):
                    for gene in gene_dict[key][model]:
                        gene_counts[gene] += 1
            
            majority_genes[key] = {g for g, c in gene_counts.items() if c > 2}
            
            # Intersection (genes in all models)
            model_gene_sets = [
                gene_dict[key][m] for m in gene_dict[key] 
                if isinstance(gene_dict[key][m], set)
            ]
            if model_gene_sets:
                intersection_genes[key] = set.intersection(*model_gene_sets)
            else:
                intersection_genes[key] = set()
    
    return {
        'union': union_genes,
        'majority': majority_genes,
        'intersection': intersection_genes
    }


def generate_experiment_key(cvs, alpha, rna_seq, n_genes, test_size, 
                           pheno_var, reg_perf, k_corr, lowcount, 
                           x_list, experiment_name, seed):
    """
    Generate a unique key for an experiment configuration.
    
    Args:
        cvs: Coefficient of variation threshold
        alpha: DESeq2 significance threshold
        rna_seq: RNA-seq data type
        n_genes: Number of genes to extract
        test_size: Test set fraction
        pheno_var: Phenotype variable name
        reg_perf: Performance metric
        k_corr: Correlation filter k
        lowcount: Low count threshold tuple
        x_list: Transformation list
        experiment_name: Name of experiment
        seed: Random seed
        
    Returns:
        str: Unique experiment key
    """
    return (f"{cvs}_{alpha}_{rna_seq}_{n_genes}_{test_size}_{pheno_var}_"
            f"{reg_perf}_{k_corr}_{lowcount}_{x_list}_{experiment_name}_{seed}")
