"""
Feature selection utilities for gene expression analysis.

This module provides:
- Correlation-based gene filtering
- Permutation feature importance
- Gene ID to symbol conversion utilities
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import mygene
import warnings
warnings.filterwarnings('ignore')


def custom_mutual_info_regression(X, y, seed=42):
    """
    Wrapper for mutual information regression with fixed random state.
    
    Used as score function for SelectKBest feature selection.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        seed (int): Random seed
        
    Returns:
        np.ndarray: Mutual information scores for each feature
    """
    return mutual_info_regression(X, y, random_state=seed)


def filterNotCorrelated(df, y, k, seed):
    """
    Filter genes to keep only those most correlated with target.
    
    Uses mutual information regression to score features and selects
    the top k most informative genes.
    
    Args:
        df (pd.DataFrame): Gene expression matrix (genes x samples)
        y (np.ndarray): Target values
        k (int): Number of top genes to keep
        seed (int): Random seed
        
    Returns:
        pd.DataFrame: Filtered dataframe with top k correlated genes
    """
    from .data_loading import transpose_df
    
    if k == 0:
        return df
    
    # Transpose to samples x genes for sklearn
    df_t = transpose_df(df, 'Unnamed: 0', 'sample')
    samples = list(df_t['sample'])
    X = df_t.drop(columns=['sample']).to_numpy()
    
    # Create score function with seed
    def score_func(X, y):
        return custom_mutual_info_regression(X, y, seed=seed)
    
    # Select top k features
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get indices of selected features
    indices = selector.get_support(indices=True)
    
    return df.iloc[indices]


def permutation_feature_importance(model, X_, y_, genes=None, scoring='accuracy', 
                                   n=20, random_state=0):
    """
    Calculate permutation feature importance for a trained model.
    
    Permutation importance measures feature importance by randomly shuffling
    each feature and measuring the decrease in model performance.
    
    Args:
        model: Trained sklearn-compatible model
        X_ (np.ndarray): Feature matrix
        y_ (np.ndarray): Target values
        genes (list): Gene names corresponding to features
        scoring (str): Scoring metric (e.g., 'r2', 'accuracy')
        n (int): Number of top features to return
        random_state (int): Random seed
        
    Returns:
        list: Top n most important gene names (Ensembl IDs)
    """
    r = permutation_importance(
        model, X_, y_, 
        n_repeats=5, 
        scoring=scoring, 
        random_state=random_state
    )
    
    means = []
    stds = []
    gene_list = []
    
    # Get indices sorted by importance (descending)
    for i in r.importances_mean.argsort()[::-1][:n]:
        gene_list.append(genes[i])
        means.append(round(r.importances_mean[i], 4))
        stds.append(round(r.importances_std[i], 4))
    
    feat_imp_df = pd.DataFrame({
        'feature': gene_list,
        'importance_mean': means,
        'importance_std': stds
    })
    
    return list(feat_imp_df.sort_values('importance_mean', ascending=False)['feature'])


def get_symbol_from_id(gene_list):
    """
    Convert Ensembl gene IDs to gene symbols using mygene.
    
    Args:
        gene_list (list): List of Ensembl gene IDs (e.g., 'ENSMUSG00000000001')
        
    Returns:
        list: List of gene symbols (e.g., 'Gnai3')
    """
    symbol_list = []
    mg = mygene.MyGeneInfo()
    
    try:
        ginfo = mg.querymany(gene_list, scopes='ensembl.gene', verbose=False)
        seen_genes = []
        
        for g in ginfo:
            if g['query'] in seen_genes:
                continue
            if 'symbol' not in g:
                symbol_list.append(g['query'])
            else:
                symbol_list.append(g['symbol'])
            seen_genes.append(g['query'])
    except Exception as e:
        print(f"Warning: Gene symbol conversion failed: {e}")
        return gene_list
    
    return symbol_list


def convert_ids_to_names(gene_id_list):
    """
    Convert Ensembl IDs to gene symbols using EnsemblConverter.
    
    Alternative method using the Ensembl_converter package.
    
    Args:
        gene_id_list (list): List of Ensembl gene IDs
        
    Returns:
        list: List of gene symbols
    """
    try:
        from Ensembl_converter import EnsemblConverter
        
        converter = EnsemblConverter()
        result = converter.convert_ids(gene_id_list)
        
        gene_symbol_list = []
        for i in range(len(result)):
            gene_symbol_list.append(result.iloc[i]['Symbol'])
        
        return gene_symbol_list
    except ImportError:
        print("Warning: Ensembl_converter not installed, using mygene fallback")
        return get_symbol_from_id(gene_id_list)
    except Exception as e:
        print(f"Warning: ID conversion failed: {e}")
        return gene_id_list


def get_id_from_symbol(symbol_list, species='mouse'):
    """
    Convert gene symbols to Ensembl IDs.
    
    Args:
        symbol_list (list): List of gene symbols
        species (str): Species ('mouse' or 'human')
        
    Returns:
        list: List of Ensembl gene IDs
    """
    mg = mygene.MyGeneInfo()
    
    scope = 'symbol'
    if species == 'mouse':
        species_filter = 'mouse'
    else:
        species_filter = 'human'
    
    try:
        ginfo = mg.querymany(
            symbol_list, 
            scopes=scope, 
            species=species_filter,
            fields='ensembl.gene',
            verbose=False
        )
        
        ensembl_ids = []
        for g in ginfo:
            if 'ensembl' in g:
                if isinstance(g['ensembl'], list):
                    ensembl_ids.append(g['ensembl'][0]['gene'])
                else:
                    ensembl_ids.append(g['ensembl']['gene'])
            else:
                ensembl_ids.append(g['query'])  # Return original if not found
        
        return ensembl_ids
    except Exception as e:
        print(f"Warning: Symbol to ID conversion failed: {e}")
        return symbol_list
