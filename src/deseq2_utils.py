"""
DESeq2 utilities for differential gene expression analysis.

This module provides wrapper functions for pydeseq2 to run
differential expression analysis and filter genes based on
significance thresholds.
"""

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def map_samples_to_classes(dfT, y_classes):
    """
    Map binary class labels to sample conditions for DESeq2.
    
    Args:
        dfT (pd.DataFrame): Transposed expression matrix (samples x genes)
                           with 'sample' column
        y_classes (np.ndarray): Binary class labels (0 or 1)
        
    Returns:
        pd.DataFrame: DataFrame with 'sample' and 'condition' columns
    """
    condition_dict = {}
    
    for i in range(len(dfT)):
        sample = dfT.iloc[i]['sample']
        condition_dict[sample] = '0' if y_classes[i] == 0 else '1'
    
    dfT["condition"] = dfT["sample"].map(condition_dict)
    conditions = dfT[['sample', 'condition']].copy()
    
    return conditions


def map_samples_to_conditions(dfT, metadata, metadata_condition_param, 
                              condition_0, condition_1):
    """
    Map experimental conditions to samples using metadata.
    
    Args:
        dfT (pd.DataFrame): Transposed expression matrix with 'sample' column
        metadata (pd.DataFrame): Sample metadata
        metadata_condition_param (str): Column name in metadata containing condition
        condition_0: Value representing condition 0 (e.g., 'Ground Control')
        condition_1: Value representing condition 1 (e.g., 'Space Flight')
        
    Returns:
        pd.DataFrame: DataFrame with 'sample' and 'condition' columns
    """
    condition_dict = {}
    
    for sample in list(dfT['sample']):
        val = metadata[metadata['Sample Name'] == sample][metadata_condition_param].values
        
        if len(val) == 0:
            print(f"Warning: No metadata for sample {sample}")
            continue
            
        if val[0] == condition_0:
            condition_dict[sample] = '0'
        else:
            condition_dict[sample] = '1'
    
    dfT["condition"] = dfT["sample"].map(condition_dict)
    conditions = dfT[['sample', 'condition']].copy()
    
    return conditions


def run_deseq2(df, metadata, y_classes=None):
    """
    Run DESeq2 differential expression analysis.
    
    Args:
        df (pd.DataFrame): Gene expression matrix (genes x samples)
                          with gene IDs in 'Unnamed: 0' column
        metadata (pd.DataFrame): Sample metadata
        y_classes (np.ndarray): Optional binary class labels.
                               If None, uses spaceflight metadata.
        
    Returns:
        DeseqDataSet: Fitted DESeq2 dataset object
    """
    # Transpose to samples x genes
    dfT = df.T
    dfT.columns = dfT.iloc[0]
    dfT = dfT.iloc[1:]
    dfT.columns.name = None
    dfT = dfT.reset_index().rename(columns={"index": "sample"})
    
    # Map conditions
    if y_classes is None:
        conditions = map_samples_to_conditions(
            dfT, metadata,
            'Factor Value[Spaceflight]',
            'Ground Control', 'Space Flight'
        )
    else:
        conditions = map_samples_to_classes(dfT, y_classes)
    
    # Prepare count matrix for DESeq2
    counts = dfT.drop(columns=['sample', 'condition']).reset_index(drop=True)
    
    # Convert to integer counts (required by DESeq2)
    counts = counts.apply(pd.to_numeric, errors='coerce')
    counts = counts.fillna(0).astype(int)
    
    # Run DESeq2
    dds = DeseqDataSet(
        counts=counts,
        metadata=conditions,
        design_factors="condition"
    )
    dds.deseq2()
    
    return dds


def get_results(dds):
    """
    Extract differential expression results from DESeq2.
    
    Args:
        dds (DeseqDataSet): Fitted DESeq2 dataset
        
    Returns:
        pd.DataFrame: Results with log2FoldChange, pvalue, padj columns
    """
    stats_results = DeseqStats(dds, contrast=('condition', '0', '1'))
    stats_results.summary()
    res = stats_results.results_df
    
    return res


def get_sig_genes(res, pval=0.05, l2fc=0):
    """
    Filter for significantly differentially expressed genes.
    
    Args:
        res (pd.DataFrame): DESeq2 results
        pval (float): Adjusted p-value threshold
        l2fc (float): Log2 fold change threshold (absolute value)
        
    Returns:
        pd.DataFrame: Filtered results with significant genes only
    """
    sigs = res[(res.padj < pval) & (abs(res.log2FoldChange) > l2fc)]
    return sigs


def get_dge_ranked_genes(res):
    """
    Rank genes by differential expression statistic.
    
    Args:
        res (pd.DataFrame): DESeq2 results
        
    Returns:
        pd.DataFrame: Genes ranked by test statistic (descending)
    """
    ranking = res[['stat']].dropna().sort_values('stat', ascending=False)
    ranking_index = list(ranking.index)
    ranking_index_upper = [x.upper() for x in ranking_index]
    ranking.index = ranking_index_upper
    
    return ranking


def filter_by_dgea(data, metadata, y_classes=None, pval=0, l2fc=0):
    """
    Filter expression data to keep only significant DEGs.
    
    Runs DESeq2 and filters the expression matrix to include only
    genes passing significance thresholds.
    
    Args:
        data (pd.DataFrame): Gene expression matrix (genes x samples)
        metadata (pd.DataFrame): Sample metadata
        y_classes (np.ndarray): Optional class labels
        pval (float): Adjusted p-value threshold (0 to skip filtering)
        l2fc (float): Log2 fold change threshold
        
    Returns:
        pd.DataFrame: Filtered expression matrix
    """
    if pval is None or pval == 0:
        return data
    
    print(f"  Running DESeq2 with pval={pval}, l2fc={l2fc}")
    
    # Run DESeq2
    dds = run_deseq2(data, metadata, y_classes)
    
    # Get results
    res = get_results(dds)
    
    # Get significant genes
    sig_genes_df = get_sig_genes(res, pval=pval, l2fc=l2fc)
    
    # Get gene IDs
    sig_genes = list(sig_genes_df.sort_values('padj').index)
    
    print(f"  Found {len(sig_genes)} significant genes")
    
    # Filter data
    return data[data['Unnamed: 0'].isin(sig_genes)]


def run_full_dgea(data, metadata, y_classes=None, alpha=0.05, l2fc=1.0):
    """
    Run complete differential expression analysis workflow.
    
    Args:
        data (pd.DataFrame): Gene expression matrix
        metadata (pd.DataFrame): Sample metadata  
        y_classes (np.ndarray): Optional class labels
        alpha (float): Significance threshold
        l2fc (float): Log2 fold change threshold
        
    Returns:
        dict: Results containing:
            - 'all_results': Full DESeq2 results
            - 'sig_genes': Significant gene IDs
            - 'upregulated': Upregulated genes (positive l2fc)
            - 'downregulated': Downregulated genes (negative l2fc)
            - 'ranking': Ranked gene list
    """
    dds = run_deseq2(data, metadata, y_classes)
    res = get_results(dds)
    sig = get_sig_genes(res, pval=alpha, l2fc=l2fc)
    ranking = get_dge_ranked_genes(res)
    
    # Separate up/down regulated
    upregulated = sig[sig.log2FoldChange > 0].index.tolist()
    downregulated = sig[sig.log2FoldChange < 0].index.tolist()
    
    return {
        'all_results': res,
        'sig_genes': sig.index.tolist(),
        'upregulated': upregulated,
        'downregulated': downregulated,
        'ranking': ranking
    }
