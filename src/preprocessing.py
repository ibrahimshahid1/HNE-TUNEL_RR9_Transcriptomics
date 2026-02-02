"""
Preprocessing utilities for gene expression data.

This module provides functions for:
- Gene filtering (variance, low-count, biotype)
- Data transformations (TPM, log, standardization, power)
- Coefficient of variation filtering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybiomart import Server
import warnings
warnings.filterwarnings('ignore')


def drop_nans(df):
    """
    Remove rows with NaN values from dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with NaN rows removed
    """
    return df.dropna(inplace=False)


def filter_cvs(df, thresh=0.5):
    """
    Filter genes by coefficient of variation.
    
    Keeps only genes with CV > threshold, which helps filter out
    genes with low variability across samples.
    
    Args:
        df (pd.DataFrame): Gene expression matrix (genes x samples)
        thresh (float): Minimum coefficient of variation threshold
        
    Returns:
        pd.DataFrame: Filtered dataframe containing only high-variance genes
    """
    cvs = []
    for i in range(len(df)):
        m = np.mean(df.iloc[i][1:])
        sd = np.std(df.iloc[i][1:])
        cvs.append(sd / m if m != 0 else 0)
    
    # Plot histogram of coefficient of variation distribution
    fig, axs = plt.subplots()
    axs.hist(cvs, bins=20)
    plt.close()
    
    # Keep genes with cv > thresh
    indices = [i for i in range(len(cvs)) if cvs[i] > thresh]
    return df.iloc[indices]


def filterGenesByPercentLowCount(df, n=0, p=0):
    """
    Filter genes with low counts across a percentage of samples.
    
    Removes genes that have counts <= n in more than p% of samples.
    
    Args:
        df (pd.DataFrame): Gene expression matrix (genes x samples)
        n (int): Count threshold (genes with counts <= n are considered low)
        p (float): Proportion of samples threshold (0-1)
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if n == 0 or p == 0:
        return df
    
    # Count samples with low expression for each gene
    count_cols = df.loc[:, df.columns != 'Unnamed: 0']
    low_count_mask = (count_cols < n).sum(axis='columns') <= int(p * len(df.columns))
    return df[low_count_mask]


def filter_genes(df, drop='non-coding'):
    """
    Filter genes based on biotype (protein-coding vs non-coding).
    
    Uses pybiomart to query Ensembl for gene biotype information.
    
    Args:
        df (pd.DataFrame): Gene expression matrix with 'Unnamed: 0' column
                          containing Ensembl gene IDs
        drop (str): Either 'non-coding' (keep only protein-coding genes)
                   or 'coding' (keep only non-coding genes)
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if drop is None or drop == 0:
        return df
    
    server = Server(host='http://www.ensembl.org')
    dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['mmusculus_gene_ensembl']
    gene_info = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'gene_biotype'])
    
    if drop == 'non-coding':
        filter_genes_list = gene_info[gene_info['Gene type'] == 'protein_coding']['Gene stable ID']
    elif drop == 'coding':
        filter_genes_list = gene_info[gene_info['Gene type'] != 'protein_coding']['Gene stable ID']
    else:
        return df
    
    return df[df['Unnamed: 0'].isin(filter_genes_list)]


def filter_data(df, y_vals, dropnans=False, dropgenes='non-coding', 
                droplowcvs=0, correlation=0, seed=None):
    """
    Orchestrate multiple filtering steps on gene expression data.
    
    Args:
        df (pd.DataFrame): Gene expression matrix
        y_vals (np.ndarray): Target values (for correlation filtering)
        dropnans (bool): Whether to drop NaN rows
        dropgenes (str): Biotype filtering ('non-coding', 'coding', or None)
        droplowcvs (float): CV threshold (0 to skip)
        correlation (int): Number of top correlated genes to keep (0 to skip)
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    from .feature_selection import filterNotCorrelated
    from .data_loading import transpose_df
    
    if dropnans:
        df = drop_nans(df)
    
    if dropgenes is not None:
        df = filter_genes(df, drop=dropgenes)
    
    if droplowcvs != 0:
        df = filter_cvs(df, droplowcvs)
    
    if correlation != 0:
        df = filterNotCorrelated(df, y_vals, correlation, seed)
    
    return df


def run_filter_data(X_orig, metadata_orig, y_vals, y_classes, cvs=0, alpha=0, 
                    lowcount=(0, 0), correlation=0, seed=None):
    """
    Execute the complete data filtering pipeline.
    
    Pipeline order:
    1. Drop NaN values
    2. Filter low-count genes
    3. Filter non-coding genes
    4. Filter by differential expression (DESeq2)
    5. Filter by coefficient of variation
    6. Filter by correlation with target
    
    Args:
        X_orig (pd.DataFrame): Gene expression matrix (genes x samples)
        metadata_orig (pd.DataFrame): Sample metadata
        y_vals (np.ndarray): Continuous target values
        y_classes (np.ndarray): Binary class labels
        cvs (float): CV threshold
        alpha (float): DESeq2 significance threshold
        lowcount (tuple): (count_threshold, proportion_threshold)
        correlation (int): Number of top correlated genes
        seed (int): Random seed
        
    Returns:
        tuple: (X_orig DataFrame, X_array numpy array)
    """
    from .deseq2_utils import filter_by_dgea
    from .data_loading import transpose_df
    
    # Filter NaNs
    X_orig = filter_data(X_orig, y_vals, dropnans=True, dropgenes=None, 
                         droplowcvs=0, correlation=0)
    print(f"  X shape after filter NaNs: {X_orig.shape}")
    
    # Filter low-count genes
    X_orig = filterGenesByPercentLowCount(X_orig, n=lowcount[0], p=lowcount[1])
    print(f"  X shape after filter lowcount={lowcount}: {X_orig.shape}")
    
    # Filter non-coding genes
    X_orig = filter_data(X_orig, y_vals, dropnans=False, dropgenes='non-coding', 
                         droplowcvs=0, correlation=0, seed=seed)
    print(f"  X shape after filter non-coding: {X_orig.shape}")
    
    # Filter by differential expression
    X_orig = filter_by_dgea(X_orig, metadata_orig, y_classes, pval=alpha, l2fc=0)
    print(f"  X shape after filter by DGEA: {X_orig.shape}")
    
    # Filter by coefficient of variation
    X_orig = filter_data(X_orig, y_vals, dropnans=False, dropgenes=None, 
                         droplowcvs=cvs, correlation=0, seed=seed)
    print(f"  X shape after filter low CVs: {X_orig.shape}")
    
    # Filter by correlation
    X_orig = filter_data(X_orig, y_vals, dropnans=False, dropgenes=None, 
                         droplowcvs=0, correlation=correlation, seed=seed)
    print(f"  X shape after filter non-correlated: {X_orig.shape}")
    
    # Transpose to samples x genes
    X_orig = transpose_df(X_orig, 'Unnamed: 0', 'sample')
    
    # Convert to numpy array
    X_array = np.array(X_orig.drop(columns=['sample']))
    
    return X_orig, X_array


def full_transform(X, x_list, gtf_path=None):
    """
    Apply a sequence of transformations to gene expression data.
    
    Supported transformations:
    - 'tpm': Transcripts Per Million normalization
    - 'log': Log2(x+1) transformation  
    - 'std': Standardization (z-score)
    - 'power': Yeo-Johnson power transformation
    - 'boxcox': Box-Cox transformation
    
    Args:
        X (pd.DataFrame or np.ndarray): Expression data (samples x genes)
        x_list (list): List of transformation names to apply in order
        gtf_path (str): Path to GTF file for TPM calculation
        
    Returns:
        np.ndarray or pd.DataFrame: Transformed data
    """
    from sklearn.preprocessing import StandardScaler, power_transform
    
    temp = X
    
    if 'tpm' in x_list:
        try:
            from rnanorm import TPM
            print(f'  Shape before TPM: {temp.shape}')
            
            if gtf_path is None:
                gtf_path = 'Mus_musculus.GRCm39.115.gtf.gz'
            
            tpm_calculator = TPM(gtf=gtf_path)
            tpm_calculator.set_output(transform="pandas")
            temp = tpm_calculator.fit_transform(temp)
        except ImportError:
            print("  Warning: rnanorm not installed, skipping TPM")
        except Exception as e:
            print(f"  Warning: TPM failed: {e}")
    
    if 'log' in x_list:
        # Log2(x+1) transformation
        # Assumes genes x samples, transpose if needed
        temp_t = temp.T if hasattr(temp, 'T') else temp
        print(f'  Shape before log: {temp_t.shape}')
        temp_log = np.log2(temp_t + 1)
        temp = temp_log.T if hasattr(temp_log, 'T') else temp_log
    
    if 'boxcox' in x_list:
        temp = myboxcox(temp)
    
    if 'std' in x_list:
        print(f'  Shape before standardization: {temp.shape}')
        # Z-score standardization
        if isinstance(temp, pd.DataFrame):
            temp_vals = temp.values
        else:
            temp_vals = temp
        scaled = (temp_vals - np.mean(temp_vals, axis=0) + 0.01) / (np.std(temp_vals, axis=0) + 0.01)
        temp = scaled
    
    if 'power' in x_list:
        # Yeo-Johnson power transformation
        temp_power = np.zeros((temp.shape[0], temp.shape[1]))
        for i in range(temp.shape[1]):
            col = temp[:, i] if isinstance(temp, np.ndarray) else temp.iloc[:, i].values
            temp_power[:, i] = power_transform(
                col.reshape(-1, 1), 
                method='yeo-johnson', 
                standardize=True
            ).reshape(-1)
        temp = temp_power
    
    return temp


def myboxcox(df):
    """
    Apply Box-Cox transformation to each row of a dataframe.
    
    Box-Cox requires positive values, so we add 1 before transforming.
    
    Args:
        df (pd.DataFrame): Input dataframe (genes x samples)
        
    Returns:
        pd.DataFrame: Box-Cox transformed dataframe
    """
    from scipy.stats import boxcox
    
    df1 = df + 1
    bcDF = pd.DataFrame()
    
    for i in range(len(df1)):
        bc_array, bc_lambda = boxcox(df1.iloc[i])
        bcDF = pd.concat([bcDF, pd.DataFrame(np.array(bc_array).reshape(1, -1))], 
                        ignore_index=True)
    
    return bcDF
