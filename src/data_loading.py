"""
Data loading utilities for NASA GLDS RNA-seq datasets.

This module handles downloading and loading data from the NASA GeneLab
Data System (GLDS), including RNA-seq count matrices, metadata, and
phenotype measurements.
"""

import pandas as pd
import zipfile
from urllib.request import urlretrieve
import warnings
warnings.filterwarnings('ignore')


def read_meta_data(dataset):
    """
    Download and read metadata for a NASA GLDS dataset.
    
    Args:
        dataset (str): Dataset ID (e.g., '255')
        
    Returns:
        pd.DataFrame: Metadata with sample information
    """
    url = (f'https://osdr.nasa.gov/geode-py/ws/studies/OSD-{dataset}/'
           f'download?source=datamanager&file=OSD-{dataset}_metadata_OSD-{dataset}-ISA.zip')
    filename = f'{dataset}-meta.zip'
    
    urlretrieve(url, filename)
    
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall()
    
    df = pd.read_csv(f's_OSD-{dataset}.txt', sep='\t', header=0)
    return df


def read_rnaseq_data(data):
    """
    Download RNA-seq count matrix from NASA GLDS.
    
    Args:
        data (str): Full filename identifier 
                   (e.g., '255_rna_seq_Normalized_Counts')
        
    Returns:
        pd.DataFrame: Count matrix (genes x samples)
    """
    dataset = data.split('_')[0]
    url = (f'https://osdr.nasa.gov/geode-py/ws/studies/OSD-{dataset}/'
           f'download?source=datamanager&file=GLDS-{data}.csv')
    df = pd.read_csv(url)
    return df


def read_phenotype_data(dataset, data):
    """
    Download phenotype/assay measurements from NASA GLDS.
    
    Args:
        dataset (str): Dataset ID (e.g., '557')
        data (str): Filename for phenotype data
        
    Returns:
        pd.DataFrame: Phenotype measurements
    """
    url = (f'https://osdr.nasa.gov//geode-py/ws/studies/OSD-{dataset}/'
           f'download?source=datamanager&file={data}.csv')
    df = pd.read_csv(url)
    return df


def read_data_and_metadata_tunel(data, metadata, rna_seq_key):
    """
    Load and align TUNEL phenotype data with RNA-seq and metadata.
    
    This function ensures proper alignment of samples across:
    - RNA-seq count matrix
    - Sample metadata
    - TUNEL phenotype measurements (cell death marker)
    
    Args:
        data (dict): Global data dictionary
        metadata (dict): Global metadata dictionary
        rna_seq_key (str): Filename key for count matrix 
                          (e.g., '255_rna_seq_STAR_Unnormalized_Counts')
    
    Returns:
        tuple: (X_orig, metadata_orig, tunel_data)
            - X_orig: Count matrix (genes x samples)
            - metadata_orig: Aligned metadata (samples x features)
            - tunel_data: Aligned phenotype data
    """
    print(f"Loading TUNEL data with RNA-seq key: {rna_seq_key}")
    
    # Load TUNEL phenotype data
    data['tunel'] = read_phenotype_data(
        '568', 
        'LSDS-5_immunostaining_microscopy_TUNELtr_TRANSFORMED'
    )

    # Load metadata if not already loaded
    if '255' not in metadata: 
        metadata['255'] = read_meta_data('255')

    # Load RNA-seq count matrix
    X_orig = read_rnaseq_data(rna_seq_key) 

    # Add 'Source Name' to tunel data for matching
    source_names = [sample.split("_")[0] for sample in data['tunel']['Sample_Name']]
    data['tunel']['Source Name'] = source_names

    # Find common samples
    common_source_names = list(
        set(data['tunel']['Source Name']) & set(metadata['255']['Source Name'])
    )
    
    # Filter metadata and get sample names
    meta_255_subset = metadata['255'][
        metadata['255']['Source Name'].isin(common_source_names)
    ]
    samples_255_to_keep = list(meta_255_subset['Sample Name'])
    
    # Filter phenotype data
    tunel_data = data['tunel'][
        data['tunel']['Source Name'].isin(common_source_names)
    ]

    # Filter count matrix to common samples
    X_orig = X_orig[['Unnamed: 0'] + samples_255_to_keep]

    # Transpose, sort, and align
    X_orig = transpose_df(X_orig, 'Unnamed: 0', 'sample')
    X_orig = X_orig.sort_values(by=['sample']).reset_index(drop=True)

    metadata_orig = meta_255_subset[
        meta_255_subset['Sample Name'].isin(X_orig['sample'])
    ]
    metadata_orig = metadata_orig.sort_values(by=['Sample Name']).reset_index(drop=True)

    # Critical alignment check
    assert list(X_orig['sample']) == list(metadata_orig['Sample Name']), \
        "FATAL: Sample names in X_orig and metadata_orig do not match!"

    # Transpose back to (genes x samples)
    X_orig = transpose_df(X_orig, 'sample', 'Unnamed: 0')
    
    print(f"  [TUNEL] X shape (genes x samples): {X_orig.shape}")
    print(f"  [TUNEL] Metadata shape: {metadata_orig.shape}")

    return X_orig, metadata_orig, tunel_data


def read_data_and_metadata_hne(data, metadata, rna_seq_key):
    """
    Load and align HNE phenotype data with RNA-seq and metadata.
    
    HNE (4-Hydroxynonenal) is a marker of lipid peroxidation/oxidative stress.
    
    Args:
        data (dict): Global data dictionary
        metadata (dict): Global metadata dictionary
        rna_seq_key (str): Filename key for count matrix
    
    Returns:
        tuple: (X_orig, metadata_orig, hne_data)
    """
    print(f"Loading HNE data with RNA-seq key: {rna_seq_key}")
    
    # Load HNE phenotype data
    data['hne'] = read_phenotype_data(
        '557', 
        'LSDS-1_immunostaining_microscopy_HNEtr_Transformed_Reusable_Results'
    )

    # Load metadata if not already loaded
    if '255' not in metadata: 
        metadata['255'] = read_meta_data('255')

    # Load RNA-seq count matrix
    X_orig = read_rnaseq_data(rna_seq_key) 

    # Find common samples based on 'Source Name'
    common_source_names = list(
        set(data['hne']['Source Name']) & set(metadata['255']['Source Name'])
    )
    
    meta_255_subset = metadata['255'][
        metadata['255']['Source Name'].isin(common_source_names)
    ]
    samples_255_to_keep = list(meta_255_subset['Sample Name'])
    
    hne_data = data['hne'][
        data['hne']['Source Name'].isin(common_source_names)
    ]

    # Filter count matrix
    X_orig = X_orig[['Unnamed: 0'] + samples_255_to_keep]

    # Align and sort
    X_orig = transpose_df(X_orig, 'Unnamed: 0', 'sample')
    X_orig = X_orig.sort_values(by=['sample']).reset_index(drop=True)

    metadata_orig = meta_255_subset[
        meta_255_subset['Sample Name'].isin(X_orig['sample'])
    ]
    metadata_orig = metadata_orig.sort_values(by=['Sample Name']).reset_index(drop=True)

    assert list(X_orig['sample']) == list(metadata_orig['Sample Name']), \
        "FATAL: Sample names do not match after sorting!"

    X_orig = transpose_df(X_orig, 'sample', 'Unnamed: 0')
    
    print(f"  [HNE] X shape (genes x samples): {X_orig.shape}")
    print(f"  [HNE] Metadata shape: {metadata_orig.shape}")

    return X_orig, metadata_orig, hne_data


def transpose_df(df, cur_index_col, new_index_col):
    """
    Transpose a dataframe and set new index.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cur_index_col (str): Column to use as index before transposing
        new_index_col (str): Name for the new index column after transpose
        
    Returns:
        pd.DataFrame: Transposed dataframe
    """
    df = df.set_index(cur_index_col).T
    df.reset_index(level=0, inplace=True)
    cols = [new_index_col] + list(df.columns)[1:]
    df.columns = cols
    return df


def run_create_Y(metadata_orig, pheno_data, pheno_var):
    """
    Create continuous target vector (Y) from phenotype data.
    
    Args:
        metadata_orig (pd.DataFrame): Sample metadata
        pheno_data (pd.DataFrame): Phenotype measurements
        pheno_var (str): Column name of phenotype variable to use
        
    Returns:
        np.ndarray: Continuous phenotype values aligned with samples
    """
    import numpy as np
    
    Y_dict = {'condition': {}, 'pheno val': {}, 'pheno class': {}}

    for i in range(len(metadata_orig)):
        sample = metadata_orig.iloc[i]['Sample Name']
        source = metadata_orig.iloc[i]['Source Name']
        
        # Find matching phenotype value
        pheno_val_series = pheno_data[
            pheno_data['Source Name'] == source
        ][pheno_var]
        
        if pheno_val_series.empty:
            print(f"Warning: No pheno data for {source}")
            continue
            
        pheno_val = pheno_val_series.values[0]
        Y_dict['pheno val'][sample] = pheno_val
        
        # Condition (flight vs ground)
        cond = 1 if metadata_orig.iloc[i]['Factor Value[Spaceflight]'] == 'Space Flight' else 0
        Y_dict['condition'][sample] = cond

    # Sort by sample name for alignment
    Y_dict_pheno_vals = dict(sorted(Y_dict['pheno val'].items()))
    y_vals = np.array(list(Y_dict_pheno_vals.values()))

    print(f'  [Y Values] n={len(y_vals)}, Mean={np.mean(y_vals):.2f}, '
          f'Median={np.median(y_vals):.2f}')
    
    return y_vals