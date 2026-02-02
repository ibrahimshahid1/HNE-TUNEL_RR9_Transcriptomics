#!/usr/bin/env python3
"""
Script 01: Download and cache NASA GeneLab data.

Downloads:
- RNA-seq count matrices from OSD-255
- Phenotype data (HNE, TUNEL)
- GTF annotation file for TPM calculation
- Sample metadata

Usage:
    python scripts/01_download_data.py [--data-dir DATA_DIR]
"""

import os
import sys
import argparse
from urllib.request import urlretrieve

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loading import read_meta_data, read_rnaseq_data, read_phenotype_data
from src.config import GTF_FILENAME, GTF_URL, DATA_DIR


def download_gtf(data_dir):
    """Download GTF annotation file for TPM calculation."""
    gtf_path = os.path.join(data_dir, GTF_FILENAME)
    
    if os.path.exists(gtf_path):
        print(f"GTF file already exists: {gtf_path}")
        return gtf_path
    
    print(f"Downloading GTF file from Ensembl...")
    urlretrieve(GTF_URL, gtf_path)
    print(f"GTF file saved to: {gtf_path}")
    
    return gtf_path


def download_rnaseq_data(data_dir):
    """Download RNA-seq count matrices."""
    datasets = [
        '255_rna_seq_Normalized_Counts',
        '255_rna_seq_RSEM_Unnormalized_Counts',
        '255_rna_seq_STAR_Unnormalized_Counts',
    ]
    
    for dataset in datasets:
        output_path = os.path.join(data_dir, f'{dataset}.csv')
        
        if os.path.exists(output_path):
            print(f"RNA-seq data already exists: {output_path}")
            continue
        
        print(f"Downloading {dataset}...")
        df = read_rnaseq_data(dataset)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")


def download_metadata(data_dir):
    """Download sample metadata."""
    output_path = os.path.join(data_dir, 'metadata_255.csv')
    
    if os.path.exists(output_path):
        print(f"Metadata already exists: {output_path}")
        return
    
    print("Downloading metadata...")
    df = read_meta_data('255')
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")


def download_phenotype_data(data_dir):
    """Download phenotype (HNE and TUNEL) data."""
    phenotypes = [
        ('557', 'LSDS-1_immunostaining_microscopy_HNEtr_Transformed_Reusable_Results', 'hne'),
        ('568', 'LSDS-5_immunostaining_microscopy_TUNELtr_TRANSFORMED', 'tunel'),
    ]
    
    for dataset_id, filename, short_name in phenotypes:
        output_path = os.path.join(data_dir, f'phenotype_{short_name}.csv')
        
        if os.path.exists(output_path):
            print(f"Phenotype data already exists: {output_path}")
            continue
        
        print(f"Downloading {short_name} phenotype data...")
        df = read_phenotype_data(dataset_id, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download NASA GeneLab data')
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default=DATA_DIR,
        help='Directory to save downloaded data'
    )
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    print("=" * 60)
    print("NASA GeneLab Data Download")
    print("=" * 60)
    
    # Download all data
    print("\n[1/4] Downloading GTF annotation file...")
    download_gtf(args.data_dir)
    
    print("\n[2/4] Downloading RNA-seq count matrices...")
    download_rnaseq_data(args.data_dir)
    
    print("\n[3/4] Downloading sample metadata...")
    download_metadata(args.data_dir)
    
    print("\n[4/4] Downloading phenotype data...")
    download_phenotype_data(args.data_dir)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data saved to: {os.path.abspath(args.data_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
