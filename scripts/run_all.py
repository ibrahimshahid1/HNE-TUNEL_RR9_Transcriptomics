#!/usr/bin/env python3
"""
Main orchestration script - runs the complete analysis pipeline.

This script coordinates all pipeline steps:
1. Download data (optional, if not cached)
2. Preprocess data
3. Train models
4. Run enrichment analysis
5. Aggregate results

Usage:
    # Run single experiment
    python scripts/run_all.py --assay hne --seed 42
    
    # Run all experiments (all seeds, both assays)
    python scripts/run_all.py --run-all
    
    # Skip download step (data already cached)
    python scripts/run_all.py --assay tunel --seed 42 --skip-download
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import SEEDS, EXPERIMENTS, RESULTS_DIR, DATA_DIR, print_config


def run_step(script_name, args_list, step_name):
    """Run a pipeline step as a subprocess."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path] + args_list
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"  Warning: {step_name} returned non-zero exit code")
        return False
    
    return True


def run_single_experiment(assay, seed, skip_download=False, skip_enrichment=False):
    """Run the complete pipeline for a single experiment."""
    success = True
    
    # Step 1: Download data
    if not skip_download:
        success = run_step(
            '01_download_data.py',
            ['--data-dir', DATA_DIR],
            'Download Data'
        )
    
    # Step 2: Preprocess
    success = run_step(
        '02_preprocess.py',
        ['--assay', assay, '--seed', str(seed), '--output-dir', RESULTS_DIR],
        f'Preprocess {assay.upper()} (seed={seed})'
    ) and success
    
    # Step 3: Train models
    success = run_step(
        '03_train_models.py',
        ['--assay', assay, '--seed', str(seed), '--data-dir', RESULTS_DIR],
        f'Train Models {assay.upper()} (seed={seed})'
    ) and success
    
    # Step 4: Enrichment (optional)
    if not skip_enrichment:
        success = run_step(
            '04_run_enrichment.py',
            ['--assay', assay, '--seed', str(seed), '--data-dir', RESULTS_DIR],
            f'Enrichment {assay.upper()} (seed={seed})'
        ) and success
    
    return success


def run_all_experiments(skip_download=False, skip_enrichment=False):
    """Run pipeline for all seeds and both assays."""
    start_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("RUNNING FULL EXPERIMENTAL SUITE")
    print("=" * 60)
    print(f"\nStart time: {start_time}")
    print(f"Seeds: {SEEDS}")
    print(f"Assays: {[e['name'] for e in EXPERIMENTS]}")
    
    total_experiments = len(SEEDS) * len(EXPERIMENTS)
    completed = 0
    failed = []
    
    # Download data once
    if not skip_download:
        run_step(
            '01_download_data.py',
            ['--data-dir', DATA_DIR],
            'Download Data (once)'
        )
    
    # Run all experiments
    for experiment in EXPERIMENTS:
        assay = experiment['name']
        
        for seed in SEEDS:
            print(f"\n{'#'*60}")
            print(f"# Experiment {completed + 1}/{total_experiments}: {assay.upper()} seed={seed}")
            print(f"{'#'*60}")
            
            try:
                success = run_single_experiment(
                    assay=assay,
                    seed=seed,
                    skip_download=True,  # Already downloaded
                    skip_enrichment=skip_enrichment,
                )
                
                if success:
                    completed += 1
                else:
                    failed.append((assay, seed))
                    
            except Exception as e:
                print(f"Error in {assay} seed={seed}: {e}")
                failed.append((assay, seed))
    
    # Aggregate results
    run_step(
        '05_aggregate_results.py',
        ['--data-dir', RESULTS_DIR],
        'Aggregate Results'
    )
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 60)
    print(f"End time: {end_time}")
    print(f"Duration: {duration}")
    print(f"Completed: {completed}/{total_experiments}")
    
    if failed:
        print(f"\nFailed experiments:")
        for assay, seed in failed:
            print(f"  - {assay} seed={seed}")


def main():
    parser = argparse.ArgumentParser(
        description='NASA RNA-seq Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python scripts/run_all.py --assay hne --seed 42
  
  # Run all experiments
  python scripts/run_all.py --run-all
  
  # Show configuration
  python scripts/run_all.py --show-config
        """
    )
    
    parser.add_argument(
        '--assay',
        type=str,
        choices=['hne', 'tunel'],
        help='Assay type for single experiment'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for single experiment'
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all experiments (all seeds, both assays)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step'
    )
    parser.add_argument(
        '--skip-enrichment',
        action='store_true',
        help='Skip enrichment analysis step'
    )
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Show configuration
    if args.show_config:
        print_config()
        return
    
    # Validate arguments
    if not args.run_all and not args.assay:
        parser.error("Either --assay or --run-all must be specified")
    
    # Print header
    print("\n" + "=" * 60)
    print("NASA RNA-seq Analysis Pipeline")
    print("=" * 60)
    print_config()
    
    # Run experiments
    if args.run_all:
        run_all_experiments(
            skip_download=args.skip_download,
            skip_enrichment=args.skip_enrichment,
        )
    else:
        run_single_experiment(
            assay=args.assay,
            seed=args.seed,
            skip_download=args.skip_download,
            skip_enrichment=args.skip_enrichment,
        )
    
    print("\nPipeline complete!")


if __name__ == '__main__':
    main()
