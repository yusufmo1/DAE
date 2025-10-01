#!/usr/bin/env python3
"""
Method Comparison Entry Point.
Generate comparison plots and tables across DAE, KNN, and baseline methods.
"""

import argparse
import os

from src.config import ComparisonConfig
from src.comparison.plots import generate_all_comparisons


def main():
    """Main entry point for generating method comparisons."""
    parser = argparse.ArgumentParser(
        description='Generate comparison plots and tables for DAE, KNN, and baseline methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--dae-results',
        type=str,
        default=None,
        help='Path to DAE results directory (default: results/dae)'
    )
    parser.add_argument(
        '--knn-results',
        type=str,
        default=None,
        help='Path to KNN results directory (default: results/knn)'
    )
    parser.add_argument(
        '--baseline-results',
        type=str,
        default=None,
        help='Path to baseline results directory (default: results/baselines)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for comparisons (default: results/comparisons)'
    )

    args = parser.parse_args()

    # Get configuration
    config = ComparisonConfig()

    # Use command-line arguments or defaults
    dae_dir = args.dae_results or config.DAE_RESULTS_DIR
    knn_dir = args.knn_results or config.KNN_RESULTS_DIR
    baseline_dir = args.baseline_results or config.BASELINE_RESULTS_DIR
    output_dir = args.output or config.OUTPUT_DIR

    # Check if result directories exist
    missing_dirs = []
    if not os.path.exists(os.path.join(dae_dir, 'summary.json')):
        missing_dirs.append(f"DAE results ({dae_dir})")
    if not os.path.exists(os.path.join(knn_dir, 'summary.json')):
        missing_dirs.append(f"KNN results ({knn_dir})")
    if not os.path.exists(os.path.join(baseline_dir, 'summary.json')):
        missing_dirs.append(f"Baseline results ({baseline_dir})")

    if missing_dirs:
        print("\n" + "="*80)
        print("WARNING: Missing experiment results!")
        print("="*80)
        print("\nThe following results were not found:")
        for missing in missing_dirs:
            print(f"  - {missing}")
        print("\nPlease run the following commands first:")
        if "DAE" in str(missing_dirs):
            print("  python run_dae.py")
        if "KNN" in str(missing_dirs):
            print("  python run_knn.py")
        if "Baseline" in str(missing_dirs):
            print("  python run_baseline.py")
        print("\n" + "="*80 + "\n")
        return

    # Generate comparisons
    print("\n" + "="*80)
    print("GENERATING METHOD COMPARISONS")
    print("="*80 + "\n")

    generate_all_comparisons(
        dae_results_dir=dae_dir,
        knn_results_dir=knn_dir,
        zero_results_dir=baseline_dir,
        output_dir=output_dir
    )

    print("\n" + "="*80)
    print("COMPARISONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - method_comparison.png")
    print("  - performance_vs_time.png")
    print("  - comparison_table.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
