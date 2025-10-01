#!/usr/bin/env python3
"""
Baseline Imputation Experiments Entry Point.
Run baseline imputation methods (zero imputation, etc.) for pharmaceutical formulation data.
"""

import argparse

from src.config import BaselineConfig, get_config
from src.baselines.experiments import run_all_experiments, generate_plots


def main():
    """Main entry point for baseline experiments."""
    parser = argparse.ArgumentParser(
        description='Run baseline imputation experiments for pharmaceutical formulation data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment control
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced parameters (~1 min)'
    )

    # Hyperparameter overrides
    parser.add_argument(
        '--missingness',
        type=float,
        nargs='+',
        default=None,
        help='Missingness rates to test (e.g., 0.01 0.05 0.10)'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Random seeds to use (e.g., 42 50 100)'
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config('baseline', quick_test=args.quick_test)

    # Run experiments
    print("\n" + "="*80)
    print("RUNNING BASELINE EXPERIMENTS")
    print("="*80 + "\n")

    all_results = run_all_experiments(
        config=config,
        missingness_rates=args.missingness,
        seeds=args.seeds
    )

    print("\n" + "="*80)
    print("ALL BASELINE EXPERIMENTS COMPLETED!")
    print("="*80)

    # Generate plots
    missingness_rates = args.missingness if args.missingness else config.MISSINGNESS_RATES
    generate_plots(config, missingness_rates)

    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nTo compare with DAE and KNN results, run:")
    print("  python run_comparison.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
