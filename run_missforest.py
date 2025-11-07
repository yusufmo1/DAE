#!/usr/bin/env python3
"""
MissForest Imputation Experiments Entry Point.
Run MissForest imputation experiments for pharmaceutical formulation data.
"""

import argparse

from src.config import MissForestConfig, get_config
from src.missforest.experiments import run_all_experiments, generate_plots


def main():
    """Main entry point for MissForest experiments."""
    parser = argparse.ArgumentParser(
        description='Run MissForest imputation experiments for pharmaceutical formulation data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment control
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced parameters (~1-2 min)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip imputation, only generate plots from existing results'
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
        '--max-iter',
        type=int,
        default=None,
        help='Maximum number of imputation iterations (default: 10)'
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
    config = get_config('missforest', quick_test=args.quick_test)

    # Override max_iter if specified
    if args.max_iter:
        config.MAX_ITER = args.max_iter

    if not args.skip_training:
        # Run experiments
        print("\n" + "="*80)
        print("RUNNING MISSFOREST EXPERIMENTS")
        print("="*80 + "\n")

        all_results = run_all_experiments(
            config=config,
            missingness_rates=args.missingness,
            max_iter=config.MAX_ITER,
            seeds=args.seeds
        )

        print("\n" + "="*80)
        print("ALL MISSFOREST EXPERIMENTS COMPLETED!")
        print("="*80)
    else:
        print("\nSkipping imputation (--skip-training), generating plots only...")

    # Generate plots
    missingness_rates = args.missingness if args.missingness else config.MISSINGNESS_RATES
    generate_plots(config, missingness_rates)

    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nTo compare with DAE/KNN results, run:")
    print("  python run_comparison.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
