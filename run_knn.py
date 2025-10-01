#!/usr/bin/env python3
"""
KNN Imputation Experiments Entry Point.
Run K-Nearest Neighbors imputation experiments for pharmaceutical formulation data.
"""

import argparse

from src.config import KNNConfig, get_config
from src.knn.experiments import run_all_experiments, generate_plots


def main():
    """Main entry point for KNN experiments."""
    parser = argparse.ArgumentParser(
        description='Run KNN imputation experiments for pharmaceutical formulation data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment control
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced parameters (~1 min)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Run experiments sequentially (default: parallel)'
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
        '--k-neighbors',
        type=int,
        nargs='+',
        default=None,
        help='K values to test (e.g., 3 5 10 20 50)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        nargs='+',
        default=None,
        choices=['uniform', 'distance'],
        help='Weighting schemes to test'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        choices=['euclidean', 'manhattan', 'cosine'],
        help='Distance metrics to test'
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
    config = get_config('knn', quick_test=args.quick_test)

    # Run experiments
    print("\n" + "="*80)
    print("RUNNING KNN EXPERIMENTS")
    print("="*80 + "\n")

    all_results = run_all_experiments(
        config=config,
        missingness_rates=args.missingness,
        n_neighbors_list=args.k_neighbors,
        weights_list=args.weights,
        metrics_list=args.metrics,
        seeds=args.seeds,
        parallel=not args.no_parallel
    )

    print("\n" + "="*80)
    print("ALL KNN EXPERIMENTS COMPLETED!")
    print("="*80)

    # Generate plots
    missingness_rates = args.missingness if args.missingness else config.MISSINGNESS_RATES
    generate_plots(config, missingness_rates)

    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nTo compare with DAE results, run:")
    print("  python run_comparison.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
