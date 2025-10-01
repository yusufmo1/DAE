#!/usr/bin/env python3
"""
DAE Imputation Experiments Entry Point.
Run Denoising Autoencoder experiments for pharmaceutical formulation data imputation.
"""

import argparse

from src.config import DAEConfig, get_config
from src.dae.experiments import run_all_experiments, generate_plots


def main():
    """Main entry point for DAE experiments."""
    parser = argparse.ArgumentParser(
        description='Run DAE experiments for pharmaceutical formulation data imputation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment control
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced parameters (~5 min)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only generate plots from existing results'
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
        '--learning-rates',
        type=float,
        nargs='+',
        default=None,
        help='Learning rates to test (e.g., 0.1 0.001 0.00001)'
    )
    parser.add_argument(
        '--neuron-sizes',
        type=int,
        nargs='+',
        default=None,
        help='Neuron sizes to test (e.g., 256 512 1024)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        default=None,
        help='Epoch settings to test (e.g., 100 500 1000 1200)'
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
    config = get_config('dae', quick_test=args.quick_test)

    # Run experiments
    if not args.skip_training:
        print("\n" + "="*80)
        print("RUNNING DAE EXPERIMENTS")
        print("="*80 + "\n")

        all_results = run_all_experiments(
            config=config,
            missingness_rates=args.missingness,
            learning_rates=args.learning_rates,
            neuron_sizes=args.neuron_sizes,
            epoch_settings=args.epochs,
            seeds=args.seeds
        )

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED!")
        print("="*80)

    # Generate plots
    missingness_rates = args.missingness if args.missingness else config.MISSINGNESS_RATES
    generate_plots(config, missingness_rates)

    print("\n" + "="*80)
    print("PIPELINE COMPLETED!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
