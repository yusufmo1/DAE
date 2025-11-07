#!/usr/bin/env python3
"""
Run All Experiments Entry Point.
Orchestrates DAE, KNN, MissForest, and baseline experiments, then generates comparisons.
"""

import argparse

from src.config import DAEConfig, KNNConfig, MissForestConfig, BaselineConfig, ComparisonConfig, get_config
from src.dae.experiments import run_all_experiments as run_dae_experiments
from src.dae.experiments import generate_plots as generate_dae_plots
from src.knn.experiments import run_all_experiments as run_knn_experiments
from src.knn.experiments import generate_plots as generate_knn_plots
from src.missforest.experiments import run_all_experiments as run_missforest_experiments
from src.missforest.experiments import generate_plots as generate_missforest_plots
from src.baselines.experiments import run_all_experiments as run_baseline_experiments
from src.baselines.experiments import generate_plots as generate_baseline_plots
from src.comparison.plots import generate_all_comparisons


def main():
    """Main entry point for running all experiments."""
    parser = argparse.ArgumentParser(
        description='Run all imputation experiments (DAE, KNN, MissForest, baselines) and generate comparisons',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick tests with reduced parameters (~10 min total)'
    )
    parser.add_argument(
        '--skip-dae',
        action='store_true',
        help='Skip DAE experiments (useful if already run)'
    )
    parser.add_argument(
        '--skip-knn',
        action='store_true',
        help='Skip KNN experiments (useful if already run)'
    )
    parser.add_argument(
        '--skip-missforest',
        action='store_true',
        help='Skip MissForest experiments (useful if already run)'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline experiments (useful if already run)'
    )
    parser.add_argument(
        '--comparison-only',
        action='store_true',
        help='Skip all experiments and only generate comparisons'
    )

    args = parser.parse_args()

    # Get configurations
    dae_config = get_config('dae', quick_test=args.quick_test)
    knn_config = get_config('knn', quick_test=args.quick_test)
    missforest_config = get_config('missforest', quick_test=args.quick_test)
    baseline_config = get_config('baseline', quick_test=args.quick_test)
    comparison_config = ComparisonConfig()

    print("\n" + "="*80)
    print("RUNNING ALL IMPUTATION EXPERIMENTS")
    print("="*80)
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***")
        print("Running reduced experiments for fast testing")
    print("\n")

    # Run DAE experiments
    if not args.skip_dae and not args.comparison_only:
        print("\n" + "="*80)
        print("STEP 1/4: DAE EXPERIMENTS")
        print("="*80 + "\n")

        run_dae_experiments(config=dae_config)
        generate_dae_plots(dae_config)

        print("\n✓ DAE experiments complete")

    # Run KNN experiments
    if not args.skip_knn and not args.comparison_only:
        print("\n" + "="*80)
        print("STEP 2/4: KNN EXPERIMENTS")
        print("="*80 + "\n")

        run_knn_experiments(config=knn_config, parallel=True)
        generate_knn_plots(knn_config)

        print("\n✓ KNN experiments complete")

    # Run MissForest experiments
    if not args.skip_missforest and not args.comparison_only:
        print("\n" + "="*80)
        print("STEP 3/4: MISSFOREST EXPERIMENTS")
        print("="*80 + "\n")

        run_missforest_experiments(config=missforest_config)
        generate_missforest_plots(missforest_config)

        print("\n✓ MissForest experiments complete")

    # Run baseline experiments
    if not args.skip_baseline and not args.comparison_only:
        print("\n" + "="*80)
        print("STEP 4/4: BASELINE EXPERIMENTS")
        print("="*80 + "\n")

        run_baseline_experiments(config=baseline_config)
        generate_baseline_plots(baseline_config)

        print("\n✓ Baseline experiments complete")

    # Generate comparisons
    print("\n" + "="*80)
    print("GENERATING METHOD COMPARISONS")
    print("="*80 + "\n")

    generate_all_comparisons(
        dae_results_dir=dae_config.RESULTS_DIR,
        knn_results_dir=knn_config.RESULTS_DIR,
        missforest_results_dir=missforest_config.RESULTS_DIR,
        zero_results_dir=baseline_config.RESULTS_DIR,
        output_dir=comparison_config.OUTPUT_DIR
    )

    print("\n✓ Comparisons complete")

    # Final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print(f"  - DAE: {dae_config.RESULTS_DIR}")
    print(f"  - KNN: {knn_config.RESULTS_DIR}")
    print(f"  - MissForest: {missforest_config.RESULTS_DIR}")
    print(f"  - Baseline: {baseline_config.RESULTS_DIR}")
    print(f"  - Comparisons: {comparison_config.OUTPUT_DIR}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
