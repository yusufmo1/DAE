"""
Baseline experiment orchestration logic.
Handles running baseline imputation experiments (zero imputation, etc.).
"""

import numpy as np
import os
import json
from typing import Dict, List

from ..common.data_preprocessing import FormulationDataPreprocessor
from ..config import BaselineConfig
from .zero_imputer import ZeroImputer, evaluate_zero_imputation
from .statistical_imputers import MeanImputer, MedianImputer, evaluate_statistical_imputation
from .plots import generate_baseline_plots


def run_zero_imputation_experiments(
    config: BaselineConfig,
    missingness_rates: List[float] = None,
    seeds: List[int] = None
) -> Dict:
    """
    Run zero imputation baseline experiments.

    Args:
        config: Baseline configuration object
        missingness_rates: List of missingness rates (uses config default if None)
        seeds: List of random seeds (uses config default if None)

    Returns:
        Dictionary of all results
    """
    # Use defaults from config if not specified
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES
    if seeds is None:
        seeds = config.SEEDS

    print("="*80)
    print("ZERO IMPUTATION BASELINE EXPERIMENTS")
    print("="*80)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = FormulationDataPreprocessor(
        data_path=config.DATA_PATH,
        metadata_cols=config.METADATA_COLS
    )
    preprocessor.load_data()
    preprocessor.normalize_data()

    # Print dataset statistics
    stats = preprocessor.get_data_statistics()
    print("\nDataset Statistics:")
    print(f"  Formulations: {stats['n_formulations']:,}")
    print(f"  Ingredients: {stats['n_ingredients']:,}")
    print(f"  Total data points: {stats['total_data_points']:,}")
    print(f"  Sparsity: {stats['sparsity_pct']:.2f}% zeros")

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Get data as numpy array (already numpy, not torch tensor)
    data = preprocessor.scaled_data

    all_results = {}

    for miss_rate in missingness_rates:
        print(f"\nTesting {miss_rate*100:.0f}% missingness...")

        seed_results = []

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')

            # Create corruption mask
            np.random.seed(seed)
            mask = np.random.rand(*data.shape) < miss_rate

            # Run evaluation
            predictions, targets, metrics, imp_time = evaluate_zero_imputation(
                data, mask, miss_rate, seed
            )

            seed_results.append(metrics)

            # Save predictions for first seed
            if seed == seeds[0]:
                pred_file = os.path.join(
                    config.PREDICTIONS_DIR,
                    f'miss{miss_rate}_seed{seed}_predictions.npz'
                )
                np.savez(pred_file, predictions=predictions, targets=targets)

            print(f"R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"Time={imp_time:.4f}s")

        # Aggregate results across seeds
        aggregated = {
            'r2_mean': np.mean([r['r2'] for r in seed_results]),
            'r2_std': np.std([r['r2'] for r in seed_results]),
            'rmse_mean': np.mean([r['rmse'] for r in seed_results]),
            'rmse_std': np.std([r['rmse'] for r in seed_results]),
            'mae_mean': np.mean([r['mae'] for r in seed_results]),
            'mae_std': np.std([r['mae'] for r in seed_results]),
            'imputation_time_mean': np.mean([r['imputation_time'] for r in seed_results]),
            'imputation_time_std': np.std([r['imputation_time'] for r in seed_results]),
            'missingness_rate': miss_rate,
            'num_seeds': len(seeds),
            'seeds': seeds
        }

        config_name = f'miss{miss_rate}_zero'
        all_results[config_name] = aggregated

        # Save aggregated metrics
        metrics_file = os.path.join(
            config.METRICS_DIR,
            f'miss{miss_rate}_zero_metrics.json'
        )
        with open(metrics_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"  Aggregated: R²={aggregated['r2_mean']:.4f}±{aggregated['r2_std']:.4f}, "
              f"RMSE={aggregated['rmse_mean']:.4f}±{aggregated['rmse_std']:.4f}")

    # Save summary
    summary_file = os.path.join(config.RESULTS_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ZERO IMPUTATION EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {config.RESULTS_DIR}/")
    print(f"  - summary.json")
    print(f"  - metrics/miss*_zero_metrics.json")
    print(f"  - predictions/miss*_seed*_predictions.npz")
    print("="*80)

    return all_results


def run_mean_imputation_experiments(
    config: BaselineConfig,
    missingness_rates: List[float] = None,
    seeds: List[int] = None
) -> Dict:
    """
    Run mean imputation baseline experiments.

    Args:
        config: Baseline configuration object
        missingness_rates: List of missingness rates (uses config default if None)
        seeds: List of random seeds (uses config default if None)

    Returns:
        Dictionary of all results
    """
    # Use defaults from config if not specified
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES
    if seeds is None:
        seeds = config.SEEDS

    print("="*80)
    print("MEAN IMPUTATION BASELINE EXPERIMENTS")
    print("="*80)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = FormulationDataPreprocessor(
        data_path=config.DATA_PATH,
        metadata_cols=config.METADATA_COLS
    )
    preprocessor.load_data()
    preprocessor.normalize_data()

    # Print dataset statistics
    stats = preprocessor.get_data_statistics()
    print("\nDataset Statistics:")
    print(f"  Formulations: {stats['n_formulations']:,}")
    print(f"  Ingredients: {stats['n_ingredients']:,}")
    print(f"  Total data points: {stats['total_data_points']:,}")
    print(f"  Sparsity: {stats['sparsity_pct']:.2f}% zeros")

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Get data as numpy array
    data = preprocessor.scaled_data

    all_results = {}

    for miss_rate in missingness_rates:
        print(f"\nTesting {miss_rate*100:.0f}% missingness...")

        seed_results = []

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')

            # Create corruption mask
            np.random.seed(seed)
            mask = np.random.rand(*data.shape) < miss_rate

            # Run evaluation
            predictions, targets, metrics, imp_time = evaluate_statistical_imputation(
                data, mask, miss_rate, method='mean', seed=seed
            )

            seed_results.append(metrics)

            # Save predictions for first seed
            if seed == seeds[0]:
                pred_file = os.path.join(
                    config.PREDICTIONS_DIR,
                    f'miss{miss_rate}_seed{seed}_mean_predictions.npz'
                )
                np.savez(pred_file, predictions=predictions, targets=targets)

            print(f"R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"Time={imp_time:.4f}s")

        # Aggregate results across seeds
        aggregated = {
            'r2_mean': np.mean([r['r2'] for r in seed_results]),
            'r2_std': np.std([r['r2'] for r in seed_results]),
            'rmse_mean': np.mean([r['rmse'] for r in seed_results]),
            'rmse_std': np.std([r['rmse'] for r in seed_results]),
            'mae_mean': np.mean([r['mae'] for r in seed_results]),
            'mae_std': np.std([r['mae'] for r in seed_results]),
            'imputation_time_mean': np.mean([r['imputation_time'] for r in seed_results]),
            'imputation_time_std': np.std([r['imputation_time'] for r in seed_results]),
            'missingness_rate': miss_rate,
            'num_seeds': len(seeds),
            'seeds': seeds
        }

        config_name = f'miss{miss_rate}_mean'
        all_results[config_name] = aggregated

        # Save aggregated metrics
        metrics_file = os.path.join(
            config.METRICS_DIR,
            f'miss{miss_rate}_mean_metrics.json'
        )
        with open(metrics_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"  Aggregated: R²={aggregated['r2_mean']:.4f}±{aggregated['r2_std']:.4f}, "
              f"RMSE={aggregated['rmse_mean']:.4f}±{aggregated['rmse_std']:.4f}")

    # Save summary
    summary_file = os.path.join(config.RESULTS_DIR, 'mean_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("MEAN IMPUTATION EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {config.RESULTS_DIR}/")
    print(f"  - mean_summary.json")
    print(f"  - metrics/miss*_mean_metrics.json")
    print(f"  - predictions/miss*_seed*_mean_predictions.npz")
    print("="*80)

    return all_results


def run_median_imputation_experiments(
    config: BaselineConfig,
    missingness_rates: List[float] = None,
    seeds: List[int] = None
) -> Dict:
    """
    Run median imputation baseline experiments.

    Args:
        config: Baseline configuration object
        missingness_rates: List of missingness rates (uses config default if None)
        seeds: List of random seeds (uses config default if None)

    Returns:
        Dictionary of all results
    """
    # Use defaults from config if not specified
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES
    if seeds is None:
        seeds = config.SEEDS

    print("="*80)
    print("MEDIAN IMPUTATION BASELINE EXPERIMENTS")
    print("="*80)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = FormulationDataPreprocessor(
        data_path=config.DATA_PATH,
        metadata_cols=config.METADATA_COLS
    )
    preprocessor.load_data()
    preprocessor.normalize_data()

    # Print dataset statistics
    stats = preprocessor.get_data_statistics()
    print("\nDataset Statistics:")
    print(f"  Formulations: {stats['n_formulations']:,}")
    print(f"  Ingredients: {stats['n_ingredients']:,}")
    print(f"  Total data points: {stats['total_data_points']:,}")
    print(f"  Sparsity: {stats['sparsity_pct']:.2f}% zeros")

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Get data as numpy array
    data = preprocessor.scaled_data

    all_results = {}

    for miss_rate in missingness_rates:
        print(f"\nTesting {miss_rate*100:.0f}% missingness...")

        seed_results = []

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ')

            # Create corruption mask
            np.random.seed(seed)
            mask = np.random.rand(*data.shape) < miss_rate

            # Run evaluation
            predictions, targets, metrics, imp_time = evaluate_statistical_imputation(
                data, mask, miss_rate, method='median', seed=seed
            )

            seed_results.append(metrics)

            # Save predictions for first seed
            if seed == seeds[0]:
                pred_file = os.path.join(
                    config.PREDICTIONS_DIR,
                    f'miss{miss_rate}_seed{seed}_median_predictions.npz'
                )
                np.savez(pred_file, predictions=predictions, targets=targets)

            print(f"R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"Time={imp_time:.4f}s")

        # Aggregate results across seeds
        aggregated = {
            'r2_mean': np.mean([r['r2'] for r in seed_results]),
            'r2_std': np.std([r['r2'] for r in seed_results]),
            'rmse_mean': np.mean([r['rmse'] for r in seed_results]),
            'rmse_std': np.std([r['rmse'] for r in seed_results]),
            'mae_mean': np.mean([r['mae'] for r in seed_results]),
            'mae_std': np.std([r['mae'] for r in seed_results]),
            'imputation_time_mean': np.mean([r['imputation_time'] for r in seed_results]),
            'imputation_time_std': np.std([r['imputation_time'] for r in seed_results]),
            'missingness_rate': miss_rate,
            'num_seeds': len(seeds),
            'seeds': seeds
        }

        config_name = f'miss{miss_rate}_median'
        all_results[config_name] = aggregated

        # Save aggregated metrics
        metrics_file = os.path.join(
            config.METRICS_DIR,
            f'miss{miss_rate}_median_metrics.json'
        )
        with open(metrics_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"  Aggregated: R²={aggregated['r2_mean']:.4f}±{aggregated['r2_std']:.4f}, "
              f"RMSE={aggregated['rmse_mean']:.4f}±{aggregated['rmse_std']:.4f}")

    # Save summary
    summary_file = os.path.join(config.RESULTS_DIR, 'median_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("MEDIAN IMPUTATION EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {config.RESULTS_DIR}/")
    print(f"  - median_summary.json")
    print(f"  - metrics/miss*_median_metrics.json")
    print(f"  - predictions/miss*_seed*_median_predictions.npz")
    print("="*80)

    return all_results


def generate_plots(config: BaselineConfig, missingness_rates: List[float] = None):
    """
    Generate all baseline visualization plots.

    Args:
        config: Baseline configuration object
        missingness_rates: List of missingness rates to plot (uses config default if None)
    """
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES

    print("\nGenerating baseline plots...")
    generate_baseline_plots(config.RESULTS_DIR, missingness_rates)
    print(f"Plots saved to {config.PLOTS_DIR}")


def run_all_experiments(config: BaselineConfig, **kwargs) -> Dict:
    """
    Run all baseline experiments (zero, mean, median imputation).

    Args:
        config: Baseline configuration object
        **kwargs: Additional arguments passed to specific baseline functions

    Returns:
        Dictionary with all baseline results merged
    """
    print("\n" + "="*80)
    print("RUNNING ALL BASELINE EXPERIMENTS")
    print("="*80)

    # Run all three baseline methods
    zero_results = run_zero_imputation_experiments(config, **kwargs)
    mean_results = run_mean_imputation_experiments(config, **kwargs)
    median_results = run_median_imputation_experiments(config, **kwargs)

    # Merge all results
    all_results = {}
    all_results.update(zero_results)
    all_results.update(mean_results)
    all_results.update(median_results)

    # Save unified summary
    summary_file = os.path.join(config.RESULTS_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ALL BASELINE EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Total configurations tested: {len(all_results)}")
    print(f"  - Zero imputation: {len(zero_results)}")
    print(f"  - Mean imputation: {len(mean_results)}")
    print(f"  - Median imputation: {len(median_results)}")
    print(f"\nUnified results saved to: {summary_file}")
    print("="*80)

    return all_results


if __name__ == "__main__":
    print("Baseline experiment orchestration module.")
    print("Use run_all_experiments() to execute baseline experiments.")
