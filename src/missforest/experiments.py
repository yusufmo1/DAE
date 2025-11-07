"""
MissForest experiment orchestration logic.
Handles running MissForest experiments across multiple missingness rates and seeds.
"""

import numpy as np
import os
import json
from typing import Dict, List
from tqdm import tqdm

from ..common.data_preprocessing import FormulationDataPreprocessor
from ..config import MissForestConfig
from .imputation import create_missforest_imputer
from .evaluate import (
    evaluate_missforest_imputation,
    aggregate_missforest_results,
    save_missforest_metrics,
    save_missforest_predictions,
    compare_missforest_configs,
    print_missforest_summary
)
from .plots import generate_all_missforest_plots


def run_single_experiment(
    preprocessor: FormulationDataPreprocessor,
    config: MissForestConfig,
    missingness_rate: float,
    max_iter: int,
    seeds: List[int],
    save_results: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run a single MissForest configuration across multiple seeds.

    Args:
        preprocessor: Data preprocessor with loaded data
        config: MissForest configuration object
        missingness_rate: Proportion of data to mask
        max_iter: Maximum number of imputation iterations
        seeds: List of random seeds
        save_results: Whether to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with aggregated results across seeds
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: Miss={missingness_rate*100:.0f}%, Max_Iter={max_iter}")
        print(f"{'='*80}")

    results_list = []
    predictions_list = []
    targets_list = []

    # Create MissForest imputer
    missforest_imputer = create_missforest_imputer(
        max_iter=max_iter,
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        n_jobs=config.N_JOBS,
        random_state=config.SEEDS[0],  # Use first seed for imputer
        verbose=verbose
    )

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nSeed {seed_idx+1}/{len(seeds)}: {seed}")

        # Prepare corrupted data (same as DAE and KNN)
        original_data, corrupted_data, mask = preprocessor.prepare_data(
            missingness_rate=missingness_rate,
            noise_std=config.NOISE_STD,
            seed=seed
        )

        # Convert to numpy (MissForest works with numpy, not torch)
        original_np = original_data.numpy()
        corrupted_np = corrupted_data.numpy()
        mask_np = mask.numpy()

        # Run MissForest imputation
        imputed_data, metrics = missforest_imputer.impute(
            original_np,
            corrupted_np,
            mask_np
        )

        if verbose:
            print(f"  R²: {metrics['r2']:.4f}, Time: {metrics['imputation_time']:.2f}s")

        results_list.append(metrics)
        predictions_list.append(imputed_data[mask_np])
        targets_list.append(original_np[mask_np])

    # Aggregate results across seeds
    aggregated = aggregate_missforest_results(results_list, seeds)

    # Save results
    if save_results:
        # Save aggregated metrics
        config_str = f"miss{missingness_rate}_mf_iter{max_iter}"
        metrics_file = os.path.join(
            config.METRICS_DIR,
            f'{config_str}_metrics.json'
        )
        save_missforest_metrics(aggregated, metrics_file)

        # Save predictions from first seed
        pred_file = os.path.join(
            config.PREDICTIONS_DIR,
            f'{config_str}_predictions.npz'
        )
        np.savez(pred_file, predictions=predictions_list[0], targets=targets_list[0])

    if verbose:
        print(f"\nAggregated Results (n={len(seeds)} runs):")
        print(f"  R² = {aggregated['r2_mean']:.4f} ± {aggregated['r2_std']:.4f}")
        print(f"  RMSE = {aggregated['rmse_mean']:.4f} ± {aggregated['rmse_std']:.4f}")
        print(f"  Time = {aggregated['imputation_time_mean']:.2f}s ± {aggregated['imputation_time_std']:.2f}s")

    return aggregated


def run_all_experiments(
    config: MissForestConfig,
    missingness_rates: List[float] = None,
    max_iter: int = None,
    seeds: List[int] = None
) -> Dict:
    """
    Run all MissForest experiments.

    Args:
        config: MissForest configuration object
        missingness_rates: List of missingness rates (uses config default if None)
        max_iter: Maximum iterations (uses config default if None)
        seeds: List of random seeds (uses config default if None)

    Returns:
        Dictionary with all results
    """
    # Use defaults from config if not specified
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES
    if max_iter is None:
        max_iter = config.MAX_ITER
    if seeds is None:
        seeds = config.SEEDS

    # Setup
    print("="*80)
    print("MISSFOREST IMPUTATION FOR PHARMACEUTICAL FORMULATION DATA")
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

    # Calculate total experiments
    total_experiments = len(missingness_rates)
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"Seeds per experiment: {len(seeds)}")
    print(f"Total imputation runs: {total_experiments * len(seeds)}")
    print(f"\n⚡ MissForest: Faster than DAE, more accurate than simple methods!")

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Run all experiments
    all_results = {}

    print(f"\nRunning {total_experiments} experiments sequentially...")
    for experiment_count, miss_rate in enumerate(missingness_rates, 1):
        print(f"\n[Experiment {experiment_count}/{total_experiments}]")

        result = run_single_experiment(
            preprocessor=preprocessor,
            config=config,
            missingness_rate=miss_rate,
            max_iter=max_iter,
            seeds=seeds,
            save_results=True,
            verbose=True
        )

        # Store result
        key = f"miss{miss_rate}_mf_iter{max_iter}"
        all_results[key] = result

    # Save summary
    summary_file = os.path.join(config.RESULTS_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Print results for each missingness rate
    print("\n" + "="*80)
    print("MISSFOREST RESULTS BY MISSINGNESS RATE")
    print("="*80)

    for miss_rate in missingness_rates:
        key = f"miss{miss_rate}_mf_iter{max_iter}"
        if key in all_results:
            result = all_results[key]
            print(f"\n{miss_rate*100:.0f}% Missingness:")
            print(f"  R² = {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
            print(f"  RMSE = {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
            print(f"  Time = {result['imputation_time_mean']:.2f}s ± {result['imputation_time_std']:.2f}s")

    return all_results


def generate_plots(config: MissForestConfig, missingness_rates: List[float] = None):
    """
    Generate all MissForest visualization plots.

    Args:
        config: MissForest configuration object
        missingness_rates: List of missingness rates to plot (uses config default if None)
    """
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES

    print("\nGenerating MissForest plots...")
    generate_all_missforest_plots(config.RESULTS_DIR, missingness_rates)
    print(f"Plots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    print("MissForest experiment orchestration module.")
    print("Use run_all_experiments() to execute the full MissForest experimental pipeline.")
