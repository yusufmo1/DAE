"""
KNN experiment orchestration logic.
Handles running KNN experiments across multiple configurations and seeds, with parallel execution support.
"""

import numpy as np
import os
import json
from typing import Dict, List
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed

from ..common.data_preprocessing import FormulationDataPreprocessor
from ..config import KNNConfig
from .imputation import create_knn_imputer
from .evaluate import (
    evaluate_knn_imputation,
    aggregate_knn_results,
    save_knn_metrics,
    save_knn_predictions,
    compare_knn_configs,
    print_knn_summary
)
from .plots import generate_all_knn_plots


def run_single_experiment(
    preprocessor: FormulationDataPreprocessor,
    config: KNNConfig,
    missingness_rate: float,
    n_neighbors: int,
    weights: str,
    metric: str,
    seeds: List[int],
    save_results: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run a single KNN configuration across multiple seeds.

    Args:
        preprocessor: Data preprocessor with loaded data
        config: KNN configuration object
        missingness_rate: Proportion of data to mask
        n_neighbors: Number of neighbors for KNN
        weights: Weighting scheme ('uniform' or 'distance')
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        seeds: List of random seeds
        save_results: Whether to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with aggregated results across seeds
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: Miss={missingness_rate*100:.0f}%, K={n_neighbors}, "
              f"Weights={weights}, Metric={metric}")
        print(f"{'='*80}")

    results_list = []
    predictions_list = []
    targets_list = []

    # Create KNN imputer
    knn_imputer = create_knn_imputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=config.NN_N_JOBS,
        verbose=verbose
    )

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nSeed {seed_idx+1}/{len(seeds)}: {seed}")

        # Prepare corrupted data (same as DAE)
        original_data, corrupted_data, mask = preprocessor.prepare_data(
            missingness_rate=missingness_rate,
            noise_std=config.NOISE_STD,
            seed=seed
        )

        # Convert to numpy (KNN works with numpy, not torch)
        original_np = original_data.numpy()
        corrupted_np = corrupted_data.numpy()
        mask_np = mask.numpy()

        # Run KNN imputation
        imputed_data, metrics = knn_imputer.impute(
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
    aggregated = aggregate_knn_results(results_list, seeds)

    # Save results
    if save_results:
        # Save aggregated metrics
        config_str = f"miss{missingness_rate}_k{n_neighbors}_{weights}_{metric}"
        metrics_file = os.path.join(
            config.METRICS_DIR,
            f'{config_str}_metrics.json'
        )
        save_knn_metrics(aggregated, metrics_file)

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


def _run_experiment_wrapper(
    args: tuple,
    preprocessor: FormulationDataPreprocessor,
    config: KNNConfig,
    seeds: List[int]
) -> tuple:
    """
    Wrapper function for parallel execution of single experiment.

    Args:
        args: Tuple of (miss_rate, k, weights, metric)
        preprocessor: Data preprocessor
        config: KNN configuration
        seeds: List of seeds

    Returns:
        Tuple of (config_key, result_dict)
    """
    miss_rate, k, weights, metric = args

    result = run_single_experiment(
        preprocessor=preprocessor,
        config=config,
        missingness_rate=miss_rate,
        n_neighbors=k,
        weights=weights,
        metric=metric,
        seeds=seeds,
        save_results=True,
        verbose=False  # Disable verbose in parallel mode
    )

    key = f"miss{miss_rate}_k{k}_{weights}_{metric}"
    return (key, result)


def run_all_experiments(
    config: KNNConfig,
    missingness_rates: List[float] = None,
    n_neighbors_list: List[int] = None,
    weights_list: List[str] = None,
    metrics_list: List[str] = None,
    seeds: List[int] = None,
    parallel: bool = True
) -> Dict:
    """
    Run all KNN experiments.

    Args:
        config: KNN configuration object
        missingness_rates: List of missingness rates (uses config default if None)
        n_neighbors_list: List of K values (uses config default if None)
        weights_list: List of weighting schemes (uses config default if None)
        metrics_list: List of distance metrics (uses config default if None)
        seeds: List of random seeds (uses config default if None)
        parallel: If True, run experiments in parallel using joblib

    Returns:
        Dictionary with all results
    """
    # Use defaults from config if not specified
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES
    if n_neighbors_list is None:
        n_neighbors_list = config.N_NEIGHBORS
    if weights_list is None:
        weights_list = config.WEIGHTS
    if metrics_list is None:
        metrics_list = config.METRICS
    if seeds is None:
        seeds = config.SEEDS

    # Setup
    print("="*80)
    print("KNN IMPUTATION FOR PHARMACEUTICAL FORMULATION DATA")
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
    total_experiments = (len(missingness_rates) * len(n_neighbors_list) *
                        len(weights_list) * len(metrics_list))
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"Seeds per experiment: {len(seeds)}")
    print(f"Total imputation runs: {total_experiments * len(seeds)}")
    print(f"\n⚡ KNN is fast - no training required! Each run takes seconds.")

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Run all experiments
    all_results = {}

    # Generate all experiment configurations
    experiment_configs = list(product(
        missingness_rates, n_neighbors_list, weights_list, metrics_list
    ))

    if parallel and len(experiment_configs) > 1:
        print(f"\nRunning {len(experiment_configs)} experiments in parallel (n_jobs={config.EXPERIMENT_N_JOBS})...")

        # Run experiments in parallel
        results = Parallel(n_jobs=config.EXPERIMENT_N_JOBS)(
            delayed(_run_experiment_wrapper)(exp_config, preprocessor, config, seeds)
            for exp_config in tqdm(experiment_configs, desc="Experiments")
        )

        # Convert list of tuples to dictionary
        all_results = dict(results)
    else:
        # Sequential execution (for quick-test or debugging)
        print(f"\nRunning {len(experiment_configs)} experiments sequentially...")
        for experiment_count, (miss_rate, k, weights, metric) in enumerate(experiment_configs, 1):
            print(f"\n[Experiment {experiment_count}/{total_experiments}]")

            result = run_single_experiment(
                preprocessor=preprocessor,
                config=config,
                missingness_rate=miss_rate,
                n_neighbors=k,
                weights=weights,
                metric=metric,
                seeds=seeds,
                save_results=True,
                verbose=True
            )

            # Store result
            key = f"miss{miss_rate}_k{k}_{weights}_{metric}"
            all_results[key] = result

    # Save summary
    summary_file = os.path.join(config.RESULTS_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # Print best configurations for each missingness rate
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY MISSINGNESS RATE")
    print("="*80)

    for miss_rate in missingness_rates:
        print(f"\n{miss_rate*100:.0f}% Missingness:")
        miss_results = {k: v for k, v in all_results.items() if f"miss{miss_rate}" in k}
        rankings = compare_knn_configs(miss_results, metric='r2', top_k=3)

        for rank, (config_name, r2) in enumerate(rankings, 1):
            print(f"  {rank}. {config_name}: R² = {r2:.4f}")

    return all_results


def generate_plots(config: KNNConfig, missingness_rates: List[float] = None):
    """
    Generate all KNN visualization plots.

    Args:
        config: KNN configuration object
        missingness_rates: List of missingness rates to plot (uses config default if None)
    """
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES

    print("\nGenerating KNN plots...")
    generate_all_knn_plots(config.RESULTS_DIR, missingness_rates)
    print(f"Plots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    print("KNN experiment orchestration module.")
    print("Use run_all_experiments() to execute the full KNN experimental pipeline.")
