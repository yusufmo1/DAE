"""
KNN imputation pipeline for pharmaceutical formulation data.
Provides classical ML baseline for comparison with DAE approach.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import argparse
from typing import Dict, List
from itertools import product
import json
from tqdm import tqdm
from joblib import Parallel, delayed

from data_preprocessing import FormulationDataPreprocessor
from knn_imputation import create_knn_imputer
from knn_evaluate import (
    evaluate_knn_imputation,
    aggregate_knn_results,
    save_knn_metrics,
    save_knn_predictions,
    compare_knn_configs,
    print_knn_summary
)


class KNNConfig:
    """Configuration for KNN experiments."""
    # Data parameters (same as DAE)
    DATA_PATH = 'data/material_name_smilesRemoved.csv'
    METADATA_COLS = 6

    # Missingness rates (same as DAE)
    MISSINGNESS_RATES = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
    SEEDS = [42, 50, 100]

    # KNN-specific parameters
    N_NEIGHBORS = [3, 5, 10, 20, 50]
    WEIGHTS = ['uniform', 'distance']
    METRICS = ['euclidean', 'manhattan', 'cosine']

    # Parallel processing for NearestNeighbors
    NN_N_JOBS = -1  # Use all CPU cores for neighbor search

    # Parallel processing for experiment queue
    EXPERIMENT_N_JOBS = 12  # Run 12 experiments in parallel

    # Output directories
    RESULTS_DIR = 'results/knn'
    METRICS_DIR = 'results/knn/metrics'
    PREDICTIONS_DIR = 'results/knn/predictions'
    PLOTS_DIR = 'results/knn/plots'


def run_single_knn_experiment(
    preprocessor: FormulationDataPreprocessor,
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
        missingness_rate: Proportion of data to mask
        n_neighbors: Number of neighbors for KNN
        weights: Weighting scheme ('uniform' or 'distance')
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        seeds: List of random seeds
        save_results: Whether to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with aggregated results
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
        n_jobs=KNNConfig.NN_N_JOBS,
        verbose=verbose
    )

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nSeed {seed_idx+1}/{len(seeds)}: {seed}")

        # Prepare corrupted data (same as DAE)
        original_data, corrupted_data, mask = preprocessor.prepare_data(
            missingness_rate=missingness_rate,
            noise_std=0.1,  # Same noise as DAE
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
            KNNConfig.METRICS_DIR,
            f'{config_str}_metrics.json'
        )
        save_knn_metrics(aggregated, metrics_file)

        # Save predictions from first seed
        pred_file = os.path.join(
            KNNConfig.PREDICTIONS_DIR,
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
    seeds: List[int]
) -> tuple:
    """
    Wrapper function for parallel execution of single experiment.

    Args:
        args: Tuple of (miss_rate, k, weights, metric)
        preprocessor: Data preprocessor
        seeds: List of seeds

    Returns:
        Tuple of (config_key, result_dict)
    """
    miss_rate, k, weights, metric = args

    result = run_single_knn_experiment(
        preprocessor=preprocessor,
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


def run_all_knn_experiments(
    missingness_rates: List[float] = None,
    n_neighbors_list: List[int] = None,
    weights_list: List[str] = None,
    metrics_list: List[str] = None,
    seeds: List[int] = None,
    quick_test: bool = False,
    parallel: bool = True
) -> Dict:
    """
    Run all KNN experiments.

    Args:
        missingness_rates: List of missingness rates
        n_neighbors_list: List of K values
        weights_list: List of weighting schemes
        metrics_list: List of distance metrics
        seeds: List of random seeds
        quick_test: If True, run reduced experiments
        parallel: If True, run experiments in parallel using joblib

    Returns:
        Dictionary with all results
    """
    # Use defaults if not specified
    if missingness_rates is None:
        missingness_rates = KNNConfig.MISSINGNESS_RATES
    if n_neighbors_list is None:
        n_neighbors_list = KNNConfig.N_NEIGHBORS
    if weights_list is None:
        weights_list = KNNConfig.WEIGHTS
    if metrics_list is None:
        metrics_list = KNNConfig.METRICS
    if seeds is None:
        seeds = KNNConfig.SEEDS

    # Quick test mode
    if quick_test:
        print("\n*** QUICK TEST MODE - Running reduced experiments ***\n")
        missingness_rates = [0.01]
        n_neighbors_list = [5]
        weights_list = ['distance']
        metrics_list = ['euclidean']
        seeds = [42]

    # Setup
    print("="*80)
    print("KNN IMPUTATION FOR PHARMACEUTICAL FORMULATION DATA")
    print("="*80)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = FormulationDataPreprocessor(
        data_path=KNNConfig.DATA_PATH,
        metadata_cols=KNNConfig.METADATA_COLS
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
    os.makedirs(KNNConfig.RESULTS_DIR, exist_ok=True)
    os.makedirs(KNNConfig.METRICS_DIR, exist_ok=True)
    os.makedirs(KNNConfig.PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(KNNConfig.PLOTS_DIR, exist_ok=True)

    # Run all experiments
    all_results = {}

    # Generate all experiment configurations
    experiment_configs = list(product(
        missingness_rates, n_neighbors_list, weights_list, metrics_list
    ))

    if parallel and len(experiment_configs) > 1:
        print(f"\nRunning {len(experiment_configs)} experiments in parallel (n_jobs={KNNConfig.EXPERIMENT_N_JOBS})...")

        # Run experiments in parallel
        results = Parallel(n_jobs=KNNConfig.EXPERIMENT_N_JOBS)(
            delayed(_run_experiment_wrapper)(config, preprocessor, seeds)
            for config in tqdm(experiment_configs, desc="Experiments")
        )

        # Convert list of tuples to dictionary
        all_results = dict(results)
    else:
        # Sequential execution (for quick-test or debugging)
        print(f"\nRunning {len(experiment_configs)} experiments sequentially...")
        for experiment_count, (miss_rate, k, weights, metric) in enumerate(experiment_configs, 1):
            print(f"\n[Experiment {experiment_count}/{total_experiments}]")

            result = run_single_knn_experiment(
                preprocessor=preprocessor,
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
    summary_file = os.path.join(KNNConfig.RESULTS_DIR, 'summary.json')
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

        for rank, (config, r2) in enumerate(rankings, 1):
            print(f"  {rank}. {config}: R² = {r2:.4f}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run KNN imputation experiments for pharmaceutical formulation data'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced parameters'
    )
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

    # Run experiments
    all_results = run_all_knn_experiments(
        missingness_rates=args.missingness,
        n_neighbors_list=args.k_neighbors,
        weights_list=args.weights,
        metrics_list=args.metrics,
        seeds=args.seeds,
        quick_test=args.quick_test
    )

    print("\n" + "="*80)
    print("ALL KNN EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {KNNConfig.RESULTS_DIR}")
    print("\nTo compare with DAE results, run:")
    print("  python -c \"from src.compare_methods import generate_all_comparisons; generate_all_comparisons()\"")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
