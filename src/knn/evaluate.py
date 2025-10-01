"""
Evaluation utilities for KNN imputation.
Provides functions to evaluate and aggregate KNN imputation results.
"""

import numpy as np
import json
import os
from typing import Dict, List
from .imputation import KNNImputer


def evaluate_knn_imputation(
    knn_imputer: KNNImputer,
    original_data: np.ndarray,
    corrupted_data: np.ndarray,
    mask: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate KNN imputation performance.

    Args:
        knn_imputer: Configured KNN imputer
        original_data: Clean data (ground truth)
        corrupted_data: Data with masked values set to zero
        mask: Boolean mask (True = was masked/missing)
        verbose: Whether to print metrics

    Returns:
        Dictionary with evaluation metrics
    """
    # Perform imputation
    imputed_data, metrics = knn_imputer.impute(
        original_data,
        corrupted_data,
        mask
    )

    if verbose:
        print("\nKNN Imputation Metrics (on masked values only):")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Number of masked values: {metrics['n_masked']:,}")
        print(f"  Imputation time: {metrics['imputation_time']:.3f}s")

    return metrics


def aggregate_knn_results(results_list: List[Dict], seeds: List[int]) -> Dict:
    """
    Aggregate results from multiple runs (different seeds).

    Args:
        results_list: List of metric dictionaries
        seeds: List of seeds used

    Returns:
        Dictionary with mean, std, and individual results
    """
    aggregated = {
        'seeds': seeds,
        'n_runs': len(results_list),
    }

    # Compute statistics for each metric
    for metric in ['r2', 'rmse', 'mse', 'mae', 'imputation_time']:
        values = [r[metric] for r in results_list]
        aggregated[f'{metric}_mean'] = float(np.mean(values))
        aggregated[f'{metric}_std'] = float(np.std(values, ddof=1))
        aggregated[f'{metric}_values'] = [float(v) for v in values]

    # Keep configuration info from first result
    if results_list:
        aggregated['n_neighbors'] = results_list[0]['n_neighbors']
        aggregated['weights'] = results_list[0]['weights']
        aggregated['metric'] = results_list[0]['metric']

    return aggregated


def save_knn_metrics(metrics: Dict, save_path: str):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {save_path}")


def save_knn_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    save_path: str
):
    """
    Save predictions and targets for masked values.

    Args:
        predictions: KNN predictions
        targets: Ground truth values
        mask: Boolean mask (True = was masked/missing)
        save_path: Path to save data
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Extract only masked values
    pred_masked = predictions[mask]
    true_masked = targets[mask]

    # Save as numpy arrays
    np.savez(
        save_path,
        predictions=pred_masked,
        targets=true_masked
    )

    print(f"Predictions saved to {save_path}")


def load_knn_predictions(load_path: str):
    """
    Load saved predictions and targets.

    Args:
        load_path: Path to load from

    Returns:
        Tuple of (predictions, targets)
    """
    data = np.load(load_path)
    return data['predictions'], data['targets']


def compare_knn_configs(
    results: Dict[str, Dict],
    metric: str = 'r2',
    top_k: int = 10
) -> List[tuple]:
    """
    Compare different KNN configurations and rank by performance.

    Args:
        results: Dictionary mapping config names to metric dicts
        metric: Metric to rank by (default: 'r2')
        top_k: Number of top configurations to return

    Returns:
        List of (config_name, metric_value) tuples, sorted by performance
    """
    rankings = []

    for config_name, metrics in results.items():
        if f'{metric}_mean' in metrics:
            value = metrics[f'{metric}_mean']
            rankings.append((config_name, value))

    # Sort by metric value (descending for R², ascending for RMSE)
    reverse = (metric in ['r2'])
    rankings.sort(key=lambda x: x[1], reverse=reverse)

    return rankings[:top_k]


def print_knn_summary(
    aggregated_results: Dict,
    config_name: str = "KNN"
):
    """
    Print summary of KNN imputation results.

    Args:
        aggregated_results: Aggregated metrics dictionary
        config_name: Name of configuration
    """
    print("\n" + "="*80)
    print(f"{config_name} IMPUTATION SUMMARY")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  K neighbors: {aggregated_results.get('n_neighbors', 'N/A')}")
    print(f"  Weights: {aggregated_results.get('weights', 'N/A')}")
    print(f"  Metric: {aggregated_results.get('metric', 'N/A')}")
    print(f"  Number of runs: {aggregated_results['n_runs']}")
    print(f"  Seeds: {aggregated_results['seeds']}")

    print(f"\nPerformance (mean ± std):")
    print(f"  R² Score: {aggregated_results['r2_mean']:.4f} ± {aggregated_results['r2_std']:.4f}")
    print(f"  RMSE: {aggregated_results['rmse_mean']:.4f} ± {aggregated_results['rmse_std']:.4f}")
    print(f"  MSE: {aggregated_results['mse_mean']:.6f} ± {aggregated_results['mse_std']:.6f}")
    print(f"  MAE: {aggregated_results['mae_mean']:.4f} ± {aggregated_results['mae_std']:.4f}")

    print(f"\nComputational Cost:")
    print(f"  Imputation time: {aggregated_results['imputation_time_mean']:.3f}s ± {aggregated_results['imputation_time_std']:.3f}s")

    print("="*80)


if __name__ == "__main__":
    print("This module provides KNN evaluation utilities.")
    print("Use knn_main.py to run KNN imputation experiments.")
