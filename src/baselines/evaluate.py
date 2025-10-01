"""
Evaluation utilities for baseline imputation methods.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple


def evaluate_imputation(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate imputation quality using standard metrics.

    Args:
        predictions: Imputed values
        targets: Ground truth values

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'r2': r2_score(targets, predictions),
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions)
    }

    return metrics


def aggregate_results(
    results_list: list,
    keys: list = ['r2', 'rmse', 'mae', 'imputation_time']
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple runs.

    Args:
        results_list: List of result dictionaries
        keys: Metrics to aggregate

    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}

    for key in keys:
        values = [r[key] for r in results_list if key in r]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    return aggregated
