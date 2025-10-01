"""
Zero Imputation Baseline.
Provides the simplest possible baseline: filling missing values with zeros.
This serves as a lower bound to demonstrate improvement of more sophisticated methods.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict
import time


class ZeroImputer:
    """
    Baseline imputer that fills all missing values with zeros.

    This is the most naive imputation strategy and serves as a baseline
    to demonstrate the value of more sophisticated approaches (KNN, DAE).

    Expected behavior:
    - For sparse pharmaceutical data (~99% zeros), this may perform reasonably
      when imputing unused ingredients
    - Will perform poorly when imputing non-zero missing values
    - Should have worst R² scores among all methods
    - Has minimal computational cost (instant)
    """

    def __init__(self):
        """Initialize zero imputer."""
        pass

    def fit(self, X: np.ndarray):
        """
        Fit method (does nothing for zero imputation).

        Args:
            X: Training data (not used)
        """
        # No fitting required for zero imputation
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values with zeros.

        Args:
            X: Data with missing values (NaN or masked)

        Returns:
            Data with NaNs replaced by zeros
        """
        X_imputed = X.copy()
        X_imputed[np.isnan(X_imputed)] = 0.0
        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Data with missing values

        Returns:
            Data with zeros imputed
        """
        self.fit(X)
        return self.transform(X)


def evaluate_zero_imputation(
    data: np.ndarray,
    mask: np.ndarray,
    missingness_rate: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """
    Evaluate zero imputation on corrupted data.

    Args:
        data: Original normalized data (samples x features)
        mask: Boolean mask (True = artificially corrupted, False = original)
        missingness_rate: Fraction of data that was corrupted
        seed: Random seed for reproducibility

    Returns:
        Tuple of (predictions, targets, metrics_dict, imputation_time)
    """
    np.random.seed(seed)

    # Create corrupted data (set masked values to NaN)
    corrupted_data = data.copy()
    corrupted_data[mask] = np.nan

    # Time the imputation
    start_time = time.time()
    imputer = ZeroImputer()
    imputed_data = imputer.fit_transform(corrupted_data)
    imputation_time = time.time() - start_time

    # Extract predictions and targets for masked values only
    predictions = imputed_data[mask]
    targets = data[mask]

    # Compute metrics
    r2 = r2_score(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)

    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'imputation_time': imputation_time,
        'missingness_rate': missingness_rate,
        'seed': seed,
        'num_imputed_values': int(mask.sum())
    }

    return predictions, targets, metrics, imputation_time


def run_zero_imputation_experiments(
    data: np.ndarray,
    missingness_rates: list = [0.01, 0.05, 0.10],
    seeds: list = [42, 50, 100],
    results_dir: str = 'results/baselines'
) -> Dict:
    """
    Run zero imputation experiments across multiple missingness rates and seeds.

    Args:
        data: Normalized formulation data
        missingness_rates: List of missingness fractions to test
        seeds: List of random seeds for statistical robustness
        results_dir: Directory to save results

    Returns:
        Dictionary of all results
    """
    import os
    import json

    print("="*80)
    print("ZERO IMPUTATION BASELINE EXPERIMENTS")
    print("="*80)

    os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'predictions'), exist_ok=True)

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
                    results_dir, 'predictions',
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
            results_dir, 'metrics',
            f'miss{miss_rate}_zero_metrics.json'
        )
        with open(metrics_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"  Aggregated: R²={aggregated['r2_mean']:.4f}±{aggregated['r2_std']:.4f}, "
              f"RMSE={aggregated['rmse_mean']:.4f}±{aggregated['rmse_std']:.4f}")

    # Save summary
    summary_file = os.path.join(results_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ZERO IMPUTATION EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved to: {results_dir}/")
    print(f"  - summary.json")
    print(f"  - metrics/miss*_zero_metrics.json")
    print(f"  - predictions/miss*_seed*_predictions.npz")
    print("="*80)

    return all_results


if __name__ == "__main__":
    print("Zero Imputation Baseline")
    print("This module provides a naive baseline for comparison.")
    print("Usage: import and call run_zero_imputation_experiments()")
