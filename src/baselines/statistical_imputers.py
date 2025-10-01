"""
Statistical Imputation Baselines.
Provides mean and median imputation strategies for comparison with sophisticated methods.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict
import time


class MeanImputer:
    """
    Mean imputation baseline - fills missing values with column-wise mean.

    For sparse pharmaceutical data, computes mean only from non-zero values
    to get meaningful average ingredient amounts.

    Expected behavior:
    - Better than zero imputation (uses actual data statistics)
    - Captures average ingredient usage patterns
    - Fast computation (single pass over data)
    - Should outperform zero but underperform KNN/DAE
    """

    def __init__(self, ignore_zeros: bool = True):
        """
        Initialize mean imputer.

        Args:
            ignore_zeros: If True, compute mean ignoring zero values (for sparse data)
        """
        self.ignore_zeros = ignore_zeros
        self.column_means = None

    def fit(self, X: np.ndarray):
        """
        Compute column-wise means from training data.

        Args:
            X: Training data (samples x features)
        """
        if self.ignore_zeros:
            # Compute mean ignoring zeros and NaN (for sparse data)
            self.column_means = np.zeros(X.shape[1])
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                # Filter out both zeros and NaN values
                valid_values = col_data[(col_data != 0) & (~np.isnan(col_data))]
                if len(valid_values) > 0:
                    self.column_means[col_idx] = np.mean(valid_values)
                else:
                    self.column_means[col_idx] = 0.0
        else:
            # Standard mean (including zeros, but ignoring NaN)
            self.column_means = np.nanmean(X, axis=0)
            self.column_means = np.nan_to_num(self.column_means, nan=0.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values with column means.

        Args:
            X: Data with missing values (NaN)

        Returns:
            Data with NaNs replaced by column means
        """
        if self.column_means is None:
            raise ValueError("Imputer must be fitted before transform. Call fit() first.")

        X_imputed = X.copy()

        # Replace NaN values with column means
        nan_mask = np.isnan(X_imputed)
        for col_idx in range(X_imputed.shape[1]):
            col_nan_mask = nan_mask[:, col_idx]
            if np.any(col_nan_mask):
                X_imputed[col_nan_mask, col_idx] = self.column_means[col_idx]

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Data with missing values

        Returns:
            Data with means imputed
        """
        self.fit(X)
        return self.transform(X)


class MedianImputer:
    """
    Median imputation baseline - fills missing values with column-wise median.

    More robust to outliers than mean imputation. For sparse data,
    computes median only from non-zero values.

    Expected behavior:
    - Similar to mean imputation but more robust
    - Better for skewed ingredient distributions
    - Slightly slower than mean (requires sorting)
    - Performance similar to mean imputation
    """

    def __init__(self, ignore_zeros: bool = True):
        """
        Initialize median imputer.

        Args:
            ignore_zeros: If True, compute median ignoring zero values (for sparse data)
        """
        self.ignore_zeros = ignore_zeros
        self.column_medians = None

    def fit(self, X: np.ndarray):
        """
        Compute column-wise medians from training data.

        Args:
            X: Training data (samples x features)
        """
        if self.ignore_zeros:
            # Compute median ignoring zeros and NaN (for sparse data)
            self.column_medians = np.zeros(X.shape[1])
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx]
                # Filter out both zeros and NaN values
                valid_values = col_data[(col_data != 0) & (~np.isnan(col_data))]
                if len(valid_values) > 0:
                    self.column_medians[col_idx] = np.median(valid_values)
                else:
                    self.column_medians[col_idx] = 0.0
        else:
            # Standard median (including zeros, but ignoring NaN)
            self.column_medians = np.nanmedian(X, axis=0)
            self.column_medians = np.nan_to_num(self.column_medians, nan=0.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values with column medians.

        Args:
            X: Data with missing values (NaN)

        Returns:
            Data with NaNs replaced by column medians
        """
        if self.column_medians is None:
            raise ValueError("Imputer must be fitted before transform. Call fit() first.")

        X_imputed = X.copy()

        # Replace NaN values with column medians
        nan_mask = np.isnan(X_imputed)
        for col_idx in range(X_imputed.shape[1]):
            col_nan_mask = nan_mask[:, col_idx]
            if np.any(col_nan_mask):
                X_imputed[col_nan_mask, col_idx] = self.column_medians[col_idx]

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Data with missing values

        Returns:
            Data with medians imputed
        """
        self.fit(X)
        return self.transform(X)


def evaluate_statistical_imputation(
    data: np.ndarray,
    mask: np.ndarray,
    missingness_rate: float,
    method: str = 'mean',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """
    Evaluate statistical imputation (mean or median) on corrupted data.

    Args:
        data: Original normalized data (samples x features)
        mask: Boolean mask (True = artificially corrupted, False = original)
        missingness_rate: Fraction of data that was corrupted
        method: 'mean' or 'median'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (predictions, targets, metrics_dict, imputation_time)
    """
    np.random.seed(seed)

    # Create corrupted data (set masked values to NaN)
    corrupted_data = data.copy()
    corrupted_data[mask] = np.nan

    # Select imputer
    if method == 'mean':
        imputer = MeanImputer(ignore_zeros=True)
    elif method == 'median':
        imputer = MedianImputer(ignore_zeros=True)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'mean' or 'median'.")

    # Time the imputation
    start_time = time.time()
    # Fit on non-masked data only (to avoid data leakage)
    train_data = data.copy()
    train_data[mask] = np.nan
    imputer.fit(train_data)
    imputed_data = imputer.transform(corrupted_data)
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
        'method': method,
        'seed': seed,
        'num_imputed_values': int(mask.sum())
    }

    return predictions, targets, metrics, imputation_time


if __name__ == "__main__":
    print("Statistical Imputation Baselines")
    print("This module provides mean and median imputation for comparison.")
    print("Usage: import and use MeanImputer or MedianImputer classes")
