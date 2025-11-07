"""
MissForest imputation for pharmaceutical formulation data.
Provides an iterative Random Forest-based imputation baseline using sklearn's IterativeImputer.
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from typing import Tuple, Dict, Optional
import time


class MissForestImputer:
    """
    MissForest imputation for missing pharmaceutical formulation data.

    Uses the MissForest algorithm with RandomForest estimators:
    - Iterative imputation: imputes each feature using other features
    - Uses Random Forest for predictions (more stable than LightGBM with sparse data)
    - Handles mixed continuous/categorical data
    - Typically more accurate than simple methods (mean, KNN)

    Advantages:
    - Captures complex nonlinear patterns
    - Robust to high missingness
    - Handles mixed data types
    - Parallel execution via RandomForest n_jobs

    Disadvantages:
    - Slower than KNN (but faster than DAE)
    - More parameters to tune
    - Requires more memory
    """

    def __init__(
        self,
        max_iter: int = 10,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize MissForest imputer.

        Args:
            max_iter: Maximum number of imputation iterations (default: 10)
                     Paper uses 10 as default, converges typically in 5-10 iterations
            n_estimators: Number of trees in forest (default: 100)
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_leaf: Minimum samples per leaf (default: 1)
            n_jobs: Number of parallel jobs (-1 = use all cores)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.imputation_time = None

        # IterativeImputer instance (will be created on first use)
        self._imputer = None

    def _create_iterative_imputer(self):
        """
        Create sklearn IterativeImputer instance with RandomForest estimator.
        """
        # Create RandomForest estimator
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        # Create IterativeImputer with RandomForest (MissForest-like behavior)
        self._imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=2 if self.verbose else 0,
            keep_empty_features=True  # Keep zero-valued features
        )

    def fit_transform(
        self,
        corrupted_data: np.ndarray,
        mask: np.ndarray,
        categorical: Optional[list] = None
    ) -> np.ndarray:
        """
        Impute missing values in corrupted data using IterativeImputer (MissForest-like).

        Args:
            corrupted_data: Data with masked values set to 0 (same as DAE preprocessing)
            mask: Boolean mask (True = was masked/missing, needs imputation)
            categorical: List of categorical column indices (not used - kept for API compatibility)

        Returns:
            Imputed data with filled values
        """
        start_time = time.time()

        # Create IterativeImputer instance
        self._create_iterative_imputer()

        # Convert to format expected by sklearn IterativeImputer
        # IterativeImputer expects NaN for missing values
        data_with_nan = corrupted_data.copy()
        data_with_nan[mask] = np.nan

        # Fit and transform
        if self.verbose:
            print(f"\nRunning MissForest-style imputation (sklearn IterativeImputer, max_iter={self.max_iter})...")

        imputed_data = self._imputer.fit_transform(data_with_nan)

        self.imputation_time = time.time() - start_time

        if self.verbose:
            print(f"Imputation completed in {self.imputation_time:.2f}s")

        return imputed_data

    def impute(
        self,
        original_data: np.ndarray,
        corrupted_data: np.ndarray,
        mask: np.ndarray,
        categorical: Optional[list] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Impute missing values and evaluate performance.

        Args:
            original_data: Clean data (for evaluation)
            corrupted_data: Data with masked values set to zero (same as DAE)
            mask: Boolean mask (True = was masked/missing)
            categorical: List of categorical column indices (optional)

        Returns:
            Tuple of (imputed_data, metrics_dict)
        """
        # Impute using MissForest
        imputed_data = self.fit_transform(corrupted_data, mask, categorical)

        # Compute metrics only on masked (originally missing) values
        metrics = self.compute_metrics(imputed_data, original_data, mask)

        return imputed_data, metrics

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics on masked values only.

        Args:
            predictions: Imputed data
            targets: Ground truth data
            mask: Boolean mask (True = was masked/missing)

        Returns:
            Dictionary with R², RMSE, MSE, MAE scores
        """
        # Extract only masked values
        pred_masked = predictions[mask]
        true_masked = targets[mask]

        # Compute R² score
        r2 = r2_score(true_masked, pred_masked)

        # Compute RMSE
        mse = mean_squared_error(true_masked, pred_masked)
        rmse = np.sqrt(mse)

        # Compute MAE
        mae = np.mean(np.abs(true_masked - pred_masked))

        metrics = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mse': float(mse),
            'mae': float(mae),
            'n_masked': int(mask.sum()),
            'max_iter': self.max_iter,
            'n_estimators': self.n_estimators,
            'imputation_time': self.imputation_time,
        }

        return metrics

    def get_config(self) -> Dict:
        """
        Get imputer configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            'method': 'MissForest',
            'max_iter': self.max_iter,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'n_jobs': self.n_jobs,
        }

    def __repr__(self) -> str:
        """String representation of imputer."""
        return (f"MissForestImputer(max_iter={self.max_iter}, "
                f"n_estimators={self.n_estimators})")


def create_missforest_imputer(
    max_iter: int = 10,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = False
) -> MissForestImputer:
    """
    Factory function to create a MissForest imputer.

    Args:
        max_iter: Maximum number of imputation iterations
        n_estimators: Number of trees in forest
        max_depth: Maximum tree depth
        min_samples_leaf: Minimum samples per leaf
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Whether to print configuration

    Returns:
        Configured MissForest imputer
    """
    imputer = MissForestImputer(
        max_iter=max_iter,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )

    if verbose:
        print(f"\nCreated MissForest Imputer:")
        print(f"  Max iterations: {max_iter}")
        print(f"  N estimators: {n_estimators}")
        print(f"  Max depth: {max_depth}")
        print(f"  Min samples per leaf: {min_samples_leaf}")
        print(f"  Parallel jobs: {n_jobs}")

    return imputer


if __name__ == "__main__":
    print("This module provides MissForest imputation functionality.")
    print("Use run_missforest.py to run MissForest imputation experiments.")
