"""
K-Nearest Neighbors (KNN) imputation for pharmaceutical formulation data.
Provides a classical ML baseline for comparison with DAE approaches.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error
from typing import Tuple, Dict, Optional
import time


class KNNImputer:
    """
    K-Nearest Neighbors imputation for missing pharmaceutical formulation data.

    Uses sklearn's KNNImputer with support for:
    - Multiple K values
    - Different distance metrics (euclidean, manhattan, cosine)
    - Different weighting schemes (uniform, distance-weighted)

    Advantages:
    - No training required (instant predictions)
    - Interpretable (similarity-based)
    - Good baseline for comparison

    Disadvantages:
    - May not capture complex nonlinear patterns
    - Sensitive to distance metric with sparse data
    - Performance degrades with high missingness
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'distance',
        metric: str = 'euclidean',
        n_jobs: int = -1
    ):
        """
        Initialize KNN imputer.

        Args:
            n_neighbors: Number of neighbors to use (default: 5)
                        Common values: 3, 5, 10, 20, 50
            weights: Weight function for neighbors (default: 'distance')
                    'uniform': All neighbors weighted equally
                    'distance': Weight inversely proportional to distance
            metric: Distance metric (default: 'euclidean')
                   Options: 'euclidean', 'manhattan', 'cosine'
            n_jobs: Number of parallel jobs (kept for API compatibility)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs
        self.imputation_time = None

    def fit_transform(self, corrupted_data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Impute missing values in corrupted data using KNN.

        Args:
            corrupted_data: Data with masked values set to 0 (same as DAE preprocessing)
            mask: Boolean mask (True = was masked/missing, needs imputation)

        Returns:
            Imputed data with filled values
        """
        start_time = time.time()

        # Create NearestNeighbors model
        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=self.n_jobs
        )

        # Fit on corrupted data
        nn.fit(corrupted_data)

        # Find neighbors for all samples
        distances, indices = nn.kneighbors(corrupted_data)

        # Impute by averaging neighbors
        imputed_data = corrupted_data.copy()

        # For each sample, impute masked values using neighbors
        for i in range(len(corrupted_data)):
            if mask[i].any():  # If this sample has missing values
                neighbor_indices = indices[i]
                neighbor_distances = distances[i]

                # Get neighbor values
                neighbor_values = corrupted_data[neighbor_indices]
                neighbor_masks = mask[neighbor_indices]

                # Apply distance weights
                if self.weights == 'distance':
                    # Distance weighting (avoid division by zero)
                    weights = 1.0 / (neighbor_distances + 1e-8)
                else:  # uniform
                    weights = np.ones(len(neighbor_distances))

                # Weighted average for masked positions only
                for j in range(corrupted_data.shape[1]):
                    if mask[i, j]:
                        # Only use neighbor values that are NOT masked
                        valid_neighbors = ~neighbor_masks[:, j]

                        if valid_neighbors.any():
                            # Use only valid neighbors for this feature
                            valid_values = neighbor_values[valid_neighbors, j]
                            valid_weights = weights[valid_neighbors]
                            valid_weights = valid_weights / valid_weights.sum()

                            imputed_data[i, j] = np.average(valid_values, weights=valid_weights)
                        else:
                            # If all neighbors are masked, use overall mean of non-masked values
                            non_masked = ~mask[:, j]
                            if non_masked.any():
                                imputed_data[i, j] = corrupted_data[non_masked, j].mean()
                            else:
                                imputed_data[i, j] = 0

        self.imputation_time = time.time() - start_time

        return imputed_data

    def impute(
        self,
        original_data: np.ndarray,
        corrupted_data: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Impute missing values and evaluate performance.

        Args:
            original_data: Clean data (for evaluation)
            corrupted_data: Data with masked values set to zero (same as DAE)
            mask: Boolean mask (True = was masked/missing)

        Returns:
            Tuple of (imputed_data, metrics_dict)
        """
        # Impute using KNN (same 0-filled preprocessing as DAE)
        imputed_data = self.fit_transform(corrupted_data, mask)

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
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
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
            'method': 'KNN',
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
        }

    def __repr__(self) -> str:
        """String representation of imputer."""
        return (f"KNNImputer(n_neighbors={self.n_neighbors}, "
                f"weights='{self.weights}', metric='{self.metric}')")


def create_knn_imputer(
    n_neighbors: int = 5,
    weights: str = 'distance',
    metric: str = 'euclidean',
    n_jobs: int = -1,
    verbose: bool = False
) -> KNNImputer:
    """
    Factory function to create a KNN imputer.

    Args:
        n_neighbors: Number of neighbors
        weights: Weighting scheme ('uniform' or 'distance')
        metric: Distance metric
        n_jobs: Number of parallel jobs (not used, kept for API compatibility)
        verbose: Whether to print configuration

    Returns:
        Configured KNN imputer
    """
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs
    )

    if verbose:
        print(f"\nCreated KNN Imputer:")
        print(f"  K neighbors: {n_neighbors}")
        print(f"  Weighting: {weights}")
        print(f"  Distance metric: {metric}")

    return imputer


if __name__ == "__main__":
    print("This module provides KNN imputation functionality.")
    print("Use knn_main.py to run KNN imputation experiments.")
