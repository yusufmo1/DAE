"""
Evaluation module for Denoising Autoencoder.
Computes metrics on imputed values.
"""

import torch
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import r2_score, mean_squared_error
import json
import os


class DAEEvaluator:
    """
    Evaluator for DAE imputation performance.

    Computes metrics (R², RMSE) only on masked (imputed) values.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize the evaluator.

        Args:
            model: Trained DAE model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def get_predictions(
        self,
        corrupted_data: torch.Tensor,
        noise_std: float = 0.1
    ) -> torch.Tensor:
        """
        Get model predictions.

        Args:
            corrupted_data: Data with masked values
            noise_std: Standard deviation of Gaussian noise to add

        Returns:
            Predictions tensor
        """
        with torch.no_grad():
            # Move to device
            corrupted_data = corrupted_data.to(self.device)

            # Add noise (same as training)
            noise = torch.randn_like(corrupted_data) * noise_std
            noisy_data = torch.clamp(corrupted_data + noise, 0, 1)

            # Get predictions
            _, reconstructed = self.model(noisy_data)

        return reconstructed.cpu()

    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics on masked values only.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            mask: Boolean mask (True = was masked/missing)

        Returns:
            Dictionary with R² and RMSE scores
        """
        # Convert to numpy and select only masked values
        pred_masked = predictions[mask].numpy()
        true_masked = targets[mask].numpy()

        # Compute R² score
        r2 = r2_score(true_masked, pred_masked)

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(true_masked, pred_masked))

        # Compute MSE for completeness
        mse = mean_squared_error(true_masked, pred_masked)

        # Compute MAE for additional insight
        mae = np.mean(np.abs(true_masked - pred_masked))

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'n_masked': mask.sum().item(),
        }

        return metrics

    def evaluate(
        self,
        original_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        mask: torch.Tensor,
        noise_std: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Evaluate model on data.

        Args:
            original_data: Clean data (targets)
            corrupted_data: Data with masked values
            mask: Boolean mask (True = was masked/missing)
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Tuple of (predictions, metrics_dict)
        """
        # Get predictions
        predictions = self.get_predictions(corrupted_data, noise_std)

        # Compute metrics
        metrics = self.compute_metrics(predictions, original_data, mask)

        return predictions, metrics

    def evaluate_multiple_runs(
        self,
        results: list,
        metric: str = 'r2'
    ) -> Tuple[float, float]:
        """
        Compute mean and std of a metric across multiple runs.

        Args:
            results: List of metric dictionaries from multiple runs
            metric: Metric name to aggregate

        Returns:
            Tuple of (mean, std)
        """
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation

        return mean, std


def evaluate_dae(
    model: torch.nn.Module,
    original_data: torch.Tensor,
    corrupted_data: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    noise_std: float = 0.1,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Convenience function to evaluate a trained DAE.

    Args:
        model: Trained DAE model
        original_data: Clean data (targets)
        corrupted_data: Data with masked values
        mask: Boolean mask (True = was masked/missing)
        device: Device to run on
        noise_std: Standard deviation of Gaussian noise
        verbose: Whether to print metrics

    Returns:
        Tuple of (predictions, metrics_dict)
    """
    evaluator = DAEEvaluator(model, device)
    predictions, metrics = evaluator.evaluate(
        original_data, corrupted_data, mask, noise_std
    )

    if verbose:
        print("\nEvaluation Metrics (on masked values only):")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Number of masked values: {metrics['n_masked']:,}")

    return predictions, metrics


def save_metrics(metrics: Dict, save_path: str):
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


def save_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    save_path: str
):
    """
    Save predictions and targets for masked values.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        mask: Boolean mask (True = was masked/missing)
        save_path: Path to save data
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Extract only masked values
    pred_masked = predictions[mask].numpy()
    true_masked = targets[mask].numpy()

    # Save as numpy arrays
    np.savez(
        save_path,
        predictions=pred_masked,
        targets=true_masked
    )

    print(f"Predictions saved to {save_path}")


def load_predictions(load_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load saved predictions and targets.

    Args:
        load_path: Path to load from

    Returns:
        Tuple of (predictions, targets)
    """
    data = np.load(load_path)
    return data['predictions'], data['targets']


def aggregate_results(results_list: list, seeds: list) -> Dict:
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
    for metric in ['r2', 'rmse', 'mse', 'mae']:
        values = [r[metric] for r in results_list]
        aggregated[f'{metric}_mean'] = float(np.mean(values))
        aggregated[f'{metric}_std'] = float(np.std(values, ddof=1))
        aggregated[f'{metric}_values'] = [float(v) for v in values]

    return aggregated


if __name__ == "__main__":
    print("This module provides DAE evaluation utilities.")
    print("Use main.py to run experiments.")
