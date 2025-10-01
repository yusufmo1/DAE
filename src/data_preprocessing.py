"""
Data preprocessing module for pharmaceutical formulation data.
Handles data loading, normalization, and corruption mechanisms for DAE training.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


class FormulationDataPreprocessor:
    """
    Preprocesses pharmaceutical formulation data for DAE training.

    The dataset is a sparse matrix where:
    - Rows represent formulations
    - Columns represent ingredients
    - Values represent composition percentages (0-100% w/w)
    - ~99% of values are zeros (unused ingredients)
    """

    def __init__(self, data_path: str, metadata_cols: int = 6, ingredient_end_col: int = 342):
        """
        Initialize the preprocessor.

        Args:
            data_path: Path to the CSV file containing formulation data
            metadata_cols: Number of metadata columns to exclude (default: 6)
                          These are: id, article, author, formulation_id, operator, reviewer
            ingredient_end_col: Column index where ingredients end (default: 342)
                              Columns 6-341 are the 336 ingredient columns
        """
        self.data_path = data_path
        self.metadata_cols = metadata_cols
        self.ingredient_end_col = ingredient_end_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        self.ingredient_names = None

    def load_data(self) -> pd.DataFrame:
        """
        Load formulation data from CSV and extract ingredient columns.
        Excludes Sulfasalazine column as per requirements.

        Returns:
            DataFrame with only ingredient composition columns
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Extract only ingredient columns (columns 6 to 341, excluding process parameters)
        self.data = df.iloc[:, self.metadata_cols:self.ingredient_end_col]

        # Handle NaN values - fill with 0 (ingredient not used)
        nan_count = self.data.isna().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count:,} NaN values, filling with 0")
            self.data = self.data.fillna(0)

        # Remove Sulfasalazine column if present (as per user requirements)
        if 'Sulfasalazine' in self.data.columns:
            print("Excluding Sulfasalazine column as per requirements")
            self.data = self.data.drop(columns=['Sulfasalazine'])

        self.ingredient_names = self.data.columns.tolist()

        print(f"Loaded {len(self.data)} formulations with {len(self.ingredient_names)} ingredients")
        print(f"Total data points: {self.data.size:,}")

        # Calculate sparsity
        non_zero_count = (self.data != 0).sum().sum()
        zero_count = (self.data == 0).sum().sum()
        sparsity = (zero_count / self.data.size) * 100
        print(f"Sparsity: {sparsity:.2f}% zeros, {100-sparsity:.2f}% non-zeros")

        return self.data

    def normalize_data(self) -> np.ndarray:
        """
        Normalize data to [0, 1] range using MinMaxScaler.

        Returns:
            Normalized data as numpy array
        """
        print("Normalizing data to [0, 1] range...")
        self.scaled_data = self.scaler.fit_transform(self.data)
        return self.scaled_data

    def create_corruption_mask(
        self,
        data_shape: Tuple[int, int],
        missingness_rate: float,
        seed: int
    ) -> np.ndarray:
        """
        Create a binary mask for data corruption.

        Args:
            data_shape: Shape of the data (n_samples, n_features)
            missingness_rate: Proportion of data to mask (e.g., 0.01 for 1%)
            seed: Random seed for reproducibility

        Returns:
            Binary mask array (1 = keep, 0 = mask/remove)
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        mask = torch.rand(data_shape) >= missingness_rate
        return mask.numpy()

    def add_gaussian_noise(
        self,
        data: np.ndarray,
        mean: float = 0.0,
        std: float = 0.1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to all data points.

        Args:
            data: Input data
            mean: Mean of Gaussian noise (default: 0.0)
            std: Standard deviation of Gaussian noise (default: 0.1)
            seed: Random seed for reproducibility

        Returns:
            Noisy data
        """
        if seed is not None:
            np.random.seed(seed)

        noise = np.random.normal(mean, std, data.shape)
        noisy_data = data + noise

        # Clip to [0, 1] range since data is normalized
        noisy_data = np.clip(noisy_data, 0, 1)

        return noisy_data

    def prepare_data(
        self,
        missingness_rate: float = 0.01,
        noise_std: float = 0.1,
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for DAE training with corruption mechanisms.

        This method:
        1. Creates a binary mask to simulate missing data
        2. Sets masked values to zero
        3. Adds Gaussian noise to all values

        Args:
            missingness_rate: Proportion of data to mask (0.01, 0.05, or 0.10)
            noise_std: Standard deviation of Gaussian noise (default: 0.1)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (original_data, corrupted_data, mask)
            - original_data: Clean normalized data as torch.Tensor
            - corrupted_data: Masked + noisy data as torch.Tensor
            - mask: Binary mask (True = was masked/missing)
        """
        if self.scaled_data is None:
            raise ValueError("Data must be normalized first. Call normalize_data().")

        print(f"\nPreparing data with {missingness_rate*100}% missingness (seed={seed})...")

        # Create mask (True = masked/missing, False = keep)
        mask = self.create_corruption_mask(
            self.scaled_data.shape,
            missingness_rate,
            seed
        )
        mask = mask == 0  # Invert: True = masked

        # Count masked entries
        n_masked = mask.sum()
        print(f"Masked {n_masked:,} values ({n_masked/self.scaled_data.size*100:.2f}%)")

        # Create corrupted data: set masked values to zero
        corrupted_data = self.scaled_data.copy()
        corrupted_data[mask] = 0

        # Add Gaussian noise to all values
        corrupted_data = self.add_gaussian_noise(
            corrupted_data,
            mean=0.0,
            std=noise_std,
            seed=seed
        )

        # Convert to torch tensors
        original_tensor = torch.FloatTensor(self.scaled_data)
        corrupted_tensor = torch.FloatTensor(corrupted_data)
        mask_tensor = torch.BoolTensor(mask)

        return original_tensor, corrupted_tensor, mask_tensor

    def get_data_statistics(self) -> dict:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data().")

        non_zero_values = self.data.values[self.data.values != 0]

        stats = {
            'n_formulations': len(self.data),
            'n_ingredients': len(self.ingredient_names),
            'total_data_points': self.data.size,
            'n_zeros': (self.data == 0).sum().sum(),
            'n_non_zeros': (self.data != 0).sum().sum(),
            'sparsity_pct': (self.data == 0).sum().sum() / self.data.size * 100,
            'unique_non_zero_values': len(np.unique(non_zero_values)),
            'non_zero_mean': non_zero_values.mean(),
            'non_zero_median': np.median(non_zero_values),
            'non_zero_min': non_zero_values.min(),
            'non_zero_max': non_zero_values.max(),
        }

        return stats


if __name__ == "__main__":
    print("This module provides data preprocessing utilities.")
    print("Use main.py to run experiments.")
