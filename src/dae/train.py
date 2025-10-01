"""
Training module for Denoising Autoencoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import os


class DAETrainer:
    """
    Trainer for Denoising Autoencoder with pharmaceutical formulation data.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        noise_std: float = 0.1
    ):
        """
        Initialize the trainer.

        Args:
            model: DAE model to train
            device: Device to train on (CPU/CUDA/MPS)
            learning_rate: Learning rate for Adam optimizer
            noise_std: Standard deviation for Gaussian noise
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.noise_std = noise_std

        # Adam optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # MSE loss function
        self.criterion = nn.MSELoss(reduction='mean')

        # Training history
        self.loss_history = []

    def add_noise_to_batch(self, data: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Add Gaussian noise to a batch of data.

        Args:
            data: Input data tensor
            seed: Random seed for reproducibility (optional)

        Returns:
            Noisy data tensor
        """
        if seed is not None:
            torch.manual_seed(seed)

        noise = torch.randn_like(data) * self.noise_std
        noisy_data = data + noise

        # Clip to [0, 1] range
        noisy_data = torch.clamp(noisy_data, 0, 1)

        return noisy_data

    def compute_masked_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only on masked (missing) values.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            mask: Boolean mask (True = was masked/missing)

        Returns:
            Loss value
        """
        # Select only masked values
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]

        # Compute MSE only on masked values
        loss = self.criterion(masked_predictions, masked_targets)

        return loss

    def train_epoch(
        self,
        original_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        mask: torch.Tensor,
        add_noise: bool = True
    ) -> float:
        """
        Train for one epoch.

        Args:
            original_data: Clean data (targets)
            corrupted_data: Data with masked values set to zero
            mask: Boolean mask (True = was masked/missing)
            add_noise: Whether to add Gaussian noise to inputs

        Returns:
            Average loss for the epoch
        """
        self.model.train()

        # Move data to device
        original_data = original_data.to(self.device)
        corrupted_data = corrupted_data.to(self.device)
        mask = mask.to(self.device)

        # Add Gaussian noise to corrupted data if specified
        if add_noise:
            input_data = self.add_noise_to_batch(corrupted_data)
        else:
            input_data = corrupted_data

        # Forward pass
        _, reconstructed = self.model(input_data)

        # Compute loss only on masked values
        loss = self.compute_masked_loss(reconstructed, original_data, mask)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def train(
        self,
        original_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        mask: torch.Tensor,
        num_epochs: int = 1000,
        verbose: bool = True,
        log_interval: int = 100
    ) -> List[float]:
        """
        Train the model for multiple epochs.

        Args:
            original_data: Clean data (targets)
            corrupted_data: Data with masked values
            mask: Boolean mask (True = was masked/missing)
            num_epochs: Number of training epochs
            verbose: Whether to show progress bar
            log_interval: Print loss every N epochs

        Returns:
            List of loss values per epoch
        """
        self.loss_history = []

        # Training loop
        iterator = tqdm(range(num_epochs), desc='Training') if verbose else range(num_epochs)

        for epoch in iterator:
            # Train one epoch
            loss = self.train_epoch(original_data, corrupted_data, mask, add_noise=True)
            self.loss_history.append(loss)

            # Update progress bar or print
            if verbose:
                iterator.set_postfix({'loss': f'{loss:.6f}'})
            elif (epoch + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6f}")

        return self.loss_history

    def evaluate(
        self,
        original_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Evaluate the model on test data.

        Args:
            original_data: Clean data (targets)
            corrupted_data: Data with masked values
            mask: Boolean mask (True = was masked/missing)

        Returns:
            Tuple of (predictions, loss)
        """
        self.model.eval()

        with torch.no_grad():
            # Move data to device
            original_data = original_data.to(self.device)
            corrupted_data = corrupted_data.to(self.device)
            mask = mask.to(self.device)

            # Add noise (same as training)
            input_data = self.add_noise_to_batch(corrupted_data)

            # Forward pass
            _, reconstructed = self.model(input_data)

            # Compute loss only on masked values
            loss = self.compute_masked_loss(reconstructed, original_data, mask)

        return reconstructed, loss.item()

    def save_model(self, save_path: str, metadata: Optional[Dict] = None):
        """
        Save the trained model and training history.

        Args:
            save_path: Path to save the model
            metadata: Additional metadata to save
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'learning_rate': self.learning_rate,
            'noise_std': self.noise_std,
        }

        if metadata:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load a trained model.

        Args:
            load_path: Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.learning_rate = checkpoint['learning_rate']
        self.noise_std = checkpoint['noise_std']

        print(f"Model loaded from {load_path}")

    def save_loss_history(self, save_path: str):
        """
        Save loss history as JSON.

        Args:
            save_path: Path to save the loss history
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.loss_history, f)

        print(f"Loss history saved to {save_path}")


def train_dae(
    model: nn.Module,
    original_data: torch.Tensor,
    corrupted_data: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    learning_rate: float = 1e-3,
    num_epochs: int = 1000,
    save_path: Optional[str] = None,
    metadata: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[nn.Module, List[float]]:
    """
    Convenience function to train a DAE model.

    Args:
        model: DAE model to train
        original_data: Clean data (targets)
        corrupted_data: Data with masked values
        mask: Boolean mask (True = was masked/missing)
        device: Device to train on
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        save_path: Path to save the trained model (optional)
        metadata: Additional metadata to save with model
        verbose: Whether to show progress

    Returns:
        Tuple of (trained_model, loss_history)
    """
    # Create trainer
    trainer = DAETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        noise_std=0.1
    )

    # Train
    loss_history = trainer.train(
        original_data=original_data,
        corrupted_data=corrupted_data,
        mask=mask,
        num_epochs=num_epochs,
        verbose=verbose
    )

    # Save model if path provided
    if save_path:
        trainer.save_model(save_path, metadata=metadata)

    return model, loss_history


if __name__ == "__main__":
    print("This module provides DAE training utilities.")
    print("Use main.py to run experiments.")
