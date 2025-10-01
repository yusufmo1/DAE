"""
Denoising Autoencoder (DAE) model for pharmaceutical formulation data imputation.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for imputing missing pharmaceutical formulation data.

    The architecture consists of:
    - Encoder: Two fully connected layers with BatchNorm and LeakyReLU
    - Decoder: Single linear layer with Sigmoid activation

    The model can be configured as either:
    - Undercomplete: latent_dim < input_dim (compression)
    - Overcomplete: latent_dim > input_dim (expansion)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        leaky_relu_slope: float = 0.2
    ):
        """
        Initialize the DAE.

        Args:
            input_dim: Number of input features (number of ingredients)
            hidden_dim: Number of neurons in first hidden layer
            latent_dim: Number of neurons in second hidden layer (latent space)
            leaky_relu_slope: Negative slope for LeakyReLU activation
        """
        super(DenoisingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: Input -> Hidden -> Latent
        self.encoder = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),

            # Second hidden layer (latent space)
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(negative_slope=leaky_relu_slope)
        )

        # Decoder: Latent -> Output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (latent_representation, reconstructed_output)
        """
        # Encode
        latent = self.encoder(x)

        # Decode
        reconstructed = self.decoder(latent)

        return latent, reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent representation
        """
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            latent: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output
        """
        return self.decoder(latent)

    def get_architecture_info(self) -> dict:
        """
        Get information about the model architecture.

        Returns:
            Dictionary with architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'architecture_type': 'overcomplete' if self.latent_dim > self.input_dim else 'undercomplete',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }

        return info


def get_device() -> torch.device:
    """
    Get the best available device for training (MPS > CUDA > CPU).

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders) GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def create_dae(
    input_dim: int,
    neuron_size: int = 256,
    device: torch.device = None
) -> DenoisingAutoencoder:
    """
    Factory function to create a DAE model.

    Args:
        input_dim: Number of input features
        neuron_size: Size of both hidden and latent layers (256, 512, or 1024)
        device: Device to place model on

    Returns:
        DAE model on specified device
    """
    if device is None:
        device = get_device()

    model = DenoisingAutoencoder(
        input_dim=input_dim,
        hidden_dim=neuron_size,
        latent_dim=neuron_size
    )

    model = model.to(device)

    # Print architecture info
    info = model.get_architecture_info()
    print(f"\nCreated {info['architecture_type'].upper()} DAE:")
    print(f"  Input dimension: {info['input_dim']}")
    print(f"  Hidden dimension: {info['hidden_dim']}")
    print(f"  Latent dimension: {info['latent_dim']}")
    print(f"  Total parameters: {info['total_parameters']:,}")

    return model


def main():
    """Example usage of the DAE model."""
    # Test on different neuron sizes
    input_dim = 382  # Number of ingredients in the dataset

    for neuron_size in [256, 512, 1024]:
        print(f"\n{'='*60}")
        print(f"Testing with {neuron_size} neurons")
        print(f"{'='*60}")

        model = create_dae(input_dim, neuron_size)

        # Test forward pass
        batch_size = 32
        x = torch.randn(batch_size, input_dim)

        device = get_device()
        x = x.to(device)

        with torch.no_grad():
            latent, reconstructed = model(x)

        print(f"\nForward pass test:")
        print(f"  Input shape: {x.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")


if __name__ == "__main__":
    main()
