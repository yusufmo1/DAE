"""
Centralized configuration for all imputation experiments.
Contains shared paths, hyperparameters, and settings for DAE, KNN, and baseline methods.
"""

from typing import List


class BaseConfig:
    """Base configuration shared across all methods."""
    # Data parameters
    DATA_PATH = 'data/material_name_smilesRemoved.csv'
    METADATA_COLS = 6

    # Experimental parameters (shared)
    MISSINGNESS_RATES = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
    SEEDS = [42, 50, 100]
    NOISE_STD = 0.1  # Gaussian noise standard deviation for corruption


class DAEConfig(BaseConfig):
    """Configuration for DAE experiments."""
    # DAE-specific hyperparameters
    LEARNING_RATES = [1e-1, 1e-3, 1e-5]
    NEURON_SIZES = [256, 512, 1024]
    EPOCH_SETTINGS = [100, 500, 1000, 1200]

    # Output directories
    RESULTS_DIR = 'results/dae'
    MODELS_DIR = 'results/dae/models'
    METRICS_DIR = 'results/dae/metrics'
    PLOTS_DIR = 'results/dae/plots'


class KNNConfig(BaseConfig):
    """Configuration for KNN experiments."""
    # KNN-specific parameters
    N_NEIGHBORS = [3, 5, 10, 20, 50]
    WEIGHTS = ['uniform', 'distance']
    METRICS = ['euclidean', 'manhattan', 'cosine']

    # Parallel processing settings
    NN_N_JOBS = -1  # Use all CPU cores for neighbor search
    EXPERIMENT_N_JOBS = 12  # Run 12 experiments in parallel

    # Output directories
    RESULTS_DIR = 'results/knn'
    METRICS_DIR = 'results/knn/metrics'
    PREDICTIONS_DIR = 'results/knn/predictions'
    PLOTS_DIR = 'results/knn/plots'


class BaselineConfig(BaseConfig):
    """Configuration for baseline experiments."""
    # Output directories
    RESULTS_DIR = 'results/baselines'
    METRICS_DIR = 'results/baselines/metrics'
    PREDICTIONS_DIR = 'results/baselines/predictions'
    PLOTS_DIR = 'results/baselines/plots'


class ComparisonConfig:
    """Configuration for cross-method comparisons."""
    # Input directories (results from other experiments)
    DAE_RESULTS_DIR = 'results/dae'
    KNN_RESULTS_DIR = 'results/knn'
    BASELINE_RESULTS_DIR = 'results/baselines'

    # Output directory
    OUTPUT_DIR = 'results/comparisons'


# Quick test configurations (reduced parameters for fast testing)
class QuickTestConfig:
    """Reduced configuration for quick testing."""
    MISSINGNESS_RATES = [0.01]
    SEEDS = [42]

    # DAE quick test
    DAE_LEARNING_RATES = [1e-3]
    DAE_NEURON_SIZES = [256]
    DAE_EPOCH_SETTINGS = [100]

    # KNN quick test
    KNN_N_NEIGHBORS = [5]
    KNN_WEIGHTS = ['distance']
    KNN_METRICS = ['euclidean']


def get_config(method: str, quick_test: bool = False):
    """
    Get configuration for a specific method.

    Args:
        method: Method name ('dae', 'knn', 'baseline', 'comparison')
        quick_test: If True, return quick test configuration

    Returns:
        Configuration class instance
    """
    config_map = {
        'dae': DAEConfig,
        'knn': KNNConfig,
        'baseline': BaselineConfig,
        'comparison': ComparisonConfig
    }

    if method not in config_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(config_map.keys())}")

    config_class = config_map[method]

    if quick_test and method in ['dae', 'knn']:
        # Overlay quick test settings
        config = config_class()
        config.MISSINGNESS_RATES = QuickTestConfig.MISSINGNESS_RATES
        config.SEEDS = QuickTestConfig.SEEDS

        if method == 'dae':
            config.LEARNING_RATES = QuickTestConfig.DAE_LEARNING_RATES
            config.NEURON_SIZES = QuickTestConfig.DAE_NEURON_SIZES
            config.EPOCH_SETTINGS = QuickTestConfig.DAE_EPOCH_SETTINGS
        elif method == 'knn':
            config.N_NEIGHBORS = QuickTestConfig.KNN_N_NEIGHBORS
            config.WEIGHTS = QuickTestConfig.KNN_WEIGHTS
            config.METRICS = QuickTestConfig.KNN_METRICS

        return config

    return config_class()


if __name__ == "__main__":
    print("Configuration module for imputation experiments")
    print("\nAvailable configurations:")
    print("  - DAEConfig: Denoising Autoencoder settings")
    print("  - KNNConfig: K-Nearest Neighbors settings")
    print("  - BaselineConfig: Baseline method settings")
    print("  - ComparisonConfig: Cross-method comparison settings")
    print("\nUse get_config(method) to retrieve configuration")
