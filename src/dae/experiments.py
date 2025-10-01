"""
DAE experiment orchestration logic.
Handles running DAE experiments across multiple configurations and seeds.
"""

import torch
import numpy as np
import os
import json
from typing import Dict, List
from itertools import product

from ..common.data_preprocessing import FormulationDataPreprocessor
from ..config import DAEConfig
from .model import create_dae, get_device
from .train import train_dae
from .evaluate import evaluate_dae, aggregate_results, save_metrics, save_predictions
from .plots import generate_all_dae_plots


def run_single_experiment(
    preprocessor: FormulationDataPreprocessor,
    device: torch.device,
    config: DAEConfig,
    missingness_rate: float,
    learning_rate: float,
    neuron_size: int,
    num_epochs: int,
    seeds: List[int],
    save_results: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run a single DAE experiment configuration across multiple seeds.

    Args:
        preprocessor: Data preprocessor with loaded data
        device: Device to run on (CPU/CUDA/MPS)
        config: DAE configuration object
        missingness_rate: Proportion of data to mask
        learning_rate: Learning rate for optimizer
        neuron_size: Number of neurons in hidden layers
        num_epochs: Number of training epochs
        seeds: List of random seeds to use
        save_results: Whether to save models and metrics
        verbose: Whether to print progress

    Returns:
        Dictionary with aggregated results across seeds
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: Miss={missingness_rate*100:.0f}%, LR={learning_rate}, "
              f"Neurons={neuron_size}, Epochs={num_epochs}")
        print(f"{'='*80}")

    input_dim = preprocessor.scaled_data.shape[1]
    results_list = []
    predictions_list = []
    targets_list = []

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nSeed {seed_idx+1}/{len(seeds)}: {seed}")

        # Prepare corrupted data
        original_data, corrupted_data, mask = preprocessor.prepare_data(
            missingness_rate=missingness_rate,
            noise_std=config.NOISE_STD,
            seed=seed
        )

        # Create model
        model = create_dae(input_dim=input_dim, neuron_size=neuron_size, device=device)

        # Train model
        metadata = {
            'missingness_rate': missingness_rate,
            'learning_rate': learning_rate,
            'neuron_size': neuron_size,
            'num_epochs': num_epochs,
            'seed': seed,
        }

        model_path = None
        if save_results:
            model_path = os.path.join(
                config.MODELS_DIR,
                f'miss{missingness_rate}_lr{learning_rate}_n{neuron_size}_ep{num_epochs}_seed{seed}.pt'
            )

        trained_model, loss_history = train_dae(
            model=model,
            original_data=original_data,
            corrupted_data=corrupted_data,
            mask=mask,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            save_path=model_path,
            metadata=metadata,
            verbose=False
        )

        # Evaluate model
        predictions, metrics = evaluate_dae(
            model=trained_model,
            original_data=original_data,
            corrupted_data=corrupted_data,
            mask=mask,
            device=device,
            noise_std=config.NOISE_STD,
            verbose=verbose
        )

        results_list.append(metrics)
        predictions_list.append(predictions[mask].numpy())
        targets_list.append(original_data[mask].numpy())

        # Save loss history for this seed
        if save_results and seed_idx == 0:  # Save loss for first seed only
            loss_file = os.path.join(
                config.METRICS_DIR,
                f'miss{missingness_rate}_lr{learning_rate}_n{neuron_size}_ep{num_epochs}_loss.json'
            )
            os.makedirs(os.path.dirname(loss_file), exist_ok=True)
            with open(loss_file, 'w') as f:
                json.dump(loss_history, f)

    # Aggregate results across seeds
    aggregated = aggregate_results(results_list, seeds)

    # Save aggregated results
    if save_results:
        metrics_file = os.path.join(
            config.METRICS_DIR,
            f'miss{missingness_rate}_lr{learning_rate}_n{neuron_size}_ep{num_epochs}_metrics.json'
        )
        save_metrics(aggregated, metrics_file)

        # Save predictions from first seed
        pred_file = os.path.join(
            config.METRICS_DIR,
            f'miss{missingness_rate}_lr{learning_rate}_n{neuron_size}_ep{num_epochs}_predictions.npz'
        )
        np.savez(pred_file, predictions=predictions_list[0], targets=targets_list[0])

    if verbose:
        print(f"\nAggregated Results (n={len(seeds)} runs):")
        print(f"  R² = {aggregated['r2_mean']:.4f} ± {aggregated['r2_std']:.4f}")
        print(f"  RMSE = {aggregated['rmse_mean']:.4f} ± {aggregated['rmse_std']:.4f}")

    return aggregated


def run_all_experiments(
    config: DAEConfig,
    missingness_rates: List[float] = None,
    learning_rates: List[float] = None,
    neuron_sizes: List[int] = None,
    epoch_settings: List[int] = None,
    seeds: List[int] = None
) -> Dict:
    """
    Run all DAE experiments in the paper.

    Args:
        config: DAE configuration object
        missingness_rates: List of missingness rates (uses config default if None)
        learning_rates: List of learning rates (uses config default if None)
        neuron_sizes: List of neuron sizes (uses config default if None)
        epoch_settings: List of epoch settings (uses config default if None)
        seeds: List of seeds (uses config default if None)

    Returns:
        Dictionary with all results
    """
    # Use defaults from config if not specified
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES
    if learning_rates is None:
        learning_rates = config.LEARNING_RATES
    if neuron_sizes is None:
        neuron_sizes = config.NEURON_SIZES
    if epoch_settings is None:
        epoch_settings = config.EPOCH_SETTINGS
    if seeds is None:
        seeds = config.SEEDS

    # Setup
    print("="*80)
    print("DAE FOR PHARMACEUTICAL FORMULATION DATA IMPUTATION")
    print("="*80)

    # Get device
    device = get_device()

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = FormulationDataPreprocessor(
        data_path=config.DATA_PATH,
        metadata_cols=config.METADATA_COLS
    )
    preprocessor.load_data()
    preprocessor.normalize_data()

    # Print dataset statistics
    stats = preprocessor.get_data_statistics()
    print("\nDataset Statistics:")
    print(f"  Formulations: {stats['n_formulations']:,}")
    print(f"  Ingredients: {stats['n_ingredients']:,}")
    print(f"  Total data points: {stats['total_data_points']:,}")
    print(f"  Sparsity: {stats['sparsity_pct']:.2f}% zeros")

    # Calculate total experiments
    total_experiments = (len(missingness_rates) * len(learning_rates) *
                        len(neuron_sizes) * len(epoch_settings))
    print(f"\nTotal experiments to run: {total_experiments}")
    print(f"Seeds per experiment: {len(seeds)}")
    print(f"Total training runs: {total_experiments * len(seeds)}")

    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Run all experiments
    all_results = {}
    experiment_count = 0

    for miss_rate, lr, neurons, epochs in product(
        missingness_rates, learning_rates, neuron_sizes, epoch_settings
    ):
        experiment_count += 1
        print(f"\n[Experiment {experiment_count}/{total_experiments}]")

        result = run_single_experiment(
            preprocessor=preprocessor,
            device=device,
            config=config,
            missingness_rate=miss_rate,
            learning_rate=lr,
            neuron_size=neurons,
            num_epochs=epochs,
            seeds=seeds,
            save_results=True,
            verbose=True
        )

        # Store result
        key = f"miss{miss_rate}_lr{lr}_n{neurons}_ep{epochs}"
        all_results[key] = result

    # Save summary
    summary_file = os.path.join(config.RESULTS_DIR, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    return all_results


def generate_plots(config: DAEConfig, missingness_rates: List[float] = None):
    """
    Generate all DAE visualization plots.

    Args:
        config: DAE configuration object
        missingness_rates: List of missingness rates to plot (uses config default if None)
    """
    if missingness_rates is None:
        missingness_rates = config.MISSINGNESS_RATES

    print("\nGenerating DAE plots...")
    generate_all_dae_plots(config.RESULTS_DIR, missingness_rates)
    print(f"Plots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    print("DAE experiment orchestration module.")
    print("Use run_all_experiments() to execute the full experimental pipeline.")
