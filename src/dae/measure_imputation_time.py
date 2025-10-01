"""
Measure imputation time for existing trained DAE models.
Updates metrics JSON files with imputation timing information.
"""

import torch
import numpy as np
import json
import os
import glob
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dae.model import DenoisingAutoencoder, get_device
from dae.evaluate import DAEEvaluator
from common.data_preprocessing import FormulationDataPreprocessor
from config import DAEConfig


def parse_model_filename(filename: str) -> dict:
    """
    Parse model filename to extract configuration parameters.

    Format: miss{rate}_lr{lr}_n{neurons}_ep{epochs}_seed{seed}.pt

    Args:
        filename: Model filename

    Returns:
        Dictionary with parsed parameters
    """
    basename = os.path.basename(filename).replace('.pt', '')
    parts = basename.split('_')

    params = {}
    for part in parts:
        if part.startswith('miss'):
            params['missingness_rate'] = float(part.replace('miss', ''))
        elif part.startswith('lr'):
            params['learning_rate'] = float(part.replace('lr', ''))
        elif part.startswith('n'):
            params['neuron_size'] = int(part.replace('n', ''))
        elif part.startswith('ep'):
            params['epochs'] = int(part.replace('ep', ''))
        elif part.startswith('seed'):
            params['seed'] = int(part.replace('seed', ''))

    return params


def measure_model_imputation_time(
    model_path: str,
    preprocessor: FormulationDataPreprocessor,
    device: torch.device,
    noise_std: float = 0.1
) -> float:
    """
    Measure imputation time for a single model.

    Args:
        model_path: Path to saved model
        preprocessor: Data preprocessor
        device: Device to run on
        noise_std: Noise standard deviation

    Returns:
        Imputation time in seconds
    """
    # Parse model filename
    params = parse_model_filename(model_path)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint['model_state_dict']

    # Get input dimension from model state
    input_dim = model_state['encoder.0.weight'].shape[1]

    # Create model
    model = DenoisingAutoencoder(
        input_dim=input_dim,
        hidden_dim=params['neuron_size'],
        latent_dim=params['neuron_size']
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    # Prepare data with same seed
    original_data, corrupted_data, mask = preprocessor.prepare_data(
        missingness_rate=params['missingness_rate'],
        noise_std=noise_std,
        seed=params['seed']
    )

    # Create evaluator and measure time
    evaluator = DAEEvaluator(model, device)
    _, metrics = evaluator.evaluate(original_data, corrupted_data, mask, noise_std)

    return metrics['imputation_time']


def update_metrics_file(metrics_path: str, imputation_times: list, seeds: list):
    """
    Update metrics JSON file with imputation timing information.

    Args:
        metrics_path: Path to metrics JSON file
        imputation_times: List of imputation times for each seed
        seeds: List of seeds
    """
    # Load existing metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Add timing information
    metrics['imputation_time_mean'] = float(np.mean(imputation_times))
    metrics['imputation_time_std'] = float(np.std(imputation_times, ddof=1))
    metrics['imputation_time_values'] = [float(t) for t in imputation_times]

    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Updated: {os.path.basename(metrics_path)}")
    print(f"    Imputation time: {metrics['imputation_time_mean']:.6f} ± {metrics['imputation_time_std']:.6f}s")


def main():
    """Main function to measure imputation times for all DAE models."""
    print("="*80)
    print("MEASURING DAE IMPUTATION TIMES")
    print("="*80)

    # Setup
    config = DAEConfig()
    device = get_device()

    # Load data
    print("\nLoading and preprocessing data...")
    preprocessor = FormulationDataPreprocessor(
        data_path=config.DATA_PATH,
        metadata_cols=config.METADATA_COLS
    )
    preprocessor.load_data()
    preprocessor.normalize_data()

    stats = preprocessor.get_data_statistics()
    print(f"Loaded {stats['n_formulations']:,} formulations with {stats['n_ingredients']:,} ingredients")

    # Find all model files
    model_dir = config.MODELS_DIR
    model_files = sorted(glob.glob(os.path.join(model_dir, '*.pt')))

    if not model_files:
        print(f"\nError: No model files found in {model_dir}")
        return

    print(f"\nFound {len(model_files)} model files")

    # Group models by configuration (excluding seed)
    configs = {}
    for model_file in model_files:
        params = parse_model_filename(model_file)
        config_key = f"miss{params['missingness_rate']}_lr{params['learning_rate']}_n{params['neuron_size']}_ep{params['epochs']}"

        if config_key not in configs:
            configs[config_key] = []
        configs[config_key].append((model_file, params['seed']))

    print(f"Processing {len(configs)} unique configurations...")

    # Measure times for each configuration
    for config_idx, (config_key, models) in enumerate(configs.items(), 1):
        print(f"\n[{config_idx}/{len(configs)}] {config_key}")

        imputation_times = []
        seeds = []

        for model_path, seed in sorted(models, key=lambda x: x[1]):
            print(f"  Measuring seed {seed}...", end=' ')

            try:
                imp_time = measure_model_imputation_time(
                    model_path,
                    preprocessor,
                    device,
                    noise_std=config.NOISE_STD
                )
                imputation_times.append(imp_time)
                seeds.append(seed)
                print(f"{imp_time:.6f}s")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        # Update metrics file if we have times
        if imputation_times:
            metrics_path = os.path.join(
                config.METRICS_DIR,
                f'{config_key}_metrics.json'
            )

            if os.path.exists(metrics_path):
                update_metrics_file(metrics_path, imputation_times, seeds)
            else:
                print(f"  Warning: Metrics file not found: {metrics_path}")

    print("\n" + "="*80)
    print("IMPUTATION TIME MEASUREMENT COMPLETE")
    print("="*80)
    print(f"\nUpdated metrics files in: {config.METRICS_DIR}/")
    print("All DAE metrics now include 'imputation_time_mean' and 'imputation_time_std'")
    print("\nYou can now regenerate comparison plots with actual DAE imputation times.")
    print("="*80)


if __name__ == "__main__":
    main()
