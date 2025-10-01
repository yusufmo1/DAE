"""
Visualization utilities for baseline imputation methods.
Uses common visualization utilities from src.common.visualization.
"""

import numpy as np
import os
import json
from typing import Dict, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.visualization import (
    plot_predictions_vs_truth,
    plot_r2_comparison,
    load_predictions_from_file,
    load_metrics_from_file
)


def generate_baseline_plots(
    results_dir: str = 'results/baselines',
    missingness_rates: List[float] = [0.01, 0.05, 0.10]
):
    """
    Generate visualization plots for baseline methods.

    Args:
        results_dir: Directory containing baseline results
        missingness_rates: List of missingness rates to plot
    """
    print("="*80)
    print("GENERATING BASELINE VISUALIZATIONS")
    print("="*80)

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for miss_rate in missingness_rates:
        print(f"\nGenerating plots for {miss_rate*100:.0f}% missingness...")

        # Load metrics
        metrics_file = os.path.join(
            results_dir, 'metrics',
            f'miss{miss_rate}_zero_metrics.json'
        )
        metrics = load_metrics_from_file(metrics_file)

        if not metrics:
            print(f"  No metrics found for {miss_rate}")
            continue

        # Load predictions
        pred_file = os.path.join(
            results_dir, 'predictions',
            f'miss{miss_rate}_seed42_predictions.npz'
        )

        if os.path.exists(pred_file):
            predictions, targets = load_predictions_from_file(pred_file)

            # Plot predictions vs truth
            plot_predictions_vs_truth(
                predictions,
                targets,
                title=f'Zero Imputation: Predicted vs Truth ({miss_rate*100:.0f}% Missing)',
                save_path=os.path.join(plots_dir, f'zero_pred_vs_truth_miss{miss_rate}.png'),
                show=False,
                add_r2=True
            )

        print(f"  Plots saved for {miss_rate*100:.0f}%")

    # Generate summary comparison across missingness rates
    r2_means = []
    r2_stds = []
    labels = []

    for miss_rate in missingness_rates:
        metrics_file = os.path.join(
            results_dir, 'metrics',
            f'miss{miss_rate}_zero_metrics.json'
        )
        metrics = load_metrics_from_file(metrics_file)

        if metrics:
            r2_means.append(metrics.get('r2_mean', 0))
            r2_stds.append(metrics.get('r2_std', 0))
            labels.append(f'{miss_rate*100:.0f}%')

    if r2_means:
        plot_r2_comparison(
            {'Zero Imputation': r2_means},
            {'Zero Imputation': r2_stds},
            labels,
            'Zero Imputation Performance Across Missingness Rates',
            save_path=os.path.join(plots_dir, 'zero_r2_summary.png'),
            show=False
        )

    print("\n" + "="*80)
    print("BASELINE VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nPlots saved to: {plots_dir}/")
    print("  - zero_pred_vs_truth_miss*.png")
    print("  - zero_r2_summary.png")
    print("="*80)


if __name__ == "__main__":
    generate_baseline_plots()
