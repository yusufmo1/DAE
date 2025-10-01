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
    missingness_rates: List[float] = [0.01, 0.05, 0.10],
    methods: List[str] = ['zero', 'mean', 'median']
):
    """
    Generate visualization plots for baseline methods.

    Args:
        results_dir: Directory containing baseline results
        missingness_rates: List of missingness rates to plot
        methods: List of baseline methods to plot (default: zero, mean, median)
    """
    print("="*80)
    print("GENERATING BASELINE VISUALIZATIONS")
    print("="*80)

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots for each method and missingness rate
    for method in methods:
        print(f"\nGenerating plots for {method} imputation...")

        for miss_rate in missingness_rates:
            # Load metrics
            metrics_file = os.path.join(
                results_dir, 'metrics',
                f'miss{miss_rate}_{method}_metrics.json'
            )
            metrics = load_metrics_from_file(metrics_file)

            if not metrics:
                print(f"  No metrics found for {method} at {miss_rate*100:.0f}%")
                continue

            # Load predictions
            pred_file = os.path.join(
                results_dir, 'predictions',
                f'miss{miss_rate}_seed42_{method}_predictions.npz'
            )

            # Zero imputation uses different naming (legacy)
            if method == 'zero' and not os.path.exists(pred_file):
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
                    title=f'{method.capitalize()} Imputation: Predicted vs Truth ({miss_rate*100:.0f}% Missing)',
                    save_path=os.path.join(plots_dir, f'{method}_pred_vs_truth_miss{miss_rate}.png'),
                    show=False,
                    add_r2=True
                )

            print(f"  {method} @ {miss_rate*100:.0f}% complete")

    # Generate summary comparison across missingness rates (all methods)
    all_r2_means = {}
    all_r2_stds = {}
    labels = [f'{miss_rate*100:.0f}%' for miss_rate in missingness_rates]

    for method in methods:
        r2_means = []
        r2_stds = []

        for miss_rate in missingness_rates:
            metrics_file = os.path.join(
                results_dir, 'metrics',
                f'miss{miss_rate}_{method}_metrics.json'
            )
            metrics = load_metrics_from_file(metrics_file)

            if metrics:
                r2_means.append(metrics.get('r2_mean', 0))
                r2_stds.append(metrics.get('r2_std', 0))
            else:
                r2_means.append(0)
                r2_stds.append(0)

        all_r2_means[f'{method.capitalize()} Imputation'] = r2_means
        all_r2_stds[f'{method.capitalize()} Imputation'] = r2_stds

    # Generate comparison plot with all methods
    if all_r2_means:
        plot_r2_comparison(
            all_r2_means,
            all_r2_stds,
            labels,
            'Baseline Methods Performance Across Missingness Rates',
            save_path=os.path.join(plots_dir, 'baseline_r2_summary.png'),
            show=False
        )

    print("\n" + "="*80)
    print("BASELINE VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nPlots saved to: {plots_dir}/")
    for method in methods:
        print(f"  - {method}_pred_vs_truth_miss*.png")
    print("  - baseline_r2_summary.png")
    print("="*80)


if __name__ == "__main__":
    generate_baseline_plots()
