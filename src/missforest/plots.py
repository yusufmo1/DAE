"""
MissForest-specific visualization module.
Creates plots for MissForest results using common visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.visualization import (
    plot_predictions_vs_truth,
    plot_r2_comparison,
    load_predictions_from_file,
    load_metrics_from_file
)


def plot_missforest_r2_bars(
    results: Dict,
    missingness_rates: List[float],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot R² scores across different missingness rates for MissForest.

    Args:
        results: Dictionary with results for all configs
        missingness_rates: List of missingness rates
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    means = []
    stds = []
    labels = []

    for miss_rate in missingness_rates:
        # Find matching config
        matching_keys = [k for k in results.keys() if f"miss{miss_rate}" in k]
        if matching_keys:
            key = matching_keys[0]
            means.append(results[key]['r2_mean'])
            stds.append(results[key]['r2_std'])
            labels.append(f"{miss_rate*100:.0f}%")

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color='#2ca02c', alpha=0.8, label='MissForest')

    ax.set_xlabel('Missingness Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel(r'$R^2$ Score', fontweight='bold', fontsize=12)
    ax.set_title('MissForest Performance Across Missingness Rates',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved R² bar plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_missforest_predictions(
    results_dir: str,
    missingness_rates: List[float],
    save_dir: Optional[str] = None,
    show: bool = False
):
    """
    Plot predicted vs. truth scatter plots for each missingness rate.

    Args:
        results_dir: Directory containing results
        missingness_rates: List of missingness rates
        save_dir: Directory to save figures
        show: Whether to display plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for miss_rate in missingness_rates:
        # Find matching prediction file
        pred_dir = os.path.join(results_dir, 'predictions')
        pred_files = [f for f in os.listdir(pred_dir) if f"miss{miss_rate}" in f and f.endswith('.npz')]

        if not pred_files:
            print(f"No prediction file found for missingness rate {miss_rate}")
            continue

        pred_file = os.path.join(pred_dir, pred_files[0])
        predictions, targets = load_predictions_from_file(pred_file)

        # Load metrics for R²
        metrics_dir = os.path.join(results_dir, 'metrics')
        metrics_files = [f for f in os.listdir(metrics_dir) if f"miss{miss_rate}" in f and f.endswith('_metrics.json')]
        if metrics_files:
            metrics_file = os.path.join(metrics_dir, metrics_files[0])
            metrics = load_metrics_from_file(metrics_file)
            r2_mean = metrics['r2_mean']
            r2_std = metrics['r2_std']
        else:
            r2_mean = None
            r2_std = None

        # Create plot
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'miss{miss_rate}_predictions.png')

        plot_predictions_vs_truth(
            predictions=predictions,
            targets=targets,
            title=f'MissForest: {miss_rate*100:.0f}% Missing Data',
            save_path=save_path,
            show=show
        )


def generate_all_missforest_plots(
    results_dir: str,
    missingness_rates: List[float],
    show: bool = False
):
    """
    Generate all MissForest visualization plots.

    Args:
        results_dir: Directory containing results
        missingness_rates: List of missingness rates to plot
        show: Whether to display plots
    """
    print("\nGenerating MissForest plots...")

    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Load summary results
    summary_file = os.path.join(results_dir, 'summary.json')
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found at {summary_file}")
        return

    with open(summary_file, 'r') as f:
        results = json.load(f)

    # 1. R² bar chart across missingness rates
    print("  - Creating R² bar chart...")
    r2_plot_path = os.path.join(plots_dir, 'missforest_r2_bars.png')
    plot_missforest_r2_bars(
        results=results,
        missingness_rates=missingness_rates,
        save_path=r2_plot_path,
        show=show
    )

    # 2. Predicted vs. truth scatter plots
    print("  - Creating prediction scatter plots...")
    plot_missforest_predictions(
        results_dir=results_dir,
        missingness_rates=missingness_rates,
        save_dir=plots_dir,
        show=show
    )

    print(f"\nAll MissForest plots saved to {plots_dir}")


if __name__ == "__main__":
    print("This module provides MissForest visualization utilities.")
    print("Use generate_all_missforest_plots() to create all plots.")
