"""
KNN-specific visualization module.
Creates plots for KNN results using common visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.visualization import (
    plot_predictions_vs_truth,
    plot_multi_predictions_vs_truth,
    plot_r2_comparison,
    load_predictions_from_file,
    load_metrics_from_file
)


def plot_knn_r2_bars(
    r2_data: Dict,
    missingness_rate: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot R² scores for different KNN configurations.

    Args:
        r2_data: Dictionary with structure:
                 {metric: {k: {weights: {'mean': float, 'std': float}}}}
        missingness_rate: Missingness rate (for title)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    metric_labels = ['Euclidean', 'Manhattan', 'Cosine']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'KNN R² Scores - {missingness_rate*100:.0f}% Missing Data',
                 fontsize=14, fontweight='bold')

    k_values = [3, 5, 10, 20, 50]
    colors = {'uniform': '#1f77b4', 'distance': '#ff7f0e'}
    x = np.arange(len(k_values))
    width = 0.35

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        if metric not in r2_data:
            continue

        uniform_means = []
        uniform_stds = []
        distance_means = []
        distance_stds = []

        for k in k_values:
            if k in r2_data[metric]:
                if 'uniform' in r2_data[metric][k]:
                    uniform_means.append(r2_data[metric][k]['uniform']['mean'])
                    uniform_stds.append(r2_data[metric][k]['uniform']['std'])
                else:
                    uniform_means.append(0)
                    uniform_stds.append(0)

                if 'distance' in r2_data[metric][k]:
                    distance_means.append(r2_data[metric][k]['distance']['mean'])
                    distance_stds.append(r2_data[metric][k]['distance']['std'])
                else:
                    distance_means.append(0)
                    distance_stds.append(0)
            else:
                uniform_means.append(0)
                uniform_stds.append(0)
                distance_means.append(0)
                distance_stds.append(0)

        # Plot bars
        ax.bar(x - width/2, uniform_means, width, yerr=uniform_stds,
               label='Uniform', capsize=5, color=colors['uniform'], alpha=0.8)
        ax.bar(x + width/2, distance_means, width, yerr=distance_stds,
               label='Distance', capsize=5, color=colors['distance'], alpha=0.8)

        ax.set_xlabel('K (Neighbors)', fontweight='bold')
        ax.set_ylabel('R² Score' if idx == 0 else '', fontweight='bold')
        ax.set_title(f'({chr(65+idx)}) {label} Distance', fontweight='bold', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KNN R² bar plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_knn_predictions_vs_truth(
    predictions_data: Dict,
    missingness_rate: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot predictions vs ground truth for different distance metrics.

    Args:
        predictions_data: Dictionary with structure:
                         {metric: {'predictions': array, 'targets': array}}
        missingness_rate: Missingness rate (for title)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    metric_labels = ['Euclidean', 'Manhattan', 'Cosine']

    subplot_titles = [f'({chr(65+idx)}) {label}'
                      for idx, label in enumerate(metric_labels)]

    plot_multi_predictions_vs_truth(
        predictions_data,
        num_cols=3,
        overall_title=f'KNN Predictions vs Truth - {missingness_rate*100:.0f}% Missing Data',
        subplot_titles=subplot_titles,
        save_path=save_path,
        show=show
    )


def plot_knn_performance_summary(
    r2_data: Dict,
    time_data: Dict,
    missingness_rate: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot performance summary: R² vs K for different metrics.

    Args:
        r2_data: R² scores by metric/k/weights
        time_data: Imputation times by metric/k/weights
        missingness_rate: Missingness rate
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'KNN Performance Summary - {missingness_rate*100:.0f}% Missing Data',
                 fontsize=14, fontweight='bold')

    metrics = ['euclidean', 'manhattan', 'cosine']
    metric_labels = ['Euclidean', 'Manhattan', 'Cosine']
    colors = {'euclidean': '#1f77b4', 'manhattan': '#ff7f0e', 'cosine': '#2ca02c'}
    k_values = [3, 5, 10, 20, 50]

    # Plot 1: R² vs K (distance-weighted)
    for metric, label in zip(metrics, metric_labels):
        if metric in r2_data:
            r2_vals = []
            for k in k_values:
                if k in r2_data[metric] and 'distance' in r2_data[metric][k]:
                    r2_vals.append(r2_data[metric][k]['distance']['mean'])
                else:
                    r2_vals.append(0)

            ax1.plot(k_values, r2_vals, marker='o', linewidth=2,
                    label=label, color=colors[metric])

    ax1.set_xlabel('K (Neighbors)', fontweight='bold')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('(A) R² vs K (Distance-Weighted)', fontweight='bold', loc='left')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xscale('log')

    # Plot 2: Imputation time vs K
    for metric, label in zip(metrics, metric_labels):
        if metric in time_data:
            time_vals = []
            for k in k_values:
                if k in time_data[metric] and 'distance' in time_data[metric][k]:
                    time_vals.append(time_data[metric][k]['distance']['mean'])
                else:
                    time_vals.append(0)

            ax2.plot(k_values, time_vals, marker='s', linewidth=2,
                    label=label, color=colors[metric])

    ax2.set_xlabel('K (Neighbors)', fontweight='bold')
    ax2.set_ylabel('Imputation Time (s)', fontweight='bold')
    ax2.set_title('(B) Computational Cost vs K', fontweight='bold', loc='left')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KNN performance summary plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_knn_results_for_plots(
    results_dir: str = 'results/knn',
    missingness_rate: float = 0.01
) -> Tuple[Dict, Dict, Dict]:
    """
    Load KNN results and organize for plotting.

    Args:
        results_dir: Directory containing KNN results
        missingness_rate: Missingness rate to load

    Returns:
        Tuple of (r2_data, time_data, predictions_data)
    """
    r2_data = {}
    time_data = {}
    predictions_data = {}

    metrics_list = ['euclidean', 'manhattan', 'cosine']
    k_values = [3, 5, 10, 20, 50]
    weights_list = ['uniform', 'distance']

    # Load R² and time data
    for metric in metrics_list:
        r2_data[metric] = {}
        time_data[metric] = {}

        for k in k_values:
            r2_data[metric][k] = {}
            time_data[metric][k] = {}

            for weights in weights_list:
                metrics_file = os.path.join(
                    results_dir, 'metrics',
                    f'miss{missingness_rate}_k{k}_{weights}_{metric}_metrics.json'
                )

                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        r2_data[metric][k][weights] = {
                            'mean': data.get('r2_mean', 0),
                            'std': data.get('r2_std', 0)
                        }
                        time_data[metric][k][weights] = {
                            'mean': data.get('imputation_time_mean', 0),
                            'std': data.get('imputation_time_std', 0)
                        }

    # Load prediction data for best configs
    for metric in metrics_list:
        # Use K=20, distance-weighted (typically best performing)
        pred_file = os.path.join(
            results_dir, 'predictions',
            f'miss{missingness_rate}_k20_distance_{metric}_predictions.npz'
        )

        if os.path.exists(pred_file):
            data = np.load(pred_file)
            predictions_data[metric] = {
                'predictions': data['predictions'],
                'targets': data['targets']
            }

    return r2_data, time_data, predictions_data


def generate_all_knn_plots(
    results_dir: str = 'results/knn',
    missingness_rates: List[float] = [0.01, 0.05, 0.10]
):
    """
    Generate all KNN visualization plots for each missingness rate.

    Args:
        results_dir: Directory containing KNN results
        missingness_rates: List of missingness rates to plot
    """
    print("="*80)
    print("GENERATING KNN VISUALIZATIONS")
    print("="*80)

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for miss_rate in missingness_rates:
        print(f"\nGenerating plots for {miss_rate*100:.0f}% missingness...")

        # Load data
        r2_data, time_data, predictions_data = load_knn_results_for_plots(
            results_dir, miss_rate
        )

        # Generate plots
        plot_knn_r2_bars(
            r2_data,
            miss_rate,
            save_path=os.path.join(plots_dir, f'knn_r2_bars_miss{miss_rate}.png'),
            show=False
        )

        if predictions_data:
            plot_knn_predictions_vs_truth(
                predictions_data,
                miss_rate,
                save_path=os.path.join(plots_dir, f'knn_predictions_miss{miss_rate}.png'),
                show=False
            )

        plot_knn_performance_summary(
            r2_data,
            time_data,
            miss_rate,
            save_path=os.path.join(plots_dir, f'knn_summary_miss{miss_rate}.png'),
            show=False
        )

    print(f"\n{'='*80}")
    print("KNN VISUALIZATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll plots saved to: {plots_dir}/")
    print("  - knn_r2_bars_miss*.png")
    print("  - knn_predictions_miss*.png")
    print("  - knn_summary_miss*.png")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    generate_all_knn_plots()
