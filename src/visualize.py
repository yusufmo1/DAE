"""
Visualization module for DAE results.
Creates plots matching the paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import json


# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def plot_loss_curves(
    loss_data: Dict,
    missingness_rate: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot loss curves for different learning rates (Figure 3 in paper).

    Args:
        loss_data: Dictionary with structure:
                   {lr: {neuron_size: {epochs: loss_history}}}
        missingness_rate: Missingness rate (for title)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    learning_rates = [1e-1, 1e-3, 1e-5]
    lr_labels = ['10⁻¹', '10⁻³', '10⁻⁵']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Loss Curves - {missingness_rate*100:.0f}% Missing Data',
                 fontsize=14, fontweight='bold')

    colors = {256: '#1f77b4', 512: '#ff7f0e', 1024: '#2ca02c'}
    neuron_sizes = [256, 512, 1024]

    for idx, (lr, lr_label) in enumerate(zip(learning_rates, lr_labels)):
        ax = axes[idx]

        if lr in loss_data:
            for neuron_size in neuron_sizes:
                if neuron_size in loss_data[lr]:
                    # Get longest training run (1200 epochs)
                    epochs_list = sorted(loss_data[lr][neuron_size].keys())
                    if not epochs_list:
                        continue
                    longest_run = epochs_list[-1]
                    loss_history = loss_data[lr][neuron_size][longest_run]

                    epochs = range(1, len(loss_history) + 1)
                    ax.plot(epochs, loss_history,
                           color=colors[neuron_size],
                           linewidth=1.5,
                           label=f'{neuron_size} neurons',
                           alpha=0.8)

        ax.set_xlabel('Epochs', fontweight='bold')
        ax.set_ylabel('R²' if idx == 0 else '', fontweight='bold')
        ax.set_title(f'({chr(65+idx)})      Learning Rate: {lr_label}',
                    fontweight='bold', loc='left')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on data
        if lr in loss_data and loss_data[lr]:
            all_losses = []
            for ns in neuron_sizes:
                if ns in loss_data[lr]:
                    for ep in loss_data[lr][ns]:
                        all_losses.extend(loss_data[lr][ns][ep])
            if all_losses:
                y_min = min(all_losses)
                y_max = max(all_losses)
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_r2_bars(
    results_data: Dict,
    missingness_rate: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot R² scores as bar charts (Figures 4-6 in paper).

    Args:
        results_data: Dictionary with structure:
                      {lr: {neuron_size: {epochs: {'mean': x, 'std': y}}}}
        missingness_rate: Missingness rate (for title)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    learning_rates = [1e-1, 1e-3, 1e-5]
    lr_labels = ['10⁻¹', '10⁻³', '10⁻⁵']
    neuron_sizes = [256, 512, 1024]
    epoch_list = [100, 500, 1000, 1200]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'R² Scores - {missingness_rate*100:.0f}% Missing Data',
                 fontsize=14, fontweight='bold')

    # Colors for different epochs
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#17becf']
    bar_width = 0.2
    x_pos = np.arange(len(neuron_sizes))

    for idx, (lr, lr_label) in enumerate(zip(learning_rates, lr_labels)):
        ax = axes[idx]

        if lr in results_data:
            for ep_idx, epochs in enumerate(epoch_list):
                means = []
                stds = []

                for neuron_size in neuron_sizes:
                    if (neuron_size in results_data[lr] and
                        epochs in results_data[lr][neuron_size]):
                        means.append(results_data[lr][neuron_size][epochs]['mean'])
                        stds.append(results_data[lr][neuron_size][epochs]['std'])
                    else:
                        means.append(0)
                        stds.append(0)

                offset = (ep_idx - 1.5) * bar_width
                ax.bar(x_pos + offset, means, bar_width,
                      yerr=stds, capsize=3,
                      color=colors[ep_idx],
                      label=f'{epochs} epochs',
                      alpha=0.8)

        ax.set_xlabel('Neurone size', fontweight='bold')
        ax.set_ylabel('R²' if idx == 0 else '', fontweight='bold')
        ax.set_title(f'({chr(65+idx)})', fontweight='bold', loc='left')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(neuron_sizes)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"R² bar plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions_vs_truth(
    predictions_data: Dict,
    missingness_rate: float,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot predicted vs ground truth scatter plots (Figure 7 in paper).

    Args:
        predictions_data: Dictionary with structure:
                          {lr: {'predictions': array, 'targets': array}}
        missingness_rate: Missingness rate (for title)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    learning_rates = [1e-1, 1e-3, 1e-5]
    lr_labels = ['10⁻¹', '10⁻³', '10⁻⁵']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Predicted vs Ground Truth - {missingness_rate*100:.0f}% Missing Data',
                 fontsize=14, fontweight='bold')

    for idx, (lr, lr_label) in enumerate(zip(learning_rates, lr_labels)):
        ax = axes[idx]

        if lr in predictions_data:
            predictions = predictions_data[lr]['predictions']
            targets = predictions_data[lr]['targets']

            # Plot scatter
            ax.scatter(targets, predictions,
                      alpha=0.5, s=20,
                      color='#1f77b4',
                      label='Individual Predictions',
                      edgecolors='none')

            # Plot diagonal line (ideal predictions)
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                   'k--', linewidth=1.5,
                   label='Ideal Predictions',
                   alpha=0.7)

        ax.set_xlabel('Ground Truth', fontweight='bold')
        ax.set_ylabel('Predicted' if idx == 0 else '', fontweight='bold')
        ax.set_title(f'({chr(65+idx)})', fontweight='bold', loc='left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions vs truth plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_results_for_plotting(results_dir: str, missingness_rate: float) -> Tuple[Dict, Dict, Dict]:
    """
    Load all results needed for plotting.

    Args:
        results_dir: Directory containing results
        missingness_rate: Missingness rate (0.01, 0.05, 0.10)

    Returns:
        Tuple of (loss_data, r2_data, predictions_data)
    """
    loss_data = {}
    r2_data = {}
    predictions_data = {}

    learning_rates = [1e-1, 1e-3, 1e-5]
    neuron_sizes = [256, 512, 1024]
    epoch_list = [100, 500, 1000, 1200]

    for lr in learning_rates:
        loss_data[lr] = {}
        r2_data[lr] = {}

        for neuron_size in neuron_sizes:
            loss_data[lr][neuron_size] = {}
            r2_data[lr][neuron_size] = {}

            for epochs in epoch_list:
                # Load loss history
                loss_file = os.path.join(
                    results_dir, 'metrics',
                    f'miss{missingness_rate}_lr{lr}_n{neuron_size}_ep{epochs}_loss.json'
                )
                if os.path.exists(loss_file):
                    with open(loss_file, 'r') as f:
                        loss_data[lr][neuron_size][epochs] = json.load(f)

                # Load R² metrics
                metrics_file = os.path.join(
                    results_dir, 'metrics',
                    f'miss{missingness_rate}_lr{lr}_n{neuron_size}_ep{epochs}_metrics.json'
                )
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        r2_data[lr][neuron_size][epochs] = {
                            'mean': metrics.get('r2_mean', 0),
                            'std': metrics.get('r2_std', 0)
                        }

        # Load predictions for one configuration (e.g., 1024 neurons, 1200 epochs)
        pred_file = os.path.join(
            results_dir, 'metrics',
            f'miss{missingness_rate}_lr{lr}_n1024_ep1200_predictions.npz'
        )
        if os.path.exists(pred_file):
            data = np.load(pred_file)
            predictions_data[lr] = {
                'predictions': data['predictions'],
                'targets': data['targets']
            }

    return loss_data, r2_data, predictions_data


def create_all_plots(results_dir: str, missingness_rates: List[float] = [0.01, 0.05, 0.10]):
    """
    Create all plots for the paper.

    Args:
        results_dir: Directory containing results
        missingness_rates: List of missingness rates to plot
    """
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for miss_rate in missingness_rates:
        print(f"\nCreating plots for {miss_rate*100:.0f}% missingness...")

        # Load data
        loss_data, r2_data, predictions_data = load_results_for_plotting(
            results_dir, miss_rate
        )

        # Plot loss curves
        plot_loss_curves(
            loss_data, miss_rate,
            save_path=os.path.join(plots_dir, f'loss_curves_{miss_rate*100:.0f}pct.png'),
            show=False
        )

        # Plot R² bars
        plot_r2_bars(
            r2_data, miss_rate,
            save_path=os.path.join(plots_dir, f'r2_bars_{miss_rate*100:.0f}pct.png'),
            show=False
        )

        # Plot predictions vs truth (only for 1% missingness as in paper)
        if miss_rate == 0.01 and predictions_data:
            plot_predictions_vs_truth(
                predictions_data, miss_rate,
                save_path=os.path.join(plots_dir, f'pred_vs_truth_{miss_rate*100:.0f}pct.png'),
                show=False
            )

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    print("This module provides visualization utilities.")
    print("Use main.py to run full experiments and generate plots.")
