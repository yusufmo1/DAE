"""
DAE-specific visualization module.
Creates plots for DAE results using common visualization utilities.
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
    plot_loss_curves,
    load_predictions_from_file,
    load_metrics_from_file
)


def plot_dae_loss_curves(
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
    fig.suptitle(f'DAE Loss Curves - {missingness_rate*100:.0f}% Missing Data',
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
        ax.set_ylabel('Loss' if idx == 0 else '', fontweight='bold')
        ax.set_title(f'({chr(65+idx)}) LR={lr_label}',
                    fontweight='bold', loc='left')
        ax.legend(loc='best', fontsize=9)
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
        print(f"DAE loss curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dae_r2_bars(
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
    fig.suptitle(f'DAE R² Scores - {missingness_rate*100:.0f}% Missing Data',
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
        ax.set_title(f'({chr(65+idx)}) LR={lr_label}', fontweight='bold', loc='left')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(neuron_sizes)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DAE R² bar plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dae_predictions_vs_truth(
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

    subplot_titles = [f'({chr(65+idx)}) LR={lr_label}'
                      for idx, lr_label in enumerate(lr_labels)]

    plot_multi_predictions_vs_truth(
        predictions_data,
        num_cols=3,
        overall_title=f'DAE: Predicted vs Ground Truth - {missingness_rate*100:.0f}% Missing Data',
        subplot_titles=subplot_titles,
        save_path=save_path,
        show=show
    )


def load_dae_results_for_plotting(results_dir: str, missingness_rate: float) -> Tuple[Dict, Dict, Dict]:
    """
    Load all DAE results needed for plotting.

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


def generate_all_dae_plots(results_dir: str = 'results/dae',
                           missingness_rates: List[float] = [0.01, 0.05, 0.10]):
    """
    Create all DAE plots for the paper.

    Args:
        results_dir: Directory containing results
        missingness_rates: List of missingness rates to plot
    """
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("="*80)
    print("GENERATING DAE VISUALIZATIONS")
    print("="*80)

    for miss_rate in missingness_rates:
        print(f"\nCreating plots for {miss_rate*100:.0f}% missingness...")

        # Load data
        loss_data, r2_data, predictions_data = load_dae_results_for_plotting(
            results_dir, miss_rate
        )

        # Plot loss curves
        plot_dae_loss_curves(
            loss_data, miss_rate,
            save_path=os.path.join(plots_dir, f'loss_curves_{miss_rate*100:.0f}pct.png'),
            show=False
        )

        # Plot R² bars
        plot_dae_r2_bars(
            r2_data, miss_rate,
            save_path=os.path.join(plots_dir, f'r2_bars_{miss_rate*100:.0f}pct.png'),
            show=False
        )

        # Plot predictions vs truth (only for 1% missingness as in paper)
        if miss_rate == 0.01 and predictions_data:
            plot_dae_predictions_vs_truth(
                predictions_data, miss_rate,
                save_path=os.path.join(plots_dir, f'pred_vs_truth_{miss_rate*100:.0f}pct.png'),
                show=False
            )

    print(f"\n{'='*80}")
    print("DAE VISUALIZATIONS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll plots saved to: {plots_dir}/")
    print("  - loss_curves_*.png")
    print("  - r2_bars_*.png")
    print("  - pred_vs_truth_*.png")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    generate_all_dae_plots()
