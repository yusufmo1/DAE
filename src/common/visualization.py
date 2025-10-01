"""
Unified visualization utilities for imputation methods.
Provides common plotting functions and styling used across DAE, KNN, and baseline methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import json


# Set consistent style for all plots
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def plot_r2_comparison(
    r2_data: Dict[str, List[float]],
    std_data: Dict[str, List[float]],
    x_labels: List[str],
    title: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Generic R² comparison bar plot.

    Args:
        r2_data: Dictionary {method_name: [r2_values]}
        std_data: Dictionary {method_name: [std_values]}
        x_labels: Labels for x-axis
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    num_groups = len(x_labels)
    num_methods = len(r2_data)
    x_pos = np.arange(num_groups)
    width = 0.8 / num_methods

    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']

    for idx, (method, r2_vals) in enumerate(r2_data.items()):
        offset = (idx - (num_methods - 1) / 2) * width
        std_vals = std_data.get(method, [0] * len(r2_vals))

        ax.bar(x_pos + offset, r2_vals, width,
               yerr=std_vals, capsize=5,
               label=method, color=colors[idx % len(colors)], alpha=0.8)

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel(r'$R^2$ Score', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"R² comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions_vs_truth(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
    add_r2: bool = True
):
    """
    Generic scatter plot of predictions vs ground truth.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
        add_r2: Whether to add R² score to plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(targets, predictions,
               alpha=0.5, s=20,
               color='#1f77b4',
               edgecolors='none')

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', linewidth=1.5,
            label='Ideal Predictions',
            alpha=0.7)

    # Add R² score if requested
    if add_r2:
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, predictions)
        ax.text(0.05, 0.95, r'$R^2$ = ' + f'{r2:.4f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Ground Truth', fontweight='bold')
    ax.set_ylabel('Predicted', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
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


def plot_multi_predictions_vs_truth(
    predictions_dict: Dict[str, Dict[str, np.ndarray]],
    num_cols: int,
    overall_title: str,
    subplot_titles: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
    use_shared_legend: bool = True
):
    """
    Create a grid of prediction vs truth plots for multiple methods/configs.

    Args:
        predictions_dict: Dictionary {method: {'predictions': arr, 'targets': arr}}
        num_cols: Number of columns in subplot grid
        overall_title: Overall figure title
        subplot_titles: List of subplot titles
        save_path: Path to save figure
        show: Whether to display the plot
        use_shared_legend: If True, use 2x2 grid with shared legend in bottom-right
    """
    num_plots = len(predictions_dict)

    # Use 2x2 layout with shared legend in bottom-right if requested
    if use_shared_legend and num_plots == 3:
        num_rows, num_cols = 2, 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
        axes = axes.flatten()
    else:
        num_rows = (num_plots + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    fig.suptitle(overall_title, fontsize=14, fontweight='bold')

    for idx, (method, data) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        predictions = data['predictions']
        targets = data['targets']

        # Scatter plot
        ax.scatter(targets, predictions, alpha=0.6, s=30,
                   color='#1f77b4', edgecolors='none',
                   label='Individual Predictions')

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=1.5, alpha=0.7,
                label='Ideal Predictions')

        # Add R²
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, predictions)
        ax.text(0.05, 0.95, r'$R^2$ = ' + f'{r2:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Ground Truth', fontweight='bold')
        ax.set_ylabel('Predicted' if idx % num_cols == 0 else '', fontweight='bold')
        ax.set_title(subplot_titles[idx] if idx < len(subplot_titles) else method,
                     fontweight='bold', loc='left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Only add legend to individual plots if not using shared legend
        if not use_shared_legend:
            ax.legend(loc='upper left', fontsize=9)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Handle shared legend in bottom-right subplot for 2x2 layout
    if use_shared_legend and num_plots == 3 and len(axes) == 4:
        # Turn off the bottom-right subplot and use it for legend
        axes[3].axis('off')

        # Create legend handles manually
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                   markersize=8, alpha=0.6, label='Individual Predictions'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                   alpha=0.7, label='Ideal Predictions')
        ]

        axes[3].legend(handles=legend_elements, loc='center', fontsize=11,
                      frameon=True, fancybox=True, shadow=True)
    else:
        # Hide extra subplots
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi predictions plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_curves(
    loss_dict: Dict[str, List[float]],
    title: str,
    xlabel: str = 'Epochs',
    ylabel: str = 'Loss',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Generic loss curve plotting.

    Args:
        loss_dict: Dictionary {label: loss_history}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (label, loss_history) in enumerate(loss_dict.items()):
        epochs = range(1, len(loss_history) + 1)
        ax.plot(epochs, loss_history,
                color=colors[idx % len(colors)],
                linewidth=1.5,
                label=label,
                alpha=0.8)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_comparison_table(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create a visual table comparing metrics across methods.

    Args:
        metrics_dict: Dictionary {method: {metric_name: value}}
        metric_names: List of metric names to display
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    methods = list(metrics_dict.keys())
    table_data = []

    for method in methods:
        row = [method]
        for metric in metric_names:
            value = metrics_dict[method].get(metric, np.nan)
            if isinstance(value, float):
                row.append(f'{value:.4f}')
            else:
                row.append(str(value))
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Method'] + metric_names,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(metric_names) + 1):
        table[(0, i)].set_facecolor('#2ca02c')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title(title, fontweight='bold', fontsize=14, pad=20)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metric comparison table saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_metrics_from_file(metrics_file: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.

    Args:
        metrics_file: Path to metrics JSON file

    Returns:
        Dictionary of metrics
    """
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found: {metrics_file}")
        return {}

    with open(metrics_file, 'r') as f:
        return json.load(f)


def load_predictions_from_file(predictions_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions from NPZ file.

    Args:
        predictions_file: Path to predictions NPZ file

    Returns:
        Tuple of (predictions, targets)
    """
    if not os.path.exists(predictions_file):
        print(f"Warning: Predictions file not found: {predictions_file}")
        return np.array([]), np.array([])

    data = np.load(predictions_file)
    return data['predictions'], data['targets']


if __name__ == "__main__":
    print("This module provides common visualization utilities.")
    print("Import and use in method-specific visualization modules.")
