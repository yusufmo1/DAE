"""
Comparison visualizations between DAE and KNN imputation methods.
Generates plots comparing neural network vs. classical ML approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Optional


# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


def load_dae_results(results_dir: str = 'results') -> Dict:
    """
    Load DAE results from summary file.

    Args:
        results_dir: Directory containing DAE results

    Returns:
        Dictionary with DAE results
    """
    summary_file = os.path.join(results_dir, 'summary.json')

    if not os.path.exists(summary_file):
        print(f"Warning: DAE results not found at {summary_file}")
        return {}

    with open(summary_file, 'r') as f:
        results = json.load(f)

    return results


def load_knn_results(results_dir: str = 'knn_results') -> Dict:
    """
    Load KNN results from summary file.

    Args:
        results_dir: Directory containing KNN results

    Returns:
        Dictionary with KNN results
    """
    summary_file = os.path.join(results_dir, 'summary.json')

    if not os.path.exists(summary_file):
        print(f"Warning: KNN results not found at {summary_file}")
        return {}

    with open(summary_file, 'r') as f:
        results = json.load(f)

    return results


def get_best_result(results: Dict, missingness_rate: float, metric: str = 'r2') -> Tuple[str, float, float]:
    """
    Get best result for a given missingness rate.

    Args:
        results: Results dictionary
        missingness_rate: Missingness rate to filter by
        metric: Metric to optimize

    Returns:
        Tuple of (config_name, mean_value, std_value)
    """
    best_config = None
    best_mean = -np.inf if metric == 'r2' else np.inf
    best_std = 0

    for config, metrics in results.items():
        if f"miss{missingness_rate}" in config:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'

            if mean_key in metrics:
                mean_val = metrics[mean_key]

                # Check if better
                if (metric == 'r2' and mean_val > best_mean) or \
                   (metric != 'r2' and mean_val < best_mean):
                    best_config = config
                    best_mean = mean_val
                    best_std = metrics.get(std_key, 0)

    return best_config, best_mean, best_std


def plot_r2_comparison(
    dae_results: Dict,
    knn_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot R² comparison between DAE and KNN.

    Args:
        dae_results: DAE results dictionary
        knn_results: KNN results dictionary
        save_path: Path to save figure
        show: Whether to display plot
    """
    missingness_rates = [0.01, 0.05, 0.10]
    x_pos = np.arange(len(missingness_rates))
    width = 0.35

    dae_means = []
    dae_stds = []
    knn_means = []
    knn_stds = []

    for miss_rate in missingness_rates:
        # Get best DAE result
        _, dae_mean, dae_std = get_best_result(dae_results, miss_rate, 'r2')
        dae_means.append(dae_mean)
        dae_stds.append(dae_std)

        # Get best KNN result
        _, knn_mean, knn_std = get_best_result(knn_results, miss_rate, 'r2')
        knn_means.append(knn_mean)
        knn_stds.append(knn_std)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(x_pos - width/2, dae_means, width, yerr=dae_stds,
           label='DAE (Neural Network)', capsize=5, color='#2ca02c', alpha=0.8)
    ax.bar(x_pos + width/2, knn_means, width, yerr=knn_stds,
           label='KNN (Classical ML)', capsize=5, color='#ff7f0e', alpha=0.8)

    ax.set_xlabel('Missingness Rate', fontweight='bold')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('DAE vs KNN: Imputation Performance Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{mr*100:.0f}%' for mr in missingness_rates])
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


def plot_performance_vs_time(
    dae_results: Dict,
    knn_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot performance vs. computational time trade-off.

    Args:
        dae_results: DAE results dictionary
        knn_results: KNN results dictionary
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Performance vs. Computational Time Trade-off', fontweight='bold', fontsize=14)

    missingness_rates = [0.01, 0.05, 0.10]

    for idx, miss_rate in enumerate(missingness_rates):
        ax = axes[idx]

        # Get all DAE results for this missingness
        dae_r2 = []
        dae_time = []
        for config, metrics in dae_results.items():
            if f"miss{miss_rate}" in config and 'r2_mean' in metrics:
                dae_r2.append(metrics['r2_mean'])
                # DAE trains, so time is significant (use epochs as proxy)
                dae_time.append(100)  # Placeholder - actual time not stored

        # Get all KNN results for this missingness
        knn_r2 = []
        knn_time = []
        for config, metrics in knn_results.items():
            if f"miss{miss_rate}" in config and 'r2_mean' in metrics:
                knn_r2.append(metrics['r2_mean'])
                knn_time.append(metrics.get('imputation_time_mean', 1))

        # Plot
        if dae_r2:
            ax.scatter(dae_time, dae_r2, s=50, alpha=0.6, color='#2ca02c', label='DAE')
        if knn_r2:
            ax.scatter(knn_time, knn_r2, s=50, alpha=0.6, color='#ff7f0e', label='KNN')

        ax.set_xlabel('Time (log scale)', fontweight='bold')
        ax.set_ylabel('R² Score' if idx == 0 else '', fontweight='bold')
        ax.set_title(f'({chr(65+idx)}) {miss_rate*100:.0f}% Missing', fontweight='bold', loc='left')
        ax.set_xscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance vs. time plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_comparison_grid(
    dae_predictions: Dict,
    knn_predictions: Dict,
    missingness_rate: float = 0.01,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot prediction quality comparison grid (DAE vs KNN).

    Args:
        dae_predictions: Dictionary with DAE predictions
        knn_predictions: Dictionary with KNN predictions
        missingness_rate: Missingness rate to plot
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Prediction Quality: DAE vs KNN ({missingness_rate*100:.0f}% Missing)',
                 fontweight='bold', fontsize=14)

    # This would require loading actual prediction arrays from .npz files
    # For now, create placeholder
    print("Note: Detailed prediction comparison requires loading .npz files")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison grid saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_comparison_table(
    dae_results: Dict,
    knn_results: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Generate text table comparing best DAE and KNN results.

    Args:
        dae_results: DAE results dictionary
        knn_results: KNN results dictionary
        save_path: Path to save table

    Returns:
        Formatted table string
    """
    table = []
    table.append("="*100)
    table.append("METHOD COMPARISON: DAE vs KNN")
    table.append("="*100)
    table.append("")
    table.append(f"{'Missingness':<15} {'Method':<10} {'R²':<20} {'RMSE':<20} {'Config':<35}")
    table.append("-"*100)

    missingness_rates = [0.01, 0.05, 0.10]

    for miss_rate in missingness_rates:
        # DAE best
        dae_config, dae_r2, dae_r2_std = get_best_result(dae_results, miss_rate, 'r2')
        _, dae_rmse, dae_rmse_std = get_best_result(dae_results, miss_rate, 'rmse')

        if dae_config:
            table.append(f"{f'{miss_rate*100:.0f}%':<15} {'DAE':<10} "
                        f"{f'{dae_r2:.4f}±{dae_r2_std:.4f}':<20} "
                        f"{f'{dae_rmse:.4f}±{dae_rmse_std:.4f}':<20} "
                        f"{dae_config[:35]:<35}")

        # KNN best
        knn_config, knn_r2, knn_r2_std = get_best_result(knn_results, miss_rate, 'r2')
        _, knn_rmse, knn_rmse_std = get_best_result(knn_results, miss_rate, 'rmse')

        if knn_config:
            table.append(f"{'':<15} {'KNN':<10} "
                        f"{f'{knn_r2:.4f}±{knn_r2_std:.4f}':<20} "
                        f"{f'{knn_rmse:.4f}±{knn_rmse_std:.4f}':<20} "
                        f"{knn_config[:35]:<35}")

        # Winner
        if dae_config and knn_config:
            winner = "DAE" if dae_r2 > knn_r2 else "KNN"
            improvement = abs(dae_r2 - knn_r2) / max(abs(knn_r2), 0.0001) * 100
            table.append(f"{'':<15} {f'Winner: {winner} (+{improvement:.1f}%)':<45}")

        table.append("")

    table.append("="*100)

    table_str = "\n".join(table)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"Comparison table saved to {save_path}")

    return table_str


def generate_all_comparisons(
    dae_results_dir: str = 'results',
    knn_results_dir: str = 'knn_results',
    output_dir: str = 'compare_results'
):
    """
    Generate all comparison visualizations and tables.

    Args:
        dae_results_dir: Directory with DAE results
        knn_results_dir: Directory with KNN results
        output_dir: Directory to save comparisons
    """
    print("="*80)
    print("GENERATING DAE VS KNN COMPARISONS")
    print("="*80)

    # Load results
    print("\nLoading results...")
    dae_results = load_dae_results(dae_results_dir)
    knn_results = load_knn_results(knn_results_dir)

    if not dae_results:
        print("Error: DAE results not found. Run main.py first.")
        return

    if not knn_results:
        print("Error: KNN results not found. Run knn_main.py first.")
        return

    print(f"Loaded {len(dae_results)} DAE configs and {len(knn_results)} KNN configs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating comparison plots...")

    plot_r2_comparison(
        dae_results,
        knn_results,
        save_path=os.path.join(output_dir, 'r2_comparison.png'),
        show=False
    )

    plot_performance_vs_time(
        dae_results,
        knn_results,
        save_path=os.path.join(output_dir, 'performance_vs_time.png'),
        show=False
    )

    # Generate table
    print("\nGenerating comparison table...")
    table = generate_comparison_table(
        dae_results,
        knn_results,
        save_path=os.path.join(output_dir, 'comparison_table.txt')
    )

    print("\n" + table)

    print("\n" + "="*80)
    print("COMPARISONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - r2_comparison.png")
    print("  - performance_vs_time.png")
    print("  - comparison_table.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    generate_all_comparisons()
