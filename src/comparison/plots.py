"""
Comparison visualizations between DAE, KNN, and baseline imputation methods.
Generates plots comparing neural network vs. classical ML vs. naive approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.visualization import plot_r2_comparison

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13


def load_results(results_dir: str) -> Dict:
    """
    Load results from summary file.

    Args:
        results_dir: Directory containing results

    Returns:
        Dictionary with results
    """
    summary_file = os.path.join(results_dir, 'summary.json')

    if not os.path.exists(summary_file):
        print(f"Warning: Results not found at {summary_file}")
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


def plot_method_comparison(
    dae_results: Dict,
    knn_results: Dict,
    zero_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot R² comparison between DAE, KNN, and Zero imputation.

    Args:
        dae_results: DAE results dictionary
        knn_results: KNN results dictionary
        zero_results: Zero imputation results dictionary
        save_path: Path to save figure
        show: Whether to display plot
    """
    missingness_rates = [0.01, 0.05, 0.10]
    x_pos = np.arange(len(missingness_rates))
    width = 0.25

    dae_means = []
    dae_stds = []
    knn_means = []
    knn_stds = []
    zero_means = []
    zero_stds = []

    for miss_rate in missingness_rates:
        # Get best DAE result
        _, dae_mean, dae_std = get_best_result(dae_results, miss_rate, 'r2')
        dae_means.append(dae_mean)
        dae_stds.append(dae_std)

        # Get best KNN result
        _, knn_mean, knn_std = get_best_result(knn_results, miss_rate, 'r2')
        knn_means.append(knn_mean)
        knn_stds.append(knn_std)

        # Get Zero imputation result
        _, zero_mean, zero_std = get_best_result(zero_results, miss_rate, 'r2')
        zero_means.append(zero_mean)
        zero_stds.append(zero_std)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(x_pos - width, dae_means, width, yerr=dae_stds,
           label='DAE (Neural Network)', capsize=5, color='#2ca02c', alpha=0.8)
    ax.bar(x_pos, knn_means, width, yerr=knn_stds,
           label='KNN (Classical ML)', capsize=5, color='#ff7f0e', alpha=0.8)
    ax.bar(x_pos + width, zero_means, width, yerr=zero_stds,
           label='Zero Imputation (Naive)', capsize=5, color='#d62728', alpha=0.8)

    ax.set_xlabel('Missingness Rate', fontweight='bold')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Method Comparison: DAE vs KNN vs Zero Imputation', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{mr*100:.0f}%' for mr in missingness_rates])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Method comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_vs_time(
    dae_results: Dict,
    knn_results: Dict,
    zero_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot performance vs. computational time trade-off.

    Args:
        dae_results: DAE results dictionary
        knn_results: KNN results dictionary
        zero_results: Zero imputation results dictionary
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

        # Get Zero imputation result
        zero_r2 = []
        zero_time = []
        for config, metrics in zero_results.items():
            if f"miss{miss_rate}" in config and 'r2_mean' in metrics:
                zero_r2.append(metrics['r2_mean'])
                zero_time.append(metrics.get('imputation_time_mean', 0.001))

        # Plot
        if dae_r2:
            ax.scatter(dae_time, dae_r2, s=50, alpha=0.6, color='#2ca02c', label='DAE')
        if knn_r2:
            ax.scatter(knn_time, knn_r2, s=50, alpha=0.6, color='#ff7f0e', label='KNN')
        if zero_r2:
            ax.scatter(zero_time, zero_r2, s=100, alpha=0.8, color='#d62728', marker='X', label='Zero')

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


def generate_comparison_table(
    dae_results: Dict,
    knn_results: Dict,
    zero_results: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Generate text table comparing best DAE, KNN, and Zero imputation results.

    Args:
        dae_results: DAE results dictionary
        knn_results: KNN results dictionary
        zero_results: Zero imputation results dictionary
        save_path: Path to save table

    Returns:
        Formatted table string
    """
    table = []
    table.append("="*110)
    table.append("METHOD COMPARISON: DAE vs KNN vs Zero Imputation")
    table.append("="*110)
    table.append("")
    table.append(f"{'Missingness':<15} {'Method':<15} {'R²':<20} {'RMSE':<20} {'Config':<40}")
    table.append("-"*110)

    missingness_rates = [0.01, 0.05, 0.10]

    for miss_rate in missingness_rates:
        # DAE best
        dae_config, dae_r2, dae_r2_std = get_best_result(dae_results, miss_rate, 'r2')
        _, dae_rmse, dae_rmse_std = get_best_result(dae_results, miss_rate, 'rmse')

        if dae_config:
            table.append(f"{f'{miss_rate*100:.0f}%':<15} {'DAE':<15} "
                        f"{f'{dae_r2:.4f}±{dae_r2_std:.4f}':<20} "
                        f"{f'{dae_rmse:.4f}±{dae_rmse_std:.4f}':<20} "
                        f"{dae_config[:40]:<40}")

        # KNN best
        knn_config, knn_r2, knn_r2_std = get_best_result(knn_results, miss_rate, 'r2')
        _, knn_rmse, knn_rmse_std = get_best_result(knn_results, miss_rate, 'rmse')

        if knn_config:
            table.append(f"{'':<15} {'KNN':<15} "
                        f"{f'{knn_r2:.4f}±{knn_r2_std:.4f}':<20} "
                        f"{f'{knn_rmse:.4f}±{knn_rmse_std:.4f}':<20} "
                        f"{knn_config[:40]:<40}")

        # Zero imputation
        zero_config, zero_r2, zero_r2_std = get_best_result(zero_results, miss_rate, 'r2')
        _, zero_rmse, zero_rmse_std = get_best_result(zero_results, miss_rate, 'rmse')

        if zero_config:
            table.append(f"{'':<15} {'Zero':<15} "
                        f"{f'{zero_r2:.4f}±{zero_r2_std:.4f}':<20} "
                        f"{f'{zero_rmse:.4f}±{zero_rmse_std:.4f}':<20} "
                        f"{zero_config[:40]:<40}")

        # Winner
        results = [
            ('DAE', dae_r2) if dae_config else None,
            ('KNN', knn_r2) if knn_config else None,
            ('Zero', zero_r2) if zero_config else None
        ]
        results = [r for r in results if r is not None]

        if results:
            winner_name, winner_r2 = max(results, key=lambda x: x[1])
            runner_up_r2 = sorted([r[1] for r in results], reverse=True)[1] if len(results) > 1 else 0
            improvement = abs(winner_r2 - runner_up_r2) / max(abs(runner_up_r2), 0.0001) * 100
            table.append(f"{'':<15} {f'Winner: {winner_name} (+{improvement:.1f}%)':<55}")

        table.append("")

    table.append("="*110)

    table_str = "\n".join(table)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"Comparison table saved to {save_path}")

    return table_str


def generate_all_comparisons(
    dae_results_dir: str = 'results/dae',
    knn_results_dir: str = 'results/knn',
    zero_results_dir: str = 'results/baselines',
    output_dir: str = 'results/comparisons'
):
    """
    Generate all comparison visualizations and tables.

    Args:
        dae_results_dir: Directory with DAE results
        knn_results_dir: Directory with KNN results
        zero_results_dir: Directory with baseline results
        output_dir: Directory to save comparisons
    """
    print("="*80)
    print("GENERATING METHOD COMPARISONS")
    print("="*80)

    # Load results
    print("\nLoading results...")
    dae_results = load_results(dae_results_dir)
    knn_results = load_results(knn_results_dir)
    zero_results = load_results(zero_results_dir)

    if not dae_results:
        print("Warning: DAE results not found. Run main.py first.")
    if not knn_results:
        print("Warning: KNN results not found. Run knn_main.py first.")
    if not zero_results:
        print("Warning: Zero imputation results not found. Run zero baseline first.")

    if not (dae_results or knn_results or zero_results):
        print("Error: No results found. Cannot generate comparisons.")
        return

    print(f"Loaded {len(dae_results)} DAE configs, {len(knn_results)} KNN configs, "
          f"{len(zero_results)} Zero configs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating comparison plots...")

    if dae_results and knn_results and zero_results:
        plot_method_comparison(
            dae_results,
            knn_results,
            zero_results,
            save_path=os.path.join(output_dir, 'method_comparison.png'),
            show=False
        )

        plot_performance_vs_time(
            dae_results,
            knn_results,
            zero_results,
            save_path=os.path.join(output_dir, 'performance_vs_time.png'),
            show=False
        )

        # Generate table
        print("\nGenerating comparison table...")
        table = generate_comparison_table(
            dae_results,
            knn_results,
            zero_results,
            save_path=os.path.join(output_dir, 'comparison_table.txt')
        )

        print("\n" + table)

    print("\n" + "="*80)
    print("COMPARISONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("  - method_comparison.png")
    print("  - performance_vs_time.png")
    print("  - comparison_table.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    generate_all_comparisons()
