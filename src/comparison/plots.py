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
    method_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot R² comparison between all imputation methods.

    Args:
        method_results: Dictionary mapping method names to their results dictionaries
                       e.g., {'DAE': dae_results, 'KNN': knn_results, 'Zero': zero_results, ...}
        save_path: Path to save figure
        show: Whether to display plot
    """
    missingness_rates = [0.01, 0.05, 0.10]
    x_pos = np.arange(len(missingness_rates))

    n_methods = len(method_results)
    width = 0.8 / n_methods  # Adjust width based on number of methods

    # Colors for each method
    color_map = {
        'DAE': '#2ca02c',
        'KNN': '#ff7f0e',
        'Zero': '#d62728',
        'Mean': '#9467bd',
        'Median': '#8c564b'
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bars for each method
    for idx, (method_name, results) in enumerate(method_results.items()):
        means = []
        stds = []

        for miss_rate in missingness_rates:
            _, mean_val, std_val = get_best_result(results, miss_rate, 'r2')
            means.append(mean_val)
            stds.append(std_val)

        offset = (idx - n_methods/2 + 0.5) * width
        color = color_map.get(method_name, f'C{idx}')
        ax.bar(x_pos + offset, means, width, yerr=stds,
               label=method_name, capsize=5, color=color, alpha=0.8)

    ax.set_xlabel('Missingness Rate', fontweight='bold')
    ax.set_ylabel(r'$R^2$ Score', fontweight='bold')
    ax.set_title('Method Comparison: All Imputation Methods', fontweight='bold', fontsize=14)
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
    method_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot performance vs. computational time trade-off.

    Args:
        method_results: Dictionary mapping method names to their results dictionaries
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Performance vs. Computational Time Trade-off', fontweight='bold', fontsize=14)

    missingness_rates = [0.01, 0.05, 0.10]

    # Colors and markers for each method
    color_map = {
        'DAE': '#2ca02c',
        'KNN': '#ff7f0e',
        'Zero': '#d62728',
        'Mean': '#9467bd',
        'Median': '#8c564b'
    }
    marker_map = {
        'DAE': 'o',
        'KNN': 'o',
        'Zero': 'X',
        'Mean': 's',
        'Median': 'D'
    }

    # Plot positions: (0,0), (0,1), (1,0), legend at (1,1)
    positions = [(0, 0), (0, 1), (1, 0)]

    for idx, miss_rate in enumerate(missingness_rates):
        ax = axes[positions[idx]]

        # Plot each method
        for method_name, results in method_results.items():
            r2_values = []
            time_values = []

            for config, metrics in results.items():
                if f"miss{miss_rate}" in config and 'r2_mean' in metrics:
                    # For DAE, only include learning rate 0.001 (10^-3)
                    if method_name == 'DAE':
                        if 'lr0.001' not in config:
                            continue

                    r2_values.append(metrics['r2_mean'])
                    # Get imputation time (all methods now have this)
                    time_values.append(metrics.get('imputation_time_mean', 0.001))

            if r2_values:
                color = color_map.get(method_name, f'C{list(method_results.keys()).index(method_name)}')
                marker = marker_map.get(method_name, 'o')
                size = 100 if method_name in ['Zero', 'Mean', 'Median'] else 50
                ax.scatter(time_values, r2_values, s=size, alpha=0.6 if method_name in ['DAE', 'KNN'] else 0.8,
                          color=color, marker=marker, label=method_name)

        ax.set_xlabel('Time (log scale)', fontweight='bold')
        # Y-axis label only on left column
        if positions[idx][1] == 0:
            ax.set_ylabel(r'$R^2$ Score', fontweight='bold')
        ax.set_title(f'({chr(65+idx)}) {miss_rate*100:.0f}% Missing', fontweight='bold', loc='left')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    # Bottom right subplot for legend
    ax_legend = axes[1, 1]
    ax_legend.axis('off')

    # Create legend entries
    handles = []
    labels = []
    for method_name in method_results.keys():
        color = color_map.get(method_name, 'gray')
        marker = marker_map.get(method_name, 'o')
        size = 100 if method_name in ['Zero', 'Mean', 'Median'] else 50
        handle = plt.scatter([], [], s=size, alpha=0.6 if method_name in ['DAE', 'KNN'] else 0.8,
                           color=color, marker=marker)
        handles.append(handle)
        labels.append(method_name)

    ax_legend.legend(handles, labels, loc='center', fontsize=12, frameon=True)

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
    method_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> str:
    """
    Generate text table comparing all imputation methods.

    Args:
        method_results: Dictionary mapping method names to their results dictionaries
        save_path: Path to save table

    Returns:
        Formatted table string
    """
    table = []
    table.append("="*110)
    table.append(f"METHOD COMPARISON: {' vs '.join(method_results.keys())}")
    table.append("="*110)
    table.append("")
    table.append(f"{'Missingness':<15} {'Method':<15} {'R²':<20} {'RMSE':<20} {'Config':<40}")
    table.append("-"*110)

    missingness_rates = [0.01, 0.05, 0.10]

    for miss_rate in missingness_rates:
        method_scores = []

        # Process each method
        for method_name, results in method_results.items():
            config, r2, r2_std = get_best_result(results, miss_rate, 'r2')
            _, rmse, rmse_std = get_best_result(results, miss_rate, 'rmse')

            if config:
                # First method shows missingness rate
                prefix = f"{f'{miss_rate*100:.0f}%':<15}" if len(method_scores) == 0 else f"{'':<15}"

                table.append(f"{prefix} {method_name:<15} "
                            f"{f'{r2:.4f}±{r2_std:.4f}':<20} "
                            f"{f'{rmse:.4f}±{rmse_std:.4f}':<20} "
                            f"{config[:40]:<40}")

                method_scores.append((method_name, r2))

        # Determine winner
        if method_scores:
            winner_name, winner_r2 = max(method_scores, key=lambda x: x[1])
            runner_up_r2 = sorted([r[1] for r in method_scores], reverse=True)[1] if len(method_scores) > 1 else 0
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
    baseline_results_dir: str = 'results/baselines',
    output_dir: str = 'results/comparisons'
):
    """
    Generate all comparison visualizations and tables.

    Args:
        dae_results_dir: Directory with DAE results
        knn_results_dir: Directory with KNN results
        baseline_results_dir: Directory with baseline results (contains zero, mean, median)
        output_dir: Directory to save comparisons
    """
    print("="*80)
    print("GENERATING METHOD COMPARISONS")
    print("="*80)

    # Load results
    print("\nLoading results...")
    dae_results = load_results(dae_results_dir)
    knn_results = load_results(knn_results_dir)
    baseline_results = load_results(baseline_results_dir)

    if not dae_results:
        print("Warning: DAE results not found. Run run_dae.py first.")
    if not knn_results:
        print("Warning: KNN results not found. Run run_knn.py first.")
    if not baseline_results:
        print("Warning: Baseline results not found. Run run_baseline.py first.")

    if not (dae_results or knn_results or baseline_results):
        print("Error: No results found. Cannot generate comparisons.")
        return

    # Separate baseline results by method
    zero_results = {k: v for k, v in baseline_results.items() if '_zero' in k}

    print(f"Loaded results:")
    print(f"  DAE: {len(dae_results)} configs")
    print(f"  KNN: {len(knn_results)} configs")
    print(f"  Zero: {len(zero_results)} configs")

    # Create method results dictionary (excluding mean/median)
    method_results = {}
    if dae_results:
        method_results['DAE'] = dae_results
    if knn_results:
        method_results['KNN'] = knn_results
    if zero_results:
        method_results['Zero'] = zero_results

    if not method_results:
        print("Error: No valid method results found.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating comparison plots...")

    plot_method_comparison(
        method_results,
        save_path=os.path.join(output_dir, 'method_comparison.png'),
        show=False
    )

    plot_performance_vs_time(
        method_results,
        save_path=os.path.join(output_dir, 'performance_vs_time.png'),
        show=False
    )

    # Generate table
    print("\nGenerating comparison table...")
    table = generate_comparison_table(
        method_results,
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
    print(f"\nMethods compared: {', '.join(method_results.keys())}")
    print("\n" + "="*80)


if __name__ == "__main__":
    generate_all_comparisons()
