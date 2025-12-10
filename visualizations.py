"""
Visualization Module for QI-VGT Results

Generates publication-quality figures for:
1. Performance comparison bar charts
2. Cross-validation stability plots
3. ROC curves with uncertainty regions
4. Ablation study results
5. Calibration plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_performance_comparison(
    results: Dict[str, Dict],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
    title: str = 'Performance Comparison on MUTAG Dataset'
):
    """
    Create bar chart comparing all methods.

    Args:
        results: Dictionary of results from each method
        metric: Metric to compare ('accuracy', 'auc_roc', 'f1', etc.)
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort by performance
    sorted_items = sorted(
        results.items(),
        key=lambda x: x[1]['stats'][metric]['mean'],
        reverse=True
    )

    names = [item[0] for item in sorted_items]
    means = [item[1]['stats'][metric]['mean'] * 100 for item in sorted_items]
    stds = [item[1]['stats'][metric]['std'] * 100 for item in sorted_items]

    # Create color palette
    colors = []
    for name in names:
        if 'QI-VGT' in name:
            colors.append('#2E86AB')  # Blue for proposed method
        elif name in ['GCN', 'GAT', 'GIN', 'DEEPGCN', 'MPNN']:
            colors.append('#A23B72')  # Purple for GNN baselines
        else:
            colors.append('#F18F01')  # Orange for traditional ML

    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.5),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel(f'{metric.upper()} (%)', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)

    plt.xticks(rotation=45, ha='right')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#2E86AB', label='QI-VGT (Proposed)'),
        mpatches.Patch(facecolor='#A23B72', label='GNN Baselines'),
        mpatches.Patch(facecolor='#F18F01', label='Traditional ML')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_cross_validation_stability(
    fold_results: List[float],
    method_name: str = 'QI-VGT',
    metric_name: str = 'Accuracy',
    save_path: Optional[str] = None
):
    """
    Plot cross-validation performance across folds with confidence interval.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    folds = list(range(1, len(fold_results) + 1))
    values = [v * 100 for v in fold_results]
    mean_val = np.mean(values)
    std_val = np.std(values)

    # Plot line with markers
    ax.plot(folds, values, 'o-', color='#2E86AB', linewidth=2,
            markersize=10, markerfacecolor='white', markeredgewidth=2,
            label=f'{method_name}')

    # Add confidence interval
    ax.fill_between(folds, mean_val - 1.96 * std_val, mean_val + 1.96 * std_val,
                    alpha=0.2, color='#2E86AB', label='95% CI')

    # Add mean line
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2f}%')

    ax.set_xlabel('Fold', fontsize=14)
    ax.set_ylabel(f'{metric_name} (%)', fontsize=14)
    ax.set_title(f'{method_name} Cross-Validation Stability', fontsize=16, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(loc='lower right')

    # Add coefficient of variation annotation
    cv = (std_val / mean_val) * 100
    ax.annotate(f'CV: {cv:.2f}%\nStd: {std_val:.2f}%',
                xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_metrics_radar(
    results: Dict[str, Dict],
    methods_to_compare: List[str],
    save_path: Optional[str] = None
):
    """
    Create radar/spider plot comparing multiple metrics across methods.
    """
    metrics = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'AUC-ROC', 'Precision', 'Recall', 'F1-Score']

    # Number of metrics
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#DC3545']

    for idx, method in enumerate(methods_to_compare):
        if method not in results:
            continue

        values = [results[method]['stats'][m]['mean'] * 100 for m in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=method,
                color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=12)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Metric Performance Comparison', fontsize=16, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_ablation_results(
    ablation_results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Create horizontal bar chart for ablation study results.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get full model accuracy as reference
    full_model_acc = None
    for name, data in ablation_results.items():
        if 'Full' in name:
            full_model_acc = data['statistics']['accuracy']['mean'] * 100
            break

    names = []
    values = []
    colors = []

    for name, data in ablation_results.items():
        acc = data['statistics']['accuracy']['mean'] * 100
        names.append(name)
        values.append(acc)

        if 'Full' in name:
            colors.append('#28A745')  # Green for full model
        elif acc < full_model_acc - 3:
            colors.append('#DC3545')  # Red for significant degradation
        elif acc < full_model_acc - 1:
            colors.append('#FFC107')  # Yellow for moderate degradation
        else:
            colors.append('#6C757D')  # Gray for similar performance

    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        diff = val - full_model_acc if full_model_acc else 0
        label = f'{val:.2f}% ({diff:+.2f}%)' if full_model_acc else f'{val:.2f}%'
        ax.annotate(label,
                    xy=(width + 0.5, bar.get_y() + bar.get_height() / 2),
                    va='center', fontsize=10)

    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_title('Ablation Study Results', fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(values) + 8)

    # Add vertical line for full model
    if full_model_acc:
        ax.axvline(x=full_model_acc, color='#28A745', linestyle='--', linewidth=2,
                   label=f'Full Model: {full_model_acc:.2f}%')
        ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_uncertainty_analysis(
    uncertainties: List[float],
    predictions: List[int],
    labels: List[int],
    save_path: Optional[str] = None
):
    """
    Create uncertainty vs accuracy relationship plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    uncertainties = np.array(uncertainties)
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Accuracy vs uncertainty bins
    bins = np.linspace(0, 1, 6)
    bin_accs = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (uncertainties >= bins[i]) & (uncertainties < bins[i + 1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(predictions[mask] == labels[mask]))
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

    axes[0].scatter(bin_centers, bin_accs, s=100, c='#2E86AB', edgecolor='black')
    axes[0].plot(bin_centers, bin_accs, '--', color='#2E86AB', alpha=0.5)
    axes[0].set_xlabel('Prediction Uncertainty', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].set_title('Uncertainty vs Accuracy Relationship', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0.5, 1.05)

    # Uncertainty distribution
    axes[1].hist(uncertainties, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Prediction Uncertainty', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=14)
    axes[1].set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')

    # Add mean uncertainty line
    mean_unc = np.mean(uncertainties)
    axes[1].axvline(x=mean_unc, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_unc:.3f}')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_literature_comparison(
    our_results: Dict[str, float],
    literature_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Compare our results with recent literature.

    Args:
        our_results: Dict with our metric values (mean)
        literature_results: Dict of {paper_name: {metric: value}}
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = ['Accuracy', 'AUC-ROC', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.15

    # Our method
    our_values = [our_results.get(m.lower().replace('-', '_').replace(' ', '_').replace('_score', ''), 0)
                  for m in metrics]

    bars = ax.bar(x - 2*width, our_values, width, label='QI-VGT (Ours)',
                  color='#2E86AB', edgecolor='black', linewidth=1.5)

    # Literature baselines
    colors = ['#A23B72', '#F18F01', '#28A745', '#6C757D', '#DC3545']
    for idx, (paper, values) in enumerate(literature_results.items()):
        paper_values = []
        for m in metrics:
            key = m.lower().replace('-', '_').replace(' ', '_').replace('_score', '')
            paper_values.append(values.get(key, 0))

        ax.bar(x + (idx - 1) * width, paper_values, width, label=paper,
               color=colors[idx % len(colors)], edgecolor='black', linewidth=1)

    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_title('Comparison with Recent Literature (2024-2025)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def generate_all_visualizations(
    results: Dict,
    ablation_results: Dict = None,
    output_dir: str = 'figures'
):
    """
    Generate all publication-quality visualizations.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # 1. Performance comparison
    plot_performance_comparison(
        results,
        metric='accuracy',
        save_path=f'{output_dir}/performance_comparison_accuracy.png'
    )

    plot_performance_comparison(
        results,
        metric='auc_roc',
        save_path=f'{output_dir}/performance_comparison_auc.png',
        title='AUC-ROC Comparison on MUTAG Dataset'
    )

    # 2. Cross-validation stability
    if 'QI-VGT' in results:
        plot_cross_validation_stability(
            results['QI-VGT']['results']['accuracy'],
            method_name='QI-VGT',
            save_path=f'{output_dir}/cv_stability.png'
        )

    # 3. Radar plot
    methods_to_compare = ['QI-VGT', 'QI-VGT++', 'GAT', 'GCN', 'Random Forest']
    methods_available = [m for m in methods_to_compare if m in results]
    plot_metrics_radar(
        results,
        methods_available,
        save_path=f'{output_dir}/radar_comparison.png'
    )

    # 4. Ablation study
    if ablation_results:
        plot_ablation_results(
            ablation_results,
            save_path=f'{output_dir}/ablation_study.png'
        )

    # 5. Literature comparison
    qi_vgt_stats = results.get('QI-VGT', {}).get('stats', {})
    our_results = {
        'accuracy': qi_vgt_stats.get('accuracy', {}).get('mean', 0) * 100,
        'auc_roc': qi_vgt_stats.get('auc_roc', {}).get('mean', 0) * 100,
        'f1': qi_vgt_stats.get('f1', {}).get('mean', 0) * 100
    }

    literature_results = {
        'GAT (2024)': {'accuracy': 89.42, 'auc_roc': 88.0, 'f1': 87.5},
        'GIN (2024)': {'accuracy': 85.14, 'auc_roc': 86.0, 'f1': 84.0},
        'DeepGCN': {'accuracy': 85.0, 'auc_roc': 87.0, 'f1': 84.5},
        'GCN': {'accuracy': 84.10, 'auc_roc': 85.0, 'f1': 83.0}
    }

    plot_literature_comparison(
        our_results,
        literature_results,
        save_path=f'{output_dir}/literature_comparison.png'
    )

    print(f"All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test with dummy data
    dummy_results = {
        'QI-VGT': {
            'results': {'accuracy': [0.84, 0.87, 0.79, 0.86, 0.86]},
            'stats': {
                'accuracy': {'mean': 0.845, 'std': 0.03},
                'auc_roc': {'mean': 0.893, 'std': 0.028},
                'precision': {'mean': 0.868, 'std': 0.048},
                'recall': {'mean': 0.912, 'std': 0.047},
                'f1': {'mean': 0.887, 'std': 0.021}
            }
        },
        'GCN': {
            'results': {'accuracy': [0.80, 0.82, 0.78, 0.83, 0.82]},
            'stats': {
                'accuracy': {'mean': 0.81, 'std': 0.02},
                'auc_roc': {'mean': 0.85, 'std': 0.03},
                'precision': {'mean': 0.82, 'std': 0.04},
                'recall': {'mean': 0.85, 'std': 0.04},
                'f1': {'mean': 0.83, 'std': 0.03}
            }
        }
    }

    print("Testing visualization module...")
    generate_all_visualizations(dummy_results, output_dir='test_figures')
