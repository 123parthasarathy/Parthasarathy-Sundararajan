"""
Publication-Quality Visualization Module
Generates figures for Q1 journal submission
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


class PublicationVisualizer:
    """
    Generate publication-quality visualizations.
    """

    def __init__(self, output_dir: str = './figures'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Color palette for consistency
        self.colors = sns.color_palette("husl", 10)
        self.method_colors = {
            'none': '#1f77b4',
            'smote': '#ff7f0e',
            'borderline_smote': '#2ca02c',
            'adasyn': '#d62728',
            'smote_tomek': '#9467bd',
            'smote_enn': '#8c564b',
            'class_weight': '#e377c2'
        }

    def plot_roc_curves(self, results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                        title: str = 'ROC Curves Comparison',
                        filename: str = 'roc_curves.png') -> None:
        """
        Plot ROC curves for multiple classifiers.

        Args:
            results: Dict mapping classifier names to (y_true, y_proba, label) tuples
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        for i, (name, (y_true, y_proba, label)) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=self.colors[i % len(self.colors)],
                    lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_pr_curves(self, results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                       title: str = 'Precision-Recall Curves',
                       filename: str = 'pr_curves.png') -> None:
        """
        Plot Precision-Recall curves for multiple classifiers.

        Args:
            results: Dict mapping classifier names to (y_true, y_proba, label) tuples
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        for i, (name, (y_true, y_proba, label)) in enumerate(results.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, y_proba)

            ax.plot(recall, precision, color=self.colors[i % len(self.colors)],
                    lw=2, label=f'{label} (AP = {ap:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_confusion_matrices(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                 class_names: List[str] = ['Benign', 'Malignant'],
                                 filename: str = 'confusion_matrices.png') -> None:
        """
        Plot confusion matrices for multiple classifiers.

        Args:
            results: Dict mapping classifier names to (y_true, y_pred) tuples
            class_names: Names of classes
            filename: Output filename
        """
        n_classifiers = len(results)
        n_cols = min(3, n_classifiers)
        n_rows = (n_classifiers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_classifiers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (name, (y_true, y_pred)) in enumerate(results.items()):
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=class_names, yticklabels=class_names,
                        cbar=False)

            # Add percentages
            for j in range(len(class_names)):
                for k in range(len(class_names)):
                    axes[i].text(k + 0.5, j + 0.7, f'({cm_normalized[j, k]:.1%})',
                                 ha='center', va='center', fontsize=9, color='gray')

            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_title(f'{name}')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_metric_comparison(self, results: Dict[str, Dict[str, float]],
                                metrics: List[str] = ['accuracy', 'AUC', 'sensitivity', 'specificity', 'g_mean'],
                                title: str = 'Performance Comparison',
                                filename: str = 'metric_comparison.png') -> None:
        """
        Plot bar chart comparing metrics across classifiers.

        Args:
            results: Dict mapping classifier names to metric dictionaries
            metrics: Metrics to compare
            title: Plot title
            filename: Output filename
        """
        n_classifiers = len(results)
        n_metrics = len(metrics)
        x = np.arange(n_classifiers)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(12, 6))

        classifier_names = list(results.keys())

        for i, metric in enumerate(metrics):
            values = []
            errors = []
            for clf_name in classifier_names:
                metric_data = results[clf_name].get(metric, {})
                if isinstance(metric_data, dict):
                    values.append(metric_data.get('mean', 0))
                    errors.append(metric_data.get('std', 0))
                else:
                    values.append(metric_data)
                    errors.append(0)

            offset = (i - n_metrics/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(),
                         color=self.colors[i], alpha=0.8, yerr=errors, capsize=3)

        ax.set_xlabel('Classifier')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(classifier_names, rotation=45, ha='right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_imbalance_method_comparison(self, results: Dict[str, Dict[str, Dict]],
                                          classifier_name: str,
                                          metrics: List[str] = ['g_mean', 'AUC', 'sensitivity', 'specificity'],
                                          filename: str = 'imbalance_comparison.png') -> None:
        """
        Compare different imbalance handling methods for a specific classifier.

        Args:
            results: Nested dict [imbalance_method][classifier][metrics]
            classifier_name: Name of classifier to analyze
            metrics: Metrics to compare
            filename: Output filename
        """
        methods = list(results.keys())
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            values = []
            errors = []
            colors = []

            for method in methods:
                clf_results = results[method].get(classifier_name, {})
                metric_data = clf_results.get(metric, {})

                if isinstance(metric_data, dict):
                    values.append(metric_data.get('mean', 0))
                    errors.append(metric_data.get('std', 0))
                else:
                    values.append(metric_data if metric_data else 0)
                    errors.append(0)

                colors.append(self.method_colors.get(method, '#333333'))

            axes[i].bar(range(len(methods)), values, yerr=errors, capsize=3,
                       color=colors, alpha=0.8)
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels([m.replace('_', '\n') for m in methods],
                                     rotation=45, ha='right', fontsize=9)
            axes[i].set_ylabel('Score')
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylim([0, 1.1])
            axes[i].grid(True, axis='y', alpha=0.3)

        plt.suptitle(f'Imbalance Method Comparison for {classifier_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_heatmap(self, results: Dict[str, Dict[str, Dict]],
                     metric: str = 'g_mean',
                     title: str = 'Performance Heatmap',
                     filename: str = 'heatmap.png') -> None:
        """
        Create heatmap of classifier vs imbalance method performance.

        Args:
            results: Nested dict [imbalance_method][classifier][metrics]
            metric: Metric to visualize
            title: Plot title
            filename: Output filename
        """
        methods = list(results.keys())
        classifiers = list(results[methods[0]].keys())

        # Build matrix
        matrix = np.zeros((len(classifiers), len(methods)))

        for i, clf in enumerate(classifiers):
            for j, method in enumerate(methods):
                metric_data = results[method].get(clf, {}).get(metric, {})
                if isinstance(metric_data, dict):
                    matrix[i, j] = metric_data.get('mean', 0)
                else:
                    matrix[i, j] = metric_data if metric_data else 0

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=[m.replace('_', '\n') for m in methods],
                    yticklabels=classifiers, ax=ax,
                    vmin=0, vmax=1, cbar_kws={'label': metric.replace('_', ' ').title()})

        ax.set_xlabel('Imbalance Handling Method')
        ax.set_ylabel('Classifier')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_specificity_improvement(self, results_baseline: Dict,
                                       results_improved: Dict,
                                       filename: str = 'specificity_improvement.png') -> None:
        """
        Highlight specificity improvement from baseline to improved methods.

        Args:
            results_baseline: Results without imbalance handling
            results_improved: Results with imbalance handling
            filename: Output filename
        """
        classifiers = list(results_baseline.keys())

        baseline_spec = []
        improved_spec = []
        improvement = []

        for clf in classifiers:
            b_spec = results_baseline[clf].get('specificity', {})
            i_spec = results_improved[clf].get('specificity', {})

            b_val = b_spec.get('mean', 0) if isinstance(b_spec, dict) else (b_spec or 0)
            i_val = i_spec.get('mean', 0) if isinstance(i_spec, dict) else (i_spec or 0)

            baseline_spec.append(b_val)
            improved_spec.append(i_val)
            improvement.append(i_val - b_val)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar comparison
        x = np.arange(len(classifiers))
        width = 0.35

        ax1.bar(x - width/2, baseline_spec, width, label='No Handling', color='#d62728', alpha=0.8)
        ax1.bar(x + width/2, improved_spec, width, label='With SMOTE', color='#2ca02c', alpha=0.8)
        ax1.set_xlabel('Classifier')
        ax1.set_ylabel('Specificity')
        ax1.set_title('Specificity Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classifiers, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, axis='y', alpha=0.3)

        # Improvement plot
        colors = ['#2ca02c' if i > 0 else '#d62728' for i in improvement]
        ax2.bar(classifiers, improvement, color=colors, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Classifier')
        ax2.set_ylabel('Improvement (Δ)')
        ax2.set_title('Specificity Improvement')
        ax2.set_xticklabels(classifiers, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)

        plt.suptitle('Impact of Imbalance Handling on Specificity', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

    def plot_sota_comparison(self, our_results: Dict, sota_results: Dict,
                              filename: str = 'sota_comparison.png') -> None:
        """
        Compare our results with state-of-the-art benchmarks.

        Args:
            our_results: Our experimental results
            sota_results: State-of-the-art benchmarks
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract our best results
        our_methods = list(our_results.keys())
        our_accuracy = [our_results[m].get('accuracy', 0) for m in our_methods]
        our_auc = [our_results[m].get('AUC', 0) for m in our_methods]

        # Extract SOTA results
        sota_methods = list(sota_results.keys())
        sota_accuracy = [sota_results[m].get('accuracy', 0) for m in sota_methods]
        sota_auc = [sota_results[m].get('AUC', 0) for m in sota_methods]

        # Accuracy comparison
        all_methods = our_methods + sota_methods
        all_accuracy = our_accuracy + sota_accuracy
        colors = ['#2ca02c'] * len(our_methods) + ['#1f77b4'] * len(sota_methods)

        y_pos = np.arange(len(all_methods))
        ax1.barh(y_pos, all_accuracy, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(all_methods, fontsize=9)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xlim([0.8, 1.0])
        ax1.axvline(x=max(our_accuracy), color='green', linestyle='--', alpha=0.5, label='Our Best')
        ax1.grid(True, axis='x', alpha=0.3)

        # AUC comparison
        all_auc = our_auc + sota_auc
        ax2.barh(y_pos, all_auc, color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_methods, fontsize=9)
        ax2.set_xlabel('AUC')
        ax2.set_title('AUC Comparison')
        ax2.set_xlim([0.8, 1.0])
        ax2.axvline(x=max(our_auc), color='green', linestyle='--', alpha=0.5, label='Our Best')
        ax2.grid(True, axis='x', alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ca02c', alpha=0.8, label='Our Methods'),
                          Patch(facecolor='#1f77b4', alpha=0.8, label='State-of-the-Art')]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))

        plt.suptitle('Comparison with State-of-the-Art Methods', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")


def generate_all_visualizations(results: Dict, output_dir: str = './figures'):
    """
    Generate all visualizations from experimental results.

    Args:
        results: Complete experimental results dictionary
        output_dir: Output directory for figures
    """
    viz = PublicationVisualizer(output_dir)

    print("\nGenerating publication-quality visualizations...")

    # 1. Performance heatmap
    viz.plot_heatmap(results, metric='g_mean',
                     title='G-Mean Performance Heatmap',
                     filename='heatmap_gmean.png')

    viz.plot_heatmap(results, metric='AUC',
                     title='AUC Performance Heatmap',
                     filename='heatmap_auc.png')

    # 2. Imbalance method comparison for best classifiers
    for clf in ['SVM-RBF', 'Random Forest']:
        if clf in results.get('none', {}):
            viz.plot_imbalance_method_comparison(
                results, clf,
                filename=f'imbalance_comparison_{clf.lower().replace("-", "_").replace(" ", "_")}.png'
            )

    # 3. Specificity improvement
    if 'none' in results and 'smote' in results:
        viz.plot_specificity_improvement(
            results['none'], results['smote'],
            filename='specificity_improvement.png'
        )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    # Test visualization with sample data
    print("Testing Visualization Module...")

    # Create sample results
    sample_results = {
        'none': {
            'SVM-RBF': {'g_mean': {'mean': 0.85, 'std': 0.02}, 'AUC': {'mean': 0.90, 'std': 0.01},
                        'sensitivity': {'mean': 0.95, 'std': 0.02}, 'specificity': {'mean': 0.75, 'std': 0.03}},
            'Random Forest': {'g_mean': {'mean': 0.88, 'std': 0.02}, 'AUC': {'mean': 0.92, 'std': 0.01},
                              'sensitivity': {'mean': 0.93, 'std': 0.02}, 'specificity': {'mean': 0.83, 'std': 0.02}},
        },
        'smote': {
            'SVM-RBF': {'g_mean': {'mean': 0.92, 'std': 0.01}, 'AUC': {'mean': 0.95, 'std': 0.01},
                        'sensitivity': {'mean': 0.94, 'std': 0.01}, 'specificity': {'mean': 0.90, 'std': 0.02}},
            'Random Forest': {'g_mean': {'mean': 0.93, 'std': 0.01}, 'AUC': {'mean': 0.96, 'std': 0.01},
                              'sensitivity': {'mean': 0.95, 'std': 0.01}, 'specificity': {'mean': 0.91, 'std': 0.02}},
        },
        'borderline_smote': {
            'SVM-RBF': {'g_mean': {'mean': 0.94, 'std': 0.01}, 'AUC': {'mean': 0.97, 'std': 0.01},
                        'sensitivity': {'mean': 0.96, 'std': 0.01}, 'specificity': {'mean': 0.92, 'std': 0.01}},
            'Random Forest': {'g_mean': {'mean': 0.94, 'std': 0.01}, 'AUC': {'mean': 0.97, 'std': 0.01},
                              'sensitivity': {'mean': 0.97, 'std': 0.01}, 'specificity': {'mean': 0.91, 'std': 0.02}},
        }
    }

    generate_all_visualizations(sample_results, './figures')
