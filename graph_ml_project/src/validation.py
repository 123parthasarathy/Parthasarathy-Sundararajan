"""
Validation and Testing Framework for Graph ML Models
Comprehensive validation of implementations, results, and research reproducibility
"""

import numpy as np
import networkx as nx
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """Validate ML model implementations and outputs"""

    def __init__(self):
        self.validation_results = {}

    def validate_graph_properties(self, G):
        """Validate basic graph properties"""
        checks = {}

        # Check graph is valid
        checks['is_valid_graph'] = isinstance(G, nx.Graph)
        checks['has_nodes'] = G.number_of_nodes() > 0
        checks['has_edges'] = G.number_of_edges() > 0

        # Check for self-loops and multi-edges
        checks['no_self_loops'] = not any(u == v for u, v in G.edges())
        checks['is_simple'] = not G.is_multigraph()

        # Check connectivity
        checks['is_connected'] = nx.is_connected(G)

        return checks

    def validate_node_features(self, features, expected_shape=None):
        """Validate node feature matrix"""
        checks = {}

        checks['is_numpy_or_tensor'] = isinstance(features, (np.ndarray, torch.Tensor))

        if isinstance(features, torch.Tensor):
            features = features.numpy()

        checks['no_nans'] = not np.any(np.isnan(features))
        checks['no_infs'] = not np.any(np.isinf(features))
        checks['is_2d'] = len(features.shape) == 2

        if expected_shape:
            checks['correct_shape'] = features.shape == expected_shape

        # Check value ranges
        checks['has_finite_values'] = np.all(np.isfinite(features))

        return checks

    def validate_model_output(self, output, num_classes=None):
        """Validate model predictions"""
        checks = {}

        if isinstance(output, torch.Tensor):
            output = output.detach().numpy()

        checks['is_valid_output'] = isinstance(output, np.ndarray)
        checks['no_nans'] = not np.any(np.isnan(output))
        checks['no_infs'] = not np.any(np.isinf(output))

        if num_classes:
            checks['correct_num_classes'] = output.shape[-1] == num_classes

            # Check if log probabilities sum to ~1 (after exp)
            probs = np.exp(output)
            prob_sums = np.sum(probs, axis=-1)
            checks['valid_probabilities'] = np.allclose(prob_sums, 1.0, atol=0.1)

        return checks

    def validate_persistence_diagram(self, dgms):
        """Validate topological persistence diagrams"""
        checks = {}

        checks['is_list'] = isinstance(dgms, list)

        if isinstance(dgms, list) and len(dgms) > 0:
            # Check each dimension
            for i, dgm in enumerate(dgms):
                if len(dgm) > 0:
                    births = dgm[:, 0]
                    deaths = dgm[:, 1]

                    # Birth should be <= death
                    checks[f'H{i}_valid_intervals'] = np.all(births <= deaths)

                    # No NaN values
                    finite_mask = np.isfinite(deaths)
                    checks[f'H{i}_has_finite_points'] = np.any(finite_mask)

        return checks

    def validate_equivariance(self, model, h, pos, edge_index, tolerance=1e-1):
        """Validate E(n) equivariance property"""
        checks = {}

        with torch.no_grad():
            # Original
            out1, pos_traj1 = model(h, pos, edge_index)

            # Translation test
            translation = torch.randn(3)
            pos_translated = pos + translation
            out2, pos_traj2 = model(h, pos_translated, edge_index)

            # Check translation equivariance
            trans_error = torch.norm(pos_traj2[-1] - (pos_traj1[-1] + translation)).item()
            checks['translation_equivariant'] = trans_error < tolerance

            # Check prediction invariance
            pred_diff = torch.norm(out1 - out2).item()
            checks['prediction_invariant'] = pred_diff < tolerance

            # Rotation test
            theta = np.pi / 4
            R = torch.tensor([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=torch.float32)

            pos_rotated = pos @ R.T
            out3, pos_traj3 = model(h, pos_rotated, edge_index)

            # Check rotation equivariance
            expected_rotated = pos_traj1[-1] @ R.T
            rot_error = torch.norm(pos_traj3[-1] - expected_rotated).item()
            checks['rotation_equivariant'] = rot_error < tolerance

            # Store error magnitudes
            checks['translation_error'] = trans_error
            checks['rotation_error'] = rot_error
            checks['invariance_error'] = pred_diff

        return checks

    def validate_file_outputs(self, output_dir='outputs'):
        """Validate that output files were generated"""
        checks = {}

        output_path = Path(output_dir)
        if output_path.exists():
            png_files = list(output_path.glob('*.png'))
            checks['output_dir_exists'] = True
            checks['num_png_files'] = len(png_files)
            checks['has_outputs'] = len(png_files) > 0

            # Check file sizes
            total_size = sum(f.stat().st_size for f in png_files)
            checks['total_size_mb'] = total_size / (1024 * 1024)
            checks['all_files_non_empty'] = all(f.stat().st_size > 0 for f in png_files)
        else:
            checks['output_dir_exists'] = False
            checks['num_png_files'] = 0
            checks['has_outputs'] = False

        return checks

    def generate_validation_report(self, save_path='outputs/validation_report.png', dpi=100):
        """Generate visual validation report"""
        if not self.validation_results:
            print("No validation results to report.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1. Overall validation summary
        all_checks = []
        all_labels = []

        for category, checks in self.validation_results.items():
            for check_name, result in checks.items():
                if isinstance(result, bool):
                    all_checks.append(result)
                    all_labels.append(f"{category}: {check_name}")

        if all_checks:
            passed = sum(all_checks)
            total = len(all_checks)

            # Pie chart
            axes[0].pie([passed, total - passed], labels=['Passed', 'Failed'],
                       colors=['#4CAF50', '#F44336'], autopct='%1.1f%%',
                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            axes[0].set_title(f'Overall Validation Results\n{passed}/{total} checks passed',
                            fontsize=14, fontweight='bold')

        # 2. Category breakdown
        category_results = {}
        for category, checks in self.validation_results.items():
            bool_checks = [v for v in checks.values() if isinstance(v, bool)]
            if bool_checks:
                category_results[category] = sum(bool_checks) / len(bool_checks) * 100

        if category_results:
            categories = list(category_results.keys())
            scores = list(category_results.values())

            colors = ['#4CAF50' if s == 100 else '#FFC107' if s >= 80 else '#F44336' for s in scores]

            axes[1].barh(categories, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[1].set_xlabel('Pass Rate (%)', fontsize=12)
            axes[1].set_title('Validation by Category', fontsize=14, fontweight='bold')
            axes[1].set_xlim(0, 100)
            axes[1].grid(True, alpha=0.3, axis='x')

            # Add percentage labels
            for i, (cat, score) in enumerate(zip(categories, scores)):
                axes[1].text(score + 2, i, f'{score:.1f}%', va='center', fontsize=10, fontweight='bold')

        # 3. Detailed checks table
        axes[2].axis('off')

        # Create table data
        table_data = []
        for category, checks in self.validation_results.items():
            for check_name, result in checks.items():
                if isinstance(result, bool):
                    status = '✓ PASS' if result else '✗ FAIL'
                    table_data.append([category, check_name, status])

        if table_data:
            # Limit to first 15 rows
            table_data = table_data[:15]

            table = axes[2].table(cellText=table_data,
                                colLabels=['Category', 'Check', 'Status'],
                                cellLoc='left', loc='center',
                                colWidths=[0.3, 0.5, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Color code status
            for i in range(1, len(table_data) + 1):
                status = table_data[i-1][2]
                if '✓' in status:
                    table[(i, 2)].set_facecolor('#C8E6C9')
                else:
                    table[(i, 2)].set_facecolor('#FFCDD2')

            # Header styling
            for j in range(3):
                table[(0, j)].set_facecolor('#2196F3')
                table[(0, j)].set_text_props(weight='bold', color='white')

        axes[2].set_title('Detailed Validation Checks', fontsize=14, fontweight='bold', pad=10)

        # 4. Numerical metrics
        axes[3].axis('off')

        metrics_text = "Numerical Validation Metrics:\n\n"

        for category, checks in self.validation_results.items():
            for check_name, result in checks.items():
                if isinstance(result, (int, float)) and not isinstance(result, bool):
                    metrics_text += f"{category} - {check_name}:\n  {result:.6f}\n\n"

        axes[3].text(0.1, 0.9, metrics_text, transform=axes[3].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[3].set_title('Numerical Metrics', fontsize=14, fontweight='bold')

        plt.suptitle('Comprehensive Validation Report', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        return save_path


def run_comprehensive_validation():
    """Run all validation tests"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION & TESTING")
    print("=" * 70)

    validator = ModelValidator()

    # 1. Validate sample graph
    print("\n1. Validating graph properties...")
    G = nx.karate_club_graph()
    graph_checks = validator.validate_graph_properties(G)
    validator.validation_results['Graph Properties'] = graph_checks

    passed = sum(graph_checks.values())
    total = len(graph_checks)
    print(f"   Graph validation: {passed}/{total} checks passed")

    # 2. Validate node features
    print("\n2. Validating node features...")
    features = np.random.randn(G.number_of_nodes(), 10)
    feature_checks = validator.validate_node_features(features, expected_shape=(34, 10))
    validator.validation_results['Node Features'] = feature_checks

    passed = sum(feature_checks.values())
    total = len(feature_checks)
    print(f"   Feature validation: {passed}/{total} checks passed")

    # 3. Validate model output
    print("\n3. Validating model outputs...")
    mock_output = np.random.randn(5, 2)
    mock_output = mock_output - np.max(mock_output, axis=1, keepdims=True)  # Log probabilities
    output_checks = validator.validate_model_output(mock_output, num_classes=2)
    validator.validation_results['Model Output'] = output_checks

    passed = sum(output_checks.values())
    total = len(output_checks)
    print(f"   Output validation: {passed}/{total} checks passed")

    # 4. Validate file outputs
    print("\n4. Validating output files...")
    file_checks = validator.validate_file_outputs('outputs')
    validator.validation_results['File Outputs'] = file_checks

    if file_checks.get('has_outputs', False):
        print(f"   ✓ Found {file_checks['num_png_files']} PNG files")
        print(f"   ✓ Total size: {file_checks['total_size_mb']:.2f} MB")
    else:
        print("   ⚠ No output files found yet")

    # 5. Generate validation report
    print("\n5. Generating validation report...")
    report_path = validator.generate_validation_report()
    if report_path:
        print(f"   ✓ Saved validation report: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_checks_count = 0
    all_passed_count = 0

    for category, checks in validator.validation_results.items():
        bool_checks = [(k, v) for k, v in checks.items() if isinstance(v, bool)]
        if bool_checks:
            passed = sum(v for k, v in bool_checks)
            total = len(bool_checks)
            all_passed_count += passed
            all_checks_count += total

            status = "✓ PASS" if passed == total else "⚠ PARTIAL" if passed > 0 else "✗ FAIL"
            print(f"{category:25} {passed:2}/{total:2} checks passed {status}")

    print("=" * 70)
    overall_rate = (all_passed_count / all_checks_count * 100) if all_checks_count > 0 else 0
    print(f"Overall Success Rate: {overall_rate:.1f}% ({all_passed_count}/{all_checks_count})")
    print("=" * 70)

    return validator


if __name__ == "__main__":
    run_comprehensive_validation()
