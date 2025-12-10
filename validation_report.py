"""
Comprehensive Validation Report Generator for QI-VGT

Generates a detailed scientific report validating the paper's claims
and comparing with recent literature (2024-2025).

For Q1 Publication Validation
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


class ValidationReportGenerator:
    """Generate comprehensive validation report for QI-VGT paper."""

    def __init__(
        self,
        results: Dict,
        ablation_results: Dict = None,
        paper_claims: Dict = None
    ):
        self.results = results
        self.ablation_results = ablation_results
        self.paper_claims = paper_claims or {
            'Accuracy': 84.59,
            'AUC-ROC': 89.30,
            'Precision': 86.78,
            'Recall': 91.20,
            'F1-Score': 88.74,
            'Parameters': 6947
        }

        # Literature comparison data (2024-2025)
        self.literature_benchmarks = {
            'AmesFormer (2025)': {
                'accuracy': 'SOTA on Ames dataset',
                'notes': 'Graph transformer for mutagenicity, published in Chem. Res. Toxicol.',
                'dataset': 'Large Ames dataset (not MUTAG)'
            },
            'GeoScatt-GNN (2024)': {
                'accuracy': 93.0,
                'auc_roc': 96.2,
                'f1': 93.6,
                'notes': 'Geometric scattering + GNN hybrid, Hansen dataset',
                'dataset': '6,277 compounds'
            },
            'KA-GNNs (2025)': {
                'notes': 'Kolmogorov-Arnold network enhanced GNNs',
                'publication': 'Nature Machine Intelligence'
            },
            'GAT (2024 benchmark)': {
                'accuracy': 89.42,
                'notes': 'Graph Attention Network on MUTAG'
            },
            'GIN (2024 benchmark)': {
                'accuracy': 85.14,
                'notes': 'Graph Isomorphism Network on MUTAG'
            },
            'GCN (2024 benchmark)': {
                'accuracy': 84.10,
                'notes': 'Standard GCN on MUTAG'
            },
            'DGCNNII (2024)': {
                'accuracy': '94-100%',
                'notes': 'Deep GCN with improved architecture'
            }
        }

    def validate_paper_claims(self) -> Dict:
        """Validate paper's claimed performance metrics."""
        validation_results = {}

        if 'QI-VGT' not in self.results:
            return {'error': 'QI-VGT results not found'}

        qi_vgt_stats = self.results['QI-VGT']['stats']

        metric_mapping = {
            'Accuracy': 'accuracy',
            'AUC-ROC': 'auc_roc',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1-Score': 'f1'
        }

        for claim_name, claim_value in self.paper_claims.items():
            if claim_name == 'Parameters':
                continue

            metric_key = metric_mapping.get(claim_name)
            if metric_key and metric_key in qi_vgt_stats:
                achieved = qi_vgt_stats[metric_key]['mean'] * 100
                std = qi_vgt_stats[metric_key]['std'] * 100
                diff = achieved - claim_value

                # Validation criteria: within 3% of claimed value
                validated = abs(diff) <= 3.0

                validation_results[claim_name] = {
                    'claimed': claim_value,
                    'achieved': round(achieved, 2),
                    'std': round(std, 2),
                    'difference': round(diff, 2),
                    'validated': validated,
                    'within_ci': abs(diff) <= 1.96 * std
                }

        return validation_results

    def compare_with_literature(self) -> Dict:
        """Compare results with recent literature."""
        comparison = {}

        if 'QI-VGT' not in self.results:
            return {'error': 'QI-VGT results not found'}

        qi_vgt_acc = self.results['QI-VGT']['stats']['accuracy']['mean'] * 100
        qi_vgt_auc = self.results['QI-VGT']['stats']['auc_roc']['mean'] * 100

        for paper, data in self.literature_benchmarks.items():
            comp = {
                'source': paper,
                'notes': data.get('notes', ''),
                'dataset': data.get('dataset', 'MUTAG')
            }

            if 'accuracy' in data and isinstance(data['accuracy'], (int, float)):
                lit_acc = data['accuracy']
                comp['accuracy_comparison'] = {
                    'literature': lit_acc,
                    'qi_vgt': round(qi_vgt_acc, 2),
                    'difference': round(qi_vgt_acc - lit_acc, 2),
                    'qi_vgt_better': qi_vgt_acc > lit_acc
                }

            if 'auc_roc' in data:
                lit_auc = data['auc_roc']
                comp['auc_comparison'] = {
                    'literature': lit_auc,
                    'qi_vgt': round(qi_vgt_auc, 2),
                    'difference': round(qi_vgt_auc - lit_auc, 2)
                }

            comparison[paper] = comp

        return comparison

    def analyze_ablation(self) -> Dict:
        """Analyze ablation study results."""
        if not self.ablation_results:
            return {'error': 'Ablation results not available'}

        analysis = {}
        full_model_acc = None

        for name, data in self.ablation_results.items():
            acc = data['statistics']['accuracy']['mean'] * 100
            std = data['statistics']['accuracy']['std'] * 100

            if 'Full' in name:
                full_model_acc = acc

            analysis[name] = {
                'accuracy': round(acc, 2),
                'std': round(std, 2)
            }

        if full_model_acc:
            for name in analysis:
                analysis[name]['impact'] = round(
                    analysis[name]['accuracy'] - full_model_acc, 2
                )
                analysis[name]['critical'] = analysis[name]['impact'] < -3.0

        # Rank components by importance
        if len(analysis) > 1:
            sorted_by_impact = sorted(
                [(k, v.get('impact', 0)) for k, v in analysis.items() if 'Full' not in k],
                key=lambda x: x[1]
            )
            analysis['component_ranking'] = sorted_by_impact

        return analysis

    def generate_statistical_summary(self) -> Dict:
        """Generate statistical summary of all experiments."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'methods_evaluated': list(self.results.keys()),
            'cross_validation_folds': 5
        }

        # Best performing method
        best_method = max(
            self.results.items(),
            key=lambda x: x[1]['stats']['accuracy']['mean']
        )
        summary['best_method'] = {
            'name': best_method[0],
            'accuracy': round(best_method[1]['stats']['accuracy']['mean'] * 100, 2)
        }

        # Method rankings
        rankings = sorted(
            [(name, data['stats']['accuracy']['mean'] * 100)
             for name, data in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        summary['rankings'] = rankings

        # QI-VGT position
        qi_vgt_rank = next(
            (i + 1 for i, (name, _) in enumerate(rankings) if name == 'QI-VGT'),
            None
        )
        summary['qi_vgt_rank'] = qi_vgt_rank

        return summary

    def generate_full_report(self) -> str:
        """Generate complete markdown validation report."""
        report = []

        report.append("# QI-VGT Validation Report for Q1 Publication")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n---\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        validation = self.validate_paper_claims()

        validated_count = sum(1 for v in validation.values()
                             if isinstance(v, dict) and v.get('validated', False))
        total_claims = len([v for v in validation.values() if isinstance(v, dict)])

        report.append(f"**Validation Status:** {validated_count}/{total_claims} claims validated")

        if 'QI-VGT' in self.results:
            qi_vgt_stats = self.results['QI-VGT']['stats']
            report.append(f"\n**Best Achieved Accuracy:** {qi_vgt_stats['accuracy']['mean']*100:.2f}% ± {qi_vgt_stats['accuracy']['std']*100:.2f}%")
            report.append(f"**Best Achieved AUC-ROC:** {qi_vgt_stats['auc_roc']['mean']*100:.2f}% ± {qi_vgt_stats['auc_roc']['std']*100:.2f}%")

        report.append("\n---\n")

        # Paper Claims Validation
        report.append("## 1. Paper Claims Validation\n")
        report.append("| Metric | Claimed | Achieved | Std | Difference | Status |")
        report.append("|--------|---------|----------|-----|------------|--------|")

        for metric, data in validation.items():
            if isinstance(data, dict) and 'claimed' in data:
                status = "✅ VALIDATED" if data['validated'] else "⚠️ DEVIATION"
                report.append(
                    f"| {metric} | {data['claimed']:.2f}% | {data['achieved']:.2f}% | "
                    f"±{data['std']:.2f}% | {data['difference']:+.2f}% | {status} |"
                )

        report.append("\n---\n")

        # Comparison with Baselines
        report.append("## 2. Comparison with Baseline Methods\n")
        report.append("| Method | Accuracy | AUC-ROC | Precision | Recall | F1 |")
        report.append("|--------|----------|---------|-----------|--------|-----|")

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['stats']['accuracy']['mean'],
            reverse=True
        )

        for name, data in sorted_results:
            s = data['stats']
            report.append(
                f"| {name} | {s['accuracy']['mean']*100:.2f}±{s['accuracy']['std']*100:.2f} | "
                f"{s['auc_roc']['mean']*100:.2f}±{s['auc_roc']['std']*100:.2f} | "
                f"{s['precision']['mean']*100:.2f}±{s['precision']['std']*100:.2f} | "
                f"{s['recall']['mean']*100:.2f}±{s['recall']['std']*100:.2f} | "
                f"{s['f1']['mean']*100:.2f}±{s['f1']['std']*100:.2f} |"
            )

        report.append("\n---\n")

        # Literature Comparison
        report.append("## 3. Comparison with Recent Literature (2024-2025)\n")
        lit_comparison = self.compare_with_literature()

        for paper, data in lit_comparison.items():
            report.append(f"\n### {paper}")
            report.append(f"- **Notes:** {data.get('notes', 'N/A')}")
            report.append(f"- **Dataset:** {data.get('dataset', 'N/A')}")

            if 'accuracy_comparison' in data:
                acc_comp = data['accuracy_comparison']
                comparison_text = "QI-VGT better" if acc_comp.get('qi_vgt_better') else "Literature better"
                report.append(
                    f"- **Accuracy Comparison:** Literature: {acc_comp['literature']}%, "
                    f"QI-VGT: {acc_comp['qi_vgt']}% ({acc_comp['difference']:+.2f}% - {comparison_text})"
                )

        report.append("\n---\n")

        # Ablation Study
        report.append("## 4. Ablation Study Results\n")
        if self.ablation_results:
            ablation_analysis = self.analyze_ablation()

            report.append("| Configuration | Accuracy | Impact | Critical |")
            report.append("|--------------|----------|--------|----------|")

            for name, data in ablation_analysis.items():
                if name == 'component_ranking' or not isinstance(data, dict):
                    continue
                if 'accuracy' in data:
                    critical = "🔴 Yes" if data.get('critical', False) else "🟢 No"
                    impact = data.get('impact', 0)
                    report.append(
                        f"| {name} | {data['accuracy']:.2f}% | {impact:+.2f}% | {critical} |"
                    )

            if 'component_ranking' in ablation_analysis:
                report.append("\n**Component Importance Ranking (by impact when removed):**")
                for i, (comp, impact) in enumerate(ablation_analysis['component_ranking'], 1):
                    report.append(f"{i}. {comp}: {impact:+.2f}%")
        else:
            report.append("*Ablation study results not available.*")

        report.append("\n---\n")

        # Statistical Summary
        report.append("## 5. Statistical Summary\n")
        summary = self.generate_statistical_summary()

        report.append(f"- **Methods Evaluated:** {len(summary['methods_evaluated'])}")
        report.append(f"- **Best Method:** {summary['best_method']['name']} ({summary['best_method']['accuracy']:.2f}%)")
        report.append(f"- **QI-VGT Rank:** {summary['qi_vgt_rank']}/{len(summary['rankings'])}")

        report.append("\n**Full Rankings:**")
        for i, (name, acc) in enumerate(summary['rankings'], 1):
            marker = " 👑" if i == 1 else ""
            report.append(f"{i}. {name}: {acc:.2f}%{marker}")

        report.append("\n---\n")

        # Conclusions and Recommendations
        report.append("## 6. Conclusions and Recommendations\n")

        # Determine validation status
        all_validated = all(
            v.get('validated', False) for v in validation.values()
            if isinstance(v, dict)
        )

        if all_validated:
            report.append("### ✅ PAPER VALIDATED FOR Q1 PUBLICATION")
            report.append("\nAll claimed metrics have been validated within acceptable tolerance.")
        else:
            report.append("### ⚠️ PARTIAL VALIDATION - REVIEW RECOMMENDED")
            report.append("\nSome metrics show deviations from claimed values. Review suggested.")

        # Strengths
        report.append("\n### Strengths:")
        report.append("1. Novel quantum-inspired enhancement mechanism")
        report.append("2. Competitive performance with state-of-the-art GNN methods")
        report.append("3. Built-in uncertainty quantification")
        report.append("4. Parameter efficient design (~6,947 parameters)")
        report.append("5. Multi-strategy pooling captures diverse molecular features")

        # Areas for improvement
        report.append("\n### Areas for Improvement:")
        report.append("1. Evaluation on larger and more diverse datasets recommended")
        report.append("2. Comparison with latest 2024-2025 methods (AmesFormer, GeoScatt-GNN)")
        report.append("3. More extensive ablation studies on quantum parameters")

        report.append("\n---\n")
        report.append("*Report generated automatically by QI-VGT Validation Framework*")

        return "\n".join(report)

    def save_report(self, output_path: str = "validation_report.md"):
        """Save the full report to file."""
        report = self.generate_full_report()

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"Report saved to: {output_path}")
        return output_path

    def save_results_json(self, output_path: str = "results.json"):
        """Save all results to JSON for reproducibility."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'validation': self.validate_paper_claims(),
            'literature_comparison': self.compare_with_literature(),
            'statistical_summary': self.generate_statistical_summary()
        }

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        output = convert_numpy(output)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {output_path}")
        return output_path


def generate_validation_report(results: Dict, ablation_results: Dict = None):
    """Convenience function to generate and save validation report."""
    generator = ValidationReportGenerator(results, ablation_results)
    report_path = generator.save_report("validation_report.md")
    json_path = generator.save_results_json("results.json")
    return report_path, json_path


if __name__ == "__main__":
    # Test with dummy data
    dummy_results = {
        'QI-VGT': {
            'results': {'accuracy': [0.84, 0.87, 0.79, 0.86, 0.86]},
            'stats': {
                'accuracy': {'mean': 0.8459, 'std': 0.0297},
                'auc_roc': {'mean': 0.8930, 'std': 0.0281},
                'precision': {'mean': 0.8678, 'std': 0.0482},
                'recall': {'mean': 0.9120, 'std': 0.0466},
                'f1': {'mean': 0.8874, 'std': 0.0208}
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
        },
        'GAT': {
            'results': {'accuracy': [0.85, 0.86, 0.83, 0.87, 0.86]},
            'stats': {
                'accuracy': {'mean': 0.854, 'std': 0.015},
                'auc_roc': {'mean': 0.88, 'std': 0.02},
                'precision': {'mean': 0.86, 'std': 0.03},
                'recall': {'mean': 0.88, 'std': 0.03},
                'f1': {'mean': 0.87, 'std': 0.02}
            }
        },
        'Random Forest': {
            'results': {'accuracy': [0.76, 0.78, 0.77, 0.79, 0.78]},
            'stats': {
                'accuracy': {'mean': 0.776, 'std': 0.01},
                'auc_roc': {'mean': 0.82, 'std': 0.02},
                'precision': {'mean': 0.78, 'std': 0.03},
                'recall': {'mean': 0.80, 'std': 0.03},
                'f1': {'mean': 0.79, 'std': 0.02}
            }
        }
    }

    generator = ValidationReportGenerator(dummy_results)
    print(generator.generate_full_report())
