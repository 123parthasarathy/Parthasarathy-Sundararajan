#!/usr/bin/env python3
"""
Main Runner Script for QI-VGT Paper Validation

This script orchestrates the complete validation pipeline:
1. Loads the MUTAG dataset
2. Trains and evaluates QI-VGT and all baselines
3. Performs ablation studies
4. Generates visualizations
5. Creates comprehensive validation report

Usage:
    python run_validation.py

For Q1 Publication Validation
"""

import sys
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if all required dependencies are available."""
    required = [
        'torch',
        'torch_geometric',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'scipy'
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing dependencies: {missing}")
        print("Install with: pip install torch torch-geometric numpy scikit-learn matplotlib seaborn scipy")
        return False

    return True


def main():
    """Run complete validation pipeline."""
    print("=" * 70)
    print("QI-VGT COMPREHENSIVE VALIDATION PIPELINE")
    print("For Q1 Publication")
    print("=" * 70)

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return

    import torch
    import numpy as np

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Import modules
    from train_evaluate import (
        main as run_experiments,
        run_cross_validation,
        run_ablation_study,
        compute_statistics
    )
    from validation_report import ValidationReportGenerator, generate_validation_report
    from visualizations import generate_all_visualizations

    # Run experiments
    print("\n" + "=" * 70)
    print("PHASE 1: Running Experiments")
    print("=" * 70)

    start_time = time.time()

    try:
        results, ablation_results = run_experiments()
    except Exception as e:
        print(f"Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        return

    experiment_time = time.time() - start_time
    print(f"\nExperiments completed in {experiment_time:.2f} seconds")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("PHASE 2: Generating Visualizations")
    print("=" * 70)

    try:
        os.makedirs('figures', exist_ok=True)
        generate_all_visualizations(results, ablation_results, output_dir='figures')
        print("Visualizations saved to figures/")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

    # Generate validation report
    print("\n" + "=" * 70)
    print("PHASE 3: Generating Validation Report")
    print("=" * 70)

    try:
        report_path, json_path = generate_validation_report(results, ablation_results)
        print(f"\nReport saved to: {report_path}")
        print(f"JSON results saved to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not generate report: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    if 'QI-VGT' in results:
        qi_vgt_stats = results['QI-VGT']['stats']
        print("\nQI-VGT Final Results:")
        print(f"  Accuracy:  {qi_vgt_stats['accuracy']['mean']*100:.2f}% ± {qi_vgt_stats['accuracy']['std']*100:.2f}%")
        print(f"  AUC-ROC:   {qi_vgt_stats['auc_roc']['mean']*100:.2f}% ± {qi_vgt_stats['auc_roc']['std']*100:.2f}%")
        print(f"  Precision: {qi_vgt_stats['precision']['mean']*100:.2f}% ± {qi_vgt_stats['precision']['std']*100:.2f}%")
        print(f"  Recall:    {qi_vgt_stats['recall']['mean']*100:.2f}% ± {qi_vgt_stats['recall']['std']*100:.2f}%")
        print(f"  F1-Score:  {qi_vgt_stats['f1']['mean']*100:.2f}% ± {qi_vgt_stats['f1']['std']*100:.2f}%")

        # Validate claims
        print("\nPaper Claims Validation:")
        paper_claims = {
            'Accuracy': 84.59,
            'AUC-ROC': 89.30,
            'Precision': 86.78,
            'Recall': 91.20,
            'F1-Score': 88.74
        }

        metric_mapping = {
            'Accuracy': 'accuracy',
            'AUC-ROC': 'auc_roc',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1-Score': 'f1'
        }

        all_validated = True
        for claim_name, claim_value in paper_claims.items():
            metric_key = metric_mapping[claim_name]
            achieved = qi_vgt_stats[metric_key]['mean'] * 100
            diff = achieved - claim_value
            validated = abs(diff) <= 3.0
            status = "✅" if validated else "⚠️"
            print(f"  {claim_name}: Claimed={claim_value:.2f}%, Achieved={achieved:.2f}%, Diff={diff:+.2f}% {status}")
            if not validated:
                all_validated = False

        print("\n" + "=" * 70)
        if all_validated:
            print("✅ ALL CLAIMS VALIDATED - PAPER READY FOR Q1 PUBLICATION")
        else:
            print("⚠️ SOME CLAIMS SHOW DEVIATION - REVIEW RECOMMENDED")
        print("=" * 70)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    print("\nGenerated files:")
    print("  - validation_report.md")
    print("  - results.json")
    print("  - figures/ (visualization directory)")

    return results, ablation_results


if __name__ == "__main__":
    main()
