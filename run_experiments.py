#!/usr/bin/env python3
"""
Main Runner Script for Q1 Journal Quality Research
Quantum-Causal Framework for Imbalanced Cancer Classification

This script executes comprehensive experiments on real UCI datasets,
compares with state-of-the-art methods, and generates publication-quality results.

Usage:
    python run_experiments.py --mode full      # Full experiments (recommended for paper)
    python run_experiments.py --mode quick     # Quick test with subset
    python run_experiments.py --mode custom    # Custom configuration
"""

import sys
import os
import argparse
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import CancerDataLoader, get_dataset_info
from experiments import ExperimentRunner, run_quick_test
from visualization import generate_all_visualizations, PublicationVisualizer
from classifiers import SOTA_BENCHMARKS


def print_header():
    """Print experiment header."""
    print("\n" + "=" * 80)
    print("  QUANTUM-CAUSAL FRAMEWORK FOR IMBALANCED CANCER CLASSIFICATION")
    print("  Q1 Journal Quality Experimental Framework")
    print("=" * 80)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 80 + "\n")


def run_full_experiments():
    """
    Run full experiments for Q1 journal publication.
    Includes all classifiers, imbalance methods, and statistical tests.
    """
    print_header()
    print("MODE: FULL EXPERIMENTS")
    print("This will take approximately 30-60 minutes.\n")

    # Configuration
    DATASET = 'wdbc'
    N_SPLITS = 10
    RANDOM_STATE = 42

    # Create directories
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./figures', exist_ok=True)

    # Show dataset info
    get_dataset_info(DATASET)

    # =========================================================================
    # EXPERIMENT 1: Original WDBC Dataset (relatively balanced)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Original WDBC Dataset")
    print("This represents a relatively balanced real-world scenario")
    print("=" * 80)

    runner1 = ExperimentRunner(
        dataset_name=DATASET,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        results_dir='./results'
    )

    results_balanced = runner1.run_comprehensive_experiments(
        imbalance_ratio=None,
        imbalance_methods=['none', 'smote', 'borderline_smote', 'adasyn',
                          'smote_tomek', 'smote_enn', 'class_weight']
    )

    runner1.save_results('results_balanced.json')
    report1 = runner1.generate_report('./results/report_balanced.txt')

    # =========================================================================
    # EXPERIMENT 2: Severely Imbalanced Dataset (clinical scenario)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Severely Imbalanced WDBC Dataset")
    print("Simulating clinical screening scenario with 6.99:1 imbalance ratio")
    print("This is the KEY experiment addressing the paper's main contribution")
    print("=" * 80)

    runner2 = ExperimentRunner(
        dataset_name='wdbc_imbalanced',
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        results_dir='./results'
    )

    results_imbalanced = runner2.run_comprehensive_experiments(
        imbalance_ratio=6.99,
        imbalance_methods=['none', 'smote', 'borderline_smote', 'adasyn',
                          'smote_tomek', 'smote_enn', 'class_weight']
    )

    runner2.save_results('results_imbalanced.json')
    report2 = runner2.generate_report('./results/report_imbalanced.txt')

    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 80)

    generate_all_visualizations(results_imbalanced, './figures')

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Best configurations from imbalanced experiments
    best_configs = runner2.find_best_configurations('g_mean')[:5]

    print("\nTOP 5 CONFIGURATIONS FOR IMBALANCED DATA (by G-Mean):")
    print("-" * 70)
    for i, config in enumerate(best_configs, 1):
        print(f"{i}. {config['classifier']} + {config['imbalance_method']}")
        print(f"   G-Mean: {config['g_mean_mean']:.4f} | AUC: {config['AUC']:.4f}")
        print(f"   Sensitivity: {config['sensitivity']:.4f} | Specificity: {config['specificity']:.4f}")
        print()

    # Compare with SOTA
    print("\nCOMPARISON WITH STATE-OF-THE-ART:")
    print("-" * 70)
    best = best_configs[0]
    print(f"Our Best: G-Mean={best['g_mean_mean']:.4f}, AUC={best['AUC']:.4f}")
    print(f"          Sensitivity={best['sensitivity']:.4f}, Specificity={best['specificity']:.4f}")
    print()

    sota = SOTA_BENCHMARKS.get('WDBC', {})
    sota_best_auc = max([v.get('AUC', 0) for v in sota.values()])
    sota_best_acc = max([v.get('accuracy', 0) for v in sota.values()])
    print(f"SOTA Best: AUC={sota_best_auc:.4f}, Accuracy={sota_best_acc:.4f}")
    print(f"Key Advantage: Our method handles severe imbalance (6.99:1 ratio)")

    # Key achievement highlight
    print("\n" + "=" * 80)
    print("KEY ACHIEVEMENT")
    print("=" * 80)
    print("SPECIFICITY TRANSFORMATION:")
    if 'none' in results_imbalanced and 'smote' in results_imbalanced:
        baseline_spec = 0
        improved_spec = 0
        for clf_name, results in results_imbalanced['none'].items():
            spec = results.get('specificity', {})
            if isinstance(spec, dict):
                baseline_spec = max(baseline_spec, spec.get('mean', 0))

        for clf_name, results in results_imbalanced['smote'].items():
            spec = results.get('specificity', {})
            if isinstance(spec, dict):
                improved_spec = max(improved_spec, spec.get('mean', 0))

        print(f"  Before (no handling): {baseline_spec*100:.1f}%")
        print(f"  After (with SMOTE):   {improved_spec*100:.1f}%")
        print(f"  Improvement:          +{(improved_spec-baseline_spec)*100:.1f} percentage points")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("Results saved in: ./results/")
    print("Figures saved in: ./figures/")
    print("=" * 80 + "\n")

    return {
        'balanced': results_balanced,
        'imbalanced': results_imbalanced,
        'best_configs': best_configs
    }


def run_quick_experiments():
    """
    Run quick experiments for testing.
    """
    print_header()
    print("MODE: QUICK TEST")
    print("Running with subset of configurations for quick validation.\n")

    # Configuration
    runner = ExperimentRunner(
        dataset_name='wdbc',
        n_splits=5,
        random_state=42,
        results_dir='./results'
    )

    results = runner.run_comprehensive_experiments(
        imbalance_ratio=None,
        imbalance_methods=['none', 'smote', 'borderline_smote'],
        classifier_names=['SVM-RBF', 'Random Forest', 'Logistic Regression', 'MLP']
    )

    print("\n" + runner.generate_report())
    runner.save_results('results_quick_test.json')

    return results


def run_custom_experiments(args):
    """
    Run custom experiments with user-specified parameters.
    """
    print_header()
    print("MODE: CUSTOM EXPERIMENTS\n")

    runner = ExperimentRunner(
        dataset_name=args.dataset,
        n_splits=args.n_splits,
        random_state=args.random_state,
        results_dir=args.output_dir
    )

    imbalance_methods = args.imbalance_methods.split(',') if args.imbalance_methods else None
    classifier_names = args.classifiers.split(',') if args.classifiers else None

    results = runner.run_comprehensive_experiments(
        imbalance_ratio=args.imbalance_ratio,
        imbalance_methods=imbalance_methods,
        classifier_names=classifier_names
    )

    print("\n" + runner.generate_report())
    runner.save_results('results_custom.json')

    if args.generate_figures:
        generate_all_visualizations(results, './figures')

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Quantum-Causal Framework for Imbalanced Cancer Classification'
    )
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['full', 'quick', 'custom'],
                        help='Experiment mode')
    parser.add_argument('--dataset', type=str, default='wdbc',
                        help='Dataset name')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='Number of CV folds')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--imbalance_ratio', type=float, default=None,
                        help='Imbalance ratio (None for original)')
    parser.add_argument('--imbalance_methods', type=str, default=None,
                        help='Comma-separated imbalance methods')
    parser.add_argument('--classifiers', type=str, default=None,
                        help='Comma-separated classifier names')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--generate_figures', action='store_true',
                        help='Generate figures')

    args = parser.parse_args()

    if args.mode == 'full':
        return run_full_experiments()
    elif args.mode == 'quick':
        return run_quick_experiments()
    else:
        return run_custom_experiments(args)


if __name__ == "__main__":
    results = main()
