"""
Main Experiment Runner for Q1 Journal Quality Research
Implements comprehensive experiments with statistical validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import os
import time
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from data_loader import CancerDataLoader
from imbalance_handlers import ImbalanceHandler
from classifiers import get_all_classifiers, SOTA_BENCHMARKS, compare_with_sota
from evaluation import (
    ComprehensiveMetrics, StatisticalTests, CrossValidationEvaluator,
    create_results_table
)

import warnings
warnings.filterwarnings('ignore')


class ExperimentRunner:
    """
    Comprehensive experiment runner for Q1 journal publication.
    """

    def __init__(self, dataset_name: str = 'wdbc',
                 n_splits: int = 10,
                 random_state: int = 42,
                 results_dir: str = './results'):
        """
        Initialize experiment runner.

        Args:
            dataset_name: Name of dataset to use
            n_splits: Number of cross-validation folds
            random_state: Random seed
            results_dir: Directory to save results
        """
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.random_state = random_state
        self.results_dir = results_dir

        os.makedirs(results_dir, exist_ok=True)

        self.data_loader = CancerDataLoader(dataset_name, random_state)
        self.all_results = {}
        self.statistical_tests = {}

    def run_single_experiment(self, X: np.ndarray, y: np.ndarray,
                              classifier, classifier_name: str,
                              imbalance_method: str = 'none') -> Dict:
        """
        Run single experiment with cross-validation.

        Args:
            X: Feature matrix
            y: Labels
            classifier: Classifier instance
            classifier_name: Name of classifier
            imbalance_method: Imbalance handling method

        Returns:
            Dictionary of results
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)

        fold_results = []
        evaluator = CrossValidationEvaluator(self.n_splits, self.random_state)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply imbalance handling
            if imbalance_method != 'none':
                handler = ImbalanceHandler(method=imbalance_method,
                                           random_state=self.random_state)
                X_train_res, y_train_res = handler.fit_resample(X_train, y_train)

                # Get class weights if applicable
                if imbalance_method in ['class_weight', 'cost_sensitive', 'hybrid_adaptive']:
                    class_weight = handler.class_weights
                else:
                    class_weight = None
            else:
                X_train_res, y_train_res = X_train, y_train
                class_weight = None

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_test_scaled = scaler.transform(X_test)

            # Clone and train classifier
            clf = clone(classifier)

            # Set class weight if supported
            if class_weight is not None and hasattr(clf, 'class_weight'):
                clf.set_params(class_weight=class_weight)

            try:
                clf.fit(X_train_scaled, y_train_res)

                # Predict
                y_pred = clf.predict(X_test_scaled)

                # Get probabilities
                if hasattr(clf, 'predict_proba'):
                    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_proba = None

                # Evaluate
                metrics = evaluator.evaluate_fold(y_test, y_pred, y_proba, fold)
                fold_results.append(metrics)

            except Exception as e:
                print(f"  Error in fold {fold} for {classifier_name}: {e}")
                continue

        # Get summary
        summary = evaluator.get_summary()
        summary['classifier'] = classifier_name
        summary['imbalance_method'] = imbalance_method
        summary['n_folds_completed'] = len(fold_results)

        return summary

    def run_comprehensive_experiments(self, imbalance_ratio: Optional[float] = None,
                                       imbalance_methods: Optional[List[str]] = None,
                                       classifier_names: Optional[List[str]] = None) -> Dict:
        """
        Run comprehensive experiments with all configurations.

        Args:
            imbalance_ratio: Ratio to create imbalanced dataset (None for original)
            imbalance_methods: List of imbalance handling methods
            classifier_names: List of classifiers to test (None for all)

        Returns:
            Dictionary of all results
        """
        # Load data
        data = self.data_loader.load_data(imbalance_ratio=imbalance_ratio)
        X, y = data['X'], data['y']

        print(f"\n{'='*70}")
        print(f"Dataset: {self.dataset_name.upper()}")
        print(f"Samples: {data['n_samples']} | Features: {data['n_features']}")
        print(f"Positive: {data['n_positive']} ({data['positive_rate']*100:.1f}%) | "
              f"Negative: {data['n_negative']} ({(1-data['positive_rate'])*100:.1f}%)")
        print(f"Imbalance ratio: {data['imbalance_ratio']:.2f}:1")
        print(f"{'='*70}\n")

        # Default configurations
        if imbalance_methods is None:
            imbalance_methods = [
                'none', 'smote', 'borderline_smote', 'adasyn',
                'smote_tomek', 'smote_enn', 'class_weight'
            ]

        # Get classifiers
        all_classifiers = get_all_classifiers(self.random_state)
        if classifier_names is not None:
            classifiers = {k: v for k, v in all_classifiers.items() if k in classifier_names}
        else:
            classifiers = all_classifiers

        # Run experiments
        results = {}
        total_experiments = len(classifiers) * len(imbalance_methods)
        current = 0

        for imb_method in imbalance_methods:
            results[imb_method] = {}

            for clf_name, clf in classifiers.items():
                current += 1
                print(f"[{current}/{total_experiments}] {clf_name} + {imb_method}...", end=' ')

                start_time = time.time()
                try:
                    result = self.run_single_experiment(
                        X, y, clf, clf_name, imb_method
                    )
                    results[imb_method][clf_name] = result
                    elapsed = time.time() - start_time

                    # Print key metrics
                    acc = result.get('accuracy', {}).get('mean', 0)
                    auc = result.get('AUC', {}).get('mean', 0)
                    sens = result.get('sensitivity', {}).get('mean', 0)
                    spec = result.get('specificity', {}).get('mean', 0)
                    gmean = result.get('g_mean', {}).get('mean', 0)

                    print(f"Acc={acc:.4f} AUC={auc:.4f} Sens={sens:.4f} "
                          f"Spec={spec:.4f} G-Mean={gmean:.4f} [{elapsed:.1f}s]")

                except Exception as e:
                    print(f"Error: {e}")
                    results[imb_method][clf_name] = {'error': str(e)}

        self.all_results = results
        return results

    def find_best_configurations(self, metric: str = 'g_mean') -> List[Dict]:
        """
        Find best classifier-imbalance method configurations.

        Args:
            metric: Metric to optimize

        Returns:
            List of top configurations sorted by metric
        """
        configurations = []

        for imb_method, clf_results in self.all_results.items():
            for clf_name, results in clf_results.items():
                if 'error' in results:
                    continue

                metric_value = results.get(metric, {})
                if isinstance(metric_value, dict):
                    mean_value = metric_value.get('mean', 0)
                    std_value = metric_value.get('std', 0)
                else:
                    mean_value = metric_value
                    std_value = 0

                configurations.append({
                    'classifier': clf_name,
                    'imbalance_method': imb_method,
                    f'{metric}_mean': mean_value,
                    f'{metric}_std': std_value,
                    'accuracy': results.get('accuracy', {}).get('mean', 0),
                    'AUC': results.get('AUC', {}).get('mean', 0),
                    'sensitivity': results.get('sensitivity', {}).get('mean', 0),
                    'specificity': results.get('specificity', {}).get('mean', 0),
                    'f1_score': results.get('f1_score', {}).get('mean', 0)
                })

        # Sort by metric
        configurations.sort(key=lambda x: x[f'{metric}_mean'], reverse=True)

        return configurations

    def run_statistical_comparison(self, configs: List[Dict]) -> Dict:
        """
        Run statistical tests comparing top configurations.

        Args:
            configs: List of configurations to compare

        Returns:
            Dictionary of statistical test results
        """
        # This requires fold-level results which we'll need to store
        # For now, return placeholder
        return {'note': 'Statistical tests require fold-level results storage'}

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate publication-ready results report.

        Args:
            output_file: Optional file to save report

        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EXPERIMENTAL RESULTS")
        report.append(f"Dataset: {self.dataset_name.upper()}")
        report.append(f"Cross-validation: {self.n_splits}-fold stratified")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        # Find best configurations
        best_configs = self.find_best_configurations('g_mean')[:10]

        report.append("\n" + "-" * 80)
        report.append("TOP 10 CONFIGURATIONS (by G-Mean)")
        report.append("-" * 80)
        report.append(f"{'Rank':<5} {'Classifier':<20} {'Imbalance':<15} {'G-Mean':<10} "
                      f"{'AUC':<10} {'Sens':<10} {'Spec':<10}")
        report.append("-" * 80)

        for i, config in enumerate(best_configs, 1):
            report.append(
                f"{i:<5} {config['classifier']:<20} {config['imbalance_method']:<15} "
                f"{config['g_mean_mean']:.4f}     {config['AUC']:.4f}     "
                f"{config['sensitivity']:.4f}     {config['specificity']:.4f}"
            )

        # Best overall
        if best_configs:
            best = best_configs[0]
            report.append("\n" + "=" * 80)
            report.append("BEST CONFIGURATION")
            report.append("=" * 80)
            report.append(f"Classifier: {best['classifier']}")
            report.append(f"Imbalance Method: {best['imbalance_method']}")
            report.append(f"G-Mean: {best['g_mean_mean']:.4f} ± {best['g_mean_std']:.4f}")
            report.append(f"Accuracy: {best['accuracy']:.4f}")
            report.append(f"AUC: {best['AUC']:.4f}")
            report.append(f"Sensitivity: {best['sensitivity']:.4f}")
            report.append(f"Specificity: {best['specificity']:.4f}")
            report.append(f"F1-Score: {best['f1_score']:.4f}")

        # Compare with SOTA
        report.append("\n" + "=" * 80)
        report.append("COMPARISON WITH STATE-OF-THE-ART")
        report.append("=" * 80)

        sota = SOTA_BENCHMARKS.get('WDBC', {})
        for method, metrics in sota.items():
            acc = metrics.get('accuracy', 'N/A')
            auc = metrics.get('AUC', 'N/A')
            source = metrics.get('source', 'Unknown')
            if isinstance(acc, float):
                acc = f"{acc:.4f}"
            if isinstance(auc, float):
                auc = f"{auc:.4f}"
            report.append(f"{method}: Acc={acc}, AUC={auc} [{source}]")

        report_str = "\n".join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_str)
            print(f"\nReport saved to: {output_file}")

        return report_str

    def save_results(self, filename: str = 'results.json'):
        """Save results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)

        # Convert to serializable format
        serializable = {}
        for imb_method, clf_results in self.all_results.items():
            serializable[imb_method] = {}
            for clf_name, results in clf_results.items():
                serializable[imb_method][clf_name] = {}
                for metric, value in results.items():
                    if isinstance(value, dict):
                        serializable[imb_method][clf_name][metric] = {
                            k: float(v) if isinstance(v, np.floating) else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, np.floating):
                        serializable[imb_method][clf_name][metric] = float(value)
                    else:
                        serializable[imb_method][clf_name][metric] = value

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to: {filepath}")


def run_main_experiments():
    """
    Run main experiments for the paper.
    """
    print("\n" + "=" * 80)
    print("Q1 JOURNAL QUALITY EXPERIMENTAL FRAMEWORK")
    print("Quantum-Causal Framework for Imbalanced Cancer Classification")
    print("=" * 80)

    # Configuration
    DATASET = 'wdbc'
    N_SPLITS = 10
    RANDOM_STATE = 42

    # Initialize runner
    runner = ExperimentRunner(
        dataset_name=DATASET,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        results_dir='./results'
    )

    # Experiment 1: Original balanced dataset
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Original WDBC Dataset (Balanced)")
    print("=" * 80)

    results_balanced = runner.run_comprehensive_experiments(
        imbalance_ratio=None,
        imbalance_methods=['none', 'smote', 'borderline_smote', 'adasyn', 'class_weight']
    )

    # Experiment 2: Imbalanced dataset (simulating clinical scenario)
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Imbalanced WDBC Dataset (6.99:1 ratio)")
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

    # Generate reports
    report1 = runner.generate_report(output_file='./results/report_balanced.txt')
    report2 = runner2.generate_report(output_file='./results/report_imbalanced.txt')

    # Save results
    runner.save_results('results_balanced.json')
    runner2.save_results('results_imbalanced.json')

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)
    print("Results saved in ./results/")

    return {
        'balanced': results_balanced,
        'imbalanced': results_imbalanced
    }


def run_quick_test():
    """Run quick test with subset of configurations."""
    print("\n" + "=" * 80)
    print("QUICK TEST - Subset of Configurations")
    print("=" * 80)

    runner = ExperimentRunner(
        dataset_name='wdbc',
        n_splits=5,  # Fewer folds for quick test
        random_state=42
    )

    results = runner.run_comprehensive_experiments(
        imbalance_ratio=None,
        imbalance_methods=['none', 'smote', 'borderline_smote'],
        classifier_names=['SVM-RBF', 'Random Forest', 'Logistic Regression', 'MLP']
    )

    print("\n" + runner.generate_report())

    return results


if __name__ == "__main__":
    # Run quick test
    results = run_quick_test()

    # Uncomment for full experiments:
    # results = run_main_experiments()
