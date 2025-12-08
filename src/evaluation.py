"""
Comprehensive Evaluation Metrics and Statistical Analysis Module
Implements Q1 journal quality evaluation framework
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score, cohen_kappa_score
)
import warnings

warnings.filterwarnings('ignore')


class ComprehensiveMetrics:
    """
    Comprehensive evaluation metrics for imbalanced classification.
    Includes all metrics required for Q1 journal publication.
    """

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of all metrics
        """
        metrics = {}

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['TP'] = tp
        metrics['TN'] = tn
        metrics['FP'] = fp
        metrics['FN'] = fn

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # Sensitivity (Recall, TPR)
        metrics['sensitivity'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall'] = metrics['sensitivity']
        metrics['TPR'] = metrics['sensitivity']

        # Specificity (TNR)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['TNR'] = metrics['specificity']

        # Precision (PPV)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['PPV'] = metrics['precision']

        # Negative Predictive Value (NPV)
        metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0

        # F1-Score
        metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # F2-Score (emphasizes recall)
        beta = 2
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f2_score'] = (1 + beta**2) * (metrics['precision'] * metrics['recall']) / \
                                   (beta**2 * metrics['precision'] + metrics['recall'])
        else:
            metrics['f2_score'] = 0

        # Geometric Mean (G-Mean)
        metrics['g_mean'] = np.sqrt(metrics['sensitivity'] * metrics['specificity'])

        # Matthews Correlation Coefficient
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

        # False Positive Rate (FPR)
        metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # False Negative Rate (FNR)
        metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Positive/Negative Likelihood Ratios
        if metrics['FPR'] > 0:
            metrics['LR_positive'] = metrics['sensitivity'] / metrics['FPR']
        else:
            metrics['LR_positive'] = float('inf')

        if metrics['specificity'] > 0:
            metrics['LR_negative'] = metrics['FNR'] / metrics['specificity']
        else:
            metrics['LR_negative'] = float('inf')

        # Youden's J Statistic
        metrics['youden_j'] = metrics['sensitivity'] + metrics['specificity'] - 1

        # AUC and PR-AUC (if probabilities available)
        if y_proba is not None:
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_proba)
                metrics['PR_AUC'] = average_precision_score(y_true, y_proba)

                # Optimal threshold using Youden's J
                fpr, tpr, thresholds = roc_curve(y_true, y_proba)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                metrics['optimal_threshold'] = thresholds[optimal_idx]
            except Exception:
                metrics['AUC'] = 0.5
                metrics['PR_AUC'] = np.mean(y_true)
                metrics['optimal_threshold'] = 0.5
        else:
            metrics['AUC'] = None
            metrics['PR_AUC'] = None
            metrics['optimal_threshold'] = 0.5

        return metrics

    @staticmethod
    def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval using bootstrap.

        Args:
            values: Array of values
            confidence: Confidence level

        Returns:
            (lower_bound, upper_bound)
        """
        n = len(values)
        mean = np.mean(values)
        se = stats.sem(values)

        # t-distribution for small samples
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

        return mean - h, mean + h

    @staticmethod
    def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray],
                              metric_func: callable,
                              n_bootstrap: int = 1000,
                              confidence: float = 0.95,
                              random_state: int = 42) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals for a metric.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            metric_func: Function to compute metric
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level
            random_state: Random seed

        Returns:
            Dictionary with mean, std, lower, upper bounds
        """
        np.random.seed(random_state)
        n = len(y_true)
        bootstrap_values = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Ensure both classes are present
            if len(np.unique(y_true_boot)) < 2:
                continue

            if y_proba is not None:
                y_proba_boot = y_proba[indices]
                value = metric_func(y_true_boot, y_pred_boot, y_proba_boot)
            else:
                value = metric_func(y_true_boot, y_pred_boot)

            bootstrap_values.append(value)

        bootstrap_values = np.array(bootstrap_values)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_values, 100 * alpha / 2)
        upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))

        return {
            'mean': np.mean(bootstrap_values),
            'std': np.std(bootstrap_values),
            'lower': lower,
            'upper': upper
        }


class StatisticalTests:
    """
    Statistical tests for comparing classifiers.
    Required for Q1 journal publication.
    """

    @staticmethod
    def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
        """
        Perform paired t-test between two classifiers.

        Args:
            scores_a: Cross-validation scores for classifier A
            scores_b: Cross-validation scores for classifier B

        Returns:
            Dictionary with t-statistic and p-value
        """
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }

    @staticmethod
    def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            scores_a: Cross-validation scores for classifier A
            scores_b: Cross-validation scores for classifier B

        Returns:
            Dictionary with statistic and p-value
        """
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            # Fallback if all differences are zero
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant_0.05': False,
                'significant_0.01': False
            }

        return {
            'statistic': stat,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }

    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                     y_pred_b: np.ndarray) -> Dict[str, float]:
        """
        Perform McNemar's test for comparing two classifiers on same test set.

        Args:
            y_true: True labels
            y_pred_a: Predictions from classifier A
            y_pred_b: Predictions from classifier B

        Returns:
            Dictionary with chi-square statistic and p-value
        """
        # Build contingency table
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # b: A correct, B wrong; c: A wrong, B correct
        b = np.sum(correct_a & ~correct_b)
        c = np.sum(~correct_a & correct_b)

        # McNemar's chi-square statistic
        if b + c > 0:
            chi2 = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2 = 0
            p_value = 1.0

        return {
            'chi_square': chi2,
            'p_value': p_value,
            'b': b,
            'c': c,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }

    @staticmethod
    def friedman_test(scores_matrix: np.ndarray) -> Dict[str, float]:
        """
        Perform Friedman test for comparing multiple classifiers.

        Args:
            scores_matrix: Matrix of shape (n_datasets, n_classifiers)

        Returns:
            Dictionary with test results
        """
        stat, p_value = stats.friedmanchisquare(*scores_matrix.T)

        return {
            'statistic': stat,
            'p_value': p_value,
            'significant_0.05': p_value < 0.05,
            'significant_0.01': p_value < 0.01
        }

    @staticmethod
    def nemenyi_post_hoc(scores_matrix: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Perform Nemenyi post-hoc test after Friedman test.

        Args:
            scores_matrix: Matrix of shape (n_datasets, n_classifiers)
            alpha: Significance level

        Returns:
            Dictionary with critical difference and rankings
        """
        n_datasets, n_classifiers = scores_matrix.shape

        # Rank classifiers for each dataset (higher is better)
        ranks = np.zeros_like(scores_matrix)
        for i in range(n_datasets):
            ranks[i] = stats.rankdata(-scores_matrix[i])  # Negative for descending

        # Average ranks
        avg_ranks = np.mean(ranks, axis=0)

        # Critical difference (CD)
        # Nemenyi critical value from tables (approximation)
        q_alpha = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
            6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
        }
        q = q_alpha.get(n_classifiers, 3.0)  # Default for large k

        cd = q * np.sqrt(n_classifiers * (n_classifiers + 1) / (6 * n_datasets))

        return {
            'average_ranks': avg_ranks,
            'critical_difference': cd,
            'n_datasets': n_datasets,
            'n_classifiers': n_classifiers
        }


class CrossValidationEvaluator:
    """
    Comprehensive cross-validation evaluation framework.
    """

    def __init__(self, n_splits: int = 10, random_state: int = 42):
        """
        Initialize evaluator.

        Args:
            n_splits: Number of CV folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = defaultdict(list)

    def evaluate_fold(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None,
                      fold: int = 0) -> Dict[str, float]:
        """
        Evaluate a single fold.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            fold: Fold number

        Returns:
            Dictionary of metrics
        """
        metrics = ComprehensiveMetrics.compute_all_metrics(y_true, y_pred, y_proba)
        metrics['fold'] = fold

        # Store results
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                self.results[key].append(value)

        return metrics

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all folds.

        Returns:
            Dictionary with mean, std, CI for each metric
        """
        summary = {}
        for metric, values in self.results.items():
            if metric == 'fold':
                continue

            values = np.array([v for v in values if v is not None and np.isfinite(v)])
            if len(values) == 0:
                continue

            lower, upper = ComprehensiveMetrics.compute_confidence_interval(values)

            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'ci_lower': lower,
                'ci_upper': upper
            }

        return summary

    def reset(self):
        """Reset stored results."""
        self.results = defaultdict(list)


def format_metric_with_ci(mean: float, ci_lower: float, ci_upper: float,
                          precision: int = 3) -> str:
    """
    Format metric with confidence interval for publication.

    Args:
        mean: Mean value
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
        precision: Decimal precision

    Returns:
        Formatted string
    """
    return f"{mean:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"


def create_results_table(results: Dict[str, Dict], metrics: List[str],
                         precision: int = 4) -> str:
    """
    Create a publication-ready results table.

    Args:
        results: Dictionary mapping model names to their metrics
        metrics: List of metrics to include
        precision: Decimal precision

    Returns:
        Formatted table string
    """
    # Header
    header = "| Model |"
    for metric in metrics:
        header += f" {metric} |"
    header += "\n"

    # Separator
    sep = "|-------|"
    for _ in metrics:
        sep += "-------|"
    sep += "\n"

    # Rows
    rows = ""
    for model_name, model_results in results.items():
        row = f"| {model_name} |"
        for metric in metrics:
            if metric in model_results:
                value = model_results[metric]
                if isinstance(value, dict) and 'mean' in value:
                    row += f" {value['mean']:.{precision}f} |"
                else:
                    row += f" {value:.{precision}f} |"
            else:
                row += " N/A |"
        rows += row + "\n"

    return header + sep + rows


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing Evaluation Metrics...")

    # Generate test data
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive
    y_proba = np.clip(y_true + np.random.randn(n_samples) * 0.3, 0, 1)
    y_pred = (y_proba > 0.5).astype(int)

    # Compute metrics
    metrics = ComprehensiveMetrics.compute_all_metrics(y_true, y_pred, y_proba)

    print("\nComprehensive Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None:
            print(f"  {key}: {value:.4f}")

    # Test statistical tests
    print("\nStatistical Tests:")
    scores_a = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
    scores_b = np.array([0.82, 0.84, 0.80, 0.83, 0.81])

    t_test = StatisticalTests.paired_t_test(scores_a, scores_b)
    print(f"  Paired t-test: p={t_test['p_value']:.4f}")

    wilcoxon = StatisticalTests.wilcoxon_test(scores_a, scores_b)
    print(f"  Wilcoxon test: p={wilcoxon['p_value']:.4f}")
