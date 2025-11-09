"""
Uncertainty Quantification Metrics

Comprehensive metrics for measuring and analyzing uncertainty in predictions:
- Entropy-based metrics
- Mutual information
- Predictive variance
- Calibration metrics
- Confidence intervals
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


class UncertaintyMetrics:
    """Collection of uncertainty quantification metrics."""

    @staticmethod
    def predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy H[y|x,D].

        Args:
            predictions: Tensor of shape (n_samples, batch_size, n_classes)

        Returns:
            Entropy values for each input
        """
        # Average predictions across samples
        mean_pred = predictions.mean(dim=0)

        # Compute entropy
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
        return entropy

    @staticmethod
    def mutual_information(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information I[y; w|x,D] (epistemic uncertainty).

        MI = H[E[y|x,w]] - E[H[y|x,w]]

        Args:
            predictions: Tensor of shape (n_samples, batch_size, n_classes)

        Returns:
            Mutual information for each input
        """
        # Expected entropy
        entropy_per_sample = -torch.sum(
            predictions * torch.log(predictions + 1e-10), dim=-1
        )
        expected_entropy = entropy_per_sample.mean(dim=0)

        # Entropy of expected
        mean_pred = predictions.mean(dim=0)
        entropy_of_expected = -torch.sum(
            mean_pred * torch.log(mean_pred + 1e-10), dim=-1
        )

        # Mutual information (epistemic uncertainty)
        mi = entropy_of_expected - expected_entropy
        return mi

    @staticmethod
    def expected_pairwise_kl(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute expected pairwise KL divergence between predictions.

        Measures disagreement between different posterior samples.

        Args:
            predictions: Tensor of shape (n_samples, batch_size, n_classes)

        Returns:
            Expected pairwise KL for each input
        """
        n_samples = predictions.shape[0]
        kl_sum = 0

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                kl = torch.sum(
                    predictions[i] * (
                        torch.log(predictions[i] + 1e-10) -
                        torch.log(predictions[j] + 1e-10)
                    ),
                    dim=-1
                )
                kl_sum += kl

        # Normalize
        n_pairs = n_samples * (n_samples - 1) / 2
        return kl_sum / n_pairs

    @staticmethod
    def variation_ratio(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute variation ratio (epistemic uncertainty for classification).

        VR = 1 - (frequency of modal class) / (number of samples)

        Args:
            predictions: Tensor of shape (n_samples, batch_size, n_classes)

        Returns:
            Variation ratio for each input
        """
        # Get predicted class for each sample
        predicted_classes = predictions.argmax(dim=-1)

        # Count mode frequency
        mode_count = torch.stack([
            torch.bincount(predicted_classes[:, i],
                          minlength=predictions.shape[-1]).max()
            for i in range(predictions.shape[1])
        ])

        vr = 1 - mode_count.float() / predictions.shape[0]
        return vr

    @staticmethod
    def predictive_variance(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive variance (for regression).

        Args:
            predictions: Tensor of shape (n_samples, batch_size, output_dim)

        Returns:
            Variance for each input and output dimension
        """
        return predictions.var(dim=0)

    @staticmethod
    def prediction_interval(predictions: torch.Tensor,
                           confidence: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prediction intervals.

        Args:
            predictions: Tensor of shape (n_samples, batch_size, output_dim)
            confidence: Confidence level (default: 95%)

        Returns:
            lower_bound, upper_bound
        """
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = torch.quantile(predictions, lower_percentile / 100, dim=0)
        upper = torch.quantile(predictions, upper_percentile / 100, dim=0)

        return lower, upper

    @staticmethod
    def coefficient_of_variation(predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute coefficient of variation (normalized uncertainty).

        CoV = std / |mean|

        Args:
            predictions: Tensor of shape (n_samples, batch_size, output_dim)

        Returns:
            Coefficient of variation
        """
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        cov = std / (torch.abs(mean) + 1e-10)
        return cov


class CalibrationMetrics:
    """Metrics for assessing calibration quality."""

    @staticmethod
    def expected_calibration_error(confidences: np.ndarray,
                                   accuracies: np.ndarray,
                                   n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            confidences: Predicted confidence values
            accuracies: Binary correctness (1 if correct, 0 otherwise)
            n_bins: Number of bins

        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0

        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (confidences >= bin_boundaries[i]) & \
                     (confidences < bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()
                bin_weight = in_bin.sum() / len(confidences)

                ece += bin_weight * abs(bin_confidence - bin_accuracy)

        return ece

    @staticmethod
    def maximum_calibration_error(confidences: np.ndarray,
                                 accuracies: np.ndarray,
                                 n_bins: int = 10) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Args:
            confidences: Predicted confidence values
            accuracies: Binary correctness
            n_bins: Number of bins

        Returns:
            MCE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0

        for i in range(n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & \
                     (confidences < bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                bin_confidence = confidences[in_bin].mean()
                bin_accuracy = accuracies[in_bin].mean()

                mce = max(mce, abs(bin_confidence - bin_accuracy))

        return mce

    @staticmethod
    def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Brier score.

        Args:
            probabilities: Predicted probabilities
            labels: True labels (one-hot encoded)

        Returns:
            Brier score
        """
        return np.mean(np.sum((probabilities - labels) ** 2, axis=1))

    @staticmethod
    def negative_log_likelihood(probabilities: np.ndarray,
                               labels: np.ndarray) -> float:
        """
        Compute negative log-likelihood.

        Args:
            probabilities: Predicted probabilities
            labels: True class indices

        Returns:
            NLL value
        """
        n_samples = len(labels)
        log_probs = np.log(probabilities[np.arange(n_samples), labels] + 1e-10)
        return -np.mean(log_probs)


class RegressionUncertaintyMetrics:
    """Specialized metrics for regression uncertainty."""

    @staticmethod
    def prediction_interval_coverage_probability(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> float:
        """
        Compute Prediction Interval Coverage Probability (PICP).

        Fraction of true values within prediction intervals.
        """
        in_interval = (y_true >= lower) & (y_true <= upper)
        return np.mean(in_interval)

    @staticmethod
    def mean_prediction_interval_width(
        lower: np.ndarray,
        upper: np.ndarray
    ) -> float:
        """
        Compute Mean Prediction Interval Width (MPIW).
        """
        return np.mean(upper - lower)

    @staticmethod
    def calibration_regression_score(
        y_true: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Compute calibration metrics for regression.

        Args:
            y_true: True values
            predictions: Predicted means
            uncertainties: Predicted standard deviations
            n_bins: Number of bins

        Returns:
            Dictionary with calibration metrics
        """
        # Compute standardized errors
        errors = np.abs(y_true - predictions)
        std_errors = errors / (uncertainties + 1e-10)

        # Expected vs observed error
        sorted_indices = np.argsort(uncertainties)
        bin_size = len(uncertainties) // n_bins

        expected_errors = []
        observed_errors = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainties)

            bin_indices = sorted_indices[start_idx:end_idx]

            expected_errors.append(np.mean(uncertainties[bin_indices]))
            observed_errors.append(np.mean(errors[bin_indices]))

        # Compute miscalibration area
        expected_errors = np.array(expected_errors)
        observed_errors = np.array(observed_errors)
        miscalibration_area = np.mean(np.abs(expected_errors - observed_errors))

        return {
            'miscalibration_area': miscalibration_area,
            'expected_errors': expected_errors,
            'observed_errors': observed_errors
        }

    @staticmethod
    def uncertainty_based_rmse(
        y_true: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        threshold_percentile: float = 90
    ) -> Dict[str, float]:
        """
        Compute RMSE for high vs low uncertainty predictions.

        Args:
            y_true: True values
            predictions: Predicted values
            uncertainties: Uncertainty estimates
            threshold_percentile: Percentile to split high/low uncertainty

        Returns:
            Dictionary with RMSE for different uncertainty levels
        """
        threshold = np.percentile(uncertainties, threshold_percentile)

        high_unc_mask = uncertainties >= threshold
        low_unc_mask = uncertainties < threshold

        rmse_high = np.sqrt(mean_squared_error(
            y_true[high_unc_mask],
            predictions[high_unc_mask]
        )) if high_unc_mask.sum() > 0 else 0

        rmse_low = np.sqrt(mean_squared_error(
            y_true[low_unc_mask],
            predictions[low_unc_mask]
        )) if low_unc_mask.sum() > 0 else 0

        rmse_all = np.sqrt(mean_squared_error(y_true, predictions))

        return {
            'rmse_all': rmse_all,
            'rmse_high_uncertainty': rmse_high,
            'rmse_low_uncertainty': rmse_low,
            'uncertainty_threshold': threshold
        }


class UncertaintyDecomposition:
    """Methods for decomposing uncertainty into components."""

    @staticmethod
    def decompose_predictive_variance(
        predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose predictive variance into aleatoric and epistemic components.

        For regression with ensemble outputting (mean, variance) pairs.

        Args:
            predictions: Tensor of shape (n_models, batch_size, 2)
                        where last dim is [mean, variance]

        Returns:
            Dictionary with total, aleatoric, and epistemic uncertainty
        """
        means = predictions[:, :, 0]
        variances = predictions[:, :, 1]

        # Aleatoric uncertainty (expected variance)
        aleatoric = variances.mean(dim=0)

        # Epistemic uncertainty (variance of means)
        epistemic = means.var(dim=0)

        # Total uncertainty
        total = aleatoric + epistemic

        return {
            'total': total,
            'aleatoric': aleatoric,
            'epistemic': epistemic
        }

    @staticmethod
    def quantile_based_decomposition(
        predictions: torch.Tensor,
        quantiles: list = [0.05, 0.25, 0.5, 0.75, 0.95]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantile-based uncertainty measures.

        Args:
            predictions: Tensor of shape (n_samples, batch_size, output_dim)
            quantiles: List of quantiles to compute

        Returns:
            Dictionary with quantile values
        """
        result = {}
        for q in quantiles:
            result[f'q{int(q*100)}'] = torch.quantile(predictions, q, dim=0)

        # Interquartile range
        result['iqr'] = result['q75'] - result['q25']

        # 90% prediction interval width
        result['pi90_width'] = result['q95'] - result['q05']

        return result
