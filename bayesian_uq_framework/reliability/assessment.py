"""
Reliability Assessment Framework

Comprehensive tools for assessing model reliability combining:
- Uncertainty-based reliability scores
- Manifold-based reliability metrics
- Confidence-based filtering
- Out-of-distribution detection
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


class ReliabilityScorer:
    """
    Compute reliability scores combining multiple uncertainty sources.
    """

    @staticmethod
    def compute_reliability_score(
        epistemic_uncertainty: np.ndarray,
        aleatoric_uncertainty: Optional[np.ndarray] = None,
        manifold_distance: Optional[np.ndarray] = None,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Compute composite reliability score.

        Score ranges from 0 (unreliable) to 1 (reliable).

        Args:
            epistemic_uncertainty: Model uncertainty
            aleatoric_uncertainty: Data uncertainty
            manifold_distance: Distance to training manifold
            weights: Weights for each component

        Returns:
            Reliability scores
        """
        if weights is None:
            weights = {
                'epistemic': 0.5,
                'aleatoric': 0.3,
                'manifold': 0.2
            }

        # Normalize uncertainties to [0, 1]
        epistemic_norm = ReliabilityScorer._normalize(epistemic_uncertainty)

        components = {'epistemic': epistemic_norm}

        if aleatoric_uncertainty is not None:
            aleatoric_norm = ReliabilityScorer._normalize(aleatoric_uncertainty)
            components['aleatoric'] = aleatoric_norm

        if manifold_distance is not None:
            manifold_norm = ReliabilityScorer._normalize(manifold_distance)
            components['manifold'] = manifold_norm

        # Compute weighted reliability (1 - weighted uncertainty)
        reliability = np.zeros_like(epistemic_norm)

        total_weight = 0
        for key, component in components.items():
            if key in weights:
                reliability += weights[key] * (1 - component)
                total_weight += weights[key]

        reliability /= total_weight

        return reliability

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-10:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def uncertainty_based_confidence(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter predictions based on confidence threshold.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            confidence_threshold: Minimum confidence (1 - normalized uncertainty)

        Returns:
            confident_predictions, confident_mask
        """
        # Convert uncertainty to confidence
        confidence = 1 - ReliabilityScorer._normalize(uncertainties)

        mask = confidence >= confidence_threshold

        return predictions[mask], mask

    @staticmethod
    def adaptive_threshold(
        uncertainties: np.ndarray,
        target_coverage: float = 0.95
    ) -> float:
        """
        Compute adaptive uncertainty threshold for target coverage.

        Args:
            uncertainties: Uncertainty estimates
            target_coverage: Desired coverage (fraction of data to keep)

        Returns:
            Uncertainty threshold
        """
        percentile = target_coverage * 100
        threshold = np.percentile(uncertainties, percentile)
        return threshold


class OutOfDistributionDetector:
    """
    Detect out-of-distribution samples using multiple methods.
    """

    def __init__(self, method: str = 'isolation_forest'):
        """
        Initialize OOD detector.

        Args:
            method: 'isolation_forest', 'elliptic_envelope', 'uncertainty', or 'ensemble'
        """
        self.method = method
        self.detector = None
        self.train_statistics = None

    def fit(self, train_features: np.ndarray,
            train_uncertainties: Optional[np.ndarray] = None):
        """
        Fit OOD detector on training data.

        Args:
            train_features: Training features
            train_uncertainties: Training uncertainties (optional)
        """
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.detector.fit(train_features)

        elif self.method == 'elliptic_envelope':
            self.detector = EllipticEnvelope(
                contamination=0.1,
                random_state=42
            )
            self.detector.fit(train_features)

        elif self.method == 'uncertainty':
            if train_uncertainties is None:
                raise ValueError("train_uncertainties required for uncertainty method")

            self.train_statistics = {
                'mean': train_uncertainties.mean(),
                'std': train_uncertainties.std()
            }

        elif self.method == 'ensemble':
            # Combine multiple methods
            self.detector = {
                'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
                'elliptic_envelope': EllipticEnvelope(contamination=0.1, random_state=42)
            }

            for detector in self.detector.values():
                detector.fit(train_features)

            if train_uncertainties is not None:
                self.train_statistics = {
                    'mean': train_uncertainties.mean(),
                    'std': train_uncertainties.std()
                }

    def predict(self, test_features: np.ndarray,
                test_uncertainties: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict OOD scores (higher = more likely OOD).

        Args:
            test_features: Test features
            test_uncertainties: Test uncertainties (optional)

        Returns:
            OOD scores (0 = in-distribution, 1 = out-of-distribution)
        """
        if self.method == 'isolation_forest':
            scores = -self.detector.score_samples(test_features)
            # Normalize to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        elif self.method == 'elliptic_envelope':
            scores = -self.detector.score_samples(test_features)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        elif self.method == 'uncertainty':
            if test_uncertainties is None:
                raise ValueError("test_uncertainties required")

            # Z-score based on training statistics
            z_scores = np.abs(
                (test_uncertainties - self.train_statistics['mean']) /
                (self.train_statistics['std'] + 1e-10)
            )

            # Convert to probability-like score
            scores = 1 - np.exp(-z_scores / 2)

        elif self.method == 'ensemble':
            scores_list = []

            # Isolation forest
            if_scores = -self.detector['isolation_forest'].score_samples(test_features)
            if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
            scores_list.append(if_scores)

            # Elliptic envelope
            ee_scores = -self.detector['elliptic_envelope'].score_samples(test_features)
            ee_scores = (ee_scores - ee_scores.min()) / (ee_scores.max() - ee_scores.min() + 1e-10)
            scores_list.append(ee_scores)

            # Uncertainty-based if available
            if test_uncertainties is not None and self.train_statistics is not None:
                z_scores = np.abs(
                    (test_uncertainties - self.train_statistics['mean']) /
                    (self.train_statistics['std'] + 1e-10)
                )
                unc_scores = 1 - np.exp(-z_scores / 2)
                scores_list.append(unc_scores)

            # Average scores
            scores = np.mean(scores_list, axis=0)

        return scores

    def is_ood(self, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary OOD classification.

        Args:
            scores: OOD scores
            threshold: Classification threshold

        Returns:
            Boolean array (True = OOD)
        """
        return scores > threshold


class ManifoldReliabilityAssessment:
    """
    Assess reliability based on manifold geometry.
    """

    def __init__(self, train_embedded: np.ndarray):
        """
        Initialize with training data manifold.

        Args:
            train_embedded: Training data in manifold space
        """
        self.train_embedded = train_embedded
        self.train_center = train_embedded.mean(axis=0)
        self.train_std = train_embedded.std(axis=0)

    def compute_manifold_distance(self, test_embedded: np.ndarray) -> np.ndarray:
        """
        Compute distance to training manifold.

        Args:
            test_embedded: Test data in manifold space

        Returns:
            Distance scores
        """
        from sklearn.neighbors import NearestNeighbors

        # Find nearest training points
        nbrs = NearestNeighbors(n_neighbors=5).fit(self.train_embedded)
        distances, _ = nbrs.kneighbors(test_embedded)

        # Average distance to k nearest neighbors
        avg_distance = distances.mean(axis=1)

        return avg_distance

    def compute_density_ratio(self, test_embedded: np.ndarray,
                             bandwidth: float = 0.1) -> np.ndarray:
        """
        Compute density ratio between test and train manifolds.

        Args:
            test_embedded: Test data in manifold space
            bandwidth: KDE bandwidth

        Returns:
            Density ratio scores (lower = more OOD)
        """
        from sklearn.neighbors import KernelDensity

        # Fit KDE on training data
        kde_train = KernelDensity(bandwidth=bandwidth)
        kde_train.fit(self.train_embedded)

        # Compute density at test points
        test_density = np.exp(kde_train.score_samples(test_embedded))

        # Normalize by maximum training density
        max_train_density = np.exp(kde_train.score_samples(self.train_embedded)).max()

        density_ratio = test_density / (max_train_density + 1e-10)

        return density_ratio

    def identify_interpolation_vs_extrapolation(
        self,
        test_embedded: np.ndarray,
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify whether test points are interpolating or extrapolating.

        Args:
            test_embedded: Test data in manifold space
            threshold: Distance threshold (in units of training std)

        Returns:
            interpolation_mask, extrapolation_mask
        """
        # Compute Mahalanobis-like distance
        centered_test = test_embedded - self.train_center
        normalized_distance = np.linalg.norm(
            centered_test / (self.train_std + 1e-10),
            axis=1
        )

        # Compute training data extent
        centered_train = self.train_embedded - self.train_center
        max_train_distance = np.linalg.norm(
            centered_train / (self.train_std + 1e-10),
            axis=1
        ).max()

        # Classify
        interpolation = normalized_distance <= max_train_distance * threshold
        extrapolation = ~interpolation

        return interpolation, extrapolation


class UncertaintyBasedRejection:
    """
    Rejection mechanism based on uncertainty.
    """

    def __init__(self, uncertainty_threshold: Optional[float] = None,
                 coverage: float = 0.95):
        """
        Initialize rejection mechanism.

        Args:
            uncertainty_threshold: Fixed threshold (if None, use coverage)
            coverage: Desired coverage if adaptive threshold
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.coverage = coverage

    def reject_samples(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Reject high-uncertainty samples.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            threshold: Override threshold

        Returns:
            accepted_predictions, accepted_mask, statistics
        """
        if threshold is None:
            if self.uncertainty_threshold is not None:
                threshold = self.uncertainty_threshold
            else:
                # Adaptive threshold
                threshold = np.percentile(uncertainties, self.coverage * 100)

        accepted_mask = uncertainties <= threshold

        statistics = {
            'n_total': len(predictions),
            'n_accepted': accepted_mask.sum(),
            'n_rejected': (~accepted_mask).sum(),
            'acceptance_rate': accepted_mask.mean(),
            'threshold': threshold,
            'mean_accepted_uncertainty': uncertainties[accepted_mask].mean() if accepted_mask.any() else 0,
            'mean_rejected_uncertainty': uncertainties[~accepted_mask].mean() if (~accepted_mask).any() else 0
        }

        return predictions[accepted_mask], accepted_mask, statistics


class ReliabilityDiagnostics:
    """
    Diagnostic tools for analyzing reliability.
    """

    @staticmethod
    def uncertainty_error_correlation(
        uncertainties: np.ndarray,
        errors: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze correlation between uncertainty and error.

        Args:
            uncertainties: Uncertainty estimates
            errors: Prediction errors

        Returns:
            Correlation statistics
        """
        from scipy.stats import pearsonr, spearmanr

        pearson_corr, pearson_pval = pearsonr(uncertainties, errors)
        spearman_corr, spearman_pval = spearmanr(uncertainties, errors)

        return {
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_pval,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_pval
        }

    @staticmethod
    def oracle_uncertainty_ranking(
        uncertainties: np.ndarray,
        errors: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze whether uncertainty ranks errors correctly.

        Args:
            uncertainties: Uncertainty estimates
            errors: Prediction errors
            n_bins: Number of bins

        Returns:
            Statistics per uncertainty bin
        """
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)

        bin_size = len(uncertainties) // n_bins
        bin_stats = {
            'bin_uncertainties': [],
            'bin_errors': [],
            'bin_sizes': []
        }

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainties)

            bin_indices = sorted_indices[start_idx:end_idx]

            bin_stats['bin_uncertainties'].append(uncertainties[bin_indices].mean())
            bin_stats['bin_errors'].append(errors[bin_indices].mean())
            bin_stats['bin_sizes'].append(len(bin_indices))

        return {
            k: np.array(v) for k, v in bin_stats.items()
        }

    @staticmethod
    def selective_prediction_analysis(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_values: np.ndarray,
        coverage_levels: List[float] = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    ) -> Dict[str, List[float]]:
        """
        Analyze performance at different coverage levels.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            true_values: Ground truth
            coverage_levels: Coverage levels to evaluate

        Returns:
            Performance metrics at each coverage level
        """
        results = {
            'coverage': [],
            'rmse': [],
            'mae': [],
            'mean_uncertainty': []
        }

        for coverage in coverage_levels:
            threshold = np.percentile(uncertainties, coverage * 100)
            mask = uncertainties <= threshold

            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((predictions[mask] - true_values[mask])**2))
                mae = np.mean(np.abs(predictions[mask] - true_values[mask]))
                mean_unc = uncertainties[mask].mean()

                results['coverage'].append(coverage)
                results['rmse'].append(rmse)
                results['mae'].append(mae)
                results['mean_uncertainty'].append(mean_unc)

        return results
