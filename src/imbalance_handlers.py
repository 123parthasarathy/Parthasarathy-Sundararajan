"""
Advanced Imbalance Handling Module
Implements multiple state-of-the-art resampling and hybrid techniques
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from collections import Counter

from imblearn.over_sampling import (
    SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE, KMeansSMOTE
)
from imblearn.under_sampling import (
    TomekLinks, EditedNearestNeighbours, ClusterCentroids,
    NearMiss, RandomUnderSampler
)
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils import compute_class_weight


class ImbalanceHandler:
    """
    Comprehensive imbalance handling with multiple strategies.
    Designed for Q1 journal quality experimental evaluation.
    """

    SUPPORTED_METHODS = [
        'none',
        'smote',
        'borderline_smote',
        'adasyn',
        'svm_smote',
        'kmeans_smote',
        'smote_tomek',
        'smote_enn',
        'class_weight',
        'hybrid_adaptive',
        'cost_sensitive'
    ]

    def __init__(self, method: str = 'smote', random_state: int = 42,
                 sampling_strategy: str = 'auto', k_neighbors: int = 5):
        """
        Initialize imbalance handler.

        Args:
            method: Resampling method to use
            random_state: Random seed
            sampling_strategy: Sampling strategy ('auto', 'minority', 'not minority', float)
            k_neighbors: Number of neighbors for SMOTE variants
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method {method} not supported. Choose from {self.SUPPORTED_METHODS}")

        self.method = method
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.sampler = None
        self.class_weights = None

    def _get_sampler(self):
        """Get the appropriate sampler based on method."""
        samplers = {
            'smote': SMOTE(
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy
            ),
            'borderline_smote': BorderlineSMOTE(
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy,
                kind='borderline-1'  # Focus on borderline samples
            ),
            'adasyn': ADASYN(
                random_state=self.random_state,
                n_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy
            ),
            'svm_smote': SVMSMOTE(
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy
            ),
            'kmeans_smote': KMeansSMOTE(
                random_state=self.random_state,
                k_neighbors=self.k_neighbors,
                sampling_strategy=self.sampling_strategy,
                kmeans_estimator=None,
                cluster_balance_threshold='auto'
            ),
            'smote_tomek': SMOTETomek(
                random_state=self.random_state,
                smote=SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors)
            ),
            'smote_enn': SMOTEENN(
                random_state=self.random_state,
                smote=SMOTE(random_state=self.random_state, k_neighbors=self.k_neighbors),
                enn=EditedNearestNeighbours(n_neighbors=3)
            )
        }
        return samplers.get(self.method, None)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply resampling to balance the dataset.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            X_resampled, y_resampled
        """
        if self.method == 'none':
            return X, y

        if self.method in ['class_weight', 'cost_sensitive']:
            # These don't modify data, only compute weights
            self.class_weights = self._compute_class_weights(y)
            return X, y

        if self.method == 'hybrid_adaptive':
            return self._hybrid_adaptive_sampling(X, y)

        self.sampler = self._get_sampler()
        if self.sampler is None:
            return X, y

        try:
            X_res, y_res = self.sampler.fit_resample(X, y)
            return X_res, y_res
        except ValueError as e:
            # Fallback to standard SMOTE if method fails
            print(f"Warning: {self.method} failed ({e}), falling back to SMOTE")
            self.sampler = SMOTE(random_state=self.random_state, k_neighbors=min(3, self.k_neighbors))
            return self.sampler.fit_resample(X, y)

    def _hybrid_adaptive_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hybrid adaptive sampling combining multiple strategies.
        1. Apply Borderline-SMOTE for focused oversampling
        2. Clean with Tomek links
        3. Apply cost-sensitive weighting
        """
        # Step 1: Borderline-SMOTE
        try:
            smote = BorderlineSMOTE(
                random_state=self.random_state,
                k_neighbors=min(self.k_neighbors, self._get_min_class_samples(y) - 1),
                sampling_strategy=0.7  # Don't fully balance
            )
            X_res, y_res = smote.fit_resample(X, y)
        except:
            X_res, y_res = X, y

        # Step 2: Clean with Tomek links
        try:
            tomek = TomekLinks()
            X_res, y_res = tomek.fit_resample(X_res, y_res)
        except:
            pass

        # Step 3: Compute class weights for remaining imbalance
        self.class_weights = self._compute_class_weights(y_res)

        return X_res, y_res

    def _get_min_class_samples(self, y: np.ndarray) -> int:
        """Get number of samples in minority class."""
        counter = Counter(y)
        return min(counter.values())

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights.

        Args:
            y: Labels

        Returns:
            Dictionary mapping class labels to weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return {c: w for c, w in zip(classes, weights)}

    def get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Get per-sample weights for training.

        Args:
            y: Labels

        Returns:
            Array of sample weights
        """
        if self.class_weights is None:
            self.class_weights = self._compute_class_weights(y)

        return np.array([self.class_weights[label] for label in y])

    def get_info(self, y_original: np.ndarray, y_resampled: np.ndarray) -> Dict:
        """
        Get information about the resampling operation.

        Args:
            y_original: Original labels
            y_resampled: Resampled labels

        Returns:
            Dictionary with resampling statistics
        """
        orig_counter = Counter(y_original)
        res_counter = Counter(y_resampled)

        return {
            'method': self.method,
            'original_distribution': dict(orig_counter),
            'resampled_distribution': dict(res_counter),
            'original_total': len(y_original),
            'resampled_total': len(y_resampled),
            'samples_added': len(y_resampled) - len(y_original),
            'original_ratio': max(orig_counter.values()) / min(orig_counter.values()),
            'resampled_ratio': max(res_counter.values()) / min(res_counter.values()) if len(res_counter) > 1 else 1.0
        }


class CostSensitiveLearning:
    """
    Cost-sensitive learning utilities for imbalanced classification.
    """

    @staticmethod
    def get_class_weights(y: np.ndarray, strategy: str = 'balanced') -> Dict[int, float]:
        """
        Compute class weights using various strategies.

        Args:
            y: Labels
            strategy: 'balanced', 'sqrt_inverse', 'effective_number'

        Returns:
            Dictionary of class weights
        """
        counter = Counter(y)
        n_samples = len(y)
        n_classes = len(counter)

        if strategy == 'balanced':
            weights = {
                c: n_samples / (n_classes * count)
                for c, count in counter.items()
            }
        elif strategy == 'sqrt_inverse':
            weights = {
                c: np.sqrt(n_samples / count)
                for c, count in counter.items()
            }
        elif strategy == 'effective_number':
            # Based on "Class-Balanced Loss Based on Effective Number of Samples"
            beta = 0.9999
            weights = {}
            for c, count in counter.items():
                effective_num = (1 - beta ** count) / (1 - beta)
                weights[c] = 1 / effective_num
            # Normalize
            total = sum(weights.values())
            weights = {c: w * n_classes / total for c, w in weights.items()}
        else:
            weights = {c: 1.0 for c in counter.keys()}

        return weights

    @staticmethod
    def focal_loss_weights(y: np.ndarray, y_pred_proba: np.ndarray,
                           gamma: float = 2.0) -> np.ndarray:
        """
        Compute focal loss weights for hard example mining.

        Args:
            y: True labels
            y_pred_proba: Predicted probabilities
            gamma: Focusing parameter

        Returns:
            Sample weights based on focal loss
        """
        # Get probability of true class
        p_t = np.where(y == 1, y_pred_proba, 1 - y_pred_proba)
        # Focal weight
        focal_weight = (1 - p_t) ** gamma
        return focal_weight


def compare_resampling_methods(X: np.ndarray, y: np.ndarray,
                                methods: Optional[List[str]] = None) -> Dict:
    """
    Compare different resampling methods on the same dataset.

    Args:
        X: Feature matrix
        y: Labels
        methods: List of methods to compare (defaults to all)

    Returns:
        Dictionary with comparison results
    """
    if methods is None:
        methods = ['none', 'smote', 'borderline_smote', 'adasyn',
                   'smote_tomek', 'smote_enn', 'hybrid_adaptive']

    results = {}
    for method in methods:
        try:
            handler = ImbalanceHandler(method=method)
            X_res, y_res = handler.fit_resample(X, y)
            results[method] = handler.get_info(y, y_res)
        except Exception as e:
            results[method] = {'error': str(e)}

    return results


if __name__ == "__main__":
    # Test imbalance handlers
    from data_loader import CancerDataLoader

    print("Testing imbalance handlers...")

    # Load imbalanced data
    loader = CancerDataLoader('wdbc')
    data = loader.load_data(imbalance_ratio=6.99)
    X, y = data['X'], data['y']

    print(f"\nOriginal distribution: {Counter(y)}")

    # Test all methods
    results = compare_resampling_methods(X, y)

    for method, info in results.items():
        if 'error' not in info:
            print(f"\n{method.upper()}:")
            print(f"  Resampled distribution: {info['resampled_distribution']}")
            print(f"  Ratio: {info['original_ratio']:.2f} -> {info['resampled_ratio']:.2f}")
        else:
            print(f"\n{method.upper()}: Error - {info['error']}")
