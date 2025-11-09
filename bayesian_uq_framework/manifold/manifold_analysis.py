"""
Manifold Learning for Uncertainty-Aware Reliability Assessment

Techniques for analyzing predictions and uncertainties in low-dimensional manifolds:
- t-SNE, UMAP for visualization
- Intrinsic dimensionality estimation
- Manifold-based reliability regions
- Geometric uncertainty analysis
"""

import numpy as np
import torch
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Optional, List
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


class ManifoldAnalyzer:
    """
    Analyze neural network predictions and uncertainties using manifold learning.
    """

    def __init__(self, method: str = 'umap', random_state: int = 42):
        """
        Initialize manifold analyzer.

        Args:
            method: 'umap', 'tsne', 'isomap', 'lle', or 'pca'
            random_state: Random seed
        """
        self.method = method
        self.random_state = random_state
        self.reducer = None

    def fit_transform(self, features: np.ndarray,
                     n_components: int = 2,
                     **kwargs) -> np.ndarray:
        """
        Fit and transform features to manifold space.

        Args:
            features: Input features (n_samples, n_features)
            n_components: Dimensionality of manifold
            **kwargs: Additional arguments for the method

        Returns:
            Embedded features (n_samples, n_components)
        """
        if self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Using t-SNE instead.")
                self.method = 'tsne'
            else:
                self.reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=self.random_state,
                    **kwargs
                )

        elif self.method == 'tsne':
            self.reducer = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                **kwargs
            )

        elif self.method == 'isomap':
            self.reducer = Isomap(
                n_components=n_components,
                **kwargs
            )

        elif self.method == 'lle':
            self.reducer = LocallyLinearEmbedding(
                n_components=n_components,
                random_state=self.random_state,
                **kwargs
            )

        elif self.method == 'pca':
            self.reducer = PCA(
                n_components=n_components,
                random_state=self.random_state,
                **kwargs
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        embedded = self.reducer.fit_transform(features)
        return embedded

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted reducer."""
        if self.reducer is None:
            raise ValueError("Must fit before transform")

        if hasattr(self.reducer, 'transform'):
            return self.reducer.transform(features)
        else:
            raise ValueError(f"{self.method} does not support transform")


class IntrinsicDimensionEstimator:
    """
    Estimate intrinsic dimensionality of data manifold.
    """

    @staticmethod
    def mle_estimation(data: np.ndarray, k: int = 10) -> float:
        """
        Maximum Likelihood Estimation of intrinsic dimensionality.

        Based on: Levina & Bickel (2005)

        Args:
            data: Input data (n_samples, n_features)
            k: Number of nearest neighbors

        Returns:
            Estimated intrinsic dimensionality
        """
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
        distances, _ = nbrs.kneighbors(data)

        # Remove first column (self-distance = 0)
        distances = distances[:, 1:]

        # MLE estimate for each point
        d_estimates = []
        for i in range(len(data)):
            r = distances[i]
            T_k = r[-1]  # k-th nearest neighbor distance
            d_i = -1 / (np.mean(np.log(r / T_k)))
            d_estimates.append(d_i)

        return np.mean(d_estimates)

    @staticmethod
    def correlation_dimension(data: np.ndarray,
                             r_min: float = 0.01,
                             r_max: float = 1.0,
                             n_steps: int = 20) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Estimate correlation dimension.

        Args:
            data: Input data
            r_min: Minimum radius
            r_max: Maximum radius
            n_steps: Number of radius steps

        Returns:
            dimension, radii, correlations
        """
        from scipy.spatial.distance import pdist, squareform

        # Compute pairwise distances
        distances = squareform(pdist(data))

        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_steps)
        correlations = []

        n_samples = len(data)

        for r in radii:
            # Count pairs within radius r
            count = np.sum(distances < r) - n_samples  # Exclude self-pairs
            C_r = count / (n_samples * (n_samples - 1))
            correlations.append(C_r)

        correlations = np.array(correlations)

        # Fit line in log-log space to estimate dimension
        log_r = np.log(radii)
        log_C = np.log(correlations + 1e-10)

        # Linear fit in middle region (avoid boundary effects)
        mid_start = len(log_r) // 4
        mid_end = 3 * len(log_r) // 4

        slope, _ = np.polyfit(
            log_r[mid_start:mid_end],
            log_C[mid_start:mid_end],
            1
        )

        return slope, radii, correlations


class UncertaintyManifold:
    """
    Analyze uncertainty structure in manifold space.
    """

    def __init__(self, manifold_analyzer: ManifoldAnalyzer):
        self.analyzer = manifold_analyzer

    def create_uncertainty_map(
        self,
        features: np.ndarray,
        uncertainties: np.ndarray,
        n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create manifold embedding colored by uncertainty.

        Args:
            features: Input features
            uncertainties: Uncertainty values
            n_components: Dimensionality of embedding

        Returns:
            embedded_features, uncertainties
        """
        embedded = self.analyzer.fit_transform(features, n_components)
        return embedded, uncertainties

    def identify_high_uncertainty_regions(
        self,
        embedded_features: np.ndarray,
        uncertainties: np.ndarray,
        threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Identify regions in manifold space with high uncertainty.

        Args:
            embedded_features: Manifold coordinates
            uncertainties: Uncertainty values
            threshold_percentile: Percentile threshold

        Returns:
            Boolean mask for high uncertainty regions
        """
        threshold = np.percentile(uncertainties, threshold_percentile)
        return uncertainties >= threshold

    def compute_local_uncertainty_density(
        self,
        embedded_features: np.ndarray,
        uncertainties: np.ndarray,
        bandwidth: float = 0.1
    ) -> np.ndarray:
        """
        Compute local uncertainty density using kernel density estimation.

        Args:
            embedded_features: Manifold coordinates
            uncertainties: Uncertainty values
            bandwidth: KDE bandwidth

        Returns:
            Local uncertainty density
        """
        from sklearn.neighbors import KernelDensity

        # Weight by uncertainty
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(embedded_features, sample_weight=uncertainties)

        density = np.exp(kde.score_samples(embedded_features))
        return density


class GeometricReliabilityAnalysis:
    """
    Geometric analysis of reliability in manifold space.
    """

    @staticmethod
    def compute_local_curvature(
        embedded_features: np.ndarray,
        k_neighbors: int = 10
    ) -> np.ndarray:
        """
        Estimate local curvature at each point in manifold.

        Args:
            embedded_features: Manifold coordinates
            k_neighbors: Number of neighbors for local fitting

        Returns:
            Curvature estimates
        """
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(embedded_features)
        distances, indices = nbrs.kneighbors(embedded_features)

        curvatures = []

        for i in range(len(embedded_features)):
            # Get local neighborhood
            neighbors = embedded_features[indices[i, 1:]]

            # Fit local quadratic
            center = embedded_features[i]
            centered_neighbors = neighbors - center

            # Estimate curvature from variance of normal deviations
            if embedded_features.shape[1] >= 2:
                # PCA to find local tangent plane
                pca = PCA(n_components=min(2, embedded_features.shape[1]))
                pca.fit(centered_neighbors)

                # Curvature approximated by explained variance ratio
                curvature = 1 - pca.explained_variance_ratio_[0]
            else:
                curvature = 0

            curvatures.append(curvature)

        return np.array(curvatures)

    @staticmethod
    def identify_boundary_regions(
        embedded_features: np.ndarray,
        threshold_percentile: float = 10
    ) -> np.ndarray:
        """
        Identify points near manifold boundaries (low density regions).

        Args:
            embedded_features: Manifold coordinates
            threshold_percentile: Percentile threshold

        Returns:
            Boolean mask for boundary regions
        """
        from sklearn.neighbors import KernelDensity

        kde = KernelDensity(bandwidth=0.1)
        kde.fit(embedded_features)

        log_density = kde.score_samples(embedded_features)
        threshold = np.percentile(log_density, threshold_percentile)

        return log_density <= threshold

    @staticmethod
    def compute_manifold_distance(
        point1: np.ndarray,
        point2: np.ndarray,
        embedded_features: np.ndarray,
        k_neighbors: int = 10
    ) -> float:
        """
        Compute geodesic distance along manifold.

        Args:
            point1: First point
            point2: Second point
            embedded_features: All manifold coordinates
            k_neighbors: Neighbors for graph construction

        Returns:
            Geodesic distance
        """
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(embedded_features)
        distances, indices = nbrs.kneighbors(embedded_features)

        # Create adjacency matrix
        n_points = len(embedded_features)
        rows = np.repeat(np.arange(n_points), k_neighbors)
        cols = indices.flatten()
        data = distances.flatten()

        graph = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

        # Find nearest points to point1 and point2
        idx1 = np.argmin(np.linalg.norm(embedded_features - point1, axis=1))
        idx2 = np.argmin(np.linalg.norm(embedded_features - point2, axis=1))

        # Compute shortest path
        dist_matrix = shortest_path(graph, indices=idx1, directed=False)
        return dist_matrix[idx2]


class ManifoldBasedReliabilityRegions:
    """
    Define and analyze reliability regions in manifold space.
    """

    def __init__(self, embedded_features: np.ndarray,
                 uncertainties: np.ndarray,
                 predictions: np.ndarray,
                 true_labels: Optional[np.ndarray] = None):
        """
        Initialize reliability region analyzer.

        Args:
            embedded_features: Manifold coordinates
            uncertainties: Uncertainty estimates
            predictions: Model predictions
            true_labels: Ground truth labels (if available)
        """
        self.embedded_features = embedded_features
        self.uncertainties = uncertainties
        self.predictions = predictions
        self.true_labels = true_labels

        if true_labels is not None:
            self.errors = np.abs(predictions - true_labels)
        else:
            self.errors = None

    def define_reliability_regions(
        self,
        n_regions: int = 3,
        method: str = 'uncertainty_quantiles'
    ) -> np.ndarray:
        """
        Define reliability regions based on uncertainty or error.

        Args:
            n_regions: Number of regions
            method: 'uncertainty_quantiles', 'error_quantiles', or 'clustering'

        Returns:
            Region labels for each point
        """
        if method == 'uncertainty_quantiles':
            quantiles = np.linspace(0, 100, n_regions + 1)
            thresholds = np.percentile(self.uncertainties, quantiles)

            labels = np.digitize(self.uncertainties, thresholds[1:-1])

        elif method == 'error_quantiles':
            if self.errors is None:
                raise ValueError("True labels required for error-based regions")

            quantiles = np.linspace(0, 100, n_regions + 1)
            thresholds = np.percentile(self.errors, quantiles)

            labels = np.digitize(self.errors, thresholds[1:-1])

        elif method == 'clustering':
            from sklearn.cluster import KMeans

            # Cluster based on manifold position and uncertainty
            features = np.column_stack([
                self.embedded_features,
                self.uncertainties.reshape(-1, 1)
            ])

            kmeans = KMeans(n_clusters=n_regions, random_state=42)
            labels = kmeans.fit_predict(features)

        else:
            raise ValueError(f"Unknown method: {method}")

        return labels

    def analyze_region_characteristics(
        self,
        region_labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Analyze characteristics of each reliability region.

        Args:
            region_labels: Region assignments

        Returns:
            Dictionary with statistics for each region
        """
        characteristics = {}
        n_regions = len(np.unique(region_labels))

        for region_id in range(n_regions):
            mask = region_labels == region_id

            stats = {
                'n_samples': mask.sum(),
                'mean_uncertainty': self.uncertainties[mask].mean(),
                'std_uncertainty': self.uncertainties[mask].std(),
                'mean_prediction': self.predictions[mask].mean(),
                'std_prediction': self.predictions[mask].std(),
            }

            if self.errors is not None:
                stats.update({
                    'mean_error': self.errors[mask].mean(),
                    'std_error': self.errors[mask].std(),
                    'rmse': np.sqrt(np.mean(self.errors[mask]**2))
                })

            characteristics[region_id] = stats

        return characteristics

    def compute_region_overlap_matrix(
        self,
        region_labels: np.ndarray,
        k_neighbors: int = 10
    ) -> np.ndarray:
        """
        Compute overlap between reliability regions in manifold space.

        Args:
            region_labels: Region assignments
            k_neighbors: Number of neighbors

        Returns:
            Overlap matrix
        """
        from sklearn.neighbors import NearestNeighbors

        n_regions = len(np.unique(region_labels))
        overlap_matrix = np.zeros((n_regions, n_regions))

        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(self.embedded_features)
        _, indices = nbrs.kneighbors(self.embedded_features)

        for i in range(len(region_labels)):
            my_region = region_labels[i]
            neighbor_regions = region_labels[indices[i]]

            for neighbor_region in neighbor_regions:
                overlap_matrix[my_region, neighbor_region] += 1

        # Normalize
        for i in range(n_regions):
            if overlap_matrix[i].sum() > 0:
                overlap_matrix[i] /= overlap_matrix[i].sum()

        return overlap_matrix
