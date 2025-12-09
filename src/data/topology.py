"""
Topological Feature Extraction using Persistent Homology.

Computes:
- Vertex-space Vietoris-Rips persistence (standard graph filtration)
- Edge-space Vietoris-Rips persistence (line graph filtration)
- Statistical summaries of persistence diagrams
"""

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from typing import List, Tuple, Optional, Dict
import warnings


def compute_distance_matrix(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Compute shortest path distance matrix from edge index."""
    adj = np.zeros((num_nodes, num_nodes))
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Ensure undirected

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = shortest_path(adj, directed=False, unweighted=True)

    # Replace inf with max_dist + 1
    max_dist = dist[dist != np.inf].max() if np.any(dist != np.inf) else 1
    dist[dist == np.inf] = max_dist + 1

    return dist


def build_line_graph(edge_index: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, int]:
    """
    Build line graph from edge index.
    In the line graph, each edge becomes a node, and two nodes are connected
    if their corresponding edges share an endpoint.
    """
    if edge_index.shape[1] == 0:
        return np.array([[], []]), 0

    # Get unique undirected edges
    edges = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        edges.add((min(u, v), max(u, v)))

    edges = list(edges)
    num_line_nodes = len(edges)

    if num_line_nodes == 0:
        return np.array([[], []]), 0

    # Build line graph edges
    line_edges = []
    for i, (u1, v1) in enumerate(edges):
        for j, (u2, v2) in enumerate(edges):
            if i < j:
                # Check if edges share an endpoint
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    line_edges.append((i, j))
                    line_edges.append((j, i))

    if len(line_edges) == 0:
        return np.array([[], []]), num_line_nodes

    line_edge_index = np.array(line_edges).T
    return line_edge_index, num_line_nodes


def compute_persistence_ripser(dist_matrix: np.ndarray, max_dim: int = 1) -> Dict[int, np.ndarray]:
    """
    Compute persistence diagrams using Ripser.
    Falls back to simple computation if Ripser is not available.
    """
    try:
        import ripser
        result = ripser.ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
        dgms = {i: result['dgms'][i] for i in range(max_dim + 1)}
    except ImportError:
        # Fallback: simple persistence approximation
        dgms = _compute_persistence_simple(dist_matrix, max_dim)

    return dgms


def _compute_persistence_simple(dist_matrix: np.ndarray, max_dim: int = 1) -> Dict[int, np.ndarray]:
    """Simple persistence computation when Ripser is not available."""
    n = dist_matrix.shape[0]
    dgms = {}

    # H0: Connected components
    # Use minimum spanning tree approach
    h0_pairs = []
    if n > 0:
        # All components start at birth=0
        # Death times based on edge weights in MST
        flat_dist = dist_matrix[np.triu_indices(n, k=1)]
        sorted_dists = np.sort(flat_dist)

        # Approximate: assume each edge death corresponds to a component merge
        for i, d in enumerate(sorted_dists[:n-1]):
            h0_pairs.append([0, d])
        # One component lives forever
        h0_pairs.append([0, np.inf])

    dgms[0] = np.array(h0_pairs) if h0_pairs else np.array([]).reshape(0, 2)

    # H1: Cycles (simplified approximation)
    h1_pairs = []
    if n > 2 and max_dim >= 1:
        # Look for triangles and cycles
        for i in range(min(n, 20)):  # Limit computation
            for j in range(i + 1, min(n, 20)):
                for k in range(j + 1, min(n, 20)):
                    # Triangle forms at max edge weight, dies at sum
                    edges = [dist_matrix[i, j], dist_matrix[j, k], dist_matrix[i, k]]
                    birth = max(edges)
                    death = birth + min(edges) * 0.5  # Approximation
                    if death > birth:
                        h1_pairs.append([birth, death])

    dgms[1] = np.array(h1_pairs) if h1_pairs else np.array([]).reshape(0, 2)

    return dgms


def extract_persistence_statistics(dgm: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a persistence diagram.

    Returns 6 features:
    - mean lifetime
    - std lifetime
    - max lifetime
    - total persistence (sum of lifetimes)
    - number of features
    - 90th percentile lifetime
    """
    if dgm.shape[0] == 0:
        return np.zeros(6)

    # Filter out infinite deaths for statistics
    finite_mask = dgm[:, 1] != np.inf
    if not np.any(finite_mask):
        # All infinite - use large value
        lifetimes = np.array([10.0])
    else:
        finite_dgm = dgm[finite_mask]
        lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]

    if len(lifetimes) == 0:
        return np.zeros(6)

    stats = np.array([
        np.mean(lifetimes),
        np.std(lifetimes) if len(lifetimes) > 1 else 0,
        np.max(lifetimes),
        np.sum(lifetimes),
        len(lifetimes),
        np.percentile(lifetimes, 90) if len(lifetimes) > 0 else 0
    ])

    return stats


def compute_topological_features(
    edge_index: np.ndarray,
    num_nodes: int,
    compute_edge_persistence: bool = True
) -> np.ndarray:
    """
    Compute topological features for a single graph.

    Returns a 24-dimensional feature vector:
    - 6 features from H0 vertex persistence
    - 6 features from H1 vertex persistence
    - 6 features from H0 edge persistence
    - 6 features from H1 edge persistence
    """
    features = []

    # Vertex-space persistence
    if num_nodes > 1:
        dist_matrix = compute_distance_matrix(edge_index, num_nodes)
        # Normalize distances
        max_dist = dist_matrix[dist_matrix != np.inf].max() if np.any(dist_matrix != np.inf) else 1
        dist_matrix = dist_matrix / (max_dist + 1e-8)

        vertex_dgms = compute_persistence_ripser(dist_matrix, max_dim=1)
        features.append(extract_persistence_statistics(vertex_dgms[0]))
        features.append(extract_persistence_statistics(vertex_dgms[1]))
    else:
        features.append(np.zeros(6))
        features.append(np.zeros(6))

    # Edge-space persistence (line graph)
    if compute_edge_persistence and edge_index.shape[1] > 0:
        line_edge_index, num_line_nodes = build_line_graph(edge_index, num_nodes)

        if num_line_nodes > 1 and line_edge_index.shape[1] > 0:
            line_dist = compute_distance_matrix(line_edge_index, num_line_nodes)
            max_dist = line_dist[line_dist != np.inf].max() if np.any(line_dist != np.inf) else 1
            line_dist = line_dist / (max_dist + 1e-8)

            edge_dgms = compute_persistence_ripser(line_dist, max_dim=1)
            features.append(extract_persistence_statistics(edge_dgms[0]))
            features.append(extract_persistence_statistics(edge_dgms[1]))
        else:
            features.append(np.zeros(6))
            features.append(np.zeros(6))
    else:
        features.append(np.zeros(6))
        features.append(np.zeros(6))

    return np.concatenate(features)


class TopologicalFeatureExtractor:
    """
    Batch extractor for topological features.
    Supports caching and parallel computation.
    """

    def __init__(self, compute_edge_persistence: bool = True, cache: bool = True):
        self.compute_edge_persistence = compute_edge_persistence
        self.cache = cache
        self._cache = {}

    def __call__(self, data) -> torch.Tensor:
        """Extract topological features for a PyG Data object."""
        # Create cache key
        if self.cache:
            key = (data.edge_index.cpu().numpy().tobytes(), data.num_nodes)
            if key in self._cache:
                return self._cache[key]

        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.num_nodes

        features = compute_topological_features(
            edge_index, num_nodes, self.compute_edge_persistence
        )

        features_tensor = torch.tensor(features, dtype=torch.float32)

        if self.cache:
            self._cache[key] = features_tensor

        return features_tensor

    def extract_batch(self, dataset, verbose: bool = True) -> torch.Tensor:
        """Extract features for entire dataset."""
        from tqdm import tqdm

        features_list = []
        iterator = tqdm(dataset, desc="Extracting topological features") if verbose else dataset

        for data in iterator:
            features = self(data)
            features_list.append(features)

        return torch.stack(features_list)

    def clear_cache(self):
        """Clear the feature cache."""
        self._cache = {}
