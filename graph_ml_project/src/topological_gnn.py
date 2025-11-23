"""
Topological Graph Neural Network with Persistence Diagrams
Based on: Line Graph Vietoris-Rips Persistence Diagram (JMLR 2024)
Novel implementation of topological features for graph representation learning
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


class PersistenceDiagramExtractor:
    """Extract topological features using Vietoris-Rips persistence diagrams"""

    def __init__(self, max_dim=2):
        self.max_dim = max_dim

    def compute_node_persistence(self, G):
        """Compute persistence diagram from node features"""
        # Extract node positions or use spectral embedding
        if nx.number_of_nodes(G) < 3:
            return None

        # Compute distance matrix using graph distances
        try:
            dist_matrix = np.array(nx.floyd_warshall_numpy(G))
            # Replace inf with max finite value
            max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
            dist_matrix[np.isinf(dist_matrix)] = max_finite * 2
        except:
            return None

        # Compute persistence using Ripser
        result = ripser(dist_matrix, distance_matrix=True, maxdim=self.max_dim)
        return result['dgms']

    def compute_edge_persistence(self, G):
        """Compute persistence diagram from line graph (edge-based)"""
        # Create line graph (nodes represent edges of original graph)
        try:
            L = nx.line_graph(G)
            if nx.number_of_nodes(L) < 3:
                return None
            return self.compute_node_persistence(L)
        except:
            return None

    def extract_persistence_statistics(self, dgms):
        """Extract statistical features from persistence diagrams"""
        if dgms is None:
            return np.zeros(12)

        features = []
        for dim_dgm in dgms[:min(len(dgms), 2)]:  # Use H0 and H1
            if len(dim_dgm) > 0:
                births = dim_dgm[:, 0]
                deaths = dim_dgm[:, 1]
                lifetimes = deaths - births

                # Remove infinite persistence points
                finite_mask = np.isfinite(lifetimes)
                lifetimes = lifetimes[finite_mask]

                if len(lifetimes) > 0:
                    # Statistical features
                    features.extend([
                        np.mean(lifetimes),
                        np.std(lifetimes),
                        np.max(lifetimes),
                        np.sum(lifetimes),
                        len(lifetimes),
                        np.percentile(lifetimes, 90)
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0, 0])

        return np.array(features[:12])


class TopologicalGNN(nn.Module):
    """
    Novel Graph Neural Network with Topological Features
    Combines message passing with persistence-based topological features
    """

    def __init__(self, num_node_features, num_classes, hidden_dim=64):
        super(TopologicalGNN, self).__init__()
        self.hidden_dim = hidden_dim

        # Graph convolutional layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Topological feature processor
        self.topo_fc1 = nn.Linear(12, hidden_dim)
        self.topo_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Fusion layer
        self.fusion_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, topo_features, batch):
        # Graph convolution branch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global pooling
        x_graph = global_mean_pool(x, batch)

        # Topological feature branch
        x_topo = self.topo_fc1(topo_features)
        x_topo = F.relu(x_topo)
        x_topo = self.dropout(x_topo)
        x_topo = self.topo_fc2(x_topo)
        x_topo = F.relu(x_topo)

        # Fusion
        x_combined = torch.cat([x_graph, x_topo], dim=1)
        x_combined = self.fusion_fc1(x_combined)
        x_combined = F.relu(x_combined)
        x_combined = self.dropout(x_combined)
        out = self.fusion_fc2(x_combined)

        return F.log_softmax(out, dim=1)


def visualize_persistence_diagram(dgms, save_path='outputs/persistence_diagram.png', dpi=100):
    """Visualize persistence diagrams for H0 and H1"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['blue', 'red']
    labels = ['H0 (Components)', 'H1 (Loops)']

    for idx, (ax, dgm, color, label) in enumerate(zip(axes, dgms[:2], colors, labels)):
        if len(dgm) > 0:
            births = dgm[:, 0]
            deaths = dgm[:, 1]

            # Remove infinite points for visualization
            finite_mask = np.isfinite(deaths)
            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]

            # Plot diagonal
            max_val = max(np.max(deaths_finite) if len(deaths_finite) > 0 else 1,
                         np.max(births_finite) if len(births_finite) > 0 else 1)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')

            # Plot points
            ax.scatter(births_finite, deaths_finite, c=color, alpha=0.6, s=50, label=label)

            # Highlight top 5 persistent features
            if len(births_finite) > 0:
                lifetimes = deaths_finite - births_finite
                top_indices = np.argsort(lifetimes)[-5:]
                ax.scatter(births_finite[top_indices], deaths_finite[top_indices],
                          c='gold', s=100, marker='*', edgecolors='black', linewidth=1,
                          label='Top 5 Persistent', zorder=5)

        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(f'{label} Persistence Diagram', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def visualize_barcode(dgms, save_path='outputs/persistence_barcode.png', dpi=100):
    """Visualize persistence as barcode"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'red', 'green']
    labels = ['H0', 'H1', 'H2']
    y_pos = 0

    for dim_idx, (dgm, color, label) in enumerate(zip(dgms, colors[:len(dgms)], labels[:len(dgms)])):
        if len(dgm) > 0:
            births = dgm[:, 0]
            deaths = dgm[:, 1]

            # Remove infinite points
            finite_mask = np.isfinite(deaths)
            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]

            # Sort by birth time
            sorted_indices = np.argsort(births_finite)
            births_finite = births_finite[sorted_indices]
            deaths_finite = deaths_finite[sorted_indices]

            # Plot bars (limit to top 20 for visibility)
            for i, (b, d) in enumerate(zip(births_finite[:20], deaths_finite[:20])):
                ax.add_patch(Rectangle((b, y_pos + i * 0.4), d - b, 0.3,
                                      facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5))

            y_pos += len(births_finite[:20]) * 0.4 + 2

    ax.set_xlabel('Filtration Value', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Persistence Barcode', fontsize=14, fontweight='bold')
    ax.set_ylim(-1, y_pos)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def create_sample_graphs():
    """Create diverse sample graphs for demonstration"""
    graphs = []

    # 1. Community structure graph
    G1 = nx.karate_club_graph()
    graphs.append(('Karate Club', G1))

    # 2. Random geometric graph
    G2 = nx.random_geometric_graph(50, 0.2, seed=42)
    graphs.append(('Random Geometric', G2))

    # 3. Barabasi-Albert (scale-free)
    G3 = nx.barabasi_albert_graph(60, 3, seed=42)
    graphs.append(('Scale-Free', G3))

    # 4. Watts-Strogatz (small-world)
    G4 = nx.watts_strogatz_graph(50, 6, 0.3, seed=42)
    graphs.append(('Small-World', G4))

    # 5. Grid graph
    G5 = nx.grid_2d_graph(8, 8)
    graphs.append(('2D Grid', G5))

    return graphs


if __name__ == "__main__":
    print("=" * 70)
    print("TOPOLOGICAL GRAPH NEURAL NETWORK")
    print("Based on: Line Graph Vietoris-Rips Persistence Diagram (JMLR 2024)")
    print("=" * 70)

    # Create sample graphs
    graphs = create_sample_graphs()
    extractor = PersistenceDiagramExtractor(max_dim=2)

    for name, G in graphs[:3]:  # Process first 3 for demonstration
        print(f"\nProcessing: {name}")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        # Compute persistence diagrams
        dgms = extractor.compute_node_persistence(G)

        if dgms is not None:
            # Extract features
            features = extractor.extract_persistence_statistics(dgms)
            print(f"  Topological features extracted: {len(features)} dimensions")

            # Visualize
            save_path_pd = f'outputs/persistence_diagram_{name.replace(" ", "_")}.png'
            save_path_bc = f'outputs/persistence_barcode_{name.replace(" ", "_")}.png'

            visualize_persistence_diagram(dgms, save_path_pd)
            visualize_barcode(dgms, save_path_bc)
            print(f"  ✓ Saved persistence visualizations")

    print("\n" + "=" * 70)
    print("Topological GNN module completed successfully!")
    print("=" * 70)
