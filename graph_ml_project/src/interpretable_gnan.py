"""
Graph Neural Additive Networks (GNAN) - Interpretable GNN
Based on: "The Intelligible and Effective Graph Neural Additive Network" (NeurIPS 2024)
Novel implementation providing interpretable graph-level and feature-level explanations
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureShapeFunction(nn.Module):
    """
    Learnable shape function for individual features (additive component)
    This provides interpretability by modeling each feature's contribution independently
    """

    def __init__(self, hidden_dim=32):
        super(FeatureShapeFunction, self).__init__()
        # Neural network to learn non-linear shape function
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """x: (batch_size, 1) - single feature values"""
        return self.net(x)


class GraphNeuralAdditiveNetwork(nn.Module):
    """
    GNAN: Interpretable Graph Neural Network using Additive Model Framework

    Key Innovation (NeurIPS 2024):
    - Decomposes prediction into interpretable additive components
    - Each feature has its own shape function visualizable as a curve
    - Provides both local and global explanations
    """

    def __init__(self, num_node_features, num_classes, hidden_dim=64, num_shape_functions=8):
        super(GraphNeuralAdditiveNetwork, self).__init__()

        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Graph structure encoder (captures graph topology)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Additive components: one shape function per feature
        self.shape_functions = nn.ModuleList([
            FeatureShapeFunction(hidden_dim=32)
            for _ in range(num_node_features)
        ])

        # Graph-level shape function
        self.graph_shape = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Final prediction layer (combines all additive components)
        self.output_layer = nn.Linear(num_node_features + hidden_dim // 2, num_classes)

        # Intercept (baseline prediction)
        self.intercept = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x, edge_index, batch, return_explanations=False):
        """
        Forward pass with optional explanation components

        Returns:
            predictions: class probabilities
            explanations: dict with feature contributions (if return_explanations=True)
        """
        batch_size = batch.max().item() + 1

        # 1. Graph structure encoding
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # Pool to graph level
        h_graph = global_mean_pool(h, batch)  # (batch_size, hidden_dim)

        # 2. Additive feature components
        feature_contributions = []
        for feat_idx in range(self.num_node_features):
            # Extract single feature across all nodes
            feat_values = x[:, feat_idx:feat_idx+1]  # (num_nodes, 1)

            # Apply shape function
            feat_contribution = self.shape_functions[feat_idx](feat_values)

            # Pool to graph level
            feat_contribution_graph = global_mean_pool(feat_contribution, batch)
            feature_contributions.append(feat_contribution_graph)

        # Stack feature contributions
        feature_contributions = torch.cat(feature_contributions, dim=1)  # (batch_size, num_features)

        # 3. Graph structure component
        graph_contribution = self.graph_shape(h_graph)  # (batch_size, hidden_dim//2)

        # 4. Combine all components (additive model)
        combined = torch.cat([feature_contributions, graph_contribution], dim=1)

        # 5. Final prediction
        logits = self.output_layer(combined) + self.intercept
        predictions = F.log_softmax(logits, dim=1)

        if return_explanations:
            explanations = {
                'feature_contributions': feature_contributions.detach(),
                'graph_contribution': graph_contribution.detach(),
                'intercept': self.intercept.detach()
            }
            return predictions, explanations
        else:
            return predictions

    def get_feature_importance(self, x, edge_index, batch):
        """Compute feature importance scores for interpretation"""
        with torch.no_grad():
            _, explanations = self.forward(x, edge_index, batch, return_explanations=True)
            feature_contribs = explanations['feature_contributions']

            # Importance = average absolute contribution
            importance = torch.abs(feature_contribs).mean(dim=0)
            return importance.numpy()

    def plot_shape_functions(self, feature_names=None, save_path='outputs/shape_functions.png', dpi=100):
        """Visualize learned shape functions for each feature"""
        num_features = self.num_node_features
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(num_features)]

        # Create grid layout
        n_cols = min(4, num_features)
        n_rows = (num_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if num_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Generate input range
        x_range = torch.linspace(-3, 3, 100).unsqueeze(1)

        with torch.no_grad():
            for i, (shape_func, ax, name) in enumerate(zip(self.shape_functions, axes, feature_names)):
                # Compute shape function output
                y_values = shape_func(x_range).squeeze().numpy()
                x_values = x_range.squeeze().numpy()

                # Plot
                ax.plot(x_values, y_values, linewidth=2, color='steelblue')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Feature Value', fontsize=10)
                ax.set_ylabel('Contribution', fontsize=10)
                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(num_features, len(axes)):
            axes[i].axis('off')

        plt.suptitle('GNAN: Interpretable Feature Shape Functions', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return save_path


def visualize_feature_importance(importance_scores, feature_names=None,
                                 save_path='outputs/feature_importance.png', dpi=100):
    """Visualize global feature importance from GNAN"""
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(importance_scores))]

    # Sort by importance
    sorted_indices = np.argsort(importance_scores)
    sorted_scores = importance_scores[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    colors = plt.cm.RdYlGn_r(sorted_scores / sorted_scores.max())

    bars = ax.barh(range(len(sorted_scores)), sorted_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('GNAN: Global Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score + 0.01 * sorted_scores.max(), i, f'{score:.3f}',
               va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return save_path


def create_synthetic_dataset(num_graphs=100):
    """Create synthetic graph dataset for demonstration"""
    graphs = []
    labels = []

    for i in range(num_graphs):
        # Create graphs with different properties
        if i % 2 == 0:
            # Class 0: Dense graphs with high clustering
            n_nodes = np.random.randint(15, 30)
            G = nx.erdos_renyi_graph(n_nodes, 0.3, seed=i)
            label = 0
        else:
            # Class 1: Sparse graphs with low clustering
            n_nodes = np.random.randint(15, 30)
            G = nx.erdos_renyi_graph(n_nodes, 0.1, seed=i)
            label = 1

        # Add node features
        features = []
        for node in G.nodes():
            degree = G.degree(node)
            clustering = nx.clustering(G, node)
            betweenness = nx.betweenness_centrality(G)[node]
            closeness = nx.closeness_centrality(G)[node]

            features.append([degree, clustering, betweenness, closeness])

        graphs.append((G, np.array(features), label))
        labels.append(label)

    return graphs


def demonstrate_gnan():
    """Demonstrate GNAN interpretability"""
    print("\n" + "=" * 70)
    print("GRAPH NEURAL ADDITIVE NETWORK (GNAN) - Interpretable GNN")
    print("Based on: NeurIPS 2024 Paper")
    print("=" * 70)

    # Create dataset
    print("\nGenerating synthetic graph dataset...")
    graphs = create_synthetic_dataset(num_graphs=50)

    # Create a sample graph for visualization
    G, features, label = graphs[0]
    print(f"Sample graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Node features: {features.shape}")

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make undirected
    x = torch.tensor(features, dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Initialize GNAN model
    num_features = features.shape[1]
    feature_names = ['Degree', 'Clustering', 'Betweenness', 'Closeness']

    model = GraphNeuralAdditiveNetwork(
        num_node_features=num_features,
        num_classes=2,
        hidden_dim=64
    )

    print(f"\nGNAN Model initialized:")
    print(f"  - Input features: {num_features}")
    print(f"  - Shape functions: {len(model.shape_functions)}")
    print(f"  - Output classes: 2")

    # Get predictions with explanations
    with torch.no_grad():
        predictions, explanations = model(x, edge_index, batch, return_explanations=True)

    print(f"\nPrediction made with interpretable components:")
    print(f"  - Feature contributions shape: {explanations['feature_contributions'].shape}")
    print(f"  - Graph structure contribution shape: {explanations['graph_contribution'].shape}")

    # Visualize shape functions
    save_path_shapes = model.plot_shape_functions(feature_names=feature_names)
    print(f"\n✓ Saved shape functions visualization: {save_path_shapes}")

    # Compute and visualize feature importance
    importance = model.get_feature_importance(x, edge_index, batch)
    save_path_importance = visualize_feature_importance(importance, feature_names=feature_names)
    print(f"✓ Saved feature importance visualization: {save_path_importance}")

    print("\n" + "=" * 70)
    print("GNAN Demonstration Complete!")
    print("Key Features:")
    print("  ✓ Interpretable predictions via additive decomposition")
    print("  ✓ Feature-level explanations through shape functions")
    print("  ✓ Global and local interpretability")
    print("=" * 70)

    return model, importance


if __name__ == "__main__":
    demonstrate_gnan()
