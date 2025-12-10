"""
Baseline Models for Comparative Evaluation

Includes:
1. Traditional ML: Random Forest, SVM, Naive Bayes
2. Graph Neural Networks: GCN, GAT, GIN, MPNN
3. Deep Learning: DNN with molecular fingerprints

For comparison with QI-VGT on MUTAG dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, NNConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
from typing import Dict, Optional


# ============================================================================
# Graph Neural Network Baselines
# ============================================================================

class GCN_Baseline(nn.Module):
    """Standard Graph Convolutional Network baseline."""

    def __init__(
        self,
        num_node_features: int = 7,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.relu(self.bn3(self.conv3(h, edge_index)))

        h = global_mean_pool(h, batch)
        return {'logits': self.classifier(h)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GAT_Baseline(nn.Module):
    """Graph Attention Network baseline."""

    def __init__(
        self,
        num_node_features: int = 7,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        num_heads: int = 4
    ):
        super().__init__()

        self.conv1 = GATConv(num_node_features, hidden_dim // num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.relu(self.bn3(self.conv3(h, edge_index)))

        h = global_mean_pool(h, batch)
        return {'logits': self.classifier(h)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GIN_Baseline(nn.Module):
    """Graph Isomorphism Network baseline (more expressive than GCN)."""

    def __init__(
        self,
        num_node_features: int = 7,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        # GIN uses MLP for neighborhood aggregation
        self.mlp1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.conv3 = GINConv(self.mlp3)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.relu(self.bn3(self.conv3(h, edge_index)))

        h = global_mean_pool(h, batch)
        return {'logits': self.classifier(h)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MPNN_Baseline(nn.Module):
    """Message Passing Neural Network baseline."""

    def __init__(
        self,
        num_node_features: int = 7,
        num_edge_features: int = 4,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super().__init__()

        # Edge network for NNConv
        self.edge_nn1 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features * hidden_dim)
        )

        self.edge_nn2 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )

        self.conv1 = NNConv(num_node_features, hidden_dim, self.edge_nn1, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nn2, aggr='mean')

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is None:
            # Create default edge features if not provided
            num_edges = edge_index.size(1)
            edge_attr = torch.ones(num_edges, 4, device=x.device)

        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))

        h = global_mean_pool(h, batch)
        return {'logits': self.classifier(h)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepGCN_Baseline(nn.Module):
    """
    Deep Graph Convolutional Network with residual connections.
    Based on architecture achieving ~85% on MUTAG in literature.
    """

    def __init__(
        self,
        num_node_features: int = 7,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 5,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Multi-scale readout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)

        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h_new = self.bns[i](h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # Residual connection

        # Multi-strategy pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)

        h = torch.cat([h_mean, h_max, h_sum], dim=1)
        return {'logits': self.classifier(h)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Traditional ML Baselines (using graph-level features)
# ============================================================================

class TraditionalMLWrapper:
    """Wrapper for traditional ML models using molecular descriptors."""

    def __init__(self, model_type: str = 'rf', **kwargs):
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        elif model_type == 'nb':
            self.model = GaussianNB()
        elif model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (64, 32)),
                activation='relu',
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)


def extract_graph_features(data) -> np.ndarray:
    """
    Extract graph-level features for traditional ML models.

    Features include:
    - Node feature statistics (mean, std, min, max)
    - Graph structure features (num nodes, num edges, density)
    - Degree statistics
    """
    features = []

    # Node feature statistics
    x = data.x.numpy() if hasattr(data.x, 'numpy') else data.x

    # Mean and std of each feature
    features.extend(np.mean(x, axis=0))
    features.extend(np.std(x, axis=0))
    features.extend(np.min(x, axis=0))
    features.extend(np.max(x, axis=0))

    # Graph structure features
    num_nodes = x.shape[0]
    num_edges = data.edge_index.shape[1] // 2  # Undirected edges counted twice

    features.append(num_nodes)
    features.append(num_edges)
    features.append(num_edges / (num_nodes * (num_nodes - 1) / 2 + 1e-10))  # Density

    # Degree statistics
    edge_index = data.edge_index.numpy() if hasattr(data.edge_index, 'numpy') else data.edge_index
    degrees = np.bincount(edge_index[0], minlength=num_nodes)
    features.append(np.mean(degrees))
    features.append(np.std(degrees))
    features.append(np.max(degrees))

    return np.array(features)


def prepare_traditional_ml_data(dataset):
    """
    Prepare dataset for traditional ML models.

    Returns X (feature matrix) and y (labels).
    """
    X = []
    y = []

    for data in dataset:
        features = extract_graph_features(data)
        X.append(features)
        y.append(data.y.item())

    return np.array(X), np.array(y)


# ============================================================================
# Model Factory
# ============================================================================

def create_baseline_model(
    model_name: str,
    num_node_features: int = 7,
    hidden_dim: int = 32,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        model_name: Name of the model ('gcn', 'gat', 'gin', 'mpnn', 'deepgcn')
        num_node_features: Number of input node features
        hidden_dim: Hidden dimension size
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    models = {
        'gcn': GCN_Baseline,
        'gat': GAT_Baseline,
        'gin': GIN_Baseline,
        'mpnn': MPNN_Baseline,
        'deepgcn': DeepGCN_Baseline,
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name.lower()](
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test of baseline models
    print("Testing baseline models...")

    for name in ['gcn', 'gat', 'gin', 'deepgcn']:
        model = create_baseline_model(name)
        print(f"{name.upper()} Parameters: {model.count_parameters()}")
