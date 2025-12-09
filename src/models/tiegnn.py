"""
TIEGNN: Topological, Interpretable, and Equivariant Graph Neural Network

A unified framework combining:
- Topological features via persistent homology (dual-space Vietoris-Rips)
- Interpretable additive architecture with learnable shape functions
- E(n)-equivariant geometric processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Optional, Tuple, List


class GCNConv(MessagePassing):
    """Standard Graph Convolutional Layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class ShapeFunction(nn.Module):
    """Learnable shape function for interpretable feature contributions."""

    def __init__(self, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1)).squeeze(-1)


class EquivariantLayer(MessagePassing):
    """E(n)-equivariant message passing layer."""

    def __init__(self, hidden_dim: int, coord_dim: int = 3):
        super().__init__(aggr='mean')
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim

        # Edge MLP for feature updates (processes invariant features)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index

        # Compute relative positions and distances (invariant)
        rel_pos = pos[row] - pos[col]
        dist = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        # Edge features (invariant)
        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)

        # Message for features
        msg_h = self.edge_mlp(edge_feat)

        # Message for coordinates (weighted by scalar)
        coord_weight = self.coord_mlp(edge_feat)
        msg_pos = rel_pos * coord_weight

        # Aggregate
        h_agg = self.propagate(edge_index, x=msg_h)
        pos_agg = self.propagate(edge_index, x=msg_pos)

        # Update
        h_new = self.node_mlp(torch.cat([h, h_agg], dim=-1))
        pos_new = pos + pos_agg

        return h_new, pos_new


class TopologicalEncoder(nn.Module):
    """Encodes precomputed topological features."""

    def __init__(self, input_dim: int = 24, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, topo_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(topo_features)


class GraphStructureEncoder(nn.Module):
    """GCN-based encoder for graph structure."""

    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        return x


class InterpretableHead(nn.Module):
    """Interpretable prediction head with additive decomposition."""

    def __init__(self, num_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Intercept
        self.mu = nn.Parameter(torch.zeros(num_classes))

        # Shape functions for each feature
        self.shape_functions = nn.ModuleList([
            ShapeFunction(hidden_dim=32) for _ in range(num_features)
        ])

        # Final linear layer for each shape function output
        self.feature_weights = nn.Parameter(torch.randn(num_features, num_classes) * 0.01)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch_size, num_features]
        Returns:
            predictions: [batch_size, num_classes]
            contributions: [batch_size, num_features] - per-feature contributions
        """
        contributions = []
        for i, shape_fn in enumerate(self.shape_functions):
            contrib = shape_fn(features[:, i])  # [batch_size]
            contributions.append(contrib)

        contributions = torch.stack(contributions, dim=1)  # [batch_size, num_features]

        # Weighted sum for each class
        output = self.mu + contributions @ self.feature_weights  # [batch_size, num_classes]

        return output, contributions


class TIEGNN(nn.Module):
    """
    Topological, Interpretable, Equivariant Graph Neural Network.

    Unified framework combining:
    - Topological features via persistent homology
    - Interpretable additive architecture
    - E(n)-equivariant processing (when coordinates available)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_gcn_layers: int = 3,
        num_equiv_layers: int = 2,
        topo_dim: int = 24,
        use_equiv: bool = True,
        use_topo: bool = True,
        use_interpretable: bool = True,
        coord_dim: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_equiv = use_equiv
        self.use_topo = use_topo
        self.use_interpretable = use_interpretable
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Graph structure encoder
        self.graph_encoder = GraphStructureEncoder(in_channels, hidden_dim, num_gcn_layers)

        # Topological encoder
        if use_topo:
            self.topo_encoder = TopologicalEncoder(topo_dim, hidden_dim, hidden_dim)

        # Equivariant layers
        if use_equiv:
            self.node_embed = nn.Linear(in_channels, hidden_dim)
            self.equiv_layers = nn.ModuleList([
                EquivariantLayer(hidden_dim, coord_dim) for _ in range(num_equiv_layers)
            ])

        # Compute fusion dimension
        fusion_dim = hidden_dim  # graph features
        if use_topo:
            fusion_dim += hidden_dim
        if use_equiv:
            fusion_dim += hidden_dim

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Prediction head
        if use_interpretable:
            self.head = InterpretableHead(hidden_dim, 32, num_classes)
        else:
            self.head = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_features: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            topo_features: Precomputed topological features [batch_size, topo_dim]
            pos: Node coordinates [num_nodes, coord_dim]

        Returns:
            predictions: [batch_size, num_classes]
            contributions: [batch_size, hidden_dim] if interpretable, else None
        """
        representations = []

        # Graph branch
        h_graph = self.graph_encoder(x, edge_index)
        h_graph = global_mean_pool(h_graph, batch)
        representations.append(h_graph)

        # Topological branch
        if self.use_topo and topo_features is not None:
            h_topo = self.topo_encoder(topo_features)
            representations.append(h_topo)

        # Equivariant branch
        if self.use_equiv and pos is not None:
            h_equiv = self.node_embed(x)
            pos_current = pos.clone()
            for layer in self.equiv_layers:
                h_equiv, pos_current = layer(h_equiv, pos_current, edge_index)
            h_equiv = global_mean_pool(h_equiv, batch)
            representations.append(h_equiv)

        # Fusion
        z = torch.cat(representations, dim=-1)
        z = self.fusion(z)
        z = self.dropout(z)

        # Prediction
        if self.use_interpretable:
            out, contributions = self.head(z)
            return out, contributions
        else:
            out = self.head(z)
            return out, None

    def get_feature_importance(self) -> torch.Tensor:
        """Get global feature importance scores."""
        if not self.use_interpretable:
            return None
        weights = self.head.feature_weights.abs().mean(dim=1)
        return weights / weights.sum()


class TIEGNNRegression(TIEGNN):
    """TIEGNN for regression tasks."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, contributions = super().forward(*args, **kwargs)
        return out.squeeze(-1), contributions
