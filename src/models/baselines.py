"""
Baseline models for comparison:
- GCN: Graph Convolutional Network (Kipf & Welling, 2017)
- GAT: Graph Attention Network (Velickovic et al., 2018)
- EGNN: E(n) Equivariant Graph Neural Network (Satorras et al., 2021)
- PersLay: Persistence Landscape Neural Network (Carriere et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree, softmax
from typing import Optional, Tuple
import math


# =============================================================================
# GCN: Graph Convolutional Network
# =============================================================================

class GCNLayer(MessagePassing):
    """Graph Convolutional Layer with symmetric normalization."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class GCN(nn.Module):
    """Graph Convolutional Network baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.classifier(x), None


# =============================================================================
# GAT: Graph Attention Network
# =============================================================================

class GATLayer(MessagePassing):
    """Graph Attention Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.zeros(1, heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        x = self.lin(x).view(-1, H, C)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, alpha=(alpha_src, alpha_dst))

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        return out + self.bias

    def message(self, x_j: torch.Tensor, alpha_j: torch.Tensor,
                alpha_i: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class GAT(nn.Module):
    """Graph Attention Network baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATLayer(in_channels, hidden_dim, heads=heads, concat=True, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATLayer(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))
        self.convs.append(GATLayer(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        self.norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.norms.append(nn.BatchNorm1d(hidden_dim * heads))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.classifier(x), None


# =============================================================================
# EGNN: E(n) Equivariant Graph Neural Network
# =============================================================================

class EGNNLayer(nn.Module):
    """E(n)-Equivariant Graph Neural Network Layer."""

    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Edge MLP
        edge_input_dim = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # Initialize coord MLP to output small values
        nn.init.zeros_(self.coord_mlp[-1].weight)

    def scatter_mean(self, src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        """Scatter mean operation without external dependencies."""
        out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
        count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)

        index_expanded = index.unsqueeze(-1).expand_as(src)
        out.scatter_add_(0, index_expanded, src)
        count.scatter_add_(0, index.unsqueeze(-1), torch.ones_like(index.unsqueeze(-1), dtype=src.dtype))

        count = count.clamp(min=1)
        return out / count

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        num_nodes = h.size(0)

        # Compute relative positions and distances
        rel_pos = pos[row] - pos[col]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        # Edge features
        edge_feat = [h[row], h[col], dist_sq]
        if edge_attr is not None:
            edge_feat.append(edge_attr)
        edge_feat = torch.cat(edge_feat, dim=-1)

        # Edge embedding
        m_ij = self.edge_mlp(edge_feat)

        # Coordinate update
        coord_weights = self.coord_mlp(m_ij)
        coord_diff = rel_pos * coord_weights

        # Aggregate messages using scatter mean
        h_agg = self.scatter_mean(m_ij, col, num_nodes)
        pos_agg = self.scatter_mean(coord_diff, col, num_nodes)

        # Update nodes
        h_new = self.node_mlp(torch.cat([h, h_agg], dim=-1))
        h_new = h + h_new  # Residual connection
        pos_new = pos + pos_agg

        return h_new, pos_new


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network baseline."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_embed = nn.Linear(in_channels, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        if pos is None:
            # Generate random positions if not provided
            pos = torch.randn(x.size(0), 3, device=x.device)

        h = self.node_embed(x)

        for layer, norm in zip(self.layers, self.norms):
            h, pos = layer(h, pos, edge_index)
            h = norm(h)
            h = self.dropout(h)

        h = global_mean_pool(h, batch)
        return self.classifier(h), None


# =============================================================================
# PersLay: Persistence Landscape Neural Network
# =============================================================================

class PersistenceStatisticsLayer(nn.Module):
    """Layer to process persistence diagram statistics."""

    def __init__(self, input_dim: int = 24, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.encoder(stats)


class PersLay(nn.Module):
    """
    PersLay-style baseline using persistence statistics.
    Based on Carriere et al., 2020.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        topo_dim: int = 24,
        num_gcn_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        # Graph encoder
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_gcn_layers)])

        # Persistence encoder
        self.pers_encoder = PersistenceStatisticsLayer(topo_dim, hidden_dim, hidden_dim)

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        topo_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        # Graph encoding
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h_graph = global_mean_pool(x, batch)

        # Persistence encoding
        if topo_features is not None:
            h_topo = self.pers_encoder(topo_features)
            h = torch.cat([h_graph, h_topo], dim=-1)
            h = self.fusion(h)
        else:
            h = h_graph

        return self.classifier(h), None


# =============================================================================
# Regression variants
# =============================================================================

class GCNRegression(GCN):
    """GCN for regression tasks."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, None]:
        out, _ = super().forward(*args, **kwargs)
        return out.squeeze(-1), None


class GATRegression(GAT):
    """GAT for regression tasks."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, None]:
        out, _ = super().forward(*args, **kwargs)
        return out.squeeze(-1), None


class EGNNRegression(EGNN):
    """EGNN for regression tasks."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, None]:
        out, _ = super().forward(*args, **kwargs)
        return out.squeeze(-1), None


class PersLayRegression(PersLay):
    """PersLay for regression tasks."""

    def __init__(self, *args, **kwargs):
        kwargs['num_classes'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, None]:
        out, _ = super().forward(*args, **kwargs)
        return out.squeeze(-1), None
