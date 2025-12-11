"""
QI-VGT Optimized: Quantum-Inspired Variational Graph Transformer
Optimized version for Q1 publication with proven techniques

Key optimizations:
1. Residual connections
2. Batch normalization
3. Learning rate scheduling
4. Better initialization
5. Gradient clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import math


class QuantumPhaseEnhancement(nn.Module):
    """
    Optimized quantum-inspired phase enhancement.
    ψ_enhanced(x) = x + α * tanh(Linear(x)) ⊙ sin(φ)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.phase = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # Better initialization
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = torch.tanh(self.gate(x))
        phase_mod = torch.sin(self.phase)
        return x + self.alpha * gate_out * phase_mod


class QI_VGT_Optimized(nn.Module):
    """
    Optimized Quantum-Inspired Variational Graph Transformer.

    Architecture:
    - Input projection with LayerNorm
    - GIN layers with quantum enhancement
    - Residual connections
    - Multi-scale pooling (mean + max + sum)
    - Dropout regularization
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.quantum_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            # GIN with MLP
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.gnn_layers.append(GINConv(mlp, train_eps=True))
            self.quantum_layers.append(QuantumPhaseEnhancement(hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Multi-scale pooling
        self.pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x.size(0) == 0:
            batch_size = 1 if batch is None else batch.max().item() + 1
            return torch.zeros(batch_size, 2, device=x.device)

        # Input projection
        x = self.input_proj(x)

        # GNN layers with quantum enhancement
        for i in range(self.num_layers):
            # Message passing
            x_new = self.gnn_layers[i](x, edge_index)

            # Quantum enhancement
            x_new = self.quantum_layers[i](x_new)

            # Batch normalization
            x_new = self.batch_norms[i](x_new)

            # Residual connection
            x = x + self.dropout(x_new)

        # Multi-scale pooling
        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_sum = global_add_pool(x, batch)

        graph_repr = torch.cat([pool_mean, pool_max, pool_sum], dim=-1)
        graph_repr = F.relu(self.pool_proj(graph_repr))

        # Classification
        return self.classifier(graph_repr)


class QI_VGT_Plus(nn.Module):
    """
    QI-VGT+ with attention-based aggregation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.quantum_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnn_layers.append(GINConv(mlp))
            self.quantum_layers.append(QuantumPhaseEnhancement(hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Attention pooling
        self.attn_proj = nn.Linear(hidden_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x.size(0) == 0:
            batch_size = 1 if batch is None else batch.max().item() + 1
            return torch.zeros(batch_size, 2, device=x.device)

        # Input
        x = self.input_proj(x)

        # GNN layers
        for i in range(self.num_layers):
            x_new = self.gnn_layers[i](x, edge_index)
            x_new = self.quantum_layers[i](x_new)
            x = self.norms[i](x + self.dropout(x_new))

        # Attention pooling
        batch_size = batch.max().item() + 1
        attn_scores = self.attn_proj(x).squeeze(-1)

        # Softmax within each graph
        graph_repr = []
        for i in range(batch_size):
            mask = (batch == i)
            scores_i = F.softmax(attn_scores[mask], dim=0)
            repr_i = (x[mask] * scores_i.unsqueeze(-1)).sum(dim=0)
            graph_repr.append(repr_i)

        graph_repr = torch.stack(graph_repr)

        return self.classifier(graph_repr)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing QI-VGT Optimized Models")
    print("=" * 50)

    # Test
    x = torch.randn(10, 7)
    edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,0]])
    batch = torch.zeros(10, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    model1 = QI_VGT_Optimized(input_dim=7)
    model2 = QI_VGT_Plus(input_dim=7)

    print(f"QI-VGT-Optimized: {count_parameters(model1):,} params")
    print(f"QI-VGT+: {count_parameters(model2):,} params")

    out1 = model1(data)
    out2 = model2(data)
    print(f"Output shapes: {out1.shape}, {out2.shape}")
