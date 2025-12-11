"""
QI-VGT-KAN: Quantum-Inspired Variational Graph Transformer
         with Kolmogorov-Arnold Network Enhancement

A novel hybrid architecture combining:
1. Quantum-inspired phase rotations (QI-VGT, 2024)
2. Kolmogorov-Arnold Network learnable activations (KAN, 2025)
3. Virtual Node global attention (AmesFormer, 2024)
4. Multi-scale graph representation

For Q1 Publication - Molecular Property Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import math
from typing import Optional, Tuple, List


class FourierKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer using Fourier series activation.
    Based on KA-GNN (Nature Machine Intelligence, 2025)

    φ(x) = Σ[A_k * cos(kx) + B_k * sin(kx)]
    """
    def __init__(self, in_features: int, out_features: int, num_harmonics: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_harmonics = num_harmonics

        # Learnable Fourier coefficients
        self.A_coeffs = nn.Parameter(torch.randn(num_harmonics, in_features, out_features) * 0.1)
        self.B_coeffs = nn.Parameter(torch.randn(num_harmonics, in_features, out_features) * 0.1)

        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable frequency scaling
        self.freq_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier-based KAN transformation.
        x: [batch, in_features]
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for k in range(1, self.num_harmonics + 1):
            # Compute cos and sin terms
            freq = k * self.freq_scale * x  # [batch, in_features]
            cos_term = torch.cos(freq)  # [batch, in_features]
            sin_term = torch.sin(freq)  # [batch, in_features]

            # Apply learnable coefficients
            # A_coeffs[k-1]: [in_features, out_features]
            output += torch.matmul(cos_term, self.A_coeffs[k-1])
            output += torch.matmul(sin_term, self.B_coeffs[k-1])

        return output + self.bias


class QuantumPhaseRotation(nn.Module):
    """
    Quantum-inspired phase rotation enhancement.
    ψ_enhanced(x) = x + α * Linear(x) ⊙ e^(iφ)

    Real implementation: x + α * Linear(x) * [cos(φ) + sin(φ)]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable quantum parameters
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.phase = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        # Quantum gate simulation
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum-inspired phase rotation."""
        gate_output = self.gate(x)
        phase_mod = torch.cos(self.phase) + torch.sin(self.phase)
        quantum_enhancement = self.alpha * gate_output * phase_mod
        return x + quantum_enhancement


class VirtualNodeAttention(nn.Module):
    """
    Virtual node for global graph-level attention.
    Based on AmesFormer (2024) and Graphormer.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Virtual node embedding
        self.vnode = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply virtual node attention.
        Returns: (updated_node_features, vnode_representation)
        """
        batch_size = batch.max().item() + 1

        # Expand virtual node for each graph in batch
        vnode_expanded = self.vnode.expand(batch_size, -1)  # [batch_size, hidden_dim]

        # Compute attention between vnode and all nodes
        Q = self.q_proj(vnode_expanded)  # [batch_size, hidden_dim]
        K = self.k_proj(x)  # [num_nodes, hidden_dim]
        V = self.v_proj(x)  # [num_nodes, hidden_dim]

        # Compute attention scores per graph
        vnode_out = []
        node_updates = torch.zeros_like(x)

        for i in range(batch_size):
            mask = (batch == i)
            nodes_i = x[mask]  # [num_nodes_i, hidden_dim]
            K_i = K[mask]
            V_i = V[mask]
            Q_i = Q[i:i+1]  # [1, hidden_dim]

            # Attention: Q_i @ K_i^T
            attn_scores = torch.matmul(Q_i, K_i.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Aggregate: weighted sum of values
            vnode_i = torch.matmul(attn_weights, V_i)  # [1, hidden_dim]
            vnode_out.append(vnode_i)

            # Update node features with vnode info
            node_updates[mask] = nodes_i + 0.1 * vnode_i.expand(nodes_i.size(0), -1)

        vnode_repr = torch.cat(vnode_out, dim=0)  # [batch_size, hidden_dim]
        updated_x = self.norm(node_updates)

        return updated_x, vnode_repr


class CentralityEncoding(nn.Module):
    """
    Centrality-based positional encoding for nodes.
    Encodes node importance based on degree.
    """
    def __init__(self, hidden_dim: int, max_degree: int = 50):
        super().__init__()
        self.degree_embedding = nn.Embedding(max_degree + 1, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Add centrality encoding to node features."""
        # Compute node degrees
        row = edge_index[0]
        degrees = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        degrees.scatter_add_(0, row, torch.ones_like(row))
        degrees = degrees.clamp(max=49)

        # Add degree embedding
        centrality = self.degree_embedding(degrees)
        return x + centrality


class QI_VGT_KAN(nn.Module):
    """
    QI-VGT-KAN: Novel hybrid architecture for molecular property prediction.

    Combines:
    - Kolmogorov-Arnold Network (KAN) learnable activations
    - Quantum-inspired phase rotations
    - Virtual node global attention
    - Multi-scale graph pooling

    Architecture:
    1. Input projection with KAN
    2. Centrality encoding
    3. GNN layers with quantum enhancement
    4. Virtual node attention
    5. Multi-scale pooling
    6. KAN classifier
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        num_harmonics: int = 8,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection with KAN
        self.input_kan = FourierKANLayer(input_dim, hidden_dim, num_harmonics)

        # Centrality encoding
        self.centrality = CentralityEncoding(hidden_dim)

        # GNN layers with quantum enhancement
        self.gnn_layers = nn.ModuleList()
        self.quantum_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # GIN-style message passing
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnn_layers.append(GINConv(mlp))
            self.quantum_layers.append(QuantumPhaseRotation(hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Virtual node attention
        self.vnode_attention = VirtualNodeAttention(hidden_dim, num_heads)

        # Multi-scale pooling projection
        self.pool_proj = nn.Linear(hidden_dim * 4, hidden_dim)  # 3 pools + vnode

        # Output KAN classifier
        self.classifier = nn.Sequential(
            FourierKANLayer(hidden_dim, hidden_dim // 2, num_harmonics),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Handle empty graphs
        if x.size(0) == 0:
            batch_size = 1 if batch is None else batch.max().item() + 1
            return torch.zeros(batch_size, 2, device=x.device)

        # 1. Input projection with KAN
        x = self.input_kan(x)

        # 2. Add centrality encoding
        x = self.centrality(x, edge_index)

        # 3. GNN layers with quantum enhancement
        for i in range(self.num_layers):
            # Message passing
            x_new = self.gnn_layers[i](x, edge_index)

            # Quantum-inspired enhancement
            x_new = self.quantum_layers[i](x_new)

            # Residual connection + normalization
            x = self.norms[i](x + x_new)
            x = self.dropout(x)

        # 4. Virtual node attention
        x, vnode_repr = self.vnode_attention(x, batch)

        # 5. Multi-scale pooling
        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_sum = global_add_pool(x, batch)

        # Combine all representations
        graph_repr = torch.cat([pool_mean, pool_max, pool_sum, vnode_repr], dim=-1)
        graph_repr = self.pool_proj(graph_repr)
        graph_repr = F.relu(graph_repr)

        # 6. KAN classifier
        out = self.classifier(graph_repr)

        return out

    def get_uncertainty(self, data: Data, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout for uncertainty quantification.
        """
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = F.softmax(self.forward(data), dim=-1)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0).mean(dim=-1)

        self.eval()
        return mean_pred, uncertainty


class QI_VGT_KAN_Light(nn.Module):
    """
    Lightweight version of QI-VGT-KAN for faster training.
    Uses simplified KAN approximation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Simplified input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # GNN layers with quantum enhancement
        self.gnn_layers = nn.ModuleList()
        self.quantum_phases = nn.ParameterList()
        self.quantum_alphas = nn.ParameterList()

        for _ in range(num_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.quantum_phases.append(nn.Parameter(torch.randn(hidden_dim) * 0.1))
            self.quantum_alphas.append(nn.Parameter(torch.tensor(0.1)))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
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

        # Input projection
        x = self.input_proj(x)

        # GNN layers with quantum enhancement
        for i, gnn in enumerate(self.gnn_layers):
            x_new = gnn(x, edge_index)

            # Quantum-inspired phase rotation
            phase_mod = torch.sin(self.quantum_phases[i])
            x_new = x_new + self.quantum_alphas[i] * x_new * phase_mod

            x = self.norms[i](x + x_new)
            x = self.dropout(x)

        # Multi-scale pooling
        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        pool_sum = global_add_pool(x, batch)

        graph_repr = torch.cat([pool_mean, pool_max, pool_sum], dim=-1)

        return self.classifier(graph_repr)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("QI-VGT-KAN: Novel Hybrid Architecture")
    print("=" * 60)

    # Create dummy data
    x = torch.randn(10, 7)  # 10 nodes, 7 features
    edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9],
                                [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8]])
    batch = torch.zeros(10, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    # Test QI-VGT-KAN
    model = QI_VGT_KAN(input_dim=7, hidden_dim=64, output_dim=2)
    out = model(data)
    print(f"\nQI-VGT-KAN:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Output shape: {out.shape}")

    # Test lightweight version
    model_light = QI_VGT_KAN_Light(input_dim=7, hidden_dim=32, output_dim=2)
    out_light = model_light(data)
    print(f"\nQI-VGT-KAN-Light:")
    print(f"  Parameters: {count_parameters(model_light):,}")
    print(f"  Output shape: {out_light.shape}")

    # Test uncertainty
    mean_pred, uncertainty = model.get_uncertainty(data)
    print(f"\nUncertainty Quantification:")
    print(f"  Mean prediction: {mean_pred}")
    print(f"  Uncertainty: {uncertainty}")

    print("\n" + "=" * 60)
    print("Model ready for validation!")
    print("=" * 60)
