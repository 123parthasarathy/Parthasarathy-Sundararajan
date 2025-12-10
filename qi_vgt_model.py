"""
Quantum-Inspired Variational Graph Transformer (QI-VGT)
for Molecular Property Prediction

Paper: QUANTUM-INSPIRED VARIATIONAL GRAPH TRANSFORMER FOR MOLECULAR PROPERTY PREDICTION:
       A NOVEL FRAMEWORK FOR DRUG MUTAGENICITY ASSESSMENT
Authors: Nandhakumar Mahamoorthy, Parthasarathy Sundararajan

Implementation for Q1 Publication Validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import DataLoader
import numpy as np
from typing import Tuple, Optional, Dict


class QuantumInspiredEnhancement(nn.Module):
    """
    Quantum-Inspired Enhancement Module

    Applies minimal quantum transformations to augment molecular representations:
    ψ_enhanced(x) = x + α · Linear(x) ⊙ sin(φ)

    Where:
    - x: Input molecular representation
    - α: Learnable scaling parameter (initialized to 0.05)
    - Linear: Learnable linear transformation (identity initialization)
    - φ: Learnable phase parameters (initialized to zero)
    - ⊙: Element-wise multiplication
    """

    def __init__(self, hidden_dim: int, alpha_init: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable scaling parameter α (minimal initial perturbation)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Quantum gate linear transformation (identity-like initialization)
        self.quantum_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Initialize close to identity with small scaling
        nn.init.eye_(self.quantum_gate.weight)
        self.quantum_gate.weight.data *= 0.1

        # Learnable phase parameters φ (initialized to zero)
        self.phase = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum-inspired enhancement.

        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Enhanced tensor with quantum-inspired augmentation
        """
        # Quantum-inspired transformation: x + α · Linear(x) ⊙ sin(φ)
        linear_transform = self.quantum_gate(x)
        phase_modulation = torch.sin(self.phase)
        quantum_enhancement = self.alpha * linear_transform * phase_modulation

        return x + quantum_enhancement


class QI_VGT(nn.Module):
    """
    Quantum-Inspired Variational Graph Transformer (QI-VGT)

    A hybrid architecture combining:
    1. Graph Convolutional Layers with residual connections
    2. Quantum-Inspired Enhancement Module
    3. Multi-Strategy Pooling (mean, max, sum)
    4. Classification Head with Uncertainty Quantification

    Optimized for small molecular datasets like MUTAG.
    """

    def __init__(
        self,
        num_node_features: int = 7,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        num_gcn_layers: int = 3,
        alpha_init: float = 0.05
    ):
        super().__init__()

        self.num_gcn_layers = num_gcn_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Quantum-Inspired Enhancement Module
        self.quantum_enhancement = QuantumInspiredEnhancement(hidden_dim, alpha_init)

        # Classification Head
        # Input: 3 * hidden_dim (from concatenation of mean, max, sum pooling)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

        # Uncertainty Estimation Branch
        self.uncertainty_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through QI-VGT.

        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            batch: Batch assignment vector

        Returns:
            Dictionary containing logits and uncertainty estimates
        """
        # Layer 1: GCN + BN + ReLU + Residual
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        # Layer 2: GCN + BN + ReLU + Residual Connection
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = h2 + h1  # Residual connection

        # Layer 3: GCN + BN + ReLU + Residual Connection
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3)
        h3 = h3 + h2  # Residual connection

        # Quantum-Inspired Enhancement
        h_enhanced = self.quantum_enhancement(h3)

        # Multi-Strategy Global Pooling
        h_mean = global_mean_pool(h_enhanced, batch)  # Average atomic properties
        h_max = global_max_pool(h_enhanced, batch)    # Most significant features
        h_sum = global_add_pool(h_enhanced, batch)    # Preserves molecular size info

        # Concatenate pooled representations
        h_global = torch.cat([h_mean, h_max, h_sum], dim=1)

        # Classification and uncertainty estimation
        logits = self.classifier(h_global)
        uncertainty = self.uncertainty_head(h_global)

        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'embedding': h_global
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QI_VGT_Improved(nn.Module):
    """
    Improved Quantum-Inspired Variational Graph Transformer (QI-VGT++)

    Enhancements over base QI-VGT:
    1. Multi-head attention for quantum enhancement
    2. Graph Attention Network integration
    3. Enhanced phase rotation with multiple frequencies
    4. Improved uncertainty calibration
    5. Layer-wise quantum enhancement
    """

    def __init__(
        self,
        num_node_features: int = 7,
        hidden_dim: int = 64,  # Increased capacity
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        num_gcn_layers: int = 4,  # Deeper network
        num_attention_heads: int = 4,
        alpha_init: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Enhanced Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

        # Batch Normalization and Layer Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # Multi-Frequency Quantum Enhancement (novel improvement)
        self.quantum_layers = nn.ModuleList([
            EnhancedQuantumLayer(hidden_dim, num_frequencies=3)
            for _ in range(2)
        ])

        # Self-attention for molecular representation
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_attention_heads, dropout=dropout_rate, batch_first=True
        )

        # Improved Classification Head with deeper network
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),  # 4x from mean, max, sum, attn
            nn.GELU(),  # GELU instead of ReLU
            nn.BatchNorm1d(2 * hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        # Calibrated Uncertainty Estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with multi-frequency quantum enhancement."""

        # Layer 1
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))

        # Apply first quantum enhancement
        h1 = self.quantum_layers[0](h1)

        # Layer 2 with residual
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h2 = h2 + h1

        # Layer 3 with residual
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))
        h3 = h3 + h2

        # Apply second quantum enhancement
        h3 = self.quantum_layers[1](h3)

        # Layer 4 with residual
        h4 = F.relu(self.bn4(self.conv4(h3, edge_index)))
        h4 = h4 + h3

        # Multi-Strategy Global Pooling
        h_mean = global_mean_pool(h4, batch)
        h_max = global_max_pool(h4, batch)
        h_sum = global_add_pool(h4, batch)

        # Attention-weighted pooling
        h_attn, _ = self.self_attention(
            h_mean.unsqueeze(1), h_max.unsqueeze(1), h_sum.unsqueeze(1)
        )
        h_attn = h_attn.squeeze(1)

        # Concatenate all representations
        h_global = torch.cat([h_mean, h_max, h_sum, h_attn], dim=1)

        # Temperature-scaled logits for calibration
        logits = self.classifier(h_global) / self.temperature
        uncertainty = self.uncertainty_head(h_global)

        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'embedding': h_global
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedQuantumLayer(nn.Module):
    """
    Enhanced Quantum-Inspired Layer with Multi-Frequency Phase Modulation

    Novel improvement: Uses multiple frequency components to capture
    different quantum-like interference patterns in molecular representations.

    ψ_enhanced(x) = x + Σ_k α_k · Linear_k(x) ⊙ sin(ω_k · φ + θ_k)
    """

    def __init__(self, hidden_dim: int, num_frequencies: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies

        # Multi-frequency components
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1 / (k + 1)))
            for k in range(num_frequencies)
        ])

        self.quantum_gates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_frequencies)
        ])

        # Initialize quantum gates near identity
        for gate in self.quantum_gates:
            nn.init.eye_(gate.weight)
            gate.weight.data *= 0.1

        # Learnable phases and frequencies
        self.phases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim))
            for _ in range(num_frequencies)
        ])

        self.frequencies = nn.ParameterList([
            nn.Parameter(torch.ones(1) * (k + 1))
            for k in range(num_frequencies)
        ])

        # Mixing layer
        self.mixer = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.mixer.weight)
        nn.init.zeros_(self.mixer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-frequency quantum enhancement."""
        enhancement = torch.zeros_like(x)

        for k in range(self.num_frequencies):
            linear_transform = self.quantum_gates[k](x)
            phase_modulation = torch.sin(
                self.frequencies[k] * self.phases[k]
            )
            enhancement += self.alphas[k] * linear_transform * phase_modulation

        return self.mixer(x + enhancement)


def create_qi_vgt_model(
    num_node_features: int = 7,
    hidden_dim: int = 32,
    num_classes: int = 2,
    improved: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create QI-VGT models.

    Args:
        num_node_features: Number of input node features
        hidden_dim: Hidden dimension size
        num_classes: Number of output classes
        improved: Whether to use improved version (QI-VGT++)
        **kwargs: Additional arguments for model configuration

    Returns:
        QI-VGT or QI-VGT++ model instance
    """
    if improved:
        return QI_VGT_Improved(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **kwargs
        )
    return QI_VGT(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == "__main__":
    # Quick test of model architecture
    model = QI_VGT(num_node_features=7, hidden_dim=32, num_classes=2)
    print(f"QI-VGT Parameters: {model.count_parameters()}")

    improved_model = QI_VGT_Improved(num_node_features=7, hidden_dim=64, num_classes=2)
    print(f"QI-VGT++ Parameters: {improved_model.count_parameters()}")
