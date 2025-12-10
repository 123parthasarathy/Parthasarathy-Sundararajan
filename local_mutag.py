"""
Local MUTAG Dataset Generator

Creates a synthetic MUTAG-like dataset for validation when network
access is unavailable.

The MUTAG dataset contains 188 nitroaromatic compounds with:
- 7 node features (atom types)
- 4 edge features (bond types)
- Binary classification (mutagenic/non-mutagenic)
- Class distribution: ~66.5% mutagenic, ~33.5% non-mutagenic
"""

import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import os
from typing import List, Tuple


class LocalMUTAGDataset(InMemoryDataset):
    """
    Local MUTAG-like dataset for validation.

    Generates realistic molecular graphs based on MUTAG statistics:
    - 188 molecules
    - Average 17.9 atoms per molecule
    - 7 atom types: C, N, O, F, I, Cl, Br
    - Binary mutagenicity labels
    """

    def __init__(
        self,
        root: str = 'data/LocalMUTAG',
        transform=None,
        pre_transform=None,
        seed: int = 42
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['mutag_data.pt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['mutag_processed.pt']

    def download(self):
        # Generate synthetic data instead of downloading
        pass

    def process(self):
        """Generate MUTAG-like molecular graphs."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_samples = 188
        n_mutagenic = 125

        # Generate labels
        labels = [1] * n_mutagenic + [0] * (n_samples - n_mutagenic)
        np.random.shuffle(labels)

        data_list = []

        for i, label in enumerate(labels):
            # Generate molecular graph
            data = self._generate_molecule(label, i)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def _generate_molecule(self, label: int, idx: int) -> Data:
        """Generate a single molecule graph."""
        # Number of atoms (based on MUTAG statistics)
        n_atoms = np.random.randint(10, 28)  # Average ~17.9

        # Node features: 7 features for atom types (one-hot encoded)
        # Atom types: C, N, O, F, I, Cl, Br (indices 0-6)
        atom_probs = [0.65, 0.15, 0.10, 0.03, 0.02, 0.03, 0.02]  # Carbon most common
        atom_types = np.random.choice(7, size=n_atoms, p=atom_probs)

        x = torch.zeros(n_atoms, 7)
        for j, atom_type in enumerate(atom_types):
            x[j, atom_type] = 1.0

        # Add some discriminative features correlated with mutagenicity
        if label == 1:  # Mutagenic compounds tend to have certain structural features
            # Nitro groups more likely
            nitro_mask = np.random.random(n_atoms) < 0.15
            x[nitro_mask, 1] = 1.0  # Nitrogen
            x[nitro_mask, 2] = 1.0  # Oxygen (nitro has both N and O)

        # Generate edges (molecular bonds)
        n_edges = int(n_atoms * 1.1)  # Typical bond ratio
        edges = []

        # Create connected graph (spanning tree first)
        for j in range(1, n_atoms):
            parent = np.random.randint(0, j)
            edges.append([parent, j])
            edges.append([j, parent])

        # Add additional edges for rings
        n_extra = n_edges - (n_atoms - 1)
        for _ in range(max(0, n_extra)):
            u = np.random.randint(0, n_atoms)
            v = np.random.randint(0, n_atoms)
            if u != v:
                edges.append([u, v])
                edges.append([v, u])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Edge features (4 features for bond types)
        n_actual_edges = edge_index.shape[1]
        edge_attr = torch.zeros(n_actual_edges, 4)

        # Bond types: single, double, triple, aromatic
        for j in range(n_actual_edges):
            bond_type = np.random.choice(4, p=[0.6, 0.2, 0.05, 0.15])
            edge_attr[j, bond_type] = 1.0

        # Label
        y = torch.tensor([label], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def create_mutag_dataset(seed: int = 42) -> InMemoryDataset:
    """Create local MUTAG dataset."""
    return LocalMUTAGDataset(seed=seed)


def get_mutag_statistics(dataset) -> dict:
    """Compute statistics of the dataset."""
    n_samples = len(dataset)
    n_mutagenic = sum(1 for data in dataset if data.y.item() == 1)

    num_nodes = [data.x.shape[0] for data in dataset]
    num_edges = [data.edge_index.shape[1] // 2 for data in dataset]

    return {
        'n_samples': n_samples,
        'n_mutagenic': n_mutagenic,
        'n_non_mutagenic': n_samples - n_mutagenic,
        'avg_nodes': np.mean(num_nodes),
        'avg_edges': np.mean(num_edges),
        'num_node_features': dataset[0].x.shape[1],
        'num_edge_features': dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
    }


if __name__ == "__main__":
    print("Creating local MUTAG dataset...")
    dataset = create_mutag_dataset()

    print("\nDataset Statistics:")
    stats = get_mutag_statistics(dataset)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nFirst molecule:")
    data = dataset[0]
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Node features: {data.x.shape[1]}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Label: {data.y.item()}")
