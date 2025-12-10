"""
Real TUDataset Loader

Loads actual benchmark datasets from TUDataset format files:
- MUTAG: 188 nitroaromatic compounds (Debnath et al., 1991)
- PTC_MR: 344 compounds for carcinogenicity
- NCI1: 4110 compounds for anti-cancer screening

This uses the REAL publicly available benchmark data.
"""

import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import os
from typing import List, Dict, Tuple
import shutil


def parse_tu_dataset(data_dir: str, name: str) -> List[Data]:
    """
    Parse TUDataset format files into PyTorch Geometric Data objects.

    TUDataset format:
    - {name}_A.txt: Edge list (comma-separated pairs)
    - {name}_graph_indicator.txt: Node to graph assignment
    - {name}_graph_labels.txt: Graph labels
    - {name}_node_labels.txt: Node labels (optional)
    - {name}_edge_labels.txt: Edge labels (optional)
    """
    prefix = os.path.join(data_dir, name)

    # Read edge list
    edges = []
    with open(f"{prefix}_A.txt", 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                u, v = int(parts[0].strip()), int(parts[1].strip())
                edges.append((u, v))

    # Read graph indicator (which graph each node belongs to)
    graph_indicator = []
    with open(f"{prefix}_graph_indicator.txt", 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip()))

    # Read graph labels
    graph_labels = []
    with open(f"{prefix}_graph_labels.txt", 'r') as f:
        for line in f:
            label = int(line.strip())
            # Convert -1 to 0 for binary classification
            graph_labels.append(0 if label == -1 else label)

    # Read node labels (if exists)
    node_labels = []
    node_labels_file = f"{prefix}_node_labels.txt"
    if os.path.exists(node_labels_file):
        with open(node_labels_file, 'r') as f:
            for line in f:
                node_labels.append(int(line.strip()))

    # Get number of graphs and nodes
    num_graphs = max(graph_indicator)
    num_nodes = len(graph_indicator)

    # Determine number of unique node labels
    if node_labels:
        num_node_labels = max(node_labels) + 1
    else:
        num_node_labels = 1

    # Build graph data objects
    data_list = []

    for g_idx in range(1, num_graphs + 1):
        # Get nodes belonging to this graph
        node_mask = [i for i, g in enumerate(graph_indicator) if g == g_idx]
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_mask)}

        n_nodes = len(node_mask)

        # Create node features (one-hot encoding of node labels)
        if node_labels:
            x = torch.zeros(n_nodes, num_node_labels)
            for new_idx, old_idx in enumerate(node_mask):
                label = node_labels[old_idx]
                if label < num_node_labels:
                    x[new_idx, label] = 1.0
        else:
            x = torch.ones(n_nodes, 1)

        # Get edges for this graph (1-indexed in TUDataset)
        graph_edges = []
        for u, v in edges:
            if u - 1 in node_mapping and v - 1 in node_mapping:
                new_u = node_mapping[u - 1]
                new_v = node_mapping[v - 1]
                graph_edges.append([new_u, new_v])

        if graph_edges:
            edge_index = torch.tensor(graph_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Graph label
        y = torch.tensor([graph_labels[g_idx - 1]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


class RealTUDataset(InMemoryDataset):
    """
    Dataset class for real TUDataset benchmarks.
    """

    def __init__(
        self,
        root: str,
        name: str,
        source_dir: str = None,
        transform=None,
        pre_transform=None
    ):
        self.name = name
        self.source_dir = source_dir or f"/tmp/{name}"
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}_A.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.name}_real_processed.pt']

    @property
    def num_node_features(self) -> int:
        return self[0].x.shape[1] if len(self) > 0 else 7

    @property
    def num_classes(self) -> int:
        return 2

    def download(self):
        pass

    def process(self):
        """Process raw TUDataset files."""
        data_list = parse_tu_dataset(self.source_dir, self.name)
        self.save(data_list, self.processed_paths[0])


def load_real_dataset(name: str, source_dir: str = None) -> RealTUDataset:
    """
    Load a real TUDataset benchmark.

    Args:
        name: Dataset name ('MUTAG', 'PTC_MR', 'NCI1')
        source_dir: Directory containing raw TUDataset files

    Returns:
        RealTUDataset object
    """
    if source_dir is None:
        source_dir = f"/tmp/{name}"

    root = f"data/RealTU_{name}"
    return RealTUDataset(root=root, name=name, source_dir=source_dir)


def get_dataset_stats(dataset) -> Dict:
    """Get statistics about a dataset."""
    n_samples = len(dataset)
    labels = [data.y.item() for data in dataset]
    n_pos = sum(labels)

    num_nodes = [data.x.shape[0] for data in dataset]
    num_edges = [data.edge_index.shape[1] // 2 for data in dataset]

    return {
        'n_samples': n_samples,
        'n_positive': n_pos,
        'n_negative': n_samples - n_pos,
        'positive_ratio': n_pos / n_samples,
        'avg_nodes': np.mean(num_nodes),
        'avg_edges': np.mean(num_edges),
        'num_features': dataset[0].x.shape[1]
    }


if __name__ == "__main__":
    print("Loading Real TUDataset Benchmarks")
    print("=" * 60)

    for name in ['MUTAG', 'PTC_MR', 'NCI1']:
        source_dir = f"/tmp/{name}"
        if os.path.exists(source_dir):
            print(f"\n{name}:")
            try:
                dataset = load_real_dataset(name, source_dir)
                stats = get_dataset_stats(dataset)
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\n{name}: Source files not found at {source_dir}")
