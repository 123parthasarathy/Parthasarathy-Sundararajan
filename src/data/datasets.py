"""
Dataset loaders for real-world graph benchmarks:
- MUTAG: Mutagenicity prediction (classification)
- PROTEINS: Protein function prediction (classification)
- QM9: Quantum molecular properties (regression)
- ZINC: Molecular property prediction (regression)
"""

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import Tuple, Optional, List, Dict, Any
import os


def get_dataset(
    name: str,
    root: str = './data',
    transform=None,
    pre_transform=None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a graph dataset.

    Args:
        name: Dataset name ('MUTAG', 'PROTEINS', 'QM9', 'ZINC')
        root: Root directory for data storage
        transform: Optional transform to apply
        pre_transform: Optional pre-transform to apply

    Returns:
        dataset: The loaded dataset
        info: Dictionary with dataset metadata
    """
    name = name.upper()

    if name == 'MUTAG':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root=root, name='MUTAG', transform=transform, pre_transform=pre_transform)
        info = {
            'task': 'classification',
            'num_classes': 2,
            'num_features': dataset.num_features,
            'num_graphs': len(dataset),
            'metric': 'accuracy',
            'has_coordinates': False
        }

    elif name == 'PROTEINS':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root=root, name='PROTEINS', transform=transform, pre_transform=pre_transform)
        info = {
            'task': 'classification',
            'num_classes': 2,
            'num_features': dataset.num_features,
            'num_graphs': len(dataset),
            'metric': 'accuracy',
            'has_coordinates': False
        }

    elif name == 'QM9':
        from torch_geometric.datasets import QM9
        dataset = QM9(root=root, transform=transform, pre_transform=pre_transform)
        # Use target 0 (dipole moment mu) as default
        info = {
            'task': 'regression',
            'num_targets': 19,
            'target_names': ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                           'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom',
                           'u298_atom', 'h298_atom', 'g298_atom', 'A', 'B', 'C'],
            'num_features': dataset.num_features,
            'num_graphs': len(dataset),
            'metric': 'mae',
            'has_coordinates': True
        }

    elif name == 'ZINC':
        from torch_geometric.datasets import ZINC
        train_dataset = ZINC(root=root, subset=True, split='train', transform=transform, pre_transform=pre_transform)
        val_dataset = ZINC(root=root, subset=True, split='val', transform=transform, pre_transform=pre_transform)
        test_dataset = ZINC(root=root, subset=True, split='test', transform=transform, pre_transform=pre_transform)

        # Combine for info, but return split datasets
        dataset = (train_dataset, val_dataset, test_dataset)
        info = {
            'task': 'regression',
            'num_features': train_dataset.num_features,
            'num_graphs': len(train_dataset) + len(val_dataset) + len(test_dataset),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'metric': 'mae',
            'has_coordinates': False,
            'pre_split': True
        }

    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: MUTAG, PROTEINS, QM9, ZINC")

    return dataset, info


def train_val_test_split(
    dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List, List, List]:
    """Split dataset into train/val/test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))

    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]

    return train_data, val_data, test_data


def get_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


class DatasetWithTopoFeatures:
    """Wrapper that adds precomputed topological features to a dataset."""

    def __init__(self, dataset, topo_features: torch.Tensor):
        self.dataset = dataset
        self.topo_features = topo_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Add topological features
        data.topo_features = self.topo_features[idx]
        return data


def prepare_dataset_with_topo(
    dataset,
    extractor=None,
    verbose: bool = True
) -> DatasetWithTopoFeatures:
    """
    Prepare dataset with precomputed topological features.

    Args:
        dataset: PyG dataset
        extractor: TopologicalFeatureExtractor instance (created if None)
        verbose: Whether to show progress

    Returns:
        DatasetWithTopoFeatures: Dataset with topological features attached
    """
    from .topology import TopologicalFeatureExtractor

    if extractor is None:
        extractor = TopologicalFeatureExtractor()

    topo_features = extractor.extract_batch(dataset, verbose=verbose)

    return DatasetWithTopoFeatures(dataset, topo_features)


def collate_with_topo(batch):
    """Custom collate function that handles topological features."""
    from torch_geometric.data import Batch

    # Check if batch items have topo_features
    has_topo = hasattr(batch[0], 'topo_features')

    # Collect topo features before batching
    if has_topo:
        topo_features = torch.stack([data.topo_features for data in batch])

    # Standard PyG batching
    batched = Batch.from_data_list(batch)

    # Add topo features to batch
    if has_topo:
        batched.topo_features = topo_features

    return batched


def get_dataloader_with_topo(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader with custom collate for topological features."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_with_topo
    )
