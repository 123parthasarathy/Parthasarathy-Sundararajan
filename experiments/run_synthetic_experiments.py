#!/usr/bin/env python3
"""
Run experiments on synthetic datasets when real datasets are not available.

Creates synthetic datasets that mimic properties of:
- MUTAG: Small molecular graphs (~18 nodes, binary classification)
- PROTEINS: Protein structure graphs (~39 nodes, binary classification)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tiegnn import TIEGNN
from src.models.baselines import GCN, GAT, PersLay
from src.data.topology import compute_topological_features
from src.training.trainer import train_model


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_synthetic_graph(
    num_nodes_mean: int,
    num_nodes_std: int,
    num_features: int,
    edge_density: float,
    label: int
) -> Data:
    """Generate a single synthetic graph."""
    # Random number of nodes
    num_nodes = max(5, int(np.random.normal(num_nodes_mean, num_nodes_std)))

    # Node features (random)
    x = torch.randn(num_nodes, num_features)

    # Add some label-dependent signal to features
    if label == 1:
        x[:, 0] += 0.5  # Slight positive shift for class 1

    # Generate edges (random graph with given density)
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = max(num_nodes - 1, int(num_possible_edges * edge_density))

    # Start with spanning tree for connectivity
    edges = []
    for i in range(1, num_nodes):
        j = np.random.randint(0, i)
        edges.append([i, j])
        edges.append([j, i])

    # Add additional edges
    while len(edges) < num_edges * 2:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and [i, j] not in edges:
            edges.append([i, j])
            edges.append([j, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label]))


def generate_synthetic_dataset(
    name: str,
    num_graphs: int,
    num_nodes_mean: int,
    num_nodes_std: int,
    num_features: int,
    edge_density: float
) -> Tuple[List[Data], Dict[str, Any]]:
    """Generate a synthetic dataset."""
    print(f"Generating {name} synthetic dataset ({num_graphs} graphs)...")

    graphs = []
    for i in range(num_graphs):
        label = i % 2  # Alternating labels for balance
        graph = generate_synthetic_graph(
            num_nodes_mean, num_nodes_std, num_features, edge_density, label
        )
        graphs.append(graph)

    info = {
        'name': name,
        'task': 'classification',
        'num_classes': 2,
        'num_features': num_features,
        'num_graphs': num_graphs,
        'metric': 'accuracy',
        'has_coordinates': False,
        'synthetic': True
    }

    return graphs, info


def add_topological_features(dataset: List[Data], verbose: bool = True) -> List[Data]:
    """Add topological features to each graph."""
    if verbose:
        print("Computing topological features...")

    for i, data in enumerate(dataset):
        edge_index = data.edge_index.numpy()
        num_nodes = data.x.size(0)

        topo = compute_topological_features(edge_index, num_nodes, compute_edge_persistence=True)
        data.topo_features = torch.tensor(topo, dtype=torch.float32)

    return dataset


class TopoDataLoader(DataLoader):
    """Custom DataLoader that properly handles topological features."""

    def __init__(self, dataset, batch_size=32, shuffle=True, **kwargs):
        # Extract topo features to separate tensor
        self.all_topo_features = torch.stack([d.topo_features.clone() for d in dataset])

        # Create dataset without topo features
        clean_dataset = []
        for i, data in enumerate(dataset):
            new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
            new_data.idx = i  # Store original index
            clean_dataset.append(new_data)

        super().__init__(clean_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self._topo_features = self.all_topo_features

    def __iter__(self):
        for batch in super().__iter__():
            # Get topo features for this batch using stored indices
            batch.topo_features = self._topo_features[batch.idx]
            yield batch


def collate_with_topo(batch: List[Data]):
    """Custom collate function for batching graphs with topo features."""
    from torch_geometric.data import Batch

    # Check if data has topo_features
    has_topo = hasattr(batch[0], 'topo_features') and batch[0].topo_features is not None

    if has_topo:
        # Extract topo features before batching
        topo_features = torch.stack([data.topo_features.clone() for data in batch])

        # Create copies without topo_features for standard batching
        batch_copy = []
        for data in batch:
            new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
            if hasattr(data, 'idx'):
                new_data.idx = data.idx
            batch_copy.append(new_data)

        batched = Batch.from_data_list(batch_copy)
        batched.topo_features = topo_features
    else:
        batched = Batch.from_data_list(batch)

    return batched


def train_val_test_split(dataset: List[Data], seed: int = 42):
    """Split dataset into train/val/test."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))

    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    return train_data, val_data, test_data


def run_experiment(
    model_name: str,
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    info: Dict[str, Any],
    device: str = 'cpu',
    epochs: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Run single experiment."""
    set_seed(seed)

    in_channels = info['num_features']
    num_classes = info['num_classes']
    hidden_dim = 64

    # Create model
    if model_name == 'GCN':
        model = GCN(in_channels, hidden_dim, num_classes)
    elif model_name == 'GAT':
        model = GAT(in_channels, hidden_dim, num_classes)
    elif model_name == 'PersLay':
        model = PersLay(in_channels, hidden_dim, num_classes, topo_dim=24)
    elif model_name == 'TIEGNN':
        model = TIEGNN(in_channels, hidden_dim, num_classes, topo_dim=24,
                      use_topo=True, use_interpretable=True, use_equiv=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create dataloaders
    train_loader = TopoDataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = TopoDataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = TopoDataLoader(test_data, batch_size=32, shuffle=False)

    # Train
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task='classification',
        epochs=epochs,
        lr=1e-3,
        device=device,
        patience=20,
        verbose=False
    )

    return {
        'model': model_name,
        'test_metrics': results.get('test_metrics', {}),
        'best_val_metric': results['best_val_metric'],
        'best_epoch': results['best_epoch']
    }


def run_cross_validation(
    model_name: str,
    dataset: List[Data],
    info: Dict[str, Any],
    n_folds: int = 5,
    epochs: int = 100,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold

    labels = np.array([data.y.item() for data in dataset])
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kfold.split(range(len(dataset)), labels))

    fold_metrics = defaultdict(list)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"    Fold {fold_idx + 1}/{n_folds}...", end=" ")

        # Split
        train_size = int(len(train_idx) * 0.9)
        train_data = [dataset[i] for i in train_idx[:train_size]]
        val_data = [dataset[i] for i in train_idx[train_size:]]
        test_data = [dataset[i] for i in test_idx]

        # Run experiment
        result = run_experiment(
            model_name=model_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            info=info,
            device=device,
            epochs=epochs,
            seed=42 + fold_idx
        )

        for metric, value in result['test_metrics'].items():
            fold_metrics[metric].append(value)

        print(f"accuracy={result['test_metrics'].get('accuracy', 0):.4f}")

    # Aggregate
    aggregated = {
        'model': model_name,
        'metrics': {}
    }

    for metric, values in fold_metrics.items():
        aggregated['metrics'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    return aggregated


def run_ablation_experiment(
    dataset: List[Data],
    info: Dict[str, Any],
    n_folds: int = 5,
    epochs: int = 100,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run ablation study on TIEGNN."""
    print("\n  Running ablation study...")

    from sklearn.model_selection import StratifiedKFold

    labels = np.array([data.y.item() for data in dataset])
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kfold.split(range(len(dataset)), labels))

    configs = {
        'TIEGNN (full)': {'use_topo': True, 'use_interpretable': True, 'use_equiv': False},
        'w/o Topo': {'use_topo': False, 'use_interpretable': True, 'use_equiv': False},
        'w/o Interp': {'use_topo': True, 'use_interpretable': False, 'use_equiv': False},
        'Graph-only': {'use_topo': False, 'use_interpretable': False, 'use_equiv': False},
    }

    ablation_results = {}

    for config_name, config in configs.items():
        print(f"\n    Configuration: {config_name}")
        fold_metrics = defaultdict(list)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"      Fold {fold_idx + 1}/{n_folds}...", end=" ")

            train_size = int(len(train_idx) * 0.9)
            train_data = [dataset[i] for i in train_idx[:train_size]]
            val_data = [dataset[i] for i in train_idx[train_size:]]
            test_data = [dataset[i] for i in test_idx]

            # Create model with specific config
            set_seed(42 + fold_idx)
            model = TIEGNN(
                in_channels=info['num_features'],
                hidden_dim=64,
                num_classes=info['num_classes'],
                topo_dim=24,
                **config
            )

            train_loader = TopoDataLoader(train_data, batch_size=32, shuffle=True)
            val_loader = TopoDataLoader(val_data, batch_size=32, shuffle=False)
            test_loader = TopoDataLoader(test_data, batch_size=32, shuffle=False)

            results = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                task='classification',
                epochs=epochs,
                lr=1e-3,
                device=device,
                patience=20,
                verbose=False
            )

            for metric, value in results['test_metrics'].items():
                fold_metrics[metric].append(value)

            print(f"accuracy={results['test_metrics'].get('accuracy', 0):.4f}")

        ablation_results[config_name] = {
            'metrics': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
                for metric, values in fold_metrics.items()
            }
        }

    return ablation_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run synthetic experiments')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("TIEGNN EXPERIMENTS ON SYNTHETIC DATA")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Generate datasets mimicking MUTAG and PROTEINS
    datasets = {
        'MUTAG-like': generate_synthetic_dataset(
            'MUTAG-like', num_graphs=188, num_nodes_mean=18, num_nodes_std=5,
            num_features=7, edge_density=0.15
        ),
        'PROTEINS-like': generate_synthetic_dataset(
            'PROTEINS-like', num_graphs=200, num_nodes_mean=39, num_nodes_std=10,
            num_features=3, edge_density=0.08
        ),
    }

    models = ['GCN', 'GAT', 'PersLay', 'TIEGNN']
    all_results = {}
    ablation_results = {}

    for dataset_name, (dataset, info) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"  Graphs: {info['num_graphs']}, Features: {info['num_features']}")
        print('='*60)

        # Add topological features
        dataset = add_topological_features(dataset, verbose=True)

        dataset_results = {}

        for model_name in models:
            print(f"\n  Model: {model_name}")
            result = run_cross_validation(
                model_name=model_name,
                dataset=dataset,
                info=info,
                n_folds=args.n_folds,
                epochs=args.epochs,
                device=args.device
            )
            dataset_results[model_name] = result

            # Print summary
            acc_stats = result['metrics']['accuracy']
            print(f"    Mean accuracy: {acc_stats['mean']:.4f} +/- {acc_stats['std']:.4f}")

        all_results[dataset_name] = dataset_results

        # Ablation study on first dataset only
        if dataset_name == 'MUTAG-like':
            ablation_results = run_ablation_experiment(
                dataset=dataset,
                info=info,
                n_folds=args.n_folds,
                epochs=args.epochs,
                device=args.device
            )

    # Save results
    results_file = os.path.join(args.output_dir, 'synthetic_benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'benchmark_results': all_results,
            'ablation_results': ablation_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    from tabulate import tabulate

    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        headers = ['Model', 'Accuracy (mean +/- std)', 'AUC', 'F1']
        rows = []

        for model_name, result in dataset_results.items():
            acc = result['metrics']['accuracy']
            auc = result['metrics'].get('auc', {'mean': 0, 'std': 0})
            f1_score = result['metrics'].get('f1', {'mean': 0, 'std': 0})
            rows.append([
                model_name,
                f"{acc['mean']:.4f} +/- {acc['std']:.4f}",
                f"{auc['mean']:.4f} +/- {auc['std']:.4f}",
                f"{f1_score['mean']:.4f} +/- {f1_score['std']:.4f}"
            ])

        print(tabulate(rows, headers=headers, tablefmt='grid'))

    # Ablation table
    if ablation_results:
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS (MUTAG-like)")
        print("="*80)

        headers = ['Configuration', 'Accuracy', 'Relative']
        rows = []
        full_acc = ablation_results['TIEGNN (full)']['metrics']['accuracy']['mean']

        for config_name, result in ablation_results.items():
            acc = result['metrics']['accuracy']
            diff = (acc['mean'] - full_acc) * 100
            rows.append([
                config_name,
                f"{acc['mean']:.4f} +/- {acc['std']:.4f}",
                f"{diff:+.2f} pp"
            ])

        print(tabulate(rows, headers=headers, tablefmt='grid'))

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
