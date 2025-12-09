#!/usr/bin/env python3
"""
Run experiments on REAL datasets: MUTAG and PROTEINS.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tiegnn import TIEGNN
from src.models.baselines import GCN, GAT, PersLay
from src.data.topology import compute_topological_features
from src.training.trainer import train_model


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_topological_features(dataset, verbose: bool = True):
    """Add topological features to each graph."""
    if verbose:
        print("Computing topological features...")

    processed = []
    for i, data in enumerate(dataset):
        if verbose and i % 50 == 0:
            print(f"  Processing graph {i}/{len(dataset)}...")

        edge_index = data.edge_index.numpy()
        num_nodes = data.x.size(0)

        topo = compute_topological_features(edge_index, num_nodes, compute_edge_persistence=True)

        # Create new data with topo features
        new_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            topo_features=torch.tensor(topo, dtype=torch.float32)
        )
        if hasattr(data, 'edge_attr'):
            new_data.edge_attr = data.edge_attr
        processed.append(new_data)

    return processed


class TopoDataLoader(DataLoader):
    """Custom DataLoader that properly handles topological features."""

    def __init__(self, dataset, batch_size=32, shuffle=True, **kwargs):
        self.all_topo_features = torch.stack([d.topo_features.clone() for d in dataset])

        clean_dataset = []
        for i, data in enumerate(dataset):
            new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
            new_data.idx = i
            if hasattr(data, 'edge_attr'):
                new_data.edge_attr = data.edge_attr
            clean_dataset.append(new_data)

        super().__init__(clean_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self._topo_features = self.all_topo_features

    def __iter__(self):
        for batch in super().__iter__():
            batch.topo_features = self._topo_features[batch.idx]
            yield batch


def run_experiment(
    model_name: str,
    train_data: List[Data],
    val_data: List[Data],
    test_data: List[Data],
    in_channels: int,
    num_classes: int,
    device: str = 'cpu',
    epochs: int = 200,
    seed: int = 42
) -> Dict[str, Any]:
    """Run single experiment."""
    set_seed(seed)

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
        patience=30,
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
    in_channels: int,
    num_classes: int,
    n_folds: int = 10,
    epochs: int = 200,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run k-fold cross-validation."""
    labels = np.array([data.y.item() for data in dataset])
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kfold.split(range(len(dataset)), labels))

    fold_metrics = defaultdict(list)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"    Fold {fold_idx + 1}/{n_folds}...", end=" ", flush=True)

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
            in_channels=in_channels,
            num_classes=num_classes,
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
    in_channels: int,
    num_classes: int,
    n_folds: int = 10,
    epochs: int = 200,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run ablation study on TIEGNN."""
    print("\n  Running ablation study...")

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
            print(f"      Fold {fold_idx + 1}/{n_folds}...", end=" ", flush=True)

            train_size = int(len(train_idx) * 0.9)
            train_data = [dataset[i] for i in train_idx[:train_size]]
            val_data = [dataset[i] for i in train_idx[train_size:]]
            test_data = [dataset[i] for i in test_idx]

            set_seed(42 + fold_idx)
            model = TIEGNN(
                in_channels=in_channels,
                hidden_dim=64,
                num_classes=num_classes,
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
                patience=30,
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

    parser = argparse.ArgumentParser(description='Run real dataset experiments')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n-folds', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("TIEGNN EXPERIMENTS ON REAL DATASETS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load datasets
    datasets_config = {
        'MUTAG': {'use_node_attr': True},
        'PROTEINS': {'use_node_attr': True},
    }

    models = ['GCN', 'GAT', 'PersLay', 'TIEGNN']
    all_results = {}
    ablation_results = {}

    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)

        # Load dataset
        raw_dataset = TUDataset(root='./data', name=dataset_name, **config)
        in_channels = raw_dataset.num_features
        num_classes = raw_dataset.num_classes

        print(f"  Graphs: {len(raw_dataset)}, Features: {in_channels}, Classes: {num_classes}")

        # Add topological features
        dataset = add_topological_features(raw_dataset, verbose=True)

        dataset_results = {}

        for model_name in models:
            print(f"\n  Model: {model_name}")
            result = run_cross_validation(
                model_name=model_name,
                dataset=dataset,
                in_channels=in_channels,
                num_classes=num_classes,
                n_folds=args.n_folds,
                epochs=args.epochs,
                device=args.device
            )
            dataset_results[model_name] = result

            # Print summary
            acc_stats = result['metrics']['accuracy']
            print(f"    Mean accuracy: {acc_stats['mean']:.4f} +/- {acc_stats['std']:.4f}")

        all_results[dataset_name] = dataset_results

        # Ablation study on MUTAG
        if dataset_name == 'MUTAG':
            ablation_results = run_ablation_experiment(
                dataset=dataset,
                in_channels=in_channels,
                num_classes=num_classes,
                n_folds=args.n_folds,
                epochs=args.epochs,
                device=args.device
            )

    # Save results
    results_file = os.path.join(args.output_dir, 'real_benchmark_results.json')
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
        print("ABLATION STUDY RESULTS (MUTAG)")
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
