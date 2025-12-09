"""
Main experiment script for benchmarking TIEGNN against baselines on real-world datasets.

Datasets:
- MUTAG: Mutagenicity prediction (188 graphs, binary classification)
- PROTEINS: Protein function prediction (1113 graphs, binary classification)
- QM9: Quantum molecular properties (130k graphs, regression)
- ZINC: Molecular property prediction (12k graphs, regression)

Baselines:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network
- EGNN: E(n) Equivariant GNN
- PersLay: Persistence-based GNN

Metrics:
- Classification: Accuracy, AUC-ROC, F1
- Regression: MAE, RMSE
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tiegnn import TIEGNN, TIEGNNRegression
from src.models.baselines import (
    GCN, GAT, EGNN, PersLay,
    GCNRegression, GATRegression, EGNNRegression, PersLayRegression
)
from src.data.datasets import (
    get_dataset, train_val_test_split, get_dataloader_with_topo,
    prepare_dataset_with_topo
)
from src.data.topology import TopologicalFeatureExtractor
from src.training.trainer import train_model, evaluate_model


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    task: str,
    hidden_dim: int = 64,
    topo_dim: int = 24,
    has_coords: bool = False
) -> torch.nn.Module:
    """Create a model by name."""
    is_regression = (task == 'regression')

    if model_name == 'TIEGNN':
        if is_regression:
            return TIEGNNRegression(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                topo_dim=topo_dim,
                use_equiv=has_coords,
                use_topo=True,
                use_interpretable=True
            )
        else:
            return TIEGNN(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                topo_dim=topo_dim,
                use_equiv=has_coords,
                use_topo=True,
                use_interpretable=True
            )

    elif model_name == 'GCN':
        if is_regression:
            return GCNRegression(in_channels, hidden_dim, num_classes)
        else:
            return GCN(in_channels, hidden_dim, num_classes)

    elif model_name == 'GAT':
        if is_regression:
            return GATRegression(in_channels, hidden_dim, num_classes)
        else:
            return GAT(in_channels, hidden_dim, num_classes)

    elif model_name == 'EGNN':
        if is_regression:
            return EGNNRegression(in_channels, hidden_dim, num_classes)
        else:
            return EGNN(in_channels, hidden_dim, num_classes)

    elif model_name == 'PersLay':
        if is_regression:
            return PersLayRegression(in_channels, hidden_dim, num_classes, topo_dim=topo_dim)
        else:
            return PersLay(in_channels, hidden_dim, num_classes, topo_dim=topo_dim)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    train_data: List,
    val_data: List,
    test_data: List,
    info: Dict,
    device: str,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42
) -> Dict[str, Any]:
    """Run a single experiment (one model on one dataset)."""
    set_seed(seed)

    # Get data info
    task = info['task']
    num_classes = info.get('num_classes', 1)
    in_channels = info['num_features']
    has_coords = info.get('has_coordinates', False)

    # Create data loaders
    train_loader = get_dataloader_with_topo(train_data, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader_with_topo(val_data, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader_with_topo(test_data, batch_size=batch_size, shuffle=False)

    # Create model
    model = get_model(
        model_name, in_channels, num_classes, task,
        hidden_dim=64, topo_dim=24, has_coords=has_coords
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Train
    start_time = time.time()
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        task=task,
        epochs=epochs,
        lr=lr,
        device=device,
        patience=30,
        verbose=False
    )
    train_time = time.time() - start_time

    return {
        'model': model_name,
        'dataset': dataset_name,
        'test_metrics': results.get('test_metrics', {}),
        'best_val_metric': results['best_val_metric'],
        'best_epoch': results['best_epoch'],
        'train_time': train_time,
        'num_params': num_params
    }


def run_cross_validation(
    model_name: str,
    dataset_name: str,
    dataset,
    info: Dict,
    device: str,
    n_folds: int = 10,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3
) -> Dict[str, Any]:
    """Run k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold, KFold

    task = info['task']

    # Prepare topological features
    print(f"  Computing topological features for {dataset_name}...")
    extractor = TopologicalFeatureExtractor()
    dataset_with_topo = prepare_dataset_with_topo(dataset, extractor, verbose=False)

    # Get labels for stratification
    if task == 'classification':
        labels = np.array([data.y.item() for data in dataset])
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(range(len(dataset)), labels))
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(range(len(dataset))))

    # Run folds
    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"    Fold {fold_idx + 1}/{n_folds}...")

        # Split data
        train_data = [dataset_with_topo[i] for i in train_idx[:int(len(train_idx)*0.9)]]
        val_data = [dataset_with_topo[i] for i in train_idx[int(len(train_idx)*0.9):]]
        test_data = [dataset_with_topo[i] for i in test_idx]

        # Run experiment
        result = run_single_experiment(
            model_name=model_name,
            dataset_name=dataset_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            info=info,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=42 + fold_idx
        )
        fold_results.append(result)

    # Aggregate results
    all_metrics = defaultdict(list)
    for result in fold_results:
        for metric, value in result['test_metrics'].items():
            all_metrics[metric].append(value)

    aggregated = {
        'model': model_name,
        'dataset': dataset_name,
        'n_folds': n_folds,
        'metrics': {}
    }

    for metric, values in all_metrics.items():
        aggregated['metrics'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }

    aggregated['train_time_mean'] = np.mean([r['train_time'] for r in fold_results])
    aggregated['num_params'] = fold_results[0]['num_params']

    return aggregated


def run_all_experiments(
    datasets: List[str],
    models: List[str],
    device: str,
    n_folds: int = 10,
    epochs: int = 200,
    output_dir: str = 'results'
) -> Dict[str, Any]:
    """Run all experiments."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)

        # Load dataset
        try:
            dataset, info = get_dataset(dataset_name)
            print(f"  Task: {info['task']}")
            print(f"  Graphs: {info['num_graphs']}")
            print(f"  Features: {info['num_features']}")
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            continue

        # Handle pre-split datasets (ZINC)
        if info.get('pre_split', False):
            train_dataset, val_dataset, test_dataset = dataset
            print(f"  Using pre-defined splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

            # Prepare topo features for each split
            extractor = TopologicalFeatureExtractor()
            train_data = prepare_dataset_with_topo(train_dataset, extractor, verbose=False)
            val_data = prepare_dataset_with_topo(val_dataset, extractor, verbose=False)
            test_data = prepare_dataset_with_topo(test_dataset, extractor, verbose=False)

            dataset_results = {}
            for model_name in models:
                print(f"\n  Model: {model_name}")
                try:
                    result = run_single_experiment(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        train_data=list(train_data),
                        val_data=list(val_data),
                        test_data=list(test_data),
                        info=info,
                        device=device,
                        epochs=epochs,
                        batch_size=32,
                        lr=1e-3,
                        seed=42
                    )
                    dataset_results[model_name] = result
                    print(f"    Results: {result['test_metrics']}")
                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()

        else:
            # Cross-validation for other datasets
            dataset_results = {}
            for model_name in models:
                print(f"\n  Model: {model_name}")
                try:
                    result = run_cross_validation(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        info=info,
                        device=device,
                        n_folds=n_folds,
                        epochs=epochs
                    )
                    dataset_results[model_name] = result

                    # Print results
                    for metric, stats in result['metrics'].items():
                        print(f"    {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()

        all_results[dataset_name] = dataset_results

    # Save results
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    print(f"\nResults saved to {results_file}")

    return all_results


def print_results_table(results: Dict[str, Any]):
    """Print results in a formatted table."""
    from tabulate import tabulate

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        print("-" * 60)

        headers = ['Model']
        rows = []

        # Determine metrics
        first_model = list(dataset_results.values())[0]
        if 'metrics' in first_model:
            # Cross-validation results
            metrics = list(first_model['metrics'].keys())
            headers.extend([f"{m} (mean +/- std)" for m in metrics])

            for model_name, result in dataset_results.items():
                row = [model_name]
                for metric in metrics:
                    stats = result['metrics'][metric]
                    row.append(f"{stats['mean']:.4f} +/- {stats['std']:.4f}")
                rows.append(row)
        else:
            # Single split results
            metrics = list(first_model['test_metrics'].keys())
            headers.extend(metrics)

            for model_name, result in dataset_results.items():
                row = [model_name]
                for metric in metrics:
                    row.append(f"{result['test_metrics'][metric]:.4f}")
                rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt='grid'))


def main():
    parser = argparse.ArgumentParser(description='Run TIEGNN benchmarks')
    parser.add_argument('--datasets', nargs='+', default=['MUTAG', 'PROTEINS'],
                       help='Datasets to evaluate')
    parser.add_argument('--models', nargs='+', default=['GCN', 'GAT', 'PersLay', 'TIEGNN'],
                       help='Models to evaluate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    print(f"Running experiments on device: {args.device}")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")

    results = run_all_experiments(
        datasets=args.datasets,
        models=args.models,
        device=args.device,
        n_folds=args.n_folds,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    print_results_table(results)


if __name__ == '__main__':
    main()
