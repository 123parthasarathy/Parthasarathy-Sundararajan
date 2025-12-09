"""
Ablation Study for TIEGNN Framework.

Evaluates the contribution of each component:
1. Topological features (persistence homology)
2. Interpretable additive architecture
3. E(n)-equivariant processing

Configurations tested:
- TIEGNN (full): All components enabled
- TIEGNN w/o Topo: Without topological features
- TIEGNN w/o Interp: Without interpretable head
- TIEGNN w/o Equiv: Without equivariant layers
- Graph-only: Only GCN base (no topo, no interp, no equiv)
"""

import os
import sys
import json
import numpy as np
import torch
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tiegnn import TIEGNN, TIEGNNRegression
from src.data.datasets import get_dataset, get_dataloader_with_topo, prepare_dataset_with_topo
from src.data.topology import TopologicalFeatureExtractor
from src.training.trainer import train_model
from sklearn.model_selection import StratifiedKFold, KFold


def create_ablation_model(
    config_name: str,
    in_channels: int,
    num_classes: int,
    task: str,
    hidden_dim: int = 64,
    topo_dim: int = 24,
    has_coords: bool = False
) -> torch.nn.Module:
    """Create TIEGNN model with specific ablation configuration."""
    is_regression = (task == 'regression')

    configs = {
        'TIEGNN (full)': {'use_topo': True, 'use_interpretable': True, 'use_equiv': has_coords},
        'w/o Topo': {'use_topo': False, 'use_interpretable': True, 'use_equiv': has_coords},
        'w/o Interp': {'use_topo': True, 'use_interpretable': False, 'use_equiv': has_coords},
        'w/o Equiv': {'use_topo': True, 'use_interpretable': True, 'use_equiv': False},
        'Graph-only': {'use_topo': False, 'use_interpretable': False, 'use_equiv': False},
    }

    cfg = configs[config_name]

    if is_regression:
        return TIEGNNRegression(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            topo_dim=topo_dim,
            **cfg
        )
    else:
        return TIEGNN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            topo_dim=topo_dim,
            **cfg
        )


def run_ablation_study(
    dataset_name: str = 'MUTAG',
    n_folds: int = 10,
    epochs: int = 200,
    device: str = 'cpu',
    output_dir: str = 'results'
) -> Dict[str, Any]:
    """Run ablation study on a dataset."""
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {dataset_name}")
    print('='*60)

    # Load dataset
    dataset, info = get_dataset(dataset_name)
    task = info['task']
    num_classes = info.get('num_classes', 1)
    in_channels = info['num_features']
    has_coords = info.get('has_coordinates', False)

    print(f"Task: {task}")
    print(f"Graphs: {info['num_graphs']}")
    print(f"Has coordinates: {has_coords}")

    # Prepare topological features
    print("Computing topological features...")
    extractor = TopologicalFeatureExtractor()
    dataset_with_topo = prepare_dataset_with_topo(dataset, extractor, verbose=True)

    # Cross-validation splits
    if task == 'classification':
        labels = np.array([data.y.item() for data in dataset])
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(range(len(dataset)), labels))
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(range(len(dataset))))

    # Ablation configurations
    configs = ['TIEGNN (full)', 'w/o Topo', 'w/o Interp', 'w/o Equiv', 'Graph-only']

    results = {}

    for config_name in configs:
        print(f"\n  Configuration: {config_name}")
        fold_metrics = defaultdict(list)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"    Fold {fold_idx + 1}/{n_folds}...", end=" ")

            # Split data
            train_data = [dataset_with_topo[i] for i in train_idx[:int(len(train_idx)*0.9)]]
            val_data = [dataset_with_topo[i] for i in train_idx[int(len(train_idx)*0.9):]]
            test_data = [dataset_with_topo[i] for i in test_idx]

            # Create data loaders
            train_loader = get_dataloader_with_topo(train_data, batch_size=32, shuffle=True)
            val_loader = get_dataloader_with_topo(val_data, batch_size=32, shuffle=False)
            test_loader = get_dataloader_with_topo(test_data, batch_size=32, shuffle=False)

            # Create model
            np.random.seed(42 + fold_idx)
            torch.manual_seed(42 + fold_idx)

            model = create_ablation_model(
                config_name, in_channels, num_classes, task,
                hidden_dim=64, topo_dim=24, has_coords=has_coords
            )

            # Train
            result = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                task=task,
                epochs=epochs,
                lr=1e-3,
                device=device,
                patience=30,
                verbose=False
            )

            # Record metrics
            for metric, value in result['test_metrics'].items():
                fold_metrics[metric].append(value)

            primary_metric = 'accuracy' if task == 'classification' else 'mae'
            print(f"{primary_metric}={result['test_metrics'][primary_metric]:.4f}")

        # Aggregate
        results[config_name] = {
            'metrics': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
                for metric, values in fold_metrics.items()
            }
        }

        # Print summary
        primary_metric = 'accuracy' if task == 'classification' else 'mae'
        stats = results[config_name]['metrics'][primary_metric]
        print(f"    Mean {primary_metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'ablation_{dataset_name}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x if not isinstance(x, np.ndarray) else x.tolist())

    print(f"\nAblation results saved to {output_file}")

    return results


def print_ablation_table(results: Dict[str, Any], task: str = 'classification'):
    """Print ablation results in a formatted table."""
    from tabulate import tabulate

    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    headers = ['Configuration']
    primary_metric = 'accuracy' if task == 'classification' else 'mae'
    headers.append(f'{primary_metric} (mean +/- std)')

    if 'auc' in list(results.values())[0]['metrics']:
        headers.append('AUC (mean +/- std)')

    rows = []
    for config_name, result in results.items():
        row = [config_name]
        stats = result['metrics'][primary_metric]
        row.append(f"{stats['mean']:.4f} +/- {stats['std']:.4f}")

        if 'auc' in result['metrics']:
            auc_stats = result['metrics']['auc']
            row.append(f"{auc_stats['mean']:.4f} +/- {auc_stats['std']:.4f}")

        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt='grid'))

    # Compute relative performance
    print("\nRelative Performance (vs full TIEGNN):")
    full_perf = results['TIEGNN (full)']['metrics'][primary_metric]['mean']

    for config_name, result in results.items():
        perf = result['metrics'][primary_metric]['mean']
        if task == 'classification':
            diff = (perf - full_perf) * 100  # Percentage points
            print(f"  {config_name}: {diff:+.2f} pp")
        else:
            diff = ((perf - full_perf) / full_perf) * 100  # Percentage change
            print(f"  {config_name}: {diff:+.2f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run TIEGNN ablation study')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                       help='Dataset for ablation study')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n-folds', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    results = run_ablation_study(
        dataset_name=args.dataset,
        n_folds=args.n_folds,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir
    )

    # Get task type
    _, info = get_dataset(args.dataset)
    print_ablation_table(results, info['task'])


if __name__ == '__main__':
    main()
