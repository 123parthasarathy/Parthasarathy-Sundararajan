"""
Runtime Analysis for TIEGNN and Baselines.

Measures:
1. Forward pass time
2. Backward pass time
3. Total training time per epoch
4. Memory usage
5. Topological feature extraction time
6. Scalability with graph size
"""

import os
import sys
import json
import time
import numpy as np
import torch
from typing import Dict, List, Any
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tiegnn import TIEGNN
from src.models.baselines import GCN, GAT, EGNN, PersLay
from src.data.topology import compute_topological_features


def create_synthetic_batch(
    num_graphs: int,
    avg_nodes: int,
    avg_edges_per_node: int,
    num_features: int,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Create a synthetic batch of graphs for benchmarking."""
    all_x = []
    all_edge_index = []
    all_batch = []
    all_topo = []
    all_pos = []

    node_offset = 0

    for g in range(num_graphs):
        # Random number of nodes
        n_nodes = max(3, int(np.random.normal(avg_nodes, avg_nodes * 0.2)))
        n_edges = max(n_nodes - 1, n_nodes * avg_edges_per_node)

        # Node features
        x = torch.randn(n_nodes, num_features)
        all_x.append(x)

        # Node positions (for equivariant models)
        pos = torch.randn(n_nodes, 3)
        all_pos.append(pos)

        # Random edges (ensure connected)
        edges = []
        # Spanning tree for connectivity
        for i in range(1, n_nodes):
            j = np.random.randint(0, i)
            edges.append((i, j))
            edges.append((j, i))

        # Additional random edges
        for _ in range(n_edges - n_nodes + 1):
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i != j:
                edges.append((i, j))
                edges.append((j, i))

        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_index = edge_index + node_offset
        all_edge_index.append(edge_index)

        # Batch assignment
        batch = torch.full((n_nodes,), g, dtype=torch.long)
        all_batch.append(batch)

        # Topological features (edges are local indices, no offset needed)
        topo = compute_topological_features(
            np.array(edges).T if len(edges) > 0 else np.array([[], []]),
            n_nodes,
            compute_edge_persistence=True
        )
        all_topo.append(torch.tensor(topo, dtype=torch.float32))

        node_offset += n_nodes

    # Combine
    batch_data = {
        'x': torch.cat(all_x, dim=0).to(device),
        'edge_index': torch.cat(all_edge_index, dim=1).to(device),
        'batch': torch.cat(all_batch, dim=0).to(device),
        'topo_features': torch.stack(all_topo, dim=0).to(device),
        'pos': torch.cat(all_pos, dim=0).to(device),
        'num_graphs': num_graphs
    }

    return batch_data


def benchmark_forward_pass(
    model: torch.nn.Module,
    batch_data: Dict[str, torch.Tensor],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """Benchmark forward pass time."""
    model.eval()

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(
                batch_data['x'],
                batch_data['edge_index'],
                batch_data['batch'],
                topo_features=batch_data.get('topo_features'),
                pos=batch_data.get('pos')
            )

    # Synchronize if using CUDA
    if batch_data['x'].is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(
                batch_data['x'],
                batch_data['edge_index'],
                batch_data['batch'],
                topo_features=batch_data.get('topo_features'),
                pos=batch_data.get('pos')
            )

        if batch_data['x'].is_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def benchmark_backward_pass(
    model: torch.nn.Module,
    batch_data: Dict[str, torch.Tensor],
    num_runs: int = 50,
    warmup_runs: int = 5
) -> Dict[str, float]:
    """Benchmark backward pass time."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Create dummy target
    target = torch.randint(0, 2, (batch_data['num_graphs'],), device=batch_data['x'].device)

    # Warmup
    for _ in range(warmup_runs):
        model.zero_grad()
        out, _ = model(
            batch_data['x'],
            batch_data['edge_index'],
            batch_data['batch'],
            topo_features=batch_data.get('topo_features'),
            pos=batch_data.get('pos')
        )
        loss = criterion(out, target)
        loss.backward()

    if batch_data['x'].is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        model.zero_grad()

        start = time.perf_counter()

        out, _ = model(
            batch_data['x'],
            batch_data['edge_index'],
            batch_data['batch'],
            topo_features=batch_data.get('topo_features'),
            pos=batch_data.get('pos')
        )
        loss = criterion(out, target)
        loss.backward()

        if batch_data['x'].is_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def benchmark_topo_extraction(
    num_graphs: int,
    avg_nodes: int,
    num_runs: int = 10
) -> Dict[str, float]:
    """Benchmark topological feature extraction time."""
    times = []

    for _ in range(num_runs):
        # Generate random graph
        n_nodes = max(3, int(np.random.normal(avg_nodes, avg_nodes * 0.2)))
        n_edges = n_nodes * 3

        edges = []
        for i in range(1, n_nodes):
            j = np.random.randint(0, i)
            edges.append((i, j))
        for _ in range(n_edges - n_nodes + 1):
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            if i != j:
                edges.append((i, j))

        edge_index = np.array(edges).T if edges else np.array([[], []])

        start = time.perf_counter()
        _ = compute_topological_features(edge_index, n_nodes, compute_edge_persistence=True)
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'per_graph_ms': np.mean(times) * 1000
    }


def run_runtime_analysis(
    device: str = 'cpu',
    output_dir: str = 'results'
) -> Dict[str, Any]:
    """Run comprehensive runtime analysis."""
    print("\n" + "="*60)
    print("RUNTIME ANALYSIS")
    print("="*60)

    results = {}

    # Model configurations
    models_config = {
        'GCN': lambda: GCN(in_channels=7, hidden_dim=64, num_classes=2),
        'GAT': lambda: GAT(in_channels=7, hidden_dim=64, num_classes=2),
        'EGNN': lambda: EGNN(in_channels=7, hidden_dim=64, num_classes=2),
        'PersLay': lambda: PersLay(in_channels=7, hidden_dim=64, num_classes=2, topo_dim=24),
        'TIEGNN': lambda: TIEGNN(in_channels=7, hidden_dim=64, num_classes=2, topo_dim=24,
                                 use_equiv=False, use_topo=True, use_interpretable=True),
    }

    # Graph size configurations
    size_configs = [
        {'num_graphs': 32, 'avg_nodes': 20, 'name': 'Small (20 nodes)'},
        {'num_graphs': 32, 'avg_nodes': 50, 'name': 'Medium (50 nodes)'},
        {'num_graphs': 32, 'avg_nodes': 100, 'name': 'Large (100 nodes)'},
    ]

    for size_cfg in size_configs:
        print(f"\n  Graph size: {size_cfg['name']}")
        print("-" * 50)

        # Create batch
        batch_data = create_synthetic_batch(
            num_graphs=size_cfg['num_graphs'],
            avg_nodes=size_cfg['avg_nodes'],
            avg_edges_per_node=3,
            num_features=7,
            device=device
        )

        size_results = {}

        for model_name, model_fn in models_config.items():
            print(f"    {model_name}...", end=" ")

            # Create model
            model = model_fn().to(device)
            num_params = sum(p.numel() for p in model.parameters())

            # Benchmark forward
            fwd_stats = benchmark_forward_pass(model, batch_data, num_runs=50, warmup_runs=5)

            # Benchmark backward
            bwd_stats = benchmark_backward_pass(model, batch_data, num_runs=30, warmup_runs=3)

            size_results[model_name] = {
                'forward': fwd_stats,
                'backward': bwd_stats,
                'total_ms': fwd_stats['mean_ms'] + bwd_stats['mean_ms'],
                'num_params': num_params
            }

            print(f"Forward: {fwd_stats['mean_ms']:.2f}ms, Backward: {bwd_stats['mean_ms']:.2f}ms")

            # Clean up
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        results[size_cfg['name']] = size_results

    # Topological feature extraction benchmark
    print("\n  Topological Feature Extraction:")
    print("-" * 50)

    topo_results = {}
    for avg_nodes in [20, 50, 100, 200]:
        topo_stats = benchmark_topo_extraction(num_graphs=1, avg_nodes=avg_nodes, num_runs=20)
        topo_results[f'{avg_nodes} nodes'] = topo_stats
        print(f"    {avg_nodes} nodes: {topo_stats['mean_ms']:.2f}ms per graph")

    results['topo_extraction'] = topo_results

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'runtime_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nRuntime analysis saved to {output_file}")

    return results


def print_runtime_table(results: Dict[str, Any]):
    """Print runtime results in a formatted table."""
    from tabulate import tabulate

    print("\n" + "="*80)
    print("RUNTIME COMPARISON (ms)")
    print("="*80)

    for size_name, size_results in results.items():
        if size_name == 'topo_extraction':
            continue

        print(f"\n{size_name}:")
        headers = ['Model', 'Forward', 'Backward', 'Total', 'Params']
        rows = []

        for model_name, stats in size_results.items():
            rows.append([
                model_name,
                f"{stats['forward']['mean_ms']:.2f} +/- {stats['forward']['std_ms']:.2f}",
                f"{stats['backward']['mean_ms']:.2f} +/- {stats['backward']['std_ms']:.2f}",
                f"{stats['total_ms']:.2f}",
                f"{stats['num_params']:,}"
            ])

        print(tabulate(rows, headers=headers, tablefmt='grid'))

    # Topological extraction table
    if 'topo_extraction' in results:
        print("\nTopological Feature Extraction:")
        headers = ['Graph Size', 'Time (ms)']
        rows = [[size, f"{stats['mean_ms']:.2f} +/- {stats['std_ms']:.2f}"]
               for size, stats in results['topo_extraction'].items()]
        print(tabulate(rows, headers=headers, tablefmt='grid'))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run runtime analysis')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    print(f"Running on device: {args.device}")

    results = run_runtime_analysis(
        device=args.device,
        output_dir=args.output_dir
    )

    print_runtime_table(results)


if __name__ == '__main__':
    main()
