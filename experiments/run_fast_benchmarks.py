#!/usr/bin/env python3
"""
Fast TIEGNN Benchmarks on Real Datasets

Runs optimized experiments with fewer epochs for quick validation.
Uses NCI1, DD, and IMDB-BINARY datasets with 5-fold CV.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GIN(nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, in_channels, hidden_dim, num_classes, num_layers=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        mlp = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp, train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(global_add_pool(x, batch))


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(global_mean_pool(x, batch))


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim // heads, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        return self.fc(global_mean_pool(x, batch))


class PersLay(nn.Module):
    def __init__(self, topo_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(topo_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, topo_features):
        return self.net(topo_features)


class TIEGNN(nn.Module):
    """TIEGNN with attention-based feature fusion"""
    def __init__(self, in_channels, hidden_dim, num_classes, topo_dim=24, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2), nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch, topo_features=None):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_repr = global_mean_pool(x, batch)

        if topo_features is not None:
            topo_repr = self.topo_encoder(topo_features)
            attn = self.fusion(torch.cat([graph_repr, topo_repr], dim=-1))
            fused = attn[:, 0:1] * graph_repr + attn[:, 1:2] * topo_repr
        else:
            fused = graph_repr

        return self.classifier(fused)


def load_dataset(name, data_dir):
    """Load TU dataset from text files"""
    path = os.path.join(data_dir, name)

    edges = []
    with open(os.path.join(path, f'{name}_A.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                edges.append([int(parts[0]) - 1, int(parts[1]) - 1])
    edges = np.array(edges).T

    graph_indicator = []
    with open(os.path.join(path, f'{name}_graph_indicator.txt'), 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip()) - 1)
    graph_indicator = np.array(graph_indicator)

    labels = []
    with open(os.path.join(path, f'{name}_graph_labels.txt'), 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    labels = np.array(labels)
    labels = labels - labels.min()

    # Node features
    node_labels_file = os.path.join(path, f'{name}_node_labels.txt')
    if os.path.exists(node_labels_file):
        node_labels = []
        with open(node_labels_file, 'r') as f:
            for line in f:
                node_labels.append(int(line.strip()))
        node_labels = np.array(node_labels)
        num_labels = node_labels.max() + 1
        node_features = np.eye(num_labels)[node_labels]
    else:
        # Degree as features for social network datasets
        num_nodes = len(graph_indicator)
        degrees = np.zeros(num_nodes)
        for i in range(edges.shape[1]):
            degrees[edges[0, i]] += 1
            degrees[edges[1, i]] += 1
        # Normalize and add constant
        degrees = degrees / (degrees.max() + 1e-6)
        node_features = np.column_stack([degrees, np.ones(num_nodes)])

    # Create graphs
    graphs = []
    for g_idx in range(labels.shape[0]):
        node_mask = graph_indicator == g_idx
        node_indices = np.where(node_mask)[0]

        if len(node_indices) == 0:
            continue

        global_to_local = {g: l for l, g in enumerate(node_indices)}
        edge_mask = np.isin(edges[0], node_indices) & np.isin(edges[1], node_indices)
        graph_edges = edges[:, edge_mask]

        if graph_edges.shape[1] == 0:
            local_edges = np.array([[0], [0]])
        else:
            local_edges = np.array([[global_to_local[e] for e in graph_edges[0]],
                                    [global_to_local[e] for e in graph_edges[1]]])

        x = torch.FloatTensor(node_features[node_mask])
        edge_index = torch.LongTensor(local_edges)
        y = torch.LongTensor([labels[g_idx]])
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs


def compute_topo_features(graphs):
    """Fast topological feature computation"""
    from scipy.spatial.distance import pdist, squareform

    features = []
    for i, data in enumerate(graphs):
        if i % 500 == 0:
            print(f"    Topo: {i}/{len(graphs)}")

        n = data.x.shape[0]
        if n < 2:
            features.append(torch.zeros(24))
            continue

        # Feature distance matrix
        X = data.x.numpy()
        if X.shape[1] > 1:
            dist = squareform(pdist(X, 'euclidean'))
        else:
            dist = squareform(pdist(X, 'cityblock'))

        if dist.max() > 0:
            dist = dist / dist.max()

        # Simple persistence features
        distances = dist[np.triu_indices(n, k=1)]
        if len(distances) == 0:
            features.append(torch.zeros(24))
            continue

        # Statistical features across thresholds
        feat = [
            np.mean(distances), np.std(distances), np.max(distances), np.min(distances),
            np.percentile(distances, 25), np.percentile(distances, 50), np.percentile(distances, 75),
            len(distances) / (n * (n-1) / 2),  # Edge density
        ]

        # Graph structure features
        edge_index = data.edge_index.numpy()
        num_edges = edge_index.shape[1]
        avg_degree = num_edges * 2 / n if n > 0 else 0

        feat.extend([
            n, num_edges, avg_degree, avg_degree / (n - 1) if n > 1 else 0,
            np.var(distances) if len(distances) > 1 else 0,
            np.sum(distances < 0.1), np.sum(distances < 0.5), np.sum(distances > 0.5)
        ])

        # Pad to 24
        while len(feat) < 24:
            feat.append(0)

        features.append(torch.FloatTensor(feat[:24]))

    return features


class TopoLoader(DataLoader):
    def __init__(self, dataset, topo_features, batch_size=32, shuffle=True, **kwargs):
        self.topo = torch.stack(topo_features)
        clean = []
        for i, d in enumerate(dataset):
            nd = Data(x=d.x, edge_index=d.edge_index, y=d.y)
            nd.idx = i
            clean.append(nd)
        super().__init__(clean, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            batch.topo_features = self.topo[batch.idx]
            yield batch


def run_experiment(name, graphs, topo_features, device, epochs=100, n_folds=5):
    """Run 5-fold CV experiment"""
    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({len(graphs)} graphs)")
    print(f"{'='*60}")

    labels = np.array([g.y.item() for g in graphs])
    num_classes = len(np.unique(labels))
    num_features = graphs[0].x.shape[1]
    print(f"  Features: {num_features}, Classes: {num_classes}")

    results = defaultdict(lambda: defaultdict(list))

    models = {
        'GCN': lambda: GCN(num_features, 64, num_classes),
        'GAT': lambda: GAT(num_features, 64, num_classes),
        'GIN': lambda: GIN(num_features, 64, num_classes),
        'PersLay': lambda: PersLay(24, 64, num_classes),
        'TIEGNN': lambda: TIEGNN(num_features, 64, num_classes),
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for model_name, model_fn in models.items():
        print(f"\n  {model_name}:", end=" ")
        use_topo = model_name in ['PersLay', 'TIEGNN']

        for fold, (train_idx, test_idx) in enumerate(skf.split(graphs, labels)):
            train_data = [graphs[i] for i in train_idx]
            test_data = [graphs[i] for i in test_idx]
            train_topo = [topo_features[i] for i in train_idx]
            test_topo = [topo_features[i] for i in test_idx]

            if use_topo:
                train_loader = TopoLoader(train_data, train_topo, batch_size=32)
                test_loader = TopoLoader(test_data, test_topo, batch_size=32, shuffle=False)
            else:
                train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

            model = model_fn().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

            best_acc = 0
            patience = 0

            for epoch in range(epochs):
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    if model_name == 'PersLay':
                        out = model(batch.topo_features)
                    elif model_name == 'TIEGNN':
                        out = model(batch.x, batch.edge_index, batch.batch, batch.topo_features)
                    else:
                        out = model(batch.x, batch.edge_index, batch.batch)

                    F.cross_entropy(out, batch.y).backward()
                    optimizer.step()

                # Evaluate
                model.eval()
                preds, true = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        if model_name == 'PersLay':
                            out = model(batch.topo_features)
                        elif model_name == 'TIEGNN':
                            out = model(batch.x, batch.edge_index, batch.batch, batch.topo_features)
                        else:
                            out = model(batch.x, batch.edge_index, batch.batch)
                        preds.extend(out.argmax(dim=-1).cpu().numpy())
                        true.extend(batch.y.cpu().numpy())

                acc = accuracy_score(true, preds)
                if acc > best_acc:
                    best_acc = acc
                    patience = 0
                else:
                    patience += 1

                if patience >= 20:
                    break

            results[model_name]['accuracy'].append(best_acc)
            print(f"{best_acc:.3f}", end=" ")

        mean = np.mean(results[model_name]['accuracy'])
        std = np.std(results[model_name]['accuracy'])
        print(f"-> {mean:.4f} +/- {std:.4f}")

    return {m: {'mean': np.mean(v['accuracy']), 'std': np.std(v['accuracy']), 'values': v['accuracy']}
            for m, v in results.items()}


def main():
    device = torch.device('cpu')
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    print("=" * 70)
    print("FAST TIEGNN BENCHMARK EXPERIMENTS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    for name in ['NCI1', 'DD', 'IMDB-BINARY']:
        try:
            print(f"\nLoading {name}...")
            graphs = load_dataset(name, data_dir)
            print(f"  Loaded {len(graphs)} graphs")

            print("  Computing topological features...")
            topo = compute_topo_features(graphs)

            results = run_experiment(name, graphs, topo, device, epochs=100, n_folds=5)
            all_results[name] = results

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Dataset':<15} {'GCN':<14} {'GAT':<14} {'GIN':<14} {'PersLay':<14} {'TIEGNN':<14}")
    print("-" * 85)

    for ds in all_results:
        row = f"{ds:<15}"
        best_acc = 0
        best_model = ""
        for m in ['GCN', 'GAT', 'GIN', 'PersLay', 'TIEGNN']:
            if m in all_results[ds]:
                acc = all_results[ds][m]['mean']
                if acc > best_acc:
                    best_acc = acc
                    best_model = m

        for m in ['GCN', 'GAT', 'GIN', 'PersLay', 'TIEGNN']:
            if m in all_results[ds]:
                acc = all_results[ds][m]['mean']
                std = all_results[ds][m]['std']
                marker = "**" if m == best_model else "  "
                row += f"{marker}{acc*100:.1f}+/-{std*100:.1f}{marker}  "
            else:
                row += "N/A           "
        print(row)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)

    json_results = {}
    for ds, models in all_results.items():
        json_results[ds] = {}
        for m, v in models.items():
            json_results[ds][m] = {'mean': float(v['mean']), 'std': float(v['std']),
                                   'values': [float(x) for x in v['values']]}

    with open(os.path.join(output_dir, 'fast_benchmark_results.json'), 'w') as f:
        json.dump({'results': json_results, 'timestamp': datetime.now().isoformat()}, f, indent=2)

    print(f"\nResults saved to {output_dir}/fast_benchmark_results.json")


if __name__ == '__main__':
    main()
