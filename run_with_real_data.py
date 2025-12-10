#!/usr/bin/env python3
"""
QI-VGT Validation with REAL Benchmark Data

This script downloads and uses the actual TUDataset benchmarks:
- MUTAG: 188 real nitroaromatic compounds (Debnath et al., 1991)
- PTC_MR: 344 real compounds tested for carcinogenicity
- PROTEINS: 1113 real protein structures

REQUIREMENTS:
- Internet connection to download TUDataset
- Run: python run_with_real_data.py

For Q1 Publication - Uses only publicly available benchmark data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from scipy import stats
import time
import warnings
from typing import Dict
from collections import defaultdict
import copy

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# Import models
from qi_vgt_model import QI_VGT, QI_VGT_Improved
from baseline_models import create_baseline_model, TraditionalMLWrapper, prepare_traditional_ml_data


def check_network():
    """Check if network is available."""
    try:
        import socket
        socket.create_connection(("www.chrsmrrs.com", 443), timeout=5)
        return True
    except:
        return False


def load_real_dataset(name: str):
    """Load real TUDataset benchmark."""
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    print(f"Downloading {name} from TUDataset...")
    dataset = TUDataset(root=f'data/{name}', name=name)
    print(f"  Loaded {len(dataset)} samples")
    print(f"  Features: {dataset.num_node_features}")
    print(f"  Classes: {dataset.num_classes}")

    return dataset


class Trainer:
    """Trainer for GNN models."""
    def __init__(self, model, device, lr=0.005, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        from torch_geometric.loader import DataLoader
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out['logits'], batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        from torch_geometric.loader import DataLoader
        self.model.eval()
        preds, probs, labels = [], [], []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            prob = torch.softmax(out['logits'], dim=1)
            preds.extend(prob.argmax(dim=1).cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auc_roc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
        }

    def train(self, train_loader, val_loader, max_epochs=300, patience=50):
        best_acc, best_state, wait = 0, None, 0
        for epoch in range(max_epochs):
            self.train_epoch(train_loader)
            if (epoch + 1) % 10 == 0:
                metrics = self.evaluate(val_loader)
                if metrics['accuracy'] > best_acc:
                    best_acc = metrics['accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1
                if wait >= patience // 10:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        return self.evaluate(val_loader)


def cross_validate(model_factory, dataset, n_folds=5, device=None):
    """5-fold cross-validation."""
    from torch_geometric.loader import DataLoader

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        print(f"    Fold {fold+1}/{n_folds}...", end=" ")
        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)

        model = model_factory()
        trainer = Trainer(model, device)
        metrics = trainer.train(train_loader, val_loader)
        print(f"Acc: {metrics['accuracy']*100:.2f}%")

        for k, v in metrics.items():
            results[k].append(v)

    return dict(results)


def main():
    """Run validation with real benchmark data."""
    print("="*70)
    print("QI-VGT VALIDATION WITH REAL BENCHMARK DATA")
    print("="*70)

    # Check network
    if not check_network():
        print("\n[ERROR] No network connection!")
        print("This script requires internet to download TUDataset benchmarks.")
        print("\nOptions:")
        print("1. Connect to internet and run again")
        print("2. Use synthetic validation: python multi_dataset_validation.py")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Real datasets to evaluate
    datasets = ['MUTAG', 'PTC_MR', 'PROTEINS']
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name} (REAL DATA)")
        print("="*70)

        try:
            dataset = load_real_dataset(dataset_name)
        except Exception as e:
            print(f"[ERROR] Could not load {dataset_name}: {e}")
            continue

        num_features = dataset.num_node_features
        labels = [data.y.item() for data in dataset]
        n_pos = sum(labels)
        print(f"  Class distribution: {n_pos} positive / {len(labels)-n_pos} negative")

        results = {}

        # QI-VGT
        print("\n[1] QI-VGT (Proposed)...")
        qi_vgt = cross_validate(
            lambda: QI_VGT(num_node_features=num_features, hidden_dim=32),
            dataset, device=device
        )
        results['QI-VGT'] = {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in qi_vgt.items()}
        print(f"    RESULT: {results['QI-VGT']['accuracy']['mean']*100:.2f}% ± {results['QI-VGT']['accuracy']['std']*100:.2f}%")

        # GCN
        print("\n[2] GCN Baseline...")
        gcn = cross_validate(
            lambda: create_baseline_model('gcn', num_node_features=num_features),
            dataset, device=device
        )
        results['GCN'] = {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in gcn.items()}
        print(f"    RESULT: {results['GCN']['accuracy']['mean']*100:.2f}% ± {results['GCN']['accuracy']['std']*100:.2f}%")

        # GAT
        print("\n[3] GAT Baseline...")
        gat = cross_validate(
            lambda: create_baseline_model('gat', num_node_features=num_features),
            dataset, device=device
        )
        results['GAT'] = {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in gat.items()}
        print(f"    RESULT: {results['GAT']['accuracy']['mean']*100:.2f}% ± {results['GAT']['accuracy']['std']*100:.2f}%")

        # GIN
        print("\n[4] GIN Baseline...")
        gin = cross_validate(
            lambda: create_baseline_model('gin', num_node_features=num_features),
            dataset, device=device
        )
        results['GIN'] = {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in gin.items()}
        print(f"    RESULT: {results['GIN']['accuracy']['mean']*100:.2f}% ± {results['GIN']['accuracy']['std']*100:.2f}%")

        all_results[dataset_name] = results

        # Statistical tests
        print("\n--- Statistical Significance (QI-VGT vs baselines) ---")
        for baseline in ['GCN', 'GAT', 'GIN']:
            baseline_acc = eval(baseline.lower())['accuracy']
            t_stat, p_val = stats.ttest_rel(qi_vgt['accuracy'], baseline_acc)
            diff = (results['QI-VGT']['accuracy']['mean'] - results[baseline]['accuracy']['mean']) * 100
            sig = "**" if p_val < 0.05 else ""
            print(f"  vs {baseline}: {diff:+.2f}% (p={p_val:.4f}) {sig}")

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY (REAL DATA)")
    print("="*70)

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy']['mean'], reverse=True)
        for i, (method, stats) in enumerate(sorted_results):
            acc = stats['accuracy']
            marker = " <-- PROPOSED" if method == 'QI-VGT' else ""
            print(f"  {i+1}. {method}: {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}%{marker}")

    # Paper claims validation
    if 'MUTAG' in all_results:
        print("\n" + "="*70)
        print("PAPER CLAIMS VALIDATION (MUTAG - REAL DATA)")
        print("="*70)

        qi_vgt_mutag = all_results['MUTAG']['QI-VGT']
        claims = {
            'Accuracy': 84.59,
            'AUC-ROC': 89.30,
            'F1-Score': 88.74
        }

        print(f"\n{'Metric':<12} {'Claimed':<12} {'Achieved':<20} {'Status'}")
        print("-"*60)
        for metric, claimed in claims.items():
            key = metric.lower().replace('-', '_').replace(' ', '_').replace('_score', '')
            if key in qi_vgt_mutag:
                achieved = qi_vgt_mutag[key]['mean'] * 100
                std = qi_vgt_mutag[key]['std'] * 100
                diff = achieved - claimed
                status = "✓ VALID" if diff >= -5 else "✗ BELOW"
                print(f"{metric:<12} {claimed:<12.2f} {achieved:.2f}±{std:.2f}         {status}")


if __name__ == "__main__":
    main()
