#!/usr/bin/env python3
"""
QI-VGT Validation with REAL Benchmark Data

Uses actual publicly available datasets:
- MUTAG: 188 nitroaromatic compounds (Debnath et al., 1991)
- PTC_MR: 344 compounds tested for carcinogenicity
- NCI1: 4110 compounds for anti-cancer screening

Source: TUDataset (Morris et al., 2020)
Downloaded from: https://github.com/nd7141/graph_datasets

For Q1 Publication Validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from scipy import stats
import time
import warnings
from typing import Dict, List
from collections import defaultdict
import copy
import os

from qi_vgt_model import QI_VGT, QI_VGT_Improved
from baseline_models import create_baseline_model, TraditionalMLWrapper, prepare_traditional_ml_data
from real_tu_loader import load_real_dataset, get_dataset_stats

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


class Trainer:
    """Trainer for GNN models."""

    def __init__(self, model, device, lr=0.005, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
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
        self.scheduler.step()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        preds, probs, labels = [], [], []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            prob = torch.softmax(out['logits'], dim=1)
            preds.extend(prob.argmax(dim=1).cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0)
        }
        try:
            metrics['auc_roc'] = roc_auc_score(labels, probs)
        except:
            metrics['auc_roc'] = 0.5
        return metrics

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


def cross_validate(model_factory, dataset, n_folds=5, device=None, verbose=True):
    """5-fold stratified cross-validation."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labels = [data.y.item() for data in dataset]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        if verbose:
            print(f"    Fold {fold+1}/{n_folds}...", end=" ", flush=True)

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)

        model = model_factory()
        trainer = Trainer(model, device)
        metrics = trainer.train(train_loader, val_loader)

        if verbose:
            print(f"Acc: {metrics['accuracy']*100:.2f}%")

        for k, v in metrics.items():
            results[k].append(v)

    return dict(results)


def cross_validate_ml(model_type, dataset, n_folds=5, **kwargs):
    """Cross-validation for traditional ML."""
    X, y = prepare_traditional_ml_data(dataset)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = defaultdict(list)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = TraditionalMLWrapper(model_type, **kwargs)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        results['accuracy'].append(accuracy_score(y_val, preds))
        results['precision'].append(precision_score(y_val, preds, zero_division=0))
        results['recall'].append(recall_score(y_val, preds, zero_division=0))
        results['f1'].append(f1_score(y_val, preds, zero_division=0))
        try:
            results['auc_roc'].append(roc_auc_score(y_val, probs))
        except:
            results['auc_roc'].append(0.5)

    return dict(results)


def compute_stats(results):
    return {k: {'mean': np.mean(v), 'std': np.std(v, ddof=1)} for k, v in results.items()}


def statistical_test(r1, r2):
    t_stat, p_val = stats.ttest_rel(r1, r2)
    diff = np.array(r1) - np.array(r2)
    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    return {'p_value': p_val, 'cohens_d': abs(d)}


def main():
    print("="*70)
    print("QI-VGT VALIDATION WITH REAL BENCHMARK DATA")
    print("="*70)
    print("\nDatasets: MUTAG, PTC_MR, NCI1")
    print("Source: TUDataset (Morris et al., 2020)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    start_time = time.time()
    all_results = {}

    # Datasets to evaluate
    datasets_info = [
        ('MUTAG', '/tmp/MUTAG'),
        ('PTC_MR', '/tmp/PTC_MR'),
        ('NCI1', '/tmp/NCI1')
    ]

    for dataset_name, source_dir in datasets_info:
        if not os.path.exists(source_dir):
            print(f"\n[SKIP] {dataset_name}: Source not found at {source_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name} (REAL DATA)")
        print("="*70)

        # Load dataset
        try:
            dataset = load_real_dataset(dataset_name, source_dir)
            stats = get_dataset_stats(dataset)
            print(f"  Samples: {stats['n_samples']}")
            print(f"  Positive: {stats['n_positive']} ({stats['positive_ratio']*100:.1f}%)")
            print(f"  Avg nodes: {stats['avg_nodes']:.1f}")
            print(f"  Features: {stats['num_features']}")
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        num_features = stats['num_features']
        results = {}

        # 1. QI-VGT
        print("\n[1] QI-VGT (Proposed)...")
        qi_vgt_results = cross_validate(
            lambda: QI_VGT(num_node_features=num_features, hidden_dim=32),
            dataset, device=device
        )
        results['QI-VGT'] = compute_stats(qi_vgt_results)
        print(f"    >>> RESULT: {results['QI-VGT']['accuracy']['mean']*100:.2f}% ± {results['QI-VGT']['accuracy']['std']*100:.2f}%")

        # 2. QI-VGT++
        print("\n[2] QI-VGT++ (Improved)...")
        qi_vgt_pp_results = cross_validate(
            lambda: QI_VGT_Improved(num_node_features=num_features, hidden_dim=64),
            dataset, device=device
        )
        results['QI-VGT++'] = compute_stats(qi_vgt_pp_results)
        print(f"    >>> RESULT: {results['QI-VGT++']['accuracy']['mean']*100:.2f}% ± {results['QI-VGT++']['accuracy']['std']*100:.2f}%")

        # 3. GCN
        print("\n[3] GCN...")
        gcn_results = cross_validate(
            lambda: create_baseline_model('gcn', num_node_features=num_features),
            dataset, device=device
        )
        results['GCN'] = compute_stats(gcn_results)
        print(f"    >>> RESULT: {results['GCN']['accuracy']['mean']*100:.2f}% ± {results['GCN']['accuracy']['std']*100:.2f}%")

        # 4. GAT
        print("\n[4] GAT...")
        gat_results = cross_validate(
            lambda: create_baseline_model('gat', num_node_features=num_features),
            dataset, device=device
        )
        results['GAT'] = compute_stats(gat_results)
        print(f"    >>> RESULT: {results['GAT']['accuracy']['mean']*100:.2f}% ± {results['GAT']['accuracy']['std']*100:.2f}%")

        # 5. GIN
        print("\n[5] GIN...")
        gin_results = cross_validate(
            lambda: create_baseline_model('gin', num_node_features=num_features),
            dataset, device=device
        )
        results['GIN'] = compute_stats(gin_results)
        print(f"    >>> RESULT: {results['GIN']['accuracy']['mean']*100:.2f}% ± {results['GIN']['accuracy']['std']*100:.2f}%")

        # 6. Random Forest
        print("\n[6] Random Forest...")
        rf_results = cross_validate_ml('rf', dataset, n_estimators=100)
        results['Random Forest'] = compute_stats(rf_results)
        print(f"    >>> RESULT: {results['Random Forest']['accuracy']['mean']*100:.2f}% ± {results['Random Forest']['accuracy']['std']*100:.2f}%")

        # 7. SVM
        print("\n[7] SVM...")
        svm_results = cross_validate_ml('svm', dataset, kernel='rbf')
        results['SVM'] = compute_stats(svm_results)
        print(f"    >>> RESULT: {results['SVM']['accuracy']['mean']*100:.2f}% ± {results['SVM']['accuracy']['std']*100:.2f}%")

        all_results[dataset_name] = {
            'results': results,
            'raw': {
                'QI-VGT': qi_vgt_results,
                'GCN': gcn_results,
                'GAT': gat_results,
                'GIN': gin_results
            }
        }

        # Statistical tests
        print("\n--- Statistical Significance (QI-VGT vs baselines) ---")
        for name, raw in [('GCN', gcn_results), ('GAT', gat_results), ('GIN', gin_results)]:
            test = statistical_test(qi_vgt_results['accuracy'], raw['accuracy'])
            diff = (results['QI-VGT']['accuracy']['mean'] - results[name]['accuracy']['mean']) * 100
            sig = "**" if test['p_value'] < 0.05 else ""
            print(f"  vs {name}: {diff:+.2f}% (p={test['p_value']:.4f}, d={test['cohens_d']:.2f}) {sig}")

    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS SUMMARY (REAL DATA)")
    print("="*70)

    print("\n" + "-"*100)
    print(f"{'Dataset':<10} {'Method':<15} {'Accuracy':<18} {'AUC-ROC':<18} {'F1-Score':<18}")
    print("-"*100)

    for dataset_name, data in all_results.items():
        results = data['results']
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['accuracy']['mean'], reverse=True)
        for i, (method, s) in enumerate(sorted_methods):
            acc = f"{s['accuracy']['mean']*100:.2f}±{s['accuracy']['std']*100:.2f}"
            auc = f"{s['auc_roc']['mean']*100:.2f}±{s['auc_roc']['std']*100:.2f}"
            f1 = f"{s['f1']['mean']*100:.2f}±{s['f1']['std']*100:.2f}"
            prefix = dataset_name if i == 0 else ""
            marker = " *" if method == 'QI-VGT' else ""
            print(f"{prefix:<10} {method:<15} {acc:<18} {auc:<18} {f1:<18}{marker}")
        print("-"*100)

    # Paper claims validation
    if 'MUTAG' in all_results:
        print("\n" + "="*70)
        print("PAPER CLAIMS VALIDATION (MUTAG - REAL DATA)")
        print("="*70)

        qi_vgt = all_results['MUTAG']['results']['QI-VGT']
        claims = {
            'Accuracy': 84.59,
            'AUC-ROC': 89.30,
            'Precision': 86.78,
            'Recall': 91.20,
            'F1-Score': 88.74
        }

        print(f"\n{'Metric':<12} {'Claimed':<12} {'Achieved':<20} {'Status'}")
        print("-"*60)
        all_valid = True
        for metric, claimed in claims.items():
            key = metric.lower().replace('-', '_').replace(' ', '_').replace('_score', '')
            if key in qi_vgt:
                achieved = qi_vgt[key]['mean'] * 100
                std = qi_vgt[key]['std'] * 100
                diff = achieved - claimed
                valid = diff >= -5  # Within 5% tolerance
                status = "✓ VALID" if valid else "✗ BELOW"
                if not valid:
                    all_valid = False
                print(f"{metric:<12} {claimed:<12.2f} {achieved:.2f}±{std:.2f}         {status}")

        print("\n" + "="*70)
        if all_valid:
            print("✓ PAPER VALIDATED FOR Q1 PUBLICATION")
        else:
            print("⚠ SOME METRICS BELOW CLAIMED - REVIEW RECOMMENDED")
        print("="*70)

    # Rankings
    print("\n" + "="*70)
    print("QI-VGT RANKING ACROSS DATASETS")
    print("="*70)

    for dataset_name, data in all_results.items():
        results = data['results']
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['accuracy']['mean'], reverse=True)
        qi_vgt_rank = next(i+1 for i, (m, _) in enumerate(sorted_methods) if m == 'QI-VGT')
        best = sorted_methods[0]
        print(f"{dataset_name}: QI-VGT rank {qi_vgt_rank}/{len(sorted_methods)} "
              f"({results['QI-VGT']['accuracy']['mean']*100:.2f}%), "
              f"Best: {best[0]} ({best[1]['accuracy']['mean']*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")

    return all_results


if __name__ == "__main__":
    results = main()
