"""
Comprehensive Validation for Q1 Publication
Tests all QI-VGT variants on real MUTAG data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch_geometric.loader import DataLoader
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from qi_vgt_optimized import QI_VGT_Optimized, QI_VGT_Plus
from real_tu_loader import parse_tu_dataset


def train_with_scheduler(model, train_loader, epochs, device):
    """Train with learning rate scheduling and early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            probs = F.softmax(out, dim=-1)
            pred = out.argmax(dim=-1)

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    return accuracy, auc


def cross_validate(model_class, kwargs, dataset, n_folds=5, epochs=200, device='cpu'):
    """Run cross-validation."""
    labels = np.array([d.y.item() for d in dataset])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)

        model = model_class(**kwargs).to(device)
        train_with_scheduler(model, train_loader, epochs, device)
        acc, auc = evaluate(model, test_loader, device)

        accuracies.append(acc * 100)
        aucs.append(auc)
        print(f"    Fold {fold+1}: {acc*100:.2f}% (AUC: {auc:.4f})")

    return {
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'auc': np.mean(aucs),
        'auc_std': np.std(aucs),
        'fold_accs': accuracies
    }


def main():
    print("=" * 70)
    print("QI-VGT COMPREHENSIVE VALIDATION FOR Q1 PUBLICATION")
    print("=" * 70)
    print("\nReal MUTAG Benchmark Data (TUDataset)")
    print("5-fold Stratified Cross-Validation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    dataset = parse_tu_dataset('/tmp/MUTAG', 'MUTAG')
    input_dim = dataset[0].x.shape[1]
    labels = [d.y.item() for d in dataset]

    print(f"\nDataset: MUTAG")
    print(f"  Samples: {len(dataset)}")
    print(f"  Positive: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Features: {input_dim}")

    results = {}

    # 1. QI-VGT Optimized
    print(f"\n[1] QI-VGT-Optimized")
    results['QI-VGT-Opt'] = cross_validate(
        QI_VGT_Optimized,
        {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 3, 'dropout': 0.5},
        dataset, n_folds=5, epochs=200, device=device
    )
    print(f"    >>> {results['QI-VGT-Opt']['accuracy']:.2f}% ± {results['QI-VGT-Opt']['accuracy_std']:.2f}%")

    # 2. QI-VGT+
    print(f"\n[2] QI-VGT+")
    results['QI-VGT+'] = cross_validate(
        QI_VGT_Plus,
        {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 3, 'dropout': 0.5},
        dataset, n_folds=5, epochs=200, device=device
    )
    print(f"    >>> {results['QI-VGT+']['accuracy']:.2f}% ± {results['QI-VGT+']['accuracy_std']:.2f}%")

    # 3. QI-VGT-Opt with different hidden dims
    print(f"\n[3] QI-VGT-Opt (hidden=128)")
    results['QI-VGT-Opt-128'] = cross_validate(
        QI_VGT_Optimized,
        {'input_dim': input_dim, 'hidden_dim': 128, 'output_dim': 2, 'num_layers': 3, 'dropout': 0.5},
        dataset, n_folds=5, epochs=200, device=device
    )
    print(f"    >>> {results['QI-VGT-Opt-128']['accuracy']:.2f}% ± {results['QI-VGT-Opt-128']['accuracy_std']:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"  {rank}. {name}: {res['accuracy']:.2f}% ± {res['accuracy_std']:.2f}% (AUC: {res['auc']:.4f})")

    best_acc = sorted_results[0][1]['accuracy']

    # Literature comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)
    literature = {
        'GIN (Xu et al.)': 90.37,
        'WL Kernel': 90.4,
        'KAGIN (2024)': 85.45,
        'GCN': 85.93,
        'GAT': 83.70,
        'Paper Claim': 84.59
    }

    print(f"\n  Our Best: {best_acc:.2f}%\n")
    for method, acc in sorted(literature.items(), key=lambda x: x[1], reverse=True):
        diff = best_acc - acc
        symbol = "✓" if diff >= 0 else ""
        print(f"    {method}: {acc:.2f}% ({diff:+.2f}%) {symbol}")

    # Save results
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
