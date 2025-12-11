"""
Final Validation Script for QI-VGT-KAN
Tests on real MUTAG benchmark data

For Q1 Publication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.loader import DataLoader
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import models
from qi_vgt_kan_enhanced import QI_VGT_KAN, QI_VGT_KAN_Light
from real_tu_loader import parse_tu_dataset


class ModelWrapper(nn.Module):
    """Wrapper to unify model interfaces."""
    def __init__(self, model, model_type='new'):
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, data):
        if self.model_type == 'old':
            # Old QI-VGT interface
            out = self.model(data.x, data.edge_index, data.batch)
            if isinstance(out, dict):
                return out['logits']
            return out
        else:
            # New interface
            return self.model(data)


def train_model(model, train_loader, epochs, lr, device):
    """Train model and return best accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


def evaluate_model(model, loader, device):
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


def cross_validate_model(model_class, model_kwargs, dataset, n_folds=5, epochs=100, lr=0.001, device='cpu'):
    """Run cross-validation for a model."""
    labels = np.array([d.y.item() for d in dataset])
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)

        model = model_class(**model_kwargs).to(device)
        train_model(model, train_loader, epochs, lr, device)
        acc, auc = evaluate_model(model, test_loader, device)

        accuracies.append(acc * 100)
        aucs.append(auc)
        print(f"    Fold {fold+1}/{n_folds}: Acc={acc*100:.2f}%, AUC={auc:.4f}")

    return {
        'accuracy': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'auc': np.mean(aucs),
        'auc_std': np.std(aucs),
        'fold_accs': accuracies
    }


def main():
    print("=" * 70)
    print("QI-VGT-KAN FINAL VALIDATION - REAL MUTAG DATA")
    print("=" * 70)
    print("\nNovel Architecture Combining:")
    print("  - Kolmogorov-Arnold Network (KAN) - Nature Machine Intelligence 2025")
    print("  - Quantum-Inspired Phase Rotations")
    print("  - Virtual Node Global Attention")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load MUTAG
    print("\nLoading MUTAG dataset...")
    dataset = parse_tu_dataset('/tmp/MUTAG', 'MUTAG')

    input_dim = dataset[0].x.shape[1]
    labels = [d.y.item() for d in dataset]

    print(f"  Samples: {len(dataset)}")
    print(f"  Positive class: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Features: {input_dim}")

    results = {}

    # 1. QI-VGT-KAN (Full)
    print(f"\n[1] QI-VGT-KAN (Novel Proposed Model)")
    print("    Architecture: KAN + Quantum Phase + Virtual Node")
    results['QI-VGT-KAN'] = cross_validate_model(
        QI_VGT_KAN,
        {'input_dim': input_dim, 'hidden_dim': 64, 'output_dim': 2,
         'num_layers': 3, 'num_harmonics': 8, 'dropout': 0.3},
        dataset, n_folds=5, epochs=150, lr=0.001, device=device
    )
    print(f"    >>> RESULT: {results['QI-VGT-KAN']['accuracy']:.2f}% ± {results['QI-VGT-KAN']['accuracy_std']:.2f}%")
    print(f"    >>> AUC: {results['QI-VGT-KAN']['auc']:.4f} ± {results['QI-VGT-KAN']['auc_std']:.4f}")

    # 2. QI-VGT-KAN-Light
    print(f"\n[2] QI-VGT-KAN-Light (Efficient Version)")
    results['QI-VGT-KAN-Light'] = cross_validate_model(
        QI_VGT_KAN_Light,
        {'input_dim': input_dim, 'hidden_dim': 32, 'output_dim': 2,
         'num_layers': 2, 'dropout': 0.3},
        dataset, n_folds=5, epochs=150, lr=0.001, device=device
    )
    print(f"    >>> RESULT: {results['QI-VGT-KAN-Light']['accuracy']:.2f}% ± {results['QI-VGT-KAN-Light']['accuracy_std']:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY - MUTAG (REAL DATA)")
    print("=" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (method, res) in enumerate(sorted_results, 1):
        print(f"  {rank}. {method}: {res['accuracy']:.2f}% ± {res['accuracy_std']:.2f}% (AUC: {res['auc']:.4f})")

    # Comparison with literature
    print("\n" + "=" * 70)
    print("COMPARISON WITH STATE-OF-THE-ART")
    print("=" * 70)
    literature = {
        'KAGIN (2024)': 85.45,
        'GIN': 90.37,
        'WL Kernel': 90.4,
        'GCN': 85.93,
        'GAT': 83.70
    }

    best_result = sorted_results[0][1]['accuracy']
    print(f"\n  Our Best (QI-VGT-KAN): {best_result:.2f}%")
    print("\n  Literature Comparison:")
    for method, acc in sorted(literature.items(), key=lambda x: x[1], reverse=True):
        diff = best_result - acc
        status = "✓" if diff >= 0 else ""
        print(f"    {method}: {acc:.2f}% ({diff:+.2f}%) {status}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
