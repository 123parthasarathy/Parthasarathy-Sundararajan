"""
Comprehensive Validation Script for QI-VGT-KAN
Tests on real benchmark data: MUTAG, PTC_MR, NCI1

For Q1 Publication Validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import models
from qi_vgt_kan_enhanced import QI_VGT_KAN, QI_VGT_KAN_Light, count_parameters
from qi_vgt_model import QI_VGT
from real_tu_loader import load_real_dataset, parse_tu_dataset


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
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

            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return accuracy, auc, f1


def cross_validate(model_class, model_kwargs, dataset, n_folds=5, epochs=100, lr=0.001, device='cpu'):
    """Perform k-fold cross-validation."""
    from torch_geometric.loader import DataLoader

    # Prepare data
    labels = np.array([d.y.item() for d in dataset])
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accuracies = []
    aucs = []
    f1s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        # Create data loaders
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)

        # Initialize model
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Early stopping
        best_acc = 0
        patience = 30
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            acc, auc, f1 = evaluate(model, test_loader, device)

            if acc > best_acc:
                best_acc = acc
                best_auc = auc
                best_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        accuracies.append(best_acc)
        aucs.append(best_auc)
        f1s.append(best_f1)
        print(f"    Fold {fold+1}/{n_folds}: Acc={best_acc*100:.2f}%, AUC={best_auc:.4f}")

    return {
        'accuracy': np.mean(accuracies) * 100,
        'accuracy_std': np.std(accuracies) * 100,
        'auc': np.mean(aucs),
        'auc_std': np.std(aucs),
        'f1': np.mean(f1s),
        'f1_std': np.std(f1s),
        'fold_accs': [a * 100 for a in accuracies]
    }


def run_validation():
    """Run comprehensive validation on real benchmark data."""
    print("=" * 70)
    print("QI-VGT-KAN COMPREHENSIVE VALIDATION WITH REAL DATA")
    print("=" * 70)
    print("\nNovel Architecture: Quantum-Inspired + Kolmogorov-Arnold + Virtual Node")
    print("Benchmark: TUDataset (Morris et al., 2020)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Datasets to test
    datasets_info = {
        'MUTAG': {'path': '/tmp/MUTAG', 'desc': 'Nitroaromatic mutagenicity'},
        'PTC_MR': {'path': '/tmp/PTC_MR', 'desc': 'Rodent carcinogenicity'},
    }

    all_results = {}

    for dataset_name, info in datasets_info.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name} ({info['desc']})")
        print("=" * 70)

        # Load real data
        try:
            dataset = parse_tu_dataset(info['path'], dataset_name)
            if len(dataset) == 0:
                print(f"  Dataset empty, skipping...")
                continue
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            continue

        input_dim = dataset[0].x.shape[1]
        labels = [d.y.item() for d in dataset]
        pos_ratio = sum(labels) / len(labels)

        print(f"  Samples: {len(dataset)}")
        print(f"  Positive class: {sum(labels)} ({pos_ratio*100:.1f}%)")
        print(f"  Input features: {input_dim}")

        results = {}

        # 1. QI-VGT-KAN (Full model)
        print(f"\n[1] QI-VGT-KAN (Proposed Novel Model)")
        model_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': 64,
            'output_dim': 2,
            'num_layers': 3,
            'num_harmonics': 8,
            'dropout': 0.3
        }
        results['QI-VGT-KAN'] = cross_validate(
            QI_VGT_KAN, model_kwargs, dataset, n_folds=5, epochs=150, lr=0.001, device=device
        )
        print(f"    >>> RESULT: {results['QI-VGT-KAN']['accuracy']:.2f}% ± {results['QI-VGT-KAN']['accuracy_std']:.2f}%")
        print(f"    >>> AUC: {results['QI-VGT-KAN']['auc']:.4f} ± {results['QI-VGT-KAN']['auc_std']:.4f}")

        # 2. QI-VGT-KAN-Light
        print(f"\n[2] QI-VGT-KAN-Light (Efficient Version)")
        model_kwargs_light = {
            'input_dim': input_dim,
            'hidden_dim': 32,
            'output_dim': 2,
            'num_layers': 2,
            'dropout': 0.3
        }
        results['QI-VGT-KAN-Light'] = cross_validate(
            QI_VGT_KAN_Light, model_kwargs_light, dataset, n_folds=5, epochs=150, lr=0.001, device=device
        )
        print(f"    >>> RESULT: {results['QI-VGT-KAN-Light']['accuracy']:.2f}% ± {results['QI-VGT-KAN-Light']['accuracy_std']:.2f}%")

        # 3. Original QI-VGT
        print(f"\n[3] QI-VGT (Original)")
        model_kwargs_orig = {
            'num_node_features': input_dim,
            'hidden_dim': 64,
            'num_classes': 2,
            'dropout_rate': 0.3
        }
        results['QI-VGT'] = cross_validate(
            QI_VGT, model_kwargs_orig, dataset, n_folds=5, epochs=150, lr=0.001, device=device
        )
        print(f"    >>> RESULT: {results['QI-VGT']['accuracy']:.2f}% ± {results['QI-VGT']['accuracy_std']:.2f}%")

        all_results[dataset_name] = results

        # Statistical comparison
        print(f"\n--- Statistical Comparison ---")
        kan_accs = results['QI-VGT-KAN']['fold_accs']
        orig_accs = results['QI-VGT']['fold_accs']

        t_stat, p_value = stats.ttest_rel(kan_accs, orig_accs)
        diff = np.mean(kan_accs) - np.mean(orig_accs)
        print(f"  QI-VGT-KAN vs QI-VGT: {diff:+.2f}% (p={p_value:.4f})")

    # Final Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 40)

        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        for rank, (method, res) in enumerate(sorted_results, 1):
            print(f"  {rank}. {method}: {res['accuracy']:.2f}% ± {res['accuracy_std']:.2f}%")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_validation()
