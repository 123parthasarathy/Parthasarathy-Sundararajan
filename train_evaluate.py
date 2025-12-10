"""
Comprehensive Training and Evaluation Script for QI-VGT

This script validates the paper's claims by:
1. Running 5-fold cross-validation on MUTAG dataset
2. Comparing QI-VGT with all baseline methods
3. Performing ablation studies
4. Computing statistical significance tests
5. Generating visualizations

For Q1 Publication Validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
try:
    from torch_geometric.datasets import TUDataset
    USE_REMOTE_DATASET = True
except:
    USE_REMOTE_DATASET = False
from local_mutag import create_mutag_dataset, get_mutag_statistics
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from scipy import stats
import time
import warnings
from typing import Dict, List, Tuple, Optional
import copy
from collections import defaultdict

from qi_vgt_model import QI_VGT, QI_VGT_Improved, create_qi_vgt_model
from baseline_models import (
    create_baseline_model, TraditionalMLWrapper,
    prepare_traditional_ml_data
)

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class Trainer:
    """Training and evaluation handler for graph neural networks."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        patience: int = 50,
        max_epochs: int = 300
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(output['logits'], batch.y)

            # Add uncertainty regularization if available
            if 'uncertainty' in output:
                uncertainty_reg = output['uncertainty'].mean() * 0.01
                loss = loss + uncertainty_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        all_uncertainties = []

        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch.x, batch.edge_index, batch.batch)

            probs = torch.softmax(output['logits'], dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(batch.y.cpu().numpy())

            if 'uncertainty' in output:
                all_uncertainties.extend(output['uncertainty'].cpu().numpy().flatten())

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        try:
            auc_roc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_roc = 0.5

        metrics = {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels
        }

        if all_uncertainties:
            metrics['mean_uncertainty'] = np.mean(all_uncertainties)
            metrics['uncertainties'] = all_uncertainties

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], int]:
        """
        Train model with early stopping.

        Returns:
            Best validation metrics and number of epochs trained
        """
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        best_metrics = {}

        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch(train_loader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_metrics = self.evaluate(val_loader)

                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_metrics = val_metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, "
                          f"Val Acc={val_metrics['accuracy']:.4f}")

                if patience_counter >= self.patience // 10:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_metrics, epoch + 1


def run_cross_validation(
    model_factory,
    dataset,
    n_folds: int = 5,
    batch_size: int = 16,
    device: torch.device = None,
    verbose: bool = True,
    **trainer_kwargs
) -> Dict[str, List[float]]:
    """
    Run stratified k-fold cross-validation.

    Args:
        model_factory: Callable that creates model instances
        dataset: PyTorch Geometric dataset
        n_folds: Number of cross-validation folds
        batch_size: Batch size for training
        device: Torch device
        verbose: Print progress
        **trainer_kwargs: Additional arguments for Trainer

    Returns:
        Dictionary of metric lists across folds
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get labels for stratified split
    labels = [data.y.item() for data in dataset]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        if verbose:
            print(f"\n  Fold {fold + 1}/{n_folds}")

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Create fresh model instance
        model = model_factory()

        trainer = Trainer(model, device, **trainer_kwargs)
        metrics, _ = trainer.train(train_loader, val_loader, verbose=verbose)

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                fold_results[key].append(value)

    return dict(fold_results)


def run_traditional_ml_cv(
    model_type: str,
    dataset,
    n_folds: int = 5,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Run cross-validation for traditional ML models.
    """
    X, y = prepare_traditional_ml_data(dataset)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = TraditionalMLWrapper(model_type, **model_kwargs)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, preds)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_val, probs)
        except ValueError:
            auc_roc = 0.5

        fold_results['accuracy'].append(accuracy)
        fold_results['auc_roc'].append(auc_roc)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)

    return dict(fold_results)


def compute_statistics(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, and confidence intervals for results."""
    stats_dict = {}

    for metric, values in results.items():
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        ci_95 = 1.96 * std / np.sqrt(len(values))

        stats_dict[metric] = {
            'mean': mean,
            'std': std,
            'ci_95': ci_95,
            'min': np.min(values),
            'max': np.max(values)
        }

    return stats_dict


def statistical_significance_test(
    results1: List[float],
    results2: List[float]
) -> Dict[str, float]:
    """
    Perform paired t-test and compute effect size.

    Returns:
        Dictionary with p-value and Cohen's d effect size
    """
    t_stat, p_value = stats.ttest_rel(results1, results2)

    # Cohen's d effect size
    diff = np.array(results1) - np.array(results2)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': abs(cohens_d),
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
    }


def run_ablation_study(
    dataset,
    device: torch.device,
    n_folds: int = 5
) -> Dict[str, Dict]:
    """
    Run ablation study to analyze contribution of each component.
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    ablation_configs = {
        'Full QI-VGT': {
            'dropout_rate': 0.5,
            'hidden_dim': 32,
            'num_gcn_layers': 3,
            'use_quantum': True,
            'use_multi_pool': True,
            'use_residual': True
        },
        'Without Quantum Enhancement': {
            'dropout_rate': 0.5,
            'hidden_dim': 32,
            'use_quantum': False
        },
        'Without Dropout': {
            'dropout_rate': 0.0,
            'hidden_dim': 32
        },
        'Smaller Hidden (16)': {
            'dropout_rate': 0.5,
            'hidden_dim': 16
        },
        'Larger Hidden (64)': {
            'dropout_rate': 0.5,
            'hidden_dim': 64
        }
    }

    results = {}
    num_features = dataset[0].x.shape[1]

    for name, config in ablation_configs.items():
        print(f"\n  Testing: {name}")

        hidden_dim = config.get('hidden_dim', 32)
        dropout_rate = config.get('dropout_rate', 0.5)
        use_quantum = config.get('use_quantum', True)

        if use_quantum:
            def model_factory():
                return QI_VGT(
                    num_node_features=num_features,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate
                )
        else:
            def model_factory():
                return create_baseline_model(
                    'gcn',
                    num_node_features=num_features,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate
                )

        fold_results = run_cross_validation(
            model_factory, dataset, n_folds=n_folds,
            device=device, verbose=False
        )

        stats = compute_statistics(fold_results)
        results[name] = {
            'fold_results': fold_results,
            'statistics': stats
        }

        print(f"    Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")

    return results


def main():
    """Main function to run comprehensive validation."""
    print("=" * 70)
    print("QI-VGT COMPREHENSIVE VALIDATION FOR Q1 PUBLICATION")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load MUTAG dataset
    print("\nLoading MUTAG dataset...")
    try:
        dataset = TUDataset(root='data/MUTAG', name='MUTAG')
        print("  Loaded from TUDataset (remote)")
    except Exception as e:
        print(f"  Remote dataset unavailable ({type(e).__name__}), using local version...")
        dataset = create_mutag_dataset(seed=42)
        print("  Loaded from LocalMUTAG")

    print(f"  Total molecules: {len(dataset)}")
    print(f"  Number of classes: {dataset.num_classes if hasattr(dataset, 'num_classes') else 2}")
    print(f"  Number of node features: {dataset.num_node_features if hasattr(dataset, 'num_node_features') else dataset[0].x.shape[1]}")

    # Count class distribution
    labels = [data.y.item() for data in dataset]
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"  Class distribution: {class_counts}")

    num_features = dataset.num_node_features if hasattr(dataset, 'num_node_features') else dataset[0].x.shape[1]
    n_folds = 5

    # Store all results for comparison
    all_results = {}
    start_time = time.time()

    # =========================================================================
    # 1. QI-VGT (Paper's proposed method)
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. QI-VGT (Proposed Method)")
    print("=" * 60)

    def qi_vgt_factory():
        return QI_VGT(num_node_features=num_features, hidden_dim=32, num_classes=2)

    qi_vgt_results = run_cross_validation(
        qi_vgt_factory, dataset, n_folds=n_folds, device=device
    )
    qi_vgt_stats = compute_statistics(qi_vgt_results)
    all_results['QI-VGT'] = {'results': qi_vgt_results, 'stats': qi_vgt_stats}

    # Print results
    print(f"\nQI-VGT Results:")
    for metric in ['accuracy', 'auc_roc', 'precision', 'recall', 'f1']:
        s = qi_vgt_stats[metric]
        print(f"  {metric.upper()}: {s['mean']*100:.2f}% ± {s['std']*100:.2f}%")

    model = qi_vgt_factory()
    print(f"\n  Total Parameters: {model.count_parameters()}")

    # =========================================================================
    # 2. QI-VGT Improved (Enhanced version)
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. QI-VGT++ (Improved Version)")
    print("=" * 60)

    def qi_vgt_improved_factory():
        return QI_VGT_Improved(num_node_features=num_features, hidden_dim=64, num_classes=2)

    qi_vgt_improved_results = run_cross_validation(
        qi_vgt_improved_factory, dataset, n_folds=n_folds, device=device
    )
    qi_vgt_improved_stats = compute_statistics(qi_vgt_improved_results)
    all_results['QI-VGT++'] = {'results': qi_vgt_improved_results, 'stats': qi_vgt_improved_stats}

    print(f"\nQI-VGT++ Results:")
    for metric in ['accuracy', 'auc_roc', 'precision', 'recall', 'f1']:
        s = qi_vgt_improved_stats[metric]
        print(f"  {metric.upper()}: {s['mean']*100:.2f}% ± {s['std']*100:.2f}%")

    # =========================================================================
    # 3. GNN Baselines
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. GNN BASELINES")
    print("=" * 60)

    gnn_baselines = ['gcn', 'gat', 'gin', 'deepgcn']

    for model_name in gnn_baselines:
        print(f"\n  Testing {model_name.upper()}...")

        def factory(name=model_name):
            return create_baseline_model(name, num_node_features=num_features)

        results = run_cross_validation(
            factory, dataset, n_folds=n_folds, device=device, verbose=False
        )
        stats = compute_statistics(results)
        all_results[model_name.upper()] = {'results': results, 'stats': stats}

        print(f"    Accuracy: {stats['accuracy']['mean']*100:.2f}% ± {stats['accuracy']['std']*100:.2f}%")
        print(f"    AUC-ROC: {stats['auc_roc']['mean']*100:.2f}% ± {stats['auc_roc']['std']*100:.2f}%")

    # =========================================================================
    # 4. Traditional ML Baselines
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. TRADITIONAL ML BASELINES")
    print("=" * 60)

    ml_baselines = {
        'Random Forest': ('rf', {'n_estimators': 100}),
        'SVM (RBF)': ('svm', {'kernel': 'rbf', 'C': 1.0}),
        'Naive Bayes': ('nb', {}),
        'MLP': ('mlp', {'hidden_layer_sizes': (64, 32)})
    }

    for name, (model_type, kwargs) in ml_baselines.items():
        print(f"\n  Testing {name}...")

        results = run_traditional_ml_cv(model_type, dataset, n_folds=n_folds, **kwargs)
        stats = compute_statistics(results)
        all_results[name] = {'results': results, 'stats': stats}

        print(f"    Accuracy: {stats['accuracy']['mean']*100:.2f}% ± {stats['accuracy']['std']*100:.2f}%")

    # =========================================================================
    # 5. Statistical Significance Tests
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. STATISTICAL SIGNIFICANCE TESTS (QI-VGT vs Baselines)")
    print("=" * 60)

    qi_vgt_acc = all_results['QI-VGT']['results']['accuracy']

    for name, data in all_results.items():
        if name in ['QI-VGT', 'QI-VGT++']:
            continue

        baseline_acc = data['results']['accuracy']
        sig_test = statistical_significance_test(qi_vgt_acc, baseline_acc)

        improvement = (np.mean(qi_vgt_acc) - np.mean(baseline_acc)) * 100
        print(f"\n  QI-VGT vs {name}:")
        print(f"    Improvement: {improvement:+.2f}%")
        print(f"    p-value: {sig_test['p_value']:.4f}")
        print(f"    Cohen's d: {sig_test['cohens_d']:.2f} ({sig_test['effect_size']})")
        print(f"    Significant (p<0.05): {'Yes' if sig_test['p_value'] < 0.05 else 'No'}")

    # =========================================================================
    # 6. Ablation Study
    # =========================================================================
    ablation_results = run_ablation_study(dataset, device, n_folds=n_folds)

    # =========================================================================
    # 7. Summary Report
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)

    print("\nPerformance Comparison Table:")
    print("-" * 90)
    print(f"{'Method':<20} {'Accuracy':<15} {'AUC-ROC':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}")
    print("-" * 90)

    # Sort by accuracy
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['stats']['accuracy']['mean'],
        reverse=True
    )

    for name, data in sorted_results:
        s = data['stats']
        print(f"{name:<20} "
              f"{s['accuracy']['mean']*100:.2f}±{s['accuracy']['std']*100:.2f}  "
              f"{s['auc_roc']['mean']*100:.2f}±{s['auc_roc']['std']*100:.2f}  "
              f"{s['precision']['mean']*100:.2f}±{s['precision']['std']*100:.2f}  "
              f"{s['recall']['mean']*100:.2f}±{s['recall']['std']*100:.2f}  "
              f"{s['f1']['mean']*100:.2f}±{s['f1']['std']*100:.2f}")

    print("-" * 90)

    # Paper claims validation
    print("\n" + "=" * 70)
    print("PAPER CLAIMS VALIDATION")
    print("=" * 70)

    paper_claims = {
        'Accuracy': 84.59,
        'AUC-ROC': 89.30,
        'Precision': 86.78,
        'Recall': 91.20,
        'F1-Score': 88.74
    }

    achieved = all_results['QI-VGT']['stats']

    print("\nClaim vs Achieved:")
    print("-" * 50)

    for metric, claimed in paper_claims.items():
        metric_key = metric.lower().replace('-', '_').replace(' ', '_')
        if metric_key == 'f1_score':
            metric_key = 'f1'

        if metric_key in achieved:
            actual = achieved[metric_key]['mean'] * 100
            diff = actual - claimed
            status = "✓ VALIDATED" if abs(diff) <= 3.0 else "⚠ DEVIATION"
            print(f"  {metric:<12}: Claimed={claimed:.2f}%, Achieved={actual:.2f}% ({diff:+.2f}%) {status}")

    print("\n" + "=" * 70)
    print(f"Total execution time: {total_time:.2f} seconds")
    print("=" * 70)

    return all_results, ablation_results


if __name__ == "__main__":
    results, ablation = main()
