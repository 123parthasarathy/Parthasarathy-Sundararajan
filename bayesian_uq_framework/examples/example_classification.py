"""
Complete Example: Bayesian UQ for Classification with Large-Scale Data

This example demonstrates:
1. Loading large-scale classification datasets
2. Training Bayesian classifiers
3. Uncertainty quantification for classification
4. Out-of-distribution detection
5. Calibration analysis

Dataset sources:
- MNIST, CIFAR-10/100 (via torchvision)
- UCI datasets (Credit Card Fraud, Adult Income, etc.)
- OpenML classification datasets
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
sys.path.append('..')

from bayesian_uq_framework.models import MCDropoutNN
from bayesian_uq_framework.uncertainty import UncertaintyMetrics, CalibrationMetrics
from bayesian_uq_framework.manifold import ManifoldAnalyzer
from bayesian_uq_framework.reliability import OutOfDistributionDetector
from bayesian_uq_framework.utils import CalibrationVisualizer, ManifoldVisualizer


# ============================================================================
# DATASET DOWNLOAD PATHS
# ============================================================================

CLASSIFICATION_DATASETS = {
    # Torchvision datasets (automatic download)
    'mnist': 'torchvision.datasets.MNIST',
    'cifar10': 'torchvision.datasets.CIFAR10',
    'cifar100': 'torchvision.datasets.CIFAR100',
    'fashion_mnist': 'torchvision.datasets.FashionMNIST',

    # UCI datasets
    'credit_card_fraud': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
    'adult_income': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    'covertype': 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',

    # Kaggle (via API)
    'titanic': 'kaggle competitions download -c titanic',
    'santander': 'kaggle competitions download -c santander-customer-transaction-prediction',

    # OpenML
    'openml': 'Use sklearn.datasets.fetch_openml with dataset IDs',
}


def load_mnist(flatten=True):
    """
    Load MNIST dataset.

    Args:
        flatten: If True, flatten images to vectors

    Returns:
        X_train, y_train, X_test, y_test
    """
    from torchvision import datasets, transforms

    print("Loading MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    X_train = train_dataset.data.numpy() / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy() / 255.0
    y_test = test_dataset.targets.numpy()

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    print(f"Dataset shape: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_openml_classification(dataset_id=40996):
    """
    Load classification dataset from OpenML.

    Example dataset IDs:
    - 40996: Australian (classification, 690 samples)
    - 1590: Adult (classification, 48k samples)
    - 40498: Wine Quality (classification, 6.5k samples)
    - 1464: Blood Transfusion (classification, 748 samples)

    Returns:
        X_train, y_train, X_test, y_test
    """
    from sklearn.datasets import fetch_openml

    print(f"Loading OpenML dataset {dataset_id}...")
    data = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')

    X = data.data
    y = data.target

    # Encode labels if categorical
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset shape: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X_train, y_train, X_test, y_test


def generate_synthetic_classification(n_samples=10000, n_features=20, n_classes=5):
    """
    Generate synthetic classification data.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes

    Returns:
        X_train, y_train, X_test, y_test
    """
    from sklearn.datasets import make_classification

    print(f"Generating synthetic classification data...")
    print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test


class BayesianClassifier(nn.Module):
    """Bayesian Neural Network for Classification."""

    def __init__(self, input_dim, hidden_dims, n_classes, dropout_rate=0.2):
        super(BayesianClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def mc_predict_proba(self, x, n_samples=100):
        """
        MC sampling for predictive distribution.

        Returns:
            mean_probs, entropy, mutual_information
        """
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        predictions = torch.stack(predictions)  # (n_samples, batch_size, n_classes)

        # Mean predictions
        mean_probs = predictions.mean(dim=0)

        # Epistemic uncertainty (mutual information)
        epistemic = UncertaintyMetrics.mutual_information(predictions)

        # Total uncertainty (predictive entropy)
        total_entropy = UncertaintyMetrics.predictive_entropy(predictions)

        return mean_probs, total_entropy, epistemic


def main():
    """Complete Bayesian UQ for Classification."""

    print("="*80)
    print("BAYESIAN UQ FOR CLASSIFICATION")
    print("="*80)

    # ========================================
    # 1. Load Data
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)

    # Load dataset
    X_train, y_train, X_test, y_test = load_mnist(flatten=True)
    # Alternative: Use OpenML
    # X_train, y_train, X_test, y_test = load_openml_classification(dataset_id=40996)

    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ========================================
    # 2. Train Bayesian Classifier
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: Training Bayesian Classifier")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)

    # Create model
    model = BayesianClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64],
        n_classes=n_classes,
        dropout_rate=0.3
    ).to(device)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 20
    batch_size = 256

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    print("Training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        acc = 100. * correct / total
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")

    # ========================================
    # 3. Uncertainty Quantification
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: Uncertainty Quantification")
    print("="*80)

    print("Computing predictions with uncertainty...")
    mean_probs, total_entropy, epistemic = model.mc_predict_proba(
        X_test_t, n_samples=100
    )

    mean_probs = mean_probs.cpu().numpy()
    total_entropy = total_entropy.cpu().numpy()
    epistemic = epistemic.cpu().numpy()

    # Get predictions
    predicted_classes = mean_probs.argmax(axis=1)
    max_probs = mean_probs.max(axis=1)

    # Accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Uncertainty statistics
    print(f"\nUncertainty Statistics:")
    print(f"  Mean Total Entropy: {total_entropy.mean():.4f}")
    print(f"  Mean Epistemic Uncertainty: {epistemic.mean():.4f}")
    print(f"  Mean Confidence: {max_probs.mean():.4f}")

    # ========================================
    # 4. Calibration Analysis
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: Calibration Analysis")
    print("="*80)

    # Compute calibration metrics
    correctness = (predicted_classes == y_test).astype(float)

    ece = CalibrationMetrics.expected_calibration_error(
        max_probs, correctness, n_bins=10
    )
    mce = CalibrationMetrics.maximum_calibration_error(
        max_probs, correctness, n_bins=10
    )

    print(f"\nCalibration Metrics:")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Maximum Calibration Error (MCE): {mce:.4f}")

    # One-hot encode for Brier score
    y_test_onehot = np.zeros((len(y_test), n_classes))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1

    brier = CalibrationMetrics.brier_score(mean_probs, y_test_onehot)
    print(f"  Brier Score: {brier:.4f}")

    # ========================================
    # 5. Out-of-Distribution Detection
    # ========================================
    print("\n" + "="*80)
    print("STEP 5: Out-of-Distribution Detection")
    print("="*80)

    # Create OOD samples (noise)
    print("Creating OOD samples...")
    X_ood = np.random.randn(*X_test[:1000].shape)
    X_ood_t = torch.FloatTensor(X_ood).to(device)

    # Get uncertainties for OOD
    _, ood_entropy, ood_epistemic = model.mc_predict_proba(
        X_ood_t, n_samples=100
    )
    ood_entropy = ood_entropy.cpu().numpy()
    ood_epistemic = ood_epistemic.cpu().numpy()

    print(f"\nOOD Uncertainty:")
    print(f"  In-Distribution Entropy: {total_entropy[:1000].mean():.4f}")
    print(f"  OOD Entropy: {ood_entropy.mean():.4f}")
    print(f"  In-Distribution Epistemic: {epistemic[:1000].mean():.4f}")
    print(f"  OOD Epistemic: {ood_epistemic.mean():.4f}")

    # ========================================
    # 6. Manifold Analysis
    # ========================================
    print("\n" + "="*80)
    print("STEP 6: Manifold Analysis")
    print("="*80)

    # Subsample for visualization
    n_viz = 5000
    indices = np.random.choice(len(X_test), n_viz, replace=False)

    print(f"Computing manifold embedding for {n_viz} samples...")
    manifold = ManifoldAnalyzer(method='umap', random_state=42)
    X_embedded = manifold.fit_transform(
        X_test[indices],
        n_components=2,
        n_neighbors=15
    )

    # ========================================
    # 7. Visualization
    # ========================================
    print("\n" + "="*80)
    print("STEP 7: Creating Visualizations")
    print("="*80)

    # Calibration curve
    fig, ax = plt.subplots(figsize=(10, 6))
    CalibrationVisualizer.plot_calibration_curve(
        max_probs, correctness, n_bins=10, ax=ax
    )
    plt.savefig('classification_calibration.png', dpi=300, bbox_inches='tight')
    print("Saved: classification_calibration.png")

    # Manifold colored by uncertainty
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Colored by class
    ManifoldVisualizer.plot_manifold_2d(
        X_embedded,
        labels=y_test[indices],
        title='Manifold: True Classes',
        ax=axes[0, 0]
    )

    # 2. Colored by entropy
    ManifoldVisualizer.plot_manifold_2d(
        X_embedded,
        colors=total_entropy[indices],
        title='Manifold: Total Uncertainty (Entropy)',
        colorbar_label='Entropy',
        ax=axes[0, 1]
    )

    # 3. Colored by epistemic uncertainty
    ManifoldVisualizer.plot_manifold_2d(
        X_embedded,
        colors=epistemic[indices],
        title='Manifold: Epistemic Uncertainty',
        colorbar_label='Mutual Information',
        ax=axes[1, 0]
    )

    # 4. Colored by correctness
    ManifoldVisualizer.plot_manifold_2d(
        X_embedded,
        colors=correctness[indices],
        title='Manifold: Prediction Correctness',
        colorbar_label='Correct (1) / Wrong (0)',
        ax=axes[1, 1]
    )

    plt.tight_layout()
    plt.savefig('classification_manifold_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: classification_manifold_analysis.png")

    # Uncertainty distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # In-distribution vs OOD
    axes[0].hist(total_entropy[:1000], bins=50, alpha=0.6, label='In-Distribution', density=True)
    axes[0].hist(ood_entropy, bins=50, alpha=0.6, label='OOD', density=True)
    axes[0].set_xlabel('Total Uncertainty (Entropy)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('In-Distribution vs OOD Uncertainty')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Correct vs Incorrect
    axes[1].hist(total_entropy[correctness == 1], bins=50, alpha=0.6,
                label='Correct', density=True)
    axes[1].hist(total_entropy[correctness == 0], bins=50, alpha=0.6,
                label='Incorrect', density=True)
    axes[1].set_xlabel('Total Uncertainty (Entropy)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Uncertainty: Correct vs Incorrect Predictions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('classification_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: classification_uncertainty_analysis.png")

    print("\n" + "="*80)
    print("CLASSIFICATION ANALYSIS COMPLETE!")
    print("="*80)

    # Save results
    results = {
        'predictions': predicted_classes,
        'probabilities': mean_probs,
        'total_uncertainty': total_entropy,
        'epistemic_uncertainty': epistemic,
        'true_labels': y_test,
        'manifold_embedding': X_embedded,
        'embedded_indices': indices,
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier
    }

    np.savez('classification_results.npz', **results)
    print("\nResults saved to: classification_results.npz")

    return results


if __name__ == '__main__':
    results = main()
    plt.show()
