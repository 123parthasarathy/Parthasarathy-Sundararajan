"""
Complete Example: Bayesian UQ for Regression with Large-Scale Data

This example demonstrates:
1. Loading large-scale regression datasets
2. Training Bayesian neural networks
3. Uncertainty quantification
4. Manifold-based reliability assessment
5. Comprehensive visualization

Dataset sources (publicly available):
- UCI ML Repository datasets
- OpenML datasets
- Kaggle datasets
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')

from bayesian_uq_framework.models import MCDropoutNN, DeepEnsemble, VariationalNN
from bayesian_uq_framework.uncertainty import UncertaintyMetrics, RegressionUncertaintyMetrics
from bayesian_uq_framework.manifold import ManifoldAnalyzer, ManifoldBasedReliabilityRegions
from bayesian_uq_framework.reliability import ReliabilityScorer, ReliabilityDiagnostics
from bayesian_uq_framework.utils import (
    UncertaintyVisualizer,
    ManifoldVisualizer,
    ReliabilityVisualizer
)


# ============================================================================
# DATASET DOWNLOAD PATHS (Big Data Sources)
# ============================================================================

DATASET_URLS = {
    # UCI Machine Learning Repository
    'california_housing': 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
    'concrete_strength': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
    'energy_efficiency': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
    'power_plant': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',

    # OpenML (accessed via sklearn or direct download)
    'openml_datasets': 'https://www.openml.org/search?type=data',

    # Large-scale datasets
    'year_prediction_msd': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
    'protein_structure': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',

    # Kaggle datasets (requires kaggle API)
    'house_prices': 'kaggle datasets download -d c/house-prices-advanced-regression-techniques',
    'bike_sharing': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip',
}


def load_california_housing():
    """
    Load California Housing dataset (sklearn built-in).

    Returns:
        X, y, feature_names
    """
    from sklearn.datasets import fetch_california_housing

    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    return X, y, feature_names


def load_openml_dataset(dataset_id=42165):
    """
    Load dataset from OpenML.

    Example dataset IDs:
    - 42165: diamonds (regression, ~54k samples)
    - 287: wine_quality (regression, 6.5k samples)
    - 41514: vehicle_sensor (regression, 100k samples)
    - 42571: Moneyball (regression, 1.2k samples)

    Returns:
        X, y
    """
    from sklearn.datasets import fetch_openml

    print(f"Loading OpenML dataset {dataset_id}...")
    data = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')
    X = data.data
    y = data.target

    # Convert to float if needed
    if y.dtype == object:
        y = y.astype(float)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    return X, y


def download_and_load_concrete_strength():
    """
    Download and load Concrete Compressive Strength dataset.

    Direct download link:
    https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls
    """
    import pandas as pd
    import urllib.request

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    filename = 'Concrete_Data.xls'

    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        df = pd.read_excel(filename)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Using synthetic data instead...")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples=10000, n_features=20, noise=0.1):
    """
    Generate synthetic regression data for demonstration.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level

    Returns:
        X, y
    """
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features...")

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # Non-linear relationship with heteroscedastic noise
    true_function = (
        np.sin(X[:, 0]) +
        0.5 * X[:, 1]**2 +
        0.3 * X[:, 2] * X[:, 3] +
        0.1 * np.sum(X[:, 4:10], axis=1)
    )

    # Add heteroscedastic noise (noise increases with |x|)
    noise_scale = noise * (1 + 0.5 * np.abs(X[:, 0]))
    y = true_function + np.random.randn(n_samples) * noise_scale

    return X, y


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Complete Bayesian UQ pipeline demonstration."""

    print("="*80)
    print("BAYESIAN UNCERTAINTY QUANTIFICATION FRAMEWORK")
    print("Manifold-Based Reliability Assessment")
    print("="*80)

    # ========================================
    # 1. Load Data
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)

    # Option 1: Use built-in dataset
    X, y, feature_names = load_california_housing()

    # Option 2: Use OpenML dataset
    # X, y = load_openml_dataset(dataset_id=42165)

    # Option 3: Use synthetic data
    # X, y = generate_synthetic_data(n_samples=10000)

    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # ========================================
    # 2. Train Bayesian Models
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: Training Bayesian Neural Networks")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Model 1: MC Dropout
    print("\nTraining MC Dropout model...")
    mc_model = MCDropoutNN(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        output_dim=1,
        dropout_rate=0.2
    ).to(device)

    optimizer = torch.optim.Adam(mc_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    n_epochs = 50
    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(n_epochs):
        mc_model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = mc_model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Model 2: Deep Ensemble (simplified version)
    print("\nTraining Deep Ensemble...")
    ensemble = DeepEnsemble(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        output_dim=1,
        n_models=5
    )

    ensemble.train_ensemble(train_loader, n_epochs=30, device=device)

    # ========================================
    # 3. Make Predictions with Uncertainty
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: Uncertainty Quantification")
    print("="*80)

    # MC Dropout predictions
    print("\nComputing MC Dropout predictions...")
    mc_mean, mc_std = mc_model.mc_predict(X_test_t, n_samples=100)
    mc_mean = mc_mean.cpu().numpy().flatten()
    mc_std = mc_std.cpu().numpy().flatten()

    # Ensemble predictions
    print("Computing Deep Ensemble predictions...")
    ens_mean, ens_epistemic, ens_aleatoric = ensemble.predict_with_uncertainty(
        X_test_t, device=device
    )
    ens_mean = ens_mean.numpy().flatten()
    ens_epistemic = ens_epistemic.numpy().flatten()
    ens_aleatoric = ens_aleatoric.numpy().flatten()
    ens_total_std = np.sqrt(ens_epistemic**2 + ens_aleatoric**2)

    # Compute metrics
    print("\nUncertainty Metrics:")
    print(f"MC Dropout - Mean uncertainty: {mc_std.mean():.4f}")
    print(f"Ensemble - Epistemic uncertainty: {ens_epistemic.mean():.4f}")
    print(f"Ensemble - Aleatoric uncertainty: {ens_aleatoric.mean():.4f}")
    print(f"Ensemble - Total uncertainty: {ens_total_std.mean():.4f}")

    # ========================================
    # 4. Manifold Analysis
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: Manifold-Based Analysis")
    print("="*80)

    # Create manifold embedding
    print("\nComputing manifold embedding (UMAP)...")
    manifold = ManifoldAnalyzer(method='umap', random_state=42)
    X_test_embedded = manifold.fit_transform(X_test, n_components=2, n_neighbors=15)

    # Reliability regions
    print("Defining reliability regions...")
    reliability_analyzer = ManifoldBasedReliabilityRegions(
        embedded_features=X_test_embedded,
        uncertainties=mc_std,
        predictions=mc_mean,
        true_labels=y_test
    )

    region_labels = reliability_analyzer.define_reliability_regions(
        n_regions=3,
        method='uncertainty_quantiles'
    )

    region_stats = reliability_analyzer.analyze_region_characteristics(region_labels)
    print("\nReliability Region Statistics:")
    for region_id, stats in region_stats.items():
        print(f"\nRegion {region_id}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    # ========================================
    # 5. Reliability Assessment
    # ========================================
    print("\n" + "="*80)
    print("STEP 5: Reliability Assessment")
    print("="*80)

    # Compute reliability scores
    reliability_scores = ReliabilityScorer.compute_reliability_score(
        epistemic_uncertainty=ens_epistemic,
        aleatoric_uncertainty=ens_aleatoric
    )

    print(f"\nMean reliability score: {reliability_scores.mean():.4f}")

    # Compute errors
    errors = np.abs(mc_mean - y_test)

    # Analyze correlation
    corr_stats = ReliabilityDiagnostics.uncertainty_error_correlation(
        mc_std, errors
    )
    print(f"\nUncertainty-Error Correlation:")
    print(f"  Pearson: {corr_stats['pearson_correlation']:.4f} (p={corr_stats['pearson_pvalue']:.4e})")
    print(f"  Spearman: {corr_stats['spearman_correlation']:.4f} (p={corr_stats['spearman_pvalue']:.4e})")

    # Selective prediction analysis
    selective_stats = ReliabilityDiagnostics.selective_prediction_analysis(
        predictions=mc_mean,
        uncertainties=mc_std,
        true_values=y_test,
        coverage_levels=[0.5, 0.7, 0.8, 0.9, 0.95]
    )

    print("\nSelective Prediction Analysis:")
    for i, cov in enumerate(selective_stats['coverage']):
        print(f"  Coverage {cov:.0%}: RMSE = {selective_stats['rmse'][i]:.4f}")

    # ========================================
    # 6. Visualization
    # ========================================
    print("\n" + "="*80)
    print("STEP 6: Creating Visualizations")
    print("="*80)

    # Create comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    fig = ReliabilityVisualizer.plot_comprehensive_dashboard(
        predictions=mc_mean,
        uncertainties=mc_std,
        true_values=y_test,
        embedded=X_test_embedded,
        x=None  # Use indices for x-axis
    )
    plt.savefig('bayesian_uq_dashboard.png', dpi=300, bbox_inches='tight')
    print("Saved: bayesian_uq_dashboard.png")

    # Manifold with reliability regions
    print("\nGenerating manifold reliability plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ManifoldVisualizer.plot_reliability_regions(
        X_test_embedded,
        region_labels,
        region_names=['High Reliability', 'Medium Reliability', 'Low Reliability'],
        ax=ax
    )
    plt.savefig('manifold_reliability_regions.png', dpi=300, bbox_inches='tight')
    print("Saved: manifold_reliability_regions.png")

    # Selective prediction curve
    print("\nGenerating selective prediction curve...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ReliabilityVisualizer.plot_selective_prediction(
        selective_stats['coverage'],
        selective_stats['rmse'],
        ax=ax
    )
    plt.savefig('selective_prediction.png', dpi=300, bbox_inches='tight')
    print("Saved: selective_prediction.png")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    # Save results
    results = {
        'mc_predictions': mc_mean,
        'mc_uncertainties': mc_std,
        'ensemble_predictions': ens_mean,
        'ensemble_epistemic': ens_epistemic,
        'ensemble_aleatoric': ens_aleatoric,
        'manifold_embedding': X_test_embedded,
        'reliability_scores': reliability_scores,
        'region_labels': region_labels,
        'true_values': y_test
    }

    np.savez('bayesian_uq_results.npz', **results)
    print("\nResults saved to: bayesian_uq_results.npz")

    return results


if __name__ == '__main__':
    results = main()
    plt.show()
