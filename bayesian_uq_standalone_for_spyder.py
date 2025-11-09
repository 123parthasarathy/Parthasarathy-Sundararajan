"""
COMPLETE BAYESIAN UNCERTAINTY QUANTIFICATION FRAMEWORK
Standalone version for Spyder IDE

This single file contains everything you need:
- Bayesian Neural Networks (MC Dropout)
- Uncertainty Quantification
- Manifold Analysis
- Visualization

USAGE IN SPYDER:
1. Copy this entire file
2. Create new file in Spyder: File → New File
3. Paste the code
4. Press F5 to run

Author: Bayesian UQ Framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Try to import UMAP, use PCA if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not available. Using PCA. Install with: pip install umap-learn")


# ============================================================================
# BAYESIAN NEURAL NETWORK
# ============================================================================

class MCDropoutNN(nn.Module):
    """
    Monte Carlo Dropout Neural Network for Bayesian inference.
    Reference: Gal & Ghahramani (2016)
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(MCDropoutNN, self).__init__()

        self.dropout_rate = dropout_rate
        layers = []

        # Build network
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def mc_predict(self, x, n_samples=100):
        """
        Perform Monte Carlo sampling for uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean: Predictive mean
            std: Epistemic uncertainty
        """
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std


# ============================================================================
# UNCERTAINTY METRICS
# ============================================================================

class UncertaintyMetrics:
    """Compute various uncertainty metrics"""

    @staticmethod
    def compute_calibration(predictions, uncertainties, true_values, n_bins=10):
        """
        Compute calibration metrics for regression.

        Returns expected vs observed errors.
        """
        errors = np.abs(true_values - predictions)

        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        bin_size = len(uncertainties) // n_bins

        expected_errors = []
        observed_errors = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainties)

            bin_indices = sorted_indices[start_idx:end_idx]

            expected_errors.append(np.mean(uncertainties[bin_indices]))
            observed_errors.append(np.mean(errors[bin_indices]))

        return np.array(expected_errors), np.array(observed_errors)

    @staticmethod
    def prediction_interval_coverage(true_values, lower, upper):
        """
        Compute Prediction Interval Coverage Probability (PICP).
        """
        in_interval = (true_values >= lower) & (true_values <= upper)
        return np.mean(in_interval)


# ============================================================================
# MANIFOLD ANALYZER
# ============================================================================

class ManifoldAnalyzer:
    """Analyze data in low-dimensional manifold"""

    def __init__(self, method='umap'):
        self.method = method
        self.reducer = None

    def fit_transform(self, features, n_components=2):
        """
        Create manifold embedding.

        Args:
            features: Input features
            n_components: Output dimensions

        Returns:
            Embedded features
        """
        if self.method == 'umap' and UMAP_AVAILABLE:
            self.reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15
            )
        elif self.method == 'pca' or not UMAP_AVAILABLE:
            self.reducer = PCA(n_components=n_components, random_state=42)
        else:
            # Fallback to PCA
            self.reducer = PCA(n_components=n_components, random_state=42)

        embedded = self.reducer.fit_transform(features)
        return embedded


# ============================================================================
# RELIABILITY SCORER
# ============================================================================

class ReliabilityScorer:
    """Compute reliability scores"""

    @staticmethod
    def compute_reliability_score(uncertainties, manifold_distances=None):
        """
        Compute composite reliability score (0=unreliable, 1=reliable).

        Args:
            uncertainties: Uncertainty estimates
            manifold_distances: Optional distances to training manifold

        Returns:
            Reliability scores
        """
        # Normalize uncertainties
        unc_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-10)

        # Reliability = 1 - normalized uncertainty
        reliability = 1 - unc_norm

        # If manifold distances provided, incorporate them
        if manifold_distances is not None:
            dist_norm = (manifold_distances - manifold_distances.min()) / (manifold_distances.max() - manifold_distances.min() + 1e-10)
            reliability = 0.7 * reliability + 0.3 * (1 - dist_norm)

        return reliability


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_predictions_with_uncertainty(x, predictions, uncertainties,
                                     true_values=None, n_std=2.0):
    """Plot predictions with uncertainty bands"""
    plt.figure(figsize=(12, 6))

    # Sort for visualization
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    pred_sorted = predictions[sort_idx]
    unc_sorted = uncertainties[sort_idx]

    # Plot uncertainty band
    plt.fill_between(x_sorted,
                     pred_sorted - n_std * unc_sorted,
                     pred_sorted + n_std * unc_sorted,
                     alpha=0.3, color='blue', label=f'{n_std}σ Uncertainty')

    # Plot predictions
    plt.plot(x_sorted, pred_sorted, 'b-', linewidth=2, label='Prediction')

    # Plot true values
    if true_values is not None:
        true_sorted = true_values[sort_idx]
        plt.plot(x_sorted, true_sorted, 'r--', linewidth=2, label='True')

    plt.xlabel('Input Index', fontsize=12)
    plt.ylabel('Output Value', fontsize=12)
    plt.title('Bayesian Predictions with Uncertainty Bands', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_vs_error(uncertainties, errors):
    """Scatter plot of uncertainty vs actual error"""
    plt.figure(figsize=(10, 6))

    plt.scatter(uncertainties, errors, alpha=0.5, s=30, c='blue', edgecolors='black', linewidth=0.5)

    # Perfect calibration line
    max_val = max(uncertainties.max(), errors.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Calibration')

    # Compute correlation
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=11)

    plt.xlabel('Predicted Uncertainty', fontsize=12)
    plt.ylabel('Actual Error', fontsize=12)
    plt.title('Uncertainty vs Error (Calibration Quality)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_manifold_with_uncertainty(embedded, uncertainties):
    """Plot 2D manifold colored by uncertainty"""
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                         c=uncertainties, cmap='viridis',
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Uncertainty', fontsize=11)

    plt.xlabel('Manifold Dimension 1', fontsize=12)
    plt.ylabel('Manifold Dimension 2', fontsize=12)
    plt.title('Data Manifold Colored by Uncertainty', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comprehensive_dashboard(predictions, uncertainties, true_values, embedded=None):
    """Create comprehensive analysis dashboard"""

    if embedded is not None:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    errors = np.abs(predictions - true_values)

    # 1. Predictions with uncertainty (first 200 points)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(min(200, len(predictions)))
    ax1.fill_between(x,
                     predictions[:200] - 2*uncertainties[:200],
                     predictions[:200] + 2*uncertainties[:200],
                     alpha=0.3)
    ax1.plot(x, predictions[:200], 'b-', linewidth=2, label='Prediction')
    ax1.plot(x, true_values[:200], 'r--', linewidth=1.5, label='True')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Predictions with Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Uncertainty distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(uncertainties, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(uncertainties.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {uncertainties.mean():.4f}')
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Uncertainty vs Error
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(uncertainties, errors, alpha=0.4, s=20)
    max_val = max(uncertainties.max(), errors.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    ax3.text(0.05, 0.95, f'Corr: {corr:.3f}',
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat'))
    ax3.set_xlabel('Uncertainty')
    ax3.set_ylabel('Error')
    ax3.set_title('Uncertainty vs Error')
    ax3.grid(True, alpha=0.3)

    # 4. Error distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(errors.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {errors.mean():.4f}')
    rmse = np.sqrt(np.mean(errors**2))
    ax4.axvline(rmse, color='blue', linestyle='--',
                linewidth=2, label=f'RMSE: {rmse:.4f}')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Manifold plot (if available)
    if embedded is not None:
        ax5 = fig.add_subplot(gs[2, :])
        scatter = ax5.scatter(embedded[:, 0], embedded[:, 1],
                            c=uncertainties, cmap='viridis',
                            alpha=0.6, s=30)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Uncertainty')
        ax5.set_xlabel('Dimension 1')
        ax5.set_ylabel('Dimension 2')
        ax5.set_title('Manifold Embedding Colored by Uncertainty')
        ax5.grid(True, alpha=0.3)

    fig.suptitle('Bayesian UQ Comprehensive Dashboard', fontsize=16, fontweight='bold')
    plt.show()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_bayesian_uq_analysis():
    """
    Complete Bayesian Uncertainty Quantification Analysis

    Returns:
        Dictionary with all results for further analysis in Spyder
    """

    print("="*80)
    print("BAYESIAN UNCERTAINTY QUANTIFICATION FRAMEWORK")
    print("Manifold-Based Reliability Assessment")
    print("="*80)

    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("\n📊 STEP 1: Loading Dataset")
    print("-" * 80)

    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"✓ Dataset: California Housing")
    print(f"✓ Train samples: {X_train.shape[0]}")
    print(f"✓ Test samples: {X_test.shape[0]}")
    print(f"✓ Features: {X_train.shape[1]}")

    # ========================================
    # 2. CREATE AND TRAIN MODEL
    # ========================================
    print("\n🧠 STEP 2: Training Bayesian Neural Network")
    print("-" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")

    model = MCDropoutNN(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        output_dim=1,
        dropout_rate=0.2
    ).to(device)

    print(f"✓ Architecture: [8, 128, 64, 32, 1]")
    print(f"✓ Dropout rate: 0.2")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=128,
        shuffle=True
    )

    print("\n⏳ Training...")
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"   Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f}")

    print("✓ Training completed!")

    # ========================================
    # 3. UNCERTAINTY QUANTIFICATION
    # ========================================
    print("\n📈 STEP 3: Uncertainty Quantification")
    print("-" * 80)

    X_test_t = torch.FloatTensor(X_test).to(device)
    mean, std = model.mc_predict(X_test_t, n_samples=100)

    mean = mean.cpu().numpy().flatten()
    std = std.cpu().numpy().flatten()

    errors = np.abs(mean - y_test)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)

    print(f"✓ MC Samples: 100")
    print(f"✓ RMSE: {rmse:.4f}")
    print(f"✓ MAE: {mae:.4f}")
    print(f"✓ Mean Uncertainty: {std.mean():.4f}")
    print(f"✓ Uncertainty-Error Correlation: {np.corrcoef(std, errors)[0,1]:.4f}")

    # Calibration
    expected, observed = UncertaintyMetrics.compute_calibration(
        mean, std, y_test, n_bins=10
    )
    miscalibration = np.mean(np.abs(expected - observed))
    print(f"✓ Miscalibration Area: {miscalibration:.4f}")

    # Prediction intervals
    lower = mean - 2 * std
    upper = mean + 2 * std
    coverage = UncertaintyMetrics.prediction_interval_coverage(y_test, lower, upper)
    print(f"✓ 95% Interval Coverage: {coverage:.2%}")

    # ========================================
    # 4. MANIFOLD ANALYSIS
    # ========================================
    print("\n🌐 STEP 4: Manifold Analysis")
    print("-" * 80)

    manifold = ManifoldAnalyzer(method='umap' if UMAP_AVAILABLE else 'pca')
    n_viz = min(5000, len(X_test))
    embedded = manifold.fit_transform(X_test[:n_viz])

    print(f"✓ Method: {'UMAP' if UMAP_AVAILABLE else 'PCA'}")
    print(f"✓ Samples embedded: {n_viz}")
    print(f"✓ Embedding dimensions: 2")

    # ========================================
    # 5. RELIABILITY ASSESSMENT
    # ========================================
    print("\n🎯 STEP 5: Reliability Assessment")
    print("-" * 80)

    # Compute manifold distances
    manifold_center = embedded.mean(axis=0)
    manifold_distances = np.linalg.norm(embedded - manifold_center, axis=1)

    # Reliability scores
    reliability = ReliabilityScorer.compute_reliability_score(
        std[:n_viz], manifold_distances
    )

    print(f"✓ Mean Reliability Score: {reliability.mean():.4f}")
    print(f"✓ High Reliability (>0.7): {(reliability > 0.7).sum()} samples ({(reliability > 0.7).mean():.1%})")
    print(f"✓ Low Reliability (<0.3): {(reliability < 0.3).sum()} samples ({(reliability < 0.3).mean():.1%})")

    # ========================================
    # 6. VISUALIZATION
    # ========================================
    print("\n📊 STEP 6: Creating Visualizations")
    print("-" * 80)

    # Plot 1: Predictions with uncertainty
    print("   Creating plot 1: Predictions with Uncertainty...")
    plot_predictions_with_uncertainty(
        x=np.arange(200),
        predictions=mean[:200],
        uncertainties=std[:200],
        true_values=y_test[:200]
    )

    # Plot 2: Uncertainty vs Error
    print("   Creating plot 2: Uncertainty vs Error...")
    plot_uncertainty_vs_error(std, errors)

    # Plot 3: Manifold
    print("   Creating plot 3: Manifold Embedding...")
    plot_manifold_with_uncertainty(embedded, std[:n_viz])

    # Plot 4: Comprehensive dashboard
    print("   Creating plot 4: Comprehensive Dashboard...")
    plot_comprehensive_dashboard(mean, std, y_test, embedded)

    print("\n✓ All visualizations created!")

    # ========================================
    # 7. RESULTS SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

    print("\n📝 Summary:")
    print(f"   • Model Performance: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    print(f"   • Uncertainty Quality: Correlation = {np.corrcoef(std, errors)[0,1]:.3f}")
    print(f"   • Calibration: Miscalibration = {miscalibration:.4f}")
    print(f"   • Coverage: {coverage:.1%} of true values in 95% intervals")
    print(f"   • Reliability: {reliability.mean():.3f} average score")

    print("\n💾 Results Dictionary Created:")
    print("   Access results in Variable Explorer or use:")
    print("   - results['predictions']")
    print("   - results['uncertainties']")
    print("   - results['errors']")
    print("   - results['manifold_embedding']")
    print("   - results['reliability_scores']")

    # Return all results
    return {
        'model': model,
        'predictions': mean,
        'uncertainties': std,
        'true_values': y_test,
        'errors': errors,
        'manifold_embedding': embedded,
        'reliability_scores': reliability,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'correlation': np.corrcoef(std, errors)[0,1],
            'miscalibration': miscalibration,
            'coverage': coverage
        },
        'X_test': X_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == '__main__':
    # Run complete analysis
    results = run_bayesian_uq_analysis()

    print("\n" + "="*80)
    print("🎉 All done! Results are available in the 'results' variable.")
    print("   You can now explore them in Spyder's Variable Explorer!")
    print("="*80)
