"""
Bayesian UQ with Results Display and File Export
Results will be printed to console and saved to your user directory
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import pandas as pd
import os
from pathlib import Path

# Get user home directory for saving files
USER_HOME = str(Path.home())
SAVE_DIR = os.path.join(USER_HOME, 'BayesianUQ_Results')
os.makedirs(SAVE_DIR, exist_ok=True)

print("="*80)
print("BAYESIAN UNCERTAINTY QUANTIFICATION - FULL RESULTS")
print("="*80)
print(f"\n📁 Results will be saved to: {SAVE_DIR}")
print("="*80)

# 1. Load data
print("\n" + "="*80)
print("STEP 1: DATA LOADING")
print("="*80)

data = fetch_california_housing()
X, y = data.data, data.target

print(f"\n📊 Dataset: California Housing")
print(f"   • Total samples: {len(X):,}")
print(f"   • Features: {X.shape[1]}")
print(f"   • Feature names: {data.feature_names}")
print(f"\n   Target variable: Median house value ($100,000s)")
print(f"   • Min value: ${y.min():.2f}00,000")
print(f"   • Max value: ${y.max():.2f}00,000")
print(f"   • Mean value: ${y.mean():.2f}00,000")

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

print(f"\n📊 Data Split:")
print(f"   • Training samples: {X_train.shape[0]:,} (80%)")
print(f"   • Test samples: {X_test.shape[0]:,} (20%)")

# 2. Train ensemble
print("\n" + "="*80)
print("STEP 2: TRAINING BAYESIAN ENSEMBLE")
print("="*80)

n_models = 5
models = []

print(f"\n🧠 Training {n_models} neural networks for ensemble...")
print(f"   Architecture: [8 → 128 → 64 → 32 → 1]")

for i in range(n_models):
    print(f"\n   Training model {i+1}/{n_models}...", end=' ')
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=100,
        random_state=i,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"✓ Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    models.append(model)

print(f"\n✅ All {n_models} models trained successfully!")

# 3. Get predictions with uncertainty
print("\n" + "="*80)
print("STEP 3: UNCERTAINTY QUANTIFICATION")
print("="*80)

predictions = []
for model in models:
    pred = model.predict(X_test)
    predictions.append(pred)

predictions = np.array(predictions)
mean = predictions.mean(axis=0)
std = predictions.std(axis=0)  # Epistemic uncertainty

errors = np.abs(mean - y_test)
squared_errors = (mean - y_test)**2

# Calculate metrics
rmse = np.sqrt(np.mean(squared_errors))
mae = np.mean(errors)
r2 = 1 - np.sum(squared_errors) / np.sum((y_test - y_test.mean())**2)
corr = np.corrcoef(std, errors)[0, 1]

print(f"\n📈 PREDICTION PERFORMANCE:")
print(f"   • RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"   • MAE (Mean Absolute Error): {mae:.4f}")
print(f"   • R² Score: {r2:.4f}")

print(f"\n🎯 UNCERTAINTY STATISTICS:")
print(f"   • Mean uncertainty: {std.mean():.4f}")
print(f"   • Min uncertainty: {std.min():.4f}")
print(f"   • Max uncertainty: {std.max():.4f}")
print(f"   • Std of uncertainty: {std.std():.4f}")

print(f"\n📊 CALIBRATION QUALITY:")
print(f"   • Uncertainty-Error Correlation: {corr:.4f}")
if corr > 0.4:
    print(f"   ✅ EXCELLENT - Model knows when it's uncertain!")
elif corr > 0.2:
    print(f"   ✓ GOOD - Reasonable calibration")
else:
    print(f"   ⚠ FAIR - Calibration could be improved")

# Coverage analysis
lower_95 = mean - 2*std
upper_95 = mean + 2*std
coverage_95 = np.mean((y_test >= lower_95) & (y_test <= upper_95))

print(f"\n📏 PREDICTION INTERVALS:")
print(f"   • 95% Interval Coverage: {coverage_95:.2%}")
print(f"   • Expected coverage: 95.0%")
if abs(coverage_95 - 0.95) < 0.05:
    print(f"   ✅ EXCELLENT - Well calibrated intervals!")
elif abs(coverage_95 - 0.95) < 0.10:
    print(f"   ✓ GOOD - Reasonable interval calibration")
else:
    print(f"   ⚠ Intervals need calibration adjustment")

# Analyze by uncertainty levels
print(f"\n🔍 DETAILED ANALYSIS BY UNCERTAINTY LEVEL:")
percentiles = [0, 25, 50, 75, 90, 100]
print(f"\n   {'Percentile':<12} {'Uncertainty':<15} {'Mean Error':<15} {'RMSE':<10}")
print(f"   {'-'*52}")
for i in range(len(percentiles)-1):
    lower_p = percentiles[i]
    upper_p = percentiles[i+1]
    mask = (std >= np.percentile(std, lower_p)) & (std <= np.percentile(std, upper_p))
    mean_unc = std[mask].mean()
    mean_err = errors[mask].mean()
    rmse_level = np.sqrt(np.mean(squared_errors[mask]))
    print(f"   {lower_p:>3}%-{upper_p:<3}%     {mean_unc:>8.4f}        {mean_err:>8.4f}        {rmse_level:>8.4f}")

# Find most/least certain predictions
print(f"\n🎲 MOST UNCERTAIN PREDICTIONS (Top 5):")
most_uncertain_idx = np.argsort(std)[-5:][::-1]
print(f"   {'Index':<8} {'Prediction':<12} {'True Value':<12} {'Uncertainty':<12} {'Error':<10}")
print(f"   {'-'*64}")
for idx in most_uncertain_idx:
    print(f"   {idx:<8} {mean[idx]:>10.4f}  {y_test[idx]:>10.4f}  {std[idx]:>10.4f}  {errors[idx]:>8.4f}")

print(f"\n🎯 MOST CONFIDENT PREDICTIONS (Top 5):")
most_confident_idx = np.argsort(std)[:5]
print(f"   {'Index':<8} {'Prediction':<12} {'True Value':<12} {'Uncertainty':<12} {'Error':<10}")
print(f"   {'-'*64}")
for idx in most_confident_idx:
    print(f"   {idx:<8} {mean[idx]:>10.4f}  {y_test[idx]:>10.4f}  {std[idx]:>10.4f}  {errors[idx]:>8.4f}")

# 4. Manifold analysis
print("\n" + "="*80)
print("STEP 4: MANIFOLD ANALYSIS")
print("="*80)

pca = PCA(n_components=2)
n_viz = min(5000, len(X_test))
embedded = pca.fit_transform(X_test[:n_viz])

print(f"\n🌐 PCA Dimensionality Reduction:")
print(f"   • Samples embedded: {n_viz:,}")
print(f"   • Original dimensions: {X_test.shape[1]}")
print(f"   • Reduced dimensions: 2")
print(f"   • Explained variance (PC1): {pca.explained_variance_ratio_[0]:.2%}")
print(f"   • Explained variance (PC2): {pca.explained_variance_ratio_[1]:.2%}")
print(f"   • Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Analyze uncertainty in manifold space
manifold_distances = np.linalg.norm(embedded - embedded.mean(axis=0), axis=1)
corr_dist = np.corrcoef(manifold_distances, std[:n_viz])[0, 1]

print(f"\n📍 Manifold-Uncertainty Relationship:")
print(f"   • Correlation (distance vs uncertainty): {corr_dist:.4f}")
if corr_dist > 0.3:
    print(f"   ✅ Higher uncertainty at manifold boundaries")
elif corr_dist > 0:
    print(f"   ✓ Weak relationship with manifold position")
else:
    print(f"   → Uncertainty not strongly related to manifold position")

# 5. Save results to files
print("\n" + "="*80)
print("STEP 5: SAVING RESULTS TO FILES")
print("="*80)

# Save predictions to CSV
results_df = pd.DataFrame({
    'True_Value': y_test,
    'Prediction': mean,
    'Uncertainty': std,
    'Absolute_Error': errors,
    'Lower_95CI': lower_95,
    'Upper_95CI': upper_95,
    'In_95CI': (y_test >= lower_95) & (y_test <= upper_95)
})

csv_path = os.path.join(SAVE_DIR, 'predictions_with_uncertainty.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n✓ Saved predictions: {csv_path}")

# Save summary statistics
summary_path = os.path.join(SAVE_DIR, 'summary_statistics.txt')
with open(summary_path, 'w') as f:
    f.write("BAYESIAN UNCERTAINTY QUANTIFICATION - SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset: California Housing\n")
    f.write(f"Total Samples: {len(X):,}\n")
    f.write(f"Training Samples: {len(X_train):,}\n")
    f.write(f"Test Samples: {len(X_test):,}\n\n")
    f.write(f"PERFORMANCE METRICS:\n")
    f.write(f"  RMSE: {rmse:.4f}\n")
    f.write(f"  MAE: {mae:.4f}\n")
    f.write(f"  R² Score: {r2:.4f}\n\n")
    f.write(f"UNCERTAINTY METRICS:\n")
    f.write(f"  Mean Uncertainty: {std.mean():.4f}\n")
    f.write(f"  Uncertainty-Error Correlation: {corr:.4f}\n")
    f.write(f"  95% Coverage: {coverage_95:.2%}\n\n")
    f.write(f"ENSEMBLE DETAILS:\n")
    f.write(f"  Number of Models: {n_models}\n")
    f.write(f"  Architecture: [8, 128, 64, 32, 1]\n")

print(f"✓ Saved summary: {summary_path}")

# 6. Create and save visualizations
print("\n" + "="*80)
print("STEP 6: CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Predictions with uncertainty
ax1 = plt.subplot(3, 3, 1)
x = np.arange(min(300, len(mean)))
ax1.fill_between(x, mean[:300] - 2*std[:300], mean[:300] + 2*std[:300],
                 alpha=0.3, label='95% CI')
ax1.plot(x, mean[:300], 'b-', linewidth=2, label='Prediction')
ax1.plot(x, y_test[:300], 'r--', linewidth=1.5, label='True')
ax1.set_xlabel('Sample Index', fontsize=10)
ax1.set_ylabel('Normalized Value', fontsize=10)
ax1.set_title('Predictions with Uncertainty Bands', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Uncertainty distribution
ax2 = plt.subplot(3, 3, 2)
ax2.hist(std, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(std.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {std.mean():.4f}')
ax2.set_xlabel('Uncertainty (Std Dev)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Uncertainty vs Error (Calibration)
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(std, errors, alpha=0.5, s=20)
max_val = max(std.max(), errors.max())
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Calibration')
ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}',
        transform=ax3.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat'))
ax3.set_xlabel('Predicted Uncertainty', fontsize=10)
ax3.set_ylabel('Actual Error', fontsize=10)
ax3.set_title('Calibration: Uncertainty vs Error', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error distribution
ax4 = plt.subplot(3, 3, 4)
ax4.hist(errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
ax4.axvline(rmse, color='blue', linestyle='--', linewidth=2,
            label=f'RMSE: {rmse:.4f}')
ax4.axvline(mae, color='green', linestyle='--', linewidth=2,
            label=f'MAE: {mae:.4f}')
ax4.set_xlabel('Absolute Error', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Manifold
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(embedded[:, 0], embedded[:, 1],
                     c=std[:n_viz], cmap='viridis', alpha=0.6, s=30)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Uncertainty', fontsize=9)
ax5.set_xlabel('Principal Component 1', fontsize=10)
ax5.set_ylabel('Principal Component 2', fontsize=10)
ax5.set_title('PCA Manifold Colored by Uncertainty', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Coverage analysis
ax6 = plt.subplot(3, 3, 6)
coverage_levels = []
interval_widths = []
for multiplier in [1, 1.5, 2, 2.5, 3]:
    lower = mean - multiplier*std
    upper = mean + multiplier*std
    cov = np.mean((y_test >= lower) & (y_test <= upper))
    coverage_levels.append(cov)
    interval_widths.append(multiplier*2)

ax6.plot(interval_widths, coverage_levels, 'o-', linewidth=2, markersize=8)
ax6.axhline(y=0.95, color='r', linestyle='--', label='Target 95%')
ax6.set_xlabel('Interval Width (× std)', fontsize=10)
ax6.set_ylabel('Coverage Probability', fontsize=10)
ax6.set_title('Prediction Interval Coverage', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Error by uncertainty level
ax7 = plt.subplot(3, 3, 7)
n_bins = 10
bins = np.percentile(std, np.linspace(0, 100, n_bins+1))
bin_errors = []
bin_centers = []
for i in range(n_bins):
    mask = (std >= bins[i]) & (std <= bins[i+1])
    if mask.sum() > 0:
        bin_errors.append(errors[mask].mean())
        bin_centers.append((bins[i] + bins[i+1]) / 2)

ax7.plot(bin_centers, bin_errors, 'o-', linewidth=2, markersize=8, color='purple')
ax7.set_xlabel('Uncertainty Level', fontsize=10)
ax7.set_ylabel('Mean Error', fontsize=10)
ax7.set_title('Error by Uncertainty Level', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Prediction scatter
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(y_test, mean, alpha=0.5, s=20, c=std, cmap='plasma')
ax8.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect')
ax8.set_xlabel('True Value', fontsize=10)
ax8.set_ylabel('Predicted Value', fontsize=10)
ax8.set_title('Predictions vs True Values', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Summary statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
SUMMARY STATISTICS

Performance:
  • RMSE: {rmse:.4f}
  • MAE: {mae:.4f}
  • R² Score: {r2:.4f}

Uncertainty:
  • Mean: {std.mean():.4f}
  • Std: {std.std():.4f}
  • Correlation: {corr:.4f}

Coverage:
  • 95% Interval: {coverage_95:.1%}

Ensemble:
  • Models: {n_models}
  • Test Samples: {len(y_test):,}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center')

plt.suptitle('Bayesian Uncertainty Quantification - Complete Analysis',
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
plot_path = os.path.join(SAVE_DIR, 'complete_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {plot_path}")

plt.show()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)

print(f"\n📁 ALL RESULTS SAVED TO:")
print(f"   {SAVE_DIR}")
print(f"\n📄 Files created:")
print(f"   1. predictions_with_uncertainty.csv - All predictions with uncertainties")
print(f"   2. summary_statistics.txt - Performance summary")
print(f"   3. complete_analysis.png - Comprehensive visualization")

print(f"\n💡 TIP: Open these files to explore the results!")
print(f"   Excel: Open the CSV file")
print(f"   Notepad: View the summary statistics")
print(f"   Image viewer: See the complete analysis plot")

print("\n" + "="*80)

# Return results for Spyder Variable Explorer
results = {
    'predictions': mean,
    'uncertainties': std,
    'true_values': y_test,
    'errors': errors,
    'manifold_embedding': embedded,
    'models': models,
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'correlation': corr,
    'coverage_95': coverage_95,
    'results_dataframe': results_df
}

print("✅ Results also available in 'results' variable in Variable Explorer")
print("="*80)
