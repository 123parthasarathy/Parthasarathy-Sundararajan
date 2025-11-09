# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Clone or download the repository
git clone https://github.com/yourusername/bayesian-uq-framework.git
cd bayesian-uq-framework

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the framework
pip install -e .
```

## Run Your First Example (2 minutes)

### Regression Example
```bash
cd examples
python example_regression.py
```

This will:
- Download the California Housing dataset automatically
- Train a Bayesian neural network with MC Dropout
- Train a Deep Ensemble (5 models)
- Compute epistemic and aleatoric uncertainty
- Create manifold embeddings with UMAP
- Generate comprehensive visualizations
- Save results to `bayesian_uq_results.npz`

### Classification Example
```bash
python example_classification.py
```

This will:
- Download MNIST dataset automatically
- Train a Bayesian classifier
- Compute predictive uncertainty and entropy
- Analyze calibration quality
- Detect out-of-distribution samples
- Create manifold visualizations
- Save results to `classification_results.npz`

## Understanding the Output

After running the examples, you'll get:

### Generated Files

1. **Visualizations** (PNG images):
   - `bayesian_uq_dashboard.png` - Comprehensive analysis dashboard
   - `manifold_reliability_regions.png` - Reliability regions in manifold space
   - `selective_prediction.png` - Performance vs coverage trade-off
   - `classification_calibration.png` - Calibration curve
   - `classification_manifold_analysis.png` - Manifold colored by uncertainty

2. **Results** (NPZ files):
   - `bayesian_uq_results.npz` - All regression results
   - `classification_results.npz` - All classification results

### Load and Explore Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
results = np.load('bayesian_uq_results.npz')

# Available data
print("Available keys:", results.files)

# Access predictions and uncertainties
predictions = results['mc_predictions']
uncertainties = results['mc_uncertainties']
true_values = results['true_values']

# Compute metrics
errors = np.abs(predictions - true_values)
print(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f}")
print(f"Mean uncertainty: {uncertainties.mean():.4f}")

# Correlation between uncertainty and error
correlation = np.corrcoef(uncertainties, errors)[0, 1]
print(f"Uncertainty-Error correlation: {correlation:.4f}")
```

## Use with Your Own Data

### Regression

```python
import torch
from bayesian_uq_framework.models import MCDropoutNN
from sklearn.preprocessing import StandardScaler

# 1. Prepare your data
X_train, y_train = your_data_loading_function()

# Normalize
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)

# 2. Create model
model = MCDropoutNN(
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],  # Adjust architecture
    output_dim=1,
    dropout_rate=0.2
)

# 3. Train (standard PyTorch training loop)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# ... your training loop ...

# 4. Get predictions with uncertainty
X_test_t = torch.FloatTensor(X_test).to(device)
mean, std = model.mc_predict(X_test_t, n_samples=100)

print(f"Predictions: {mean.cpu().numpy()}")
print(f"Uncertainties: {std.cpu().numpy()}")
```

### Classification

```python
from bayesian_uq_framework.models import MCDropoutNN
import torch.nn.functional as F

# 1. Create classifier
model = MCDropoutNN(
    input_dim=784,  # e.g., flattened 28x28 images
    hidden_dims=[256, 128, 64],
    output_dim=10,  # number of classes
    dropout_rate=0.3
)

# 2. Train with CrossEntropyLoss
# ... training loop ...

# 3. Get predictions with uncertainty
model.train()  # Enable dropout
predictions = []

with torch.no_grad():
    for _ in range(100):  # MC samples
        logits = model(X_test)
        probs = F.softmax(logits, dim=-1)
        predictions.append(probs)

predictions = torch.stack(predictions)

# Mean predictions
mean_probs = predictions.mean(dim=0)

# Uncertainty (entropy)
entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
```

## Next Steps

### 1. Explore Different Models

```python
# Deep Ensemble
from bayesian_uq_framework.models import DeepEnsemble

ensemble = DeepEnsemble(
    input_dim=20,
    hidden_dims=[128, 64],
    output_dim=1,
    n_models=5
)

# Variational Inference
from bayesian_uq_framework.models import VariationalNN

vnn = VariationalNN(
    input_dim=20,
    hidden_dims=[128, 64],
    output_dim=1,
    prior_std=1.0
)
```

### 2. Manifold Analysis

```python
from bayesian_uq_framework.manifold import ManifoldAnalyzer, ManifoldBasedReliabilityRegions

# Create embedding
analyzer = ManifoldAnalyzer(method='umap')
embedded = analyzer.fit_transform(X_test, n_components=2)

# Define reliability regions
regions = ManifoldBasedReliabilityRegions(
    embedded_features=embedded,
    uncertainties=uncertainties,
    predictions=predictions,
    true_labels=y_test
)

labels = regions.define_reliability_regions(n_regions=3)
stats = regions.analyze_region_characteristics(labels)
```

### 3. Reliability Assessment

```python
from bayesian_uq_framework.reliability import (
    ReliabilityScorer,
    OutOfDistributionDetector,
    UncertaintyBasedRejection
)

# Reliability scoring
scores = ReliabilityScorer.compute_reliability_score(
    epistemic_uncertainty=epistemic,
    aleatoric_uncertainty=aleatoric
)

# OOD detection
detector = OutOfDistributionDetector(method='ensemble')
detector.fit(X_train, train_uncertainties)
ood_scores = detector.predict(X_test, test_uncertainties)

# Rejection mechanism
rejector = UncertaintyBasedRejection(coverage=0.95)
accepted_preds, mask, stats = rejector.reject_samples(
    predictions, uncertainties
)
```

### 4. Visualization

```python
from bayesian_uq_framework.utils import (
    UncertaintyVisualizer,
    ManifoldVisualizer,
    CalibrationVisualizer,
    ReliabilityVisualizer
)

# Predictions with uncertainty bands
UncertaintyVisualizer.plot_predictions_with_uncertainty(
    x=x_values,
    predictions=predictions,
    uncertainties=uncertainties,
    true_values=y_true
)

# Manifold colored by uncertainty
ManifoldVisualizer.plot_uncertainty_manifold(
    embedded, uncertainties
)

# Calibration curve
CalibrationVisualizer.plot_calibration_curve(
    confidences, accuracies
)

# Comprehensive dashboard
ReliabilityVisualizer.plot_comprehensive_dashboard(
    predictions, uncertainties, true_values, embedded
)
```

## Dataset Download Paths

### Automatic Downloads (No Manual Setup)

```python
# California Housing
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

# MNIST
from torchvision.datasets import MNIST
dataset = MNIST(root='./data', download=True)

# OpenML datasets
from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=42165)  # Diamonds dataset
```

### Manual Downloads

```python
# Direct URLs for large datasets
URLS = {
    'concrete': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
    'energy': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
    'power_plant': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',
    'year_prediction': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
}

# Download with urllib
import urllib.request
urllib.request.urlretrieve(URLS['concrete'], 'Concrete_Data.xls')
```

## Common Issues

### 1. UMAP Installation Error

```bash
# If umap-learn fails to install
pip install numba
pip install umap-learn
```

### 2. CUDA/GPU Issues

```python
# Force CPU if GPU has issues
device = 'cpu'
model = model.to(device)
```

### 3. Memory Issues with Large Datasets

```python
# Use batching for predictions
batch_size = 1000
predictions = []

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    pred = model.mc_predict(X_batch, n_samples=100)
    predictions.append(pred)
```

## Getting Help

- **Documentation**: See `README.md` for full documentation
- **Examples**: Check `examples/` directory for complete workflows
- **Issues**: Report bugs on GitHub Issues
- **Questions**: Open a discussion or contact the authors

## Performance Tips

1. **Use GPU**: Set `device='cuda'` for 10-100x speedup
2. **Reduce MC Samples**: Start with 50 samples, increase if needed
3. **Batch Processing**: Process data in batches to manage memory
4. **Manifold Subsampling**: Use subset of data for visualization

```python
# Example: Fast uncertainty estimation
mean, std = model.mc_predict(X_test, n_samples=50)  # Faster

# Example: Subsample for manifold
indices = np.random.choice(len(X_test), 5000, replace=False)
X_subset = X_test[indices]
embedded = manifold.fit_transform(X_subset)
```

---

**You're ready to go! Start with the examples and adapt them to your use case.**
