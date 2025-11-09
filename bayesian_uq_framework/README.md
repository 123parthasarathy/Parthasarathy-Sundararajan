# Bayesian Uncertainty Quantification in Deep Learning: A Manifold-Based Reliability Assessment Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for uncertainty quantification in deep learning that combines:
- **Bayesian Inference** for neural networks
- **Uncertainty Quantification** (epistemic & aleatoric)
- **Manifold Learning** for geometric reliability analysis
- **Comprehensive Visualization** tools

## Features

### 🧠 Bayesian Neural Networks
- **Monte Carlo Dropout** (Gal & Ghahramani, 2016)
- **Deep Ensembles** (Lakshminarayanan et al., 2017)
- **Variational Inference** (Bayes by Backprop)
- **Concrete Dropout** (automatic dropout tuning)

### 📊 Uncertainty Quantification
- Predictive entropy and mutual information
- Aleatoric and epistemic uncertainty decomposition
- Prediction intervals and confidence estimation
- Calibration metrics (ECE, MCE, Brier score)
- Regression-specific metrics (PICP, MPIW)

### 🌐 Manifold Learning
- t-SNE, UMAP, Isomap, PCA embeddings
- Intrinsic dimensionality estimation
- Geometric reliability analysis
- Local curvature and boundary detection
- Manifold-based reliability regions

### 🎯 Reliability Assessment
- Composite reliability scoring
- Out-of-distribution (OOD) detection
- Uncertainty-based rejection
- Selective prediction analysis
- Interpolation vs extrapolation detection

### 📈 Visualization
- Uncertainty plots with prediction bands
- Calibration curves and reliability diagrams
- Manifold embeddings colored by uncertainty
- Comprehensive dashboards
- Interactive visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bayesian-uq-framework.git
cd bayesian-uq-framework

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Regression Example

```python
import numpy as np
from bayesian_uq_framework.models import MCDropoutNN, DeepEnsemble
from bayesian_uq_framework.uncertainty import UncertaintyMetrics
from bayesian_uq_framework.manifold import ManifoldAnalyzer
from bayesian_uq_framework.reliability import ReliabilityScorer

# Load your data
X_train, y_train, X_test, y_test = load_your_data()

# Train Bayesian model
model = MCDropoutNN(input_dim=X_train.shape[1],
                    hidden_dims=[128, 64],
                    output_dim=1)
# ... training code ...

# Get predictions with uncertainty
mean, std = model.mc_predict(X_test, n_samples=100)

# Manifold analysis
manifold = ManifoldAnalyzer(method='umap')
embedded = manifold.fit_transform(X_test)

# Reliability assessment
reliability = ReliabilityScorer.compute_reliability_score(
    epistemic_uncertainty=std,
    manifold_distance=compute_manifold_distance(embedded)
)
```

### Classification Example

```python
from bayesian_uq_framework.models import MCDropoutNN
from bayesian_uq_framework.uncertainty import CalibrationMetrics

# Train classifier
model = BayesianClassifier(input_dim=784, n_classes=10)
# ... training ...

# Get predictions with uncertainty
probs, entropy, epistemic = model.mc_predict_proba(X_test, n_samples=100)

# Calibration analysis
ece = CalibrationMetrics.expected_calibration_error(
    confidences=probs.max(axis=1),
    accuracies=correctness
)
```

## Complete Examples

The `examples/` directory contains comprehensive demonstrations:

### 1. Regression Analysis
```bash
python examples/example_regression.py
```

Features:
- Multiple dataset options (California Housing, UCI datasets, synthetic)
- MC Dropout and Deep Ensemble training
- Comprehensive uncertainty quantification
- Manifold-based reliability regions
- Full visualization dashboard

### 2. Classification Analysis
```bash
python examples/example_classification.py
```

Features:
- MNIST and other classification datasets
- Bayesian classifier training
- Calibration analysis
- OOD detection
- Manifold visualization by uncertainty

## Large-Scale Dataset Sources

The framework includes support for downloading and processing large-scale datasets:

### Regression Datasets

| Dataset | Size | Features | Download |
|---------|------|----------|----------|
| California Housing | 20,640 | 8 | `sklearn.datasets.fetch_california_housing()` |
| Concrete Strength | 1,030 | 8 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) |
| Year Prediction MSD | 515,345 | 90 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD) |
| OpenML Diamonds | 53,940 | 10 | `fetch_openml(data_id=42165)` |

### Classification Datasets

| Dataset | Size | Classes | Download |
|---------|------|---------|----------|
| MNIST | 70,000 | 10 | `torchvision.datasets.MNIST` |
| CIFAR-10 | 60,000 | 10 | `torchvision.datasets.CIFAR10` |
| Fashion-MNIST | 70,000 | 10 | `torchvision.datasets.FashionMNIST` |
| Covertype | 581,012 | 7 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Covertype) |

### Direct Download Links

```python
# UCI Machine Learning Repository
DATASET_URLS = {
    'concrete': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
    'energy': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
    'power_plant': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',
    'year_msd': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
    'protein': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',
    'bike_sharing': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip',
}

# OpenML (via sklearn)
from sklearn.datasets import fetch_openml

# Diamonds dataset (regression)
data = fetch_openml(data_id=42165, as_frame=False)

# Vehicle sensor dataset (large-scale, 100k samples)
data = fetch_openml(data_id=41514, as_frame=False)
```

## Framework Architecture

```
bayesian_uq_framework/
├── models/
│   ├── bayesian_nn.py          # Bayesian neural network implementations
│   └── __init__.py
├── uncertainty/
│   ├── metrics.py               # Uncertainty quantification metrics
│   └── __init__.py
├── manifold/
│   ├── manifold_analysis.py     # Manifold learning & analysis
│   └── __init__.py
├── reliability/
│   ├── assessment.py            # Reliability scoring & diagnostics
│   └── __init__.py
├── utils/
│   ├── visualization.py         # Comprehensive plotting tools
│   └── __init__.py
└── examples/
    ├── example_regression.py
    └── example_classification.py
```

## Key Components

### Bayesian Models

#### Monte Carlo Dropout
```python
model = MCDropoutNN(
    input_dim=20,
    hidden_dims=[128, 64, 32],
    output_dim=1,
    dropout_rate=0.2
)

# Predict with uncertainty
mean, std = model.mc_predict(X, n_samples=100)
```

#### Deep Ensemble
```python
ensemble = DeepEnsemble(
    input_dim=20,
    hidden_dims=[128, 64],
    output_dim=1,
    n_models=5
)

# Decomposed uncertainty
mean, epistemic, aleatoric = ensemble.predict_with_uncertainty(X)
```

#### Variational Inference
```python
model = VariationalNN(
    input_dim=20,
    hidden_dims=[128, 64],
    output_dim=1,
    prior_std=1.0
)

# ELBO loss for training
loss = model.elbo_loss(X, y, n_samples=10)
```

### Uncertainty Metrics

```python
from bayesian_uq_framework.uncertainty import UncertaintyMetrics

# For classification
entropy = UncertaintyMetrics.predictive_entropy(predictions)
mi = UncertaintyMetrics.mutual_information(predictions)

# For regression
variance = UncertaintyMetrics.predictive_variance(predictions)
lower, upper = UncertaintyMetrics.prediction_interval(predictions, confidence=0.95)
```

### Manifold Analysis

```python
from bayesian_uq_framework.manifold import ManifoldAnalyzer, ManifoldBasedReliabilityRegions

# Create manifold embedding
analyzer = ManifoldAnalyzer(method='umap')
embedded = analyzer.fit_transform(features, n_components=2)

# Define reliability regions
regions = ManifoldBasedReliabilityRegions(
    embedded_features=embedded,
    uncertainties=uncertainties,
    predictions=predictions
)

labels = regions.define_reliability_regions(n_regions=3)
stats = regions.analyze_region_characteristics(labels)
```

### Reliability Assessment

```python
from bayesian_uq_framework.reliability import ReliabilityScorer, OutOfDistributionDetector

# Composite reliability score
score = ReliabilityScorer.compute_reliability_score(
    epistemic_uncertainty=epistemic,
    aleatoric_uncertainty=aleatoric,
    manifold_distance=distances
)

# OOD detection
detector = OutOfDistributionDetector(method='ensemble')
detector.fit(train_features, train_uncertainties)
ood_scores = detector.predict(test_features, test_uncertainties)
```

## Visualization

### Comprehensive Dashboard
```python
from bayesian_uq_framework.utils import ReliabilityVisualizer

fig = ReliabilityVisualizer.plot_comprehensive_dashboard(
    predictions=predictions,
    uncertainties=uncertainties,
    true_values=true_values,
    embedded=manifold_embedding
)
```

### Calibration Curves
```python
from bayesian_uq_framework.utils import CalibrationVisualizer

CalibrationVisualizer.plot_calibration_curve(
    confidences=confidences,
    accuracies=accuracies,
    n_bins=10
)
```

### Manifold Plots
```python
from bayesian_uq_framework.utils import ManifoldVisualizer

ManifoldVisualizer.plot_uncertainty_manifold(
    embedded=embedded,
    uncertainties=uncertainties
)
```

## Research Background

This framework implements methods from key papers in Bayesian deep learning:

1. **Gal & Ghahramani (2016)** - Dropout as a Bayesian Approximation
2. **Lakshminarayanan et al. (2017)** - Simple and Scalable Predictive Uncertainty Estimation
3. **Kendall & Gal (2017)** - What Uncertainties Do We Need in Bayesian Deep Learning?
4. **Blundell et al. (2015)** - Weight Uncertainty in Neural Networks

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{bayesian_uq_framework,
  title={Bayesian Uncertainty Quantification in Deep Learning: A Manifold-Based Reliability Assessment Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/bayesian-uq-framework}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with PyTorch, scikit-learn, and UMAP
- Inspired by research in Bayesian deep learning and uncertainty quantification
- Dataset sources: UCI ML Repository, OpenML, Kaggle

## Contact

For questions and feedback:
- Open an issue on GitHub
- Email: your.email@example.com

## Roadmap

- [ ] Add Gaussian Process integration
- [ ] Implement conformal prediction
- [ ] Add more Bayesian architectures (BNNs, SNGPs)
- [ ] Support for time series uncertainty
- [ ] Active learning integration
- [ ] TensorFlow/Keras backend
- [ ] Web-based interactive dashboard

---

**Note**: This is a research framework for educational and experimental purposes. For production use, additional validation and testing is recommended.
