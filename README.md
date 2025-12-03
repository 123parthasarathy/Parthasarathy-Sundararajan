# Improved SVM Methods for Medical Diagnosis

Advanced SVM techniques for heart disease prediction using the Cleveland Heart Disease dataset (UCI Repository).

## Dataset

- **Name**: StatLog Heart Disease Dataset
- **Source**: UCI Machine Learning Repository (OpenML ID 53)
- **Samples**: 270 patients
- **Features**: 13 clinical attributes
- **Task**: Binary classification (heart disease presence/absence)

## Improved Methods Implemented

### 1. SMOTE (Synthetic Minority Over-sampling)
Generates synthetic samples for minority class by interpolating between existing samples and their k-nearest neighbors. Addresses class imbalance without simple duplication.

### 2. Polynomial Feature Interactions
Creates degree-2 interaction features (e.g., age*cholesterol) to capture non-linear relationships between clinical measurements.

### 3. Bayesian-Inspired Hyperparameter Optimization
Uses exploration-exploitation strategy to find optimal C and gamma parameters more efficiently than grid search.

### 4. Multiple Kernel Learning (MKL)
Learns optimal weighted combination of kernels:
- K_combined = w1*K_linear + w2*K_rbf + w3*K_poly
- Weights are optimized via cross-validation

### 5. Nystroem Kernel Approximation
Approximates RBF kernel using subset of training samples, enabling linear SVM in approximated feature space. Best performing method.

## Results

| Model | CV Accuracy | Holdout Accuracy |
|-------|-------------|------------------|
| RBF-SVM (Baseline) | 83.33% | 81.48% |
| SMOTE + SVM | 82.59% | 81.48% |
| PolyFeatures + SVM | 80.74% | 83.33% |
| Bayesian-Opt SVM | 81.85% | **87.04%** |
| MKL-SVM | 75.19% | 77.78% |
| **Nystroem-SVM** | **84.44%** | 85.19% |

**Best CV Performance**: Nystroem-SVM (+1.11% over baseline)
**Best Holdout Performance**: Bayesian-Opt SVM (87.04%)

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
python ensemble_svm_analysis.py
```

## Output Files

- `results_comparison.png` - Visualization of results
- `model_comparison_results.csv` - Performance summary
- `statistical_comparison.csv` - Statistical test results
