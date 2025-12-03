# Advanced SVM Methods for Medical Diagnosis

Comparison of advanced SVM-specific techniques for heart disease prediction using the Cleveland Heart Disease dataset (UCI Repository).

## Dataset

- **Name**: StatLog Heart Disease Dataset
- **Source**: UCI Machine Learning Repository (OpenML ID 53)
- **Samples**: 270 patients
- **Features**: 13 clinical attributes
- **Task**: Binary classification (heart disease presence/absence)

## SVM Methods Implemented

### 1. SVM-RFE (Recursive Feature Elimination)
Uses SVM coefficients to iteratively rank and select features. Unlike generic filter methods (chi-square, ANOVA), SVM-RFE:
- Uses linear SVM weights to rank feature importance
- Iteratively removes least important features
- Re-trains SVM at each step to update rankings

### 2. Stacked SVM
A meta-learning approach where:
- Base SVMs with different kernels (linear, RBF, polynomial) generate predictions
- Cross-validated predictions become meta-features
- An SVM meta-learner combines base predictions optimally

### 3. Calibrated SVM (Platt Scaling)
Applies sigmoid calibration to improve probability estimates:
- Standard SVM produces uncalibrated scores
- Platt scaling fits a sigmoid to convert scores to probabilities
- Measured using Brier score (lower = better calibration)

### 4. Cost-Sensitive SVM
Handles class imbalance by weighting classes inversely proportional to frequency.

## Key Methodological Features

- **SVM-specific feature selection** (not generic filter methods)
- **Nested cross-validation** for unbiased performance estimates
- **Proper statistical tests** (paired t-test for matched CV folds)
- **Calibration analysis** using Brier score

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
python ensemble_svm_analysis.py
```

## Output Files

- `results_comparison.png` - Visualization of results
- `model_comparison_results.csv` - Performance summary
- `statistical_comparison.csv` - Statistical test results
- `svm_rfe_feature_ranking.csv` - SVM-RFE feature rankings
