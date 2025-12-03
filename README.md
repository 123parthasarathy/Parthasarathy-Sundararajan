# Ensemble SVM Comparison Study

Empirical comparison of SVM ensemble methods for medical diagnosis using the Wisconsin Diagnostic Breast Cancer dataset.

## Overview

This repository contains a methodologically rigorous comparison of SVM variants, addressing common pitfalls in machine learning evaluation studies.

## Key Methodological Features

- **No data leakage**: Feature selection performed within cross-validation folds using sklearn Pipeline
- **Fair comparison**: Hyperparameter tuning for all models including baseline
- **Proper statistical tests**: Paired t-test for matched CV folds
- **Honest interpretation**: Results reported according to statistical significance

## Files

- `ensemble_svm_analysis.py` - Main analysis script
- `METHODOLOGY_CORRECTIONS.md` - Documentation of methodological corrections

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
python ensemble_svm_analysis.py
```

## Output

The script generates:
- `results_comparison.png` - Visualization of results
- `model_comparison_results.csv` - Performance summary
- `statistical_comparison.csv` - Statistical test results

## Models Compared

1. RBF-SVM (Tuned) - Baseline with hyperparameter tuning
2. Linear-SVM
3. Polynomial-SVM
4. Multi-Kernel Ensemble (Soft Voting)
5. Bagging-SVM

## Methodology Notes

This analysis acknowledges that:
- The techniques used are established methods, not novel contributions
- Results are interpreted according to statistical significance (p < 0.05)
- Negative results (no significant improvement) are reported honestly
