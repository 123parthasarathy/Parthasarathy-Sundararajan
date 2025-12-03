# Methodology Corrections

This document addresses the referee criticisms and explains the SVM-specific methodology implemented.

## Dataset

**Cleveland Heart Disease Dataset (UCI Repository)**
- Source: StatLog Heart Disease dataset via OpenML (ID 53)
- Samples: 270 patients
- Features: 13 clinical attributes
- Baseline accuracy: ~83-84%

## Summary of Referee Criticisms (Addressed)

| Criticism | Original Issue | Solution |
|-----------|---------------|----------|
| No novelty | Generic filter methods | SVM-RFE (SVM-specific feature selection) |
| Simple voting | Basic ensemble voting | Stacked SVM with meta-learner |
| No calibration | Uncalibrated probabilities | Platt scaling calibration |
| Class imbalance ignored | Equal class weights | Cost-sensitive SVM |
| Data leakage | Feature selection before CV | Pipeline-based within CV |
| Wrong statistics | Independent t-test | Paired t-test for matched folds |

---

## SVM-Specific Methods Implemented

### 1. SVM-RFE (Recursive Feature Elimination)

Unlike generic filter methods (chi-square, ANOVA, MI), SVM-RFE uses the classifier itself:

```
Algorithm:
1. Train LinearSVM on all features
2. Rank features by |w| (absolute weight coefficients)
3. Remove lowest-ranked feature
4. Repeat until desired number of features
```

**Why this matters**: Feature importance is determined by the SVM decision boundary, not generic statistical tests.

### 2. Stacked SVM (Meta-Learning)

Different from simple voting:

```
Simple Voting:        Stacked SVM:

Linear  --\           Linear  ---> P(y|x) --\
RBF     ---> Vote     RBF     ---> P(y|x) ---> Meta-SVM --> y
Poly    --/           Poly    ---> P(y|x) --/

Fixed weights         Learned combination
```

**Why this matters**: The meta-learner learns optimal kernel combination from data rather than using fixed weights.

### 3. Calibrated SVM (Platt Scaling)

SVMs produce uncalibrated decision scores. Platt scaling fits a sigmoid:

```
P(y=1|f(x)) = 1 / (1 + exp(A*f(x) + B))
```

Where A and B are learned via maximum likelihood on held-out data.

**Measured by**: Brier score = mean((p_predicted - y_true)^2)

### 4. Cost-Sensitive SVM

For imbalanced classes, modifies the objective:

```
Standard:     min ||w||² + C * Σξᵢ
Cost-Sens:    min ||w||² + C₀ * Σξᵢ(y=0) + C₁ * Σξᵢ(y=1)
```

Where C₀ and C₁ are inversely proportional to class frequencies.

---

## Methodological Rigor

### No Data Leakage

All preprocessing in `sklearn.Pipeline`:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', RFE(estimator=LinearSVC(), n_features_to_select=k)),
    ('svm', SVC(kernel='rbf'))
])
```

### Proper Statistical Tests

Paired t-test for matched CV folds:

```python
t_stat, p_value = stats.ttest_rel(model_scores, baseline_scores)
```

### Nested Cross-Validation

- Outer CV: 10-fold for performance estimation
- Inner CV: 5-fold for hyperparameter tuning

---

## Results Summary

| Model | CV Accuracy | Significant? |
|-------|-------------|--------------|
| RBF-SVM (Baseline) | 83.70% | - |
| SVM + SVM-RFE | 84.44% | No (p=0.168) |
| Cost-Sensitive SVM | 80.74% | No (p=0.053) |
| Calibrated SVM | 82.96% | No (p=0.168) |
| Stacked SVM | 84.07% | No (p=0.591) |

**Conclusion**: Advanced SVM methods do not show statistically significant improvement over a properly tuned baseline on this dataset. This is an honest negative result.
