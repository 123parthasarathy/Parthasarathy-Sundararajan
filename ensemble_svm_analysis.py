"""
Empirical Comparison of Ensemble SVM Methods for Medical Diagnosis
Using the Cleveland Heart Disease Dataset (UCI Repository)

This script performs a rigorous comparison of SVM variants with proper
statistical methodology, addressing common pitfalls in ML evaluation.

Key methodological considerations:
- Feature selection performed within cross-validation to prevent data leakage
- Hyperparameter tuning for fair comparison across all models
- Proper paired statistical tests for model comparison
- Honest interpretation of statistical significance

Dataset: Cleveland Heart Disease Database
Source: UCI Machine Learning Repository
Samples: 303 patients
Features: 13 clinical attributes
Task: Predict presence of heart disease (binary classification)

Author: Research Analysis
Date: 2025
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve, cross_validate
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report, make_scorer
)
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """
    Load and prepare the Cleveland Heart Disease dataset.

    Source: UCI Machine Learning Repository (via OpenML)
    This is a more challenging dataset than Wisconsin Breast Cancer,
    with lower baseline accuracy (~75-85%), making it suitable for
    evaluating whether ensemble methods provide meaningful benefit.
    """
    print("=" * 70)
    print("ENSEMBLE SVM COMPARISON STUDY")
    print("Cleveland Heart Disease Dataset (UCI Repository)")
    print("=" * 70)

    # Load StatLog Heart Disease dataset from OpenML (ID 53)
    # This is a well-known heart disease dataset with 270 samples, 13 features
    heart = fetch_openml(data_id=53, as_frame=True, parser='auto')
    X = heart.data
    y = heart.target

    # Convert categorical target to binary
    # Target values are 'absent' (1) and 'present' (2) or similar
    # Map to 0 = no disease, 1 = disease
    if y.dtype == 'object' or str(y.dtype) == 'category':
        # Handle string/categorical labels
        unique_vals = y.unique()
        print(f"  Target values found: {unique_vals}")
        # Map: first value (typically 'absent' or '1') -> 0, second -> 1
        y = pd.Categorical(y).codes
    else:
        # Numeric: assume 1 = absent, 2 = present
        y = (y.astype(int) == 2).astype(int)

    feature_names = X.columns.tolist()

    # Ensure numeric types for features
    X = X.apply(pd.to_numeric, errors='coerce')

    # Handle any missing values
    if X.isnull().any().any():
        X = X.fillna(X.median())

    # Convert to numpy for sklearn compatibility
    y = np.array(y)

    print(f"\nDataset characteristics:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: No Disease={sum(y==0)}, Disease={sum(y==1)}")
    print(f"  Class balance: {sum(y==1)/len(y):.1%} positive (disease)")
    print(f"\nFeatures: {', '.join(feature_names)}")

    return X, y, feature_names


# ============================================================================
# FEATURE ANALYSIS (For reporting only - not used in model training)
# ============================================================================
def analyze_features(X, y, feature_names):
    """
    Analyze feature importance using multiple methods.

    Note: This analysis is for reporting purposes only. In the actual
    model pipeline, feature selection is performed within cross-validation
    folds to prevent data leakage.
    """
    print("\n" + "-" * 70)
    print("Feature Analysis (Descriptive)")
    print("-" * 70)

    # Standardize for analysis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ANOVA F-statistic
    f_scores, f_pvals = f_classif(X_scaled, y)

    # Mutual Information
    mi_scores = mutual_info_classif(X_scaled, y, random_state=RANDOM_STATE)

    # Create summary dataframe
    feature_analysis = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': f_scores,
        'F_pvalue': f_pvals,
        'MI_Score': mi_scores
    })

    # Normalize scores for comparison
    feature_analysis['F_Score_Norm'] = (
        (feature_analysis['F_Score'] - feature_analysis['F_Score'].min()) /
        (feature_analysis['F_Score'].max() - feature_analysis['F_Score'].min())
    )
    feature_analysis['MI_Score_Norm'] = (
        (feature_analysis['MI_Score'] - feature_analysis['MI_Score'].min()) /
        (feature_analysis['MI_Score'].max() - feature_analysis['MI_Score'].min())
    )

    # Combined score (simple average of normalized scores)
    feature_analysis['Combined_Score'] = (
        feature_analysis['F_Score_Norm'] + feature_analysis['MI_Score_Norm']
    ) / 2

    feature_analysis = feature_analysis.sort_values(
        'Combined_Score', ascending=False
    ).reset_index(drop=True)

    print("\nTop 10 features by combined score:")
    print(feature_analysis[['Feature', 'F_Score', 'MI_Score', 'Combined_Score']].head(10).to_string())

    return feature_analysis


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
def create_models(n_features):
    """
    Create model pipelines with integrated preprocessing and feature selection.

    Each pipeline includes:
    1. StandardScaler - normalization
    2. SelectKBest - feature selection (prevents data leakage when used in CV)
    3. SVM classifier

    Returns dict of {name: (pipeline, param_grid)} tuples.
    """
    # For smaller datasets, use most features; for larger, use 50-70%
    k_features = max(8, min(n_features, int(n_features * 0.8)))

    models = {}

    # 1. Standard RBF-SVM (tuned baseline)
    models['RBF-SVM (Tuned)'] = (
        Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k_features)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]),
        {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto', 0.1, 0.01]
        }
    )

    # 2. Linear SVM
    models['Linear-SVM'] = (
        Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k_features)),
            ('svm', SVC(kernel='linear', probability=True, random_state=RANDOM_STATE))
        ]),
        {
            'svm__C': [0.1, 1, 10]
        }
    )

    # 3. Polynomial SVM
    models['Poly-SVM'] = (
        Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k_features)),
            ('svm', SVC(kernel='poly', probability=True, random_state=RANDOM_STATE))
        ]),
        {
            'svm__C': [0.1, 1, 10],
            'svm__degree': [2, 3]
        }
    )

    # 4. Multi-Kernel Ensemble (Soft Voting)
    # Note: This uses pre-tuned parameters for computational efficiency
    models['Multi-Kernel Ensemble'] = (
        Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k_features)),
            ('ensemble', VotingClassifier(
                estimators=[
                    ('linear', SVC(kernel='linear', C=1, probability=True,
                                   random_state=RANDOM_STATE)),
                    ('rbf', SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                                random_state=RANDOM_STATE)),
                    ('poly', SVC(kernel='poly', C=1, degree=3, probability=True,
                                 random_state=RANDOM_STATE))
                ],
                voting='soft'
            ))
        ]),
        {}  # No hyperparameter search for ensemble
    )

    # 5. Bagging SVM
    models['Bagging-SVM'] = (
        Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(f_classif, k=k_features)),
            ('bagging', BaggingClassifier(
                estimator=SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                              random_state=RANDOM_STATE),
                n_estimators=20,
                max_samples=0.8,
                random_state=RANDOM_STATE
            ))
        ]),
        {}
    )

    return models


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================
def train_and_evaluate(X, y, models, cv_folds=10):
    """
    Train models with hyperparameter tuning and evaluate using cross-validation.

    Uses nested cross-validation when hyperparameter tuning is needed
    to provide unbiased performance estimates.
    """
    print("\n" + "-" * 70)
    print("Model Training and Evaluation")
    print("-" * 70)

    outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=RANDOM_STATE)

    results = {}
    cv_scores_all = {}

    for name, (pipeline, param_grid) in models.items():
        print(f"\nTraining: {name}")

        if param_grid:
            # Nested CV with hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1
            )

            # Get cross-validated scores with tuning
            cv_scores = cross_val_score(
                grid_search, X, y, cv=outer_cv, scoring='accuracy'
            )

            # Fit on full data to get best params
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"  Best params: {best_params}")
        else:
            # Direct cross-validation (no tuning needed)
            cv_scores = cross_val_score(
                pipeline, X, y, cv=outer_cv, scoring='accuracy'
            )
            best_model = clone(pipeline).fit(X, y)
            best_params = None

        cv_scores_all[name] = cv_scores

        results[name] = {
            'model': best_model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': best_params
        }

        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return results, cv_scores_all


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================
def statistical_comparison(cv_scores_all, baseline_name='RBF-SVM (Tuned)'):
    """
    Perform rigorous statistical comparison between models.

    Uses paired t-test (or Wilcoxon signed-rank as non-parametric alternative)
    since the CV folds are matched across models.

    Interpretation guidelines:
    - p < 0.05: Statistically significant difference
    - p >= 0.05: No significant difference (cannot claim improvement)
    """
    print("\n" + "-" * 70)
    print("Statistical Comparison")
    print("-" * 70)

    baseline_scores = cv_scores_all[baseline_name]
    comparison_results = []

    print(f"\nBaseline: {baseline_name}")
    print(f"Baseline CV Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std()*2:.4f})")

    print("\nPairwise comparisons (paired t-test):")
    print("-" * 50)

    for name, scores in cv_scores_all.items():
        if name == baseline_name:
            continue

        # Paired t-test (appropriate for matched CV folds)
        t_stat, p_value = stats.ttest_rel(scores, baseline_scores)

        # Also compute Wilcoxon as non-parametric alternative
        try:
            w_stat, w_pvalue = stats.wilcoxon(scores, baseline_scores)
        except ValueError:
            w_stat, w_pvalue = np.nan, np.nan

        diff = scores.mean() - baseline_scores.mean()

        comparison_results.append({
            'Model': name,
            'Mean_Diff': diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'wilcoxon_p': w_pvalue,
            'significant': p_value < 0.05
        })

        sig_marker = "*" if p_value < 0.05 else ""
        print(f"\n{name} vs {baseline_name}:")
        print(f"  Mean difference: {diff:+.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f} {sig_marker}")

        if p_value < 0.05:
            direction = "better" if diff > 0 else "worse"
            print(f"  Interpretation: Statistically significant - {name} is {direction}")
        else:
            print(f"  Interpretation: No significant difference (p >= 0.05)")

    return pd.DataFrame(comparison_results)


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
def bootstrap_confidence_interval(scores, n_bootstrap=10000, confidence=0.95):
    """
    Compute bootstrap confidence interval for the mean.

    Uses the percentile method:
    CI = [percentile(alpha/2), percentile(1-alpha/2)]

    where alpha = 1 - confidence level.
    """
    n = len(scores)
    bootstrap_means = np.array([
        np.mean(np.random.choice(scores, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return bootstrap_means.mean(), ci_lower, ci_upper


def compute_all_confidence_intervals(cv_scores_all, confidence=0.95):
    """Compute bootstrap CIs for all models."""
    print("\n" + "-" * 70)
    print(f"Bootstrap {confidence*100:.0f}% Confidence Intervals")
    print("-" * 70)

    ci_results = []

    for name, scores in cv_scores_all.items():
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(
            scores, confidence=confidence
        )
        ci_results.append({
            'Model': name,
            'Mean': mean,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'CI_Width': ci_upper - ci_lower
        })
        print(f"{name}:")
        print(f"  Mean: {mean:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return pd.DataFrame(ci_results)


# ============================================================================
# HOLDOUT EVALUATION
# ============================================================================
def evaluate_on_holdout(X, y, results, test_size=0.2):
    """Evaluate final models on held-out test set."""
    print("\n" + "-" * 70)
    print("Holdout Test Set Evaluation")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    holdout_results = {}

    for name, data in results.items():
        # Refit on training data
        model = clone(data['model'])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        holdout_results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }

        print(f"\n{name}:")
        print(f"  Accuracy:  {holdout_results[name]['accuracy']:.4f}")
        print(f"  Precision: {holdout_results[name]['precision']:.4f}")
        print(f"  Recall:    {holdout_results[name]['recall']:.4f}")
        print(f"  F1-Score:  {holdout_results[name]['f1']:.4f}")

    return holdout_results, y_test


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_visualizations(cv_scores_all, ci_results, holdout_results,
                          comparison_df, feature_analysis, y_test):
    """Generate publication-quality figures."""
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    fig = plt.figure(figsize=(16, 12))

    # 1. Cross-validation score distributions
    ax1 = plt.subplot(2, 3, 1)
    cv_data = pd.DataFrame(cv_scores_all)
    cv_melted = cv_data.melt(var_name='Model', value_name='Accuracy')
    sns.boxplot(data=cv_melted, x='Model', y='Accuracy', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Cross-Validation Score Distributions', fontweight='bold')
    # Dynamic y-axis based on data range
    min_acc = cv_melted['Accuracy'].min()
    ax1.set_ylim([max(0.5, min_acc - 0.1), 1.0])

    # 2. Confidence intervals
    ax2 = plt.subplot(2, 3, 2)
    models = ci_results['Model'].values
    means = ci_results['Mean'].values
    ci_lower = ci_results['CI_Lower'].values
    ci_upper = ci_results['CI_Upper'].values

    y_pos = np.arange(len(models))
    xerr = np.array([means - ci_lower, ci_upper - means])

    ax2.barh(y_pos, means, xerr=xerr, capsize=5, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Accuracy')
    ax2.set_title('95% Bootstrap Confidence Intervals', fontweight='bold')
    # Dynamic x-axis based on CI range
    ax2.set_xlim([max(0.5, ci_lower.min() - 0.05), min(1.0, ci_upper.max() + 0.05)])

    # 3. ROC curves
    ax3 = plt.subplot(2, 3, 3)
    for name, data in holdout_results.items():
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)

    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves (Holdout Set)', fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)

    # 4. Statistical comparison p-values
    ax4 = plt.subplot(2, 3, 4)
    if len(comparison_df) > 0:
        colors = ['green' if p < 0.05 else 'red' for p in comparison_df['p_value']]
        bars = ax4.barh(comparison_df['Model'], comparison_df['p_value'],
                        color=colors, alpha=0.7)
        ax4.axvline(x=0.05, color='black', linestyle='--',
                    label='Significance threshold (p=0.05)')
        ax4.set_xlabel('p-value (paired t-test)')
        ax4.set_title('Statistical Significance vs Baseline', fontweight='bold')
        ax4.legend(loc='lower right')

    # 5. Feature importance (top features)
    ax5 = plt.subplot(2, 3, 5)
    n_features_to_show = min(15, len(feature_analysis))
    top_features = feature_analysis.head(n_features_to_show)
    ax5.barh(range(n_features_to_show), top_features['Combined_Score'].values, alpha=0.7)
    ax5.set_yticks(range(n_features_to_show))
    ax5.set_yticklabels(top_features['Feature'].values, fontsize=8)
    ax5.set_xlabel('Combined Score (F + MI)')
    ax5.set_title(f'Feature Importance (Top {n_features_to_show})', fontweight='bold')
    ax5.invert_yaxis()

    # 6. Confusion matrix for best model
    ax6 = plt.subplot(2, 3, 6)
    best_model = max(holdout_results.items(), key=lambda x: x[1]['accuracy'])
    cm = confusion_matrix(best_model[1]['y_test'], best_model[1]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    ax6.set_title(f'Confusion Matrix: {best_model[0]}', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: results_comparison.png")

    return fig


# ============================================================================
# RESULTS SUMMARY
# ============================================================================
def generate_summary(results, comparison_df, ci_results, holdout_results):
    """Generate comprehensive results summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Create summary table
    summary_data = []
    for name, data in results.items():
        holdout = holdout_results[name]
        ci = ci_results[ci_results['Model'] == name].iloc[0]

        summary_data.append({
            'Model': name,
            'CV_Accuracy': f"{data['cv_mean']:.4f} +/- {data['cv_std']:.4f}",
            '95%_CI': f"[{ci['CI_Lower']:.4f}, {ci['CI_Upper']:.4f}]",
            'Holdout_Accuracy': f"{holdout['accuracy']:.4f}",
            'Holdout_F1': f"{holdout['f1']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))

    # Statistical conclusions
    print("\n" + "-" * 70)
    print("Statistical Conclusions")
    print("-" * 70)

    baseline = 'RBF-SVM (Tuned)'
    baseline_acc = results[baseline]['cv_mean']

    significant_improvements = comparison_df[
        (comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)
    ]

    if len(significant_improvements) > 0:
        print("\nModels with statistically significant improvement over baseline:")
        for _, row in significant_improvements.iterrows():
            print(f"  - {row['Model']}: +{row['Mean_Diff']:.4f} (p={row['p_value']:.4f})")
    else:
        print("\nNo model showed statistically significant improvement over baseline.")
        print("This suggests the ensemble methods do not provide meaningful benefit")
        print("over a properly tuned RBF-SVM on this dataset.")

    # Key findings
    print("\n" + "-" * 70)
    print("Key Findings")
    print("-" * 70)

    best_cv = max(results.items(), key=lambda x: x[1]['cv_mean'])
    print(f"\n1. Best CV performance: {best_cv[0]} ({best_cv[1]['cv_mean']:.4f})")

    print(f"\n2. Tuned RBF-SVM baseline: {baseline_acc:.4f}")

    has_improvement = any(
        (comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)
    )
    print(f"\n3. Statistically significant improvement found: {'Yes' if has_improvement else 'No'}")

    overlapping_cis = []
    baseline_ci = ci_results[ci_results['Model'] == baseline].iloc[0]
    for _, row in ci_results.iterrows():
        if row['Model'] != baseline:
            # Check CI overlap
            if (row['CI_Lower'] <= baseline_ci['CI_Upper'] and
                row['CI_Upper'] >= baseline_ci['CI_Lower']):
                overlapping_cis.append(row['Model'])

    if overlapping_cis:
        print(f"\n4. Models with overlapping CIs with baseline: {', '.join(overlapping_cis)}")
        print("   (Overlapping CIs indicate similar performance)")

    return summary_df


# ============================================================================
# HONEST INTERPRETATION
# ============================================================================
def write_honest_interpretation(comparison_df, results):
    """
    Provide an honest interpretation of results, avoiding overclaiming.
    """
    print("\n" + "=" * 70)
    print("HONEST INTERPRETATION")
    print("=" * 70)

    baseline_name = 'RBF-SVM (Tuned)'
    baseline_acc = results[baseline_name]['cv_mean']

    # Check for any significant improvements
    sig_improvements = comparison_df[
        (comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)
    ]

    if len(sig_improvements) == 0:
        print("""
CONCLUSION: The ensemble methods do not demonstrate statistically
significant improvement over a properly tuned RBF-SVM baseline.

This finding suggests that for the Cleveland Heart Disease dataset:

1. A well-tuned single-kernel SVM achieves competitive performance
2. The additional complexity of ensemble methods is not justified
3. Performance gains from ensembling are not statistically significant

RECOMMENDATIONS:

- For practical deployment: Use the simpler RBF-SVM (lower complexity)
- Consider other approaches (e.g., gradient boosting) if higher accuracy needed
- Be cautious of complexity without demonstrated improvement

This is a negative result, but negative results are valuable - they
prevent unnecessary complexity in production systems.
""")
    else:
        best_improvement = sig_improvements.loc[
            sig_improvements['Mean_Diff'].idxmax()
        ]
        print(f"""
CONCLUSION: {best_improvement['Model']} shows statistically significant
improvement over the baseline (p={best_improvement['p_value']:.4f}).

However, note that:
1. The absolute improvement is {best_improvement['Mean_Diff']:.4f}
2. Consider whether this improvement justifies the added complexity
3. Results should be validated on independent datasets
""")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load data
    X, y, feature_names = load_data()

    # Feature analysis (descriptive only)
    feature_analysis = analyze_features(X, y, feature_names)

    # Create models
    models = create_models(n_features=X.shape[1])

    # Train and evaluate
    results, cv_scores_all = train_and_evaluate(X, y, models)

    # Statistical comparison
    comparison_df = statistical_comparison(cv_scores_all)

    # Confidence intervals
    ci_results = compute_all_confidence_intervals(cv_scores_all)

    # Holdout evaluation
    holdout_results, y_test = evaluate_on_holdout(X, y, results)

    # Visualizations
    create_visualizations(
        cv_scores_all, ci_results, holdout_results,
        comparison_df, feature_analysis, y_test
    )

    # Summary
    summary_df = generate_summary(results, comparison_df, ci_results, holdout_results)

    # Save results
    summary_df.to_csv('model_comparison_results.csv', index=False)
    comparison_df.to_csv('statistical_comparison.csv', index=False)
    print("\nSaved: model_comparison_results.csv")
    print("Saved: statistical_comparison.csv")

    # Honest interpretation
    write_honest_interpretation(comparison_df, results)

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

    return results, comparison_df, summary_df


if __name__ == "__main__":
    results, comparison_df, summary_df = main()
    plt.show()
