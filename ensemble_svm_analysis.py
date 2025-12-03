"""
Advanced SVM Methods for Medical Diagnosis:
SVM-RFE Feature Selection, Stacked SVMs, and Calibration

This script implements and compares advanced SVM-specific techniques:
1. SVM-RFE (Recursive Feature Elimination using SVM weights)
2. Stacked SVM (using SVM predictions as meta-features)
3. Calibrated SVM (Platt scaling for probability calibration)
4. Cost-Sensitive SVM (handling class imbalance)

Dataset: Cleveland Heart Disease Database (UCI Repository)
Samples: 270 patients | Features: 13 clinical attributes

Key methodological features:
- SVM-specific feature selection (not generic filter methods)
- Proper nested cross-validation for unbiased estimates
- Stacking with SVM as meta-learner
- Statistical comparison with paired tests

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
    GridSearchCV, cross_val_predict
)
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


# ============================================================================
# CUSTOM STACKED SVM CLASSIFIER
# ============================================================================
class StackedSVM(BaseEstimator, ClassifierMixin):
    """
    Stacked SVM: Uses multiple SVM base learners with different kernels,
    then combines their predictions using an SVM meta-learner.

    This is a proper stacking approach where:
    1. Base SVMs with different kernels generate cross-validated predictions
    2. These predictions become features for a meta-SVM
    3. The meta-SVM learns optimal combination weights

    This differs from simple voting by learning the combination.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_learners = [
            ('linear', SVC(kernel='linear', C=1, probability=True,
                          random_state=random_state)),
            ('rbf', SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                       random_state=random_state)),
            ('poly', SVC(kernel='poly', C=1, degree=2, probability=True,
                        random_state=random_state))
        ]
        self.meta_learner = SVC(kernel='rbf', C=1, probability=True,
                                random_state=random_state)
        self.fitted_base_learners_ = []

    def fit(self, X, y):
        """Fit base learners and meta-learner using stacking."""
        X = np.array(X)
        y = np.array(y)

        # Generate cross-validated predictions from base learners
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=self.random_state)

        meta_features = np.zeros((len(y), len(self.base_learners) * 2))

        for i, (name, learner) in enumerate(self.base_learners):
            # Get cross-validated probability predictions
            proba = cross_val_predict(clone(learner), X, y, cv=cv,
                                     method='predict_proba')
            meta_features[:, i*2:(i+1)*2] = proba

            # Fit on full data for prediction phase
            fitted = clone(learner).fit(X, y)
            self.fitted_base_learners_.append((name, fitted))

        # Fit meta-learner on stacked features
        self.meta_learner.fit(meta_features, y)
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict using stacked ensemble."""
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X):
        """Predict probabilities using stacked ensemble."""
        meta_features = self._get_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)

    def _get_meta_features(self, X):
        """Generate meta-features from base learner predictions."""
        X = np.array(X)
        meta_features = np.zeros((len(X), len(self.fitted_base_learners_) * 2))

        for i, (name, learner) in enumerate(self.fitted_base_learners_):
            proba = learner.predict_proba(X)
            meta_features[:, i*2:(i+1)*2] = proba

        return meta_features


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load Cleveland Heart Disease dataset from UCI Repository."""
    print("=" * 70)
    print("ADVANCED SVM METHODS FOR MEDICAL DIAGNOSIS")
    print("Cleveland Heart Disease Dataset (UCI Repository)")
    print("=" * 70)

    # Load StatLog Heart Disease dataset (OpenML ID 53)
    heart = fetch_openml(data_id=53, as_frame=True, parser='auto')
    X = heart.data
    y = heart.target

    # Convert target to binary
    if y.dtype == 'object' or str(y.dtype) == 'category':
        unique_vals = y.unique()
        print(f"  Target values: {list(unique_vals)}")
        y = pd.Categorical(y).codes
    else:
        y = (y.astype(int) == 2).astype(int)

    feature_names = X.columns.tolist()
    X = X.apply(pd.to_numeric, errors='coerce')

    if X.isnull().any().any():
        X = X.fillna(X.median())

    y = np.array(y)

    # Calculate class weights for cost-sensitive learning
    class_counts = np.bincount(y)
    class_weights = {0: len(y) / (2 * class_counts[0]),
                     1: len(y) / (2 * class_counts[1])}

    print(f"\nDataset characteristics:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: No Disease={class_counts[0]}, Disease={class_counts[1]}")
    print(f"  Class balance: {class_counts[1]/len(y):.1%} positive")
    print(f"  Computed class weights: {class_weights}")

    return X, y, feature_names, class_weights


# ============================================================================
# SVM-RFE FEATURE SELECTION
# ============================================================================
def svm_rfe_analysis(X, y, feature_names):
    """
    SVM-RFE: Recursive Feature Elimination using SVM weights.

    Unlike generic filter methods (chi-square, ANOVA), SVM-RFE:
    1. Uses SVM coefficients to rank features
    2. Iteratively removes least important features
    3. Re-trains SVM at each step to update rankings

    This is an SVM-specific embedded feature selection method.
    """
    print("\n" + "-" * 70)
    print("SVM-RFE Feature Selection")
    print("-" * 70)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use LinearSVC for RFE (has coef_ attribute)
    svm = LinearSVC(C=1, random_state=RANDOM_STATE, max_iter=10000, dual=True)

    # RFE with SVM
    rfe = RFE(estimator=svm, n_features_to_select=1, step=1)
    rfe.fit(X_scaled, y)

    # Get feature rankings
    feature_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values('Ranking')

    print("\nFeature rankings (1 = most important):")
    print(feature_ranking.to_string(index=False))

    # Determine optimal number of features using CV
    print("\nFinding optimal feature count...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    n_features_range = range(3, len(feature_names) + 1)
    cv_scores = []

    for n_features in n_features_range:
        rfe_temp = RFE(estimator=clone(svm), n_features_to_select=n_features, step=1)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rfe', rfe_temp),
            ('svm', SVC(kernel='rbf', C=1, gamma='scale', random_state=RANDOM_STATE))
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        cv_scores.append(scores.mean())

    optimal_n = n_features_range[np.argmax(cv_scores)]
    print(f"  Optimal number of features: {optimal_n}")
    print(f"  CV accuracy with {optimal_n} features: {max(cv_scores):.4f}")

    # Get optimal feature set
    optimal_features = feature_ranking[feature_ranking['Ranking'] <= optimal_n]['Feature'].tolist()
    print(f"  Selected features: {optimal_features}")

    return feature_ranking, optimal_n, optimal_features, cv_scores


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
def create_models(n_features, optimal_n_features, class_weights):
    """
    Create SVM model variants with different methodological approaches.

    Models:
    1. Standard RBF-SVM (baseline)
    2. SVM with SVM-RFE feature selection
    3. Cost-Sensitive SVM (weighted for class imbalance)
    4. Calibrated SVM (Platt scaling for better probabilities)
    5. Stacked SVM (SVM meta-learner combining kernel variants)
    """

    models = {}

    # 1. Standard RBF-SVM (tuned baseline)
    models['RBF-SVM (Baseline)'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]),
        'param_grid': {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 0.1, 0.01]
        }
    }

    # 2. SVM with SVM-RFE feature selection
    svm_for_rfe = LinearSVC(C=1, random_state=RANDOM_STATE, max_iter=10000, dual=True)
    models['SVM + SVM-RFE'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('rfe', RFE(estimator=svm_for_rfe, n_features_to_select=optimal_n_features, step=1)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]),
        'param_grid': {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 0.1]
        }
    }

    # 3. Cost-Sensitive SVM
    models['Cost-Sensitive SVM'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, class_weight=class_weights,
                       random_state=RANDOM_STATE))
        ]),
        'param_grid': {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 0.1]
        }
    }

    # 4. Calibrated SVM (Platt scaling)
    # CalibratedClassifierCV applies Platt scaling to improve probability estimates
    base_svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=RANDOM_STATE)
    models['Calibrated SVM'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('calibrated_svm', CalibratedClassifierCV(base_svm, method='sigmoid', cv=5))
        ]),
        'param_grid': {}  # Calibration is done internally
    }

    # 5. Stacked SVM (meta-learner approach)
    models['Stacked SVM'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('stacked', StackedSVM(random_state=RANDOM_STATE))
        ]),
        'param_grid': {}
    }

    return models


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================
def train_and_evaluate(X, y, models, cv_folds=10):
    """Train and evaluate models with proper cross-validation."""
    print("\n" + "-" * 70)
    print("Model Training and Evaluation")
    print("-" * 70)

    outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=RANDOM_STATE)

    results = {}
    cv_scores_all = {}

    for name, model_config in models.items():
        print(f"\nTraining: {name}")
        pipeline = model_config['pipeline']
        param_grid = model_config['param_grid']

        if param_grid:
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1
            )
            cv_scores = cross_val_score(
                grid_search, X, y, cv=outer_cv, scoring='accuracy'
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"  Best params: {best_params}")
        else:
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
def statistical_comparison(cv_scores_all, baseline_name='RBF-SVM (Baseline)'):
    """Perform paired statistical tests between models."""
    print("\n" + "-" * 70)
    print("Statistical Comparison (Paired t-test)")
    print("-" * 70)

    baseline_scores = cv_scores_all[baseline_name]
    comparison_results = []

    print(f"\nBaseline: {baseline_name}")
    print(f"Baseline CV Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std()*2:.4f})")
    print("\n" + "-" * 50)

    for name, scores in cv_scores_all.items():
        if name == baseline_name:
            continue

        t_stat, p_value = stats.ttest_rel(scores, baseline_scores)
        diff = scores.mean() - baseline_scores.mean()

        comparison_results.append({
            'Model': name,
            'CV_Mean': scores.mean(),
            'Mean_Diff': diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

        sig = "*" if p_value < 0.05 else ""
        direction = "better" if diff > 0 else "worse" if diff < 0 else "equal"

        print(f"\n{name}:")
        print(f"  CV Accuracy: {scores.mean():.4f}")
        print(f"  Difference: {diff:+.4f} ({direction})")
        print(f"  p-value: {p_value:.4f} {sig}")

        if p_value < 0.05:
            print(f"  => Statistically significant difference")
        else:
            print(f"  => No significant difference")

    return pd.DataFrame(comparison_results)


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
def compute_confidence_intervals(cv_scores_all, n_bootstrap=10000):
    """Compute bootstrap 95% confidence intervals."""
    print("\n" + "-" * 70)
    print("Bootstrap 95% Confidence Intervals")
    print("-" * 70)

    ci_results = []

    for name, scores in cv_scores_all.items():
        n = len(scores)
        bootstrap_means = np.array([
            np.mean(np.random.choice(scores, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        ci_results.append({
            'Model': name,
            'Mean': scores.mean(),
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        })

        print(f"{name}: {scores.mean():.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    return pd.DataFrame(ci_results)


# ============================================================================
# HOLDOUT EVALUATION WITH CALIBRATION ANALYSIS
# ============================================================================
def evaluate_holdout(X, y, results):
    """Evaluate on holdout set with calibration analysis."""
    print("\n" + "-" * 70)
    print("Holdout Test Set Evaluation")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    holdout_results = {}

    for name, data in results.items():
        model = clone(data['model'])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Brier score measures calibration quality (lower is better)
        brier = brier_score_loss(y_test, y_proba)

        holdout_results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'brier_score': brier,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test
        }

        print(f"\n{name}:")
        print(f"  Accuracy:    {holdout_results[name]['accuracy']:.4f}")
        print(f"  Precision:   {holdout_results[name]['precision']:.4f}")
        print(f"  Recall:      {holdout_results[name]['recall']:.4f}")
        print(f"  F1-Score:    {holdout_results[name]['f1']:.4f}")
        print(f"  Brier Score: {holdout_results[name]['brier_score']:.4f} (calibration)")

    return holdout_results, y_test


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_visualizations(cv_scores_all, ci_results, holdout_results,
                          comparison_df, feature_ranking, rfe_cv_scores):
    """Generate publication-quality figures."""
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    fig = plt.figure(figsize=(16, 12))

    # 1. CV Score Distributions
    ax1 = plt.subplot(2, 3, 1)
    cv_data = pd.DataFrame(cv_scores_all)
    cv_melted = cv_data.melt(var_name='Model', value_name='Accuracy')
    sns.boxplot(data=cv_melted, x='Model', y='Accuracy', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Cross-Validation Score Distributions', fontweight='bold')
    min_acc = cv_melted['Accuracy'].min()
    ax1.set_ylim([max(0.5, min_acc - 0.1), 1.0])

    # 2. SVM-RFE Feature Ranking
    ax2 = plt.subplot(2, 3, 2)
    sorted_features = feature_ranking.sort_values('Ranking')
    colors = ['green' if r <= 8 else 'lightgray'
              for r in sorted_features['Ranking']]
    ax2.barh(range(len(sorted_features)),
             max(sorted_features['Ranking']) - sorted_features['Ranking'] + 1,
             color=colors, alpha=0.7)
    ax2.set_yticks(range(len(sorted_features)))
    ax2.set_yticklabels(sorted_features['Feature'].values, fontsize=8)
    ax2.set_xlabel('SVM-RFE Importance Score')
    ax2.set_title('SVM-RFE Feature Ranking', fontweight='bold')
    ax2.invert_yaxis()

    # 3. RFE Feature Count vs CV Accuracy
    ax3 = plt.subplot(2, 3, 3)
    n_features_range = range(3, len(feature_ranking) + 1)
    ax3.plot(list(n_features_range), rfe_cv_scores, 'b-o', linewidth=2, markersize=6)
    optimal_idx = np.argmax(rfe_cv_scores)
    ax3.axvline(x=list(n_features_range)[optimal_idx], color='r', linestyle='--',
                label=f'Optimal: {list(n_features_range)[optimal_idx]} features')
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('CV Accuracy')
    ax3.set_title('SVM-RFE: Feature Count vs Performance', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. ROC Curves
    ax4 = plt.subplot(2, 3, 4)
    for name, data in holdout_results.items():
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=2)
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves', fontweight='bold')
    ax4.legend(loc='lower right', fontsize=7)

    # 5. Calibration Comparison (Brier Scores)
    ax5 = plt.subplot(2, 3, 5)
    brier_scores = {name: data['brier_score'] for name, data in holdout_results.items()}
    models_sorted = sorted(brier_scores.items(), key=lambda x: x[1])
    names = [m[0] for m in models_sorted]
    scores = [m[1] for m in models_sorted]
    colors = ['green' if 'Calibrated' in n else 'steelblue' for n in names]
    ax5.barh(names, scores, color=colors, alpha=0.7)
    ax5.set_xlabel('Brier Score (lower = better calibration)')
    ax5.set_title('Probability Calibration Quality', fontweight='bold')

    # 6. Confusion Matrix (Best Model)
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

    summary_data = []
    for name, data in results.items():
        holdout = holdout_results[name]
        ci = ci_results[ci_results['Model'] == name].iloc[0]

        summary_data.append({
            'Model': name,
            'CV_Accuracy': f"{data['cv_mean']:.4f} +/- {data['cv_std']:.4f}",
            '95%_CI': f"[{ci['CI_Lower']:.4f}, {ci['CI_Upper']:.4f}]",
            'Holdout_Acc': f"{holdout['accuracy']:.4f}",
            'Brier': f"{holdout['brier_score']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))

    # Statistical conclusions
    print("\n" + "-" * 70)
    print("Statistical Conclusions")
    print("-" * 70)

    sig_improvements = comparison_df[
        (comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)
    ]

    if len(sig_improvements) > 0:
        print("\nModels with significant improvement over baseline:")
        for _, row in sig_improvements.iterrows():
            print(f"  - {row['Model']}: +{row['Mean_Diff']:.4f} (p={row['p_value']:.4f})")
    else:
        print("\nNo model showed statistically significant improvement over baseline.")

    # Best calibration
    best_calib = min(holdout_results.items(), key=lambda x: x[1]['brier_score'])
    print(f"\nBest calibration: {best_calib[0]} (Brier={best_calib[1]['brier_score']:.4f})")

    return summary_df


# ============================================================================
# INTERPRETATION
# ============================================================================
def write_interpretation(comparison_df, results, holdout_results):
    """Provide interpretation of results."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    sig_improvements = comparison_df[
        (comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)
    ]

    if len(sig_improvements) > 0:
        best = sig_improvements.loc[sig_improvements['Mean_Diff'].idxmax()]
        print(f"""
FINDINGS: {best['Model']} shows statistically significant improvement
over the baseline RBF-SVM (p={best['p_value']:.4f}).

Key observations:
1. SVM-RFE provides principled feature selection using SVM weights
2. Stacked SVM learns optimal kernel combination through meta-learning
3. Calibrated SVM improves probability estimates (Platt scaling)
4. Cost-Sensitive SVM handles class imbalance

The improvement of {best['Mean_Diff']:.4f} in CV accuracy suggests
that the advanced SVM method provides meaningful benefit.
""")
    else:
        print("""
FINDINGS: No advanced SVM method shows statistically significant
improvement over the baseline RBF-SVM.

Key observations:
1. The baseline RBF-SVM with proper tuning is highly competitive
2. Additional complexity does not guarantee better performance
3. The dataset may not require advanced techniques

However, note that:
- Calibrated SVM may still provide better probability estimates
- Cost-Sensitive SVM handles class imbalance more appropriately
- SVM-RFE provides interpretable feature selection
""")

    # Calibration analysis
    baseline_brier = holdout_results['RBF-SVM (Baseline)']['brier_score']
    calibrated_brier = holdout_results['Calibrated SVM']['brier_score']

    if calibrated_brier < baseline_brier:
        improvement = (baseline_brier - calibrated_brier) / baseline_brier * 100
        print(f"\nCalibration improvement: {improvement:.1f}% reduction in Brier score")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""
    # Load data
    X, y, feature_names, class_weights = load_data()

    # SVM-RFE feature analysis
    feature_ranking, optimal_n, optimal_features, rfe_cv_scores = svm_rfe_analysis(
        X, y, feature_names
    )

    # Create models
    models = create_models(
        n_features=X.shape[1],
        optimal_n_features=optimal_n,
        class_weights=class_weights
    )

    # Train and evaluate
    results, cv_scores_all = train_and_evaluate(X, y, models)

    # Statistical comparison
    comparison_df = statistical_comparison(cv_scores_all)

    # Confidence intervals
    ci_results = compute_confidence_intervals(cv_scores_all)

    # Holdout evaluation
    holdout_results, y_test = evaluate_holdout(X, y, results)

    # Visualizations
    create_visualizations(
        cv_scores_all, ci_results, holdout_results,
        comparison_df, feature_ranking, rfe_cv_scores
    )

    # Summary
    summary_df = generate_summary(results, comparison_df, ci_results, holdout_results)

    # Save results
    summary_df.to_csv('model_comparison_results.csv', index=False)
    comparison_df.to_csv('statistical_comparison.csv', index=False)
    feature_ranking.to_csv('svm_rfe_feature_ranking.csv', index=False)
    print("\nSaved: model_comparison_results.csv")
    print("Saved: statistical_comparison.csv")
    print("Saved: svm_rfe_feature_ranking.csv")

    # Interpretation
    write_interpretation(comparison_df, results, holdout_results)

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

    return results, comparison_df, summary_df


if __name__ == "__main__":
    results, comparison_df, summary_df = main()
    plt.show()
