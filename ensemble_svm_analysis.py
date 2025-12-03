"""
Improved SVM Methods for Medical Diagnosis:
SMOTE Resampling, Bayesian Optimization, and Multiple Kernel Learning

This script implements advanced techniques to improve SVM performance:
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. Bayesian Hyperparameter Optimization (Optuna)
3. Multiple Kernel Learning (MKL) - weighted kernel combination
4. Polynomial Feature Interactions
5. Nystroem Kernel Approximation

Dataset: Cleveland Heart Disease Database (UCI Repository)
Samples: 270 patients | Features: 13 clinical attributes

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
    cross_val_predict
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


# ============================================================================
# SMOTE IMPLEMENTATION (No external dependency)
# ============================================================================
class SimpleSMOTE:
    """
    Synthetic Minority Over-sampling Technique (SMOTE).

    Creates synthetic samples for the minority class by interpolating
    between existing minority samples and their nearest neighbors.

    This addresses class imbalance without simply duplicating samples.
    """

    def __init__(self, sampling_ratio=1.0, k_neighbors=5, random_state=42):
        self.sampling_ratio = sampling_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y)

        # Find minority and majority classes
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        X_minority = X[minority_indices]
        n_minority = len(X_minority)
        n_majority = len(majority_indices)

        # Calculate how many synthetic samples to generate
        n_synthetic = int((n_majority - n_minority) * self.sampling_ratio)

        if n_synthetic <= 0:
            return X, y

        # Generate synthetic samples
        synthetic_samples = []

        for _ in range(n_synthetic):
            # Pick a random minority sample
            idx = np.random.randint(0, n_minority)
            sample = X_minority[idx]

            # Find k nearest neighbors within minority class
            distances = np.linalg.norm(X_minority - sample, axis=1)
            neighbor_indices = np.argsort(distances)[1:self.k_neighbors+1]

            # Pick a random neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[neighbor_idx]

            # Generate synthetic sample along the line between sample and neighbor
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)

        synthetic_samples = np.array(synthetic_samples)
        synthetic_labels = np.full(n_synthetic, minority_class)

        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.concatenate([y, synthetic_labels])

        # Shuffle
        indices = np.random.permutation(len(y_resampled))

        return X_resampled[indices], y_resampled[indices]


# ============================================================================
# MULTIPLE KERNEL LEARNING (MKL)
# ============================================================================
class MultipleKernelSVM(BaseEstimator, ClassifierMixin):
    """
    Multiple Kernel Learning (MKL) SVM.

    Learns optimal combination of multiple kernels:
    K_combined = Σ μ_k * K_k(x, x')

    where μ_k are learned kernel weights.

    This goes beyond simple voting by optimizing kernel combination.
    """

    def __init__(self, C=1.0, random_state=42):
        self.C = C
        self.random_state = random_state
        self.kernel_weights_ = None
        self.svm_ = None
        self.X_train_ = None

    def _compute_kernels(self, X1, X2=None):
        """Compute individual kernel matrices."""
        if X2 is None:
            X2 = X1

        X1 = np.array(X1)
        X2 = np.array(X2)

        # Linear kernel
        K_linear = X1 @ X2.T

        # RBF kernel (gamma = 1/n_features)
        gamma = 1.0 / X1.shape[1]
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                   np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        K_rbf = np.exp(-gamma * sq_dists)

        # Polynomial kernel (degree 2)
        K_poly = (1 + X1 @ X2.T) ** 2

        return [K_linear, K_rbf, K_poly]

    def _combined_kernel(self, X1, X2=None):
        """Compute weighted combination of kernels."""
        kernels = self._compute_kernels(X1, X2)
        K_combined = sum(w * K for w, K in zip(self.kernel_weights_, kernels))
        return K_combined

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train_ = X
        self.classes_ = np.unique(y)

        # Initialize kernel weights uniformly
        n_kernels = 3
        self.kernel_weights_ = np.ones(n_kernels) / n_kernels

        # Simple optimization: try different weight combinations
        best_score = 0
        best_weights = self.kernel_weights_.copy()

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        # Grid search over kernel weights
        for w1 in np.linspace(0.1, 0.8, 5):
            for w2 in np.linspace(0.1, 0.8, 5):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue

                weights = np.array([w1, w2, w3])
                self.kernel_weights_ = weights

                # Cross-validate
                scores = []
                for train_idx, val_idx in cv.split(X, y):
                    K_train = self._combined_kernel(X[train_idx])
                    K_val = self._combined_kernel(X[val_idx], X[train_idx])

                    svm = SVC(kernel='precomputed', C=self.C,
                             random_state=self.random_state)
                    svm.fit(K_train, y[train_idx])
                    score = svm.score(K_val, y[val_idx])
                    scores.append(score)

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_weights = weights.copy()

        self.kernel_weights_ = best_weights

        # Fit final model
        K_train = self._combined_kernel(X)
        self.svm_ = SVC(kernel='precomputed', C=self.C, probability=True,
                       random_state=self.random_state)
        self.svm_.fit(K_train, y)

        return self

    def predict(self, X):
        K_test = self._combined_kernel(np.array(X), self.X_train_)
        return self.svm_.predict(K_test)

    def predict_proba(self, X):
        K_test = self._combined_kernel(np.array(X), self.X_train_)
        return self.svm_.predict_proba(K_test)


# ============================================================================
# BAYESIAN-INSPIRED HYPERPARAMETER OPTIMIZATION
# ============================================================================
class BayesianOptimizedSVM(BaseEstimator, ClassifierMixin):
    """
    SVM with Bayesian-inspired hyperparameter optimization.

    Uses random search with adaptive sampling to find optimal
    hyperparameters more efficiently than grid search.
    """

    def __init__(self, n_iter=50, random_state=42):
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.svm_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        best_score = 0
        best_params = {'C': 1.0, 'gamma': 'scale'}

        # Parameter space
        C_space = np.logspace(-3, 3, 20)
        gamma_space = np.logspace(-4, 1, 20)

        # Random search with exploitation
        for i in range(self.n_iter):
            if i < self.n_iter // 2:
                # Exploration phase
                C = np.random.choice(C_space)
                gamma = np.random.choice(gamma_space)
            else:
                # Exploitation phase - sample near best
                C = best_params['C'] * np.random.uniform(0.5, 2.0)
                gamma = best_params['gamma'] if best_params['gamma'] == 'scale' else \
                        best_params['gamma'] * np.random.uniform(0.5, 2.0)

            try:
                svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=self.random_state)
                scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'C': C, 'gamma': gamma}
            except:
                continue

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.classes_ = np.unique(y)

        # Fit final model
        self.svm_ = SVC(kernel='rbf', probability=True, random_state=self.random_state,
                       **self.best_params_)
        self.svm_.fit(X, y)

        return self

    def predict(self, X):
        return self.svm_.predict(X)

    def predict_proba(self, X):
        return self.svm_.predict_proba(X)


# ============================================================================
# NYSTROEM KERNEL APPROXIMATION SVM
# ============================================================================
class NystroemSVM(BaseEstimator, ClassifierMixin):
    """
    SVM with Nystroem kernel approximation.

    Approximates the RBF kernel using a subset of training samples,
    enabling linear SVM in the approximated feature space.

    This can capture non-linear patterns while being more scalable.
    """

    def __init__(self, n_components=100, gamma='scale', C=1.0, random_state=42):
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.random_state = random_state
        self.nystroem_ = None
        self.svm_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        n_components = min(self.n_components, X.shape[0])

        gamma = self.gamma
        if gamma == 'scale':
            gamma = 1.0 / (X.shape[1] * X.var())

        self.nystroem_ = Nystroem(kernel='rbf', gamma=gamma,
                                  n_components=n_components,
                                  random_state=self.random_state)
        X_transformed = self.nystroem_.fit_transform(X)

        self.svm_ = SVC(kernel='linear', C=self.C, probability=True,
                       random_state=self.random_state)
        self.svm_.fit(X_transformed, y)

        return self

    def predict(self, X):
        X_transformed = self.nystroem_.transform(X)
        return self.svm_.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.nystroem_.transform(X)
        return self.svm_.predict_proba(X_transformed)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load Cleveland Heart Disease dataset."""
    print("=" * 70)
    print("IMPROVED SVM METHODS FOR MEDICAL DIAGNOSIS")
    print("Cleveland Heart Disease Dataset (UCI Repository)")
    print("=" * 70)

    heart = fetch_openml(data_id=53, as_frame=True, parser='auto')
    X = heart.data
    y = heart.target

    if y.dtype == 'object' or str(y.dtype) == 'category':
        unique_vals = y.unique()
        print(f"  Target values: {list(unique_vals)}")
        y = pd.Categorical(y).codes

    feature_names = X.columns.tolist()
    X = X.apply(pd.to_numeric, errors='coerce')

    if X.isnull().any().any():
        X = X.fillna(X.median())

    y = np.array(y)
    X = np.array(X)

    class_counts = np.bincount(y)
    print(f"\nDataset characteristics:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: No Disease={class_counts[0]}, Disease={class_counts[1]}")
    print(f"  Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")

    return X, y, feature_names


# ============================================================================
# CREATE MODELS
# ============================================================================
def create_models():
    """Create improved SVM model variants."""

    models = {}

    # 1. Baseline RBF-SVM
    models['RBF-SVM (Baseline)'] = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                   random_state=RANDOM_STATE))
    ])

    # 2. SMOTE + SVM (addresses class imbalance)
    models['SMOTE + SVM'] = 'smote_svm'  # Special handling

    # 3. Polynomial Features + SVM
    models['PolyFeatures + SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True,
                                    include_bias=False)),
        ('scaler2', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                   random_state=RANDOM_STATE))
    ])

    # 4. Bayesian Optimized SVM
    models['Bayesian-Opt SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', BayesianOptimizedSVM(n_iter=30, random_state=RANDOM_STATE))
    ])

    # 5. Multiple Kernel Learning
    models['MKL-SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('mkl', MultipleKernelSVM(C=1.0, random_state=RANDOM_STATE))
    ])

    # 6. Nystroem Approximation SVM
    models['Nystroem-SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('nystroem_svm', NystroemSVM(n_components=100, C=1.0,
                                     random_state=RANDOM_STATE))
    ])

    return models


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_and_evaluate(X, y, models, cv_folds=10):
    """Train and evaluate models with cross-validation."""
    print("\n" + "-" * 70)
    print("Model Training and Evaluation")
    print("-" * 70)

    outer_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                random_state=RANDOM_STATE)

    results = {}
    cv_scores_all = {}
    smote = SimpleSMOTE(sampling_ratio=1.0, k_neighbors=5, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\nTraining: {name}")

        if name == 'SMOTE + SVM':
            # Special handling for SMOTE
            scores = []
            for train_idx, val_idx in outer_cv.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                # Apply SMOTE only to training data
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                # Scale and train
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_resampled)
                X_val_scaled = scaler.transform(X_val)

                svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                         random_state=RANDOM_STATE)
                svm.fit(X_train_scaled, y_train_resampled)
                score = svm.score(X_val_scaled, y_val)
                scores.append(score)

            cv_scores = np.array(scores)
            best_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                           random_state=RANDOM_STATE))
            ])
            # Fit on SMOTE-resampled data
            X_resampled, y_resampled = smote.fit_resample(X, y)
            best_model.fit(X_resampled, y_resampled)
        else:
            cv_scores = cross_val_score(model, X, y, cv=outer_cv, scoring='accuracy')
            best_model = clone(model).fit(X, y)

        cv_scores_all[name] = cv_scores
        results[name] = {
            'model': best_model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return results, cv_scores_all


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================
def statistical_comparison(cv_scores_all, baseline_name='RBF-SVM (Baseline)'):
    """Perform paired statistical tests."""
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

        sig = "**" if p_value < 0.05 else ""
        direction = "BETTER" if diff > 0 else "worse"

        print(f"\n{name}:")
        print(f"  CV Accuracy: {scores.mean():.4f}")
        print(f"  Difference: {diff:+.4f} ({direction}) {sig}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05 and diff > 0:
            print(f"  => STATISTICALLY SIGNIFICANT IMPROVEMENT!")

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
# HOLDOUT EVALUATION
# ============================================================================
def evaluate_holdout(X, y, results):
    """Evaluate on holdout test set."""
    print("\n" + "-" * 70)
    print("Holdout Test Set Evaluation")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    holdout_results = {}
    smote = SimpleSMOTE(sampling_ratio=1.0, k_neighbors=5, random_state=RANDOM_STATE)

    for name, data in results.items():
        if name == 'SMOTE + SVM':
            # Apply SMOTE to training data
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(kernel='rbf', C=1, gamma='scale', probability=True,
                       random_state=RANDOM_STATE)
            model.fit(X_train_scaled, y_train_resampled)

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model = clone(data['model'])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        holdout_results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'brier_score': brier_score_loss(y_test, y_proba),
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
def create_visualizations(cv_scores_all, ci_results, holdout_results, comparison_df):
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
    ax1.axhline(y=cv_scores_all['RBF-SVM (Baseline)'].mean(), color='red',
                linestyle='--', alpha=0.7, label='Baseline')
    ax1.legend()

    # 2. Improvement over Baseline
    ax2 = plt.subplot(2, 3, 2)
    if len(comparison_df) > 0:
        colors = ['green' if d > 0 else 'red' for d in comparison_df['Mean_Diff']]
        bars = ax2.barh(comparison_df['Model'], comparison_df['Mean_Diff'] * 100,
                        color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Improvement over Baseline (%)')
        ax2.set_title('Performance Improvement', fontweight='bold')

        # Mark significant improvements
        for i, (idx, row) in enumerate(comparison_df.iterrows()):
            if row['p_value'] < 0.05 and row['Mean_Diff'] > 0:
                ax2.annotate('*', (row['Mean_Diff']*100 + 0.2, i), fontsize=16, fontweight='bold')

    # 3. Confidence Intervals
    ax3 = plt.subplot(2, 3, 3)
    models = ci_results['Model'].values
    means = ci_results['Mean'].values
    ci_lower = ci_results['CI_Lower'].values
    ci_upper = ci_results['CI_Upper'].values

    y_pos = np.arange(len(models))
    xerr = np.array([means - ci_lower, ci_upper - means])

    colors = ['green' if m > means[0] else 'steelblue' for m in means]
    ax3.barh(y_pos, means, xerr=xerr, capsize=5, alpha=0.7, color=colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(models)
    ax3.set_xlabel('Accuracy')
    ax3.set_title('95% Bootstrap Confidence Intervals', fontweight='bold')
    ax3.axvline(x=means[0], color='red', linestyle='--', alpha=0.7)

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

    # 5. P-values
    ax5 = plt.subplot(2, 3, 5)
    if len(comparison_df) > 0:
        colors = ['green' if p < 0.05 else 'gray' for p in comparison_df['p_value']]
        ax5.barh(comparison_df['Model'], comparison_df['p_value'], color=colors, alpha=0.7)
        ax5.axvline(x=0.05, color='red', linestyle='--', label='p=0.05 threshold')
        ax5.set_xlabel('p-value')
        ax5.set_title('Statistical Significance', fontweight='bold')
        ax5.legend()

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
            'F1': f"{holdout['f1']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))

    # Highlight improvements
    print("\n" + "-" * 70)
    print("Key Findings")
    print("-" * 70)

    sig_improvements = comparison_df[
        (comparison_df['p_value'] < 0.05) & (comparison_df['Mean_Diff'] > 0)
    ]

    if len(sig_improvements) > 0:
        print("\n*** STATISTICALLY SIGNIFICANT IMPROVEMENTS FOUND ***")
        for _, row in sig_improvements.iterrows():
            print(f"  - {row['Model']}: +{row['Mean_Diff']*100:.2f}% (p={row['p_value']:.4f})")
    else:
        # Find best performing even if not significant
        best = comparison_df.loc[comparison_df['Mean_Diff'].idxmax()]
        print(f"\nBest improvement: {best['Model']} (+{best['Mean_Diff']*100:.2f}%)")
        print(f"p-value: {best['p_value']:.4f}")
        if best['p_value'] < 0.1:
            print("Note: Approaching significance (p < 0.10)")

    return summary_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main execution function."""
    # Load data
    X, y, feature_names = load_data()

    # Create models
    models = create_models()

    # Train and evaluate
    results, cv_scores_all = train_and_evaluate(X, y, models)

    # Statistical comparison
    comparison_df = statistical_comparison(cv_scores_all)

    # Confidence intervals
    ci_results = compute_confidence_intervals(cv_scores_all)

    # Holdout evaluation
    holdout_results, y_test = evaluate_holdout(X, y, results)

    # Visualizations
    create_visualizations(cv_scores_all, ci_results, holdout_results, comparison_df)

    # Summary
    summary_df = generate_summary(results, comparison_df, ci_results, holdout_results)

    # Save results
    summary_df.to_csv('model_comparison_results.csv', index=False)
    comparison_df.to_csv('statistical_comparison.csv', index=False)
    print("\nSaved: model_comparison_results.csv")
    print("Saved: statistical_comparison.csv")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

    return results, comparison_df, summary_df


if __name__ == "__main__":
    results, comparison_df, summary_df = main()
    plt.show()
