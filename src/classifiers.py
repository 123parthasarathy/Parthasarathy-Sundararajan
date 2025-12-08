"""
Classifier Collection Module
Implements all baseline and advanced classifiers for comparison
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


def get_baseline_classifiers(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get dictionary of baseline classifiers.

    Args:
        random_state: Random seed

    Returns:
        Dictionary mapping classifier names to instances
    """
    classifiers = {
        # Linear models
        'Logistic Regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced'
        ),

        # Support Vector Machines
        'SVM-RBF': SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state,
            class_weight='balanced',
            C=1.0,
            gamma='scale'
        ),

        'SVM-Linear': SVC(
            kernel='linear',
            probability=True,
            random_state=random_state,
            class_weight='balanced',
            C=1.0
        ),

        # Tree-based methods
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state,
            class_weight='balanced',
            max_depth=10
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight='balanced',
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1
        ),

        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=5,
            learning_rate=0.1
        ),

        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            random_state=random_state,
            learning_rate=0.5
        ),

        # Instance-based
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        ),

        # Probabilistic
        'Naive Bayes': GaussianNB(),

        # Neural Networks
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            learning_rate_init=0.001
        ),
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=200,
            random_state=random_state,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,  # Will be adjusted based on class imbalance
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        classifiers['LightGBM'] = LGBMClassifier(
            n_estimators=200,
            random_state=random_state,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            n_jobs=-1,
            verbose=-1
        )

    return classifiers


def get_ensemble_classifiers(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get advanced ensemble classifiers.

    Args:
        random_state: Random seed

    Returns:
        Dictionary of ensemble classifiers
    """
    base_rf = RandomForestClassifier(
        n_estimators=50, random_state=random_state, class_weight='balanced'
    )
    base_svm = SVC(probability=True, random_state=random_state, class_weight='balanced')
    base_lr = LogisticRegression(random_state=random_state, max_iter=1000)

    classifiers = {
        'Bagging-RF': BaggingClassifier(
            estimator=clone(base_rf),
            n_estimators=10,
            random_state=random_state,
            n_jobs=-1
        ),

        'Voting-Hard': VotingClassifier(
            estimators=[
                ('rf', clone(base_rf)),
                ('svm', clone(base_svm)),
                ('lr', clone(base_lr))
            ],
            voting='hard'
        ),

        'Voting-Soft': VotingClassifier(
            estimators=[
                ('rf', clone(base_rf)),
                ('svm', clone(base_svm)),
                ('lr', clone(base_lr))
            ],
            voting='soft'
        ),
    }

    return classifiers


class HybridQuantumCausalClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid classifier combining Quantum-Inspired NN with Causal Learning
    and ensemble strategies for optimal performance.
    """

    def __init__(self, use_quantum: bool = True, use_causal: bool = True,
                 use_ensemble: bool = True, random_state: int = 42,
                 class_weight: Optional[Dict] = None):
        """
        Initialize hybrid classifier.

        Args:
            use_quantum: Include quantum-inspired component
            use_causal: Include causal learning component
            use_ensemble: Use ensemble combination
            random_state: Random seed
            class_weight: Class weights for imbalanced data
        """
        self.use_quantum = use_quantum
        self.use_causal = use_causal
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        self.class_weight = class_weight
        self.models = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HybridQuantumCausalClassifier':
        """Train the hybrid classifier."""
        from quantum_neural_network import QINNClassifier
        from causal_learning import CausalNeuralClassifier

        self.models = []

        if self.use_quantum:
            qinn = QINNClassifier(
                hidden_dims=[64, 32],
                n_quantum_states=3,
                n_epochs=80,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            qinn.fit(X, y)
            self.models.append(('QINN', qinn))

        if self.use_causal:
            causal = CausalNeuralClassifier(
                hidden_dims=[64, 32],
                causal_method='correlation',
                n_epochs=80,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            causal.fit(X, y)
            self.models.append(('Causal', causal))

        if self.use_ensemble or len(self.models) == 0:
            # Add a strong baseline for ensemble
            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            rf.fit(X, y)
            self.models.append(('RF', rf))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction probabilities."""
        if not self.models:
            raise ValueError("Model not fitted")

        # Collect predictions from all models
        all_proba = []
        for name, model in self.models:
            proba = model.predict_proba(X)
            all_proba.append(proba)

        # Average probabilities
        avg_proba = np.mean(all_proba, axis=0)
        return avg_proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


def get_all_classifiers(random_state: int = 42,
                        include_neural: bool = True,
                        include_ensemble: bool = True) -> Dict[str, BaseEstimator]:
    """
    Get all available classifiers for comparison.

    Args:
        random_state: Random seed
        include_neural: Include neural network classifiers
        include_ensemble: Include ensemble classifiers

    Returns:
        Dictionary of all classifiers
    """
    classifiers = get_baseline_classifiers(random_state)

    if include_ensemble:
        classifiers.update(get_ensemble_classifiers(random_state))

    return classifiers


# State-of-the-art benchmark results for comparison
SOTA_BENCHMARKS = {
    'WDBC': {
        # From literature review
        'Stacking Ensemble (2023)': {
            'accuracy': 0.9989,
            'AUC': 0.999,
            'source': 'PMC 2023'
        },
        'Deep CNN (ResearchGate)': {
            'accuracy': 0.9970,
            'AUC': 0.998,
            'source': 'ResearchGate'
        },
        'Shallow ANN (ScienceDirect 2021)': {
            'accuracy': 0.9947,
            'AUC': 0.997,
            'source': 'ScienceDirect 2021'
        },
        'SVM (MDPI 2023)': {
            'accuracy': 0.993,
            'AUC': 0.994,
            'source': 'MDPI 2023'
        },
        'Hybrid CNN+EfficientNetV2B3 (2024)': {
            'accuracy': 0.963,
            'recall': 0.864,
            'precision': 0.934,
            'AUC': 0.975,
            'source': 'PMC 2024'
        },
        'ResNet50+SMOTE (IEEE 2024)': {
            'accuracy': 0.95,
            'AUC': 0.97,
            'source': 'IEEE Xplore 2024'
        },
        'Random Forest (AUC)': {
            'AUC': 0.9751,
            'source': 'Network Modeling 2025'
        },
        'Q-GBGWO (Best Reported)': {
            'accuracy': 0.988,
            'AUC': 0.99,
            'source': 'Literature'
        },
    }
}


def compare_with_sota(results: Dict[str, Dict], dataset: str = 'WDBC') -> Dict:
    """
    Compare experimental results with state-of-the-art benchmarks.

    Args:
        results: Dictionary of experimental results
        dataset: Dataset name for SOTA comparison

    Returns:
        Comparison dictionary
    """
    sota = SOTA_BENCHMARKS.get(dataset, {})

    comparison = {
        'experimental': results,
        'sota': sota,
        'analysis': {}
    }

    # Find best experimental results
    best_accuracy = 0
    best_auc = 0
    best_model = None

    for model_name, metrics in results.items():
        acc = metrics.get('accuracy', {}).get('mean', 0) if isinstance(metrics.get('accuracy'), dict) else metrics.get('accuracy', 0)
        auc = metrics.get('AUC', {}).get('mean', 0) if isinstance(metrics.get('AUC'), dict) else metrics.get('AUC', 0)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model_name

        if auc > best_auc:
            best_auc = auc

    comparison['analysis']['best_accuracy'] = best_accuracy
    comparison['analysis']['best_auc'] = best_auc
    comparison['analysis']['best_model'] = best_model

    # Compare with SOTA
    sota_best_acc = max([v.get('accuracy', 0) for v in sota.values()])
    sota_best_auc = max([v.get('AUC', 0) for v in sota.values()])

    comparison['analysis']['sota_best_accuracy'] = sota_best_acc
    comparison['analysis']['sota_best_auc'] = sota_best_auc
    comparison['analysis']['accuracy_gap'] = sota_best_acc - best_accuracy
    comparison['analysis']['auc_gap'] = sota_best_auc - best_auc

    return comparison


if __name__ == "__main__":
    # Test classifiers
    print("Testing Classifiers...")

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Load data
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get classifiers
    classifiers = get_all_classifiers(random_state=42)

    print(f"\nTesting {len(classifiers)} classifiers...")
    results = {}

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            results[name] = {'accuracy': acc, 'AUC': auc}
            print(f"  {name}: Acc={acc:.4f}, AUC={auc:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Compare with SOTA
    comparison = compare_with_sota(results)
    print(f"\nBest experimental accuracy: {comparison['analysis']['best_accuracy']:.4f}")
    print(f"SOTA best accuracy: {comparison['analysis']['sota_best_accuracy']:.4f}")
    print(f"Gap: {comparison['analysis']['accuracy_gap']:.4f}")
