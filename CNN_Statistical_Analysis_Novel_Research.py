"""
================================================================================
NOVEL RESEARCH CODE: Uncertainty-Aware Multi-Scale CNN with Statistical
Hypothesis Testing and Explainable AI (UA-MSCNN-SHE)

Title: "A Novel Uncertainty-Aware Multi-Scale Convolutional Neural Network
        with Integrated Statistical Hypothesis Testing and Explainability
        for Medical Image Classification"

Author: [Your Name]
Date: 2025

Publication Target: Q1 Journals (Nature Scientific Reports, IEEE TNNLS,
                    Pattern Recognition, Expert Systems with Applications)

Novel Contributions:
1. Multi-Scale Feature Pyramid with Learnable Fusion Weights
2. Monte Carlo Dropout for Bayesian Uncertainty Quantification
3. Statistical Hypothesis Testing Framework for Model Comparison
4. Integrated SHAP + Grad-CAM Explainability Pipeline
5. Bootstrap Confidence Intervals for Performance Metrics

Dataset: MedMNIST - PathMNIST (Colorectal Cancer Histology)
         Direct Download: https://zenodo.org/records/10519652

Requirements: pip install tensorflow numpy scipy scikit-learn matplotlib
              seaborn shap pandas tqdm medmnist statsmodels pingouin

COMPATIBLE WITH: TensorFlow 2.20.0 / Keras 3.x
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, mannwhitneyu
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Statistical Analysis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# Progress Bar
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("UA-MSCNN-SHE: Uncertainty-Aware Multi-Scale CNN with Statistical")
print("              Hypothesis Testing and Explainability")
print("=" * 80)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ============================================================================
# SECTION 2: DATA LOADING - MedMNIST PathMNIST Dataset
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Loading MedMNIST PathMNIST Dataset")
print("=" * 80)

def load_medmnist_pathmnist():
    """
    Load PathMNIST dataset from MedMNIST collection.
    PathMNIST: Colorectal Cancer Histology - 9 classes

    Dataset Info:
    - 107,180 training images
    - 10,004 validation images
    - 7,180 test images
    - 28x28 RGB images
    - 9 classes (different tissue types)
    """
    try:
        # Try loading from medmnist package
        import medmnist
        from medmnist import PathMNIST

        print("Loading PathMNIST from medmnist package...")

        # Download and load data
        train_dataset = PathMNIST(split='train', download=True, size=28)
        val_dataset = PathMNIST(split='val', download=True, size=28)
        test_dataset = PathMNIST(split='test', download=True, size=28)

        # Extract numpy arrays
        X_train = train_dataset.imgs
        y_train = train_dataset.labels.squeeze()
        X_val = val_dataset.imgs
        y_val = val_dataset.labels.squeeze()
        X_test = test_dataset.imgs
        y_test = test_dataset.labels.squeeze()

        print(f"✓ Data loaded successfully from medmnist package")

    except ImportError:
        print("medmnist package not found. Installing and loading directly from URL...")

        # Direct download from Zenodo
        import urllib.request
        import os

        # Create data directory
        data_dir = "./medmnist_data"
        os.makedirs(data_dir, exist_ok=True)

        # Download PathMNIST
        url = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
        filepath = os.path.join(data_dir, "pathmnist.npz")

        if not os.path.exists(filepath):
            print(f"Downloading PathMNIST from {url}...")
            urllib.request.urlretrieve(url, filepath)
            print("✓ Download complete!")

        # Load the data
        data = np.load(filepath)
        X_train = data['train_images']
        y_train = data['train_labels'].squeeze()
        X_val = data['val_images']
        y_val = data['val_labels'].squeeze()
        X_test = data['test_images']
        y_test = data['test_labels'].squeeze()

        print(f"✓ Data loaded successfully from Zenodo")

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Image shape: {X_train.shape[1:]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")

    # Class names for PathMNIST
    class_names = [
        'Adipose', 'Background', 'Debris', 'Lymphocytes',
        'Mucus', 'Smooth Muscle', 'Normal Colon Mucosa',
        'Cancer-Associated Stroma', 'Colorectal Adenocarcinoma'
    ]

    return X_train, y_train, X_val, y_val, X_test, y_test, class_names

# Load data
X_train, y_train, X_val, y_val, X_test, y_test, class_names = load_medmnist_pathmnist()
n_classes = len(class_names)

# ============================================================================
# SECTION 3: NOVEL MULTI-SCALE ATTENTION CNN ARCHITECTURE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Building Novel UA-MSCNN Architecture")
print("=" * 80)

class ChannelAttention(layers.Layer):
    """
    Novel Channel Attention Module with Learnable Temperature Scaling
    """
    def __init__(self, reduction_ratio=8, temperature=1.0, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.temperature = temperature

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation=None)
        self.temp = self.add_weight(
            name='temperature',
            shape=(1,),
            initializer=tf.constant_initializer(self.temperature),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # Global average pooling
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        # Global max pooling
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        # Shared MLP
        avg_out = self.fc2(self.fc1(avg_pool))
        max_out = self.fc2(self.fc1(max_pool))

        # Temperature-scaled attention
        attention = tf.nn.sigmoid((avg_out + max_out) / self.temp)

        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'temperature': self.temperature
        })
        return config

class SpatialAttention(layers.Layer):
    """
    Novel Spatial Attention Module with Multi-Scale Receptive Fields
    Compatible with TensorFlow 2.20 / Keras 3.x
    """
    def __init__(self, kernel_sizes=[3, 5, 7], **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

    def build(self, input_shape):
        # Create convolutions for each kernel size
        self.conv_3 = layers.Conv2D(1, 3, padding='same', activation=None, name='spatial_conv_3')
        self.conv_5 = layers.Conv2D(1, 5, padding='same', activation=None, name='spatial_conv_5')
        self.conv_7 = layers.Conv2D(1, 7, padding='same', activation=None, name='spatial_conv_7')

        self.fusion_weights = self.add_weight(
            name='fusion_weights',
            shape=(3,),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # Channel-wise statistics
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Multi-scale convolutions
        out_3 = self.conv_3(concat)
        out_5 = self.conv_5(concat)
        out_7 = self.conv_7(concat)

        # Learnable fusion using TensorFlow operations (Keras 3.x compatible)
        weights = tf.nn.softmax(self.fusion_weights)

        # Stack and weighted sum
        stacked = tf.stack([out_3, out_5, out_7], axis=-1)  # [B, H, W, 1, 3]
        weights_expanded = tf.reshape(weights, [1, 1, 1, 1, 3])
        attention = tf.reduce_sum(stacked * weights_expanded, axis=-1)  # [B, H, W, 1]

        attention = tf.nn.sigmoid(attention)

        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_sizes': self.kernel_sizes
        })
        return config

class MultiScaleFeaturePyramid(layers.Layer):
    """
    Novel Multi-Scale Feature Pyramid with Learnable Fusion
    Compatible with TensorFlow 2.20 / Keras 3.x
    """
    def __init__(self, filters, **kwargs):
        super(MultiScaleFeaturePyramid, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # Multi-scale convolutions
        self.conv1x1 = layers.Conv2D(self.filters, 1, padding='same', activation='relu')
        self.conv3x3 = layers.Conv2D(self.filters, 3, padding='same', activation='relu')
        self.conv5x5 = layers.Conv2D(self.filters, 5, padding='same', activation='relu')

        # Dilated convolutions for larger receptive fields
        self.conv3x3_d2 = layers.Conv2D(self.filters, 3, padding='same',
                                         dilation_rate=2, activation='relu')

        # Learnable fusion weights
        self.fusion_weights = self.add_weight(
            name='pyramid_weights',
            shape=(4,),
            initializer='ones',
            trainable=True
        )

        # Batch normalization
        self.bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Extract multi-scale features
        f1 = self.conv1x1(inputs)
        f3 = self.conv3x3(inputs)
        f5 = self.conv5x5(inputs)
        fd = self.conv3x3_d2(inputs)

        # Learnable weighted fusion using TensorFlow operations
        weights = tf.nn.softmax(self.fusion_weights)

        # Stack and weighted sum
        stacked = tf.stack([f1, f3, f5, fd], axis=-1)  # [B, H, W, C, 4]
        weights_expanded = tf.reshape(weights, [1, 1, 1, 1, 4])
        fused = tf.reduce_sum(stacked * weights_expanded, axis=-1)  # [B, H, W, C]

        return self.bn(fused, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters
        })
        return config

class MonteCarloDropout(layers.Layer):
    """
    Monte Carlo Dropout Layer - Active during both training and inference
    for Bayesian uncertainty estimation
    """
    def __init__(self, rate, **kwargs):
        super(MonteCarloDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        # Always apply dropout for MC sampling
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            'rate': self.rate
        })
        return config

def build_ua_mscnn(input_shape, n_classes, dropout_rate=0.3):
    """
    Build Uncertainty-Aware Multi-Scale CNN (UA-MSCNN)

    Novel Architecture Features:
    1. Multi-Scale Feature Pyramid with learnable fusion
    2. Channel and Spatial Attention modules
    3. Monte Carlo Dropout for uncertainty estimation
    4. Skip connections with feature refinement
    """
    inputs = layers.Input(shape=input_shape)

    # Initial Feature Extraction
    x = layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)

    # Stage 1: Multi-Scale Feature Pyramid
    x = MultiScaleFeaturePyramid(64)(x)
    x = ChannelAttention(reduction_ratio=8)(x)
    x = SpatialAttention(kernel_sizes=[3, 5, 7])(x)
    x = layers.MaxPooling2D(2)(x)
    x = MonteCarloDropout(dropout_rate)(x)
    skip1 = x

    # Stage 2
    x = MultiScaleFeaturePyramid(128)(x)
    x = ChannelAttention(reduction_ratio=8)(x)
    x = SpatialAttention(kernel_sizes=[3, 5, 7])(x)
    x = layers.MaxPooling2D(2)(x)
    x = MonteCarloDropout(dropout_rate)(x)
    skip2 = x

    # Stage 3
    x = MultiScaleFeaturePyramid(256)(x)
    x = ChannelAttention(reduction_ratio=16)(x)
    x = SpatialAttention(kernel_sizes=[3, 5, 7])(x)
    x = MonteCarloDropout(dropout_rate)(x)

    # Feature Aggregation with Skip Connections
    skip1_down = layers.MaxPooling2D(2)(skip1)
    skip1_proj = layers.Conv2D(256, 1, padding='same')(skip1_down)
    skip2_proj = layers.Conv2D(256, 1, padding='same')(skip2)

    x = layers.Add()([x, skip1_proj, skip2_proj])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Global Feature Extraction
    x = layers.GlobalAveragePooling2D()(x)

    # Classification Head with MC Dropout
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = MonteCarloDropout(dropout_rate)(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = MonteCarloDropout(dropout_rate)(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='UA_MSCNN')

    return model

# Build the model
model = build_ua_mscnn(X_train.shape[1:], n_classes, dropout_rate=0.3)
model.summary()

# Compile with label smoothing for better generalization
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# ============================================================================
# SECTION 4: TRAINING WITH ADVANCED CALLBACKS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Training UA-MSCNN Model")
print("=" * 80)

# Convert labels to categorical
y_train_cat = to_categorical(y_train, n_classes)
y_val_cat = to_categorical(y_val, n_classes)
y_test_cat = to_categorical(y_test, n_classes)

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'ua_mscnn_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# SECTION 5: MONTE CARLO DROPOUT UNCERTAINTY QUANTIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Bayesian Uncertainty Quantification (MC Dropout)")
print("=" * 80)

def mc_dropout_predict(model, X, n_samples=100):
    """
    Perform Monte Carlo Dropout inference for uncertainty estimation.

    Returns:
    - mean_predictions: Average predictions across MC samples
    - predictive_uncertainty: Total uncertainty (epistemic + aleatoric)
    - epistemic_uncertainty: Model uncertainty (reducible with more data)
    - aleatoric_uncertainty: Data uncertainty (irreducible)
    """
    predictions = []

    print(f"Performing {n_samples} MC Dropout forward passes...")
    for i in tqdm(range(n_samples)):
        # Forward pass with dropout active
        pred = model(X, training=True)
        predictions.append(pred.numpy())

    predictions = np.array(predictions)  # Shape: (n_samples, n_test, n_classes)

    # Mean prediction
    mean_predictions = np.mean(predictions, axis=0)

    # Predictive entropy (total uncertainty)
    predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-10), axis=-1)

    # Expected entropy (aleatoric uncertainty)
    individual_entropies = -np.sum(predictions * np.log(predictions + 1e-10), axis=-1)
    expected_entropy = np.mean(individual_entropies, axis=0)

    # Mutual information (epistemic uncertainty)
    epistemic_uncertainty = predictive_entropy - expected_entropy

    # Prediction variance
    prediction_variance = np.var(predictions, axis=0)

    return {
        'mean_predictions': mean_predictions,
        'prediction_std': np.std(predictions, axis=0),
        'predictive_entropy': predictive_entropy,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': expected_entropy,
        'prediction_variance': prediction_variance,
        'all_predictions': predictions
    }

# Perform MC Dropout inference on test set
mc_results = mc_dropout_predict(model, X_test, n_samples=100)

# Get predictions and uncertainties
y_pred_proba = mc_results['mean_predictions']
y_pred = np.argmax(y_pred_proba, axis=1)
uncertainty = mc_results['predictive_entropy']
epistemic = mc_results['epistemic_uncertainty']
aleatoric = mc_results['aleatoric_uncertainty']

print(f"\nUncertainty Statistics:")
print(f"  Mean Predictive Entropy: {np.mean(uncertainty):.4f} ± {np.std(uncertainty):.4f}")
print(f"  Mean Epistemic Uncertainty: {np.mean(epistemic):.4f} ± {np.std(epistemic):.4f}")
print(f"  Mean Aleatoric Uncertainty: {np.mean(aleatoric):.4f} ± {np.std(aleatoric):.4f}")

# ============================================================================
# SECTION 6: STATISTICAL HYPOTHESIS TESTING FRAMEWORK
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Statistical Hypothesis Testing Framework")
print("=" * 80)

class StatisticalTestingFramework:
    """
    Comprehensive statistical testing framework for model evaluation
    """

    def __init__(self, y_true, y_pred, y_pred_proba, class_names):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class_names = class_names
        self.n_classes = len(class_names)

    def bootstrap_confidence_interval(self, metric_func, n_bootstrap=1000,
                                       confidence_level=0.95):
        """
        Compute bootstrap confidence intervals for any metric
        """
        n_samples = len(self.y_true)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = self.y_true[indices]
            y_pred_boot = self.y_pred[indices]

            # Compute metric
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except:
                continue

        bootstrap_scores = np.array(bootstrap_scores)

        # Compute confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_scores, alpha/2 * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        mean = np.mean(bootstrap_scores)

        return mean, lower, upper, bootstrap_scores

    def compute_all_metrics_with_ci(self, n_bootstrap=1000):
        """
        Compute all classification metrics with bootstrap confidence intervals
        """
        metrics = {}

        # Accuracy
        mean, lower, upper, _ = self.bootstrap_confidence_interval(
            accuracy_score, n_bootstrap
        )
        metrics['Accuracy'] = {'mean': mean, 'lower': lower, 'upper': upper}

        # Per-class metrics
        for avg in ['macro', 'weighted']:
            # Precision
            mean, lower, upper, _ = self.bootstrap_confidence_interval(
                lambda y_t, y_p: precision_score(y_t, y_p, average=avg, zero_division=0),
                n_bootstrap
            )
            metrics[f'Precision ({avg})'] = {'mean': mean, 'lower': lower, 'upper': upper}

            # Recall
            mean, lower, upper, _ = self.bootstrap_confidence_interval(
                lambda y_t, y_p: recall_score(y_t, y_p, average=avg, zero_division=0),
                n_bootstrap
            )
            metrics[f'Recall ({avg})'] = {'mean': mean, 'lower': lower, 'upper': upper}

            # F1-Score
            mean, lower, upper, _ = self.bootstrap_confidence_interval(
                lambda y_t, y_p: f1_score(y_t, y_p, average=avg, zero_division=0),
                n_bootstrap
            )
            metrics[f'F1-Score ({avg})'] = {'mean': mean, 'lower': lower, 'upper': upper}

        return metrics

    def mcnemar_test(self, y_pred_model1, y_pred_model2):
        """
        McNemar's test for comparing two classifiers
        """
        # Create contingency table
        correct1 = (y_pred_model1 == self.y_true)
        correct2 = (y_pred_model2 == self.y_true)

        # n01: model1 wrong, model2 correct
        n01 = np.sum(~correct1 & correct2)
        # n10: model1 correct, model2 wrong
        n10 = np.sum(correct1 & ~correct2)

        # McNemar's test statistic
        if n01 + n10 == 0:
            return 1.0, 0  # No difference

        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return p_value, statistic

    def cohens_kappa(self):
        """
        Compute Cohen's Kappa with confidence interval
        """
        from sklearn.metrics import cohen_kappa_score

        kappa = cohen_kappa_score(self.y_true, self.y_pred)

        # Bootstrap CI
        mean, lower, upper, _ = self.bootstrap_confidence_interval(
            lambda y_t, y_p: cohen_kappa_score(y_t, y_p),
            n_bootstrap=1000
        )

        return {'kappa': kappa, 'lower': lower, 'upper': upper}

    def per_class_statistical_analysis(self):
        """
        Statistical analysis for each class
        """
        results = {}
        y_true_bin = label_binarize(self.y_true, classes=range(self.n_classes))
        y_pred_bin = label_binarize(self.y_pred, classes=range(self.n_classes))

        for i, class_name in enumerate(self.class_names):
            class_true = y_true_bin[:, i]
            class_pred = y_pred_bin[:, i]
            class_proba = self.y_pred_proba[:, i]

            # Sensitivity (Recall)
            tp = np.sum((class_true == 1) & (class_pred == 1))
            fn = np.sum((class_true == 1) & (class_pred == 0))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Specificity
            tn = np.sum((class_true == 0) & (class_pred == 0))
            fp = np.sum((class_true == 0) & (class_pred == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # AUC
            if len(np.unique(class_true)) > 1:
                auc_score = roc_auc_score(class_true, class_proba)
            else:
                auc_score = 0.5

            results[class_name] = {
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'AUC': auc_score,
                'Support': np.sum(class_true)
            }

        return results

# Initialize framework
stat_framework = StatisticalTestingFramework(y_test, y_pred, y_pred_proba, class_names)

# Compute metrics with confidence intervals
print("\nComputing metrics with bootstrap confidence intervals (1000 iterations)...")
metrics_ci = stat_framework.compute_all_metrics_with_ci(n_bootstrap=1000)

print("\nClassification Metrics with 95% Confidence Intervals:")
print("-" * 60)
for metric_name, values in metrics_ci.items():
    print(f"{metric_name:25s}: {values['mean']:.4f} [{values['lower']:.4f}, {values['upper']:.4f}]")

# Cohen's Kappa
kappa_results = stat_framework.cohens_kappa()
print(f"\nCohen's Kappa: {kappa_results['kappa']:.4f} [{kappa_results['lower']:.4f}, {kappa_results['upper']:.4f}]")

# Per-class analysis
print("\nPer-Class Statistical Analysis:")
print("-" * 80)
per_class_stats = stat_framework.per_class_statistical_analysis()
stats_df = pd.DataFrame(per_class_stats).T
print(stats_df.round(4))

# ============================================================================
# SECTION 7: CROSS-VALIDATION WITH STATISTICAL TESTING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: 5-Fold Cross-Validation with Statistical Testing")
print("=" * 80)

def cross_validation_with_stats(X, y, n_folds=5):
    """
    Perform k-fold cross-validation with statistical analysis
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }

    print(f"\nPerforming {n_folds}-fold cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_folds}")

        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Convert to categorical
        y_fold_train_cat = to_categorical(y_fold_train, n_classes)
        y_fold_val_cat = to_categorical(y_fold_val, n_classes)

        # Build and train model
        fold_model = build_ua_mscnn(X.shape[1:], n_classes, dropout_rate=0.3)
        fold_model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        fold_model.fit(
            X_fold_train, y_fold_train_cat,
            validation_data=(X_fold_val, y_fold_val_cat),
            epochs=30,
            batch_size=128,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=10,
                             restore_best_weights=True, verbose=0)
            ],
            verbose=0
        )

        # Evaluate
        y_fold_pred_proba = fold_model.predict(X_fold_val, verbose=0)
        y_fold_pred = np.argmax(y_fold_pred_proba, axis=1)

        # Store metrics
        fold_results['accuracy'].append(accuracy_score(y_fold_val, y_fold_pred))
        fold_results['precision'].append(precision_score(y_fold_val, y_fold_pred,
                                                          average='weighted', zero_division=0))
        fold_results['recall'].append(recall_score(y_fold_val, y_fold_pred,
                                                    average='weighted', zero_division=0))
        fold_results['f1'].append(f1_score(y_fold_val, y_fold_pred,
                                           average='weighted', zero_division=0))

        # AUC (multi-class)
        y_fold_val_bin = label_binarize(y_fold_val, classes=range(n_classes))
        fold_results['auc'].append(roc_auc_score(y_fold_val_bin, y_fold_pred_proba,
                                                  average='weighted', multi_class='ovr'))

        print(f"  Accuracy: {fold_results['accuracy'][-1]:.4f}")

        # Clear session to free memory
        keras.backend.clear_session()

    return fold_results

# Note: For faster execution, using combined train+val set for CV
X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)

# Subsample for faster CV (adjust as needed)
subsample_size = min(20000, len(X_combined))
indices = np.random.choice(len(X_combined), subsample_size, replace=False)
X_cv = X_combined[indices]
y_cv = y_combined[indices]

cv_results = cross_validation_with_stats(X_cv, y_cv, n_folds=5)

# Statistical analysis of CV results
print("\n" + "=" * 60)
print("Cross-Validation Results Summary")
print("=" * 60)

cv_summary = pd.DataFrame(cv_results)
print("\nFold-wise Results:")
print(cv_summary.round(4))

print("\nAggregated Results (Mean ± Std):")
for metric in cv_results.keys():
    values = cv_results[metric]
    print(f"  {metric.capitalize():12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Normality test (Shapiro-Wilk)
print("\nStatistical Tests on CV Results:")
for metric in cv_results.keys():
    values = cv_results[metric]
    stat, p_value = stats.shapiro(values)
    print(f"  Shapiro-Wilk test for {metric}: W={stat:.4f}, p={p_value:.4f}")

# ============================================================================
# SECTION 8: EXPLAINABLE AI - GRAD-CAM VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: Explainable AI - Grad-CAM Visualization")
print("=" * 80)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretation
    """
    def __init__(self, model, layer_name=None):
        self.model = model

        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break

        self.layer_name = layer_name
        print(f"Using layer: {layer_name}")

        # Create gradient model
        self.grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, image, class_idx=None):
        """
        Compute Grad-CAM heatmap for a single image
        """
        # Expand dimensions if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            class_output = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

        return heatmap.numpy()

    def visualize(self, image, class_idx=None, alpha=0.4):
        """
        Overlay Grad-CAM heatmap on image
        """
        try:
            import cv2
            use_cv2 = True
        except ImportError:
            use_cv2 = False

        heatmap = self.compute_heatmap(image, class_idx)

        # Resize heatmap to image size
        if use_cv2:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = heatmap.astype(np.float32) / 255.0
        else:
            # Fallback without OpenCV
            from scipy.ndimage import zoom
            zoom_factor = (image.shape[0] / heatmap.shape[0],
                          image.shape[1] / heatmap.shape[1])
            heatmap = zoom(heatmap, zoom_factor)
            # Create RGB heatmap using matplotlib colormap
            cmap = plt.cm.jet
            heatmap = cmap(heatmap)[:, :, :3]

        # Overlay
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        superimposed = heatmap * alpha + image * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 1)

        return superimposed, heatmap

# Initialize Grad-CAM
try:
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer:
        gradcam = GradCAM(model, last_conv_layer)

        # Generate Grad-CAM visualizations for sample images
        print("\nGenerating Grad-CAM visualizations...")

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        # Select samples from different classes
        for i in range(3):
            # Select a sample
            idx = np.where(y_test == i)[0][0]
            image = X_test[idx]
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred[idx]]
            conf = y_pred_proba[idx].max()
            uncert = uncertainty[idx]

            # Original image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Original\nTrue: {true_label[:10]}")
            axes[i, 0].axis('off')

            # Grad-CAM heatmap
            superimposed, heatmap = gradcam.visualize(image)

            axes[i, 1].imshow(heatmap)
            axes[i, 1].set_title("Grad-CAM Heatmap")
            axes[i, 1].axis('off')

            # Overlay
            axes[i, 2].imshow(superimposed)
            axes[i, 2].set_title(f"Overlay\nPred: {pred_label[:10]}")
            axes[i, 2].axis('off')

            # Uncertainty visualization
            axes[i, 3].bar(['Epistemic', 'Aleatoric'],
                           [epistemic[idx], aleatoric[idx]],
                           color=['blue', 'orange'])
            axes[i, 3].set_title(f"Uncertainty\nConf: {conf:.3f}")
            axes[i, 3].set_ylim(0, max(epistemic.max(), aleatoric.max()) * 1.2)

        plt.tight_layout()
        plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Grad-CAM visualization saved to 'gradcam_visualization.png'")
    else:
        print("No convolutional layer found for Grad-CAM")
except Exception as e:
    print(f"Grad-CAM visualization error: {e}")

# ============================================================================
# SECTION 9: SHAP EXPLAINABILITY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: SHAP Explainability Analysis")
print("=" * 80)

try:
    import shap

    print("Initializing SHAP DeepExplainer...")

    # Use a subset for SHAP (computationally expensive)
    background = X_train[np.random.choice(len(X_train), 100, replace=False)]
    test_samples = X_test[:50]

    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, background)

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(test_samples)

    # Plot SHAP summary
    print("Generating SHAP visualizations...")

    # For multi-class, shap_values is a list
    if isinstance(shap_values, list):
        # Combine SHAP values across classes
        shap_combined = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_combined = np.abs(shap_values)

    # Mean absolute SHAP value per pixel location
    mean_shap = shap_combined.mean(axis=0)

    # Visualize mean importance
    plt.figure(figsize=(10, 8))

    if len(mean_shap.shape) == 3:
        # Color image
        plt.imshow(mean_shap.mean(axis=-1), cmap='hot')
    else:
        plt.imshow(mean_shap, cmap='hot')

    plt.colorbar(label='Mean |SHAP value|')
    plt.title('Mean SHAP Feature Importance Across Test Samples')
    plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ SHAP importance map saved to 'shap_importance.png'")

except ImportError:
    print("SHAP not installed. Install with: pip install shap")
except Exception as e:
    print(f"SHAP analysis error: {e}")

# ============================================================================
# SECTION 10: COMPREHENSIVE RESULTS VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: Comprehensive Results Visualization")
print("=" * 80)

# Create comprehensive figure
fig = plt.figure(figsize=(20, 16))

# 1. Training History
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Confusion Matrix
ax2 = fig.add_subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
            xticklabels=[c[:8] for c in class_names],
            yticklabels=[c[:8] for c in class_names])
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
ax2.set_title('Normalized Confusion Matrix')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# 3. ROC Curves
ax3 = fig.add_subplot(2, 3, 3)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

for i in range(min(5, n_classes)):  # Plot first 5 classes
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, label=f'{class_names[i][:10]} (AUC={roc_auc:.3f})')

ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves (Selected Classes)')
ax3.legend(loc='lower right', fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Uncertainty Distribution
ax4 = fig.add_subplot(2, 3, 4)
correct_mask = (y_pred == y_test)
ax4.hist(uncertainty[correct_mask], bins=50, alpha=0.7, label='Correct', color='green')
ax4.hist(uncertainty[~correct_mask], bins=50, alpha=0.7, label='Incorrect', color='red')
ax4.set_xlabel('Predictive Entropy')
ax4.set_ylabel('Count')
ax4.set_title('Uncertainty Distribution: Correct vs Incorrect')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Confidence vs Accuracy Calibration
ax5 = fig.add_subplot(2, 3, 5)
confidences = np.max(y_pred_proba, axis=1)
bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(confidences, bins) - 1
bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

bin_accuracies = []
bin_confidences = []
bin_counts = []

for i in range(len(bins) - 1):
    mask = bin_indices == i
    if mask.sum() > 0:
        bin_accuracies.append(correct_mask[mask].mean())
        bin_confidences.append(confidences[mask].mean())
        bin_counts.append(mask.sum())
    else:
        bin_accuracies.append(0)
        bin_confidences.append((bins[i] + bins[i+1]) / 2)
        bin_counts.append(0)

ax5.bar(range(len(bin_accuracies)), bin_accuracies, alpha=0.7, label='Accuracy')
ax5.plot(range(len(bin_confidences)), bin_confidences, 'r-o', label='Confidence')
ax5.plot([0, len(bins)], [0, 1], 'k--', label='Perfect Calibration')
ax5.set_xlabel('Confidence Bin')
ax5.set_ylabel('Accuracy / Confidence')
ax5.set_title('Confidence Calibration Plot')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Per-Class Performance
ax6 = fig.add_subplot(2, 3, 6)
class_metrics = []
for i, name in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (y_pred[mask] == y_test[mask]).mean()
        class_metrics.append((name[:15], acc))

class_metrics.sort(key=lambda x: x[1])
names, accs = zip(*class_metrics)
colors = plt.cm.RdYlGn(np.array(accs))
ax6.barh(range(len(names)), accs, color=colors)
ax6.set_yticks(range(len(names)))
ax6.set_yticklabels(names)
ax6.set_xlabel('Accuracy')
ax6.set_title('Per-Class Accuracy')
ax6.set_xlim(0, 1)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Comprehensive results saved to 'comprehensive_results.png'")

# ============================================================================
# SECTION 11: STATISTICAL COMPARISON WITH BASELINE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: Statistical Comparison with Baseline Models")
print("=" * 80)

def build_baseline_cnn(input_shape, n_classes):
    """Simple baseline CNN without novel components"""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def build_resnet_style(input_shape, n_classes):
    """ResNet-style baseline with skip connections"""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Residual block 1
    skip = x
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.Add()([x, skip])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Residual block 2
    skip = layers.Conv2D(128, 1, padding='same')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.Add()([x, skip])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return Model(inputs, outputs)

print("Training baseline models for comparison...")

# Use a subset for faster comparison
X_sub_train = X_train[:10000]
y_sub_train = y_train[:10000]
y_sub_train_cat = to_categorical(y_sub_train, n_classes)

# Train Baseline CNN
print("\n1. Training Simple Baseline CNN...")
baseline_cnn = build_baseline_cnn(X_train.shape[1:], n_classes)
baseline_cnn.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
baseline_cnn.fit(X_sub_train, y_sub_train_cat, epochs=20, batch_size=128,
                  validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True, verbose=0)])
baseline_pred = np.argmax(baseline_cnn.predict(X_test, verbose=0), axis=1)
baseline_acc = accuracy_score(y_test, baseline_pred)
print(f"   Accuracy: {baseline_acc:.4f}")

# Train ResNet-style
print("\n2. Training ResNet-style CNN...")
resnet_model = build_resnet_style(X_train.shape[1:], n_classes)
resnet_model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
resnet_model.fit(X_sub_train, y_sub_train_cat, epochs=20, batch_size=128,
                  validation_split=0.1, verbose=0,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True, verbose=0)])
resnet_pred = np.argmax(resnet_model.predict(X_test, verbose=0), axis=1)
resnet_acc = accuracy_score(y_test, resnet_pred)
print(f"   Accuracy: {resnet_acc:.4f}")

# Our UA-MSCNN
ua_mscnn_acc = accuracy_score(y_test, y_pred)
print(f"\n3. UA-MSCNN (Proposed): {ua_mscnn_acc:.4f}")

# Statistical Comparison
print("\n" + "-" * 60)
print("Statistical Comparison (McNemar's Test):")
print("-" * 60)

# McNemar's test: UA-MSCNN vs Baseline
p_val_baseline, stat_baseline = stat_framework.mcnemar_test(y_pred, baseline_pred)
print(f"UA-MSCNN vs Baseline CNN: χ²={stat_baseline:.4f}, p={p_val_baseline:.4f}")
print(f"  → {'Significant difference (p<0.05)' if p_val_baseline < 0.05 else 'No significant difference'}")

# McNemar's test: UA-MSCNN vs ResNet
p_val_resnet, stat_resnet = stat_framework.mcnemar_test(y_pred, resnet_pred)
print(f"UA-MSCNN vs ResNet-style: χ²={stat_resnet:.4f}, p={p_val_resnet:.4f}")
print(f"  → {'Significant difference (p<0.05)' if p_val_resnet < 0.05 else 'No significant difference'}")

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Compute per-sample correctness for effect size
ua_correct = (y_pred == y_test).astype(float)
baseline_correct = (baseline_pred == y_test).astype(float)
resnet_correct = (resnet_pred == y_test).astype(float)

print("\nEffect Size (Cohen's d):")
print(f"  UA-MSCNN vs Baseline: d = {cohens_d(ua_correct, baseline_correct):.4f}")
print(f"  UA-MSCNN vs ResNet: d = {cohens_d(ua_correct, resnet_correct):.4f}")

# Summary table
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Baseline CNN', 'ResNet-style', 'UA-MSCNN (Proposed)'],
    'Accuracy': [baseline_acc, resnet_acc, ua_mscnn_acc],
    'F1-Score (weighted)': [
        f1_score(y_test, baseline_pred, average='weighted'),
        f1_score(y_test, resnet_pred, average='weighted'),
        f1_score(y_test, y_pred, average='weighted')
    ],
    'Has Uncertainty': ['No', 'No', 'Yes'],
    'Explainable': ['Limited', 'Limited', 'Full (SHAP+GradCAM)']
})

print(comparison_df.to_string(index=False))

# ============================================================================
# SECTION 12: FINAL RESULTS AND PUBLICATION-READY SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 12: Publication-Ready Results Summary")
print("=" * 80)

# Compute final metrics
final_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
    'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
    'F1-Score (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
    'Precision (weighted)': precision_score(y_test, y_pred, average='weighted', zero_division=0),
    'Recall (weighted)': recall_score(y_test, y_pred, average='weighted', zero_division=0),
    'F1-Score (weighted)': f1_score(y_test, y_pred, average='weighted', zero_division=0),
}

# Multi-class AUC
y_test_bin = label_binarize(y_test, classes=range(n_classes))
final_metrics['AUC (weighted OvR)'] = roc_auc_score(y_test_bin, y_pred_proba,
                                                      average='weighted', multi_class='ovr')

print("\n┌" + "─" * 58 + "┐")
print("│" + " FINAL PERFORMANCE METRICS ".center(58) + "│")
print("├" + "─" * 58 + "┤")

for metric, value in final_metrics.items():
    ci = metrics_ci.get(metric, {})
    if ci:
        print(f"│ {metric:30s} │ {value:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] │")
    else:
        print(f"│ {metric:30s} │ {value:.4f}                    │")

print("├" + "─" * 58 + "┤")
print("│" + " UNCERTAINTY METRICS ".center(58) + "│")
print("├" + "─" * 58 + "┤")
print(f"│ {'Mean Predictive Entropy':30s} │ {np.mean(uncertainty):.4f} ± {np.std(uncertainty):.4f}          │")
print(f"│ {'Mean Epistemic Uncertainty':30s} │ {np.mean(epistemic):.4f} ± {np.std(epistemic):.4f}          │")
print(f"│ {'Mean Aleatoric Uncertainty':30s} │ {np.mean(aleatoric):.4f} ± {np.std(aleatoric):.4f}          │")
print("└" + "─" * 58 + "┘")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=[c[:20] for c in class_names]))

# Save results to CSV
results_df = pd.DataFrame({
    'Metric': list(final_metrics.keys()),
    'Value': list(final_metrics.values())
})
results_df.to_csv('final_results.csv', index=False)
print("\n✓ Results saved to 'final_results.csv'")

# Save model
model.save('ua_mscnn_final.keras')
print("✓ Model saved to 'ua_mscnn_final.keras'")

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)
print("""
Generated Files:
1. ua_mscnn_best.keras      - Best model checkpoint
2. ua_mscnn_final.keras     - Final trained model
3. comprehensive_results.png - Visualization of all results
4. gradcam_visualization.png - Grad-CAM explainability maps
5. shap_importance.png      - SHAP feature importance
6. final_results.csv        - Numerical results for publication

Novel Contributions for Q1 Journal Publication:
1. Multi-Scale Feature Pyramid with Learnable Fusion Weights
2. Temperature-Scaled Channel Attention Mechanism
3. Monte Carlo Dropout for Bayesian Uncertainty Quantification
4. Comprehensive Statistical Hypothesis Testing Framework
5. Integrated Explainability (SHAP + Grad-CAM)
6. Bootstrap Confidence Intervals for All Metrics

Suggested Journal Targets:
- Nature Scientific Reports (IF: 4.6)
- IEEE Transactions on Neural Networks and Learning Systems (IF: 14.2)
- Pattern Recognition (IF: 8.0)
- Expert Systems with Applications (IF: 8.5)
- Computers in Biology and Medicine (IF: 7.7)
- Artificial Intelligence in Medicine (IF: 7.5)
""")
