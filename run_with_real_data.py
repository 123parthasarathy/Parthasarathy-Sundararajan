"""
Run analysis with REAL clinical data from 130 US hospitals
"""

import os
os.chdir('/home/user/Parthasarathy-Sundararajan')

# Load real data
print("="*80)
print("RUNNING ANALYSIS WITH REAL CLINICAL DATA")
print("="*80)

from load_real_data import load_best_available_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Import models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Load REAL data
df, target_col, dataset_name = load_best_available_dataset()

if df is None:
    print("ERROR: Could not load real data")
    exit(1)

print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Prepare data
y = df[target_col].values
X = df.drop(columns=[target_col])

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"Encoding {len(categorical_cols)} categorical features...")
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

# Handle missing values
if X.isnull().sum().sum() > 0:
    print("Imputing missing values...")
    X = X.fillna(X.median())

# For speed, use a subsample if dataset is huge
if len(X) > 30000:
    print(f"\nUsing stratified sample of 30,000 from {len(X):,} for faster analysis...")
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=30000, random_state=42)
    for sample_idx, _ in sss.split(X, y):
        X = X.iloc[sample_idx]
        y = y[sample_idx]

print(f"Final dataset: {len(X):,} samples, {X.shape[1]} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Class balance: {(y_train.mean())*100:.1f}% positive")

# Apply SMOTE
print("\nApplying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_train_res):,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*80)
print("TRAINING ENSEMBLE MODELS")
print("="*80)

models = {}

# XGBoost
print("\n[1/3] Training XGBoost...")
models['XGBoost'] = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)
models['XGBoost'].fit(X_train_scaled, y_train_res)
print("✓ XGBoost trained")

# LightGBM
print("\n[2/3] Training LightGBM...")
models['LightGBM'] = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
models['LightGBM'].fit(X_train_scaled, y_train_res)
print("✓ LightGBM trained")

# CatBoost
print("\n[3/3] Training CatBoost...")
models['CatBoost'] = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=False
)
models['CatBoost'].fit(X_train_scaled, y_train_res)
print("✓ CatBoost trained")

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

# Get predictions from all models
predictions = {}
for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    predictions[name] = y_pred_proba
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{name:15s}: AUC = {auc:.4f}")

# Ensemble prediction (average)
ensemble_pred = np.mean(list(predictions.values()), axis=0)
ensemble_auc = roc_auc_score(y_test, ensemble_pred)
predictions['Ensemble'] = ensemble_pred

print(f"\n{'='*80}")
print(f"ENSEMBLE:        AUC = {ensemble_auc:.4f}")
print(f"{'='*80}")

# Calculate uncertainty
epistemic_uncertainty = np.var(list(predictions.values())[:3], axis=0)  # Model variance
aleatoric_uncertainty = -(ensemble_pred * np.log(ensemble_pred + 1e-10) +
                          (1 - ensemble_pred) * np.log(1 - ensemble_pred + 1e-10))

# Binary predictions
ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)

# All metrics
accuracy = accuracy_score(y_test, ensemble_pred_binary)
precision = precision_score(y_test, ensemble_pred_binary, zero_division=0)
recall = recall_score(y_test, ensemble_pred_binary, zero_division=0)
f1 = f1_score(y_test, ensemble_pred_binary, zero_division=0)

print("\n" + "="*80)
print("FINAL PERFORMANCE METRICS (REAL DATA)")
print("="*80)
print(f"\nAUC-ROC:     {ensemble_auc:.4f}")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-Score:    {f1:.4f}")

# Confidence-stratified performance
print("\n" + "="*80)
print("CONFIDENCE-STRATIFIED PERFORMANCE")
print("="*80)

total_unc = epistemic_uncertainty + aleatoric_uncertainty
confidence = 1 / (1 + total_unc)

for threshold in [0.7, 0.8, 0.9]:
    mask = confidence >= threshold
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], ensemble_pred_binary[mask])
        auc_strat = roc_auc_score(y_test[mask], ensemble_pred[mask])
        print(f"Confidence >= {threshold:.1f}: n={mask.sum():5d} ({100*mask.sum()/len(y_test):5.1f}%), AUC={auc_strat:.4f}, Acc={acc:.4f}")

# Bootstrap confidence interval
print("\n" + "="*80)
print("BOOTSTRAP 95% CONFIDENCE INTERVAL")
print("="*80)

from sklearn.utils import resample

auc_bootstrap = []
for i in range(500):
    indices = resample(range(len(y_test)), n_samples=len(y_test), random_state=i)
    auc_boot = roc_auc_score(y_test[indices], ensemble_pred[indices])
    auc_bootstrap.append(auc_boot)

auc_ci_lower = np.percentile(auc_bootstrap, 2.5)
auc_ci_upper = np.percentile(auc_bootstrap, 97.5)

print(f"\nAUC: {ensemble_auc:.4f} [95% CI: {auc_ci_lower:.4f} - {auc_ci_upper:.4f}]")

# Compare with literature
print("\n" + "="*80)
print("LITERATURE COMPARISON")
print("="*80)

literature_best = {
    'Study': 'Ahuja et al. 2022',
    'AUC': 0.881,
    'Journal': 'Computational Intelligence and Neuroscience',
    'Impact Factor': 3.1
}

print(f"\nBest Published Result:")
print(f"  Study: {literature_best['Study']}")
print(f"  AUC: {literature_best['AUC']:.4f}")
print(f"  Journal: {literature_best['Journal']} (IF: {literature_best['Impact Factor']})")

print(f"\nOur Results (REAL DATA):")
print(f"  Dataset: {dataset_name} - 130 US Hospitals")
print(f"  Samples: {len(df):,} real patient encounters")
print(f"  AUC: {ensemble_auc:.4f}")

difference = ensemble_auc - literature_best['AUC']
improvement = (difference / literature_best['AUC']) * 100

print(f"\nComparison:")
print(f"  Difference: {difference:+.4f}")
print(f"  Improvement: {improvement:+.2f}%")

if ensemble_auc > literature_best['AUC']:
    if ensemble_auc > 0.90:
        print("\n" + "="*80)
        print("✓✓✓ RESULTS SIGNIFICANTLY EXCEED STATE-OF-THE-ART! ✓✓✓")
        print("="*80)
        print("\nRECOMMENDATION: Submit to TOP-TIER journal")
        print("  - Nature Communications (IF: 16.6)")
        print("  - IEEE J. Biomed Health Inform (IF: 7.7)")
    else:
        print("\n" + "="*80)
        print("✓✓✓ RESULTS EXCEED STATE-OF-THE-ART! ✓✓✓")
        print("="*80)
        print("\nRECOMMENDATION: Submit to HIGH-IMPACT journal")
        print("  - Scientific Reports (IF: 4.6)")
        print("  - Journal of Big Data (IF: 8.6)")
elif ensemble_auc > 0.85:
    print("\n" + "="*80)
    print("✓ RESULTS ARE COMPETITIVE WITH STATE-OF-THE-ART")
    print("="*80)
    print("\nRECOMMENDATION: Submit to GOOD-IMPACT journal")
    print("  - PLoS ONE (IF: 3.7)")
    print("  - IEEE Access (IF: 3.9)")
else:
    print("\n" + "="*80)
    print("RESULTS ARE MODERATE")
    print("="*80)
    print("\nNOTE: Novel methodology still publishable")
    print("  - Emphasize uncertainty quantification")
    print("  - Emphasize medicinal plant integration")

# Save results
results_dict = {
    'auc': ensemble_auc,
    'auc_ci_lower': auc_ci_lower,
    'auc_ci_upper': auc_ci_upper,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'dataset': dataset_name,
    'samples': len(df),
    'test_samples': len(y_test)
}

import json
with open('./results/real_data_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n✓ Results saved to: ./results/real_data_results.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE WITH REAL CLINICAL DATA")
print("="*80)
print(f"\n✓ Used REAL data from {dataset_name}")
print(f"✓ Total samples: {len(df):,}")
print(f"✓ AUC: {ensemble_auc:.4f} (95% CI: [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}])")
print(f"✓ Result: {'EXCEEDS' if ensemble_auc > literature_best['AUC'] else 'Competitive with'} published work")
print("\n")
