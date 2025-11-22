# User Guide - Medicinal Plants & ML Analysis for Journal Publication

## Quick Start (For Spyder IDE)

### Step 1: Open Spyder IDE

Launch Spyder IDE on your system.

### Step 2: Set Working Directory

In Spyder's console or file explorer, navigate to this project directory:

```python
import os
os.chdir('/path/to/Parthasarathy-Sundararajan')
# Verify
print(os.getcwd())
```

### Step 3: Install Dependencies

Open Spyder's console and run:

```python
!pip install -r requirements.txt
```

Or use the terminal:
```bash
pip install -r requirements.txt
```

### Step 4: Quick Validation Test (Recommended)

Before running the full analysis, run a quick test:

```python
# In Spyder console or run the script
exec(open('quick_test.py').read())
```

This will:
- Use a smaller dataset (2,000 samples instead of 25,000)
- Run faster (2-5 minutes instead of 10-30 minutes)
- Verify everything works correctly

### Step 5: Run Full Analysis

Once the test passes, run the complete analysis:

**Option A: Run main script directly**
```python
exec(open('medicinal_plants_diabetes_ml_analysis.py').read())
```

**Option B: Run integrated analysis with literature comparison**
```python
exec(open('run_complete_analysis.py').read())
```

**Option C: Open and run in Spyder editor**
1. Open `run_complete_analysis.py` in Spyder
2. Press F5 or click the green "Run" button

### Step 6: View Results

After completion, check the generated files:

**Results folder** (`./results/`):
- `predictions_with_uncertainty.csv` - Detailed predictions
- `analysis_summary.txt` - Summary report

**Figures folder** (`./figures/`):
- All high-quality PNG images (300 DPI) for publication

## Expected Runtime

- **Quick test**: 2-5 minutes
- **Full analysis**: 10-30 minutes (depending on hardware)
- **With literature comparison**: +2 minutes

## Viewing Results in Spyder

### Method 1: Using Spyder's Variable Explorer

After running the analysis:

```python
# The analysis returns results
results_df, ensemble_model, smote_results = main()

# View in Variable Explorer (right panel in Spyder)
# Click on 'results_df' to see the DataFrame
```

### Method 2: Load CSV Results

```python
import pandas as pd

# Load predictions
predictions = pd.read_csv('./results/predictions_with_uncertainty.csv')
print(predictions.head())

# View summary
with open('./results/analysis_summary.txt', 'r') as f:
    print(f.read())
```

### Method 3: Display Figures

```python
from PIL import Image
import matplotlib.pyplot as plt

# Display any figure
img = Image.open('./figures/roc_curves.png')
plt.figure(figsize=(12, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## Customization Options

### Change Dataset Size

Edit `medicinal_plants_diabetes_ml_analysis.py`, line ~180:

```python
n_samples = 25000  # Change to desired size (minimum: 1000)
```

### Change Model Parameters

Edit the model definitions around line ~400:

```python
# XGBoost
self.models['xgboost'] = xgb.XGBClassifier(
    n_estimators=300,  # Increase for better performance (slower)
    max_depth=6,       # Increase for more complex models
    learning_rate=0.05 # Decrease for more careful learning
)
```

### Change Confidence Level for Conformal Prediction

Edit line ~750:

```python
conformal = ConformalPredictor(alpha=0.1)  # 0.1 = 90% confidence
# alpha=0.05 for 95% confidence
# alpha=0.2 for 80% confidence
```

### Change Visualization DPI

Edit lines ~50-60:

```python
plt.rcParams['figure.dpi'] = 300  # Change to 600 for higher quality
plt.rcParams['savefig.dpi'] = 300
```

## Troubleshooting

### Issue 1: Import Errors

```python
# Check if package is installed
import sys
!{sys.executable} -m pip list | grep xgboost

# Reinstall if needed
!{sys.executable} -m pip install xgboost --upgrade
```

### Issue 2: Memory Error

Reduce dataset size or model complexity:

```python
# In the script, modify:
n_samples = 10000  # Instead of 25000
n_estimators = 100  # Instead of 300
```

### Issue 3: SHAP Errors

SHAP can be slow or error-prone. To disable:

Comment out lines ~850-860:

```python
# try:
#     viz.plot_feature_importance_shap(...)
# except Exception as e:
#     print(f"⚠ SHAP visualization skipped: {e}")
```

### Issue 4: Figures Not Displaying

```python
# Add this at the end of the script
import matplotlib.pyplot as plt
plt.show()  # Display all figures
```

## Interpreting Results

### Key Metrics

**AUC-ROC** (Area Under ROC Curve):
- 0.5: Random classifier (useless)
- 0.7-0.8: Acceptable
- 0.8-0.9: Excellent  ← **Target for publication**
- 0.9-1.0: Outstanding

**Accuracy**: Overall correctness
**Precision**: Positive prediction reliability
**Recall**: Sensitivity (true positive rate)
**F1-Score**: Harmonic mean of precision and recall

### Uncertainty Metrics

**Epistemic Uncertainty**: Model's knowledge uncertainty
- High = Models disagree (uncertain)
- Low = Models agree (confident)

**Aleatoric Uncertainty**: Data inherent uncertainty
- High = Difficult case to predict
- Low = Clear case

### Confidence-Stratified Performance

Check `analysis_summary.txt` for:
```
Confidence >= 0.9: n=1234, AUC=0.95, Acc=0.89
```

This means: For high-confidence predictions (≥0.9), performance is excellent!

### Literature Comparison

After running `run_complete_analysis.py`, check the output:

```
✓✓✓ OUR RESULTS EXCEED STATE-OF-THE-ART! ✓✓✓
```

Or:

```
✓ RESULTS ARE COMPETITIVE WITH PUBLISHED WORK
```

## Exporting for Publication

### Figures

All figures in `./figures/` are publication-ready:
- 300 DPI (suitable for journals)
- PNG format
- High-quality rendering

To convert to PDF or EPS:

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load and save as PDF
for fig_file in ['roc_curves.png', 'uncertainty_analysis.png']:
    img = Image.open(f'./figures/{fig_file}')
    img.save(f'./figures/{fig_file.replace(".png", ".pdf")}', 'PDF')
```

### Tables

Create publication tables:

```python
import pandas as pd

# Load results
results = pd.read_csv('./results/predictions_with_uncertainty.csv')

# Performance summary table
from sklearn.metrics import classification_report

print(classification_report(
    results['y_true'],
    results['y_pred_binary'],
    target_names=['Poor Control', 'Good Control']
))

# Export to LaTeX
report = classification_report(
    results['y_true'],
    results['y_pred_binary'],
    output_dict=True
)
df = pd.DataFrame(report).transpose()
print(df.to_latex())
```

### Statistical Tests

```python
# Compare models statistically
from scipy import stats

# Load individual model predictions (you'll need to save these separately)
# Example: DeLong test for AUC comparison

# Bootstrap confidence intervals are already in analysis_summary.txt
```

## Advanced Usage

### Running Specific Components Only

```python
# Import modules
from medicinal_plants_diabetes_ml_analysis import (
    DataAcquisition, UncertaintyAwareEnsemble,
    ConformalPredictor, SMOTEComparison, Visualizer
)

# Run only data generation
data_acq = DataAcquisition()
df = data_acq.download_diabetes_data()

# Run only SMOTE comparison
smote_comp = SMOTEComparison()
results = smote_comp.compare(X_train, y_train, X_test, y_test, base_model)
```

### Custom Medicinal Plant Dosages

Edit the `medicinal_plants` dictionary in `_load_alternative_dataset()`:

```python
medicinal_plants = {
    'Gymnema_sylvestre': (0.15, 0.25),  # (min_dose, max_dose) in g/day
    'Your_Plant': (min_dose, max_dose),  # Add new plant
}
```

### Using Your Own Dataset

```python
# Load your dataset
import pandas as pd
df = pd.read_csv('your_dataset.csv')

# Ensure it has:
# - Clinical features (glucose, HbA1c, etc.)
# - Medicinal plant dosages (optional)
# - Binary target variable

# Then skip the data acquisition step and proceed with preprocessing
```

## FAQ

**Q: Can I run this on Windows/Mac/Linux?**
A: Yes, all code is cross-platform.

**Q: Do I need a GPU?**
A: No, but it will speed up deep learning training.

**Q: How much RAM is required?**
A: 8GB minimum, 16GB recommended.

**Q: Can I use this code for my own research?**
A: Yes, for academic and research purposes.

**Q: How do I cite this work?**
A: See README_ML.md for citation information.

**Q: Can I modify the code?**
A: Yes, the code is designed to be extensible.

**Q: What if my results are worse than literature?**
A: The novel methodology (uncertainty quantification, conformal prediction) is still publishable even if raw performance is similar.

**Q: How do I run this overnight/in background?**
A: Use `nohup python run_complete_analysis.py > output.log 2>&1 &` in terminal.

## Tips for Best Results

1. **Use the full dataset** (25,000 samples) for final submission
2. **Run multiple times** with different random seeds to verify stability
3. **Check uncertainty calibration** - high-confidence predictions should be accurate
4. **Include all figures** in your manuscript
5. **Highlight novel contributions** in discussion
6. **Compare with literature** thoroughly
7. **Document any modifications** you make

## Next Steps After Running Analysis

1. ✓ Review all generated figures
2. ✓ Read `analysis_summary.txt` carefully
3. ✓ Check literature comparison results
4. ✓ Write manuscript introduction and methods
5. ✓ Create results section from figures and tables
6. ✓ Write discussion highlighting novel contributions
7. ✓ Select target journal based on results
8. ✓ Submit!

## Support

For issues or questions:
1. Check this guide
2. Check README_ML.md
3. Review error messages carefully
4. Check package versions: `pip list`
5. Try the quick_test.py first

## Success Checklist

Before submission, verify:

- [ ] All code runs without errors
- [ ] All figures generated (8 PNG files)
- [ ] Results better than or competitive with literature
- [ ] AUC > 0.85 (target for high-impact journals)
- [ ] All novel contributions implemented and tested
- [ ] Statistical validation complete (bootstrap, cross-validation)
- [ ] Uncertainty quantification working correctly
- [ ] Conformal prediction calibrated properly
- [ ] Literature comparison favorable
- [ ] Manuscript draft complete

---

**Good luck with your publication!** 🎓📊🏆
