# ARTEMIS: Adaptive Resource and Temporal Emergency Management Intelligent System

**Q1 Journal Quality Implementation** with real-time MQTT integration for emergency dispatch optimization.

## Quick Start (Clean Environment)

### Option 1: Conda (Recommended)
```bash
# Create new environment
conda env create -f environment.yml

# Activate
conda activate artemis

# Run
python artemis_q1_enhanced.py
```

### Option 2: pip
```bash
# Create new environment
conda create -n artemis python=3.10
conda activate artemis

# Install dependencies
pip install -r requirements.txt

# Run
python artemis_q1_enhanced.py
```

### Option 3: Manual Installation
```bash
conda create -n artemis python=3.10
conda activate artemis

pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2
pip install scikit-learn==1.3.0 scipy==1.11.1 requests==2.31.0 statsmodels==0.14.0
pip install xgboost==2.0.0 lightgbm==4.1.0 catboost==1.2
pip install paho-mqtt==1.6.1

python artemis_q1_enhanced.py
```

## Features

### Machine Learning Models
| Category | Models |
|----------|--------|
| Linear | Ridge, Lasso, ElasticNet |
| Tree-based | Random Forest, Extra Trees, AdaBoost |
| Gradient Boosting | **XGBoost**, **LightGBM**, **CatBoost**, Sklearn GB |
| Deep Learning | LSTM, GRU, Transformer (optional) |

> **Note**: XGBoost/LightGBM/CatBoost often outperform deep learning on tabular data!

### Statistical Rigor (Q1 Quality)
- Walk-forward cross-validation (proper time-series validation)
- Bootstrap 95% confidence intervals
- Paired statistical tests (Wilcoxon signed-rank, paired t-test)
- Effect size (Cohen's d)
- Ablation studies

### Real Public Data Sources
| City | Dataset | Source |
|------|---------|--------|
| San Francisco | Fire Department Calls | data.sfgov.org |
| New York City | EMS Incidents | data.cityofnewyork.us |
| Seattle | Fire 911 Calls | data.seattle.gov |
| Chicago | Fire Incidents | data.cityofchicago.org |
| Los Angeles | Fire Incidents | data.lacity.org |
| Austin | Fire Incidents | data.austintexas.gov |
| Boston | Fire Incidents | data.boston.gov |

**Total: 50,000+ real records - NO synthetic data**

## Output Files

```
artemis_q1_outputs/
├── results_table.csv           # Publication-ready results
├── ablation_results.csv        # Feature importance
├── fig_model_comparison.png    # Model comparison with CI
├── fig_data_summary.png        # Dataset visualization
├── fig_statistical.png         # Statistical significance
```

## Expected Results

With XGBoost/LightGBM on real emergency dispatch data:
- **R² Score**: 0.45 - 0.65
- **MAE**: 2.5 - 4.0 minutes
- **MAPE**: 25 - 40%

These are realistic results for response time prediction using only temporal features.

## Troubleshooting

### TensorFlow Issues
If TensorFlow causes problems, the code will automatically use XGBoost/LightGBM instead.
These gradient boosting methods often perform better than deep learning on tabular data.

### Memory Issues
Reduce data size:
```python
city_data = downloader.download_all_available(target_records=20000)
```

## Citation

```bibtex
@article{artemis2024,
  title={ARTEMIS: Adaptive Resource and Temporal Emergency Management Intelligent System},
  author={Department of Mathematics, SRM Institute of Science and Technology},
  year={2024}
}
```

## Author

Department of Mathematics, SRM Institute of Science and Technology
