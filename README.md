# ARTEMIS: Adaptive Resource and Temporal Emergency Management Intelligent System

**Q1 Journal Quality Implementation** with PAHO MQTT real-time data integration for emergency dispatch optimization.

## Key Features

### Core Modules
- **STQM Module**: Spatial-Temporal Queuing Model for zone clustering
- **DLRP Module**: Response time prediction with ML and Deep Learning models
- **PRO Module**: Multi-objective resource optimization with Pareto front generation

### Deep Learning Models (TensorFlow/Keras)
- **Bidirectional LSTM**: With batch normalization and dropout regularization
- **Bidirectional GRU**: Gated Recurrent Units for sequence modeling
- **Transformer**: Multi-head attention architecture
- **CNN-LSTM Hybrid**: Convolutional feature extraction with LSTM

### Real-Time Integration
- **PAHO MQTT Streaming**: Eclipse Paho MQTT client for live data
- **Message Transformation**: JSON, CSV, and binary format support
- **City-Specific Transformers**: Automatic field mapping for different data sources

### Statistical Rigor (Q1 Quality)
- Walk-forward (expanding window) cross-validation
- Bootstrap confidence intervals
- Paired statistical tests (Wilcoxon, t-test)
- Effect size analysis (Cohen's d)
- Friedman test with Nemenyi post-hoc
- Ablation studies framework

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- TensorFlow 2.10+
- scikit-learn 1.0+
- paho-mqtt 1.6+

## Usage

### Q1 Enhanced Analysis (Recommended)
```bash
python artemis_q1_enhanced.py
```

This runs:
1. Multi-city real data download (SF, NYC, Seattle, Chicago, LA, Austin, Boston)
2. Comprehensive preprocessing and feature engineering
3. Traditional ML model evaluation (8 algorithms)
4. Deep Learning model evaluation (LSTM, GRU, Transformer, CNN-LSTM)
5. Statistical significance testing
6. Ablation studies
7. Publication-quality figure generation

### Real-Time Mode with MQTT
```bash
python artemis_realtime.py --mode realtime --broker broker.hivemq.com --duration 300
```

### Batch Analysis Only
```bash
python artemis_realtime.py --mode batch
```

## Real Public Data Sources

| City | Dataset | Records |
|------|---------|---------|
| San Francisco | Fire Department Calls for Service | ~15,000+ |
| New York City | EMS Incident Dispatch Data | ~15,000+ |
| Seattle | Real-Time Fire 911 Calls | ~15,000+ |
| Chicago | Fire Incidents | ~10,000+ |
| Los Angeles | Fire Department Incidents | ~10,000+ |
| Austin | Fire Incidents | ~5,000+ |
| Boston | Fire Incident Reporting | ~5,000+ |

**Total: 50,000+ real records from 7+ cities**

## Output Files

```
artemis_q1_outputs/
├── results_table.csv           # Publication-ready results
├── ablation_results.csv        # Feature importance analysis
├── fig_model_comparison.png    # Model comparison with CI
├── fig_data_summary.png        # Dataset visualization
├── fig_learning_curves.png     # DL training curves
├── fig_ablation.png            # Ablation study results
└── fig_statistical.png         # Statistical significance heatmap
```

## Models Evaluated

### Traditional ML
- Ridge Regression
- Lasso Regression
- ElasticNet
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- Extra Trees
- AdaBoost

### Deep Learning
- Bidirectional LSTM (3-layer)
- Bidirectional GRU (3-layer)
- Transformer (Multi-head attention)
- CNN-LSTM Hybrid

### Baseline Methods
- Historical Average
- Hourly Average
- Moving Average
- Exponential Smoothing
- ARIMA

## Citation

If you use this code in your research, please cite:

```bibtex
@article{artemis2024,
  title={ARTEMIS: Adaptive Resource and Temporal Emergency Management Intelligent System},
  author={Department of Mathematics, SRM Institute of Science and Technology},
  year={2024}
}
```

## Author

Department of Mathematics, SRM Institute of Science and Technology
