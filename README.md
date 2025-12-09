# TIEGNN: Topological, Interpretable, Equivariant Graph Neural Network

A unified framework combining persistent homology, interpretable additive architecture, and E(n)-equivariant processing for graph neural networks.

## Overview

TIEGNN addresses three fundamental challenges in graph neural networks:
1. **Capturing higher-order structural patterns** via persistent homology
2. **Ensuring model transparency** through additive decomposition
3. **Maintaining geometric consistency** with E(n)-equivariant processing

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── tiegnn.py          # Main TIEGNN model
│   │   └── baselines.py       # GCN, GAT, EGNN, PersLay baselines
│   ├── data/
│   │   ├── datasets.py        # Dataset loaders (MUTAG, PROTEINS, QM9, ZINC)
│   │   └── topology.py        # Topological feature extraction
│   └── training/
│       ├── trainer.py         # Training utilities
│       └── metrics.py         # Evaluation metrics
├── experiments/
│   ├── run_benchmarks.py      # Main benchmark experiments
│   ├── ablation_study.py      # Ablation study
│   ├── runtime_analysis.py    # Runtime benchmarks
│   └── run_synthetic_experiments.py  # Synthetic data experiments
├── results/
│   └── EXPERIMENTAL_RESULTS.md  # Comprehensive results
└── requirements.txt
```

## Usage

### Training TIEGNN

```python
from src.models.tiegnn import TIEGNN
from src.data.datasets import get_dataset
from src.training.trainer import train_model

# Load dataset
dataset, info = get_dataset('MUTAG')

# Create model
model = TIEGNN(
    in_channels=info['num_features'],
    hidden_dim=64,
    num_classes=info['num_classes'],
    use_topo=True,
    use_interpretable=True,
    use_equiv=False
)

# Train
results = train_model(model, train_loader, val_loader, test_loader)
```

### Running Experiments

```bash
# Run benchmarks on real datasets
python experiments/run_benchmarks.py --datasets MUTAG PROTEINS --models GCN GAT PersLay TIEGNN

# Run ablation study
python experiments/ablation_study.py --dataset MUTAG

# Run runtime analysis
python experiments/runtime_analysis.py

# Run synthetic experiments (when datasets unavailable)
python experiments/run_synthetic_experiments.py
```

## Key Results

### Benchmark Performance (10-fold CV on Real Datasets)

#### MUTAG (135 graphs, binary classification)

| Model | Accuracy | AUC |
|-------|----------|-----|
| GCN | 69.62% +/- 3.80% | 74.97% |
| GAT | 71.10% +/- 7.11% | 75.19% |
| PersLay | **81.54%** +/- 10.39% | **90.72%** |
| TIEGNN | 75.60% +/- 7.21% | 85.42% |

#### PROTEINS (975 graphs, binary classification)

| Model | Accuracy | AUC |
|-------|----------|-----|
| GCN | 66.77% +/- 9.73% | 67.04% |
| GAT | 58.60% +/- 14.02% | 67.31% |
| PersLay | 74.47% +/- 3.77% | 79.04% |
| **TIEGNN** | **75.08% +/- 1.97%** | 77.93% |

**TIEGNN achieves best accuracy (75.08%) with lowest variance (1.97%) on PROTEINS.**

### Ablation Study (MUTAG)

| Configuration | Accuracy | Relative |
|--------------|----------|----------|
| TIEGNN (full) | 75.60% | baseline |
| w/o Topology | 72.58% | -3.02 pp |
| w/o Interpretable | 81.43% | +5.83 pp |
| Graph-only | 73.96% | -1.64 pp |

### Runtime (ms, batch of 32 graphs)

| Model | 20 nodes | 50 nodes | 100 nodes |
|-------|----------|----------|-----------|
| GCN | 12.5 | 16.5 | 22.6 |
| TIEGNN | 32.5 | 35.2 | 35.7 |

## Features

- **Dual-space persistence**: Computes Vietoris-Rips filtrations in both node and edge spaces
- **Interpretable predictions**: Additive decomposition with learnable shape functions
- **E(n)-equivariance**: Preserves rotational, translational, and reflective symmetries
- **Flexible architecture**: Components can be enabled/disabled based on application needs

## Citation

If you use this code, please cite:

```bibtex
@article{tiegnn2024,
  title={A Unified Framework for Topological, Interpretable, and Equivariant Graph Neural Networks},
  author={Rajeswari R and Sujatha N},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
