# QI-VGT Validation Report for Q1 Publication

**Generated:** 2025-12-10 15:33:31

---

## Executive Summary

**Validation Status:** 1/5 claims validated

**Best Achieved Accuracy:** 92.57% ± 5.98%
**Best Achieved AUC-ROC:** 95.27% ± 4.93%

---

## 1. Paper Claims Validation

| Metric | Claimed | Achieved | Std | Difference | Status |
|--------|---------|----------|-----|------------|--------|
| Accuracy | 84.59% | 92.57% | ±5.98% | +7.98% | ⚠️ DEVIATION |
| AUC-ROC | 89.30% | 95.27% | ±4.93% | +5.97% | ⚠️ DEVIATION |
| Precision | 86.78% | 98.25% | ±2.41% | +11.47% | ⚠️ DEVIATION |
| Recall | 91.20% | 90.40% | ±8.29% | -0.80% | ✅ VALIDATED |
| F1-Score | 88.74% | 94.02% | ±5.01% | +5.28% | ⚠️ DEVIATION |

---

## 2. Comparison with Baseline Methods

| Method | Accuracy | AUC-ROC | Precision | Recall | F1 |
|--------|----------|---------|-----------|--------|-----|
| GAT | 95.78±4.77 | 96.89±3.42 | 100.00±0.00 | 93.60±7.27 | 96.58±3.93 |
| QI-VGT++ | 93.66±5.10 | 96.52±4.12 | 96.87±4.88 | 93.60±5.37 | 95.11±4.04 |
| GCN | 93.12±6.32 | 94.59±4.75 | 100.00±0.00 | 89.60±9.63 | 94.29±5.48 |
| GIN | 93.12±5.43 | 95.69±4.13 | 96.06±5.60 | 93.60±3.58 | 94.77±4.11 |
| QI-VGT | 92.57±5.98 | 95.27±4.93 | 98.25±2.41 | 90.40±8.29 | 94.02±5.01 |
| DEEPGCN | 91.51±5.04 | 96.50±3.52 | 97.66±3.46 | 89.60±9.21 | 93.15±4.52 |
| Random Forest | 83.02±5.97 | 88.85±3.88 | 87.97±5.48 | 86.40±6.69 | 87.05±4.86 |
| MLP | 82.46±3.44 | 91.83±3.86 | 86.63±2.26 | 87.20±7.16 | 86.75±3.22 |
| Naive Bayes | 72.82±7.60 | 87.50±4.22 | 87.33±13.02 | 73.60±19.31 | 77.49±8.53 |
| SVM (RBF) | 66.50±0.97 | 86.39±4.08 | 66.50±0.97 | 100.00±0.00 | 79.88±0.70 |

---

## 3. Comparison with Recent Literature (2024-2025)


### AmesFormer (2025)
- **Notes:** Graph transformer for mutagenicity, published in Chem. Res. Toxicol.
- **Dataset:** Large Ames dataset (not MUTAG)

### GeoScatt-GNN (2024)
- **Notes:** Geometric scattering + GNN hybrid, Hansen dataset
- **Dataset:** 6,277 compounds
- **Accuracy Comparison:** Literature: 93.0%, QI-VGT: 92.57% (-0.43% - Literature better)

### KA-GNNs (2025)
- **Notes:** Kolmogorov-Arnold network enhanced GNNs
- **Dataset:** MUTAG

### GAT (2024 benchmark)
- **Notes:** Graph Attention Network on MUTAG
- **Dataset:** MUTAG
- **Accuracy Comparison:** Literature: 89.42%, QI-VGT: 92.57% (+3.15% - QI-VGT better)

### GIN (2024 benchmark)
- **Notes:** Graph Isomorphism Network on MUTAG
- **Dataset:** MUTAG
- **Accuracy Comparison:** Literature: 85.14%, QI-VGT: 92.57% (+7.43% - QI-VGT better)

### GCN (2024 benchmark)
- **Notes:** Standard GCN on MUTAG
- **Dataset:** MUTAG
- **Accuracy Comparison:** Literature: 84.1%, QI-VGT: 92.57% (+8.47% - QI-VGT better)

### DGCNNII (2024)
- **Notes:** Deep GCN with improved architecture
- **Dataset:** MUTAG

---

## 4. Ablation Study Results

| Configuration | Accuracy | Impact | Critical |
|--------------|----------|--------|----------|
| Full QI-VGT | 92.57% | -0.00% | 🟢 No |
| Without Quantum Enhancement | 93.64% | +1.07% | 🟢 No |
| Without Dropout | 92.05% | -0.52% | 🟢 No |
| Smaller Hidden (16) | 93.64% | +1.07% | 🟢 No |
| Larger Hidden (64) | 93.12% | +0.55% | 🟢 No |

**Component Importance Ranking (by impact when removed):**
1. Without Dropout: -0.52%
2. Larger Hidden (64): +0.55%
3. Without Quantum Enhancement: +1.07%
4. Smaller Hidden (16): +1.07%

---

## 5. Statistical Summary

- **Methods Evaluated:** 10
- **Best Method:** GAT (95.78%)
- **QI-VGT Rank:** 5/10

**Full Rankings:**
1. GAT: 95.78% 👑
2. QI-VGT++: 93.66%
3. GCN: 93.12%
4. GIN: 93.12%
5. QI-VGT: 92.57%
6. DEEPGCN: 91.51%
7. Random Forest: 83.02%
8. MLP: 82.46%
9. Naive Bayes: 72.82%
10. SVM (RBF): 66.50%

---

## 6. Conclusions and Recommendations

### ⚠️ PARTIAL VALIDATION - REVIEW RECOMMENDED

Some metrics show deviations from claimed values. Review suggested.

### Strengths:
1. Novel quantum-inspired enhancement mechanism
2. Competitive performance with state-of-the-art GNN methods
3. Built-in uncertainty quantification
4. Parameter efficient design (~6,947 parameters)
5. Multi-strategy pooling captures diverse molecular features

### Areas for Improvement:
1. Evaluation on larger and more diverse datasets recommended
2. Comparison with latest 2024-2025 methods (AmesFormer, GeoScatt-GNN)
3. More extensive ablation studies on quantum parameters

---

*Report generated automatically by QI-VGT Validation Framework*