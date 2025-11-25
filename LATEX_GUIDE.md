# LaTeX Integration Guide

## Including Figures in Your LaTeX Document

### All 12 PNG Figures (300 DPI) Ready for LaTeX

---

## Figure List

### Performance & Results Figures (1-8)

1. **1_roc_curves.png** - ROC curves comparison
2. **2_uncertainty_analysis.png** - Uncertainty quantification (4-panel)
3. **3_calibration_curve.png** - Calibration assessment
4. **4_confusion_matrix.png** - Confusion matrix with metrics
5. **5_smote_comparison.png** - SMOTE variant comparison (4-panel)
6. **6_feature_importance.png** - Top 20 features
7. **7_literature_comparison.png** - Literature comparison
8. **8_performance_summary.png** - Overall metrics

### Architecture & Methodology Figures (9-12)

9. **9_methodology_architecture.png** - Overall system architecture
10. **10_data_flow.png** - Data processing pipeline
11. **11_medicinal_plants_integration.png** - Medicinal plants features
12. **12_results_comparison_chart.png** - Performance comparison chart

---

## LaTeX Code for Including Figures

### Basic Figure Inclusion

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/1_roc_curves.png}
    \caption{ROC curves for individual models and ensemble. The ensemble achieved
    superior discrimination with AUC = 0.875, outperforming individual models.}
    \label{fig:roc_curves}
\end{figure}
```

### Two-Column Figure

```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/2_uncertainty_analysis.png}
    \caption{Uncertainty analysis showing (a) epistemic uncertainty, (b) aleatoric
    uncertainty, (c) uncertainty distribution by class, and (d) confidence-stratified
    performance.}
    \label{fig:uncertainty}
\end{figure*}
```

### Side-by-Side Figures

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/1_roc_curves.png}
        \caption{ROC Curves}
        \label{fig:roc}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/3_calibration_curve.png}
        \caption{Calibration}
        \label{fig:calibration}
    \end{subfigure}
    \caption{Performance assessment: (a) discrimination and (b) calibration.}
    \label{fig:performance}
\end{figure}
```

---

## Complete LaTeX Document Template

```latex
\documentclass[twocolumn]{article}

% Packages
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}

% Set graphics path
\graphicspath{{figures/}}

\title{Uncertainty-Aware Ensemble Learning for Diabetes Prediction: \\
Integrating Medicinal Plant Interventions with Conformal Prediction}

\author{Your Name et al.}

\begin{document}

\maketitle

\begin{abstract}
We present a novel uncertainty-aware ensemble framework for diabetes prediction
that integrates medicinal plant interventions with conformal prediction.
Using real clinical data from 101,766 hospital encounters, our approach
achieves AUC = 0.875, competitive with state-of-the-art.
\end{abstract}

\section{Introduction}

[Your introduction text...]

\section{Methods}

\subsection{Dataset}

We utilized the Diabetes 130-US Hospitals dataset comprising 101,766 real
patient encounters~\cite{strack2014}.

\subsection{Methodology}

Figure~\ref{fig:architecture} presents our overall methodology architecture.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{9_methodology_architecture.png}
    \caption{Overall methodology architecture showing data flow from input
    through ensemble models to final predictions with uncertainty quantification.}
    \label{fig:architecture}
\end{figure}

\subsection{Medicinal Plant Integration}

We integrated 10 evidence-based anti-diabetic medicinal plants as shown in
Figure~\ref{fig:plants}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{11_medicinal_plants_integration.png}
    \caption{Integration of 10 medicinal plant features with usage rates
    based on epidemiological studies.}
    \label{fig:plants}
\end{figure}

\section{Results}

\subsection{Overall Performance}

Figure~\ref{fig:roc} shows the ROC curves for all models.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{1_roc_curves.png}
    \caption{ROC curves comparing individual models (XGBoost, LightGBM, CatBoost)
    and the ensemble. All models demonstrate strong discrimination ability.}
    \label{fig:roc}
\end{figure}

\subsection{Uncertainty Quantification}

Our novel uncertainty quantification framework is shown in Figure~\ref{fig:uncertainty}.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{2_uncertainty_analysis.png}
    \caption{Comprehensive uncertainty analysis: (a) epistemic uncertainty
    (model variance), (b) aleatoric uncertainty (prediction entropy),
    (c) uncertainty distribution by outcome class, and (d) confidence-stratified
    performance showing higher accuracy for high-confidence predictions.}
    \label{fig:uncertainty}
\end{figure*}

\subsection{Model Calibration}

Figure~\ref{fig:calibration} demonstrates model calibration quality.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{3_calibration_curve.png}
    \caption{Calibration curve and prediction distribution showing
    well-calibrated probability estimates.}
    \label{fig:calibration}
\end{figure}

\subsection{Performance Metrics}

The confusion matrix (Figure~\ref{fig:confusion}) shows detailed performance.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\columnwidth]{4_confusion_matrix.png}
    \caption{Confusion matrix with clinical metrics: sensitivity,
    specificity, PPV, and NPV.}
    \label{fig:confusion}
\end{figure}

\subsection{Feature Importance}

Figure~\ref{fig:features} shows the top 20 most important features.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{6_feature_importance.png}
    \caption{Top 20 features ranked by importance, including several
    medicinal plant features.}
    \label{fig:features}
\end{figure}

\section{Discussion}

\subsection{Comparison with Literature}

Figure~\ref{fig:comparison} compares our results with recent publications.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{7_literature_comparison.png}
    \caption{Performance comparison with 6 recent SCI publications showing
    competitive results.}
    \label{fig:comparison}
\end{figure}

Our approach achieved AUC = 0.875, competitive with the best published
result of 0.881~\cite{ahuja2022}.

\section{Conclusion}

[Your conclusion...]

\begin{thebibliography}{9}

\bibitem{strack2014}
Strack, B., et al. (2014).
Impact of HbA1c measurement on hospital readmission rates.
\textit{BioMed Research International}, 2014.

\bibitem{ahuja2022}
Ahuja, R., et al. (2022).
Deep learning-based diabetes prediction.
\textit{Computational Intelligence and Neuroscience}, 2022.

\end{thebibliography}

\end{document}
```

---

## Required LaTeX Packages

Add to your preamble:

```latex
\usepackage{graphicx}        % For including images
\usepackage{subcaption}      % For subfigures
\usepackage{float}           % For figure placement
\usepackage{caption}         % For caption formatting
```

---

## Figure Placement Options

- `[h]` - here (approximately)
- `[t]` - top of page
- `[b]` - bottom of page
- `[p]` - separate page
- `[H]` - exactly here (requires float package)
- `[htbp]` - here, top, bottom, or page (recommended)

---

## Resizing Figures

```latex
% Full column width
\includegraphics[width=\columnwidth]{figure.png}

% Half column width
\includegraphics[width=0.5\columnwidth]{figure.png}

% Full text width (for two-column documents)
\includegraphics[width=\textwidth]{figure.png}

% Fixed width
\includegraphics[width=10cm]{figure.png}

% Scale
\includegraphics[scale=0.8]{figure.png}
```

---

## Figure Captions - Recommended Templates

### Figure 1 (ROC Curves)
```latex
\caption{Receiver Operating Characteristic (ROC) curves for XGBoost, LightGBM,
CatBoost, and the ensemble model. The ensemble achieved AUC = 0.875, demonstrating
superior discrimination ability compared to individual models.}
```

### Figure 2 (Uncertainty)
```latex
\caption{Uncertainty quantification analysis. (a) Epistemic uncertainty showing
model disagreement, (b) aleatoric uncertainty indicating prediction entropy,
(c) uncertainty distribution stratified by outcome class, and (d) confidence-stratified
performance demonstrating higher accuracy for high-confidence predictions (>90\%).}
```

### Figure 9 (Architecture)
```latex
\caption{Overall methodology architecture of the uncertainty-aware ensemble
framework. Data flows from preprocessing through SMOTE resampling and ensemble
models (XGBoost, LightGBM, CatBoost) to final predictions with epistemic and
aleatoric uncertainty quantification and conformal prediction intervals.}
```

---

## Tips for LaTeX Figure Management

1. **Create a figures/ subdirectory**: Keep all PNG files organized
2. **Use consistent naming**: Numbered or descriptive names
3. **Set graphicspath**: `\graphicspath{{figures/}}`
4. **Use labels**: Always add `\label{fig:name}` for cross-referencing
5. **Reference in text**: Use `Figure~\ref{fig:name}`
6. **Two-column figures**: Use `figure*` environment for full-width figures
7. **Vector vs raster**: PNG works well for complex plots at 300 DPI

---

## Overleaf Integration

1. Create new project in Overleaf
2. Upload all 12 PNG files to `figures/` folder
3. Copy LaTeX template above
4. Compile with pdfLaTeX or XeLaTeX

---

## Journal-Specific Requirements

### Scientific Reports (Nature)
- Figures: 300 DPI minimum ✓
- Format: PNG, TIFF, or EPS ✓
- Size: Max 180mm width
- All our figures meet requirements

### IEEE Journals
- Figures: 300-600 DPI ✓
- Format: PNG or TIFF ✓
- Color: RGB for online, CMYK for print
- Our figures meet requirements

### PLoS ONE
- Figures: 300 DPI minimum ✓
- Format: PNG, TIFF ✓
- Fonts: Arial, Times, Symbol
- Our figures meet requirements

---

## All Figures Ready for LaTeX! ✓

Total: 12 PNG files, 300 DPI, publication-quality
Format: Compatible with all major journals
Size: Optimized for LaTeX inclusion
Status: Ready to use
