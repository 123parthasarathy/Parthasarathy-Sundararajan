"""
Visualization utilities for Bayesian UQ framework.

Comprehensive plotting functions for:
- Uncertainty visualization
- Manifold plots
- Reliability diagrams
- Calibration plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class UncertaintyVisualizer:
    """Visualize uncertainty estimates."""

    @staticmethod
    def plot_predictions_with_uncertainty(
        x: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_values: Optional[np.ndarray] = None,
        n_std: float = 2.0,
        title: str = "Predictions with Uncertainty",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot predictions with uncertainty bands.

        Args:
            x: Input values (for x-axis)
            predictions: Predicted values
            uncertainties: Uncertainty estimates (std)
            true_values: Ground truth values
            n_std: Number of std deviations for bands
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by x for better visualization
        if x.ndim == 1:
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            pred_sorted = predictions[sort_idx]
            unc_sorted = uncertainties[sort_idx]

            # Plot uncertainty bands
            ax.fill_between(
                x_sorted,
                pred_sorted - n_std * unc_sorted,
                pred_sorted + n_std * unc_sorted,
                alpha=0.3,
                label=f'{n_std}σ Uncertainty'
            )

            # Plot predictions
            ax.plot(x_sorted, pred_sorted, 'b-', linewidth=2, label='Prediction')

            # Plot true values if available
            if true_values is not None:
                true_sorted = true_values[sort_idx]
                ax.plot(x_sorted, true_sorted, 'r--', linewidth=2, label='True')

        else:
            # For high-dimensional x, just scatter plot
            ax.scatter(range(len(predictions)), predictions, alpha=0.6, label='Prediction')
            ax.errorbar(
                range(len(predictions)),
                predictions,
                yerr=n_std * uncertainties,
                fmt='none',
                alpha=0.3,
                label=f'{n_std}σ Uncertainty'
            )

            if true_values is not None:
                ax.scatter(range(len(true_values)), true_values, alpha=0.6,
                          marker='x', label='True')

        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_uncertainty_distribution(
        uncertainties: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Uncertainty Distribution",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot distribution of uncertainties.

        Args:
            uncertainties: Uncertainty values (can be 2D for multiple types)
            labels: Labels for different uncertainty types
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if uncertainties.ndim == 1:
            uncertainties = uncertainties.reshape(-1, 1)
            if labels is None:
                labels = ['Uncertainty']

        for i in range(uncertainties.shape[1]):
            ax.hist(
                uncertainties[:, i],
                bins=50,
                alpha=0.6,
                label=labels[i] if labels else f'Type {i}',
                edgecolor='black'
            )

        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_uncertainty_vs_error(
        uncertainties: np.ndarray,
        errors: np.ndarray,
        title: str = "Uncertainty vs Error",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Scatter plot of uncertainty vs actual error.

        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(uncertainties, errors, alpha=0.5, s=20)

        # Add diagonal reference
        max_val = max(uncertainties.max(), errors.max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect calibration')

        # Compute and display correlation
        corr = np.corrcoef(uncertainties, errors)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Actual Error')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class ManifoldVisualizer:
    """Visualize manifold embeddings."""

    @staticmethod
    def plot_manifold_2d(
        embedded: np.ndarray,
        colors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        title: str = "Manifold Embedding",
        colorbar_label: str = "Value",
        ax: Optional[plt.Axes] = None,
        **scatter_kwargs
    ) -> plt.Axes:
        """
        Plot 2D manifold embedding.

        Args:
            embedded: Embedded coordinates (n_samples, 2)
            colors: Color values for each point
            labels: Discrete labels for each point
            title: Plot title
            colorbar_label: Label for colorbar
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            # Discrete labels
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    label=f'Class {label}',
                    alpha=0.6,
                    s=30,
                    **scatter_kwargs
                )
            ax.legend()

        elif colors is not None:
            # Continuous colors
            scatter = ax.scatter(
                embedded[:, 0],
                embedded[:, 1],
                c=colors,
                cmap='viridis',
                alpha=0.6,
                s=30,
                **scatter_kwargs
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)

        else:
            # No colors
            ax.scatter(
                embedded[:, 0],
                embedded[:, 1],
                alpha=0.6,
                s=30,
                **scatter_kwargs
            )

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_uncertainty_manifold(
        embedded: np.ndarray,
        uncertainties: np.ndarray,
        title: str = "Uncertainty in Manifold Space",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot manifold colored by uncertainty.

        Args:
            embedded: Embedded coordinates
            uncertainties: Uncertainty values
            title: Plot title
            ax: Matplotlib axes
        """
        return ManifoldVisualizer.plot_manifold_2d(
            embedded,
            colors=uncertainties,
            title=title,
            colorbar_label='Uncertainty',
            ax=ax
        )

    @staticmethod
    def plot_reliability_regions(
        embedded: np.ndarray,
        region_labels: np.ndarray,
        region_names: Optional[List[str]] = None,
        title: str = "Reliability Regions",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot reliability regions in manifold space.

        Args:
            embedded: Embedded coordinates
            region_labels: Region assignments
            region_names: Names for each region
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        unique_regions = np.unique(region_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_regions)))

        for i, region in enumerate(unique_regions):
            mask = region_labels == region
            label = region_names[region] if region_names else f'Region {region}'

            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class CalibrationVisualizer:
    """Visualize calibration quality."""

    @staticmethod
    def plot_calibration_curve(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot calibration curve (reliability diagram).

        Args:
            confidences: Predicted confidences
            accuracies: Binary correctness (1 if correct)
            n_bins: Number of bins
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & \
                     (confidences < bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                bin_confidences.append(confidences[in_bin].mean())
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_counts.append(in_bin.sum())

        bin_confidences = np.array(bin_confidences)
        bin_accuracies = np.array(bin_accuracies)
        bin_counts = np.array(bin_counts)

        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(bin_confidences, bin_accuracies, 'o-',
               linewidth=2, markersize=8, label='Model')

        # Add bar chart for sample counts
        ax2 = ax.twinx()
        ax2.bar(bin_confidences, bin_counts, alpha=0.3,
               width=1.0/n_bins, color='gray')
        ax2.set_ylabel('Count', color='gray')

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        return ax

    @staticmethod
    def plot_regression_calibration(
        expected_errors: np.ndarray,
        observed_errors: np.ndarray,
        title: str = "Regression Calibration",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot calibration for regression (expected vs observed errors).

        Args:
            expected_errors: Expected errors (uncertainties)
            observed_errors: Observed errors
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot points
        ax.scatter(expected_errors, observed_errors, alpha=0.6, s=100)

        # Perfect calibration line
        max_val = max(expected_errors.max(), observed_errors.max())
        ax.plot([0, max_val], [0, max_val], 'r--',
               linewidth=2, label='Perfect calibration')

        # Compute miscalibration
        miscal = np.mean(np.abs(expected_errors - observed_errors))
        ax.text(0.05, 0.95, f'Miscalibration: {miscal:.4f}',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Expected Error (Uncertainty)')
        ax.set_ylabel('Observed Error')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_prediction_intervals(
        predictions: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        true_values: np.ndarray,
        x: Optional[np.ndarray] = None,
        confidence: float = 0.95,
        title: str = "Prediction Intervals",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot prediction intervals with coverage analysis.

        Args:
            predictions: Predicted values
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
            true_values: Ground truth
            x: X-axis values (optional)
            confidence: Confidence level
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if x is None:
            x = np.arange(len(predictions))

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        predictions = predictions[sort_idx]
        lower_bounds = lower_bounds[sort_idx]
        upper_bounds = upper_bounds[sort_idx]
        true_values = true_values[sort_idx]

        # Plot intervals
        ax.fill_between(x, lower_bounds, upper_bounds,
                        alpha=0.3, label=f'{confidence*100:.0f}% Interval')

        # Plot predictions
        ax.plot(x, predictions, 'b-', linewidth=2, label='Prediction')

        # Plot true values
        ax.scatter(x, true_values, c='red', s=20, alpha=0.6, label='True', zorder=5)

        # Compute coverage
        in_interval = (true_values >= lower_bounds) & (true_values <= upper_bounds)
        coverage = in_interval.mean()

        ax.text(0.05, 0.95, f'Coverage: {coverage:.2%}',
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class ReliabilityVisualizer:
    """Visualize reliability analysis results."""

    @staticmethod
    def plot_selective_prediction(
        coverage_levels: List[float],
        rmse_values: List[float],
        title: str = "Selective Prediction Performance",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot performance vs coverage trade-off.

        Args:
            coverage_levels: Coverage levels
            rmse_values: RMSE at each coverage
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(coverage_levels, rmse_values, 'o-',
               linewidth=2, markersize=8)

        ax.set_xlabel('Coverage (fraction of samples retained)')
        ax.set_ylabel('RMSE')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

        return ax

    @staticmethod
    def plot_uncertainty_ranking(
        bin_uncertainties: np.ndarray,
        bin_errors: np.ndarray,
        title: str = "Uncertainty Ranking Quality",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot error vs uncertainty bins.

        Args:
            bin_uncertainties: Average uncertainty per bin
            bin_errors: Average error per bin
            title: Plot title
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(bin_uncertainties, bin_errors, 'o-',
               linewidth=2, markersize=8)

        # Add reference line
        ax.plot([bin_uncertainties.min(), bin_uncertainties.max()],
               [bin_errors.min(), bin_errors.max()],
               'r--', alpha=0.5, label='Ideal ranking')

        ax.set_xlabel('Mean Uncertainty (bin)')
        ax.set_ylabel('Mean Error (bin)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_comprehensive_dashboard(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_values: np.ndarray,
        embedded: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization dashboard.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            true_values: Ground truth
            embedded: Manifold embedding (optional)
            x: Input values (optional)

        Returns:
            Figure with multiple subplots
        """
        errors = np.abs(predictions - true_values)

        if embedded is not None:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Predictions with uncertainty
        ax1 = fig.add_subplot(gs[0, 0])
        UncertaintyVisualizer.plot_predictions_with_uncertainty(
            x if x is not None else np.arange(len(predictions))[:100],
            predictions[:100],
            uncertainties[:100],
            true_values[:100],
            ax=ax1
        )

        # 2. Uncertainty distribution
        ax2 = fig.add_subplot(gs[0, 1])
        UncertaintyVisualizer.plot_uncertainty_distribution(
            uncertainties,
            ax=ax2
        )

        # 3. Uncertainty vs Error
        ax3 = fig.add_subplot(gs[1, 0])
        UncertaintyVisualizer.plot_uncertainty_vs_error(
            uncertainties,
            errors,
            ax=ax3
        )

        # 4. Error distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Distribution')
        ax4.grid(True, alpha=0.3)

        # 5. Prediction interval coverage
        ax5 = fig.add_subplot(gs[2, 0])
        lower = predictions - 2 * uncertainties
        upper = predictions + 2 * uncertainties
        CalibrationVisualizer.plot_prediction_intervals(
            predictions[:100],
            lower[:100],
            upper[:100],
            true_values[:100],
            x=x[:100] if x is not None else None,
            ax=ax5
        )

        # 6. Correlation heatmap
        ax6 = fig.add_subplot(gs[2, 1])
        corr_data = np.column_stack([predictions, uncertainties, errors])
        corr_matrix = np.corrcoef(corr_data.T)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=['Pred', 'Unc', 'Error'],
            yticklabels=['Pred', 'Unc', 'Error'],
            cmap='coolwarm',
            center=0,
            ax=ax6
        )
        ax6.set_title('Correlation Matrix')

        # 7. Manifold plots if available
        if embedded is not None:
            ax7 = fig.add_subplot(gs[0, 2])
            ManifoldVisualizer.plot_uncertainty_manifold(
                embedded,
                uncertainties,
                ax=ax7
            )

            ax8 = fig.add_subplot(gs[1, 2])
            ManifoldVisualizer.plot_manifold_2d(
                embedded,
                colors=errors,
                colorbar_label='Error',
                title='Error in Manifold Space',
                ax=ax8
            )

            ax9 = fig.add_subplot(gs[2, 2])
            ManifoldVisualizer.plot_manifold_2d(
                embedded,
                colors=predictions,
                colorbar_label='Prediction',
                title='Predictions in Manifold Space',
                ax=ax9
            )

        fig.suptitle('Bayesian UQ Comprehensive Dashboard', fontsize=16, y=0.995)

        return fig
