"""
Evaluation metrics for graph neural networks.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Dict, Union, List


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)
    return accuracy_score(y_true, y_pred)


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ROC-AUC score."""
    try:
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] == 2:
                # Binary classification - use probability of positive class
                y_score = y_pred[:, 1]
            else:
                # Multi-class - use one-vs-rest
                return roc_auc_score(y_true, y_pred, multi_class='ovr')
        else:
            y_score = y_pred
        return roc_auc_score(y_true, y_score)
    except ValueError:
        # AUC not defined (e.g., only one class in y_true)
        return 0.5


def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """Compute F1 score."""
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    task: str = 'classification'
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a task.

    Args:
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values (logits for classification)
        task: 'classification' or 'regression'

    Returns:
        Dictionary of metric name -> value
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    metrics = {}

    if task == 'classification':
        # Apply softmax if logits
        if len(y_pred.shape) > 1:
            y_prob = np.exp(y_pred) / np.exp(y_pred).sum(axis=1, keepdims=True)
            y_class = y_pred.argmax(axis=1)
        else:
            y_prob = y_pred
            y_class = (y_pred > 0.5).astype(int)

        metrics['accuracy'] = accuracy(y_true, y_class)
        metrics['auc'] = auc_score(y_true, y_prob)
        metrics['f1'] = f1(y_true, y_class)

    else:  # regression
        metrics['mae'] = mae(y_true, y_pred)
        metrics['rmse'] = rmse(y_true, y_pred)

    return metrics


class MetricTracker:
    """Track metrics over training epochs."""

    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.history = {m: [] for m in metrics}
        self.history['train_loss'] = []
        self.history['val_loss'] = []

    def update(self, values: Dict[str, float]):
        """Add new metric values."""
        for k, v in values.items():
            if k in self.history:
                self.history[k].append(v)

    def get_best(self, metric: str, mode: str = 'max') -> float:
        """Get best value for a metric."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        if mode == 'max':
            return max(self.history[metric])
        else:
            return min(self.history[metric])

    def get_last(self, metric: str) -> float:
        """Get most recent value for a metric."""
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        return self.history[metric][-1]
