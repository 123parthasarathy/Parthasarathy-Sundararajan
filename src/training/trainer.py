"""
Training and evaluation utilities for graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from tqdm import tqdm
import time

from .metrics import compute_metrics, MetricTracker


class Trainer:
    """
    Trainer for graph neural networks.

    Supports:
    - Classification and regression tasks
    - Early stopping
    - Learning rate scheduling
    - Metric tracking
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = 'classification',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
        patience: int = 20,
        min_epochs: int = 50
    ):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.patience = patience
        self.min_epochs = min_epochs

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss function
        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.L1Loss()  # MAE for regression

        # Metric tracking
        metrics = ['accuracy', 'auc', 'f1'] if task == 'classification' else ['mae', 'rmse']
        self.tracker = MetricTracker(metrics)

        # Best model state
        self.best_state = None
        self.best_metric = float('-inf') if task == 'classification' else float('inf')

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            topo_features = getattr(batch, 'topo_features', None)
            pos = getattr(batch, 'pos', None)

            out, _ = self.model(
                batch.x.float(),
                batch.edge_index,
                batch.batch,
                topo_features=topo_features,
                pos=pos
            )

            # Get target
            if self.task == 'classification':
                target = batch.y.long()
            else:
                target = batch.y.float()
                if hasattr(batch, 'y_target_idx'):
                    target = target[:, batch.y_target_idx]
                if len(target.shape) > 1:
                    target = target.squeeze(-1)

            # Compute loss
            loss = self.criterion(out, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, Dict[str, float]]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_preds = []
        all_targets = []

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            topo_features = getattr(batch, 'topo_features', None)
            pos = getattr(batch, 'pos', None)

            out, _ = self.model(
                batch.x.float(),
                batch.edge_index,
                batch.batch,
                topo_features=topo_features,
                pos=pos
            )

            # Get target
            if self.task == 'classification':
                target = batch.y.long()
            else:
                target = batch.y.float()
                if hasattr(batch, 'y_target_idx'):
                    target = target[:, batch.y_target_idx]
                if len(target.shape) > 1:
                    target = target.squeeze(-1)

            loss = self.criterion(out, target)
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(out.cpu())
            all_targets.append(target.cpu())

        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_metrics(all_targets, all_preds, self.task)

        return total_loss / max(num_batches, 1), metrics

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 200,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Dictionary with training history and best metrics
        """
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        best_epoch = 0
        patience_counter = 0

        # Determine which metric to track for early stopping
        primary_metric = 'accuracy' if self.task == 'classification' else 'mae'
        mode = 'max' if self.task == 'classification' else 'min'

        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_loss, val_metrics = self.evaluate(val_loader)

            # Update tracker
            self.tracker.update({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            })

            # Check for improvement
            current_metric = val_metrics[primary_metric]
            improved = (mode == 'max' and current_metric > self.best_metric) or \
                      (mode == 'min' and current_metric < self.best_metric)

            if improved:
                self.best_metric = current_metric
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Update scheduler
            scheduler.step()

            # Progress update
            if verbose:
                metric_str = f"{primary_metric}={current_metric:.4f}"
                iterator.set_postfix({
                    'loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    primary_metric: f'{current_metric:.4f}'
                })

            # Early stopping
            if patience_counter >= self.patience and epoch >= self.min_epochs:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {
            'best_epoch': best_epoch,
            'best_metric': self.best_metric,
            'history': self.tracker.history
        }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader=None,
    task: str = 'classification',
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cpu',
    patience: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to train a model.

    Returns:
        Dictionary with training results and test metrics
    """
    trainer = Trainer(
        model=model,
        task=task,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        patience=patience
    )

    # Train
    train_results = trainer.train(train_loader, val_loader, epochs=epochs, verbose=verbose)

    # Test evaluation
    results = {
        'train_history': train_results['history'],
        'best_epoch': train_results['best_epoch'],
        'best_val_metric': train_results['best_metric']
    }

    if test_loader is not None:
        test_loss, test_metrics = trainer.evaluate(test_loader)
        results['test_loss'] = test_loss
        results['test_metrics'] = test_metrics

    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    task: str = 'classification',
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Returns:
        Dictionary of metrics
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        topo_features = getattr(batch, 'topo_features', None)
        pos = getattr(batch, 'pos', None)

        out, _ = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch,
            topo_features=topo_features,
            pos=pos
        )

        target = batch.y.float() if task == 'regression' else batch.y.long()
        if len(target.shape) > 1:
            target = target.squeeze(-1)

        all_preds.append(out.cpu())
        all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    return compute_metrics(all_targets, all_preds, task)
