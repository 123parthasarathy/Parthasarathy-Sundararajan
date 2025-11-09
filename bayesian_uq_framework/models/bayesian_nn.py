"""
Bayesian Neural Network implementations using multiple techniques:
1. Monte Carlo Dropout
2. Deep Ensembles
3. Variational Inference (Bayes by Backprop)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple, Optional


class MCDropoutNN(nn.Module):
    """
    Monte Carlo Dropout Neural Network for Bayesian inference.

    Uses dropout at test time to approximate Bayesian posterior.
    Reference: Gal & Ghahramani (2016)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, dropout_rate: float = 0.2):
        super(MCDropoutNN, self).__init__()

        self.dropout_rate = dropout_rate
        layers = []

        # Build network
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo sampling for uncertainty estimation.

        Returns:
            mean: Predictive mean
            std: Predictive uncertainty (epistemic)
        """
        self.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer using variational inference.
    Implements Bayes by Backprop algorithm.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # Prior
        self.prior_std = prior_std
        self.prior = Normal(0, prior_std)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_mu, 0, 0.1)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.normal_(self.bias_mu, 0, 0.1)
        nn.init.constant_(self.bias_rho, -3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        # Sample bias
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))

        # KL for weights
        kl_weight = self._kl_normal(self.weight_mu, weight_std)
        # KL for bias
        kl_bias = self._kl_normal(self.bias_mu, bias_std)

        return kl_weight + kl_bias

    def _kl_normal(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """KL divergence between N(mu, std) and N(0, prior_std)."""
        kl = 0.5 * (
            2 * np.log(self.prior_std) - 2 * torch.log(std) +
            (std**2 + mu**2) / (self.prior_std**2) - 1
        )
        return kl.sum()


class VariationalNN(nn.Module):
    """
    Variational Neural Network using Bayes by Backprop.
    Full Bayesian treatment with uncertainty quantification.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, prior_std: float = 1.0):
        super(VariationalNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_std))
            prev_dim = hidden_dim

        layers.append(BayesianLinear(prev_dim, output_dim, prior_std))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all layers."""
        kl = 0
        for layer in self.layers:
            kl += layer.kl_divergence()
        return kl

    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor,
                  n_samples: int = 1, beta: float = 1.0) -> torch.Tensor:
        """
        Compute ELBO (Evidence Lower Bound) loss.

        Args:
            x: Input data
            y: Target data
            n_samples: Number of samples for MC estimation
            beta: Weight for KL term
        """
        # Likelihood term (averaged over samples)
        nll = 0
        for _ in range(n_samples):
            pred = self.forward(x)
            nll += F.mse_loss(pred, y)
        nll /= n_samples

        # KL divergence term
        kl = self.kl_divergence() / x.size(0)  # Normalize by batch size

        # ELBO = -log p(D|w) + KL(q(w)||p(w))
        return nll + beta * kl

    def predict_with_uncertainty(self, x: torch.Tensor,
                                n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty quantification.

        Returns:
            mean: Predictive mean
            std: Epistemic uncertainty
        """
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return mean, std


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty quantification.
    Trains multiple independent networks and aggregates predictions.

    Reference: Lakshminarayanan et al. (2017)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 output_dim: int, n_models: int = 5):
        self.n_models = n_models
        self.models = [
            self._create_model(input_dim, hidden_dims, output_dim)
            for _ in range(n_models)
        ]

    def _create_model(self, input_dim: int, hidden_dims: List[int],
                     output_dim: int) -> nn.Module:
        """Create a single model in the ensemble."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Output both mean and variance for aleatoric uncertainty
        layers.append(nn.Linear(prev_dim, output_dim * 2))

        return nn.Sequential(*layers)

    def train_ensemble(self, train_loader, n_epochs: int = 100,
                      lr: float = 0.001, device: str = 'cuda'):
        """Train all models in the ensemble."""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(n_epochs):
                total_loss = 0
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    optimizer.zero_grad()
                    output = model(x_batch)

                    # Split into mean and log variance
                    mean, log_var = torch.chunk(output, 2, dim=-1)

                    # Negative log-likelihood loss
                    loss = self._nll_loss(y_batch, mean, log_var)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def _nll_loss(self, y_true: torch.Tensor, mean: torch.Tensor,
                  log_var: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for Gaussian outputs."""
        precision = torch.exp(-log_var)
        return 0.5 * (log_var + precision * (y_true - mean)**2).mean()

    def predict_with_uncertainty(self, x: torch.Tensor,
                                device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with both aleatoric and epistemic uncertainty.

        Returns:
            mean: Predictive mean
            epistemic_std: Epistemic uncertainty (model uncertainty)
            aleatoric_std: Aleatoric uncertainty (data noise)
        """
        means = []
        variances = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                x_device = x.to(device)
                output = model(x_device)
                mean, log_var = torch.chunk(output, 2, dim=-1)

                means.append(mean.cpu())
                variances.append(torch.exp(log_var).cpu())

        means = torch.stack(means)
        variances = torch.stack(variances)

        # Predictive mean
        pred_mean = means.mean(dim=0)

        # Aleatoric uncertainty (expected data noise)
        aleatoric_var = variances.mean(dim=0)

        # Epistemic uncertainty (model uncertainty)
        epistemic_var = means.var(dim=0)

        return pred_mean, torch.sqrt(epistemic_var), torch.sqrt(aleatoric_var)


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout for automatic tuning of dropout rate.

    Reference: Gal et al. (2017)
    """

    def __init__(self, weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        # Learnable dropout probability in logit space
        self.p_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Apply concrete dropout to layer output."""
        p = torch.sigmoid(self.p_logit)

        # Apply dropout
        out = layer(self._concrete_dropout(x, p))

        return out

    def _concrete_dropout(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Apply dropout with learned rate."""
        eps = 1e-7
        temp = 0.1

        # Sample from concrete distribution
        unif_noise = torch.rand_like(x)
        drop_prob = (
            torch.log(p + eps) - torch.log(1 - p + eps) +
            torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps)
        )

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob

        return x * random_tensor

    def regularization(self) -> torch.Tensor:
        """Compute regularization term."""
        p = torch.sigmoid(self.p_logit)
        return self.weight_regularizer * (1 - p) + self.dropout_regularizer * p
