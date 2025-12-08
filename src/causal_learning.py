"""
Causal Structure Learning Module
Implements temporal causal extraction and causal-constrained neural networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import GraphicalLasso
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class CausalGraphDiscovery:
    """
    Discover causal relationships between features using multiple methods.
    """

    def __init__(self, method: str = 'correlation', threshold: float = 0.3,
                 alpha: float = 0.01, random_state: int = 42):
        """
        Initialize causal discovery.

        Args:
            method: Discovery method ('correlation', 'partial_correlation', 'granger')
            threshold: Edge threshold for graph construction
            alpha: Significance level for statistical tests
            random_state: Random seed
        """
        self.method = method
        self.threshold = threshold
        self.alpha = alpha
        self.random_state = random_state
        self.adjacency_matrix = None
        self.feature_importance = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CausalGraphDiscovery':
        """
        Discover causal structure from data.

        Args:
            X: Feature matrix
            y: Optional labels (used for supervised causal discovery)

        Returns:
            self
        """
        n_features = X.shape[1]

        if self.method == 'correlation':
            self.adjacency_matrix = self._correlation_graph(X)
        elif self.method == 'partial_correlation':
            self.adjacency_matrix = self._partial_correlation_graph(X)
        elif self.method == 'granger':
            self.adjacency_matrix = self._granger_causality_graph(X)
        else:
            # Default to correlation-based
            self.adjacency_matrix = self._correlation_graph(X)

        # Compute feature importance based on graph structure
        self.feature_importance = np.sum(np.abs(self.adjacency_matrix), axis=0)
        self.feature_importance /= self.feature_importance.max() + 1e-10

        return self

    def _correlation_graph(self, X: np.ndarray) -> np.ndarray:
        """Construct graph from correlation matrix."""
        corr_matrix = np.corrcoef(X.T)
        # Threshold weak correlations
        adj_matrix = np.abs(corr_matrix)
        adj_matrix[adj_matrix < self.threshold] = 0
        # Remove self-loops
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    def _partial_correlation_graph(self, X: np.ndarray) -> np.ndarray:
        """Construct graph from partial correlations using Graphical Lasso."""
        try:
            model = GraphicalLasso(alpha=0.1, max_iter=500)
            model.fit(X)
            precision_matrix = model.precision_
            # Convert precision to partial correlation
            d = np.sqrt(np.diag(precision_matrix))
            partial_corr = -precision_matrix / np.outer(d, d)
            np.fill_diagonal(partial_corr, 0)
            # Threshold
            partial_corr[np.abs(partial_corr) < self.threshold] = 0
            return np.abs(partial_corr)
        except Exception:
            return self._correlation_graph(X)

    def _granger_causality_graph(self, X: np.ndarray) -> np.ndarray:
        """
        Simplified Granger-like causality based on predictive power.
        For cross-sectional data, uses conditional independence tests.
        """
        n_features = X.shape[1]
        adj_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    # Use correlation as proxy for causal strength
                    corr, p_value = stats.pearsonr(X[:, i], X[:, j])
                    if p_value < self.alpha:
                        adj_matrix[i, j] = abs(corr)

        # Threshold
        adj_matrix[adj_matrix < self.threshold] = 0
        return adj_matrix

    def get_causal_graph(self) -> np.ndarray:
        """Return the discovered causal adjacency matrix."""
        return self.adjacency_matrix

    def get_important_features(self, top_k: int = 10) -> List[int]:
        """Return indices of most important features."""
        return np.argsort(self.feature_importance)[-top_k:][::-1].tolist()


class CausalConstrainedLayer(nn.Module):
    """
    Neural network layer with causal structure constraints.
    Connections follow the discovered causal graph.
    """

    def __init__(self, in_features: int, out_features: int,
                 causal_mask: Optional[torch.Tensor] = None):
        """
        Initialize causal-constrained layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            causal_mask: Binary mask encoding allowed connections
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Causal mask (learnable if not provided)
        if causal_mask is not None:
            self.register_buffer('causal_mask', causal_mask)
        else:
            self.register_buffer('causal_mask', torch.ones(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal constraints."""
        # Apply causal mask to weights
        masked_weight = self.weight * self.causal_mask
        return F.linear(x, masked_weight, self.bias)


class TemporalCausalNetwork(nn.Module):
    """
    Neural network that respects causal structure.
    Combines causal discovery with neural network learning.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 causal_graph: Optional[np.ndarray] = None,
                 dropout_rate: float = 0.3):
        """
        Initialize temporal causal network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            causal_graph: Adjacency matrix of causal graph
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Convert causal graph to mask if provided
        if causal_graph is not None:
            # Create mask: each output neuron gets input from causally related features
            # For first layer, use causal graph; for others, full connections
            self.causal_mask = torch.FloatTensor(causal_graph + np.eye(input_dim))
            self.causal_mask = (self.causal_mask > 0).float()
        else:
            self.causal_mask = None

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0 and self.causal_mask is not None:
                # First layer uses causal constraints
                layers.append(CausalConstrainedLayer(
                    prev_dim, hidden_dim,
                    self._expand_mask(self.causal_mask, hidden_dim)
                ))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))

            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def _expand_mask(self, mask: torch.Tensor, out_features: int) -> torch.Tensor:
        """Expand causal mask to match layer dimensions."""
        # Each output neuron sees a subset of inputs based on causal structure
        # Simple expansion: tile the mask
        in_features = mask.shape[0]
        expanded = mask.unsqueeze(0).repeat(out_features, 1, 1)
        # Take mean across feature groups
        return expanded.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class CausalNeuralClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible causal neural network classifier.
    """

    def __init__(self, hidden_dims: List[int] = [64, 32],
                 causal_method: str = 'correlation',
                 causal_threshold: float = 0.3,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 early_stopping_patience: int = 15,
                 class_weight: Optional[Dict] = None,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Initialize causal neural classifier.
        """
        self.hidden_dims = hidden_dims
        self.causal_method = causal_method
        self.causal_threshold = causal_threshold
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.causal_graph = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'CausalNeuralClassifier':
        """
        Train the causal neural network.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Discover causal structure
        causal_discovery = CausalGraphDiscovery(
            method=self.causal_method,
            threshold=self.causal_threshold,
            random_state=self.random_state
        )
        causal_discovery.fit(X, y)
        self.causal_graph = causal_discovery.get_causal_graph()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # Create validation set if not provided
        if X_val is None:
            n_val = int(0.15 * len(X))
            indices = np.random.permutation(len(X))
            val_idx, train_idx = indices[:n_val], indices[n_val:]

            X_train = X_tensor[train_idx]
            y_train = y_tensor[train_idx]
            X_val_t = X_tensor[val_idx]
            y_val_t = y_tensor[val_idx]
        else:
            X_train = X_tensor
            y_train = y_tensor
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Initialize model
        self.model = TemporalCausalNetwork(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            causal_graph=self.causal_graph,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Loss function with class weighting
        if self.class_weight is not None:
            pos_weight = torch.tensor([self.class_weight.get(1, 1.0) /
                                       self.class_weight.get(0, 1.0)]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCELoss()

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                       lr=self.learning_rate,
                                       weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()

        return np.hstack([1 - proba, proba])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_causal_info(self) -> Dict:
        """Get information about discovered causal structure."""
        if self.causal_graph is None:
            return {}

        n_edges = np.sum(self.causal_graph > 0)
        max_weight = np.max(self.causal_graph)

        return {
            'n_features': self.causal_graph.shape[0],
            'n_edges': n_edges,
            'density': n_edges / (self.causal_graph.shape[0] ** 2),
            'max_edge_weight': max_weight,
            'causal_graph': self.causal_graph
        }


if __name__ == "__main__":
    # Test causal learning
    print("Testing Causal Structure Learning...")

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Load data
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target  # Flip for convention

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Test causal discovery
    discovery = CausalGraphDiscovery(method='correlation', threshold=0.3)
    discovery.fit(X_train)
    print(f"Discovered causal graph with {np.sum(discovery.adjacency_matrix > 0)} edges")
    print(f"Top important features: {discovery.get_important_features(5)}")

    # Train causal neural classifier
    clf = CausalNeuralClassifier(
        hidden_dims=[64, 32],
        causal_method='correlation',
        n_epochs=50,
        verbose=True
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Get causal info
    causal_info = clf.get_causal_info()
    print(f"\nCausal graph: {causal_info['n_edges']} edges, density={causal_info['density']:.3f}")
