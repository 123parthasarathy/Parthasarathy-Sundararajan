"""
Quantum-Inspired Neural Network (QINN) Module
Implements quantum-mimetic optimization for neural network training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')


class QuantumRotationGate:
    """
    Quantum-inspired rotation gate for parameter optimization.
    Implements rotation in the probability amplitude space.
    """

    def __init__(self, delta_theta: float = 0.01 * np.pi):
        """
        Initialize rotation gate.

        Args:
            delta_theta: Base rotation angle
        """
        self.delta_theta = delta_theta

    def compute_rotation(self, alpha: float, beta: float,
                         fitness_current: float, fitness_best: float,
                         x_current: int, x_best: int) -> Tuple[float, float]:
        """
        Compute rotation for quantum bit based on fitness comparison.

        Args:
            alpha, beta: Current probability amplitudes
            fitness_current: Fitness of current solution
            fitness_best: Best known fitness
            x_current: Current binary value
            x_best: Best binary value

        Returns:
            Updated (alpha, beta) amplitudes
        """
        # Determine rotation direction
        if fitness_current < fitness_best:
            if x_current == 0 and x_best == 1:
                sign = 1 if alpha * beta > 0 else -1
            elif x_current == 1 and x_best == 0:
                sign = -1 if alpha * beta > 0 else 1
            else:
                sign = 0
        else:
            sign = 0

        # Apply rotation
        theta = sign * self.delta_theta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        alpha_new = cos_theta * alpha - sin_theta * beta
        beta_new = sin_theta * alpha + cos_theta * beta

        # Normalize
        norm = np.sqrt(alpha_new**2 + beta_new**2)
        if norm > 0:
            alpha_new /= norm
            beta_new /= norm

        return alpha_new, beta_new


class QuantumInspiredLayer(nn.Module):
    """
    Neural network layer with quantum-inspired activation.
    Uses superposition-like computation for enhanced expressiveness.
    """

    def __init__(self, in_features: int, out_features: int,
                 n_quantum_states: int = 2):
        """
        Initialize quantum-inspired layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            n_quantum_states: Number of quantum states to superpose
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_quantum_states = n_quantum_states

        # Multiple weight matrices for superposition
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, out_features) * 0.01)
            for _ in range(n_quantum_states)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features))
            for _ in range(n_quantum_states)
        ])

        # Probability amplitudes (learnable)
        self.amplitudes = nn.Parameter(
            torch.ones(n_quantum_states) / np.sqrt(n_quantum_states)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantum superposition.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Normalize amplitudes to ensure |alpha|^2 + |beta|^2 = 1
        probs = F.softmax(self.amplitudes ** 2, dim=0)

        # Compute superposition of outputs
        output = torch.zeros(x.size(0), self.out_features, device=x.device)
        for i in range(self.n_quantum_states):
            state_output = F.linear(x, self.weights[i].t(), self.biases[i])
            output += probs[i] * state_output

        return output


class QuantumInspiredNeuralNetwork(nn.Module):
    """
    Full quantum-inspired neural network architecture.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 n_quantum_states: int = 2, dropout_rate: float = 0.3):
        """
        Initialize QINN.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            n_quantum_states: Number of quantum states per layer
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_quantum_states = n_quantum_states

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(QuantumInspiredLayer(prev_dim, hidden_dim, n_quantum_states))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (classical)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

    def get_quantum_amplitudes(self) -> Dict[str, np.ndarray]:
        """Get learned quantum amplitudes from all layers."""
        amplitudes = {}
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, QuantumInspiredLayer):
                probs = F.softmax(layer.amplitudes ** 2, dim=0)
                amplitudes[f'layer_{i}'] = probs.detach().cpu().numpy()
        return amplitudes


class QINNClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for Quantum-Inspired Neural Network.
    """

    def __init__(self, hidden_dims: List[int] = [64, 32],
                 n_quantum_states: int = 2,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 early_stopping_patience: int = 15,
                 class_weight: Optional[Dict] = None,
                 random_state: int = 42,
                 verbose: bool = False):
        """
        Initialize QINN classifier.

        Args:
            hidden_dims: Hidden layer dimensions
            n_quantum_states: Quantum states per layer
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            n_epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            class_weight: Class weights for imbalanced data
            random_state: Random seed
            verbose: Print training progress
        """
        self.hidden_dims = hidden_dims
        self.n_quantum_states = n_quantum_states
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'QINNClassifier':
        """
        Train the QINN model.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            self
        """
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

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
            X_val = X_tensor[val_idx]
            y_val = y_tensor[val_idx]
        else:
            X_train = X_tensor
            y_train = y_tensor
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True)

        # Initialize model
        self.model = QuantumInspiredNeuralNetwork(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            n_quantum_states=self.n_quantum_states,
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

        for epoch in range(self.n_epochs):
            # Training
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
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, 2)
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()

        return np.hstack([1 - proba, proba])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_quantum_info(self) -> Dict:
        """Get information about learned quantum states."""
        if self.model is None:
            return {}

        return {
            'amplitudes': self.model.get_quantum_amplitudes(),
            'n_quantum_states': self.n_quantum_states
        }


class QuantumEvolutionaryOptimizer:
    """
    Quantum-inspired evolutionary algorithm for hyperparameter optimization.
    """

    def __init__(self, n_qubits: int, n_population: int = 20,
                 n_generations: int = 50, random_state: int = 42):
        """
        Initialize optimizer.

        Args:
            n_qubits: Number of qubits (hyperparameters)
            n_population: Population size
            n_generations: Number of generations
            random_state: Random seed
        """
        self.n_qubits = n_qubits
        self.n_population = n_population
        self.n_generations = n_generations
        self.random_state = random_state

        np.random.seed(random_state)

        # Initialize quantum population (probability amplitudes)
        self.Q = np.ones((n_population, n_qubits, 2)) / np.sqrt(2)
        self.rotation_gate = QuantumRotationGate()

        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = []

    def measure(self, individual_idx: int) -> np.ndarray:
        """
        Measure quantum state to get classical binary solution.

        Args:
            individual_idx: Index of individual in population

        Returns:
            Binary solution array
        """
        solution = np.zeros(self.n_qubits, dtype=int)
        for j in range(self.n_qubits):
            alpha = self.Q[individual_idx, j, 0]
            # Probability of measuring |0>
            p0 = alpha ** 2
            if np.random.random() < p0:
                solution[j] = 0
            else:
                solution[j] = 1
        return solution

    def update(self, individual_idx: int, solution: np.ndarray,
               fitness: float) -> None:
        """
        Update quantum state based on fitness.

        Args:
            individual_idx: Index of individual
            solution: Current solution
            fitness: Fitness of solution
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()

        # Update each qubit
        for j in range(self.n_qubits):
            alpha, beta = self.Q[individual_idx, j]
            alpha_new, beta_new = self.rotation_gate.compute_rotation(
                alpha, beta, fitness, self.best_fitness,
                solution[j], self.best_solution[j]
            )
            self.Q[individual_idx, j] = [alpha_new, beta_new]

    def optimize(self, fitness_func) -> Tuple[np.ndarray, float]:
        """
        Run optimization.

        Args:
            fitness_func: Function that takes binary solution and returns fitness

        Returns:
            Best solution and fitness
        """
        for gen in range(self.n_generations):
            gen_best_fitness = -np.inf

            for i in range(self.n_population):
                # Measure to get solution
                solution = self.measure(i)
                # Evaluate fitness
                fitness = fitness_func(solution)
                # Update quantum state
                self.update(i, solution, fitness)

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness

            self.history.append(gen_best_fitness)

        return self.best_solution, self.best_fitness


if __name__ == "__main__":
    # Test QINN
    print("Testing Quantum-Inspired Neural Network...")

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train QINN
    qinn = QINNClassifier(
        hidden_dims=[64, 32],
        n_quantum_states=3,
        n_epochs=50,
        verbose=True
    )
    qinn.fit(X_train, y_train)

    # Evaluate
    y_pred = qinn.predict(X_test)
    y_proba = qinn.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Get quantum info
    quantum_info = qinn.get_quantum_info()
    print(f"\nQuantum amplitudes: {quantum_info['amplitudes']}")
