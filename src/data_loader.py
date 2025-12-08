"""
Data Loading and Preprocessing Module for Cancer Classification
Uses REAL datasets from UCI ML Repository with direct download
Supports multiple real-world datasets with various imbalance ratios
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, List, Optional
import urllib.request
import os
import warnings

warnings.filterwarnings('ignore')

# Direct download URLs for real datasets
UCI_WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
UCI_WPBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"
UCI_WBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"


class CancerDataLoader:
    """
    Comprehensive data loader for cancer classification datasets.
    Downloads REAL data directly from UCI ML Repository.
    Supports WDBC, WPBC, and WBC datasets with various imbalance scenarios.
    """

    SUPPORTED_DATASETS = ['wdbc', 'wdbc_imbalanced', 'wpbc', 'wbc', 'wdbc_uci']

    # Feature names for WDBC dataset (30 features)
    WDBC_FEATURE_NAMES = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    def __init__(self, dataset_name: str = 'wdbc', random_state: int = 42,
                 data_dir: str = './data'):
        """
        Initialize the data loader.

        Args:
            dataset_name: Name of dataset to load
            random_state: Random seed for reproducibility
            data_dir: Directory to store downloaded data
        """
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = None

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def _download_file(self, url: str, filename: str) -> str:
        """
        Download file from URL if not already present.

        Args:
            url: URL to download from
            filename: Local filename to save as

        Returns:
            Path to downloaded file
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from UCI ML Repository...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")
        return filepath

    def load_wdbc_from_uci(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Download and load WDBC directly from UCI ML Repository.

        UCI ML Repository URL:
        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

        Dataset format:
        - Column 0: ID number
        - Column 1: Diagnosis (M = malignant, B = benign)
        - Columns 2-31: 30 real-valued input features

        Returns:
            X: Feature matrix (569 samples, 30 features)
            y: Labels (0=benign, 1=malignant)
            feature_names: List of feature names
        """
        # Download the file
        filepath = self._download_file(UCI_WDBC_URL, "wdbc.data")

        # Load the data
        data = pd.read_csv(filepath, header=None)

        # Extract features (columns 2-31) and labels (column 1)
        X = data.iloc[:, 2:].values
        y_str = data.iloc[:, 1].values

        # Convert labels: M=1 (malignant), B=0 (benign)
        y = np.array([1 if label == 'M' else 0 for label in y_str])

        self.feature_names = self.WDBC_FEATURE_NAMES
        self.class_names = ['Benign', 'Malignant']

        print(f"Loaded WDBC from UCI: {len(y)} samples, {X.shape[1]} features")
        print(f"Class distribution: Benign={np.sum(y==0)}, Malignant={np.sum(y==1)}")

        return X, y, self.feature_names

    def load_wbc_from_uci(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Download and load WBC (Wisconsin Breast Cancer) from UCI.
        This is the original dataset with 9 features.

        Returns:
            X: Feature matrix
            y: Labels (0=benign, 1=malignant)
            feature_names: List of feature names
        """
        filepath = self._download_file(UCI_WBC_URL, "wbc.data")

        # Load data
        column_names = ['id', 'clump_thickness', 'cell_size_uniformity',
                        'cell_shape_uniformity', 'marginal_adhesion',
                        'single_epithelial_cell_size', 'bare_nuclei',
                        'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

        data = pd.read_csv(filepath, header=None, names=column_names, na_values='?')

        # Remove rows with missing values
        data = data.dropna()

        # Extract features and labels
        X = data.iloc[:, 1:-1].values.astype(float)
        y = (data.iloc[:, -1].values == 4).astype(int)  # 4=malignant, 2=benign

        self.feature_names = column_names[1:-1]
        self.class_names = ['Benign', 'Malignant']

        print(f"Loaded WBC from UCI: {len(y)} samples, {X.shape[1]} features")

        return X, y, self.feature_names

    def load_wdbc(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load Wisconsin Diagnostic Breast Cancer dataset.
        Uses sklearn's built-in dataset (same as UCI WDBC).

        UCI ML Repository:
        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

        Citation:
        Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995).
        Breast Cancer Wisconsin (Diagnostic) Data Set.
        UCI Machine Learning Repository.

        Returns:
            X: Feature matrix (569 samples, 30 features)
            y: Labels (0=benign, 1=malignant)
            feature_names: List of feature names
        """
        data = load_breast_cancer()
        X = data.data
        # Original sklearn: 0=malignant, 1=benign
        # We convert to: 0=benign (negative), 1=malignant (positive)
        # This matches clinical convention where positive=disease
        y = 1 - data.target  # Flip labels

        self.feature_names = list(data.feature_names)
        self.class_names = ['Benign', 'Malignant']

        return X, y, self.feature_names

    def create_imbalanced_dataset(self, X: np.ndarray, y: np.ndarray,
                                   imbalance_ratio: float = 6.99,
                                   minority_class: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create imbalanced dataset by undersampling one class.

        Args:
            X: Feature matrix
            y: Labels
            imbalance_ratio: Desired majority/minority ratio
            minority_class: Which class to make minority (0 or 1)

        Returns:
            X_imb, y_imb: Imbalanced dataset
        """
        np.random.seed(self.random_state)

        # Separate classes
        majority_class = 1 - minority_class
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]

        # Calculate how many minority samples to keep
        n_minority_target = int(len(X_majority) / imbalance_ratio)
        n_minority_target = max(n_minority_target, 10)  # Keep at least 10

        if n_minority_target < len(X_minority):
            # Undersample minority class
            indices = np.random.choice(len(X_minority), n_minority_target, replace=False)
            X_minority = X_minority[indices]
            y_minority = y_minority[indices]

        # Combine
        X_imb = np.vstack([X_majority, X_minority])
        y_imb = np.hstack([y_majority, y_minority])

        # Shuffle
        shuffle_idx = np.random.permutation(len(y_imb))

        return X_imb[shuffle_idx], y_imb[shuffle_idx]

    def load_data(self, imbalance_ratio: Optional[float] = None) -> Dict:
        """
        Load dataset based on configuration.
        Uses REAL data from UCI ML Repository.

        Args:
            imbalance_ratio: If provided, creates imbalanced version

        Returns:
            Dictionary containing data and metadata
        """
        if self.dataset_name in ['wdbc', 'wdbc_imbalanced']:
            X, y, feature_names = self.load_wdbc()

            if self.dataset_name == 'wdbc_imbalanced' or imbalance_ratio is not None:
                ratio = imbalance_ratio if imbalance_ratio else 6.99
                X, y = self.create_imbalanced_dataset(X, y, ratio, minority_class=0)

        elif self.dataset_name == 'wdbc_uci':
            # Direct download from UCI ML Repository
            X, y, feature_names = self.load_wdbc_from_uci()
            if imbalance_ratio is not None:
                X, y = self.create_imbalanced_dataset(X, y, imbalance_ratio, minority_class=0)

        elif self.dataset_name == 'wbc':
            # Original WBC dataset with 9 features
            X, y, feature_names = self.load_wbc_from_uci()
            if imbalance_ratio is not None:
                X, y = self.create_imbalanced_dataset(X, y, imbalance_ratio, minority_class=0)

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Supported: {self.SUPPORTED_DATASETS}")

        # Calculate statistics
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        actual_ratio = max(n_positive, n_negative) / min(n_positive, n_negative)

        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'class_names': self.class_names,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_positive': n_positive,
            'n_negative': n_negative,
            'imbalance_ratio': actual_ratio,
            'positive_rate': n_positive / len(y)
        }

    def get_cross_validation_splits(self, X: np.ndarray, y: np.ndarray,
                                     n_splits: int = 10) -> List[Tuple]:
        """
        Generate stratified cross-validation splits.

        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV folds

        Returns:
            List of (train_idx, test_idx) tuples
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=self.random_state)
        return list(skf.split(X, y))

    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray,
                   method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features using specified scaling method.

        Args:
            X_train: Training features
            X_test: Test features
            method: 'standard' or 'minmax'

        Returns:
            Scaled X_train, X_test
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled


def get_dataset_info(dataset_name: str = 'wdbc') -> None:
    """Print dataset information."""
    loader = CancerDataLoader(dataset_name)
    data = loader.load_data()

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Total samples: {data['n_samples']}")
    print(f"Number of features: {data['n_features']}")
    print(f"Positive (Malignant): {data['n_positive']} ({data['positive_rate']*100:.1f}%)")
    print(f"Negative (Benign): {data['n_negative']} ({(1-data['positive_rate'])*100:.1f}%)")
    print(f"Imbalance ratio: {data['imbalance_ratio']:.2f}:1")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")

    # Test WDBC
    get_dataset_info('wdbc')

    # Test imbalanced WDBC
    loader = CancerDataLoader('wdbc')
    data = loader.load_data(imbalance_ratio=6.99)
    print(f"Imbalanced WDBC: {data['n_positive']} positive, {data['n_negative']} negative")
    print(f"Imbalance ratio: {data['imbalance_ratio']:.2f}:1")
