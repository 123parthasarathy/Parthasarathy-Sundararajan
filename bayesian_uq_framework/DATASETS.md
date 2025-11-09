# Dataset Reference Guide

Complete reference for all datasets compatible with the Bayesian UQ Framework, including direct download links and loading instructions.

## Table of Contents
- [Regression Datasets](#regression-datasets)
- [Classification Datasets](#classification-datasets)
- [Large-Scale Datasets](#large-scale-datasets)
- [Download Instructions](#download-instructions)

---

## Regression Datasets

### 1. California Housing (Built-in)
- **Samples**: 20,640
- **Features**: 8
- **Target**: Median house value
- **Source**: scikit-learn built-in
- **Download**: Automatic

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
```

### 2. Concrete Compressive Strength
- **Samples**: 1,030
- **Features**: 8 (cement, water, age, etc.)
- **Target**: Compressive strength (MPa)
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls

```python
import pandas as pd
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
urllib.request.urlretrieve(url, 'Concrete_Data.xls')
df = pd.read_excel('Concrete_Data.xls')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

### 3. Energy Efficiency
- **Samples**: 768
- **Features**: 8 (building parameters)
- **Target**: Heating/cooling load
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx

```python
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
df = pd.read_excel(url)
X = df.iloc[:, :-2].values
y = df.iloc[:, -2].values  # Heating load
```

### 4. Combined Cycle Power Plant
- **Samples**: 9,568
- **Features**: 4 (temperature, pressure, humidity, vacuum)
- **Target**: Net hourly electrical energy output
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip

```python
import pandas as pd
import zipfile
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
urllib.request.urlretrieve(url, 'CCPP.zip')

with zipfile.ZipFile('CCPP.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

df = pd.read_excel('CCPP/Folds5x2_pp.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

### 5. Year Prediction MSD (Large-Scale)
- **Samples**: 515,345
- **Features**: 90 (audio features)
- **Target**: Year of release
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
- **Size**: ~400 MB

```python
import pandas as pd
import zipfile
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
print("Downloading (400MB)...")
urllib.request.urlretrieve(url, 'YearPredictionMSD.zip')

with zipfile.ZipFile('YearPredictionMSD.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

df = pd.read_csv('YearPredictionMSD.txt', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
```

### 6. Protein Tertiary Structure
- **Samples**: 45,730
- **Features**: 9
- **Target**: RMSD (size of residue)
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv

```python
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv'
df = pd.read_csv(url)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
```

### 7. Bike Sharing Dataset
- **Samples**: 17,389
- **Features**: 16 (weather, time, etc.)
- **Target**: Bike rental count
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip

```python
import pandas as pd
import zipfile
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
urllib.request.urlretrieve(url, 'Bike-Sharing-Dataset.zip')

with zipfile.ZipFile('Bike-Sharing-Dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

df = pd.read_csv('hour.csv')
X = df.iloc[:, 2:-3].values
y = df.iloc[:, -1].values
```

### 8. OpenML Datasets (via sklearn)

#### Diamonds (Large)
- **Samples**: 53,940
- **Dataset ID**: 42165
```python
from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=42165, as_frame=False, parser='auto')
X, y = data.data, data.target
```

#### Moneyball
- **Samples**: 1,232
- **Dataset ID**: 42571
```python
data = fetch_openml(data_id=42571, as_frame=False, parser='auto')
```

#### Wine Quality
- **Samples**: 6,497
- **Dataset ID**: 287
```python
data = fetch_openml(data_id=287, as_frame=False, parser='auto')
```

---

## Classification Datasets

### 1. MNIST (Built-in)
- **Samples**: 70,000 (60k train, 10k test)
- **Features**: 784 (28x28 images)
- **Classes**: 10 (digits 0-9)
- **Source**: torchvision
- **Download**: Automatic

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
```

### 2. Fashion-MNIST (Built-in)
- **Samples**: 70,000
- **Features**: 784
- **Classes**: 10 (clothing items)
- **Source**: torchvision
- **Download**: Automatic

```python
from torchvision import datasets
train_dataset = datasets.FashionMNIST('./data', train=True, download=True)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True)
```

### 3. CIFAR-10 (Built-in)
- **Samples**: 60,000
- **Features**: 3072 (32x32x3 images)
- **Classes**: 10
- **Source**: torchvision
- **Download**: Automatic

```python
from torchvision import datasets
train_dataset = datasets.CIFAR10('./data', train=True, download=True)
test_dataset = datasets.CIFAR10('./data', train=False, download=True)
```

### 4. CIFAR-100 (Built-in)
- **Samples**: 60,000
- **Features**: 3072
- **Classes**: 100
- **Source**: torchvision
- **Download**: Automatic

```python
from torchvision import datasets
train_dataset = datasets.CIFAR100('./data', train=True, download=True)
```

### 5. Credit Card Default
- **Samples**: 30,000
- **Features**: 23
- **Classes**: 2 (default/no default)
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls

```python
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
df = pd.read_excel(url, skiprows=1)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
```

### 6. Adult Income (Census)
- **Samples**: 48,842
- **Features**: 14 (mix of categorical and continuous)
- **Classes**: 2 (>50K, <=50K)
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

```python
import pandas as pd

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

# Preprocessing required for categorical features
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

### 7. Covertype (Large-Scale)
- **Samples**: 581,012
- **Features**: 54
- **Classes**: 7 (forest cover types)
- **Source**: UCI ML Repository
- **Download Link**: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
- **Size**: ~70 MB

```python
import pandas as pd
import gzip
import urllib.request

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
print("Downloading (70MB)...")
urllib.request.urlretrieve(url, 'covtype.data.gz')

with gzip.open('covtype.data.gz', 'rb') as f:
    df = pd.read_csv(f, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values - 1  # Convert to 0-indexed
```

### 8. OpenML Classification Datasets

#### Australian Credit
- **Samples**: 690
- **Dataset ID**: 40996
```python
from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=40996, as_frame=False, parser='auto')
```

#### Blood Transfusion
- **Samples**: 748
- **Dataset ID**: 1464
```python
data = fetch_openml(data_id=1464, as_frame=False, parser='auto')
```

---

## Large-Scale Datasets Summary

| Dataset | Task | Samples | Features | Size | Download Time* |
|---------|------|---------|----------|------|----------------|
| Year Prediction MSD | Regression | 515,345 | 90 | 400 MB | ~5-10 min |
| Covertype | Classification | 581,012 | 54 | 70 MB | ~2-5 min |
| Protein Structure | Regression | 45,730 | 9 | 5 MB | <1 min |
| Diamonds | Regression | 53,940 | 10 | 3 MB | <1 min |
| Adult Income | Classification | 48,842 | 14 | 5 MB | <1 min |
| Credit Card Default | Classification | 30,000 | 23 | 2 MB | <1 min |

*Approximate download times on typical broadband connection

---

## Download Instructions

### Method 1: Direct Download with Python

```python
import urllib.request
import pandas as pd

def download_dataset(url, filename):
    """Download dataset from URL."""
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved to {filename}")
    return filename

# Example
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
filename = download_dataset(url, 'Concrete_Data.xls')
df = pd.read_excel(filename)
```

### Method 2: Using sklearn (OpenML)

```python
from sklearn.datasets import fetch_openml

# List of interesting dataset IDs
datasets = {
    'diamonds': 42165,
    'wine_quality': 287,
    'moneyball': 42571,
    'vehicle_sensor': 41514,
    'australian': 40996,
    'blood_transfusion': 1464,
}

# Load any dataset
data = fetch_openml(data_id=datasets['diamonds'], as_frame=False, parser='auto')
X, y = data.data, data.target
```

### Method 3: Using torchvision

```python
from torchvision import datasets

# All these download automatically
mnist = datasets.MNIST('./data', download=True)
fashion = datasets.FashionMNIST('./data', download=True)
cifar10 = datasets.CIFAR10('./data', download=True)
cifar100 = datasets.CIFAR100('./data', download=True)
```

### Method 4: Kaggle API (for Kaggle datasets)

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API credentials (~/.kaggle/kaggle.json)
# Download dataset
kaggle competitions download -c house-prices-advanced-regression-techniques
kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009
```

---

## Dataset Preprocessing Template

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def prepare_dataset(X, y, test_size=0.2, normalize=True):
    """
    Standard preprocessing pipeline.

    Args:
        X: Features
        y: Target
        test_size: Fraction for test set
        normalize: Whether to normalize features

    Returns:
        X_train, X_test, y_train, y_test (all preprocessed)
    """
    # Handle categorical targets
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Example usage
X, y = load_your_data()
X_train, X_test, y_train, y_test = prepare_dataset(X, y)
```

---

## Troubleshooting

### SSL Certificate Errors

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Slow Downloads

```python
# Use mirror or cache
import requests
from io import BytesIO

def download_with_progress(url):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with BytesIO() as buffer:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            buffer.write(chunk)
            print(f"\rProgress: {downloaded/total*100:.1f}%", end='')

        buffer.seek(0)
        return buffer
```

### Memory Issues with Large Datasets

```python
# Load in chunks
import pandas as pd

chunk_size = 10000
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    processed = process(chunk)
    chunks.append(processed)

df = pd.concat(chunks)
```

---

## Citation Information

When using datasets, please cite appropriately:

**UCI ML Repository**:
```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
School of Information and Computer Science.
```

**OpenML**:
```
Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2014).
OpenML: networked science in machine learning. ACM SIGKDD Explorations
Newsletter, 15(2), 49-60.
```

---

## Additional Resources

- **UCI ML Repository**: https://archive.ics.uci.edu/ml/index.php
- **OpenML**: https://www.openml.org
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Papers with Code Datasets**: https://paperswithcode.com/datasets
- **Google Dataset Search**: https://datasetsearch.research.google.com

---

**Last Updated**: 2025
**Maintained by**: Bayesian UQ Framework Team
