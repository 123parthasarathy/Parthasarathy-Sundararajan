# How to Copy and Run in Spyder IDE

## 📋 **OPTION 1: Direct Copy to Your Computer (Recommended for Spyder)**

### **Step 1: Download from GitHub**

```bash
# Method A: Using Git
git clone https://github.com/123parthasarathy/Parthasarathy-Sundararajan.git
cd Parthasarathy-Sundararajan/bayesian_uq_framework
```

Or download the ZIP:
1. Go to: https://github.com/123parthasarathy/Parthasarathy-Sundararajan
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Navigate to `Parthasarathy-Sundararajan/bayesian_uq_framework/`

### **Step 2: Install Dependencies in Spyder**

Open Spyder, then in the IPython console:

```python
# Install required packages
!pip install numpy scipy torch torchvision scikit-learn umap-learn matplotlib seaborn pandas openpyxl
```

Or use Anaconda Navigator:
- Go to Environments → Select your environment
- Search and install: numpy, scipy, pytorch, torchvision, scikit-learn, umap-learn, matplotlib, seaborn, pandas, openpyxl

### **Step 3: Open Files in Spyder**

In Spyder:
1. **File → Open** → Navigate to `bayesian_uq_framework/examples/example_regression.py`
2. Click **Run** (F5) or click the green play button

---

## 📂 **OPTION 2: Copy Individual Files**

If you want to create your own script in Spyder, here are the key code snippets:

### **Create a New File in Spyder: `my_bayesian_analysis.py`**

```python
"""
Complete Bayesian UQ Analysis in Spyder IDE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# BAYESIAN NEURAL NETWORK - MONTE CARLO DROPOUT
# ============================================================================

class MCDropoutNN(nn.Module):
    """Monte Carlo Dropout Neural Network"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(MCDropoutNN, self).__init__()

        self.dropout_rate = dropout_rate
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def mc_predict(self, x, n_samples=100):
        """Get predictions with uncertainty"""
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


# ============================================================================
# MANIFOLD ANALYSIS WITH UMAP
# ============================================================================

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class ManifoldAnalyzer:
    """Analyze data in manifold space"""

    def __init__(self, method='umap'):
        self.method = method
        self.reducer = None

    def fit_transform(self, features, n_components=2):
        """Create manifold embedding"""
        if self.method == 'umap' and UMAP_AVAILABLE:
            self.reducer = umap.UMAP(n_components=n_components, random_state=42)
        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=n_components, random_state=42)
        elif self.method == 'pca':
            self.reducer = PCA(n_components=n_components, random_state=42)
        else:
            print("Using PCA as fallback")
            self.reducer = PCA(n_components=n_components, random_state=42)

        return self.reducer.fit_transform(features)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_predictions_with_uncertainty(x, predictions, uncertainties, true_values=None):
    """Plot predictions with uncertainty bands"""
    plt.figure(figsize=(12, 6))

    # Sort by x for better visualization
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    pred_sorted = predictions[sort_idx]
    unc_sorted = uncertainties[sort_idx]

    # Plot uncertainty band
    plt.fill_between(x_sorted,
                     pred_sorted - 2*unc_sorted,
                     pred_sorted + 2*unc_sorted,
                     alpha=0.3, label='95% Confidence')

    # Plot predictions
    plt.plot(x_sorted, pred_sorted, 'b-', linewidth=2, label='Prediction')

    # Plot true values
    if true_values is not None:
        true_sorted = true_values[sort_idx]
        plt.plot(x_sorted, true_sorted, 'r--', linewidth=2, label='True')

    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Bayesian Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_manifold_with_uncertainty(embedded, uncertainties):
    """Plot manifold colored by uncertainty"""
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                         c=uncertainties, cmap='viridis',
                         alpha=0.6, s=30)

    plt.colorbar(scatter, label='Uncertainty')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Manifold Embedding Colored by Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Complete Bayesian UQ Analysis"""

    print("="*80)
    print("BAYESIAN UNCERTAINTY QUANTIFICATION - Spyder IDE")
    print("="*80)

    # Load data
    print("\n1. Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # Create model
    print("\n2. Creating Bayesian Neural Network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    model = MCDropoutNN(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        output_dim=1,
        dropout_rate=0.2
    ).to(device)

    # Training
    print("\n3. Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=128,
        shuffle=True
    )

    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Predictions with uncertainty
    print("\n4. Computing predictions with uncertainty...")
    X_test_t = torch.FloatTensor(X_test).to(device)
    mean, std = model.mc_predict(X_test_t, n_samples=100)

    mean = mean.cpu().numpy().flatten()
    std = std.cpu().numpy().flatten()

    # Compute metrics
    errors = np.abs(mean - y_test)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"   RMSE: {rmse:.4f}")
    print(f"   Mean Uncertainty: {std.mean():.4f}")
    print(f"   Uncertainty-Error Correlation: {np.corrcoef(std, errors)[0,1]:.4f}")

    # Manifold analysis
    print("\n5. Creating manifold embedding...")
    manifold = ManifoldAnalyzer(method='umap' if UMAP_AVAILABLE else 'pca')
    embedded = manifold.fit_transform(X_test[:5000])

    # Visualizations
    print("\n6. Creating visualizations...")

    # Plot 1: Predictions with uncertainty
    plot_predictions_with_uncertainty(
        x=np.arange(100),
        predictions=mean[:100],
        uncertainties=std[:100],
        true_values=y_test[:100]
    )

    # Plot 2: Uncertainty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(std, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Uncertainty (Std Dev)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictive Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot 3: Uncertainty vs Error
    plt.figure(figsize=(10, 6))
    plt.scatter(std, errors, alpha=0.5, s=20)
    plt.plot([0, std.max()], [0, std.max()], 'r--', label='Perfect calibration')
    plt.xlabel('Predicted Uncertainty')
    plt.ylabel('Actual Error')
    plt.title('Uncertainty vs Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot 4: Manifold with uncertainty
    plot_manifold_with_uncertainty(embedded, std[:5000])

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    # Return results for further analysis in Spyder
    return {
        'model': model,
        'predictions': mean,
        'uncertainties': std,
        'true_values': y_test,
        'errors': errors,
        'manifold_embedding': embedded,
        'rmse': rmse
    }


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == '__main__':
    results = main()

    # Results are now available in Spyder's variable explorer
    print("\nResults saved in 'results' dictionary:")
    print("- results['predictions']")
    print("- results['uncertainties']")
    print("- results['true_values']")
    print("- results['manifold_embedding']")

    # You can now explore these in Spyder's Variable Explorer!
```

---

## 🎯 **OPTION 3: Use Jupyter Notebook in Spyder**

Spyder 5+ supports Jupyter notebooks. Create a notebook file:

1. In Spyder: **File → New file → Notebook**
2. Save as `bayesian_uq_analysis.ipynb`
3. Copy and paste code cells

---

## 📁 **OPTION 4: Set Python Path in Spyder**

To use the full framework in Spyder:

**Method 1: Via Spyder Console**
```python
import sys
sys.path.append('/path/to/Parthasarathy-Sundararajan')

from bayesian_uq_framework.models import MCDropoutNN
from bayesian_uq_framework.uncertainty import UncertaintyMetrics
# etc...
```

**Method 2: Via Spyder Settings**
1. **Tools → Preferences → Python Interpreter → PYTHONPATH Manager**
2. Click **Add path**
3. Browse to `/path/to/Parthasarathy-Sundararajan`
4. Click **OK**
5. Restart Spyder

---

## 🚀 **QUICK START IN SPYDER**

### **Step-by-Step:**

1. **Open Spyder IDE**

2. **Create new file:** `File → New File`

3. **Copy the complete code above** (from the "Create a New File" section)

4. **Save as:** `bayesian_analysis.py`

5. **Run:** Press **F5** or click the green **Run** button

6. **View results:**
   - Plots will appear in the Plots pane
   - Variables will appear in Variable Explorer
   - Console shows progress

---

## 💡 **TIPS FOR USING IN SPYDER**

### **1. Interactive Mode**
Run code section by section using `#%%` cells:

```python
#%% Load Data
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target

#%% Train Model
model = MCDropoutNN(input_dim=8, hidden_dims=[128, 64], output_dim=1)
# training code...

#%% Analyze Results
mean, std = model.mc_predict(X_test_t, n_samples=100)
plt.plot(mean.cpu().numpy())
```

### **2. View Variables**
All variables are visible in **Variable Explorer** pane:
- Double-click arrays to see data
- Right-click → Plot to visualize
- Right-click → Save to export

### **3. Debug Mode**
Set breakpoints and inspect:
- Click left of line number to set breakpoint
- Run with **Debug → Debug** (Ctrl+F5)
- Inspect variables at each step

---

## 📦 **FILES TO COPY**

If you downloaded from GitHub, copy these to your Spyder workspace:

```
Your Spyder Workspace/
├── my_bayesian_analysis.py          ← The complete code above
└── data/                              ← Auto-created when running
```

Or use the full framework:
```
Your Spyder Workspace/
├── bayesian_uq_framework/            ← Copy entire folder
│   ├── models/
│   ├── uncertainty/
│   ├── manifold/
│   ├── reliability/
│   ├── utils/
│   └── examples/
└── my_project.py                     ← Your code that imports framework
```

---

## ✅ **VERIFY INSTALLATION IN SPYDER**

Run this in Spyder console:

```python
# Test imports
import numpy as np
import torch
import sklearn
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("Scikit-learn version:", sklearn.__version__)

# Test UMAP (optional)
try:
    import umap
    print("UMAP available!")
except:
    print("UMAP not available - will use PCA/t-SNE")

print("\n✅ All core dependencies installed!")
```

---

## 🎓 **NEXT STEPS IN SPYDER**

1. **Run the example script** - See how it works
2. **Modify for your data** - Replace dataset loading
3. **Experiment with parameters** - Change hidden_dims, dropout_rate
4. **Export results** - Save plots and data
5. **Build your analysis** - Create custom pipelines

---

**The complete standalone code above will run directly in Spyder with no external files needed!** 🚀

Just copy the code into a new `.py` file in Spyder and click Run!
