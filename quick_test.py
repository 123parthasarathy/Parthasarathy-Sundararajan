"""
Quick Test Script

Tests the analysis pipeline with a smaller dataset to verify everything works.
Run this before running the full analysis.
"""

import os
import sys

# Modify the main script temporarily for quick testing
print("="*80)
print("QUICK VALIDATION TEST")
print("="*80)
print("\nThis will run a quick test with a smaller dataset to verify everything works.")
print("For full analysis, run: python medicinal_plants_diabetes_ml_analysis.py")
print()

# Read the main script
with open('medicinal_plants_diabetes_ml_analysis.py', 'r') as f:
    code = f.read()

# Modify for quick test: reduce dataset size and iterations
code = code.replace('n_samples = 25000', 'n_samples = 2000')
code = code.replace('n_estimators=300', 'n_estimators=50')
code = code.replace('iterations=300', 'iterations=50')
code = code.replace('epochs=100', 'epochs=20')
code = code.replace('n_bootstrap = 1000', 'n_bootstrap = 100')
code = code.replace('X[:1000]', 'X[:200]')

# Execute the modified code
print("\n[Starting quick validation test with reduced parameters...]")
print("  - Dataset: 2,000 samples (instead of 25,000)")
print("  - Tree models: 50 estimators (instead of 300)")
print("  - Neural network: 20 epochs (instead of 100)")
print("  - Bootstrap: 100 iterations (instead of 1000)")
print("  - SHAP: 200 samples (instead of 1000)")
print()

try:
    exec(code)
    print("\n" + "="*80)
    print("✓✓✓ VALIDATION TEST PASSED! ✓✓✓")
    print("="*80)
    print("\nThe code is working correctly!")
    print("\nTo run the full analysis with optimal parameters:")
    print("  python medicinal_plants_diabetes_ml_analysis.py")
    print("\nOr use the integrated runner:")
    print("  python run_complete_analysis.py")
    print()
except Exception as e:
    print("\n" + "="*80)
    print("✗ ERROR DURING TESTING")
    print("="*80)
    print(f"\nError: {e}")
    print("\nPlease check the error message and ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print()
    sys.exit(1)
