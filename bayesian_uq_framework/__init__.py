"""
Bayesian Uncertainty Quantification in Deep Learning:
A Manifold-Based Reliability Assessment Framework

This framework combines:
- Bayesian inference for neural networks
- Uncertainty quantification (aleatoric & epistemic)
- Manifold learning for reliability assessment
- Comprehensive visualization tools
"""

__version__ = "1.0.0"
__author__ = "Bayesian UQ Research Team"

from .models import *
from .uncertainty import *
from .manifold import *
from .reliability import *

__all__ = ['models', 'uncertainty', 'manifold', 'reliability', 'utils']
