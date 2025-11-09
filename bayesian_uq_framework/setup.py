"""
Setup script for Bayesian UQ Framework
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bayesian-uq-framework",
    version="1.0.0",
    author="Bayesian UQ Research Team",
    author_email="research@example.com",
    description="Bayesian Uncertainty Quantification in Deep Learning with Manifold-Based Reliability Assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bayesian-uq-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
        "all": [
            "tensorflow>=2.10.0",
            "tensorflow-probability>=0.18.0",
            "gpytorch>=1.9.0",
            "pyro-ppl>=1.8.0",
        ]
    },
    include_package_data=True,
    keywords=[
        "bayesian",
        "uncertainty quantification",
        "deep learning",
        "manifold learning",
        "reliability assessment",
        "monte carlo dropout",
        "deep ensembles",
        "variational inference",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bayesian-uq-framework/issues",
        "Source": "https://github.com/yourusername/bayesian-uq-framework",
        "Documentation": "https://bayesian-uq-framework.readthedocs.io",
    },
)
