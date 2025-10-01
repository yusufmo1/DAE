"""
K-Nearest Neighbors (KNN) imputation module.
"""

from .imputation import KNNImputer
from . import evaluate
from . import plots

__all__ = ['KNNImputer', 'evaluate', 'plots']
