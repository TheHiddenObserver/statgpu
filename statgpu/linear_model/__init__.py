"""
Linear models for regression and classification.
"""

from ._linear import LinearRegression
from ._logistic import LogisticRegression
from ._ridge import Ridge
from ._lasso import Lasso

__all__ = ['LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso']
