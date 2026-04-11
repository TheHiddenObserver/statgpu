"""
Linear models for regression and classification.
"""

from ._linear import LinearRegression
from ._logistic import LogisticRegression
from ._ridge import Ridge
from ._lasso import Lasso, LassoCV
from ._ridge_cv import RidgeCV
from ._logistic_cv import LogisticRegressionCV

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'LogisticRegressionCV',
    'Ridge',
    'RidgeCV',
    'Lasso',
    'LassoCV',
]
