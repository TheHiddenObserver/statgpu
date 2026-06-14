"""Basic model wrappers (thin wrappers over PenalizedGLM / GLM base)."""

from ._linear import LinearRegression
from ._ridge import Ridge
from ._lasso import Lasso
from ._elasticnet import ElasticNet
from ._logistic import LogisticRegression
from ._gamma import GammaRegression
from ._poisson import PoissonRegression
from ._inverse_gaussian import InverseGaussianRegression
from ._negative_binomial import NegativeBinomialRegression
from ._tweedie import TweedieRegression

__all__ = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "GammaRegression",
    "PoissonRegression",
    "InverseGaussianRegression",
    "NegativeBinomialRegression",
    "TweedieRegression",
]
