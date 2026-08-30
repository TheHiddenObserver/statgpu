"""Basic model wrappers (thin wrappers over PenalizedGLM / GLM base)."""

from ._linear import LinearRegression
from ._ridge import Ridge
from ._lasso import Lasso
from ._elasticnet import ElasticNet
from ._adaptive_lasso import AdaptiveLasso
from ._scad import SCADRegression
from ._mcp import MCPRegression
from ._logistic import LogisticRegression
from ._gamma import GammaRegression
from ._poisson import PoissonRegression
from ._inverse_gaussian import InverseGaussianRegression
from ._negative_binomial import NegativeBinomialRegression
from ._tweedie import TweedieRegression
from ._quantile import QuantileRegression

# LinearRegression has direct backend conversion/alignment outside the backend
# solver methods. Install a narrow fail-closed refit transaction after the class
# is defined so conversion/device failures cannot leave stale fitted state.
from . import _linear_fit_transaction_contract as _linear_fit_transaction_contract

__all__ = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "AdaptiveLasso",
    "SCADRegression",
    "MCPRegression",
    "LogisticRegression",
    "GammaRegression",
    "PoissonRegression",
    "InverseGaussianRegression",
    "NegativeBinomialRegression",
    "TweedieRegression",
    "QuantileRegression",
]
