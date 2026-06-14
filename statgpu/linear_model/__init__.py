"""
Linear models for regression and classification.
"""

# Wrappers (basic model classes)
from .wrappers import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    GammaRegression,
    PoissonRegression,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    TweedieRegression,
)

# GLM base
from ._glm_base import GeneralizedLinearModel, OrderedGeneralizedLinearModel

# Penalized models
from .penalized import PenalizedGeneralizedLinearModel
from .penalized._penalized_linear import PenalizedLinearRegression
from .penalized._penalized_logistic import PenalizedLogisticRegression
from .penalized._penalized_poisson import PenalizedPoissonRegression
from .penalized._penalized_gamma import PenalizedGammaRegression
from .penalized._penalized_inverse_gaussian import PenalizedInverseGaussianRegression
from .penalized._penalized_negative_binomial import PenalizedNegativeBinomialRegression
from .penalized._penalized_tweedie import PenalizedTweedieRegression

# CV models
from .cv import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from .penalized._penalized_cv import PenalizedGLM_CV, ApproximateCVWarning

# Ordered models
from ._ordered_logit import OrderedLogitRegression
from ._ordered_probit import OrderedProbitRegression

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'LogisticRegressionCV',
    'PoissonRegression',
    'GammaRegression',
    'InverseGaussianRegression',
    'NegativeBinomialRegression',
    'TweedieRegression',
    'GeneralizedLinearModel',
    'OrderedGeneralizedLinearModel',
    'PenalizedGeneralizedLinearModel',
    'PenalizedGLM_CV',
    'PenalizedLinearRegression',
    'PenalizedLogisticRegression',
    'PenalizedPoissonRegression',
    'PenalizedGammaRegression',
    'PenalizedInverseGaussianRegression',
    'PenalizedNegativeBinomialRegression',
    'PenalizedTweedieRegression',
    'Ridge',
    'RidgeCV',
    'Lasso',
    'LassoCV',
    'ElasticNet',
    'ElasticNetCV',
    'OrderedLogitRegression',
    'OrderedProbitRegression',
    'ApproximateCVWarning',
]
