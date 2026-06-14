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
from ._penalized import PenalizedGeneralizedLinearModel
from ._penalized_linear import PenalizedLinearRegression
from ._penalized_logistic import PenalizedLogisticRegression
from ._penalized_poisson import PenalizedPoissonRegression
from ._penalized_gamma import PenalizedGammaRegression
from ._penalized_inverse_gaussian import PenalizedInverseGaussianRegression
from ._penalized_negative_binomial import PenalizedNegativeBinomialRegression
from ._penalized_tweedie import PenalizedTweedieRegression

# CV models
from ._lasso_cv import LassoCV
from ._ridge_cv import RidgeCV
from ._elasticnet_cv import ElasticNetCV
from ._logistic_cv import LogisticRegressionCV
from ._penalized_cv import PenalizedGLM_CV, ApproximateCVWarning

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
