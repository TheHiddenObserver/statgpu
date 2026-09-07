"""
Linear models for regression and classification.
"""

# Wrappers (basic model classes)
from .wrappers import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    AdaptiveLasso,
    SCADRegression,
    MCPRegression,
    LogisticRegression,
    GammaRegression,
    PoissonRegression,
    InverseGaussianRegression,
    NegativeBinomialRegression,
    TweedieRegression,
    QuantileRegression,
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

# Non-GLM penalized models
from .penalized._penalized_robust import PenalizedRobustRegression
from .penalized._penalized_quantile import PenalizedQuantileRegression
from .penalized._penalized_cox import PenalizedCoxPHModel

# CV models
from .cv import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from .penalized._penalized_cv import PenalizedGLM_CV, ApproximateCVWarning

# Ordered models
from ._ordered_logit import OrderedLogitRegression
from ._ordered_probit import OrderedProbitRegression

# Shared Gaussian helpers support explicit Torch CPU execution when no native
# tensor or concrete device is available to select a CUDA device.
from . import _gaussian_inference_device_contract as _gaussian_inference_device_contract

# The unified penalized engine uses one backend-neutral direct-fit `solver`.
# Keep the historical `cpu_solver` constructor argument for one compatibility
# cycle, but make meaningful legacy use visibly deprecated. LassoCV owns the
# separate CV-path migration to `cv_solver`.
from . import _penalized_solver_api_contract as _penalized_solver_api_contract

_penalized_solver_api_contract.install_penalized_solver_api_contract()

# Keep statistical method identity separate from execution hardware. The
# historical cpu_ols/gpu_ols values remain one-cycle compatibility aliases for
# the canonical post_selection_ols inference method.
from . import _penalized_inference_api_contract as _penalized_inference_api_contract

_penalized_inference_api_contract.install_penalized_inference_api_contract()

# A fresh full-diff review after physical acceptance found two cross-path gaps:
# generic squared-error sparse estimators were not receiving the same migration
# contract, and the first weighted GPU objective fix disabled backend-native
# debiased inference. Install the focused closure after the main API migration.
from . import (
    _post_selection_ols_review_fix_contract as _post_selection_ols_review_fix_contract,
)

_post_selection_ols_review_fix_contract.install_post_selection_ols_review_fix_contract()

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'LogisticRegressionCV',
    'PoissonRegression',
    'GammaRegression',
    'InverseGaussianRegression',
    'NegativeBinomialRegression',
    'TweedieRegression',
    'QuantileRegression',
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
    'PenalizedRobustRegression',
    'PenalizedQuantileRegression',
    'PenalizedCoxPHModel',
    'AdaptiveLasso',
    'SCADRegression',
    'MCPRegression',
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
