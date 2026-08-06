"""Penalized GLM models (split via mixin pattern)."""

from ._base import PenalizedGeneralizedLinearModel, SelectivePenalty
from ._penalized_linear import PenalizedLinearRegression
from ._penalized_logistic import PenalizedLogisticRegression
from ._penalized_poisson import PenalizedPoissonRegression
from ._penalized_gamma import PenalizedGammaRegression
from ._penalized_inverse_gaussian import PenalizedInverseGaussianRegression
from ._penalized_negative_binomial import PenalizedNegativeBinomialRegression
from ._penalized_tweedie import PenalizedTweedieRegression
from ._penalized_cv import PenalizedGLM_CV, ApproximateCVWarning

# Non-GLM penalized models (LossBase subclasses)
from ._penalized_robust import PenalizedRobustRegression
from ._penalized_quantile import PenalizedQuantileRegression
from ._penalized_cox import PenalizedCoxPHModel

# Install transactional group-penalty design-width validation only after the
# estimator and CV classes above are fully defined.  The hook patches their
# existing methods in place, so specialized subclasses and direct historical
# imports share the same contract.
from . import _group_penalty_model_contract as _group_penalty_model_contract

# Install strict penalized-Cox grid validation and restore the public class
# introspection contract after the estimator and survival CV modules exist.
from . import _penalized_cox_public_contract as _penalized_cox_public_contract

__all__ = [
    "PenalizedGeneralizedLinearModel",
    "SelectivePenalty",
    "PenalizedLinearRegression",
    "PenalizedLogisticRegression",
    "PenalizedPoissonRegression",
    "PenalizedGammaRegression",
    "PenalizedInverseGaussianRegression",
    "PenalizedNegativeBinomialRegression",
    "PenalizedTweedieRegression",
    "PenalizedRobustRegression",
    "PenalizedQuantileRegression",
    "PenalizedCoxPHModel",
    "PenalizedGLM_CV",
    "ApproximateCVWarning",
]
