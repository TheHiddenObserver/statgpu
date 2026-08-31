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

# Ordinary squared-error L2 inference must reuse the fit's converted arrays and
# include pre-dispatch conversion/alignment in the same fail-closed transaction.
# Install this after the group hook so the two narrow wrappers compose.
from . import _gaussian_fit_transaction_contract as _gaussian_fit_transaction_contract

# Estimation-only GPU fits already release backend caches inside the executed
# backend path. Suppress only the redundant later cleanup while preserving
# failure cleanup and the inference-enabled cleanup contract above.
from . import _no_inference_cleanup_contract as _no_inference_cleanup_contract

# BaseEstimator's public finite-input guard can reject a refit before the
# estimation-only transaction wrapper is entered. Extend the reset hook after
# the cleanup contract is available so such failures also clear stale results.
from . import (
    _no_inference_public_validation_reset_contract
    as _no_inference_public_validation_reset_contract,
)

# Install strict penalized-Cox grid validation and restore the public class
# introspection contract after the estimator and survival CV modules exist.
from . import _penalized_cox_public_contract as _penalized_cox_public_contract

# Final review contracts compose outside the earlier fit/reset hooks: current-
# attempt device provenance is published before remaining conversions, public
# finite-validation cleanup is best effort, and exact weighted GPU inference
# retains raw outcomes for diagnostics without changing weighted numerics.
from . import _latest_review_fix_contract as _latest_review_fix_contract

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
