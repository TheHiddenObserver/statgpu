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

# The outer finite guard can run while inner Gaussian/no-inference transactions
# have installed cleanup delegates that still see prior-fit device provenance.
# Capture direct cleanup callables outside that wrapper stack and use the
# finite-check exception's concrete device when validation fails.
from . import (
    _final_finite_validation_cleanup_contract
    as _final_finite_validation_cleanup_contract,
)

# Quantile L2/no-penalty fits historically reported FISTA while the FISTA branch
# internally substituted IRLS. Reconcile the runtime policy after the generic
# estimator/CV classes are fully defined so direct and CV consumers share the
# same truthful solver identity.
from . import _quantile_solver_contract as _quantile_solver_contract

# Generic FISTA-BB and shared ADMM require smooth-gradient structure that
# Quantile/check loss does not provide. Install the narrow additional boundary
# after the main Quantile contract so existing, more-specific rejection
# semantics remain authoritative and only previously-open rows are closed.
from . import (
    _quantile_unsupported_solver_guard_contract
    as _quantile_unsupported_solver_guard_contract,
)

# The provenance repair above intentionally failed explicit smooth-Quantile
# FISTA closed because the historical FISTA branch actually executed IRLS.
# Complete that public capability after unsupported FISTA-BB/ADMM boundaries
# are installed: auto still prefers IRLS, while an explicit ordinary FISTA
# request now reaches the generic FISTA engine without silent substitution.
from . import (
    _quantile_smooth_fista_contract as _quantile_smooth_fista_contract,
)

# The non-convex Quantile solver receives an automatically generated
# continuation path. Mark that internal path after the solver-support contracts
# are installed so the public Proximal IRLS-CD boundary can align its start with
# non-uniform analytic weights without rewriting user-supplied low-level paths.
from . import (
    _quantile_continuation_contract as _quantile_continuation_contract,
)

# Group SCAD/MCP use the canonical group-aware LLA surrogate. Install this last
# so it sees the fully composed Quantile solver/weight contracts and can restore
# the documented Quantile Group FISTA-LLA route without perturbing other losses.
from . import _quantile_group_lla_contract as _quantile_group_lla_contract

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
