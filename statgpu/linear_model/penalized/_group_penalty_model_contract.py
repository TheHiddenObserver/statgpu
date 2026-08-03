"""Transactional model contracts for public group penalties.

Penalty constructors can validate index syntax but do not know the eventual
design width. This module installs narrow in-place hooks so all specialized
estimators and historical direct imports share the same behavior:

- direct estimators clone externally supplied group penalty objects before
  design-width completion, so fitting never mutates constructor parameters;
- direct estimators validate/complete group coverage immediately after penalty
  resolution and before solver/backend work;
- direct and CV refits clear prior fitted state before validation and again on
  failure, so stale coefficients, selection results, or formula metadata cannot
  survive a rejected fit;
- pending private coefficient/intercept warm starts are preserved for exactly
  one fit call, then cleared on both success and failure;
- PenalizedGLM_CV validates/completes coverage before alpha-grid generation,
  fold construction, or candidate fitting, using fit-local penalty/kwargs state
  and restoring the original constructor parameters afterward;
- temporary CV penalty objects are rebuilt at each candidate's alpha, so object
  penalties and string penalties evaluate the same regularization grid;
- the selected final estimator exposes an unmarked penalty snapshot matching
  its resolved groups and selected alpha;
- every Group Lasso objective uses the actual loss gradient plus the exact
  Euclidean Group Lasso proximal operator. The historical Gaussian block update
  is bypassed because its inverse-Gram-then-threshold formula is exact only for
  orthonormal group blocks, a condition the public design does not require;
- bypassing that block update never overrides an explicitly requested generic
  proximal solver such as FISTA-BB or ADMM;
- all group penalties are explicitly estimation-only until a group-preserving
  inference implementation exists. In particular, the generic residual
  bootstrap is rejected because it currently refits ordinary L1 models.
"""

from __future__ import annotations

import copy

import numpy as np

from ._base import PenalizedGeneralizedLinearModel
from ._fit_mixin import _PenalizedFitMixin
from ._penalized_cv import PenalizedGLM_CV


_GROUP_PENALTY_NAMES = frozenset(
    {
        "group_lasso",
        "gl",
        "group_mcp",
        "gmcp",
        "group_scad",
        "gscad",
    }
)
_GROUP_LASSO_NAMES = frozenset({"group_lasso", "gl"})
_CV_ALPHA_MARKER = "_statgpu_cv_alpha_from_estimator"


def _public_penalty_name(estimator):
    return str(getattr(estimator.penalty, "name", estimator.penalty)).lower().strip()


def _clone_group_penalty(penalty, *, alpha=None):
    params = penalty.get_params(deep=False)
    if alpha is not None:
        params = dict(params)
        params["alpha"] = alpha
    try:
        return type(penalty)(**params)
    except Exception:
        if alpha is not None:
            raise
        clone = getattr(penalty, "clone", None)
        if callable(clone):
            return clone()
        return copy.deepcopy(penalty)


def _validate_resolved_group_penalty(penalty, n_features):
    validator = getattr(penalty, "validate_n_features", None)
    if validator is not None:
        validator(n_features)
    return penalty


def _resolved_penalty_name(estimator):
    return str(
        getattr(getattr(estimator, "_penalty", None), "name", estimator.penalty)
    ).lower().strip()


def _reset_direct_group_fit_state(estimator):
    """Clear all result-bearing state without changing constructor parameters."""
    estimator._penalty = None
    estimator._loss = None
    estimator.coef_ = None
    estimator.intercept_ = None
    estimator.n_iter_ = 0
    estimator._lla_n_iters_ = 0
    estimator._selected_solver = None
    estimator._selected_backend_name = None
    estimator._init_coef = None
    estimator._init_intercept = None
    estimator._feature_names = None
    estimator._design_info = None
    estimator._formula_has_intercept = None
    estimator._use_intercept = None
    estimator._inference_precomputed = False
    estimator._precomputed_gaussian_state = None
    estimator._conf_int_simultaneous = None
    estimator._simultaneous_enabled = False
    estimator._debiased_M_cpu = None
    estimator._clear_inference_state()
    estimator._fitted = False
    if hasattr(estimator, "n_features_in_"):
        delattr(estimator, "n_features_in_")


def _install_direct_fit_transaction():
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, "_statgpu_group_fit_transaction", False):
        return

    def _fit_with_group_transaction(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        current_name = _public_penalty_name(self)
        previous_name = _resolved_penalty_name(self)
        if (
            current_name not in _GROUP_PENALTY_NAMES
            and previous_name not in _GROUP_PENALTY_NAMES
        ):
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )

        pending_init_coef = getattr(self, "_init_coef", None)
        pending_init_intercept = getattr(self, "_init_intercept", None)
        _reset_direct_group_fit_state(self)
        self._init_coef = pending_init_coef
        self._init_intercept = pending_init_intercept
        try:
            result = current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        except Exception:
            _reset_direct_group_fit_state(self)
            raise
        self._init_coef = None
        self._init_intercept = None
        return result

    _fit_with_group_transaction._statgpu_group_fit_transaction = True
    _fit_with_group_transaction._statgpu_original = current
    PenalizedGeneralizedLinearModel.fit = _fit_with_group_transaction


def _install_direct_contract():
    current = PenalizedGeneralizedLinearModel._resolve_penalty
    if getattr(current, "_statgpu_group_contract", False):
        return

    def _resolve_penalty_with_group_contract(self):
        penalty = current(self)
        penalty_name = str(getattr(penalty, "name", "")).lower().strip()
        if penalty is self.penalty and penalty_name in _GROUP_PENALTY_NAMES:
            forced_alpha = (
                self.alpha
                if bool(getattr(penalty, _CV_ALPHA_MARKER, False))
                else None
            )
            penalty = _clone_group_penalty(penalty, alpha=forced_alpha)
        n_features = getattr(self, "n_features_in_", None)
        if n_features is not None:
            _validate_resolved_group_penalty(penalty, n_features)
        return penalty

    _resolve_penalty_with_group_contract._statgpu_group_contract = True
    _resolve_penalty_with_group_contract._statgpu_original = current
    PenalizedGeneralizedLinearModel._resolve_penalty = (
        _resolve_penalty_with_group_contract
    )


def _install_inference_contract():
    current = PenalizedGeneralizedLinearModel._validate_inference_request
    if getattr(current, "_statgpu_group_inference_contract", False):
        return

    def _validate_inference_with_group_contract(self):
        if self.compute_inference and _resolved_penalty_name(self) in _GROUP_PENALTY_NAMES:
            raise NotImplementedError(
                "Group Lasso, Group MCP, and Group SCAD are currently "
                "estimation-only. Group-preserving covariance/bootstrap "
                "inference is not implemented; set compute_inference=False."
            )
        return current(self)

    _validate_inference_with_group_contract._statgpu_group_inference_contract = True
    _validate_inference_with_group_contract._statgpu_original = current
    PenalizedGeneralizedLinearModel._validate_inference_request = (
        _validate_inference_with_group_contract
    )


def _cv_design_width(X):
    """Resolve public array-like design width without starting CV work."""
    shape = getattr(X, "shape", None)
    ndim = getattr(X, "ndim", None)
    if shape is not None and ndim == 2:
        return int(shape[1])
    try:
        host = np.asarray(X)
    except Exception:
        return None
    if host.ndim != 2:
        return None
    return int(host.shape[1])


def _prepare_cv_group_penalty(estimator, X):
    """Prepare fit-local group metadata without mutating constructor state."""
    penalty_name = _public_penalty_name(estimator)
    if penalty_name not in _GROUP_PENALTY_NAMES:
        return
    n_features = _cv_design_width(X)
    if n_features is None:
        return

    original_penalty = estimator.penalty
    is_penalty_object = getattr(original_penalty, "validate_n_features", None) is not None
    if is_penalty_object:
        penalty = _clone_group_penalty(original_penalty)
    else:
        from statgpu.penalties import get_penalty

        kwargs = dict(getattr(estimator, "_penalty_kwargs", None) or {})
        kwargs["alpha"] = 1.0
        penalty = get_penalty(penalty_name, **kwargs)

    _validate_resolved_group_penalty(penalty, n_features)

    if isinstance(original_penalty, str):
        kwargs = dict(getattr(estimator, "_penalty_kwargs", None) or {})
        kwargs["groups"] = penalty.groups
        estimator._penalty_kwargs = kwargs
        return

    setattr(penalty, _CV_ALPHA_MARKER, True)
    estimator.penalty = penalty


def _install_cv_contract():
    current = PenalizedGLM_CV.fit
    if getattr(current, "_statgpu_group_contract", False):
        return

    def _fit_with_group_contract(self, X, y, sample_weight=None):
        if _public_penalty_name(self) not in _GROUP_PENALTY_NAMES:
            return current(self, X, y, sample_weight=sample_weight)

        # The wrapped implementation also resets at entry and on failure. This
        # first reset is required because group coverage validation intentionally
        # runs before entering that implementation.
        self._reset_cv_fit_state()
        original_penalty = self.penalty
        original_penalty_kwargs = self._penalty_kwargs
        try:
            if str(self.loss).lower() != "cox_ph":
                _prepare_cv_group_penalty(self, X)
            result = current(self, X, y, sample_weight=sample_weight)
            if (
                not isinstance(original_penalty, str)
                and getattr(self, "estimator_", None) is not None
                and getattr(self.estimator_, "_penalty", None) is not None
            ):
                self.estimator_.penalty = _clone_group_penalty(
                    self.estimator_._penalty
                )
            return result
        except Exception:
            self._reset_cv_fit_state()
            raise
        finally:
            self.penalty = original_penalty
            self._penalty_kwargs = original_penalty_kwargs

    _fit_with_group_contract._statgpu_group_contract = True
    _fit_with_group_contract._statgpu_original = current
    PenalizedGLM_CV.fit = _fit_with_group_contract


def _install_exact_group_lasso_solver_contract():
    current = _PenalizedFitMixin._fit_loss_backend
    if getattr(current, "_statgpu_group_loss_contract", False):
        return

    def _fit_loss_backend_with_group_contract(
        self,
        X,
        y,
        sample_weight,
        solver_name,
        backend_name,
    ):
        penalty_name = str(
            getattr(getattr(self, "_penalty", None), "name", "")
        ).lower()
        if penalty_name not in _GROUP_LASSO_NAMES:
            return current(
                self,
                X,
                y,
                sample_weight,
                solver_name,
                backend_name,
            )

        # A shallow copy with a private routing name bypasses only the legacy
        # Gaussian BCD branch. Value/proximal semantics and all group metadata
        # remain unchanged. Preserve the selected/explicit generic solver so
        # user intent is not silently rewritten while solving the advertised
        # composite objective.
        original_penalty = self._penalty
        routed_penalty = copy.copy(original_penalty)
        routed_penalty.name = "_group_lasso_generic"
        self._penalty = routed_penalty
        try:
            return current(
                self,
                X,
                y,
                sample_weight,
                solver_name,
                backend_name,
            )
        finally:
            self._penalty = original_penalty

    _fit_loss_backend_with_group_contract._statgpu_group_loss_contract = True
    _fit_loss_backend_with_group_contract._statgpu_original = current
    _PenalizedFitMixin._fit_loss_backend = (
        _fit_loss_backend_with_group_contract
    )


_install_direct_fit_transaction()
_install_direct_contract()
_install_inference_contract()
_install_cv_contract()
_install_exact_group_lasso_solver_contract()
