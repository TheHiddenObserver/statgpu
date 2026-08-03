"""Transactional model contracts for public group penalties.

Penalty constructors can validate index syntax but do not know the eventual
design width. This module installs narrow in-place hooks so all specialized
estimators and historical direct imports share the same behavior:

- direct estimators validate/complete group coverage immediately after penalty
  resolution and before solver/backend work;
- PenalizedGLM_CV validates/completes coverage before alpha-grid generation,
  fold construction, or candidate fitting, then writes canonical groups back;
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


def _validate_resolved_group_penalty(penalty, n_features):
    validator = getattr(penalty, "validate_n_features", None)
    if validator is not None:
        validator(n_features)
    return penalty


def _resolved_penalty_name(estimator):
    return str(
        getattr(getattr(estimator, "_penalty", None), "name", estimator.penalty)
    ).lower().strip()


def _install_direct_contract():
    current = PenalizedGeneralizedLinearModel._resolve_penalty
    if getattr(current, "_statgpu_group_contract", False):
        return

    def _resolve_penalty_with_group_contract(self):
        penalty = current(self)
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
    penalty_name = str(
        getattr(estimator.penalty, "name", estimator.penalty)
    ).lower().strip()
    if penalty_name not in _GROUP_PENALTY_NAMES:
        return
    n_features = _cv_design_width(X)
    if n_features is None:
        return

    penalty = estimator.penalty
    if getattr(penalty, "validate_n_features", None) is None:
        from statgpu.penalties import get_penalty

        kwargs = dict(getattr(estimator, "_penalty_kwargs", None) or {})
        kwargs["alpha"] = 1.0
        penalty = get_penalty(penalty_name, **kwargs)

    _validate_resolved_group_penalty(penalty, n_features)

    if isinstance(estimator.penalty, str):
        kwargs = dict(getattr(estimator, "_penalty_kwargs", None) or {})
        kwargs["groups"] = penalty.groups
        estimator._penalty_kwargs = kwargs
    else:
        estimator.penalty = penalty


def _install_cv_contract():
    current = PenalizedGLM_CV.fit
    if getattr(current, "_statgpu_group_contract", False):
        return

    def _fit_with_group_contract(self, X, y, sample_weight=None):
        if str(self.loss).lower() != "cox_ph":
            _prepare_cv_group_penalty(self, X)
        return current(self, X, y, sample_weight=sample_weight)

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


_install_direct_contract()
_install_inference_contract()
_install_cv_contract()
_install_exact_group_lasso_solver_contract()
