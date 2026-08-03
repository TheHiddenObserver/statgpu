"""Transactional design-width validation for public group penalties.

Penalty constructors can validate index syntax but do not know the eventual
design width. This module installs two narrow public-boundary hooks:

- direct estimators validate/complete group coverage immediately after penalty
  resolution and before solver/backend work;
- PenalizedGLM_CV validates/completes coverage before alpha-grid generation,
  fold construction, or candidate fitting, then writes the canonical groups
  back so every fold and the final refit use the same penalty.
"""

from __future__ import annotations

from ._base import PenalizedGeneralizedLinearModel
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


def _validate_resolved_group_penalty(penalty, n_features):
    validator = getattr(penalty, "validate_n_features", None)
    if validator is not None:
        validator(n_features)
    return penalty


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


def _prepare_cv_group_penalty(estimator, X):
    penalty_name = str(
        getattr(estimator.penalty, "name", estimator.penalty)
    ).lower().strip()
    if penalty_name not in _GROUP_PENALTY_NAMES:
        return
    shape = getattr(X, "shape", None)
    ndim = getattr(X, "ndim", None)
    if shape is None or ndim != 2:
        return
    n_features = int(shape[1])

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


_install_direct_contract()
_install_cv_contract()
