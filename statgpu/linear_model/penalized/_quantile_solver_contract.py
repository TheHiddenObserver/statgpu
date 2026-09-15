"""Truthful solver routing for penalized Quantile models.

Historically the generic auto-dispatch labeled every Quantile fit as FISTA,
while the fit path internally substituted different algorithms for selected
penalties. That made ``_selected_solver`` and explicit solver requests disagree
with the algorithm that actually ran.

Keep this compatibility installer narrow and composable with the existing
penalized-model contract installers:

* auto + Quantile + L2/none resolves to ordinary Quantile IRLS;
* sparse Quantile auto routes keep the existing FISTA-family policy;
* auto + Quantile + SCAD/MCP resolves to the dedicated Proximal IRLS-CD route;
* incompatible explicit Quantile solver requests fail before backend numerical
  dispatch instead of being silently substituted by another algorithm.

``proximal_irls_cd`` is an internal resolved-provenance label, not a new public
``solver=`` keyword. ``PenalizedGLM_CV`` is allowed to pass that resolved label
to its private SCAD/MCP child estimators only inside a call-local context; a
user who constructs an estimator with that spelling still fails closed for
all Quantile penalties.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

from . import _fit_mixin as _fit_mixin
from ._base import PenalizedGeneralizedLinearModel
from ._penalized_cv import PenalizedGLM_CV


_POLICY_MARKER = "_statgpu_quantile_solver_policy_contract"
_VALIDATE_MARKER = "_statgpu_quantile_solver_validate_contract"
_CV_CONTEXT_MARKER = "_statgpu_quantile_solver_cv_context_contract"
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})
_NONCONVEX_QUANTILE_PENALTIES = frozenset({"scad", "mcp"})
_DEDICATED_NONCONVEX_SOLVER = "proximal_irls_cd"
_INTERNAL_CV_RESOLVED_SOLVER = ContextVar(
    "statgpu_quantile_internal_cv_resolved_solver", default=False
)


def _loss_name(value) -> str:
    return str(getattr(value, "name", value) or "").lower().strip()


def _penalty_name(value) -> str:
    return str(getattr(value, "name", value) or "").lower().strip()


def _install_policy_contract() -> None:
    current = _fit_mixin._preferred_penalized_glm_solver
    if getattr(current, _POLICY_MARKER, False):
        return

    @wraps(current)
    def _preferred_with_truthful_quantile_route(
        loss_name,
        penalty_name,
        backend_name=None,
        l1_ratio=0.5,
        cv_mode=False,
        problem_size=None,
    ):
        resolved_loss = _loss_name(loss_name)
        resolved_penalty = _penalty_name(penalty_name)
        if resolved_loss == "quantile":
            if resolved_penalty in _SMOOTH_PENALTIES:
                return "irls"
            if resolved_penalty in _NONCONVEX_QUANTILE_PENALTIES:
                return _DEDICATED_NONCONVEX_SOLVER
        return current(
            loss_name,
            penalty_name,
            backend_name=backend_name,
            l1_ratio=l1_ratio,
            cv_mode=cv_mode,
            problem_size=problem_size,
        )

    setattr(_preferred_with_truthful_quantile_route, _POLICY_MARKER, True)
    _preferred_with_truthful_quantile_route._statgpu_original = current
    _fit_mixin._preferred_penalized_glm_solver = _preferred_with_truthful_quantile_route


def _install_cv_internal_context() -> None:
    if getattr(PenalizedGLM_CV, _CV_CONTEXT_MARKER, False):
        return

    current_fold = PenalizedGLM_CV._cv_fold_general
    current_refit = PenalizedGLM_CV._refit_best

    @wraps(current_fold)
    def _cv_fold_with_internal_resolved_solver(*args, **kwargs):
        token = _INTERNAL_CV_RESOLVED_SOLVER.set(True)
        try:
            return current_fold(*args, **kwargs)
        finally:
            _INTERNAL_CV_RESOLVED_SOLVER.reset(token)

    @wraps(current_refit)
    def _refit_with_internal_resolved_solver(*args, **kwargs):
        token = _INTERNAL_CV_RESOLVED_SOLVER.set(True)
        try:
            return current_refit(*args, **kwargs)
        finally:
            _INTERNAL_CV_RESOLVED_SOLVER.reset(token)

    PenalizedGLM_CV._cv_fold_general = _cv_fold_with_internal_resolved_solver
    PenalizedGLM_CV._refit_best = _refit_with_internal_resolved_solver
    setattr(PenalizedGLM_CV, _CV_CONTEXT_MARKER, True)


def _install_explicit_route_guard() -> None:
    current = PenalizedGeneralizedLinearModel._validate_solver_penalty
    if getattr(current, _VALIDATE_MARKER, False):
        return

    @wraps(current)
    def _validate_with_quantile_route_guard(self):
        current(self)
        solver_name = str(getattr(self, "_solver", "") or "").lower()
        loss_name = str(getattr(self, "loss", "") or "").lower()
        penalty_name = _penalty_name(getattr(self, "_penalty", self.penalty))

        if loss_name != "quantile":
            return

        if solver_name == _DEDICATED_NONCONVEX_SOLVER:
            if (
                penalty_name in _NONCONVEX_QUANTILE_PENALTIES
                and _INTERNAL_CV_RESOLVED_SOLVER.get()
            ):
                return
            raise ValueError(
                f"solver='{solver_name}' is an internal resolved Quantile "
                "solver label, not a public explicit solver; use solver='auto'."
            )

        if (
            solver_name in ("fista", "fista_bb")
            and penalty_name in _SMOOTH_PENALTIES
        ):
            raise ValueError(
                f"solver='{solver_name}' is not a maintained smooth Quantile "
                "route for L2/no-penalty objectives; use solver='irls' or "
                "solver='auto'."
            )

        if penalty_name in _NONCONVEX_QUANTILE_PENALTIES:
            if solver_name == "auto":
                return
            raise ValueError(
                f"solver='{solver_name}' is not a public explicit Quantile "
                f"{penalty_name.upper()} route; use solver='auto' so the "
                "dedicated Proximal IRLS-CD algorithm is selected."
            )

    setattr(_validate_with_quantile_route_guard, _VALIDATE_MARKER, True)
    _validate_with_quantile_route_guard._statgpu_original = current
    PenalizedGeneralizedLinearModel._validate_solver_penalty = (
        _validate_with_quantile_route_guard
    )


def install_quantile_solver_contract() -> None:
    """Install the Quantile solver/provenance reconciliation idempotently."""
    _install_policy_contract()
    _install_cv_internal_context()
    _install_explicit_route_guard()


install_quantile_solver_contract()
