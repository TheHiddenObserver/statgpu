"""Truthful solver routing and scoring for penalized Quantile models.

Historically the generic auto-dispatch labeled every Quantile fit as FISTA,
while the fit path internally substituted different algorithms for selected
penalties. That made ``_selected_solver`` and explicit solver requests disagree
with the algorithm that actually ran.

A fresh review of the same consumer graph also found that several accelerated
CV scoring paths constructed their validation registry calls without carrying
the requested Quantile level. Training children received ``loss_kwargs`` while
those registry calls fell back to ``tau=0.5``. The contract installed here
therefore keeps the requested Quantile level in a call-local CV context shared
by the residual/validation registries.

Keep this compatibility installer narrow and composable with the existing
penalized-model contract installers:

* auto + Quantile + L2/none resolves to ordinary Quantile IRLS;
* sparse Quantile auto routes keep the existing FISTA-family policy;
* auto + Quantile + SCAD/MCP resolves to the dedicated Proximal IRLS-CD route;
* incompatible explicit Quantile solver requests fail before backend numerical
  dispatch instead of being silently substituted by another algorithm;
* every Quantile CV candidate/validation route uses the caller's requested
  Quantile level, including fold-batched sparse and weighted registry paths.

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
from . import _penalized_cv as _cv_mod
from ._base import PenalizedGeneralizedLinearModel

PenalizedGLM_CV = _cv_mod.PenalizedGLM_CV


_POLICY_MARKER = "_statgpu_quantile_solver_policy_contract"
_VALIDATE_MARKER = "_statgpu_quantile_solver_validate_contract"
_CV_CONTEXT_MARKER = "_statgpu_quantile_solver_cv_context_contract"
_CV_SCORE_CONTEXT_MARKER = "_statgpu_quantile_cv_score_context_contract"
_CV_REGISTRY_MARKER = "_statgpu_quantile_cv_registry_contract"
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})
_NONCONVEX_QUANTILE_PENALTIES = frozenset({"scad", "mcp"})
_DEDICATED_NONCONVEX_SOLVER = "proximal_irls_cd"
_INTERNAL_CV_RESOLVED_SOLVER = ContextVar(
    "statgpu_quantile_internal_cv_resolved_solver", default=False
)
_QUANTILE_CV_LEVEL = ContextVar(
    "statgpu_quantile_cv_level", default=None
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


def _inject_requested_quantile(fn):
    if getattr(fn, _CV_REGISTRY_MARKER, False):
        return fn

    @wraps(fn)
    def _with_requested_quantile(eta, y, **kwargs):
        if "quantile" not in kwargs:
            quantile = _QUANTILE_CV_LEVEL.get()
            if quantile is not None:
                kwargs = {**kwargs, "quantile": float(quantile)}
        return fn(eta, y, **kwargs)

    setattr(_with_requested_quantile, _CV_REGISTRY_MARKER, True)
    _with_requested_quantile._statgpu_original = fn
    return _with_requested_quantile


def _install_cv_registry_contract() -> None:
    residual = _cv_mod._LOSS_RESIDUAL_FNS.get("quantile")
    val_loss = _cv_mod._LOSS_VALLOSS_FNS.get("quantile")
    if residual is None or val_loss is None:
        raise RuntimeError("Quantile CV registry entries are unavailable")

    residual_wrapped = _inject_requested_quantile(residual)
    val_wrapped = _inject_requested_quantile(val_loss)
    _cv_mod._LOSS_RESIDUAL_FNS["quantile"] = residual_wrapped
    _cv_mod._LOSS_VALLOSS_FNS["quantile"] = val_wrapped

    eval_entry = _cv_mod._LOSS_EVAL_DISPATCH.get("quantile")
    if eval_entry is not None:
        _cv_mod._LOSS_EVAL_DISPATCH["quantile"] = (
            val_wrapped,
            eval_entry[1],
        )


def _install_cv_score_context() -> None:
    current = PenalizedGLM_CV._compute_cv_scores
    if getattr(current, _CV_SCORE_CONTEXT_MARKER, False):
        return

    @wraps(current)
    def _compute_cv_scores_with_quantile_level(self, *args, **kwargs):
        if _loss_name(getattr(self, "loss", "")) != "quantile":
            return current(self, *args, **kwargs)

        loss_kwargs = getattr(self, "_loss_kwargs", None) or {}
        quantile = float(loss_kwargs.get("quantile", 0.5))
        token = _QUANTILE_CV_LEVEL.set(quantile)
        try:
            return current(self, *args, **kwargs)
        finally:
            _QUANTILE_CV_LEVEL.reset(token)

    setattr(
        _compute_cv_scores_with_quantile_level,
        _CV_SCORE_CONTEXT_MARKER,
        True,
    )
    _compute_cv_scores_with_quantile_level._statgpu_original = current
    PenalizedGLM_CV._compute_cv_scores = _compute_cv_scores_with_quantile_level


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
    """Install Quantile solver/provenance/scoring reconciliation idempotently."""
    _install_policy_contract()
    _install_cv_internal_context()
    _install_cv_registry_contract()
    _install_cv_score_context()
    _install_explicit_route_guard()


install_quantile_solver_contract()
