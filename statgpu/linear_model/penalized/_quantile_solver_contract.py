"""Truthful solver routing and CV scoring for penalized Quantile models.

This compatibility contract reconciles two existing-capability mismatches:

* reported Quantile solver identity must match the algorithm that executes;
* Quantile CV scoring must use the caller's requested quantile level rather
  than silently falling back to the median objective.

The repair is deliberately narrow. It does not register a new Quantile
fold-batched/FISTA residual or SCAD/MCP fast-path implementation. Private
accelerated Quantile routes whose loss-parameter contract is incomplete return
``None`` so the existing CV dispatcher uses the maintained per-fold estimator
path instead.
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
_CV_FIT_VALIDATE_MARKER = "_statgpu_quantile_cv_fit_validate_contract"
_DIRECT_FIT_SOLVER_SYNC_MARKER = "_statgpu_quantile_direct_fit_solver_sync_contract"
_CV_FIT_SOLVER_SYNC_MARKER = "_statgpu_quantile_cv_fit_solver_sync_contract"
_CV_PUBLIC_SOLVER_MARKER = "_statgpu_quantile_cv_public_solver_contract"
_CV_CONTEXT_MARKER = "_statgpu_quantile_solver_cv_context_contract"
_CV_SCORE_CONTEXT_MARKER = "_statgpu_quantile_cv_score_context_contract"
_CV_EVAL_MARKER = "_statgpu_quantile_cv_eval_contract"
_CV_SCAD_MARKER = "_statgpu_quantile_cv_scad_contract"
_CV_FOLDBATCH_MARKER = "_statgpu_quantile_cv_foldbatch_contract"
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})
_NONCONVEX_QUANTILE_PENALTIES = frozenset({"scad", "mcp"})
_DEDICATED_NONCONVEX_SOLVER = "proximal_irls_cd"
_INTERNAL_CV_RESOLVED_SOLVER = ContextVar(
    "statgpu_quantile_internal_cv_resolved_solver", default=False
)
_QUANTILE_CV_LEVEL = ContextVar("statgpu_quantile_cv_level", default=None)


def _loss_name(value) -> str:
    return str(getattr(value, "name", value) or "").lower().strip()


def _penalty_name(value) -> str:
    return str(getattr(value, "name", value) or "").lower().strip()


def _requested_quantile(owner) -> float:
    loss_kwargs = getattr(owner, "_loss_kwargs", None) or {}
    if "quantile" in loss_kwargs:
        return float(loss_kwargs["quantile"])
    resolved = _fit_mixin._resolve_loss_name(
        getattr(owner, "loss", "quantile"), loss_kwargs=loss_kwargs
    )
    return float(getattr(resolved, "_tau", getattr(resolved, "quantile", 0.5)))


def _validate_quantile_solver_request(
    *,
    loss_name,
    penalty_name,
    solver_name,
    allow_internal_nonconvex=False,
) -> None:
    """Validate the public Quantile solver contract before numerical dispatch."""
    resolved_loss = _loss_name(loss_name)
    resolved_penalty = _penalty_name(penalty_name)
    resolved_solver = str(solver_name or "").lower().strip()
    if resolved_loss != "quantile":
        return

    if resolved_solver == _DEDICATED_NONCONVEX_SOLVER:
        if (
            allow_internal_nonconvex
            and resolved_penalty in _NONCONVEX_QUANTILE_PENALTIES
        ):
            return
        raise ValueError(
            f"solver='{resolved_solver}' is an internal resolved Quantile "
            "solver label, not a public explicit solver; use solver='auto'."
        )

    if resolved_penalty in _NONCONVEX_QUANTILE_PENALTIES:
        if resolved_solver == "auto":
            return
        raise ValueError(
            f"solver='{resolved_solver}' is not a public explicit Quantile "
            f"{resolved_penalty.upper()} route; use solver='auto' so the "
            "dedicated Proximal IRLS-CD algorithm is selected."
        )

    if resolved_solver == "irls" and resolved_penalty not in _SMOOTH_PENALTIES:
        raise ValueError(
            "solver='irls' only supports L2 or no-penalty Quantile objectives."
        )

    if resolved_solver in ("newton", "lbfgs", "exact"):
        raise ValueError(
            f"solver='{resolved_solver}' requires Hessian-compatible smooth "
            "structure, but quantile loss has no Hessian. Use solver='auto', "
            "'irls', or a supported sparse solver as appropriate."
        )

    if (
        resolved_solver in ("fista", "fista_bb")
        and resolved_penalty in _SMOOTH_PENALTIES
    ):
        raise ValueError(
            f"solver='{resolved_solver}' is not supported for L2/no-penalty "
            "Quantile objectives; use solver='irls' or solver='auto'."
        )


def _sync_public_solver(owner) -> None:
    """Make direct public solver replacement authoritative for the next fit."""
    solver = getattr(owner, "solver", getattr(owner, "_solver", "auto"))
    owner._solver = solver.lower() if isinstance(solver, str) else solver


def _install_public_solver_refit_sync() -> None:
    """Synchronize the public Quantile solver before validation/dispatch."""

    current_direct_fit = PenalizedGeneralizedLinearModel.fit
    if not getattr(current_direct_fit, _DIRECT_FIT_SOLVER_SYNC_MARKER, False):

        @wraps(current_direct_fit)
        def _fit_with_current_public_solver(self, *args, **kwargs):
            if _loss_name(getattr(self, "loss", "")) == "quantile":
                _sync_public_solver(self)
            return current_direct_fit(self, *args, **kwargs)

        setattr(
            _fit_with_current_public_solver,
            _DIRECT_FIT_SOLVER_SYNC_MARKER,
            True,
        )
        _fit_with_current_public_solver._statgpu_original = current_direct_fit
        PenalizedGeneralizedLinearModel.fit = _fit_with_current_public_solver

    current_cv_fit = PenalizedGLM_CV.fit
    if not getattr(current_cv_fit, _CV_FIT_SOLVER_SYNC_MARKER, False):

        @wraps(current_cv_fit)
        def _cv_fit_with_current_public_solver(self, *args, **kwargs):
            if _loss_name(getattr(self, "loss", "")) == "quantile":
                _sync_public_solver(self)
            return current_cv_fit(self, *args, **kwargs)

        setattr(
            _cv_fit_with_current_public_solver,
            _CV_FIT_SOLVER_SYNC_MARKER,
            True,
        )
        _cv_fit_with_current_public_solver._statgpu_original = current_cv_fit
        PenalizedGLM_CV.fit = _cv_fit_with_current_public_solver


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


def _install_cv_fit_route_guard() -> None:
    """Reject invalid explicit Quantile CV solvers before alpha-grid work."""
    current = PenalizedGLM_CV._fit_standard
    if getattr(current, _CV_FIT_VALIDATE_MARKER, False):
        return

    @wraps(current)
    def _fit_standard_with_quantile_solver_guard(self, *args, **kwargs):
        _validate_quantile_solver_request(
            loss_name=getattr(self, "loss", ""),
            penalty_name=getattr(self, "penalty", ""),
            solver_name=getattr(self, "_solver", ""),
            allow_internal_nonconvex=False,
        )
        return current(self, *args, **kwargs)

    setattr(
        _fit_standard_with_quantile_solver_guard,
        _CV_FIT_VALIDATE_MARKER,
        True,
    )
    _fit_standard_with_quantile_solver_guard._statgpu_original = current
    PenalizedGLM_CV._fit_standard = _fit_standard_with_quantile_solver_guard


def _install_cv_public_solver_guard() -> None:
    """Keep the dedicated Quantile provenance label internal to CV auto routing."""
    current = PenalizedGLM_CV._solver_for_cv
    if getattr(current, _CV_PUBLIC_SOLVER_MARKER, False):
        return

    @wraps(current)
    def _solver_for_cv_with_public_boundary(self, *args, **kwargs):
        if (
            _loss_name(getattr(self, "loss", "")) == "quantile"
            and str(getattr(self, "_solver", "") or "").lower().strip()
            == _DEDICATED_NONCONVEX_SOLVER
        ):
            raise ValueError(
                f"solver='{_DEDICATED_NONCONVEX_SOLVER}' is an internal resolved "
                "Quantile solver label, not a public explicit solver; use "
                "solver='auto'."
            )
        return current(self, *args, **kwargs)

    setattr(
        _solver_for_cv_with_public_boundary,
        _CV_PUBLIC_SOLVER_MARKER,
        True,
    )
    _solver_for_cv_with_public_boundary._statgpu_original = current
    PenalizedGLM_CV._solver_for_cv = _solver_for_cv_with_public_boundary


def _install_cv_internal_context() -> None:
    if getattr(PenalizedGLM_CV, _CV_CONTEXT_MARKER, False):
        return

    current_fold = PenalizedGLM_CV._cv_fold_general
    current_refit = PenalizedGLM_CV._refit_best

    @wraps(current_fold)
    def _cv_fold_with_internal_resolved_solver(self, *args, **kwargs):
        if _loss_name(getattr(self, "loss", "")) != "quantile":
            return current_fold(self, *args, **kwargs)
        token = _INTERNAL_CV_RESOLVED_SOLVER.set(True)
        try:
            return current_fold(self, *args, **kwargs)
        finally:
            _INTERNAL_CV_RESOLVED_SOLVER.reset(token)

    @wraps(current_refit)
    def _refit_with_internal_resolved_solver(self, *args, **kwargs):
        if _loss_name(getattr(self, "loss", "")) != "quantile":
            return current_refit(self, *args, **kwargs)
        token = _INTERNAL_CV_RESOLVED_SOLVER.set(True)
        try:
            return current_refit(self, *args, **kwargs)
        finally:
            _INTERNAL_CV_RESOLVED_SOLVER.reset(token)

    PenalizedGLM_CV._cv_fold_general = _cv_fold_with_internal_resolved_solver
    PenalizedGLM_CV._refit_best = _refit_with_internal_resolved_solver
    setattr(PenalizedGLM_CV, _CV_CONTEXT_MARKER, True)


def _install_cv_eval_contract() -> None:
    entry = _cv_mod._LOSS_EVAL_DISPATCH.get("quantile")
    if entry is None:
        raise RuntimeError("Quantile CV evaluation entry is unavailable")

    current_eval_fn, uses_design = entry
    if not getattr(current_eval_fn, _CV_EVAL_MARKER, False):
        @wraps(current_eval_fn)
        def _eval_with_requested_quantile(eta, y, **kwargs):
            if "quantile" not in kwargs:
                quantile = _QUANTILE_CV_LEVEL.get()
                if quantile is not None:
                    kwargs = {**kwargs, "quantile": float(quantile)}
            return current_eval_fn(eta, y, **kwargs)

        setattr(_eval_with_requested_quantile, _CV_EVAL_MARKER, True)
        _eval_with_requested_quantile._statgpu_original = current_eval_fn
        _cv_mod._LOSS_EVAL_DISPATCH["quantile"] = (
            _eval_with_requested_quantile,
            uses_design,
        )

    current_numpy_eval = _cv_mod._evaluate_loss_numpy
    if getattr(current_numpy_eval, _CV_EVAL_MARKER, False):
        return

    @wraps(current_numpy_eval)
    def _evaluate_loss_numpy_with_requested_quantile(
        loss_name,
        loss_fn,
        X_val_np,
        y_val_np,
        coef_np,
        intercept,
        fit_intercept,
        sample_weight=None,
    ):
        if _loss_name(loss_name) == "quantile":
            quantile = _QUANTILE_CV_LEVEL.get()
            if quantile is not None:
                loss_fn = _fit_mixin._resolve_loss_name(
                    "quantile", loss_kwargs={"quantile": float(quantile)}
                )
        return current_numpy_eval(
            loss_name,
            loss_fn,
            X_val_np,
            y_val_np,
            coef_np,
            intercept,
            fit_intercept,
            sample_weight=sample_weight,
        )

    setattr(_evaluate_loss_numpy_with_requested_quantile, _CV_EVAL_MARKER, True)
    _evaluate_loss_numpy_with_requested_quantile._statgpu_original = current_numpy_eval
    _cv_mod._evaluate_loss_numpy = _evaluate_loss_numpy_with_requested_quantile


def _install_scad_quantile_guard() -> None:
    current = _cv_mod._scad_mcp_cv_path
    if getattr(current, _CV_SCAD_MARKER, False):
        return

    @wraps(current)
    def _scad_mcp_without_unmaintained_quantile_fast_path(*args, **kwargs):
        loss_name = kwargs.get("loss_name", args[0] if args else "")
        if _loss_name(loss_name) == "quantile":
            return None
        return current(*args, **kwargs)

    setattr(
        _scad_mcp_without_unmaintained_quantile_fast_path,
        _CV_SCAD_MARKER,
        True,
    )
    _scad_mcp_without_unmaintained_quantile_fast_path._statgpu_original = current
    _cv_mod._scad_mcp_cv_path = _scad_mcp_without_unmaintained_quantile_fast_path


def _install_incomplete_fold_batch_guard() -> None:
    current = _cv_mod._glm_sparse_cv_folds
    if getattr(current, _CV_FOLDBATCH_MARKER, False):
        return

    @wraps(current)
    def _fold_batch_without_unmaintained_quantile_route(*args, **kwargs):
        loss_name = kwargs.get("loss_name", args[8] if len(args) > 8 else "")
        if _loss_name(loss_name) == "quantile":
            return None
        return current(*args, **kwargs)

    setattr(
        _fold_batch_without_unmaintained_quantile_route,
        _CV_FOLDBATCH_MARKER,
        True,
    )
    _fold_batch_without_unmaintained_quantile_route._statgpu_original = current
    _cv_mod._glm_sparse_cv_folds = _fold_batch_without_unmaintained_quantile_route


def _install_cv_score_context() -> None:
    current = PenalizedGLM_CV._compute_cv_scores
    if getattr(current, _CV_SCORE_CONTEXT_MARKER, False):
        return

    @wraps(current)
    def _compute_cv_scores_with_quantile_level(self, *args, **kwargs):
        if _loss_name(getattr(self, "loss", "")) != "quantile":
            return current(self, *args, **kwargs)

        token = _QUANTILE_CV_LEVEL.set(_requested_quantile(self))
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
        _validate_quantile_solver_request(
            loss_name=getattr(self, "loss", ""),
            penalty_name=getattr(self, "_penalty", self.penalty),
            solver_name=getattr(self, "_solver", ""),
            allow_internal_nonconvex=_INTERNAL_CV_RESOLVED_SOLVER.get(),
        )

    setattr(_validate_with_quantile_route_guard, _VALIDATE_MARKER, True)
    _validate_with_quantile_route_guard._statgpu_original = current
    PenalizedGeneralizedLinearModel._validate_solver_penalty = (
        _validate_with_quantile_route_guard
    )


def install_quantile_solver_contract() -> None:
    """Install Quantile solver/provenance/scoring reconciliation idempotently."""
    _install_public_solver_refit_sync()
    _install_policy_contract()
    _install_cv_fit_route_guard()
    _install_cv_public_solver_guard()
    _install_cv_internal_context()
    _install_cv_eval_contract()
    _install_scad_quantile_guard()
    _install_incomplete_fold_batch_guard()
    _install_cv_score_context()
    _install_explicit_route_guard()


install_quantile_solver_contract()
