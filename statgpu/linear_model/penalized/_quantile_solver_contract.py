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

import copy
from contextvars import ContextVar
import sys
from functools import wraps
from numbers import Integral, Real

import numpy as np

from statgpu._config import Device
from statgpu.penalties._categories import GROUP as _GROUP_PENALTY_NAMES

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
_RESOLVE_PENALTY_MARKER = "_statgpu_quantile_scalar_cv_penalty_object_contract"
_SCALAR_CV_ALPHA_MARKER = "_statgpu_quantile_scalar_cv_alpha_from_estimator"
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})
_NONCONVEX_QUANTILE_PENALTIES = frozenset({"scad", "mcp"})
_GROUP_NONCONVEX_QUANTILE_PENALTIES = frozenset(
    {"group_scad", "gscad", "group_mcp", "gmcp"}
)
_QUANTILE_LLA_PENALTIES = (
    _NONCONVEX_QUANTILE_PENALTIES | _GROUP_NONCONVEX_QUANTILE_PENALTIES
)
_DEDICATED_NONCONVEX_SOLVER = "proximal_irls_cd"
_INTERNAL_CV_RESOLVED_SOLVER = ContextVar(
    "statgpu_quantile_internal_cv_resolved_solver", default=False
)
_QUANTILE_CV_LEVEL = ContextVar("statgpu_quantile_cv_level", default=None)


def _wrapper_chain_has_marker(function, marker: str) -> bool:
    """Return whether a contract marker exists anywhere in a wrapper chain."""
    current = function
    seen = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, marker, False)):
            return True
        wrapped = getattr(current, "_statgpu_original", None)
        if not callable(wrapped):
            wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped):
            break
        current = wrapped
    return False


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


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _clone_scalar_penalty(penalty, *, alpha=None):
    """Clone a scalar penalty without dropping constructor or learned state."""
    cloned = copy.deepcopy(penalty)
    if alpha is not None:
        cloned.alpha = float(alpha)
    # The marker belongs only to the CV owner's routed source object. Child
    # penalties are ordinary resolved snapshots and must not request another
    # candidate-alpha rewrite on later introspection/refit.
    if hasattr(cloned, _SCALAR_CV_ALPHA_MARKER):
        delattr(cloned, _SCALAR_CV_ALPHA_MARKER)
    return cloned


def _sync_public_quantile_fit_controls(owner, *, cv: bool) -> None:
    """Validate current public Quantile refit controls and sync runtime mirrors."""
    solver = getattr(owner, "solver", getattr(owner, "_solver", "auto"))
    owner._solver = solver.lower() if isinstance(solver, str) else solver

    device = getattr(owner, "device", getattr(owner, "_device", Device.AUTO))
    try:
        owner._device = device if isinstance(device, Device) else Device(device)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "device must be one of 'auto', 'cpu', 'cuda', or 'torch'"
        ) from exc

    owner._max_iter = _positive_integer(owner.max_iter, "max_iter")
    owner._tol = _finite_positive(owner.tol, "tol")
    lipschitz_L = getattr(owner, "lipschitz_L", None)
    if lipschitz_L is not None:
        if (
            isinstance(lipschitz_L, (bool, np.bool_))
            or not isinstance(lipschitz_L, Real)
            or not np.isfinite(float(lipschitz_L))
            or float(lipschitz_L) <= 0.0
        ):
            raise ValueError(
                "lipschitz_L must be None or a finite positive number"
            )
        owner.lipschitz_L = float(lipschitz_L)

    if cv:
        cv_value = _positive_integer(owner.cv, "cv")
        if cv_value < 2:
            raise ValueError("cv must be an integer greater than or equal to 2")
        owner._cv = cv_value

        owner._n_alphas = _positive_integer(owner.n_alphas, "n_alphas")

        penalty_name = _penalty_name(getattr(owner, "penalty", ""))
        if (
            isinstance(getattr(owner, "penalty", None), str)
            and penalty_name in ("elasticnet", "en")
        ):
            l1_ratio = getattr(owner, "l1_ratio", 0.5)
            if (
                isinstance(l1_ratio, (bool, np.bool_))
                or not isinstance(l1_ratio, Real)
            ):
                raise ValueError(
                    "l1_ratio must be a finite real number in [0, 1]"
                )
            l1_ratio = float(l1_ratio)
            if not np.isfinite(l1_ratio) or not (0.0 <= l1_ratio <= 1.0):
                raise ValueError(
                    "l1_ratio must be a finite real number in [0, 1]"
                )
            owner.l1_ratio = l1_ratio

        strategy = getattr(owner, "cv_strategy", "strict")
        if not isinstance(strategy, str):
            raise ValueError("cv_strategy must be either 'strict' or 'two_stage'")
        strategy = strategy.lower()
        if strategy not in ("strict", "two_stage"):
            raise ValueError("cv_strategy must be either 'strict' or 'two_stage'")
        owner._cv_strategy = strategy

        acknowledge = getattr(owner, "acknowledge_approx", False)
        if not isinstance(acknowledge, (bool, np.bool_)):
            raise ValueError("acknowledge_approx must be boolean")
        owner._acknowledge_approx = bool(acknowledge)

        owner._refine_top_k = _positive_integer(
            owner.refine_top_k, "refine_top_k"
        )

        owner._loss_kwargs = dict(
            getattr(owner, "loss_kwargs", None) or {}
        )
        owner._penalty_kwargs = dict(
            getattr(owner, "penalty_kwargs", None) or {}
        )
        if hasattr(owner, "alpha_grid"):
            owner._alpha_grid_input = owner.alpha_grid

    if not cv:
        fit_intercept = getattr(owner, "fit_intercept", True)
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        owner._fit_intercept = bool(fit_intercept)

    penalty_name = _penalty_name(getattr(owner, "penalty", ""))
    solver_name = str(getattr(owner, "_solver", "") or "").lower().strip()
    scalar_internal_lla = (
        penalty_name in _NONCONVEX_QUANTILE_PENALTIES
        and solver_name == _DEDICATED_NONCONVEX_SOLVER
        and _INTERNAL_CV_RESOLVED_SOLVER.get()
    )
    uses_quantile_lla = (
        (
            penalty_name in _NONCONVEX_QUANTILE_PENALTIES
            and solver_name == "auto"
        )
        or scalar_internal_lla
        or (
            penalty_name in _GROUP_NONCONVEX_QUANTILE_PENALTIES
            and solver_name == "auto"
        )
    )
    if not cv and uses_quantile_lla:
        lla = getattr(owner, "lla", getattr(owner, "_lla_enabled", True))
        if not isinstance(lla, (bool, np.bool_)):
            raise ValueError("lla must be boolean for Quantile non-convex penalties")
        owner._lla_enabled = bool(lla)
        if not owner._lla_enabled:
            raise ValueError(
                "Quantile SCAD/MCP and Group SCAD/MCP require lla=True"
            )

        owner._max_lla_iters = _positive_integer(
            owner.max_lla_iters, "max_lla_iters"
        )
        min_lla_steps = int(_fit_mixin._N_CONT_STEPS_NONSMOOTH)
        if owner._max_lla_iters < min_lla_steps:
            raise ValueError(
                f"max_lla_iters must be at least {min_lla_steps} for the Quantile "
                "continuation path so every alpha step can run once"
            )
        owner._lla_tol = _finite_positive(owner.lla_tol, "lla_tol")


def _install_scalar_cv_penalty_object_contract() -> None:
    current = PenalizedGeneralizedLinearModel._resolve_penalty
    if _wrapper_chain_has_marker(current, _RESOLVE_PENALTY_MARKER):
        return

    @wraps(current)
    def _resolve_penalty_with_scalar_quantile_cv_alpha(self):
        penalty = current(self)
        if (
            _loss_name(getattr(self, "loss", "")) != "quantile"
            or _penalty_name(penalty) in _GROUP_PENALTY_NAMES
        ):
            return penalty

        marked_cv_source = bool(
            getattr(penalty, _SCALAR_CV_ALPHA_MARKER, False)
        )
        direct_public_object = (
            penalty is getattr(self, "penalty", None)
            and not isinstance(getattr(self, "penalty", None), str)
            and _penalty_name(penalty) in ("adaptive_l1", "adaptive_lasso")
        )
        if not marked_cv_source and not direct_public_object:
            return penalty

        resolved = _clone_scalar_penalty(
            penalty,
            alpha=float(self.alpha) if marked_cv_source else None,
        )
        if marked_cv_source:
            # Internal CV children own this clone, so their public penalty
            # reports the same alpha as the numerical penalty that actually fits.
            self.penalty = resolved
        return resolved

    setattr(
        _resolve_penalty_with_scalar_quantile_cv_alpha,
        _RESOLVE_PENALTY_MARKER,
        True,
    )
    _resolve_penalty_with_scalar_quantile_cv_alpha._statgpu_original = current
    PenalizedGeneralizedLinearModel._resolve_penalty = (
        _resolve_penalty_with_scalar_quantile_cv_alpha
    )


def _invalidate_rejected_direct_quantile_refit(owner) -> None:
    """Clear all fit-derived Quantile state after pre-dispatch rejection."""
    from ._no_inference_cleanup_contract import _invalidate_failed_no_inference_fit

    _invalidate_failed_no_inference_fit(owner)
    for name in (
        "_X_design",
        "_y",
        "_resid",
        "_raw_resid",
        "_scale",
        "_nobs",
        "_df_resid",
        "_sample_weight_fit",
        "_loss",
        "_penalty",
        "_init_coef",
        "_init_intercept",
    ):
        if hasattr(owner, name):
            setattr(owner, name, None)


def _install_public_solver_refit_sync() -> None:
    """Synchronize the public Quantile solver before validation/dispatch."""

    current_direct_fit = PenalizedGeneralizedLinearModel.fit
    if not _wrapper_chain_has_marker(
        current_direct_fit, _DIRECT_FIT_SOLVER_SYNC_MARKER
    ):

        @wraps(current_direct_fit)
        def _fit_with_current_public_solver(self, *args, **kwargs):
            if _loss_name(getattr(self, "loss", "")) == "quantile":
                try:
                    _sync_public_quantile_fit_controls(self, cv=False)
                except Exception:
                    _invalidate_rejected_direct_quantile_refit(self)
                    raise
            return current_direct_fit(self, *args, **kwargs)

        setattr(
            _fit_with_current_public_solver,
            _DIRECT_FIT_SOLVER_SYNC_MARKER,
            True,
        )
        _fit_with_current_public_solver._statgpu_original = current_direct_fit
        PenalizedGeneralizedLinearModel.fit = _fit_with_current_public_solver

    current_cv_fit = PenalizedGLM_CV.fit
    if not _wrapper_chain_has_marker(
        current_cv_fit, _CV_FIT_SOLVER_SYNC_MARKER
    ):

        @wraps(current_cv_fit)
        def _cv_fit_with_current_public_solver(self, *args, **kwargs):
            if _loss_name(getattr(self, "loss", "")) != "quantile":
                return current_cv_fit(self, *args, **kwargs)

            try:
                _sync_public_quantile_fit_controls(self, cv=True)
            except Exception:
                self._reset_cv_fit_state()
                raise

            original_penalty = self.penalty
            penalty_name = _penalty_name(original_penalty)
            use_scalar_object_clone = (
                not isinstance(original_penalty, str)
                and penalty_name not in _GROUP_PENALTY_NAMES
                and hasattr(original_penalty, "alpha")
            )
            if use_scalar_object_clone:
                routed_penalty = _clone_scalar_penalty(original_penalty)
                setattr(routed_penalty, _SCALAR_CV_ALPHA_MARKER, True)
                self.penalty = routed_penalty
            try:
                return current_cv_fit(self, *args, **kwargs)
            finally:
                if use_scalar_object_clone:
                    self.penalty = original_penalty

        setattr(
            _cv_fit_with_current_public_solver,
            _CV_FIT_SOLVER_SYNC_MARKER,
            True,
        )
        _cv_fit_with_current_public_solver._statgpu_original = current_cv_fit
        PenalizedGLM_CV.fit = _cv_fit_with_current_public_solver


def _install_policy_contract() -> None:
    current = _fit_mixin._preferred_penalized_glm_solver
    if _wrapper_chain_has_marker(current, _POLICY_MARKER):
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
    if _wrapper_chain_has_marker(current, _CV_FIT_VALIDATE_MARKER):
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
    if _wrapper_chain_has_marker(current, _CV_PUBLIC_SOLVER_MARKER):
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

        strict = kwargs.get("strict", args[7] if len(args) > 7 else True)
        from statgpu.solvers import _proximal_irls_quantile as _prox_kernel

        scalar_strict = (
            bool(strict)
            and _penalty_name(getattr(self, "penalty", ""))
            in _NONCONVEX_QUANTILE_PENALTIES
        )
        token = _INTERNAL_CV_RESOLVED_SOLVER.set(True)
        convergence_token = _prox_kernel._STRICT_CV_TARGET.set(scalar_strict)
        try:
            return current_fold(self, *args, **kwargs)
        finally:
            _prox_kernel._STRICT_CV_TARGET.reset(convergence_token)
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
    if not _wrapper_chain_has_marker(current_eval_fn, _CV_EVAL_MARKER):
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
    if _wrapper_chain_has_marker(current_numpy_eval, _CV_EVAL_MARKER):
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
    if _wrapper_chain_has_marker(current, _CV_SCAD_MARKER):
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
    if _wrapper_chain_has_marker(current, _CV_FOLDBATCH_MARKER):
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
    if _wrapper_chain_has_marker(current, _CV_SCORE_CONTEXT_MARKER):
        return

    @wraps(current)
    def _compute_cv_scores_with_quantile_level(self, *args, **kwargs):
        if _loss_name(getattr(self, "loss", "")) != "quantile":
            return current(self, *args, **kwargs)

        token = _QUANTILE_CV_LEVEL.set(_requested_quantile(self))
        try:
            scores = current(self, *args, **kwargs)
        finally:
            _QUANTILE_CV_LEVEL.reset(token)

        strict = kwargs.get("strict", args[8] if len(args) > 8 else True)
        # Strict Quantile CV requires complete fold evidence for every alpha,
        # independent of penalty family.  A numerical/convergence failure in
        # one fold must not be averaged away by the finite-column mean.
        if bool(strict):
            values = np.asarray(scores, dtype=np.float64)
            if values.ndim == 2 and values.shape[0] > 0:
                incomplete = ~np.all(np.isfinite(values), axis=0)
                if np.any(incomplete):
                    values = np.array(values, copy=True)
                    values[:, incomplete] = np.nan
                    scores = values
        return scores

    setattr(
        _compute_cv_scores_with_quantile_level,
        _CV_SCORE_CONTEXT_MARKER,
        True,
    )
    _compute_cv_scores_with_quantile_level._statgpu_original = current
    PenalizedGLM_CV._compute_cv_scores = _compute_cv_scores_with_quantile_level


def _install_explicit_route_guard() -> None:
    current = PenalizedGeneralizedLinearModel._validate_solver_penalty
    if _wrapper_chain_has_marker(current, _VALIDATE_MARKER):
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


def _reinstall_loaded_quantile_solver_layers() -> None:
    """Restore layered Quantile solver semantics after this module is reloaded.

    Normal package import installs the dependent contracts later from the
    penalized package initializer. A direct importlib.reload of this base
    contract recreates its module-level validator, so any already-loaded
    wrappers around that function must be re-applied in the same order to keep
    public solver behavior import-order invariant.
    """
    for module_name, installer_name in (
        (
            "statgpu.linear_model.penalized._quantile_unsupported_solver_guard_contract",
            "install_quantile_unsupported_solver_guard_contract",
        ),
        (
            "statgpu.linear_model.penalized._quantile_smooth_fista_contract",
            "install_quantile_smooth_fista_contract",
        ),
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        installer = getattr(module, installer_name, None)
        if callable(installer):
            installer()


def install_quantile_solver_contract() -> None:
    """Install Quantile solver/provenance/scoring reconciliation idempotently."""
    _install_scalar_cv_penalty_object_contract()
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
    _reinstall_loaded_quantile_solver_layers()


install_quantile_solver_contract()
