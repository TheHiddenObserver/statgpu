"""Quantile Group SCAD/MCP automatic routing through Proximal IRLS-LLA.

Convex Group Lasso remains on loss-gradient Group FISTA. Group SCAD/MCP need a
non-convex local-linear-approximation (LLA) outer loop. For Quantile/check loss,
the automatic route uses the loss's IRLS/MM quadratic majorization and solves
each convex Adaptive-Group-Lasso surrogate with a backend-native exact-WLS ADMM
inner solver. This avoids treating a discontinuous pinball subgradient as a
smooth fixed-step FISTA problem.

The wrapper applies only to the public ``solver='auto'`` policy. An explicit
``solver='fista'`` request remains explicit proximal FISTA in direct fits and in
CV children/final refits.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._array_ops import _xp_asarray
from statgpu.backends._utils import _get_xp
from statgpu.solvers._quantile_continuation import (
    resolve_auto_quantile_continuation_path,
)
from . import _fit_mixin
from . import _quantile_continuation_contract as _continuation_contract
from . import _quantile_solver_contract as _quantile_contract


_MARKER = "_statgpu_quantile_group_lla_contract"
_CV_MARKER = "_statgpu_quantile_group_lla_cv_auto_context_contract"
_GROUP_NONCONVEX = frozenset({"group_mcp", "gmcp", "group_scad", "gscad"})
_AUTO_CV_GROUP_LLA = ContextVar("statgpu_quantile_group_auto_cv_lla", default=False)
_CV_GROUP_LLA_CANDIDATE_STRICT = ContextVar(
    "statgpu_quantile_group_cv_candidate_strict", default=False
)
_EXECUTED_SOLVER = "group_proximal_irls_lla"


def _penalty_name(owner) -> str:
    return str(
        getattr(getattr(owner, "_penalty", None), "name", getattr(owner, "penalty", ""))
        or ""
    ).lower().strip()


def _loss_name(owner) -> str:
    return str(
        getattr(getattr(owner, "_loss", None), "name", getattr(owner, "loss", ""))
        or ""
    ).lower().strip()


def _is_quantile_group_nonconvex(owner) -> bool:
    return _loss_name(owner) == "quantile" and _penalty_name(owner) in _GROUP_NONCONVEX


def _public_solver_is_auto(owner) -> bool:
    # Public constructor parameters are refit controls. Read the public value
    # here because this predicate can run in an outer CV wrapper before the
    # Quantile fit-entry synchronization wrapper has copied it to _solver.
    value = getattr(owner, "solver", getattr(owner, "_solver", ""))
    return str(value or "").lower().strip() == "auto"


def _use_auto_group_lla(owner, solver_name) -> bool:
    return (
        _is_quantile_group_nonconvex(owner)
        and str(solver_name or "").lower().strip() == "fista"
        and (_public_solver_is_auto(owner) or _AUTO_CV_GROUP_LLA.get())
    )


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _validate_direct_group_lla_controls(owner) -> None:
    # Direct public parameter replacement is part of the penalized refit API.
    # Validate the current public values, then synchronize the normalized
    # runtime fields consumed by continuation/solver code.
    owner._max_iter = _positive_integer(owner.max_iter, "max_iter")
    owner._max_lla_iters = _positive_integer(
        owner.max_lla_iters, "max_lla_iters"
    )
    owner._tol = _finite_positive(owner.tol, "tol")
    owner._lla_tol = _finite_positive(owner.lla_tol, "lla_tol")


def _validate_cv_group_lla_controls(owner) -> None:
    owner._max_iter = _positive_integer(owner.max_iter, "max_iter")
    owner._tol = _finite_positive(owner.tol, "tol")


def _install_cv_auto_context() -> None:
    """Own auto Quantile Group CV routing and candidate eligibility."""
    CV = _quantile_contract.PenalizedGLM_CV
    if getattr(CV, _CV_MARKER, False):
        return

    current_fit = CV.fit
    current_fold = CV._cv_fold_general
    current_refit = CV._refit_best
    current_scores = CV._compute_cv_scores

    def _is_auto_quantile_group_cv(owner) -> bool:
        penalty_name = str(
            getattr(getattr(owner, "penalty", None), "name", getattr(owner, "penalty", ""))
            or ""
        ).lower().strip()
        loss_name = str(getattr(owner, "loss", "") or "").lower().strip()
        return (
            loss_name == "quantile"
            and penalty_name in _GROUP_NONCONVEX
            and _public_solver_is_auto(owner)
        )

    @wraps(current_fit)
    def _fit_with_quantile_group_stopping_controls(self, *args, **kwargs):
        if _is_auto_quantile_group_cv(self):
            # Synchronize public refit-time controls before _fit_standard()
            # derives two-stage screening budgets from the private runtime
            # fields. If pre-validation fails, clear any prior successful CV
            # result because this wrapper sits outside the group transaction.
            try:
                _validate_cv_group_lla_controls(self)
            except Exception:
                self._reset_cv_fit_state()
                raise
        return current_fit(self, *args, **kwargs)

    @wraps(current_fold)
    def _cv_fold_with_quantile_group_auto_context(self, *args, **kwargs):
        if not _is_auto_quantile_group_cv(self):
            return current_fold(self, *args, **kwargs)
        strict = kwargs.get("strict", args[7] if len(args) > 7 else True)
        auto_token = _AUTO_CV_GROUP_LLA.set(True)
        strict_token = _CV_GROUP_LLA_CANDIDATE_STRICT.set(bool(strict))
        try:
            return current_fold(self, *args, **kwargs)
        finally:
            _CV_GROUP_LLA_CANDIDATE_STRICT.reset(strict_token)
            _AUTO_CV_GROUP_LLA.reset(auto_token)

    @wraps(current_refit)
    def _refit_with_quantile_group_auto_context(self, *args, **kwargs):
        if not _is_auto_quantile_group_cv(self):
            return current_refit(self, *args, **kwargs)
        _validate_cv_group_lla_controls(self)
        token = _AUTO_CV_GROUP_LLA.set(True)
        try:
            return current_refit(self, *args, **kwargs)
        finally:
            _AUTO_CV_GROUP_LLA.reset(token)

    @wraps(current_scores)
    def _scores_with_complete_quantile_group_candidates(self, *args, **kwargs):
        if not _is_auto_quantile_group_cv(self):
            return current_scores(self, *args, **kwargs)
        _validate_cv_group_lla_controls(self)
        scores = current_scores(self, *args, **kwargs)
        strict = kwargs.get("strict", args[8] if len(args) > 8 else True)
        if not bool(strict):
            return scores
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            return scores
        incomplete = ~np.all(np.isfinite(values), axis=0)
        if not np.any(incomplete):
            return scores
        values = np.array(values, copy=True)
        values[:, incomplete] = np.nan
        return values

    CV.fit = _fit_with_quantile_group_stopping_controls
    CV._cv_fold_general = _cv_fold_with_quantile_group_auto_context
    CV._refit_best = _refit_with_quantile_group_auto_context
    CV._compute_cv_scores = _scores_with_complete_quantile_group_candidates
    setattr(CV, _CV_MARKER, True)


def _install_quantile_group_lla_route() -> None:
    current = _fit_mixin._PenalizedFitMixin._fit_loss_backend
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def _fit_loss_backend_with_quantile_group_lla(
        self, X, y, sample_weight, solver_name, backend_name
    ):
        if not _use_auto_group_lla(self, solver_name):
            return current(self, X, y, sample_weight, solver_name, backend_name)

        _validate_direct_group_lla_controls(self)
        xp = _get_xp(backend_name)
        ref = X if not isinstance(X, np.ndarray) else xp.zeros(1, dtype=xp.float64)
        X_arr = _xp_asarray(X, xp.float64, ref)
        y_arr = _xp_asarray(y, xp.float64, X_arr)
        p = int(X_arr.shape[1])

        # Unweighted/equal-weight intercept fits keep the historical Quantile
        # continuation path exactly. Non-uniform weights or a fixed-zero
        # intercept are objective-dependent cases whose start is recomputed on
        # the selected backend, so only target/length metadata is needed first.
        needs_objective_resolve = (
            not self._effective_intercept
            or _continuation_contract._is_nonuniform_weight(sample_weight)
        )
        if needs_objective_resolve:
            alpha_path, max_lla_per_step, mi_path = (
                _continuation_contract._quantile_path_metadata(self)
            )
        else:
            alpha_path, max_lla_per_step, mi_path = self._compute_lla_path(
                X_arr,
                y_arr,
                p,
                "quantile",
            )
        alpha_path = resolve_auto_quantile_continuation_path(
            self._loss,
            X_arr,
            y_arr,
            alpha_path,
            sample_weight=sample_weight,
            fit_intercept=self._effective_intercept,
        )

        from statgpu.solvers._quantile_group_proximal_irls_lla import (
            quantile_group_proximal_irls_lla_solver,
        )

        coef, intercept, n_iter = quantile_group_proximal_irls_lla_solver(
            self._loss,
            self._penalty,
            X_arr,
            y_arr,
            alpha_path=alpha_path,
            max_lla_per_step=max_lla_per_step,
            lla_tol=getattr(self, "_lla_tol", 1e-6),
            max_iter=mi_path,
            tol=self._tol,
            fit_intercept=self._effective_intercept,
            sample_weight=sample_weight,
            init_coef=getattr(self, "_init_coef", None),
            init_intercept=getattr(self, "_init_intercept", None),
            fail_on_target_nonconvergence=_CV_GROUP_LLA_CANDIDATE_STRICT.get(),
        )

        coef_np = np.asarray(_to_numpy(coef), dtype=np.float64).reshape(-1)
        self.n_iter_ = int(n_iter)
        self.coef_ = coef_np.copy()
        self.intercept_ = float(intercept) if self._effective_intercept else 0.0
        if self._effective_intercept:
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self._params = self.coef_.copy()
        self._df_resid = self._nobs - (
            p + (1 if self._effective_intercept else 0)
        )
        self._selected_solver = _EXECUTED_SOLVER

        if backend_name == "cupy":
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()
        return None

    setattr(_fit_loss_backend_with_quantile_group_lla, _MARKER, True)
    _fit_loss_backend_with_quantile_group_lla._statgpu_original = current
    _fit_mixin._PenalizedFitMixin._fit_loss_backend = (
        _fit_loss_backend_with_quantile_group_lla
    )


_install_cv_auto_context()
_install_quantile_group_lla_route()


__all__ = []