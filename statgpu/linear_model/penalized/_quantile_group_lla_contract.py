"""Quantile Group SCAD/MCP routing through the canonical group FISTA-LLA path.

The generic group-penalty contract already routes convex Group Lasso through
loss-gradient FISTA rather than the historical Gaussian block update. Group
SCAD/MCP have a separate automatic contract: local linear approximation (LLA)
with an Adaptive Group Lasso surrogate. Quantile was accidentally excluded from
that automatic branch by the legacy ``_SPECIAL_LLA_LOSSES`` classification, so
``solver='auto'`` fell through to direct non-convex proximal FISTA instead of
the documented Group FISTA-LLA algorithm.

This narrow wrapper restores the declared automatic route without changing the
historical meaning of an explicit ``solver='fista'`` request. The same
distinction is preserved inside CV: only an auto-resolved child/final refit uses
Group FISTA-LLA; explicit-FISTA CV remains explicit FISTA.
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


class _QuantileGroupStepScaleProxy:
    """Retain analytic weights across Quantile FISTA-LLA step-scale refreshes.

    The fused FISTA-LLA engine supplies ``sample_weight`` to its initial
    ``loss.lipschitz`` call but omits it from periodic refreshes.  Quantile's
    first-order step scale is objective-weighted, so this route-local proxy
    remembers the initial backend-native weight vector and reuses it later.
    Keeping this state here avoids changing the shared Group FISTA-LLA behavior
    of Huber/GLM losses that is outside PR #166's physical-validation scope.
    """

    def __init__(self, loss):
        self._loss = loss
        self._sample_weight = None

    def __getattr__(self, name):
        return getattr(self._loss, name)

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        if sample_weight is not None:
            self._sample_weight = sample_weight
        effective_weight = (
            sample_weight if sample_weight is not None else self._sample_weight
        )
        return self._loss.lipschitz(
            X,
            coef,
            y=y,
            sample_weight=effective_weight,
        )


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
    return str(getattr(owner, "_solver", "") or "").lower().strip() == "auto"


def _use_auto_group_lla(owner, solver_name) -> bool:
    return (
        _is_quantile_group_nonconvex(owner)
        and str(solver_name or "").lower().strip() == "fista"
        and (_public_solver_is_auto(owner) or _AUTO_CV_GROUP_LLA.get())
    )


def _install_cv_auto_context() -> None:
    """Mark only auto-routed Quantile group CV child/refit fits as LLA-owned."""
    CV = _quantile_contract.PenalizedGLM_CV
    if getattr(CV, _CV_MARKER, False):
        return

    current_fold = CV._cv_fold_general
    current_refit = CV._refit_best

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

    @wraps(current_fold)
    def _cv_fold_with_quantile_group_auto_context(self, *args, **kwargs):
        if not _is_auto_quantile_group_cv(self):
            return current_fold(self, *args, **kwargs)
        token = _AUTO_CV_GROUP_LLA.set(True)
        try:
            return current_fold(self, *args, **kwargs)
        finally:
            _AUTO_CV_GROUP_LLA.reset(token)

    @wraps(current_refit)
    def _refit_with_quantile_group_auto_context(self, *args, **kwargs):
        if not _is_auto_quantile_group_cv(self):
            return current_refit(self, *args, **kwargs)
        token = _AUTO_CV_GROUP_LLA.set(True)
        try:
            return current_refit(self, *args, **kwargs)
        finally:
            _AUTO_CV_GROUP_LLA.reset(token)

    CV._cv_fold_general = _cv_fold_with_quantile_group_auto_context
    CV._refit_best = _refit_with_quantile_group_auto_context
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

        xp = _get_xp(backend_name)
        ref = X if not isinstance(X, np.ndarray) else xp.zeros(1, dtype=xp.float64)
        X_arr = _xp_asarray(X, xp.float64, ref)
        y_arr = _xp_asarray(y, xp.float64, X_arr)
        p = int(X_arr.shape[1])

        # Unweighted/equal-weight intercept fits keep the historical Quantile
        # continuation path exactly. Non-uniform weights or a fixed zero
        # intercept are objective-dependent cases that the backend-native
        # resolver recomputes anyway, so they can skip the legacy full-host
        # path calculation and carry only target/length metadata into it.
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

        from statgpu.solvers import fista_lla_path

        lla_loss = _QuantileGroupStepScaleProxy(self._loss)
        coef, intercept, n_iter = fista_lla_path(
            lla_loss,
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
