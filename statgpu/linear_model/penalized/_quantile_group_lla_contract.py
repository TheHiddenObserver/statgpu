"""Quantile Group SCAD/MCP routing through the canonical group FISTA-LLA path.

The generic group-penalty contract already routes convex Group Lasso through
loss-gradient FISTA rather than the historical Gaussian block update. Group
SCAD/MCP have a separate canonical contract: local linear approximation (LLA)
with an Adaptive Group Lasso surrogate. Quantile was accidentally excluded from
that branch by the legacy ``_SPECIAL_LLA_LOSSES`` classification, so its public
``solver='auto'`` path fell through to direct non-convex proximal FISTA instead
of the documented Group FISTA-LLA algorithm.

This narrow wrapper restores the declared route without changing other losses.
It also resolves the Quantile continuation start from the same analytic weights,
intercept policy, and normalized pinball score used by the fitted objective.
"""

from __future__ import annotations

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


_MARKER = "_statgpu_quantile_group_lla_contract"
_GROUP_NONCONVEX = frozenset({"group_mcp", "gmcp", "group_scad", "gscad"})


def _penalty_name(owner) -> str:
    return str(
        getattr(getattr(owner, "_penalty", None), "name", getattr(owner, "penalty", ""))
        or ""
    ).lower().strip()


def _is_quantile_group_nonconvex(owner) -> bool:
    return (
        str(getattr(getattr(owner, "_loss", None), "name", "")).lower().strip()
        == "quantile"
        and _penalty_name(owner) in _GROUP_NONCONVEX
    )


def _install_quantile_group_lla_route() -> None:
    current = _fit_mixin._PenalizedFitMixin._fit_loss_backend
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def _fit_loss_backend_with_quantile_group_lla(
        self, X, y, sample_weight, solver_name, backend_name
    ):
        if not _is_quantile_group_nonconvex(self):
            return current(self, X, y, sample_weight, solver_name, backend_name)

        # Group SCAD/MCP are public auto-routed LLA objectives. Do not let an
        # unrelated explicit solver request silently enter this resolved route.
        if str(solver_name or "").lower().strip() != "fista":
            return current(self, X, y, sample_weight, solver_name, backend_name)

        xp = _get_xp(backend_name)
        ref = X if not isinstance(X, np.ndarray) else xp.zeros(1, dtype=xp.float64)
        X_arr = _xp_asarray(X, xp.float64, ref)
        y_arr = _xp_asarray(y, xp.float64, X_arr)
        p = int(X_arr.shape[1])

        # Build only target/length metadata here. The weighted resolver computes
        # the actual lambda-style start from backend-native X/y/weights.
        alpha_path, max_lla_per_step, mi_path = (
            _continuation_contract._quantile_path_metadata(self)
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

        coef, intercept, n_iter = fista_lla_path(
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


_install_quantile_group_lla_route()


__all__ = []
