"""Ordinary-GLM weighted explicit Newton/L-BFGS public contract.

The ordinary ``GeneralizedLinearModel`` source predates the shared weighted
Newton repair and still rejects every weighted explicit smooth-solver request
before solver entry.  Keep this installer narrow: it opens only that public
boundary, preserves the existing solver algorithms, and publishes fit-recorded
solver/backend/device provenance after a successful fit.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.linear_model._glm_base import (
    GeneralizedLinearModel,
    _torch_promoted_float_dtype,
)


_INIT_MARKER = "_statgpu_weighted_explicit_solver_init_contract"
_FIT_MARKER = "_statgpu_weighted_explicit_solver_fit_contract"
_SMOOTH_MARKER = "_statgpu_weighted_explicit_solver_smooth_contract"


def _resolved_ordinary_solver(self) -> str:
    requested = self._solver.lower() if isinstance(self._solver, str) else self._solver
    if requested != "auto":
        return requested
    pen = getattr(self, "_penalty", None)
    pname = str(getattr(pen, "name", "none")).lower() if pen is not None else "none"
    if pname in (
        "l1",
        "scad",
        "mcp",
        "adaptive_l1",
        "adaptive_lasso",
        "group_lasso",
        "group_mcp",
        "group_scad",
    ):
        return "fista"
    return "irls"


def _fit_device_label(X_design, backend: str) -> str:
    if backend == "numpy":
        return "cpu"
    if backend == "torch":
        return str(X_design.device)
    if backend == "cupy":
        return f"cuda:{int(X_design.device.id)}"
    return str(backend)


def _inverse_gamma_intercept_start(
    loss,
    y,
    sample_weight,
    *,
    backend_name: str,
    p: int,
    dtype,
):
    """Return a stable inverse-link Gamma start when an intercept is present.

    The inverse link requires strictly positive linear predictors.  Starting
    the augmented smooth problem at all zeros places every observation on the
    clipping boundary and can make the first Newton line search fail before a
    meaningful weighted step is evaluated.  Match the existing GLM/FISTA
    family-aware initialization: zero slopes and intercept ``1 / mean(y)``.
    With genuine analytic weights, use the same normalized weighted mean as the
    fitted objective.  Uniform/effectively-uniform weights are normalized away
    before this helper is called so the historical unweighted initialization is
    preserved exactly.  Other families return ``None`` and keep their zero start.
    """
    if getattr(loss, "name", "") != "gamma" or getattr(loss, "link", None) != "inverse_power":
        return None

    if backend_name == "torch":
        import torch

        y_work = y.to(dtype=dtype)
        if sample_weight is None:
            y_mean = torch.mean(y_work)
        else:
            weights = sample_weight.to(y_work.device).to(dtype)
            y_mean = torch.sum(weights * y_work) / torch.sum(weights)
        init = torch.zeros(p + 1, dtype=dtype, device=y_work.device)
        init[-1] = 1.0 / torch.clamp(y_mean, min=1e-12)
        return init

    if backend_name == "cupy":
        import cupy as cp

        y_work = cp.asarray(y, dtype=dtype)
        if sample_weight is None:
            y_mean = cp.mean(y_work)
        else:
            weights = cp.asarray(sample_weight, dtype=dtype)
            y_mean = cp.sum(weights * y_work) / cp.sum(weights)
        init = cp.zeros(p + 1, dtype=dtype)
        init[-1] = 1.0 / cp.maximum(y_mean, cp.asarray(1e-12, dtype=dtype))
        return init

    y_work = np.asarray(y, dtype=dtype)
    if sample_weight is None:
        y_mean = float(np.mean(y_work))
    else:
        weights = np.asarray(sample_weight, dtype=dtype)
        y_mean = float(np.sum(weights * y_work) / np.sum(weights))
    init = np.zeros(p + 1, dtype=dtype)
    init[-1] = 1.0 / max(y_mean, 1e-12)
    return init


def _install_init_contract() -> None:
    current = GeneralizedLinearModel.__init__
    if getattr(current, _INIT_MARKER, False):
        return

    @wraps(current)
    def _init_with_provenance(self, *args, **kwargs):
        current(self, *args, **kwargs)
        self._selected_solver = None
        self._selected_backend_name = None
        self._selected_backend_device = None

    setattr(_init_with_provenance, _INIT_MARKER, True)
    _init_with_provenance._statgpu_original = current
    GeneralizedLinearModel.__init__ = _init_with_provenance


def _install_smooth_solver_contract() -> None:
    current = GeneralizedLinearModel._fit_smooth_solver
    if getattr(current, _SMOOTH_MARKER, False):
        return

    @wraps(current)
    def _fit_smooth_solver_with_weights(
        self,
        X,
        y,
        sample_weight,
        solver_name,
        backend_name,
    ):
        from statgpu.glm_core import get_glm_loss
        from statgpu.solvers import lbfgs_solver, newton_solver

        loss_kwargs = self._get_loss_kwargs()
        loss = get_glm_loss(self.family_to_loss(), **loss_kwargs)
        if not getattr(loss, "has_hessian", False):
            raise ValueError(f"solver='{solver_name}' requires a Hessian.")

        # Inverse-link Gamma has an extra domain contract.  Prepare its weight
        # identity once with the same reviewed rule used by Newton/L-BFGS:
        # validation -> execution-backend/dtype alignment -> uniformity check.
        # The prepared vector is then authoritative both for the no-intercept
        # capability boundary and for the family-valid intercept warm start.
        gamma_start_weight = sample_weight
        is_inverse_gamma = (
            getattr(loss, "name", "") == "gamma"
            and getattr(loss, "link", None) == "inverse_power"
        )
        if is_inverse_gamma and sample_weight is not None:
            from statgpu.solvers._newton import _prepare_newton_sample_weight

            gamma_start_weight = _prepare_newton_sample_weight(
                sample_weight,
                X.shape[0],
                backend_name,
                X,
            )
            # Without an intercept there is no generic way to guarantee that
            # an arbitrary design admits X @ beta > 0 for every row and no
            # maintained public init_coef exists to provide such a feasible
            # point.  Keep only the genuine-nonuniform newly opened row closed;
            # uniform/effectively-uniform weights retain the historical path.
            if not self._effective_intercept and gamma_start_weight is not None:
                raise ValueError(
                    "weighted explicit Newton/L-BFGS for Gamma inverse_power "
                    "requires fit_intercept=True for genuine non-uniform "
                    "sample_weight; no maintained family-valid no-intercept "
                    "initialization is available."
                )

        init_coef = None
        if self._effective_intercept:
            from statgpu.backends._utils import _get_xp

            xp = _get_xp(backend_name)
            if backend_name == "cupy":
                x_dtype = X.dtype if getattr(X.dtype, "kind", "") == "f" else xp.float64
                X_float = X.astype(x_dtype, copy=False)
                X_work = xp.column_stack(
                    [X_float, xp.ones(X.shape[0], dtype=x_dtype)]
                )
            elif backend_name == "torch":
                import torch

                x_dtype = _torch_promoted_float_dtype(X, y)
                X_float = X.to(dtype=x_dtype)
                y = y.to(X.device).to(x_dtype)
                X_work = torch.column_stack(
                    [
                        X_float,
                        torch.ones(
                            X.shape[0], dtype=x_dtype, device=X.device
                        ),
                    ]
                )
            else:
                x_dtype = (
                    X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
                )
                X_float = X.astype(x_dtype, copy=False)
                X_work = np.column_stack(
                    [X_float, np.ones(X.shape[0], dtype=x_dtype)]
                )
            p = X.shape[1]
            init_coef = _inverse_gamma_intercept_start(
                loss,
                y,
                gamma_start_weight,
                backend_name=backend_name,
                p=p,
                dtype=x_dtype,
            )
        else:
            if backend_name == "torch":
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_work = X.to(dtype=x_dtype)
                y = y.to(X.device).to(x_dtype)
            else:
                X_work = X
            p = X.shape[1]

        if solver_name == "newton":
            params, n_iter = newton_solver(
                loss,
                None,
                X_work,
                y,
                max_iter=self._max_iter,
                tol=self._tol,
                init_coef=init_coef,
                sample_weight=sample_weight,
            )
        else:
            params, n_iter = lbfgs_solver(
                loss,
                None,
                X_work,
                y,
                max_iter=self._max_iter,
                tol=self._tol,
                init_coef=init_coef,
                sample_weight=sample_weight,
            )

        params_np = _to_numpy(params)
        self.n_iter_ = n_iter
        if self._effective_intercept:
            self.coef_ = params_np[:p]
            self.intercept_ = float(params_np[p])
        else:
            self.coef_ = params_np.copy()
            self.intercept_ = 0.0
        self._params = (
            np.concatenate([[self.intercept_], self.coef_])
            if self._effective_intercept
            else self.coef_.copy()
        )
        self._df_resid = self._nobs - (
            X.shape[1] + (1 if self._effective_intercept else 0)
        )

    setattr(_fit_smooth_solver_with_weights, _SMOOTH_MARKER, True)
    _fit_smooth_solver_with_weights._statgpu_original = current
    GeneralizedLinearModel._fit_smooth_solver = _fit_smooth_solver_with_weights


def _install_fit_provenance_contract() -> None:
    current = GeneralizedLinearModel.fit
    if getattr(current, _FIT_MARKER, False):
        return

    @wraps(current)
    def _fit_with_execution_provenance(self, *args, **kwargs):
        # Publish new provenance only after the existing fit transaction has
        # returned successfully.  A failed refit therefore leaves the previous
        # successful provenance untouched, matching the existing ordinary-GLM
        # state behavior rather than inventing a new invalidation contract.
        result = current(self, *args, **kwargs)
        solver_name = _resolved_ordinary_solver(self)
        X_design = getattr(self, "_X_design", None)
        if X_design is None:
            raise RuntimeError(
                "Successful GLM fit did not retain its numerical design for provenance."
            )
        backend = _resolve_backend("auto", X_design)
        self._selected_solver = solver_name
        self._selected_backend_name = backend
        self._selected_backend_device = _fit_device_label(X_design, backend)
        return result

    setattr(_fit_with_execution_provenance, _FIT_MARKER, True)
    _fit_with_execution_provenance._statgpu_original = current
    GeneralizedLinearModel.fit = _fit_with_execution_provenance


def install_glm_weighted_explicit_solver_contract() -> None:
    """Install the bounded ordinary-GLM weighted smooth-solver contract."""

    _install_init_contract()
    _install_smooth_solver_contract()
    _install_fit_provenance_contract()
