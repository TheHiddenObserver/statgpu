"""Ordinary-GLM weighted explicit Newton/L-BFGS public contract.

The ordinary ``GeneralizedLinearModel`` source predates the shared weighted
Newton repair and still rejects every weighted explicit smooth-solver request
before solver entry. Keep this installer narrow: it opens only that public
boundary, preserves the requested solver, and publishes fit-recorded
solver/backend/device provenance after a successful fit.

Family-specific numerical domains belong to the loss/solver layer. In
particular, inverse-power Gamma now obtains and preserves its smooth-domain
interior through ``GammaLoss`` plus the shared Newton/L-BFGS domain hooks rather
than through an estimator-specific ``fit_intercept`` guard or duplicate start.
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
_ALIGNMENT_MARKER = "_statgpu_weighted_explicit_solver_inference_alignment_contract"


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


def _install_inference_alignment_contract() -> None:
    """Keep post-fit GLM design/parameters on a floating numerical dtype.

    Smooth solvers promote integral public designs before optimization. The
    historical inference-state builder instead reused the original design dtype,
    which could cast a successful floating GPU coefficient vector back to an
    integer dtype before log-likelihood or M-estimation inference. Align the
    retained design to at least the fitted coefficient precision before the
    existing layout helper reconstructs ``_X_design`` and ``_params``.
    """
    current = GeneralizedLinearModel._aligned_inference_design_glm
    if getattr(current, _ALIGNMENT_MARKER, False):
        return

    @wraps(current)
    def _aligned_inference_design_with_float_state(self, X_orig):
        backend = _resolve_backend("auto", X_orig)
        if backend == "torch":
            import torch

            coef_dtype = torch.as_tensor(np.asarray(self.coef_)).dtype
            x_dtype = X_orig.dtype if torch.is_floating_point(X_orig) else coef_dtype
            dtype = torch.promote_types(x_dtype, coef_dtype)
            if not dtype.is_floating_point:
                dtype = torch.float64
            X_orig = X_orig.to(dtype=dtype)
        elif backend == "cupy":
            from statgpu.backends._utils import _get_xp

            xp = _get_xp("cupy")
            coef_dtype = xp.asarray(self.coef_).dtype
            dtype = xp.result_type(X_orig.dtype, coef_dtype)
            if getattr(dtype, "kind", "") != "f":
                dtype = xp.float64
            X_orig = X_orig.astype(dtype, copy=False)
        # The existing NumPy branch already converts the inference design to
        # floating point and therefore cannot truncate fitted coefficients.
        return current(self, X_orig)

    setattr(_aligned_inference_design_with_float_state, _ALIGNMENT_MARKER, True)
    _aligned_inference_design_with_float_state._statgpu_original = current
    GeneralizedLinearModel._aligned_inference_design_glm = (
        _aligned_inference_design_with_float_state
    )


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

        from statgpu.backends._utils import _get_xp

        xp = _get_xp(backend_name)
        if self._effective_intercept:
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
        else:
            # Smooth solvers require floating arithmetic even when the public
            # design container is integral. Promote before solver entry so
            # fractional analytic weights cannot be truncated while aligning
            # to the executed design dtype.
            if backend_name == "cupy":
                x_dtype = X.dtype if getattr(X.dtype, "kind", "") == "f" else xp.float64
                X_work = X.astype(x_dtype, copy=False)
            elif backend_name == "torch":
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_work = X.to(dtype=x_dtype)
                y = y.to(X.device).to(x_dtype)
            else:
                x_dtype = (
                    X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
                )
                X_work = X.astype(x_dtype, copy=False)
            p = X.shape[1]

        # ``init_coef=None`` is intentional. Ordinary GLMs expose no public
        # smooth-solver warm start, and losses with a maintained numerical
        # domain (currently inverse-power Gamma) construct their own
        # backend-native interior start inside Newton/L-BFGS. Other losses keep
        # the historical zero/default solver start.
        solver = newton_solver if solver_name == "newton" else lbfgs_solver
        params, n_iter = solver(
            loss,
            None,
            X_work,
            y,
            max_iter=self._max_iter,
            tol=self._tol,
            init_coef=None,
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
    # The later inverse-Gamma consumer installer must not add a second ordinary
    # wrapper. This ordinary owner already delegates family-domain work to the
    # shared loss/solver hooks; marking that contract lets the later installer
    # no-op its historical compatibility wrapper while still patching penalized
    # and CV consumers.
    setattr(_fit_smooth_solver_with_weights, "_statgpu_inverse_gamma_domain_ordinary", True)
    _fit_smooth_solver_with_weights._statgpu_original = current
    GeneralizedLinearModel._fit_smooth_solver = _fit_smooth_solver_with_weights


def _install_fit_provenance_contract() -> None:
    current = GeneralizedLinearModel.fit
    if getattr(current, _FIT_MARKER, False):
        return

    @wraps(current)
    def _fit_with_execution_provenance(self, *args, **kwargs):
        # Publish new provenance only after the existing fit transaction has
        # returned successfully. A failed refit therefore leaves the previous
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
    _install_inference_alignment_contract()
    _install_smooth_solver_contract()
    _install_fit_provenance_contract()
