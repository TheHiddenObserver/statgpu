"""Weighted explicit Newton/L-BFGS GLM public-contract reconciliation.

The ordinary ``GeneralizedLinearModel`` source predates the shared weighted
Newton repair and still rejects every weighted explicit smooth-solver request
before solver entry. Keep this installer narrow: it opens that ordinary public
boundary, preserves the requested solver, publishes fit-recorded
solver/backend/device provenance, and keeps ordinary/penalized post-fit
inference consumers on the same effective weight objective as the smooth solve.

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
_INFERENCE_MARKER = "_statgpu_weighted_explicit_solver_inference_weight_contract"
_PENALIZED_INFERENCE_MARKER = (
    "_statgpu_weighted_explicit_solver_penalized_inference_weight_contract"
)


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


def _canonicalize_smooth_inference_weight_state(self, solver_name) -> None:
    """Mirror the ordinary smooth solver's actual weight classification."""
    if solver_name not in ("newton", "lbfgs"):
        return
    sample_weight = getattr(self, "_sample_weight_inf", None)
    if sample_weight is None:
        return

    # The ordinary smooth owner records the classification on the exact
    # solver-side design dtype/device after a successful solve. Prefer that
    # identity to reclassifying from the post-fit inference design, because the
    # NumPy reporting layer may promote a float32 fit design to float64.
    fitted_unweighted = getattr(
        self, "_statgpu_smooth_effective_unweighted", None
    )
    if fitted_unweighted is True:
        self._sample_weight_inf = None
        return
    if fitted_unweighted is False:
        return

    # Compatibility fallback for an object fitted before this marker existed.
    X_design = getattr(self, "_X_design", None)
    if X_design is None:
        return
    from statgpu.solvers._smooth_domain import _prepare_analytic_sample_weight

    backend = _resolve_backend("auto", X_design)
    prepared = _prepare_analytic_sample_weight(
        sample_weight,
        X_design.shape[0],
        backend,
        X_design,
    )
    if prepared is None:
        self._sample_weight_inf = None


def _penalized_inference_sample_weight(self, sample_weight):
    """Return the weight state matching a penalized smooth GLM fit objective."""
    if sample_weight is None:
        return None
    solver_name = str(getattr(self, "_selected_solver", "") or "").lower()
    if solver_name not in ("newton", "lbfgs"):
        return sample_weight

    from statgpu.glm_core._base import GLMLoss

    if not isinstance(getattr(self, "_loss", None), GLMLoss):
        return sample_weight

    backend = str(getattr(self, "_selected_backend_name", "") or "").lower()
    if backend not in ("numpy", "cupy", "torch"):
        return sample_weight

    # ``_PenalizedFitMixin._fit_loss_backend`` converts the smooth GLM design
    # to float64 before Newton/L-BFGS. Reproduce that exact numerical dtype for
    # the effective-uniform classification instead of classifying from the
    # caller's original (possibly float32/integer) design or weight dtype.
    from statgpu.backends._utils import _get_xp, xp_asarray
    from statgpu.solvers._smooth_domain import _prepare_analytic_sample_weight

    xp = _get_xp(backend)
    ref = xp_asarray(
        sample_weight,
        dtype=xp.float64,
        xp=xp,
        ref_arr=sample_weight,
    )
    prepared = _prepare_analytic_sample_weight(
        sample_weight,
        int(ref.shape[0]),
        backend,
        ref,
    )
    return None if prepared is None else sample_weight


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
        self._statgpu_smooth_effective_unweighted = None

    setattr(_init_with_provenance, _INIT_MARKER, True)
    _init_with_provenance._statgpu_original = current
    GeneralizedLinearModel.__init__ = _init_with_provenance


def _install_inference_alignment_contract() -> None:
    """Keep post-fit ordinary-GLM design/parameters on a floating dtype."""
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
            coef_dtype = np.asarray(self.coef_).dtype
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
        from statgpu.solvers._smooth_domain import _prepare_analytic_sample_weight

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

        # Record the exact weight identity used by the successful solver before
        # later reporting/inference layers can change dtype. This is a boolean
        # provenance fact, not an additional public fitted parameter.
        prepared_weight = _prepare_analytic_sample_weight(
            sample_weight,
            X_work.shape[0],
            backend_name,
            X_work,
        )
        self._statgpu_smooth_effective_unweighted = prepared_weight is None

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
    setattr(_fit_smooth_solver_with_weights, "_statgpu_inverse_gamma_domain_ordinary", True)
    _fit_smooth_solver_with_weights._statgpu_original = current
    GeneralizedLinearModel._fit_smooth_solver = _fit_smooth_solver_with_weights


def _install_inference_weight_contract() -> None:
    current = GeneralizedLinearModel._compute_inference
    if getattr(current, _INFERENCE_MARKER, False):
        return

    @wraps(current)
    def _compute_inference_with_fit_weight_contract(self, *args, **kwargs):
        solver_name = getattr(self, "_fit_metadata", {}).get("solver_used")
        _canonicalize_smooth_inference_weight_state(self, solver_name)
        return current(self, *args, **kwargs)

    setattr(_compute_inference_with_fit_weight_contract, _INFERENCE_MARKER, True)
    _compute_inference_with_fit_weight_contract._statgpu_original = current
    GeneralizedLinearModel._compute_inference = _compute_inference_with_fit_weight_contract


def _install_penalized_inference_weight_contract() -> None:
    from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

    current = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    if getattr(current, _PENALIZED_INFERENCE_MARKER, False):
        return

    @wraps(current)
    def _compute_post_fit_with_fit_weight_contract(
        self, X, y, sample_weight=None
    ):
        effective_weight = _penalized_inference_sample_weight(self, sample_weight)
        return current(self, X, y, sample_weight=effective_weight)

    setattr(
        _compute_post_fit_with_fit_weight_contract,
        _PENALIZED_INFERENCE_MARKER,
        True,
    )
    _compute_post_fit_with_fit_weight_contract._statgpu_original = current
    PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = (
        _compute_post_fit_with_fit_weight_contract
    )


def _install_fit_provenance_contract() -> None:
    current = GeneralizedLinearModel.fit
    if getattr(current, _FIT_MARKER, False):
        return

    @wraps(current)
    def _fit_with_execution_provenance(self, *args, **kwargs):
        result = current(self, *args, **kwargs)
        solver_name = _resolved_ordinary_solver(self)
        _canonicalize_smooth_inference_weight_state(self, solver_name)
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
    """Install the bounded weighted smooth-GLM contract."""

    _install_init_contract()
    _install_inference_alignment_contract()
    _install_smooth_solver_contract()
    _install_inference_weight_contract()
    _install_penalized_inference_weight_contract()
    _install_fit_provenance_contract()
