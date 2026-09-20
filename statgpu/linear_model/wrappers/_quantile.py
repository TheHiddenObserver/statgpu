"""Quantile regression with bootstrap inference support."""

import hashlib
import math as _math
import warnings
from numbers import Integral, Real
from typing import Optional
import numpy as np

# Pre-computed scalar constants (Python floats, safe for GPU tensor broadcast)
_INV_SQRT_2PI = 1.0 / _math.sqrt(2.0 * _math.pi)
# Bound the temporary B_chunk x p x p normal-equation tensor used by
# batched bootstrap IRLS. The cap is in matrix elements, independent of dtype.
_BOOTSTRAP_IRLS_GRAM_ELEMENT_CAP = 2_000_000
_BOOTSTRAP_IRLS_EPS = 1e-8


def _align_quantile_fit_inputs(X_arr, y_arr, sample_weight_arr, backend_name):
    """Align response/weights to the concrete device that owns the design."""
    if backend_name == "numpy":
        return y_arr, sample_weight_arr, "cpu"

    if backend_name == "torch":
        target = X_arr.device
        y_arr = y_arr.to(target)
        if sample_weight_arr is not None:
            sample_weight_arr = sample_weight_arr.to(target)
        return y_arr, sample_weight_arr, str(target)

    if backend_name == "cupy":
        from statgpu.backends._utils import _cupy_asarray_on_device

        device_id = int(X_arr.device.id)
        y_arr = _cupy_asarray_on_device(y_arr, device_id)
        if sample_weight_arr is not None:
            sample_weight_arr = _cupy_asarray_on_device(
                sample_weight_arr,
                device_id,
            )
        return y_arr, sample_weight_arr, f"cuda:{device_id}"

    raise RuntimeError(
        f"Unsupported QuantileRegression backend provenance: {backend_name!r}"
    )


def _bootstrap_schedule_to_backend(schedule, resid, backend, xp):
    """Move a deterministic resampling schedule to the exact numerical device."""
    if backend == "torch":
        import torch

        return torch.as_tensor(
            schedule,
            dtype=torch.long,
            device=resid.device,
        )
    if backend == "cupy":
        with xp.cuda.Device(int(resid.device.id)):
            return xp.asarray(schedule, dtype=xp.int64)
    return schedule


def _center_quantile_bootstrap_residuals(resid, tau, xp):
    """Center residuals at their empirical tau-quantile on the active backend."""
    from statgpu.solvers._quantile_continuation import (
        _lower_empirical_quantile_backend,
    )

    center = _lower_empirical_quantile_backend(resid, float(tau), xp)
    return resid - center


class _BootstrapIRLSConvergenceError(RuntimeError):
    """Internal bootstrap-IRLS exhaustion carrying diagnostic progress."""

    def __init__(self, message, *, n_iter, chunk_size):
        super().__init__(message)
        self.n_iter = int(n_iter)
        self.chunk_size = int(chunk_size)


def _batched_quantile_irls(
    X,
    y_matrix,
    init_params,
    tau,
    *,
    max_iter,
    tol,
    xp,
):
    """Solve independent bootstrap Quantile problems by batched IRLS/MM.

    Parameters
    ----------
    X : array, shape (n, p)
        Numerical design, including the intercept column when requested.
    y_matrix : array, shape (n, B)
        Bootstrap responses, one draw per column.
    init_params : array, shape (p,)
        Fitted full-sample parameter vector used as a warm start.
    tau : float
        Target quantile.
    max_iter : int
        Maximum IRLS iterations for every draw chunk.
    tol : float
        Maximum coefficient-change norm required for convergence.
    xp : module
        NumPy, CuPy, or Torch module owning the numerical arrays.

    Returns
    -------
    params : array, shape (p, B)
        Converged bootstrap parameter matrix on the active backend/device.
    n_iter : int
        Maximum IRLS iterations used by any draw chunk.
    chunk_size : int
        Number of bootstrap draws solved together in each normal-equation batch.
    """
    from statgpu.backends._utils import xp_asarray, xp_eye

    X_work = xp_asarray(
        X,
        dtype=xp.float64,
        xp=xp,
        ref_arr=X,
    )
    y_work = xp_asarray(
        y_matrix,
        dtype=xp.float64,
        xp=xp,
        ref_arr=X_work,
    )
    init = xp_asarray(
        init_params,
        dtype=xp.float64,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)

    n = int(X_work.shape[0])
    p = int(X_work.shape[1])
    B = int(y_work.shape[1])
    if int(y_work.shape[0]) != n:
        raise ValueError("bootstrap response matrix must have n_samples rows")
    if int(init.shape[0]) != p:
        raise ValueError("bootstrap init_params must match numerical design width")
    if B < 1:
        raise ValueError("bootstrap response matrix must contain at least one draw")

    gram_per_draw = max(p * p, 1)
    chunk_size = max(
        1,
        min(
            B,
            int(_BOOTSTRAP_IRLS_GRAM_ELEMENT_CAP // gram_per_draw),
        ),
    )
    eye = xp_eye(
        p,
        xp.float64,
        xp,
        ref_arr=X_work,
    )
    chunks = []
    max_used_iter = 0
    tau = float(tau)
    eps = float(_BOOTSTRAP_IRLS_EPS)

    for start in range(0, B, chunk_size):
        stop = min(start + chunk_size, B)
        y_chunk = y_work[:, start:stop]
        width = int(stop - start)
        if xp.__name__ == "torch":
            beta = init.reshape(-1, 1).repeat(1, width)
        else:
            beta = xp.repeat(init.reshape(-1, 1), width, axis=1)

        converged = False
        for iteration in range(int(max_iter)):
            residual = y_chunk - X_work @ beta
            abs_residual = xp.abs(residual)
            if xp.__name__ == "torch":
                eps_arr = xp.as_tensor(
                    eps,
                    dtype=abs_residual.dtype,
                    device=abs_residual.device,
                )
                abs_safe = xp.maximum(abs_residual, eps_arr)
                negative = (residual < 0.0).to(abs_residual.dtype)
            else:
                abs_safe = xp.maximum(abs_residual, eps)
                negative = (residual < 0.0).astype(
                    abs_residual.dtype,
                    copy=False,
                )

            weight = (
                tau + (1.0 - 2.0 * tau) * negative
            ) / abs_safe

            # One weighted normal equation per bootstrap draw. Keep the
            # draw dimension batched while chunking B to bound B*p*p memory.
            gram = xp.einsum(
                "ni,nb,nj->bij",
                X_work,
                weight,
                X_work,
            )
            rhs = xp.einsum(
                "ni,nb->bi",
                X_work,
                weight * y_chunk,
            )
            system = gram + eps * eye[None, :, :]
            solved = xp.linalg.solve(
                system,
                rhs[..., None],
            )[..., 0]
            beta_new = solved.T

            finite = xp.all(xp.isfinite(beta_new))
            if not bool(finite.item() if hasattr(finite, "item") else finite):
                raise FloatingPointError(
                    "QuantileRegression bootstrap IRLS produced non-finite "
                    "parameters"
                )

            delta_by_draw = xp.sqrt(
                xp.sum((beta_new - beta) * (beta_new - beta), axis=0)
            )
            max_delta = xp.max(delta_by_draw)
            max_delta_value = float(
                max_delta.item() if hasattr(max_delta, "item") else max_delta
            )
            beta = beta_new
            if max_delta_value < float(tol):
                converged = True
                used_iter = iteration + 1
                break

        if not converged:
            raise _BootstrapIRLSConvergenceError(
                "QuantileRegression bootstrap IRLS did not converge within "
                f"{int(max_iter)} iterations",
                n_iter=int(max_iter),
                chunk_size=chunk_size,
            )

        max_used_iter = max(max_used_iter, int(used_iter))
        chunks.append(beta)

    if len(chunks) == 1:
        params = chunks[0]
    elif xp.__name__ == "torch":
        params = xp.cat(chunks, dim=1)
    else:
        params = xp.concatenate(chunks, axis=1)
    return params, max_used_iter, chunk_size


from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.losses._quantile import QuantileLoss
from statgpu.solvers import fista_solver
from statgpu.solvers._convergence import ConvergenceWarning


class QuantileRegression(BaseEstimator):
    """Quantile regression with bootstrap inference.

    Minimizes pinball loss: (1/n) Σ ρ_τ(y_i - η_i).
    Uses FISTA solver (no Hessian required).

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile in (0, 1).
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    tol : float, default=1e-4
    device : str or Device, default='auto'
    compute_inference : bool, default=False
        If True, compute SE, p-values, CI.
    inference_method : str, default='kernel'
        'kernel': Powell (1991) sandwich covariance with kernel density.
        'bootstrap': i.i.d. residual bootstrap (batched Quantile IRLS/MM) with percentile
            CI, bootstrap sign-test p-values, and bootstrap std errors. Fitted
            residuals are centered at their empirical tau-quantile before
            resampling so the bootstrap error distribution preserves a zero
            tau-quantile. This is an exchangeable-residual procedure; it is not
            a wild/multiplier
            bootstrap and does not claim heteroscedastic-robust coverage.
            All backends (CPU/GPU) use the same batched solver for consistency.
    kernel : str, default='epa'
        Kernel for sparsity estimation: 'epa' (Epanechnikov), 'gau' (Gaussian),
        'biw' (Biweight), 'cos' (Cosine), 'par' (Parzen).  Only for
        inference_method='kernel'.
    bandwidth : str, default='hsheather'
        Bandwidth rule: 'hsheather' (Hall-Sheather), 'bofinger' (Bofinger),
        'chamberlain' (Chamberlain).  Only for inference_method='kernel'.
    n_bootstrap : int, default=200
        Number of bootstrap resamples; bootstrap inference requires at least 2.
    random_state : int or None, default=42
        Seed for residual-bootstrap resampling.
    gpu_memory_cleanup : bool, default=False
    """

    def __init__(
        self,
        quantile: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        device: Device = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = False,
        inference_method: str = "kernel",
        kernel: str = "epa",
        bandwidth: str = "hsheather",
        n_bootstrap: int = 200,
        random_state: int = 42,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        if (
            isinstance(quantile, (bool, np.bool_))
            or not isinstance(quantile, Real)
            or not np.isfinite(float(quantile))
            or not 0.0 < float(quantile) < 1.0
        ):
            raise ValueError(f"quantile must be a finite real number in (0, 1), got {quantile}")
        self.quantile = float(quantile)
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.compute_inference = compute_inference
        self.inference_method = inference_method
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.gpu_memory_cleanup = gpu_memory_cleanup

        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = None
        self._params = None
        self._bse = None
        self._zvalues = None
        self._tvalues = None
        self._statistic = None
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._bootstrap_n_iter_ = None
        self._bootstrap_schedule_sha256_ = None
        self._bootstrap_residual_centering_ = None
        self._bootstrap_draw_chunk_size_ = None
        self._fitted = False

    def _reset_fit_state(self):
        """Clear all result-bearing state after a failed or new fit attempt."""
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = None
        self._params = None
        self._bse = None
        self._zvalues = None
        self._tvalues = None
        self._statistic = None
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._bootstrap_n_iter_ = None
        self._bootstrap_schedule_sha256_ = None
        self._bootstrap_residual_centering_ = None
        self._bootstrap_draw_chunk_size_ = None
        self._fitted = False
        self._selected_backend_name = None
        self._selected_backend_device = None
        self.__dict__.pop("n_features_in_", None)

    def _validate_public_controls(self):
        """Validate and synchronize mutable public controls before numerics."""
        # Direct public attribute replacement is a supported refit pattern.
        # Keep clone-facing public objects untouched while rebuilding the
        # normalized runtime mirrors consumed by numerical/inference code.
        quantile = self.quantile
        fit_intercept = self.fit_intercept
        max_iter = self.max_iter
        tol = self.tol
        compute_inference = self.compute_inference
        inference_method = self.inference_method
        n_bootstrap = self.n_bootstrap
        gpu_memory_cleanup = self.gpu_memory_cleanup
        device = self.device

        if (
            isinstance(quantile, (bool, np.bool_))
            or not isinstance(quantile, Real)
            or not np.isfinite(float(quantile))
            or not 0.0 < float(quantile) < 1.0
        ):
            raise ValueError("quantile must be a finite real number in (0, 1)")
        self._quantile = float(quantile)

        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        self._fit_intercept = bool(fit_intercept)

        if not isinstance(compute_inference, (bool, np.bool_)):
            raise ValueError("compute_inference must be boolean")
        self._compute_inference_enabled = bool(compute_inference)

        if not isinstance(gpu_memory_cleanup, (bool, np.bool_)):
            raise ValueError("gpu_memory_cleanup must be boolean")
        self._gpu_memory_cleanup = bool(gpu_memory_cleanup)

        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
            max_iter, Integral
        ) or int(max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        self._max_iter = int(max_iter)

        if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real):
            raise ValueError("tol must be a finite positive number")
        tol_value = float(tol)
        if not np.isfinite(tol_value) or tol_value <= 0.0:
            raise ValueError("tol must be a finite positive number")
        self._tol = tol_value

        self._inference_method = inference_method
        self._n_bootstrap = n_bootstrap

        try:
            self._device = device if isinstance(device, Device) else Device(device)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "device must be one of 'auto', 'cpu', 'cuda', or 'torch'"
            ) from exc

        if (
            isinstance(self._quantile, (bool, np.bool_))
            or not isinstance(self._quantile, Real)
            or not np.isfinite(float(self._quantile))
            or not 0.0 < float(self._quantile) < 1.0
        ):
            raise ValueError("quantile must be a finite real number in (0, 1)")
        if self._compute_inference_enabled:
            if not isinstance(self._inference_method, str) or self._inference_method not in {
                "kernel", "bootstrap"
            }:
                raise ValueError(
                    f"Unknown inference_method='{self._inference_method}'. "
                    "Valid options: ['bootstrap', 'kernel']."
                )
            if self._inference_method == "kernel":
                if not isinstance(self.kernel, str) or self.kernel not in {
                    "epa", "gau", "biw", "cos", "par"
                }:
                    raise ValueError(
                        "kernel must be one of ['epa', 'gau', 'biw', 'cos', 'par']"
                    )
                if not isinstance(self.bandwidth, str) or self.bandwidth not in {
                    "hsheather", "bofinger", "chamberlain"
                }:
                    raise ValueError(
                        "bandwidth must be 'hsheather', 'bofinger', or 'chamberlain'"
                    )
            else:
                if (
                    isinstance(self._n_bootstrap, (bool, np.bool_))
                    or not isinstance(self._n_bootstrap, Integral)
                    or int(self._n_bootstrap) < 2
                ):
                    raise ValueError(
                        "n_bootstrap must be an integer greater than or equal to 2 "
                        "for QuantileRegression bootstrap inference"
                    )
                random_state = self.random_state
                if (
                    random_state is not None
                    and (
                        isinstance(random_state, (bool, np.bool_))
                        or not isinstance(random_state, Integral)
                        or int(random_state) < 0
                    )
                ):
                    raise ValueError(
                        "random_state must be None or a non-negative integer "
                        "for QuantileRegression bootstrap inference"
                    )

    @staticmethod
    def _input_cleanup_targets(*values):
        """Return GPU input backend/device pairs used before fit provenance exists."""
        targets = []
        for value in values:
            if value is None:
                continue
            module = str(type(value).__module__ or "")
            if module.startswith("cupy"):
                device = getattr(value, "device", None)
                device_id = getattr(device, "id", None)
                if device_id is not None:
                    targets.append(("cupy", f"cuda:{int(device_id)}"))
            elif module.startswith("torch"):
                device = getattr(value, "device", None)
                label = str(device or "")
                if label.startswith("cuda"):
                    targets.append(("torch", label))
        # Preserve first-seen order while avoiding duplicate cleanup calls.
        return tuple(dict.fromkeys(targets))


    @staticmethod
    def _has_nonuniform_weight(sample_weight):
        module = type(sample_weight).__module__
        values = sample_weight.reshape(-1)
        if module.startswith("torch"):
            import torch
            uniform = torch.all(values == values[0])
        elif module.startswith("cupy"):
            import cupy as cp
            uniform = cp.all(values == values[0])
        else:
            uniform = np.all(np.asarray(values) == np.asarray(values)[0])
        return not bool(uniform.item() if hasattr(uniform, "item") else uniform)

    def fit(self, X, y, sample_weight=None):
        cleanup_targets = self._input_cleanup_targets(X, y, sample_weight)
        self._reset_fit_state()
        try:
            return self._fit_impl(X, y, sample_weight=sample_weight)
        except Exception:
            if self._gpu_memory_cleanup:
                backend_name = getattr(self, "_selected_backend_name", None)
                backend_device = getattr(self, "_selected_backend_device", None)
                selected = (
                    (backend_name, backend_device)
                    if backend_name is not None
                    else None
                )
                targets = list(cleanup_targets)
                if selected is not None and selected not in targets:
                    targets.append(selected)
                for cleanup_backend, cleanup_device in targets:
                    self._cleanup_backend_memory(
                        cleanup_backend,
                        cleanup_device,
                    )
            self._reset_fit_state()
            raise

    def _fit_impl(self, X, y, sample_weight=None):
        self._validate_public_controls()
        from statgpu.glm_core._validation import (
            validate_glm_design_matrix,
            validate_glm_sample_weight,
        )
        from statgpu.backends import _to_numpy

        X_native = validate_glm_design_matrix(X)
        loss = QuantileLoss(quantile=self._quantile)
        y_native = loss.validate_response(y)
        if int(y_native.shape[0]) != int(X_native.shape[0]):
            raise ValueError("Response length must match the number of X rows.")

        sample_weight_native = None
        if sample_weight is not None:
            sample_weight_native = validate_glm_sample_weight(
                sample_weight, X_native.shape[0]
            )

        if self._compute_inference_enabled:
            if sample_weight_native is not None and self._has_nonuniform_weight(
                sample_weight_native
            ):
                raise NotImplementedError(
                    "Standalone QuantileRegression inference does not support "
                    "non-uniform sample_weight. Fit with compute_inference=False "
                    "or omit/use uniform weights."
                )

        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        self._selected_backend_name = backend_name
        X_arr = self._to_array(X_native, backend=backend_name)
        y_arr = self._to_array(y_native, backend=backend_name)
        sample_weight_arr = (
            None
            if sample_weight_native is None
            else self._to_array(sample_weight_native, backend=backend_name)
        )
        y_arr, sample_weight, backend_device = _align_quantile_fit_inputs(
            X_arr,
            y_arr,
            sample_weight_arr,
            backend_name,
        )
        self._selected_backend_device = backend_device
        n, p = X_arr.shape
        self.n_features_in_ = int(p)

        if self._fit_intercept:
            from statgpu.penalties._l2 import L2Penalty
            from statgpu.backends._utils import _get_xp, xp_ones
            xp = _get_xp(backend_name)
            ones = xp_ones(n, X_arr.dtype, xp, ref_arr=X_arr)
            X_aug = xp.column_stack([X_arr, ones])
            pen = L2Penalty(alpha=0.0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                params, n_iter = fista_solver(
                    loss,
                    pen,
                    X_aug,
                    y_arr,
                    max_iter=self._max_iter,
                    tol=self._tol,
                    sample_weight=sample_weight,
                )
            self._handle_estimation_convergence_warnings(caught)
            self.coef_ = np.asarray(_to_numpy(params[:-1]))
            self.intercept_ = float(_to_numpy(params[-1]))
        else:
            from statgpu.penalties._l2 import L2Penalty
            pen = L2Penalty(alpha=0.0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                params, n_iter = fista_solver(
                    loss,
                    pen,
                    X_arr,
                    y_arr,
                    max_iter=self._max_iter,
                    tol=self._tol,
                    sample_weight=sample_weight,
                )
            self._handle_estimation_convergence_warnings(caught)
            self.coef_ = np.asarray(_to_numpy(params))
            self.intercept_ = 0.0

        self.n_iter_ = n_iter
        if self._fit_intercept:
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self._params = self.coef_.copy()
        if self._compute_inference_enabled:
            self._compute_inference(X_arr, y_arr, loss,
                                     backend_name=backend_name)

        self._fitted = True
        if self._gpu_memory_cleanup:
            self._cleanup_backend_memory(
                backend_name,
                self._selected_backend_device,
            )

        return self

    def _handle_estimation_convergence_warnings(self, caught):
        """Preserve solver warnings while making inference require convergence."""
        convergence_warnings = []
        for item in caught:
            if issubclass(item.category, ConvergenceWarning):
                convergence_warnings.append(item)
                continue
            # catch_warnings(record=True) captures all visible warnings emitted
            # during FISTA, including backend/NumPy RuntimeWarnings. Never hide
            # those merely because convergence ownership is handled here.
            warnings.warn(
                item.message,
                item.category,
                stacklevel=3,
            )

        if not convergence_warnings:
            return

        message = str(convergence_warnings[-1].message)
        if self._compute_inference_enabled:
            raise RuntimeError(
                "QuantileRegression cannot compute inference because the "
                f"point-estimation FISTA solve did not converge: {message}"
            )
        for item in convergence_warnings:
            warnings.warn(
                item.message,
                item.category,
                stacklevel=3,
            )

    def _compute_inference(self, X, y, loss, backend_name="numpy"):
        """Dispatch to kernel-based or bootstrap inference."""
        _valid = {"kernel", "bootstrap"}
        if self._inference_method not in _valid:
            raise ValueError(
                f"Unknown inference_method='{self._inference_method}'. "
                f"Valid options: {sorted(_valid)}."
            )
        if self._inference_method == "bootstrap":
            self._compute_inference_bootstrap(X, y)
        elif backend_name == "numpy":
            self._compute_inference_kernel(X, y)
        else:
            self._compute_inference_kernel_gpu(X, y)

    # ---- Kernel helpers (matching statsmodels) ----
    @staticmethod
    def _get_kernel_fn(name, xp=None):
        """Backend-agnostic kernel function."""
        if xp is None:
            import numpy as _np
            xp = _np
        if name == 'gau':
            return lambda u: xp.exp(-0.5 * u * u) * _INV_SQRT_2PI
        _KERNELS = {
            'epa': lambda u: 0.75 * (1 - u**2) * (xp.abs(u) <= 1),
            'biw': lambda u: 15./16 * (1 - u**2)**2 * (xp.abs(u) <= 1),
            'cos': lambda u: (xp.abs(u) <= 0.5) * (1 + xp.cos(2*xp.pi*u)),
            'par': lambda u: xp.where(xp.abs(u) <= 0.5,
                    4./3 - 8*u**2 + 8*xp.abs(u)**3,
                    xp.where(xp.abs(u) <= 1, 8*(1-xp.abs(u))**3/3., 0)),
        }
        if name not in _KERNELS:
            raise ValueError(f"kernel must be one of {list(_KERNELS.keys())}, got '{name}'")
        return _KERNELS[name]

    @staticmethod
    def _validate_inference_outputs(**arrays):
        """Reject non-finite inference snapshots before estimator publication."""
        for name, value in arrays.items():
            arr = np.asarray(value)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"QuantileRegression inference produced non-finite {name}"
                )
        if "pvalues" in arrays:
            pvalues = np.asarray(arrays["pvalues"], dtype=np.float64)
            if np.any((pvalues < 0.0) | (pvalues > 1.0)):
                raise ValueError(
                    "QuantileRegression inference produced p-values outside [0, 1]"
                )

    @staticmethod
    def _validate_kernel_density_estimate(fhat):
        """Require a finite positive residual density at zero for inference."""
        fhat = float(fhat)
        if not np.isfinite(fhat) or fhat <= 0.0:
            raise ValueError(
                "Quantile kernel inference requires a finite positive residual "
                "density estimate at zero. Try a different kernel/bandwidth, "
                "more observations, or bootstrap inference."
            )
        return fhat

    @staticmethod
    def _get_bandwidth_h_from_scale(n, q, rule, scale):
        """Evaluate the Quantile bandwidth rule from an already-reduced scale."""
        from statgpu.inference._distributions_backend import get_distribution

        _norm = get_distribution("norm", backend="numpy")
        import numpy as _np

        if rule == 'hsheather':
            z = _norm.ppf(q)
            h_base = n**(-1./3) * _norm.ppf(0.975)**(2./3) * (
                1.5 * _norm.pdf(z)**2 / (2*z**2 + 1))**(1./3)
        elif rule == 'bofinger':
            z = _norm.ppf(q)
            h_base = n**(-1./5) * (
                4.5 * _norm.pdf(2*z)**4 / (2*z**2 + 1)**2)**(1./5)
        elif rule == 'chamberlain':
            h_base = _norm.ppf(0.975) * _np.sqrt(q * (1-q) / n)
        else:
            raise ValueError(
                f"bandwidth must be 'hsheather', 'bofinger', or 'chamberlain', got '{rule}'"
            )

        lower_prob = float(q - h_base)
        upper_prob = float(q + h_base)
        if not (0.0 < lower_prob < upper_prob < 1.0):
            raise ValueError(
                "Quantile inference bandwidth is undefined for this quantile/sample-size "
                "combination because q ± h leaves the probability interval (0, 1). "
                "Use a less extreme quantile, more observations, or a different "
                "validated bandwidth rule."
            )
        bandwidth_value = float(scale) * (
            _norm.ppf(upper_prob) - _norm.ppf(lower_prob)
        )
        bandwidth_value = float(bandwidth_value)
        if not np.isfinite(bandwidth_value) or bandwidth_value <= 0.0:
            raise ValueError(
                "Quantile inference bandwidth must be finite and positive; "
                "the response/residual scale is degenerate for kernel inference."
            )
        return bandwidth_value

    @staticmethod
    def _get_bandwidth_h(n, q, rule, resid, y_std):
        import numpy as _np

        iqre = float(_np.percentile(resid, 75) - _np.percentile(resid, 25))
        scale = min(float(y_std), iqre / 1.34)
        return QuantileRegression._get_bandwidth_h_from_scale(
            n,
            q,
            rule,
            scale,
        )

    def _compute_inference_kernel(self, X, y):
        """Kernel-based sandwich covariance (Powell 1991).

        Matches statsmodels ``QuantReg`` with configurable kernel and bandwidth.
        Default: Epanechnikov kernel + Hall-Sheather bandwidth (se='nid').
        """
        from statgpu.inference._distributions_backend import get_distribution
        _norm = get_distribution("norm", backend="numpy")
        import numpy as _np

        if self._fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            X_design = X
            params = self.coef_.copy()

        n, k = X_design.shape
        resid = y - X_design @ params
        tau = self._quantile

        # Bandwidth
        h = self._get_bandwidth_h(n, tau, self.bandwidth, resid, float(np.std(y)))

        # Sparsity via kernel density
        kernel_fn = self._get_kernel_fn(self.kernel)
        u = resid / h
        fhat = self._validate_kernel_density_estimate(
            _np.sum(kernel_fn(u)) / (n * h)
        )
        sparsity = 1.0 / fhat

        # Powell (1991) sandwich covariance
        D = _np.where(resid > 0, (tau / fhat) ** 2, ((1.0 - tau) / fhat) ** 2)

        XtX = X_design.T @ X_design
        try:
            XtX_inv = _np.linalg.solve(XtX, _np.eye(k))
        except _np.linalg.LinAlgError:
            raise _np.linalg.LinAlgError(
                "Quantile regression design matrix is singular — cannot compute "
                "kernel standard errors. This may indicate collinear features. "
                "Consider using inference_method='bootstrap' instead."
            )

        XtDX = X_design.T @ (X_design * D[:, None])
        cov = XtX_inv @ XtDX @ XtX_inv

        bse = _np.sqrt(_np.maximum(_np.diag(cov), 0.0))
        zvalues = params / (bse + 1e-30)
        pvalues = 2.0 * _norm.sf(_np.abs(zvalues))
        z_crit = _norm.ppf(0.975)
        conf_int = _np.column_stack([
            params - z_crit * bse,
            params + z_crit * bse,
        ])
        self._validate_inference_outputs(
            params=params,
            bse=bse,
            statistic=zvalues,
            pvalues=pvalues,
            conf_int=conf_int,
        )

        from statgpu.inference._results import ParameterInferenceResult
        result = ParameterInferenceResult(
            method="kernel",
            params=params.copy(),
            bse=bse.copy(),
            statistic=zvalues.copy(),
            statistic_name="z",
            pvalues=pvalues.copy(),
            conf_int=conf_int.copy(),
            distribution="normal",
            metadata={
                "method": "powell_1991_sandwich",
                "kernel": self.kernel,
                "bandwidth_rule": self.bandwidth,
                "bandwidth": float(h),
                "sparsity": float(sparsity),
                "quantile": tau,
                "numerical_backend": "numpy",
                "numerical_device": "cpu",
                "reporting_backend": "numpy",
            },
        )
        result.apply_to(self)

    def _compute_inference_kernel_gpu(self, X, y):
        """GPU-native kernel-based sandwich covariance (Powell 1991)."""
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_ones, xp_eye, xp_asarray
        from statgpu.backends._array_ops import (
            _clip,
            _linalg_exception_is_rank_failure,
        )
        from statgpu.inference._distributions_backend import get_distribution

        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)
        is_torch = (backend == "torch")
        dev = X.device if is_torch else None
        n = X.shape[0]

        if self._fit_intercept:
            ones = xp_ones((n, 1), X.dtype, xp, ref_arr=X)
            X_design = xp.cat([ones, X], dim=1) if is_torch else xp.column_stack([ones, X])
            inter = xp_asarray([self.intercept_], dtype=X.dtype, xp=xp, ref_arr=X)
            coef = xp_asarray(self.coef_, dtype=X.dtype, xp=xp, ref_arr=X)
            params = xp.concatenate([inter, coef])
        else:
            X_design = X
            params = xp_asarray(self.coef_, dtype=X.dtype, xp=xp, ref_arr=X)

        k = X_design.shape[1]
        resid = (y - X_design @ params).ravel()
        tau = self._quantile

        # Bandwidth rules need only scalar scale statistics. Keep residual
        # quantiles and response dispersion on the executed backend/device.
        q75 = xp.quantile(resid, 0.75)
        q25 = xp.quantile(resid, 0.25)
        iqre = float(q75 - q25)
        # Match NumPy/CuPy population-standard-deviation semantics exactly.
        # torch.std defaults to Bessel correction (unbiased=True), which would
        # otherwise change the bandwidth and therefore the reported inference.
        y_std = (
            float(xp.std(y, unbiased=False))
            if is_torch
            else float(xp.std(y))
        )
        scale = min(y_std, iqre / 1.34)
        h = self._get_bandwidth_h_from_scale(
            n,
            tau,
            self.bandwidth,
            scale,
        )

        # Sparsity
        kernel_fn = self._get_kernel_fn(self.kernel, xp)
        u = resid / h
        fhat = self._validate_kernel_density_estimate(
            float(xp.sum(kernel_fn(u))) / (n * h)
        )
        sparsity = 1.0 / fhat

        # Sandwich covariance
        D = xp.where(resid > 0, (tau / fhat) ** 2, ((1.0 - tau) / fhat) ** 2)
        XtX = X_design.T @ X_design
        try:
            XtX_inv = xp.linalg.solve(
                XtX,
                xp_eye(k, X.dtype, xp, ref_arr=X),
            )
        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            raise np.linalg.LinAlgError(
                "Quantile regression design matrix is singular — cannot compute "
                "kernel standard errors. This may indicate collinear features. "
                "Consider using inference_method='bootstrap' instead."
            ) from exc
        XtDX = X_design.T @ (X_design * D[:, None])
        cov = XtX_inv @ XtDX @ XtX_inv

        cov_diag = xp.diag(cov)
        bse = xp.sqrt(_clip(cov_diag, 0.0, None))
        z_values = params / (bse + 1e-30)
        _norm = get_distribution(
            "norm",
            backend=backend,
            device=str(dev) if is_torch else None,
        )
        pvalues = 2.0 * _norm.sf(xp.abs(z_values))
        q_crit = xp_asarray(
            0.975,
            dtype=X.dtype,
            xp=xp,
            ref_arr=X,
        )
        z_crit = _norm.ppf(q_crit)

        params_np = np.asarray(_to_numpy(params))
        bse_np = np.asarray(_to_numpy(bse))
        zvalues_np = np.asarray(_to_numpy(z_values))
        pvalues_np = np.asarray(_to_numpy(pvalues))
        conf_int_np = np.column_stack([
            np.asarray(_to_numpy(params - z_crit * bse)),
            np.asarray(_to_numpy(params + z_crit * bse)),
        ])
        self._validate_inference_outputs(
            params=params_np,
            bse=bse_np,
            statistic=zvalues_np,
            pvalues=pvalues_np,
            conf_int=conf_int_np,
        )

        from statgpu.inference._results import ParameterInferenceResult
        result = ParameterInferenceResult(
            method="kernel", params=params_np.copy(), bse=bse_np.copy(),
            statistic=zvalues_np.copy(), statistic_name="z",
            pvalues=pvalues_np.copy(), conf_int=conf_int_np.copy(),
            distribution="normal",
            metadata={
                "method": "powell_1991_sandwich",
                "kernel": self.kernel,
                "bandwidth_rule": self.bandwidth,
                "bandwidth": float(h),
                "sparsity": float(sparsity),
                "quantile": tau,
                "backend": backend,
                "numerical_backend": backend,
                "numerical_device": self._selected_backend_device,
                "reporting_backend": "numpy",
            })
        result.apply_to(self)

    def _compute_bootstrap_batched(self, X, y):
        """Solve all residual-bootstrap Quantile refits with batched IRLS/MM.

        Bootstrap responses stay on the fit-recorded numerical backend/device.
        Draws share the same design but own independent IRLS weights and
        weighted normal equations. The draw dimension is chunked to bound
        temporary B*p*p memory while NumPy, CuPy, and Torch execute the same
        float64 numerical algorithm.
        """
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_ones, xp_asarray

        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)
        is_torch = backend == "torch"
        n = int(X.shape[0])
        tau = self._quantile

        if self._fit_intercept:
            ones = xp_ones((n, 1), X.dtype, xp, ref_arr=X)
            Xd = (
                xp.cat([ones, X], dim=1)
                if is_torch
                else xp.column_stack([ones, X])
            )
            inter = xp_asarray(
                [self.intercept_],
                dtype=X.dtype,
                xp=xp,
                ref_arr=X,
            )
            cf = xp_asarray(
                self.coef_,
                dtype=X.dtype,
                xp=xp,
                ref_arr=X,
            )
            params = xp.concatenate([inter, cf])
        else:
            Xd = X
            params = xp_asarray(
                self.coef_,
                dtype=X.dtype,
                xp=xp,
                ref_arr=X,
            )

        eta = Xd @ params
        resid = (y - eta).ravel()
        resid = _center_quantile_bootstrap_residuals(resid, tau, xp)
        self._bootstrap_residual_centering_ = "empirical_tau_quantile"

        B = self._n_bootstrap
        rng = np.random.default_rng(self.random_state)
        schedule = np.stack(
            [
                rng.integers(0, n, size=n, dtype=np.int64)
                for _ in range(B)
            ],
            axis=0,
        )
        normalized_schedule = np.ascontiguousarray(schedule, dtype="<i8")
        self._bootstrap_schedule_sha256_ = hashlib.sha256(
            normalized_schedule.view(np.uint8)
        ).hexdigest()

        # Only the deterministic integer resampling schedule lives on the CPU
        # control plane. Residual gathering and response construction stay on
        # the concrete numerical backend/device.
        schedule_native = _bootstrap_schedule_to_backend(
            schedule,
            resid,
            backend,
            xp,
        )
        y_boot = eta[None, :] + resid[schedule_native]
        y_matrix = y_boot.T

        try:
            boot_coef, used_iter, chunk_size = _batched_quantile_irls(
                Xd,
                y_matrix,
                params,
                tau,
                max_iter=self._max_iter,
                tol=self._tol,
                xp=xp,
            )
        except _BootstrapIRLSConvergenceError as exc:
            self._bootstrap_n_iter_ = int(exc.n_iter)
            self._bootstrap_draw_chunk_size_ = int(exc.chunk_size)
            raise
        self._bootstrap_n_iter_ = int(used_iter)
        self._bootstrap_draw_chunk_size_ = int(chunk_size)

        return np.asarray(_to_numpy(boot_coef.T)), params, Xd

    def _compute_inference_bootstrap(self, X, y):
        """I.i.d. residual-bootstrap inference for quantile regression.

        This procedure centers fitted residuals at their empirical tau-quantile
        and resamples the centered residuals as exchangeable draws. The
        centering keeps the bootstrap error distribution's tau-quantile at
        zero, including for no-intercept fits. It is not the wild/multiplier
        bootstrap used for general heteroscedastic quantile-regression inference.

        Uses batched Quantile IRLS/MM for all backends (CPU/GPU). Draws share
        the design but own independent IRLS weights and weighted normal
        equations; draw chunks bound temporary Gram-matrix memory while keeping
        the numerical work backend-native.

        Inference outputs:
        - Standard errors: bootstrap std (ddof=1)
        - p-values: two-sided bootstrap sign-test
        - Confidence intervals: percentile bootstrap (2.5%, 97.5%)

        .. note::
           Bootstrap child refits use the Quantile-specific IRLS/MM objective,
           not the generic smooth-gradient FISTA route. This avoids treating the
           pinball loss's set-valued kink subgradient as a smooth gradient.
        """
        if self._fit_intercept:
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            params = self.coef_.copy()

        boot_params, _, _ = self._compute_bootstrap_batched(X, y)
        boot_params = np.asarray(boot_params)
        bse = np.std(boot_params, axis=0, ddof=1)
        zvalues = params / (bse + 1e-30)
        pvalues = np.array([
            min(
                2.0 * min(
                    np.mean(boot_params[:, i] <= 0.0),
                    np.mean(boot_params[:, i] >= 0.0),
                ),
                1.0,
            )
            for i in range(len(params))
        ])
        conf_int = np.column_stack([
            np.quantile(boot_params, 0.025, axis=0),
            np.quantile(boot_params, 0.975, axis=0),
        ])
        self._validate_inference_outputs(
            params=params,
            boot_params=boot_params,
            bse=bse,
            statistic=zvalues,
            pvalues=pvalues,
            conf_int=conf_int,
        )

        from statgpu.inference._results import ParameterInferenceResult
        result = ParameterInferenceResult(
            method="bootstrap", params=params.copy(), bse=bse.copy(),
            statistic=zvalues.copy(),
            statistic_name="estimate_over_bootstrap_se",
            pvalues=pvalues.copy(), conf_int=conf_int.copy(),
            distribution="bootstrap_percentile",
            metadata={
                "n_bootstrap": self._n_bootstrap,
                "bootstrap_type": "iid_residual",
                "residual_centering": "empirical_tau_quantile",
                "heteroscedastic_robust": False,
                "ci_method": "percentile",
                "pvalue_method": "bootstrap_sign_test",
                "statistic_method": "estimate_over_bootstrap_se",
                "resampling_schedule": "numpy_generator_control_plane",
                "resampling_schedule_sha256": self._bootstrap_schedule_sha256_,
                "random_state": (
                    int(self.random_state)
                    if isinstance(self.random_state, (int, np.integer))
                    and not isinstance(self.random_state, (bool, np.bool_))
                    else repr(self.random_state)
                ),
                "response_construction": "backend_native",
                "solver": "batched_quantile_irls",
                "solver_n_iter": int(self._bootstrap_n_iter_),
                "draw_chunk_size": int(self._bootstrap_draw_chunk_size_),
                "backend": getattr(self, "_selected_backend_name", "numpy"),
                "numerical_backend": getattr(
                    self, "_selected_backend_name", "numpy"
                ),
                "numerical_device": self._selected_backend_device,
                "reporting_backend": "numpy",
            })
        result.apply_to(self)
        # Backward-compatible aliases: these are estimate/SE ratios only.
        # Bootstrap p-values are sign-test p-values, not normal z-tail p-values.
        self._zvalues = zvalues.copy()
        self._tvalues = zvalues.copy()

    def _prediction_array_on_fit_device(self, X, backend_name):
        """Normalize prediction design to float64 on the recorded fit device."""
        if backend_name == "cupy":
            import cupy as cp
            from statgpu.backends._utils import _cupy_asarray_on_device

            selected = str(
                getattr(self, "_selected_backend_device", "") or ""
            ).lower()
            if selected.startswith("cuda:"):
                device_id = int(selected.split(":", 1)[1])
                with cp.cuda.Device(device_id):
                    converted = self._to_array(
                        X,
                        Device.CUDA,
                        backend="cupy",
                    )
                    return _cupy_asarray_on_device(
                        converted,
                        device_id,
                        dtype=cp.float64,
                    )
            converted = self._to_array(
                X,
                Device.CUDA,
                backend="cupy",
            )
            return cp.asarray(converted, dtype=cp.float64)

        if backend_name == "torch":
            import torch

            selected = str(
                getattr(self, "_selected_backend_device", "") or ""
            ).lower()
            target = selected if selected.startswith("cuda:") else "cuda"
            return self._to_torch(X, device=target).to(
                device=target,
                dtype=torch.float64,
            )

        return np.asarray(X, dtype=np.float64)

    def predict(self, X):
        self._check_is_fitted()
        from statgpu.glm_core._validation import validate_glm_design_matrix

        X_native = validate_glm_design_matrix(X)
        if int(X_native.shape[1]) != int(np.asarray(self.coef_).size):
            raise ValueError(
                "X must have the same number of features as the fitted QuantileRegression"
            )
        backend_name = self._selected_backend_name or "numpy"
        X_arr = self._prediction_array_on_fit_device(X_native, backend_name)
        from statgpu.backends._utils import _get_xp, xp_asarray
        xp = _get_xp(backend_name)
        coef = xp_asarray(
            self.coef_,
            dtype=X_arr.dtype,
            xp=xp,
            ref_arr=X_arr,
        )
        intercept = xp_asarray(
            self.intercept_,
            dtype=X_arr.dtype,
            xp=xp,
            ref_arr=X_arr,
        )
        raw = X_arr @ coef + intercept
        from statgpu.backends import _to_numpy
        result = np.asarray(_to_numpy(raw)) if backend_name != "numpy" else raw
        if self._gpu_memory_cleanup:
            self._cleanup_backend_memory(
                backend_name,
                self._selected_backend_device,
            )
        return result

    def score(self, X, y, sample_weight=None):
        """Return negative pinball loss on test data; higher is better."""
        self._check_is_fitted()
        from statgpu.backends import _to_numpy
        from statgpu.glm_core._validation import validate_glm_sample_weight

        y_np = np.asarray(_to_numpy(y))
        if y_np.ndim != 1:
            raise ValueError("y must be one-dimensional for QuantileRegression score")
        if y_np.dtype.kind not in "biuf":
            raise ValueError(
                "y must contain real numeric values for QuantileRegression score"
            )
        if sample_weight is not None:
            sw = validate_glm_sample_weight(sample_weight, y_np.shape[0])
            sw = np.asarray(_to_numpy(sw), dtype=np.float64)
        else:
            sw = None

        pred = np.asarray(self.predict(X), dtype=np.float64)
        if y_np.shape[0] != pred.shape[0]:
            raise ValueError(
                "y must have the same number of observations as X for "
                "QuantileRegression score"
            )
        residual = y_np - pred
        tau = float(self._quantile)
        per_sample = np.where(
            residual >= 0.0,
            tau * residual,
            (tau - 1.0) * residual,
        )
        if sw is None:
            loss = float(np.mean(per_sample))
        else:
            loss = float(np.average(per_sample, weights=sw))
        return -loss

    # ---- GPU memory management ----

    def _cleanup_cuda_memory(self, device_label=None):
        if not self._gpu_memory_cleanup:
            return
        try:
            import cupy as cp

            selected = str(
                device_label
                or getattr(self, "_selected_backend_device", "")
                or ""
            )
            if selected.startswith("cuda:"):
                device_id = int(selected.split(":", 1)[1])
                with cp.cuda.Device(device_id):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
            else:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self, device_label=None):
        if not self._gpu_memory_cleanup:
            return
        try:
            import torch

            selected = str(
                device_label
                or getattr(self, "_selected_backend_device", "")
                or ""
            )
            if selected.startswith("cuda:"):
                target = torch.device(selected)
                with torch.cuda.device(target):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(target)
            elif selected in ("", "cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def _cleanup_backend_memory(self, backend_name, device_label=None):
        if backend_name in ("cuda", "cupy"):
            self._cleanup_cuda_memory(device_label)
        elif backend_name == "torch":
            self._cleanup_torch_memory(device_label)

    def __del__(self):
        try:
            backend_name = getattr(self, "_selected_backend_name", None)
            if backend_name is not None:
                self._cleanup_backend_memory(
                    backend_name,
                    getattr(self, "_selected_backend_device", None),
                )
        except Exception:
            pass

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def summary(self):
        if not self._fitted:
            return f"{self.__class__.__name__}(not fitted)"
        lines = [
            f"{'='*60}",
            f"  QuantileRegression (τ={self._quantile})",
            f"{'='*60}",
        ]
        if self._inference_result is not None:
            try:
                df = self._inference_result.to_dataframe()
                lines.append(str(df.to_string(index=False)))
            except Exception:
                lines.append(f"  coef: {self._params}")
                if self._bse is not None:
                    method = str(getattr(self._inference_result, "method", "inference"))
                    lines.append(f"  std err ({method}): {self._bse}")
        else:
            lines.append(f"  coef: {self._params}")
            lines.append("  (inference not computed)")
        lines.append(f"{'='*60}")
        return "\n".join(lines)
