"""Quantile regression with bootstrap inference support."""

import math as _math
from numbers import Integral, Real
from typing import Optional
import numpy as np

# Pre-computed scalar constants (Python floats, safe for GPU tensor broadcast)
_INV_SQRT_2PI = 1.0 / _math.sqrt(2.0 * _math.pi)


def _pinball_eta_gradient_values(tau):
    """Return d rho_tau(y-eta) / d eta on nonnegative/negative residuals."""
    tau = float(tau)
    return -tau, 1.0 - tau

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.losses._quantile import QuantileLoss
from statgpu.solvers import fista_solver


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
        'bootstrap': residual bootstrap (batched FISTA) with percentile CI,
            bootstrap sign-test p-values, and bootstrap std errors.  All
            backends (CPU/GPU) use the same batched solver for consistency.
    kernel : str, default='epa'
        Kernel for sparsity estimation: 'epa' (Epanechnikov), 'gau' (Gaussian),
        'biw' (Biweight), 'cos' (Cosine), 'par' (Parzen).  Only for
        inference_method='kernel'.
    bandwidth : str, default='hsheather'
        Bandwidth rule: 'hsheather' (Hall-Sheather), 'bofinger' (Bofinger),
        'chamberlain' (Chamberlain).  Only for inference_method='kernel'.
    n_bootstrap : int, default=200
        Number of bootstrap resamples; bootstrap inference requires at least 2.
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
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._fitted = False

    def _reset_fit_state(self):
        """Clear all result-bearing state after a failed or new fit attempt."""
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = None
        self._params = None
        self._bse = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None
        self._fitted = False
        self._selected_backend_name = None
        self.__dict__.pop("n_features_in_", None)

    def _validate_public_controls(self):
        """Validate mutable public controls before backend or numerical work."""
        if (
            isinstance(self._quantile, (bool, np.bool_))
            or not isinstance(self._quantile, Real)
            or not np.isfinite(float(self._quantile))
            or not 0.0 < float(self._quantile) < 1.0
        ):
            raise ValueError("quantile must be a finite real number in (0, 1)")
        if not isinstance(self._fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        if not isinstance(self._compute_inference_enabled, (bool, np.bool_)):
            raise ValueError("compute_inference must be boolean")
        if not isinstance(self._gpu_memory_cleanup, (bool, np.bool_)):
            raise ValueError("gpu_memory_cleanup must be boolean")

        if isinstance(self._max_iter, (bool, np.bool_)) or not isinstance(
            self._max_iter, Integral
        ) or int(self._max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(self._tol, (bool, np.bool_)) or not isinstance(self._tol, Real):
            raise ValueError("tol must be a finite positive number")
        tol = float(self._tol)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be a finite positive number")

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
        self._reset_fit_state()
        try:
            return self._fit_impl(X, y, sample_weight=sample_weight)
        except Exception:
            backend_name = getattr(self, "_selected_backend_name", None)
            if self._gpu_memory_cleanup and backend_name is not None:
                self._cleanup_backend_memory(backend_name)
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
        n, p = X_arr.shape
        self.n_features_in_ = int(p)
        sample_weight = sample_weight_native

        if self._fit_intercept:
            from statgpu.penalties._l2 import L2Penalty
            from statgpu.backends._utils import _get_xp, xp_ones
            xp = _get_xp(backend_name)
            ones = xp_ones(n, X_arr.dtype, xp, ref_arr=X_arr)
            X_aug = xp.column_stack([X_arr, ones])
            pen = L2Penalty(alpha=0.0)
            params, n_iter = fista_solver(loss, pen, X_aug, y_arr,
                                          max_iter=self._max_iter, tol=self._tol,
                                          sample_weight=sample_weight)
            self.coef_ = np.asarray(_to_numpy(params[:-1]))
            self.intercept_ = float(_to_numpy(params[-1]))
        else:
            from statgpu.penalties._l2 import L2Penalty
            pen = L2Penalty(alpha=0.0)
            params, n_iter = fista_solver(loss, pen, X_arr, y_arr,
                                          max_iter=self._max_iter, tol=self._tol,
                                          sample_weight=sample_weight)
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
            self._cleanup_backend_memory(backend_name)

        return self

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
    def _get_bandwidth_h(n, q, rule, resid, y_std):
        from statgpu.inference._distributions_backend import get_distribution
        _norm = get_distribution("norm", backend="numpy")
        import numpy as _np
        iqre = float(_np.percentile(resid, 75) - _np.percentile(resid, 25))
        scale = min(y_std, iqre / 1.34)

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
            raise ValueError(f"bandwidth must be 'hsheather', 'bofinger', or 'chamberlain', got '{rule}'")

        lower_prob = float(q - h_base)
        upper_prob = float(q + h_base)
        if not (0.0 < lower_prob < upper_prob < 1.0):
            raise ValueError(
                "Quantile inference bandwidth is undefined for this quantile/sample-size "
                "combination because q ± h leaves the probability interval (0, 1). "
                "Use a less extreme quantile, more observations, or a different "
                "validated bandwidth rule."
            )
        bandwidth_value = scale * (
            _norm.ppf(upper_prob) - _norm.ppf(lower_prob)
        )
        bandwidth_value = float(bandwidth_value)
        if not np.isfinite(bandwidth_value) or bandwidth_value <= 0.0:
            raise ValueError(
                "Quantile inference bandwidth must be finite and positive; "
                "the response/residual scale is degenerate for kernel inference."
            )
        return bandwidth_value

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

        self._bse = _np.sqrt(_np.maximum(_np.diag(cov), 0.0))
        self._zvalues = params / (self._bse + 1e-30)
        self._pvalues = 2.0 * _norm.sf(_np.abs(self._zvalues))
        z_crit = _norm.ppf(0.975)
        self._conf_int = _np.column_stack([
            params - z_crit * self._bse,
            params + z_crit * self._bse,
        ])

        from statgpu.inference._results import ParameterInferenceResult
        self._inference_result = ParameterInferenceResult(
            method="kernel",
            params=params.copy(),
            bse=self._bse.copy(),
            statistic=self._zvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="normal",
            metadata={
                "method": "powell_1991_sandwich",
                "kernel": self.kernel,
                "bandwidth_rule": self.bandwidth,
                "bandwidth": float(h),
                "sparsity": float(sparsity),
                "quantile": tau,
            },
        )
        self._inference_result.apply_to(self)

    def _compute_inference_kernel_gpu(self, X, y):
        """GPU-native kernel-based sandwich covariance (Powell 1991)."""
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_ones, xp_eye, xp_asarray
        from statgpu.backends._array_ops import _clip
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

        # Bandwidth (scipy operates on CPU scalars only)
        resid_cpu = np.asarray(_to_numpy(resid)).ravel()
        y_std = float(xp.std(y))
        h = self._get_bandwidth_h(n, tau, self.bandwidth, resid_cpu, y_std)

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
        XtX_inv = xp.linalg.solve(XtX, xp_eye(k, X.dtype, xp, ref_arr=X))
        XtDX = X_design.T @ (X_design * D[:, None])
        cov = XtX_inv @ XtDX @ XtX_inv

        cov_diag = xp.diag(cov)
        bse = xp.sqrt(_clip(cov_diag, 0.0, None))
        z_values = params / (bse + 1e-30)
        _norm = get_distribution("norm", backend=backend)
        pvalues = 2.0 * _norm.sf(xp.abs(z_values))
        z_crit = _norm.ppf(0.975)

        self._bse = np.asarray(_to_numpy(bse))
        self._zvalues = np.asarray(_to_numpy(z_values))
        self._pvalues = np.asarray(_to_numpy(pvalues))
        self._conf_int = np.column_stack([
            np.asarray(_to_numpy(params - z_crit * bse)),
            np.asarray(_to_numpy(params + z_crit * bse))])
        self._params = np.asarray(_to_numpy(params))

        from statgpu.inference._results import ParameterInferenceResult
        self._inference_result = ParameterInferenceResult(
            method="kernel", params=self._params.copy(), bse=self._bse.copy(),
            statistic=self._zvalues.copy(), statistic_name="z",
            pvalues=self._pvalues.copy(), conf_int=self._conf_int.copy(),
            distribution="normal",
            metadata={"method": "powell_1991_sandwich", "kernel": self.kernel,
                       "bandwidth_rule": self.bandwidth, "bandwidth": float(h),
                       "sparsity": float(sparsity), "quantile": tau, "backend": backend})
        self._inference_result.apply_to(self)

    def _compute_bootstrap_batched(self, X, y):
        """Batched pinball FISTA — solves all B bootstrap samples in parallel.

        Works on all backends (CPU/GPU).  The pinball loss is convex but not
        strictly convex; different backends may produce different but equally
        valid solutions that minimize the same objective.
        """
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp, xp_ones, xp_zeros, xp_asarray
        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)
        is_torch = (backend == "torch")
        n = X.shape[0]; tau = self._quantile; p = X.shape[1]

        if self._fit_intercept:
            ones = xp_ones((n, 1), X.dtype, xp, ref_arr=X)
            Xd = xp.cat([ones, X], dim=1) if is_torch else xp.column_stack([ones, X])
            inter = xp_asarray([self.intercept_], dtype=X.dtype, xp=xp, ref_arr=X)
            cf = xp_asarray(self.coef_, dtype=X.dtype, xp=xp, ref_arr=X)
            params = xp.concatenate([inter, cf])
        else:
            Xd = X; p = X.shape[1]
            params = xp_asarray(self.coef_, dtype=X.dtype, xp=xp, ref_arr=X)
        p = Xd.shape[1]

        eta = Xd @ params
        resid_cpu = np.asarray(_to_numpy((y - eta).ravel()))
        eta_cpu = np.asarray(_to_numpy(eta))
        B = self._n_bootstrap
        rng = np.random.default_rng(self.random_state)
        y_batch = np.array([eta_cpu + resid_cpu[rng.integers(0, n, size=n)] for _ in range(B)])
        y_gpu = xp_asarray(y_batch, dtype=X.dtype, xp=xp, ref_arr=X)

        # Lipschitz constant + backtracking line search
        L0 = max(float(xp.linalg.norm(Xd, ord=2)) ** 2 / n, 1e-10)
        coef = xp_zeros((p, B), X.dtype, xp, ref_arr=X)
        z = coef.clone() if is_torch else coef.copy()
        c1 = 1e-4
        t_iter = 1.0
        is_cupy = (not is_torch and hasattr(xp, 'fuse'))

        # CuPy: pre-allocate scratch arrays to avoid allocation in hot loop
        if is_cupy:
            _d_eta_buf = xp.empty_like(y_gpu.T)
            _loss_buf = xp.empty_like(y_gpu.T)
            grad_nonnegative, grad_negative = _pinball_eta_gradient_values(tau)
            @xp.fuse()
            def _pinball_grad_kernel(_r, _out):
                _out[:] = xp.where(
                    _r >= 0, float(grad_nonnegative), float(grad_negative)
                )
            @xp.fuse()
            def _pinball_loss_kernel(_r, _out):
                _out[:] = xp.where(_r > 0, float(tau) * _r, float(tau - 1.0) * _r)

        for iteration in range(self._max_iter):
            # ---- Gradient (all backends) ----
            pred_z = Xd @ z
            r_z = y_gpu.T - pred_z  # (n, B)

            # Element-wise pinball gradient
            if is_cupy:
                _pinball_grad_kernel(r_z, _d_eta_buf)
                d_eta = _d_eta_buf
            else:
                grad_nonnegative, grad_negative = _pinball_eta_gradient_values(tau)
                d_eta = xp.where(
                    r_z >= 0, float(grad_nonnegative), float(grad_negative)
                )
                if is_torch: d_eta = d_eta.to(Xd.dtype)

            grad = Xd.T @ d_eta / n

            # ---- Convergence check ----
            if float(xp.max(xp.abs(grad))) < self._tol:
                break

            # ---- Backtracking line search ----
            step = 1.0 / L0
            # Compute loss once; reuse for Armijo checks
            if is_cupy:
                _pinball_loss_kernel(r_z, _loss_buf)
                loss_z = xp.sum(_loss_buf) / n
            else:
                loss_z = xp.sum(xp.where(r_z > 0, tau * r_z, (tau - 1.0) * r_z)) / n
            grad_norm_sq = xp.sum(grad * grad)

            for _ in range(10):
                coef_new = z - step * grad
                pred_new = Xd @ coef_new
                r_new = y_gpu.T - pred_new
                if is_cupy:
                    _pinball_loss_kernel(r_new, _loss_buf)
                    loss_new = xp.sum(_loss_buf) / n
                else:
                    loss_new = xp.sum(xp.where(r_new > 0, tau * r_new, (tau - 1.0) * r_new)) / n
                if float(loss_new - loss_z + c1 * step * grad_norm_sq) <= 0:
                    break
                step *= 0.5

            # ---- FISTA momentum update ----
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_iter * t_iter) ** 0.5)
            z = coef_new + ((t_iter - 1.0) / t_new) * (coef_new - coef)
            coef = coef_new; t_iter = t_new

        return np.asarray(_to_numpy(coef.T)), params, Xd

    def _compute_inference_bootstrap(self, X, y):
        """Residual bootstrap inference for quantile regression.

        Uses batched pinball FISTA for all backends (CPU/GPU), solving all B
        bootstrap samples in parallel via a single ``(p, B)`` coefficient matrix.
        This gives numerically identical results across backends (same rng seed).

        Inference outputs:
        - Standard errors: bootstrap std (ddof=1)
        - p-values: two-sided bootstrap sign-test
        - Confidence intervals: percentile bootstrap (2.5%, 97.5%)

        .. note::
           Batched FISTA does NOT call ``fista_solver()`` because the general
           solver API operates on ``(p,)`` coefficients.  The batched variant
           works on ``(p, B)`` coefficients for parallel solves and includes
           its own backtracking line search with Armijo condition.
        """
        if self._fit_intercept:
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            params = self.coef_.copy()

        boot_params, _, _ = self._compute_bootstrap_batched(X, y)
        boot_params = np.asarray(boot_params)
        self._bse = np.std(boot_params, axis=0, ddof=1)
        self._zvalues = params / (self._bse + 1e-30)
        pvalues = np.array([min(2.0 * min(np.mean(boot_params[:, i] <= 0.0),
                                           np.mean(boot_params[:, i] >= 0.0)), 1.0)
                            for i in range(len(params))])
        self._pvalues = pvalues
        self._conf_int = np.column_stack([
            np.quantile(boot_params, 0.025, axis=0),
            np.quantile(boot_params, 0.975, axis=0)])

        from statgpu.inference._results import ParameterInferenceResult
        self._inference_result = ParameterInferenceResult(
            method="bootstrap", params=params.copy(), bse=self._bse.copy(),
            statistic=self._zvalues.copy(), statistic_name="z",
            pvalues=self._pvalues.copy(), conf_int=self._conf_int.copy(),
            distribution="bootstrap_percentile",
            metadata={
                "n_bootstrap": self._n_bootstrap,
                "ci_method": "percentile",
                "pvalue_method": "bootstrap_sign_test",
                "solver": "batched_pinball_fista",
                "backend": getattr(self, '_selected_backend_name', 'numpy'),
            })
        self._inference_result.apply_to(self)

    def predict(self, X):
        self._check_is_fitted()
        from statgpu.glm_core._validation import validate_glm_design_matrix

        X_native = validate_glm_design_matrix(X)
        if int(X_native.shape[1]) != int(np.asarray(self.coef_).size):
            raise ValueError(
                "X must have the same number of features as the fitted QuantileRegression"
            )
        backend_name = self._selected_backend_name or "numpy"
        X_arr = self._to_array(X_native, backend=backend_name)
        from statgpu.backends._utils import _get_xp, xp_asarray
        xp = _get_xp(backend_name)
        coef = xp_asarray(self.coef_, xp=xp, ref_arr=X_arr)
        intercept = xp_asarray(self.intercept_, xp=xp, ref_arr=X_arr)
        raw = X_arr @ coef + intercept
        from statgpu.backends import _to_numpy
        result = np.asarray(_to_numpy(raw)) if backend_name != "numpy" else raw
        if self._gpu_memory_cleanup:
            self._cleanup_backend_memory(backend_name)
        return result

    def score(self, X, y, sample_weight=None):
        """Return negative pinball loss on test data; higher is better."""
        self._check_is_fitted()
        from statgpu.backends import _to_numpy
        from statgpu.glm_core._validation import validate_glm_sample_weight

        y_np = np.asarray(_to_numpy(y))
        if y_np.ndim != 1:
            raise ValueError("y must be one-dimensional for QuantileRegression score")
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

    def _cleanup_cuda_memory(self):
        if not self._gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        if not self._gpu_memory_cleanup:
            return
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    def _cleanup_backend_memory(self, backend_name):
        if backend_name in ("cuda", "cupy"):
            self._cleanup_cuda_memory()
        elif backend_name == "torch":
            self._cleanup_torch_memory()

    def __del__(self):
        try:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()
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
