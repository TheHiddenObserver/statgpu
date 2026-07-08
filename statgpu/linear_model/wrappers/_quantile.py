"""Quantile regression with bootstrap inference support."""

import math as _math
from typing import Optional
import numpy as np

# Pre-computed scalar constants (Python floats, safe for GPU tensor broadcast)
_INV_SQRT_2PI = 1.0 / _math.sqrt(2.0 * _math.pi)

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
        'bootstrap': residual bootstrap with percentile CI.
    kernel : str, default='epa'
        Kernel for sparsity estimation: 'epa' (Epanechnikov), 'gau' (Gaussian),
        'biw' (Biweight), 'cos' (Cosine), 'par' (Parzen).  Only for
        inference_method='kernel'.
    bandwidth : str, default='hsheather'
        Bandwidth rule: 'hsheather' (Hall-Sheather), 'bofinger' (Bofinger),
        'chamberlain' (Chamberlain).  Only for inference_method='kernel'.
    n_bootstrap : int, default=200
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
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
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

    def fit(self, X, y, sample_weight=None):
        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        from statgpu.backends import _to_numpy

        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)
        n, p = X_arr.shape

        loss = QuantileLoss(quantile=self.quantile)

        if self.fit_intercept:
            from statgpu.penalties._l2 import L2Penalty
            from statgpu.backends._utils import _get_xp, xp_ones
            xp = _get_xp(backend_name)
            ones = xp_ones(n, X_arr.dtype, xp, ref_arr=X_arr)
            X_aug = xp.column_stack([X_arr, ones])
            pen = L2Penalty(alpha=0.0)
            params, n_iter = fista_solver(loss, pen, X_aug, y_arr,
                                          max_iter=self.max_iter, tol=self.tol,
                                          sample_weight=sample_weight)
            self.coef_ = np.asarray(_to_numpy(params[:-1]))
            self.intercept_ = float(_to_numpy(params[-1]))
        else:
            from statgpu.penalties._l2 import L2Penalty
            pen = L2Penalty(alpha=0.0)
            params, n_iter = fista_solver(loss, pen, X_arr, y_arr,
                                          max_iter=self.max_iter, tol=self.tol,
                                          sample_weight=sample_weight)
            self.coef_ = np.asarray(_to_numpy(params))
            self.intercept_ = 0.0

        self.n_iter_ = n_iter
        if self.fit_intercept:
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self._params = self.coef_.copy()
        self._selected_backend_name = backend_name
        self._fitted = True

        if self.compute_inference:
            self._compute_inference(X_arr, y_arr, loss,
                                     backend_name=backend_name)

        return self

    def _compute_inference(self, X, y, loss, backend_name="numpy"):
        """Dispatch to kernel-based or bootstrap inference."""
        _valid = {"kernel", "bootstrap"}
        if self.inference_method not in _valid:
            raise ValueError(
                f"Unknown inference_method='{self.inference_method}'. "
                f"Valid options: {sorted(_valid)}."
            )
        if self.inference_method == "bootstrap":
            self._compute_inference_bootstrap(X, y)
        else:
            self._compute_inference_kernel_gpu(X, y) if backend_name != "numpy" else self._compute_inference_kernel(X, y)

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
    def _get_bandwidth_h(n, q, rule, resid, y_std):
        from scipy.stats import norm as _norm, scoreatpercentile
        import numpy as _np
        iqre = float(scoreatpercentile(resid, 75) - scoreatpercentile(resid, 25))
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

        return scale * (_norm.ppf(q + h_base) - _norm.ppf(q - h_base))

    def _compute_inference_kernel(self, X, y):
        """Kernel-based sandwich covariance (Powell 1991).

        Matches statsmodels ``QuantReg`` with configurable kernel and bandwidth.
        Default: Epanechnikov kernel + Hall-Sheather bandwidth (se='nid').
        """
        from scipy.stats import norm as _norm
        import numpy as _np

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X.shape[0]), X])
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            X_design = X
            params = self.coef_.copy()

        n, k = X_design.shape
        resid = y - X_design @ params
        tau = self.quantile

        # Bandwidth
        h = self._get_bandwidth_h(n, tau, self.bandwidth, resid, float(np.std(y)))

        # Sparsity via kernel density
        kernel_fn = self._get_kernel_fn(self.kernel)
        u = resid / h
        fhat = _np.sum(kernel_fn(u)) / (n * h)
        sparsity = 1.0 / max(fhat, 1e-10)

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
        self._pvalues = 2.0 * (1.0 - _norm.cdf(_np.abs(self._zvalues)))
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
        from statgpu.backends._utils import _get_xp
        from statgpu.inference._distributions_backend import get_distribution

        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)
        is_torch = (backend == "torch")
        dev = X.device if is_torch else None
        n = X.shape[0]

        if self.fit_intercept:
            ones = xp.ones((n, 1), dtype=X.dtype) if not is_torch else xp.ones((n,1), dtype=X.dtype, device=dev)
            X_design = xp.cat([ones, X], dim=1) if is_torch else xp.column_stack([ones, X])
            inter = xp.asarray([self.intercept_], dtype=X.dtype)
            coef = xp.asarray(self.coef_, dtype=X.dtype)
            if is_torch:
                inter = inter.to(dev); coef = coef.to(dev)
            params = xp.concatenate([inter, coef])
        else:
            X_design = X
            params = xp.asarray(self.coef_, dtype=X.dtype)

        k = X_design.shape[1]
        resid = (y - X_design @ params).ravel()
        tau = self.quantile

        # Bandwidth (scipy operates on CPU scalars only)
        resid_cpu = np.asarray(_to_numpy(resid)).ravel()
        h = self._get_bandwidth_h(n, tau, self.bandwidth, resid_cpu,
                                   float(np.std(_to_numpy(y))))

        # Sparsity
        kernel_fn = self._get_kernel_fn(self.kernel, xp)
        u = resid / h
        fhat = float(xp.sum(kernel_fn(u))) / (n * h)
        sparsity = 1.0 / max(fhat, 1e-10)

        # Sandwich covariance
        D = xp.where(resid > 0, (tau / fhat) ** 2, ((1.0 - tau) / fhat) ** 2)
        XtX = X_design.T @ X_design
        eye = xp.eye(k, dtype=X.dtype) if not is_torch else xp.eye(k, dtype=X.dtype, device=dev)
        XtX_inv = xp.linalg.solve(XtX, eye)
        XtDX = X_design.T @ (X_design * D[:, None])
        cov = XtX_inv @ XtDX @ XtX_inv

        cov_diag = xp.diag(cov)
        bse = xp.sqrt(xp.clamp(cov_diag, min=0.0) if is_torch else xp.maximum(cov_diag, 0.0))
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

    def _compute_inference_bootstrap_batched(self, X, y):
        """Batched pinball FISTA bootstrap for GPU — correct subgradient."""
        from statgpu.backends import _to_numpy, _resolve_backend
        from statgpu.backends._utils import _get_xp
        backend = _resolve_backend("auto", X)
        xp = _get_xp(backend)
        is_torch = (backend == "torch")
        n = X.shape[0]; tau = self.quantile

        if self.fit_intercept:
            ones = xp.ones((n, 1), dtype=X.dtype)
            if is_torch: ones = xp.ones((n, 1), dtype=X.dtype, device=X.device)
            X_design = xp.cat([ones, X], dim=1) if is_torch else xp.column_stack([ones, X])
            inter = xp.asarray([self.intercept_], dtype=X.dtype)
            coef_arr = xp.asarray(self.coef_, dtype=X.dtype)
            if is_torch: inter = inter.to(X.device); coef_arr = coef_arr.to(X.device)
            params = xp.concatenate([inter, coef_arr])
        else:
            X_design = X
            params = xp.asarray(self.coef_, dtype=X.dtype)

        eta = X_design @ params
        resid = (y - eta).ravel()
        p = X_design.shape[1]
        B = self.n_bootstrap

        # Generate all B bootstrap y samples at once on CPU
        eta_cpu = np.asarray(_to_numpy(eta))
        resid_cpu = np.asarray(_to_numpy(resid))
        rng = np.random.default_rng(self.random_state)
        y_batch = np.array([eta_cpu + resid_cpu[rng.integers(0, n, size=n)]
                            for _ in range(B)])

        # Convert to GPU
        y_gpu = xp.asarray(y_batch, dtype=X.dtype)
        if is_torch: y_gpu = y_gpu.to(X.device)

        # Batched pinball FISTA: correct subgradient, no proximal op needed
        L = max(float(xp.linalg.norm(X_design, ord=2)) ** 2 / n, 1e-10)
        step = 1.0 / L
        coef = xp.zeros((p, B), dtype=X.dtype)  # (p, B) for efficient GEMM
        if is_torch: coef = coef.to(X.device)
        z = coef.clone() if is_torch else coef.copy()
        t_val = 1.0

        for _ in range(self.max_iter):
            pred = X_design @ z  # (n, B)
            resid_batch = y_gpu.T - pred  # (n, B), r = y - X@coef
            # Pinball subgradient: tau if r>0, -(1-tau) if r<0
            # Use Python float; torch/cupy auto-cast to match X_design.dtype
            subgrad_pos = float(tau)
            subgrad_neg = float(tau - 1.0)
            subgrad = xp.where(resid_batch > 0, subgrad_pos, subgrad_neg)
            if is_torch:
                subgrad = subgrad.to(X_design.dtype)
            grad = X_design.T @ subgrad / n  # (p, B)
            coef_new = z - step * grad
            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t_val * t_val) ** 0.5)
            z = coef_new + ((t_val - 1.0) / t_new) * (coef_new - coef)
            coef = coef_new; t_val = t_new

        boot_params = np.asarray(_to_numpy(coef.T))  # (B, p)
        return boot_params, params, X_design

    def _compute_inference_bootstrap(self, X, y):
        """Residual bootstrap inference for quantile regression. Backend-aware."""
        from statgpu.backends import _to_numpy
        dev = getattr(self, '_selected_backend_name', 'cpu') or 'cpu'
        dev = {'numpy': 'cpu', 'cupy': 'cuda', 'torch': 'torch'}.get(dev, dev)

        # Use correct pinball batched FISTA for GPU backends
        if dev != 'cpu' and self.fit_intercept is not None:
            boot_params, params_gpu, _ = self._compute_inference_bootstrap_batched(X, y)
            boot_params = np.asarray(boot_params)
            params_cpu = np.asarray(_to_numpy(params_gpu))
            self._bse = np.std(boot_params, axis=0, ddof=1)
            self._zvalues = params_cpu / (self._bse + 1e-30)
            pvalues = np.array([min(2.0 * min(np.mean(boot_params[:, i] <= 0.0),
                                               np.mean(boot_params[:, i] >= 0.0)), 1.0)
                                for i in range(len(params_cpu))])
            self._pvalues = pvalues
            z_crit = 1.96
            self._conf_int = np.column_stack([params_cpu - z_crit * self._bse,
                                               params_cpu + z_crit * self._bse])
            from statgpu.inference._results import ParameterInferenceResult
            self._inference_result = ParameterInferenceResult(
                method="bootstrap", params=params_cpu.copy(), bse=self._bse.copy(),
                statistic=self._zvalues.copy(), statistic_name="z",
                pvalues=self._pvalues.copy(), conf_int=self._conf_int.copy(),
                distribution="normal", metadata={"n_bootstrap": self.n_bootstrap})
            self._inference_result.apply_to(self)
            return

        X_cpu = np.asarray(_to_numpy(X), dtype=float)
        y_cpu = np.asarray(_to_numpy(y), dtype=float).ravel()
        n = X_cpu.shape[0]

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X_cpu])
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            X_design = X_cpu
            params = self.coef_.copy()

        eta = X_design @ params
        resid = y_cpu - eta
        y_fitted = eta

        B = self.n_bootstrap
        rng = np.random.default_rng(self.random_state)
        boot_params = np.zeros((B, len(params)), dtype=float)

        # For GPU backends, pre-convert X_design once to avoid _to_array per iteration
        X_refit = X_design
        if dev != 'cpu':
            X_refit = self._to_array(X_design, backend={'cuda':'cupy','torch':'torch'}.get(dev, dev))

        for b in range(B):
            idx = rng.integers(0, n, size=n)
            y_star = y_fitted + resid[idx]
            if dev != 'cpu':
                y_star = self._to_array(y_star, backend={'cuda':'cupy','torch':'torch'}.get(dev, dev))
            m = QuantileRegression(
                quantile=self.quantile, fit_intercept=False,
                max_iter=self.max_iter, tol=self.tol, device=dev,
                compute_inference=False,
            )
            m.fit(X_refit, y_star)
            boot_params[b, :] = m._params

        self._bse = np.std(boot_params, axis=0, ddof=1)
        self._zvalues = params / (self._bse + 1e-30)

        pvalues = np.zeros(len(params), dtype=float)
        for i in range(len(params)):
            pvalues[i] = min(2.0 * min(
                np.mean(boot_params[:, i] <= 0.0),
                np.mean(boot_params[:, i] >= 0.0)
            ), 1.0)
        self._pvalues = pvalues

        self._conf_int = np.column_stack([
            np.quantile(boot_params, 0.025, axis=0),
            np.quantile(boot_params, 0.975, axis=0),
        ])

        from statgpu.inference._results import ParameterInferenceResult
        self._inference_result = ParameterInferenceResult(
            method="bootstrap",
            params=params.copy(),
            bse=self._bse.copy(),
            statistic=self._zvalues.copy(),
            statistic_name="z",
            pvalues=self._pvalues.copy(),
            conf_int=self._conf_int.copy(),
            distribution="bootstrap_percentile",
            metadata={"n_bootstrap": B, "method": "residual_bootstrap",
                      "quantile": self.quantile},
        )
        self._inference_result.apply_to(self)

    def predict(self, X):
        self._check_is_fitted()
        backend_name = self._selected_backend_name or "numpy"
        X_arr = self._to_array(X, backend=backend_name)
        from statgpu.backends._utils import _get_xp, xp_asarray
        xp = _get_xp(backend_name)
        coef = xp_asarray(self.coef_, xp=xp, ref_arr=X_arr)
        intercept = xp_asarray(self.intercept_, xp=xp, ref_arr=X_arr)
        raw = X_arr @ coef + intercept
        from statgpu.backends import _to_numpy
        return np.asarray(_to_numpy(raw)) if backend_name != "numpy" else raw

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def summary(self):
        if not self._fitted:
            return f"{self.__class__.__name__}(not fitted)"
        lines = [
            f"{'='*60}",
            f"  QuantileRegression (τ={self.quantile})",
            f"{'='*60}",
        ]
        if self._inference_result is not None:
            try:
                df = self._inference_result.to_dataframe()
                lines.append(str(df.to_string(index=False)))
            except Exception:
                lines.append(f"  coef: {self._params}")
                if self._bse is not None:
                    lines.append(f"  std err (bootstrap): {self._bse}")
        else:
            lines.append(f"  coef: {self._params}")
            lines.append("  (bootstrap inference not computed)")
        lines.append(f"{'='*60}")
        return "\n".join(lines)
