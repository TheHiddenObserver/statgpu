"""Quantile regression with bootstrap inference support."""

from typing import Optional
import numpy as np

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
        backend = self._get_backend(device=self.device)
        backend_name = backend.name

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
            self.coef_ = np.asarray(params[:-1])
            self.intercept_ = float(params[-1])
        else:
            from statgpu.penalties._l2 import L2Penalty
            pen = L2Penalty(alpha=0.0)
            params, n_iter = fista_solver(loss, pen, X_arr, y_arr,
                                          max_iter=self.max_iter, tol=self.tol,
                                          sample_weight=sample_weight)
            self.coef_ = np.asarray(params)
            self.intercept_ = 0.0

        self.n_iter_ = n_iter
        if self.fit_intercept:
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self._params = self.coef_.copy()
        self._fitted = True

        if self.compute_inference:
            self._compute_inference(X_np, y_np, loss)

        return self

    def _compute_inference(self, X, y, loss):
        """Dispatch to kernel-based or bootstrap inference."""
        if self.inference_method == "bootstrap":
            self._compute_inference_bootstrap(X, y)
        else:
            self._compute_inference_kernel(X, y)

    # ---- Kernel helpers (matching statsmodels) ----
    @staticmethod
    def _get_kernel_fn(name):
        import numpy as _np
        from scipy.stats import norm as _norm
        _KERNELS = {
            'epa': lambda u: 0.75 * (1 - u**2) * _np.where(_np.abs(u) <= 1, 1, 0),
            'gau': _norm.pdf,
            'biw': lambda u: 15./16 * (1 - u**2)**2 * _np.where(_np.abs(u) <= 1, 1, 0),
            'cos': lambda u: _np.where(_np.abs(u) <= 0.5, 1 + _np.cos(2*_np.pi*u), 0),
            'par': lambda u: _np.where(_np.abs(u) <= 0.5,
                    4./3 - 8*u**2 + 8*_np.abs(u)**3,
                    _np.where(_np.abs(u) <= 1, 8*(1-_np.abs(u))**3/3., 0)),
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
            XtX_inv = _np.linalg.pinv(XtX)

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

    def _compute_inference_bootstrap(self, X, y):
        """Residual bootstrap inference for quantile regression."""
        n = X.shape[0]

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            X_design = X
            params = self.coef_.copy()

        eta = X_design @ params
        resid = y - eta
        y_fitted = eta

        B = self.n_bootstrap
        rng = np.random.default_rng(self.random_state)
        boot_params = np.zeros((B, len(params)), dtype=float)

        for b in range(B):
            idx = rng.integers(0, n, size=n)
            y_star = y_fitted + resid[idx]
            m = QuantileRegression(
                quantile=self.quantile, fit_intercept=False,
                max_iter=self.max_iter, tol=self.tol, device="cpu",
                compute_inference=False,
            )
            m.fit(X_design, y_star)
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
        return np.asarray(X) @ self.coef_ + self.intercept_

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
