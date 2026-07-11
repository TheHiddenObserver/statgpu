"""
Cox Proportional Hazards regression with GPU acceleration.

Implements Cox PH model using Breslow and Efron approximations for ties with
Newton-Raphson optimization. Matches R's survival::coxph() API.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
import os
import numpy as np
from scipy import stats

from statgpu._base import BaseEstimator
from statgpu._config import Device

# Optional Cython import for faster Efron gradient/Hessian computation
try:
    from ._cox_efron_cy import efron_grad_hess as _efron_grad_hess_cython
    HAS_CYTHON_EFRON = True
except ImportError:
    HAS_CYTHON_EFRON = False
    _efron_grad_hess_cython = None

try:
    from statgpu.survival._cox_efron_triton import _find_p_ce
    HAS_TRITON_EFRON = True
except ImportError:
    HAS_TRITON_EFRON = False
    _find_p_ce = None


def _unpack_efron_pre6(efron_pre):
    """``(uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft)`` — supports legacy 5-tuple in tests only."""
    if len(efron_pre) == 6:
        return efron_pre
    if len(efron_pre) == 5:
        uft, uft_ix, re, rx, nuft = efron_pre
        return uft, uft_ix, re, rx, nuft, None
    raise ValueError(f"invalid efron_pre length {len(efron_pre)}")


# ── Numba JIT-compiled Efron backward scan (opt-in via env var) ─────
_USE_NUMBA = (
    os.environ.get("STATGPU_USE_NUMBA", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
_HAS_NUMBA_EFRON = False
if _USE_NUMBA:
    try:
        from numba import njit

        @njit(cache=True)
        def _efron_backward_scan_numba(
            X, e_linpred, risk_sum, risk_X_sum,
            first_idx_uft, fail_ptr, fail_ind,
            nuft, n, p,
        ):
            """Numba-compiled Efron backward scan — eliminates Python loop overhead."""
            xp0 = 0.0
            xp1 = np.zeros(p)
            xp2 = np.zeros((p, p))
            grad = np.zeros(p)
            hess = np.zeros((p, p))

            for g in range(nuft - 1, -1, -1):
                enter_start = first_idx_uft[g]
                enter_end = n if g == nuft - 1 else first_idx_uft[g + 1]
                if enter_end > enter_start:
                    xp0 += risk_sum[enter_start] - risk_sum[enter_end]
                    for j in range(p):
                        xp1[j] += risk_X_sum[enter_start, j] - risk_X_sum[enter_end, j]
                    for r in range(enter_start, enter_end):
                        elx = e_linpred[r]
                        for j in range(p):
                            for k in range(p):
                                xp2[j, k] += elx * X[r, j] * X[r, k]

                fs = fail_ptr[g]
                fe = fail_ptr[g + 1]
                d = fe - fs
                if d == 0:
                    continue

                xp0f = 0.0
                xp1f = np.zeros(p)
                xp2f = np.zeros((p, p))
                for idx in range(fs, fe):
                    r = fail_ind[idx]
                    elx = e_linpred[r]
                    xp0f += elx
                    for j in range(p):
                        xp1f[j] += elx * X[r, j]
                        for k in range(p):
                            xp2f[j, k] += elx * X[r, j] * X[r, k]

                sum_inv = 0.0
                sum_J = 0.0
                sum_aa = 0.0
                sum_bb = 0.0
                sum_ab = 0.0
                for k in range(d):
                    c0 = xp0 - (float(k) / float(d)) * xp0f
                    if c0 < 1e-300:
                        c0 = 1e-300
                    inv_k = 1.0 / c0
                    J_k = float(k) / float(d) * inv_k
                    sum_inv += inv_k
                    sum_J += J_k
                    sum_aa += inv_k * inv_k
                    sum_bb += J_k * J_k
                    sum_ab += inv_k * J_k

                for idx in range(fs, fe):
                    r = fail_ind[idx]
                    for j in range(p):
                        grad[j] += X[r, j]
                for j in range(p):
                    grad[j] -= xp1[j] * sum_inv - xp1f[j] * sum_J

                for j in range(p):
                    for k in range(p):
                        hess[j, k] -= xp2[j, k] * sum_inv
                        hess[j, k] += xp2f[j, k] * sum_J
                        hess[j, k] += sum_aa * xp1[j] * xp1[k]
                        hess[j, k] += sum_bb * xp1f[j] * xp1f[k]
                        hess[j, k] -= sum_ab * (xp1[j] * xp1f[k] + xp1f[j] * xp1[k])

            return grad, -hess

        _HAS_NUMBA_EFRON = True
    except ImportError:
        pass


def _efron_backward_scan_python(
    X, e_linpred, risk_sum, risk_X_sum,
    first_idx_uft, uft_ix, nuft, n, p,
):
    """Pure Python fallback — same algorithm, no Numba required."""
    xp0 = 0.0
    xp1 = np.zeros(p, dtype=np.float64)
    xp2 = np.zeros((p, p), dtype=np.float64)
    grad = np.zeros(p, dtype=np.float64)
    hess = np.zeros((p, p), dtype=np.float64)

    for g in range(nuft - 1, -1, -1):
        enter_start = int(first_idx_uft[g])
        enter_end = n if g == nuft - 1 else int(first_idx_uft[g + 1])
        if enter_end > enter_start:
            xp0 += risk_sum[enter_start] - risk_sum[enter_end]
            xp1 += risk_X_sum[enter_start] - risk_X_sum[enter_end]
            xp2 += X[enter_start:enter_end].T @ (
                X[enter_start:enter_end] * e_linpred[enter_start:enter_end, None]
            )

        ix_ev = uft_ix[g]
        d = len(ix_ev)
        if d == 0:
            continue

        v = X[ix_ev]
        elx = e_linpred[ix_ev]
        xp0f = float(elx.sum())
        xp1f = v.T @ elx
        xp2f = (v * elx[:, None]).T @ v

        J = np.arange(d, dtype=np.float64) / d
        c0 = xp0 - J * xp0f
        np.maximum(c0, 1e-300, out=c0)
        inv = 1.0 / c0
        J_inv = J * inv
        sum_inv = inv.sum()
        sum_J = J_inv.sum()
        sum_aa = np.dot(inv, inv)
        sum_bb = np.dot(J_inv, J_inv)
        sum_ab = np.dot(inv, J_inv)

        grad += v.sum(axis=0)
        grad -= xp1 * sum_inv - xp1f * sum_J

        hess -= xp2 * sum_inv
        hess += xp2f * sum_J
        hess += sum_aa * np.outer(xp1, xp1)
        hess += sum_bb * np.outer(xp1f, xp1f)
        hess -= sum_ab * (np.outer(xp1, xp1f) + np.outer(xp1f, xp1))

    return grad, -hess


def _efron_backward_scan_vectorized(
    X, e_linpred, risk_sum, risk_X_sum,
    first_idx_uft, uft_ix, nuft, n, p,
):
    """Vectorized Efron gradient/Hessian via suffix outer products.

    Properly handles tied failures with Efron's k/d correction.
    O(n·p²) memory for suffix outer products; O(nuft·d·p) for Efron loop.
    """
    X_exp = X * e_linpred[:, None]
    total = X_exp.T @ X  # (p, p)

    # Suffix outer products: risk_X2[g] = sum_{i >= first_idx[g]} X_i exp(eta_i) X_i'
    fi = first_idx_uft.astype(np.int64)
    flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
    prefix_flat = np.cumsum(flat, axis=0)  # (n, p*p)

    prefix_at_g = np.zeros((nuft, p, p), dtype=np.float64)
    mask = fi > 0
    if mask.any():
        prefix_at_g[mask] = prefix_flat[fi[mask] - 1].reshape(-1, p, p)
    risk_X2 = total[None, :, :] - prefix_at_g  # (nuft, p, p)

    # Efron gradient/Hessian with proper tied-event correction
    grad = np.zeros(p, dtype=np.float64)
    hess = np.zeros((p, p), dtype=np.float64)

    for g in range(nuft):
        ix_ev = uft_ix[g]
        d = len(ix_ev)
        if d == 0:
            continue

        # Risk set quantities at this failure time
        s0 = float(risk_sum[fi[g]])
        s1 = risk_X_sum[fi[g]]  # (p,)

        # Tied failure quantities
        v = X[ix_ev]  # (d, p) — ALL failures, not just first
        elx = e_linpred[ix_ev]  # (d,)
        xp0f = float(elx.sum())
        xp1f = v.T @ elx  # (p,) — weighted sum of failure covariates

        # Efron correction: for k=0..d-1, denominator = s0 - (k/d)*xp0f
        J = np.arange(d, dtype=np.float64) / d  # (d,)
        c0 = s0 - J * xp0f  # (d,)
        np.maximum(c0, 1e-300, out=c0)
        inv = 1.0 / c0  # (d,)
        J_inv = J * inv  # (d,)
        sum_inv = inv.sum()
        sum_J = J_inv.sum()
        sum_aa = np.dot(inv, inv)
        sum_bb = np.dot(J_inv, J_inv)
        sum_ab = np.dot(inv, J_inv)

        # Gradient: sum of ALL failure X's minus Efron-corrected risk term
        grad += v.sum(axis=0)  # sum_{i in D_g} X_i
        grad -= s1 * sum_inv - xp1f * sum_J

        # Hessian: Efron-corrected second moment
        hess -= risk_X2[g] * sum_inv
        hess += (v * elx[:, None]).T @ v * sum_J  # xp2f * sum_J
        hess += sum_aa * np.outer(s1, s1)
        hess += sum_bb * np.outer(xp1f, xp1f)
        hess -= sum_ab * (np.outer(s1, xp1f) + np.outer(xp1f, s1))

    return grad, -hess


class CoxPH(BaseEstimator):
    """
    Cox Proportional Hazards regression with GPU acceleration.
    
    Parameters
    ----------
    ties : str, default='breslow'
        Method for handling ties: 'breslow' or 'efron'.
    tol : float, default=1e-9
        Convergence tolerance for Newton-Raphson.
    max_iter : int, default=100
        Maximum number of iterations.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    compute_inference : bool, default=True
        If True, compute standard errors/tests/baseline hazard on CPU after fitting.
        Set to False to reduce CPU-GPU data transfers in CUDA mode.
    compute_cindex : bool, default=True
        If True, compute training-set C-index during fit. Disabling this can
        significantly reduce fit time, especially on CUDA/Torch for moderate n.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients (log hazard ratios).
    hazard_ratios_ : ndarray of shape (n_features,)
        exp(coef) = hazard ratios.
    """
    
    def __init__(
        self,
        ties: str = 'breslow',
        tol: float = 1e-9,
        max_iter: int = 100,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        compute_cindex: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        penalty: float = 0.0,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.ties = ties.lower()
        self.tol = tol
        self.max_iter = max_iter
        self.compute_inference = compute_inference
        self.compute_cindex = bool(compute_cindex)
        self.cov_type = cov_type.lower()
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.penalty = float(penalty)

        if self.ties not in ('breslow', 'efron'):
            raise ValueError("ties must be 'breslow' or 'efron'")
        if self.cov_type not in ("nonrobust", "hc0", "hc1", "cluster"):
            raise ValueError("cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'")
        if self.penalty < 0:
            raise ValueError("penalty must be non-negative")
        
        # Fitted attributes
        self.coef_ = None
        self.hazard_ratios_ = None
        
        # Internal storage for inference
        self._time = None
        self._event = None
        self._X = None
        self._entry = None
        self._nobs = None
        self._nevents = None
        self._bse = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        self._log_likelihood = None
        self._log_likelihood_null = None
        self._iterations = 0
        self._converged = False
        self._var_matrix = None
        self._score_test_stat = None
        self._baseline_hazard = None
        self._baseline_cumulative_hazard = None
        self._unique_times = None
        self._cindex = None
        self._feature_names = None
        self._wald_test_stat = None
        self._wald_test_pvalue = None
        self._lr_test_stat = None
        self._lr_test_pvalue = None
        self._score_test_pvalue = None
        # Efron only: cached (uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft); depends only on sorted time/event.
        self._efron_pre = None
        # Efron optimization: True when all failure groups are singletons (no ties),
        # in which case Efron equals Breslow and we can use faster vectorized paths.
        self._efron_all_singletons = False
        # Efron only: cached CSR packed indices for GPU kernels.
        # (enter_ptr, enter_ind, exit_ptr, exit_ind, fail_ptr, fail_ind, first_idx_uft, nuft)
        self._efron_pre_csr = None
        # Breslow only: cached (first_idx_uft, counts_uft) on CPU.
        self._breslow_pre = None
        # Breslow only: cached (first_idx_uft_gpu, counts_uft_gpu) on GPU.
        self._breslow_pre_gpu = None

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Best-effort Torch CUDA cache cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    def __del__(self):
        try:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()
        except Exception:
            pass

    @staticmethod
    def _extract_convergence_status(result):
        """Best-effort convergence extraction from statsmodels results."""
        conv_attr = getattr(result, "converged", None)
        if conv_attr is not None:
            return bool(conv_attr)

        mle_retvals = getattr(result, "mle_retvals", None)
        if isinstance(mle_retvals, dict):
            conv_attr = mle_retvals.get("converged")
            if conv_attr is not None:
                return bool(conv_attr)
        elif mle_retvals is not None:
            conv_attr = getattr(mle_retvals, "converged", None)
            if conv_attr is not None:
                return bool(conv_attr)
        return None
        
    def fit(self, X=None, time=None, event=None, entry=None, cluster=None, init_coef=None, formula=None, data=None):
        """
        Fit Cox Proportional Hazards model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix. Required if ``formula`` is None.
        time : array-like of shape (n_samples,)
            Time to event or censoring. Required if ``formula`` is None.
        event : array-like of shape (n_samples,)
            Event indicator (1 = event, 0 = censored). Required if ``formula`` is None.
        entry : array-like of shape (n_samples,), optional
            Entry time for delayed entry (left truncation).
        cluster : array-like of shape (n_samples,), optional
            Cluster ids for cluster-robust covariance when `cov_type='cluster'`.
        init_coef : array-like of shape (n_features,), optional
            Initial coefficient guess for warm-start optimization.
        formula : str or None
            R-style formula with Surv() response, e.g.
            ``"Surv(time, event) ~ x1 + x2 + C(sex)"``.
        data : pd.DataFrame or None
            DataFrame used with ``formula`` for column lookup.

        Returns
        -------
        self : CoxPH
            Fitted estimator.
        """
        # Handle formula interface
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            from statgpu.core.formula import _surv, make_surv_env
            import patsy
            from patsy import EvalEnvironment

            env = make_surv_env()
            # Create evaluation environment with custom Surv function
            custom_env = EvalEnvironment([env])
            y_patsy, X_patsy = patsy.dmatrices(
                formula, data, eval_env=custom_env, return_type="matrix",
            )
            design_info = X_patsy.design_info
            # y_patsy is the result of Surv(time, event) -> shape (n, 2)
            y_arr = np.asarray(y_patsy)
            if y_arr.ndim == 1:
                raise ValueError(
                    "Formula response must be Surv(time, event), not a single variable. "
                    "Use: formula='Surv(time, event) ~ x1 + x2'"
                )
            time = y_arr[:, 0]
            event = y_arr[:, 1]
            X_arr = np.asarray(X_patsy)

            # Drop intercept column from design matrix (CoxPH doesn't use intercept)
            self._feature_names = list(design_info.column_names)
            if "Intercept" in self._feature_names:
                self._feature_names.remove("Intercept")
                X_arr = X_arr[:, 1:]
            self._design_info = design_info
            X = X_arr
        else:
            if X is None or time is None or event is None:
                raise ValueError(
                    "Either formula+data or X+time+event must be provided."
                )
            self._design_info = None
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            import cupy as cp
            
            X_gpu = cp.asarray(self._to_array(X), dtype=cp.float64)
            time_gpu = cp.asarray(self._to_array(time), dtype=cp.float64)
            event_gpu = cp.asarray(self._to_array(event), dtype=cp.int32)
            entry_gpu = None if entry is None else cp.asarray(self._to_array(entry), dtype=cp.float64)
            
            if X_gpu.ndim == 1:
                X_gpu = X_gpu.reshape(-1, 1)
            if entry_gpu is not None and entry_gpu.shape[0] != X_gpu.shape[0]:
                raise ValueError("entry must have shape (n_samples,)")
            
            self._nobs = int(X_gpu.shape[0])
            self._nevents = int(cp.sum(event_gpu).item())
            if self._feature_names is None:
                self._feature_names = [f'x{i+1}' for i in range(int(X_gpu.shape[1]))]
            
            # Keep CPU copies only when CPU-side inference/baseline stats are requested.
            if self.compute_inference:
                self._X = cp.asnumpy(X_gpu)
                self._time = cp.asnumpy(time_gpu)
                self._event = cp.asnumpy(event_gpu)
                self._entry = None if entry_gpu is None else cp.asnumpy(entry_gpu)
            else:
                self._X = None
                self._time = None
                self._event = None
                self._entry = None
            
            cluster_gpu = None if cluster is None else cp.asarray(self._to_array(cluster), dtype=cp.int64)
            self._fit_gpu(X_gpu, time_gpu, event_gpu, entry_gpu, cluster_gpu, init_coef=init_coef)
        elif device == Device.TORCH:
            import torch

            torch_device = "cuda"

            X_torch = self._to_array(X, Device.TORCH, backend="torch").to(dtype=torch.float64)
            time_torch = self._to_array(time, Device.TORCH, backend="torch").to(dtype=torch.float64)
            event_torch = self._to_array(event, Device.TORCH, backend="torch").to(dtype=torch.int32)
            entry_torch = None if entry is None else self._to_array(
                entry, Device.TORCH, backend="torch"
            ).to(dtype=torch.float64)

            if X_torch.ndim == 1:
                X_torch = X_torch.reshape(-1, 1)
            if entry_torch is not None and entry_torch.shape[0] != X_torch.shape[0]:
                raise ValueError("entry must have shape (n_samples,)")

            self._nobs = int(X_torch.shape[0])
            self._nevents = int(torch.sum(event_torch).item())
            if self._feature_names is None:
                self._feature_names = [f'x{i+1}' for i in range(int(X_torch.shape[1]))]

            # Keep CPU copies only when CPU-side inference/baseline stats are requested.
            if self.compute_inference:
                self._X = X_torch.cpu().numpy()
                self._time = time_torch.cpu().numpy()
                self._event = event_torch.cpu().numpy()
                self._entry = None if entry_torch is None else entry_torch.cpu().numpy()
            else:
                self._X = None
                self._time = None
                self._event = None
                self._entry = None

            cluster_torch = None if cluster is None else self._to_array(
                cluster, Device.TORCH, backend="torch"
            ).to(dtype=torch.int64)
            self._fit_torch(
                X_torch,
                time_torch,
                event_torch,
                entry_torch,
                cluster_torch,
                torch_device,
                init_coef=init_coef,
            )
        else:
            X_np = np.asarray(self._to_array(X, Device.CPU), dtype=np.float64)
            time_np = np.asarray(self._to_array(time, Device.CPU), dtype=np.float64)
            event_np = np.asarray(self._to_array(event, Device.CPU), dtype=np.int32)
            entry_np = None if entry is None else np.asarray(self._to_array(entry, Device.CPU), dtype=np.float64)
            
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            if entry_np is not None and entry_np.shape[0] != X_np.shape[0]:
                raise ValueError("entry must have shape (n_samples,)")
            
            self._nobs = X_np.shape[0]
            self._nevents = np.sum(event_np)
            
            # Store original data (CPU mode is CPU-only)
            self._time = time_np.copy()
            self._event = event_np.copy()
            self._X = X_np.copy()
            self._entry = None if entry_np is None else entry_np.copy()
            if self._feature_names is None:
                self._feature_names = [f'x{i+1}' for i in range(X_np.shape[1])]
            
            cluster_np = None if cluster is None else np.asarray(self._to_array(cluster, Device.CPU))
            self._fit_cpu(X_np, time_np, event_np, entry_np, cluster_np, init_coef=init_coef)
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, time, event, entry=None, cluster=None, init_coef=None):
        """Fit using CPU (NumPy)."""
        if entry is not None:
            self._fit_cpu_with_entry(X, time, event, np.asarray(entry, dtype=np.float64), cluster)
            return
        n_samples, n_features = X.shape
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = np.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        entry_sorted = None if entry is None else np.asarray(entry, dtype=np.float64)[order]
        cluster_sorted = None if cluster is None else np.asarray(cluster)[order]
        
        self._efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
            self._efron_pre = self._efron_unique_failure_indices(time_sorted, event_sorted)
            try:
                uft, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(self._efron_pre)
                self._efron_all_singletons = bool(nuft > 0) and all(
                    len(ix) == 1 for ix in uft_ix
                )
            except Exception:
                self._efron_all_singletons = False
        else:
            self._efron_all_singletons = False
            self._breslow_pre = self._breslow_unique_failure_groups(
                time_sorted, event_sorted
            )
            if entry_sorted is not None:
                event_idx_np = np.flatnonzero(event_sorted.astype(np.int32) == 1)
                event_times_np = time_sorted[event_idx_np].astype(np.float64, copy=False)
                uft_np, inv_np = np.unique(event_times_np, return_inverse=True)
                self._entry_fail_groups_np = [
                    event_idx_np[inv_np == g].astype(np.int64, copy=False)
                    for g in range(len(uft_np))
                ]
                self._entry_fail_times_np = uft_np.astype(np.float64, copy=False)
                self._entry_order_np = np.argsort(entry_sorted).astype(np.int64, copy=False)
                self._entry_add_end_np = np.searchsorted(
                    entry_sorted, uft_np, side="left"
                ).astype(np.int64, copy=False)
                self._entry_rem_end_np = np.searchsorted(
                    time_sorted, uft_np, side="left"
                ).astype(np.int64, copy=False)
            else:
                self._entry_fail_groups_np = None
                self._entry_fail_times_np = None
                self._entry_order_np = None
                self._entry_add_end_np = None
                self._entry_rem_end_np = None
        
        # Initialize coefficients (supports warm-start path in CV)
        if init_coef is None:
            beta = np.zeros(n_features, dtype=np.float64)
        else:
            beta = np.asarray(init_coef, dtype=np.float64).reshape(-1)
            if beta.shape[0] != n_features:
                raise ValueError("init_coef must have shape (n_features,)")

        # Compute null log-likelihood (beta = 0)
        self._log_likelihood_null = self._compute_log_likelihood(
            np.zeros(n_features), X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
        )

        # Newton-Raphson optimization with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0
        # Preferred Newton direction for CPU path; updated adaptively.
        preferred_direction = -1.0
        iteration = -1  # default if max_iter=0

        for iteration in range(self.max_iter):
            # Compute gradient and Hessian
            grad, hess = self._compute_gradient_hessian(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
            )

            # Add penalty terms: gradient -= 2*penalty*beta, hessian -= 2*penalty*I
            if use_penalty:
                grad = grad - 2 * penalty * beta
                hess = hess - 2 * penalty * np.eye(n_features, dtype=np.float64)

            # Solve a Newton-like step on (-hess). In practice, different tie paths
            # may expose Hessian with different sign conventions, so we choose the
            # ascent direction adaptively below using objective evaluation.
            try:
                delta = np.linalg.solve(-hess, grad)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                delta = np.linalg.lstsq(-hess, grad, rcond=None)[0]

            # Line search with step halving
            # Compute log-likelihood at current point
            old_ll = self._compute_log_likelihood(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
            )
            if use_penalty:
                old_ll = old_ll - penalty * np.sum(beta ** 2)

            # Fast path: try preferred direction first, only test opposite
            # when the preferred full step does not improve.
            direction = preferred_direction
            new_beta = beta + direction * delta
            new_ll = self._compute_log_likelihood(
                new_beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
            )
            if use_penalty:
                new_ll = new_ll - penalty * np.sum(new_beta ** 2)

            if new_ll <= old_ll - 1e-8:
                # Probe the opposite direction only when needed.
                if entry_sorted is None:
                    alt_direction = -direction
                    alt_beta = beta + alt_direction * delta
                    alt_ll = self._compute_log_likelihood(
                        alt_beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
                    )
                    if use_penalty:
                        alt_ll = alt_ll - penalty * np.sum(alt_beta ** 2)
                    if alt_ll > new_ll:
                        direction = alt_direction
                        preferred_direction = alt_direction
                        new_beta = alt_beta
                        new_ll = alt_ll

                # Backtracking line search from step=0.5; step=1 was already evaluated.
                if new_ll <= old_ll - 1e-8:
                    step = 0.5
                    for _ in range(20):
                        trial_beta = beta + direction * step * delta
                        trial_ll = self._compute_log_likelihood(
                            trial_beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
                        )
                        if use_penalty:
                            trial_ll = trial_ll - penalty * np.sum(trial_beta ** 2)
                        if trial_ll > old_ll - 1e-8:
                            new_beta = trial_beta
                            new_ll = trial_ll
                            break
                        step *= 0.5
                else:
                    step = 1.0
            else:
                # Keep successful direction for the next iteration.
                preferred_direction = direction
                step = 1.0

            # Check convergence
            if np.linalg.norm(delta) * step < self.tol:
                self._converged = True
                beta = new_beta
                break

            beta = new_beta
        
        self._iterations = iteration + 1
        self.coef_ = beta
        self.hazard_ratios_ = np.exp(beta)
        
        # Compute final log-likelihood
        self._log_likelihood = self._compute_log_likelihood(
            beta, X_sorted, time_sorted, event_sorted, self._efron_pre, entry=entry_sorted
        )
        
        # Compute optional inference statistics
        if self.compute_inference:
            self._compute_inference_cpu(X_sorted, time_sorted, event_sorted, cluster_sorted)
            self._compute_baseline_hazard(X_sorted, time_sorted, event_sorted)
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None

        # Release large temporary GPU tensors early.
        try:
            del X_sorted
        except Exception:
            pass
        try:
            del time_sorted
        except Exception:
            pass
        try:
            del event_sorted
        except Exception:
            pass
        try:
            del grad
        except Exception:
            pass
        try:
            del hess
        except Exception:
            pass
        try:
            del delta
        except Exception:
            pass
        self._cleanup_cuda_memory()
        if self.compute_cindex:
            self._compute_cindex()
        else:
            self._cindex = None

    def _fit_cpu_with_entry(self, X, time, event, entry, cluster=None):
        """Fit using statsmodels PHReg when delayed entry is provided.

        Note: L2 penalty is not applied in this path (statsmodels PHReg
        does not support penalized fitting). A warning is emitted when
        penalty is specified.
        """
        if float(self.penalty) > 0:
            import warnings
            warnings.warn(
                "CoxPH with entry (delayed entry) does not support penalties via "
                "statsmodels PHReg. The penalty will be ignored. "
                "Use the GPU/torch path for penalized Cox with delayed entry.",
                UserWarning, stacklevel=3,
            )
        import statsmodels.duration.api as smd

        n_samples, n_features = X.shape
        model = smd.PHReg(time, X, status=event, entry=entry, ties=self.ties)
        res = model.fit(disp=0)

        self._iterations = int(getattr(res, "iterations", 0) or 0)
        conv_attr = self._extract_convergence_status(res)
        self._converged = bool(conv_attr) if conv_attr is not None else False
        self.coef_ = np.asarray(res.params, dtype=np.float64)
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood = float(res.llf)

        try:
            null_model = smd.PHReg(time, np.zeros((n_samples, 1), dtype=np.float64), status=event, entry=entry, ties=self.ties)
            null_res = null_model.fit(disp=0)
            self._log_likelihood_null = float(null_res.llf)
        except Exception:
            self._log_likelihood_null = np.nan

        cov = np.asarray(res.cov_params(), dtype=np.float64)
        if cov.shape != (n_features, n_features):
            cov = np.full((n_features, n_features), np.nan, dtype=np.float64)
        self._var_matrix = cov
        self._bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
        self._zvalues = self.coef_ / (self._bse + 1e-30)
        self._pvalues = 2 * (1 - stats.norm.cdf(np.abs(self._zvalues)))
        self._conf_int = np.asarray(res.conf_int(), dtype=np.float64)

        # Delayed-entry robust covariance override is intentionally skipped:
        # current internal robust score/hessian helpers do not account for entry.

        self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
        self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
        try:
            var_inv = np.linalg.solve(self._var_matrix, np.eye(n_features))
            self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
        except np.linalg.LinAlgError:
            self._wald_test_stat = np.nan
        self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
        self._score_test_stat = np.nan
        self._score_test_pvalue = np.nan

        # Baseline hazard from PHReg output.
        try:
            base = res.baseline_cumulative_hazard[0]
            self._unique_times = np.asarray(base[0], dtype=np.float64)
            self._baseline_cumulative_hazard = np.asarray(base[1], dtype=np.float64)
            if self._baseline_cumulative_hazard.size > 0:
                self._baseline_hazard = np.diff(
                    np.concatenate([[0.0], self._baseline_cumulative_hazard])
                )
            else:
                self._baseline_hazard = np.array([], dtype=np.float64)
        except Exception:
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None

        if self.compute_cindex:
            self._compute_cindex()
        else:
            self._cindex = None
    
    def _fit_gpu(self, X, time, event, entry=None, cluster=None, init_coef=None):
        """Fit using GPU with full GPU computation."""
        import cupy as cp
        from statgpu.inference._distributions_backend import norm
        
        n_samples, n_features = X.shape

        # Transfer to GPU once
        X = cp.asarray(X, dtype=cp.float64)
        time = cp.asarray(time, dtype=cp.float64)
        event = cp.asarray(event, dtype=cp.int32)
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = cp.argsort(time, kind="stable")
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        entry_sorted = None if entry is None else entry[order]
        cluster_sorted = None if cluster is None else cluster[order]
        event_idx_sorted = cp.where(event_sorted == 1)[0]
        self._event_idx_gpu = event_idx_sorted
        self._event_X_sum_gpu = (
            cp.sum(X_sorted[event_idx_sorted], axis=0)
            if int(event_idx_sorted.size) > 0
            else cp.zeros(n_features, dtype=cp.float64)
        )
        
        # Precompute Efron tie structure once (depends only on time/event order).
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
            if entry_sorted is None:
                efron_pre = self._efron_unique_failure_indices(
                    cp.asnumpy(time_sorted), cp.asnumpy(event_sorted)
                )
                self._efron_pre = efron_pre
                try:
                    _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                    self._efron_all_singletons = bool(nuft > 0) and all(
                        len(ix) == 1 for ix in uft_ix
                    )
                except Exception:
                    self._efron_all_singletons = False
                # Pack enter/exit/fail indices once; reuse across Newton steps on GPU.
                try:
                    from ._cox_efron_cuda import efron_indices_to_csr

                    uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = _unpack_efron_pre6(
                        efron_pre
                    )
                    (
                        enter_ptr,
                        enter_ind,
                        exit_ptr,
                        exit_ind,
                        fail_ptr,
                        fail_ind,
                    ) = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
                    self._efron_pre_csr = (
                        enter_ptr,
                        enter_ind,
                        exit_ptr,
                        exit_ind,
                        fail_ptr,
                        fail_ind,
                        first_idx_uft,
                        nuft,
                    )
                    self._efron_pre_csr_gpu = (
                        cp.asarray(enter_ptr, dtype=cp.int32),
                        cp.asarray(enter_ind, dtype=cp.int32),
                        cp.asarray(exit_ptr, dtype=cp.int32),
                        cp.asarray(exit_ind, dtype=cp.int32),
                        cp.asarray(fail_ptr, dtype=cp.int32),
                        cp.asarray(fail_ind, dtype=cp.int32),
                        cp.asarray(first_idx_uft, dtype=cp.int32),
                        int(nuft),
                    )
                except Exception:
                    self._efron_pre_csr = None
                    self._efron_pre_csr_gpu = None
            else:
                self._efron_pre = None
                self._efron_pre_csr = None
                self._efron_pre_csr_gpu = None
            try:
                _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                n_events = int(cp.asnumpy(cp.sum(event_sorted)))
                avg_tie = float(n_events / max(1, int(nuft)))
            except Exception:
                avg_tie = 1.0
        else:
            self._efron_pre = None
            self._efron_all_singletons = False
            self._efron_pre_csr = None
            self._efron_pre_csr_gpu = None
            first_idx_uft, counts_uft = self._breslow_unique_failure_groups(
                cp.asnumpy(time_sorted), cp.asnumpy(event_sorted)
            )
            self._breslow_pre = (first_idx_uft, counts_uft)
            self._breslow_pre_gpu = (
                cp.asarray(first_idx_uft, dtype=cp.int32),
                cp.asarray(counts_uft, dtype=cp.int32),
            )
            self._breslow_counts_f_gpu = cp.asarray(counts_uft, dtype=cp.float64)
            self._breslow_first_idx_np = np.asarray(first_idx_uft, dtype=np.int64)
            self._breslow_counts_np = np.asarray(counts_uft, dtype=np.float64)
            if entry_sorted is not None:
                # Entry path: avoid stale index cache drift across different sort permutations.
                self._entry_fail_groups_gpu = None
                self._entry_fail_times_gpu = None
                self._entry_order_gpu = None
                self._entry_add_end_np_gpu = None
                self._entry_rem_end_np_gpu = None
            else:
                self._entry_fail_groups_gpu = None
                self._entry_fail_times_gpu = None
                self._entry_order_gpu = None
                self._entry_add_end_np_gpu = None
                self._entry_rem_end_np_gpu = None
            n_events = int(cp.asnumpy(cp.sum(event_sorted)))
            avg_tie = float(n_events / max(1, int(len(counts_uft))))

        # Initialize coefficients on GPU (supports warm-start path in CV)
        if init_coef is None:
            beta = cp.zeros(n_features, dtype=cp.float64)
        else:
            beta = cp.asarray(np.asarray(init_coef, dtype=np.float64), dtype=cp.float64).reshape(-1)
            if int(beta.shape[0]) != int(n_features):
                raise ValueError("init_coef must have shape (n_features,)")

        # Compute null log-likelihood on GPU
        entry_ctx_gpu = None
        if entry_sorted is not None:
            _ctx = self._build_entry_ctx_gpu(time_sorted, event_sorted, entry_sorted, cp)
            event_idx_ctx = _ctx[5]
            entry_ctx_gpu = (
                _ctx[0], _ctx[1], _ctx[2], _ctx[3],
                cp.ascontiguousarray(X_sorted[_ctx[0]]),
                cp.ascontiguousarray(X_sorted),
                event_idx_ctx,
                cp.sum(X_sorted[event_idx_ctx], axis=0),
                _ctx[6],
            )
        loglik_null_gpu = self._compute_log_likelihood_gpu(
            cp.zeros(n_features, dtype=cp.float64),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
            entry=entry_sorted,
            entry_ctx=entry_ctx_gpu,
        )

        # Newton-Raphson optimization on GPU with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0
        diag_idx = cp.arange(n_features, dtype=cp.int64) if use_penalty else None
        eye_cache = (
            cp.eye(n_features, dtype=cp.float64)
            if (self.compute_inference or use_penalty)
            else None
        )

        # Newton-Raphson optimization on GPU
        loglik_gpu = None
        current_obj = None
        iteration = -1  # default if max_iter=0
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian on GPU
            grad, hess, aux_stats = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True, entry=entry_sorted, entry_ctx=entry_ctx_gpu
            )

            # Add penalty terms: gradient -= 2*penalty*beta, hessian -= 2*penalty*I
            if use_penalty:
                grad = grad - 2 * penalty * beta
                # In-place diagonal shift avoids allocating a new dense eye each iteration.
                hess[diag_idx, diag_idx] -= 2 * penalty

            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_gpu(hess, grad, cp, eye_cache=eye_cache)
            step = 1.0
            accepted_step = True
            if entry_sorted is not None:
                if current_obj is None:
                    old_ll = self._compute_log_likelihood_gpu_from_stats(
                        aux_stats[0], aux_stats[1], aux_stats[2], time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_gpu
                    )
                    if use_penalty:
                        old_ll = old_ll - penalty * cp.sum(beta * beta)
                    current_obj = old_ll
                else:
                    old_ll = current_obj
                new_beta = beta - delta
                new_ll = self._compute_log_likelihood_gpu(
                    new_beta, X_sorted, time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_gpu
                )
                if use_penalty:
                    new_ll = new_ll - penalty * cp.sum(new_beta * new_beta)
                if float((new_ll - old_ll).item()) <= -1e-8:
                    step = 0.5
                    accepted = False
                    for _ in range(20):
                        trial_beta = beta - step * delta
                        trial_ll = self._compute_log_likelihood_gpu(
                            trial_beta, X_sorted, time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_gpu
                        )
                        if use_penalty:
                            trial_ll = trial_ll - penalty * cp.sum(trial_beta * trial_beta)
                        if float((trial_ll - old_ll).item()) > -1e-8:
                            beta = trial_beta
                            current_obj = trial_ll
                            accepted = True
                            break
                        step *= 0.5
                    if not accepted:
                        accepted_step = False
                else:
                    beta = new_beta
                    current_obj = new_ll
            else:
                beta = beta - delta

            # Check convergence on GPU
            if entry_sorted is not None:
                delta_norm = float(cp.linalg.norm(delta).item())
                if accepted_step and delta_norm * step < self.tol:
                    self._converged = True
                    loglik_gpu = self._compute_log_likelihood_gpu(
                        beta, X_sorted, time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_gpu
                    )
                    break
            else:
                grad_norm = float(cp.linalg.norm(grad).item())
                delta_norm = float(cp.linalg.norm(delta).item())
                if accepted_step and grad_norm < max(self.tol * 10.0, 1e-8) and delta_norm * step < self.tol:
                    self._converged = True
                    # Reuse current iteration statistics to avoid an extra
                    # Efron log-likelihood setup pass when converged.
                    eta_cur, exp_eta_cur, risk_sum_cur = aux_stats
                    loglik_gpu = self._compute_log_likelihood_gpu_from_stats(
                        eta_cur, exp_eta_cur, risk_sum_cur, time_sorted, event_sorted, efron_pre, entry=entry_sorted
                    )
                    break
        
        # Compute final log-likelihood on GPU unless already obtained on convergence.
        if loglik_gpu is None:
            loglik_gpu = self._compute_log_likelihood_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre
                , entry=entry_sorted, entry_ctx=entry_ctx_gpu
            )
        
        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = cp.asnumpy(beta)
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(cp.asnumpy(loglik_null_gpu))
        self._log_likelihood = float(cp.asnumpy(loglik_gpu))
        if self.compute_cindex:
            cindex_gpu = self._compute_cindex_gpu(X_sorted, time_sorted, event_sorted, beta)
            self._cindex = float(cp.asnumpy(cindex_gpu))
        else:
            self._cindex = None
        
        # Inference:
        # - nonrobust: stay on GPU to avoid expensive host transfers/recompute
        # - hc0/hc1/cluster: use CPU inference path (current implementation)
        if self.compute_inference:
            if self.cov_type == "nonrobust":
                try:
                    info = -hess
                    rhs_eye = eye_cache if eye_cache is not None else cp.eye(info.shape[0], dtype=info.dtype)
                    var_gpu = cp.linalg.solve(info, rhs_eye)
                except Exception:
                    var_gpu = cp.linalg.pinv(-hess)
                bse_gpu = cp.sqrt(cp.maximum(cp.diag(var_gpu), 0.0))
                z_gpu = beta / (bse_gpu + 1e-30)
                p_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_gpu)))
                z_crit = norm.ppf(0.975)
                ci_gpu = cp.stack([beta - z_crit * bse_gpu, beta + z_crit * bse_gpu], axis=1)

                self._bse = cp.asnumpy(bse_gpu)
                self._zvalues = cp.asnumpy(z_gpu)
                self._pvalues = cp.asnumpy(p_gpu)
                self._conf_int = cp.asnumpy(ci_gpu)
                self._var_matrix = np.diag(np.square(self._bse))
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
                # Keep baseline hazard optional in CUDA fast path to reduce transfer overhead.
                self._baseline_hazard = None
                self._baseline_cumulative_hazard = None
                self._unique_times = None
            else:
                score_resid_gpu = self._compute_robust_score_residuals_gpu(X_sorted, time_sorted, event_sorted)
                try:
                    info = -hess
                    rhs_eye = eye_cache if eye_cache is not None else cp.eye(info.shape[0], dtype=info.dtype)
                    bread = cp.linalg.solve(info, rhs_eye)
                except Exception:
                    bread = cp.linalg.pinv(-hess)

                if self.cov_type == "cluster":
                    if cluster_sorted is None:
                        raise ValueError("cov_type='cluster' requires cluster ids in fit(..., cluster=...)")
                    unique_clusters = cp.unique(cluster_sorted)
                    meat = cp.zeros((n_features, n_features), dtype=cp.float64)
                    for g in unique_clusters:
                        u_g = cp.sum(score_resid_gpu[cluster_sorted == g], axis=0)
                        meat += cp.outer(u_g, u_g)
                else:
                    meat = score_resid_gpu.T @ score_resid_gpu
                    if self.cov_type == "hc1":
                        n = X_sorted.shape[0]
                        k = X_sorted.shape[1]
                        if n > k:
                            meat = meat * (n / (n - k))

                var_gpu = bread @ meat @ bread
                bse_gpu = cp.sqrt(cp.maximum(cp.diag(var_gpu), 0.0))
                z_gpu = beta / (bse_gpu + 1e-30)
                p_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_gpu)))
                z_crit = norm.ppf(0.975)
                ci_gpu = cp.stack([beta - z_crit * bse_gpu, beta + z_crit * bse_gpu], axis=1)

                self._var_matrix = cp.asnumpy(var_gpu)
                self._bse = cp.asnumpy(bse_gpu)
                self._zvalues = cp.asnumpy(z_gpu)
                self._pvalues = cp.asnumpy(p_gpu)
                self._conf_int = cp.asnumpy(ci_gpu)
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
                # Compute baseline hazard on GPU
                self._compute_baseline_hazard_gpu(X_sorted, time_sorted, event_sorted, beta)
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None

    def _fit_torch(self, X, time, event, entry=None, cluster=None, torch_device="cuda", init_coef=None):
        """Fit using Torch with full GPU computation."""
        import torch
        from statgpu.inference._distributions_backend import norm

        n_samples, n_features = X.shape

        # Sort by time ascending so risk-set terms are suffix sums
        order = torch.argsort(time, stable=True)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        entry_sorted = None if entry is None else entry[order]
        cluster_sorted = None if cluster is None else cluster[order]

        # Precompute Efron tie structure once (depends only on time/event order)
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_torch = None
        if self.ties == "efron":
            if entry_sorted is None:
                efron_pre = self._efron_unique_failure_indices(
                    time_sorted.cpu().numpy(), event_sorted.cpu().numpy()
                )
                self._efron_pre = efron_pre
                try:
                    _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                    self._efron_all_singletons = bool(nuft > 0) and all(
                        len(ix) == 1 for ix in uft_ix
                    )
                except Exception:
                    self._efron_all_singletons = False
                # Reuse CUDA CSR packing for Torch-CUDA fused kernels when available.
                try:
                    import cupy as cp
                    from ._cox_efron_cuda import efron_indices_to_csr

                    uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = _unpack_efron_pre6(
                        efron_pre
                    )
                    (
                        enter_ptr,
                        enter_ind,
                        exit_ptr,
                        exit_ind,
                        fail_ptr,
                        fail_ind,
                    ) = efron_indices_to_csr(uft_ix, risk_enter, risk_exit, nuft)
                    self._efron_pre_csr = (
                        enter_ptr,
                        enter_ind,
                        exit_ptr,
                        exit_ind,
                        fail_ptr,
                        fail_ind,
                        first_idx_uft,
                        nuft,
                    )
                    self._efron_pre_csr_gpu = (
                        cp.asarray(enter_ptr, dtype=cp.int32),
                        cp.asarray(enter_ind, dtype=cp.int32),
                        cp.asarray(exit_ptr, dtype=cp.int32),
                        cp.asarray(exit_ind, dtype=cp.int32),
                        cp.asarray(fail_ptr, dtype=cp.int32),
                        cp.asarray(fail_ind, dtype=cp.int32),
                        cp.asarray(first_idx_uft, dtype=cp.int32),
                        int(nuft),
                    )
                except Exception:
                    self._efron_pre_csr = None
                    self._efron_pre_csr_gpu = None
            else:
                self._efron_pre = None
                self._efron_pre_csr = None
                self._efron_pre_csr_gpu = None
            try:
                _, uft_ix, _, _, nuft, _ = _unpack_efron_pre6(efron_pre)
                n_events = int(torch.sum(event_sorted).item())
                avg_tie = float(n_events / max(1, int(nuft)))
            except Exception:
                avg_tie = 1.0
        else:
            self._efron_pre = None
            self._efron_all_singletons = False
            self._efron_pre_csr = None
            self._efron_pre_csr_gpu = None
            first_idx_uft, counts_uft = self._breslow_unique_failure_groups(
                time_sorted.cpu().numpy(), event_sorted.cpu().numpy()
            )
            self._breslow_pre = (first_idx_uft, counts_uft)
            self._breslow_pre_torch = (
                torch.tensor(first_idx_uft, dtype=torch.int32, device=torch_device),
                torch.tensor(counts_uft, dtype=torch.int32, device=torch_device),
            )
            if entry_sorted is not None:
                # Entry path: avoid stale index cache drift across different sort permutations.
                self._entry_fail_groups_torch = None
                self._entry_fail_times_torch = None
                self._entry_order_torch = None
                self._entry_add_end_np_torch = None
                self._entry_rem_end_np_torch = None
            else:
                self._entry_fail_groups_torch = None
                self._entry_fail_times_torch = None
                self._entry_order_torch = None
                self._entry_add_end_np_torch = None
                self._entry_rem_end_np_torch = None
            n_events = int(torch.sum(event_sorted).item())
            avg_tie = float(n_events / max(1, int(len(counts_uft))))

        # Initialize coefficients on Torch device (supports warm-start path in CV)
        if init_coef is None:
            beta = torch.zeros(n_features, dtype=torch.float64, device=torch_device)
        else:
            beta = torch.as_tensor(init_coef, dtype=torch.float64, device=torch_device).reshape(-1)
            if int(beta.shape[0]) != int(n_features):
                raise ValueError("init_coef must have shape (n_features,)")

        # Compute null log-likelihood on Torch
        entry_ctx_torch = None
        if entry_sorted is not None:
            _ctx = self._build_entry_ctx_torch(time_sorted, event_sorted, entry_sorted, torch_device)
            event_idx_ctx = _ctx[5]
            entry_ctx_torch = (
                _ctx[0],
                _ctx[1],
                _ctx[2],
                _ctx[3],
                X_sorted.index_select(0, _ctx[0]).contiguous(),
                X_sorted.contiguous(),
                event_idx_ctx,
                torch.sum(X_sorted.index_select(0, event_idx_ctx), dim=0),
                _ctx[6],
            )
        loglik_null_torch = self._compute_log_likelihood_torch(
            torch.zeros(n_features, dtype=torch.float64, device=torch_device),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
            entry=entry_sorted,
            entry_ctx=entry_ctx_torch,
        )

        # Newton-Raphson optimization on Torch with L2 penalty
        penalty = float(self.penalty) if hasattr(self, 'penalty') else 0.0
        use_penalty = penalty > 0.0
        diag_idx = torch.arange(n_features, dtype=torch.long, device=torch_device) if use_penalty else None

        # Newton-Raphson optimization on Torch
        iteration = 0
        loglik_torch = None
        current_obj = None
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian on Torch
            grad, hess, aux_stats = self._compute_gradient_hessian_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre, return_aux=True, entry=entry_sorted, entry_ctx=entry_ctx_torch
            )

            # Add penalty terms: gradient -= 2*penalty*beta, hessian -= 2*penalty*I
            if use_penalty:
                grad = grad - 2 * penalty * beta
                hess[diag_idx, diag_idx] -= 2 * penalty

            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_torch(hess, grad)
            step = 1.0
            accepted_step = True
            if entry_sorted is not None:
                if current_obj is None:
                    old_ll = self._compute_log_likelihood_torch_from_stats(
                        aux_stats[0], aux_stats[1], aux_stats[2], time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_torch
                    )
                    if use_penalty:
                        old_ll = old_ll - penalty * torch.sum(beta * beta)
                    current_obj = old_ll
                else:
                    old_ll = current_obj
                new_beta = beta - delta
                new_ll = self._compute_log_likelihood_torch(
                    new_beta, X_sorted, time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_torch
                )
                if use_penalty:
                    new_ll = new_ll - penalty * torch.sum(new_beta * new_beta)
                if float((new_ll - old_ll).item()) <= -1e-8:
                    step = 0.5
                    accepted = False
                    for _ in range(20):
                        trial_beta = beta - step * delta
                        trial_ll = self._compute_log_likelihood_torch(
                            trial_beta, X_sorted, time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_torch
                        )
                        if use_penalty:
                            trial_ll = trial_ll - penalty * torch.sum(trial_beta * trial_beta)
                        if float((trial_ll - old_ll).item()) > -1e-8:
                            beta = trial_beta
                            current_obj = trial_ll
                            accepted = True
                            break
                        step *= 0.5
                    if not accepted:
                        accepted_step = False
                else:
                    beta = new_beta
                    current_obj = new_ll
            else:
                beta = beta - delta

            # Check convergence
            if entry_sorted is not None:
                delta_norm = float(torch.linalg.norm(delta).item())
                if accepted_step and delta_norm * step < self.tol:
                    self._converged = True
                    loglik_torch = self._compute_log_likelihood_torch(
                        beta, X_sorted, time_sorted, event_sorted, efron_pre, entry=entry_sorted, entry_ctx=entry_ctx_torch
                    )
                    break
            else:
                grad_norm = float(torch.linalg.norm(grad).item())
                delta_norm = float(torch.linalg.norm(delta).item())
                if accepted_step and grad_norm < max(self.tol * 10.0, 1e-8) and delta_norm * step < self.tol:
                    self._converged = True
                    eta_cur, exp_eta_cur, risk_sum_cur = aux_stats
                    loglik_torch = self._compute_log_likelihood_torch_from_stats(
                        eta_cur, exp_eta_cur, risk_sum_cur, time_sorted, event_sorted, efron_pre, entry=entry_sorted
                    )
                    break

        # Compute final log-likelihood on Torch unless already obtained.
        if loglik_torch is None:
            loglik_torch = self._compute_log_likelihood_torch(
                beta, X_sorted, time_sorted, event_sorted, efron_pre
                , entry=entry_sorted, entry_ctx=entry_ctx_torch
            )

        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = beta.cpu().numpy()
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(loglik_null_torch.item())
        self._log_likelihood = float(loglik_torch.item())
        if self.compute_cindex:
            cindex_torch = self._compute_cindex_torch(X_sorted, time_sorted, event_sorted, beta)
            self._cindex = float(cindex_torch.item())
        else:
            self._cindex = None

        # Inference: nonrobust on Torch, other types fall back to CPU
        if self.compute_inference:
            if self.cov_type == "nonrobust":
                try:
                    info = -hess
                    var_torch = torch.linalg.solve(info, torch.eye(info.shape[0], dtype=info.dtype, device=torch_device))
                except Exception:
                    var_torch = torch.linalg.pinv(-hess)
                bse_torch = torch.sqrt(torch.maximum(torch.diag(var_torch), torch.tensor(0.0, dtype=torch.float64, device=torch_device)))
                z_torch = beta / (bse_torch + 1e-30)
                p_torch = torch.minimum(torch.tensor(1.0, device=torch_device), 2.0 * norm.sf(torch.abs(z_torch)))
                z_crit = norm.ppf(0.975)
                ci_torch = torch.stack([beta - z_crit * bse_torch, beta + z_crit * bse_torch], dim=1)

                self._bse = bse_torch.cpu().numpy()
                self._zvalues = z_torch.cpu().numpy()
                self._pvalues = p_torch.cpu().numpy()
                self._conf_int = ci_torch.cpu().numpy()
                self._var_matrix = np.diag(np.square(self._bse))
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.solve(self._var_matrix, np.eye(self._var_matrix.shape[0]))
                    self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
                except np.linalg.LinAlgError:
                    self._wald_test_stat = np.nan
                self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
                self._score_test_stat = np.nan
                self._score_test_pvalue = np.nan
                # Compute baseline hazard on Torch
                self._compute_baseline_hazard_torch(X_sorted, time_sorted, event_sorted, beta)
            else:
                # For hc0/hc1/cluster, use CPU inference path
                self._compute_inference_cpu(X_sorted.cpu().numpy(), time_sorted.cpu().numpy(), event_sorted.cpu().numpy(),
                                           cluster_sorted.cpu().numpy() if cluster_sorted is not None else None)
                self._baseline_hazard = None
                self._baseline_cumulative_hazard = None
                self._unique_times = None
        else:
            self._var_matrix = None
            self._bse = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            self._score_test_stat = None
            self._score_test_pvalue = None
            self._wald_test_stat = None
            self._wald_test_pvalue = None
            self._lr_test_stat = None
            self._lr_test_pvalue = None
            self._baseline_hazard = None
            self._baseline_cumulative_hazard = None
            self._unique_times = None
        self._cleanup_torch_memory()

    def _compute_log_likelihood(self, beta, X, time, event, efron_pre=None, entry=None):
        """Compute log partial likelihood (Breslow/Efron tie handling)."""
        eta = X @ beta
        eta_eff = eta
        if entry is not None and self.ties == "breslow":
            eta_eff = eta - np.max(eta)
        # Note: We do NOT center eta here. While centering prevents exp overflow,
        # it introduces a beta-dependent shift that complicates numeric gradient verification.
        # In practice, exp(eta) overflow is rare when beta is near convergence.
        exp_eta = np.exp(eta_eff)

        # Risk set suffix sums for standard (no-entry) path.
        risk_sum = np.cumsum(exp_eta[::-1])[::-1] if entry is None else None

        event_mask = event == 1
        if not np.any(event_mask):
            return 0.0

        if self.ties == "breslow":
            if entry is not None:
                fail_groups = getattr(self, "_entry_fail_groups_np", None)
                add_end_np = getattr(self, "_entry_add_end_np", None)
                rem_end_np = getattr(self, "_entry_rem_end_np", None)
                order_np = getattr(self, "_entry_order_np", None)
                if (
                    fail_groups is None
                    or add_end_np is None
                    or rem_end_np is None
                    or order_np is None
                ):
                    event_idx = np.flatnonzero(event_mask)
                    event_times = time[event_idx]
                    uft_np, inv_np = np.unique(event_times, return_inverse=True)
                    fail_groups = [
                        event_idx[inv_np == g].astype(np.int64, copy=False)
                        for g in range(len(uft_np))
                    ]
                    order_np = np.argsort(np.asarray(entry, dtype=np.float64)).astype(np.int64, copy=False)
                    add_end_np = np.searchsorted(
                        np.asarray(entry, dtype=np.float64)[order_np], uft_np, side="left"
                    ).astype(np.int64, copy=False)
                    rem_end_np = np.searchsorted(time, uft_np, side="left").astype(np.int64, copy=False)

                s0 = 0.0
                add_ptr = 0
                rem_ptr = 0
                ll = 0.0
                for g, fail_idx in enumerate(fail_groups):
                    add_end = int(add_end_np[g])
                    if add_end > add_ptr:
                        idx_add = order_np[add_ptr:add_end]
                        s0 += float(np.sum(exp_eta[idx_add]))
                        add_ptr = add_end
                    rem_end = int(rem_end_np[g])
                    if rem_end > rem_ptr:
                        s0 -= float(np.sum(exp_eta[rem_ptr:rem_end]))
                        rem_ptr = rem_end
                    d_t = int(fail_idx.shape[0])
                    if d_t <= 0:
                        continue
                    s0_safe = max(s0, 1e-300)
                    ll += float(np.sum(eta_eff[fail_idx]) - d_t * np.log(s0_safe))
                return float(ll)

            # l(β) = sum_i(eta_i) - sum_t(d_t * log(S0(t)))
            breslow_pre = getattr(self, "_breslow_pre", None)
            if (
                breslow_pre is not None
                and len(breslow_pre) == 2
                and breslow_pre[0].size > 0
            ):
                first_idx = breslow_pre[0].astype(np.int64, copy=False)
                counts = breslow_pre[1].astype(np.float64, copy=False)
            else:
                event_times = time[event_mask]
                uft, counts_i = np.unique(event_times, return_counts=True)
                first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
                counts = counts_i.astype(np.float64)
            risk_at = risk_sum[first_idx]
            # With centering: ll = sum(eta_i - eta_max) - sum(d_t * log(S0(t) * exp(-eta_max)))
            #              = sum(eta_i) - n_events*eta_max - sum(d_t * (log(S0(t)) - eta_max))
            #              = sum(eta_i) - n_events*eta_max - sum(d_t * log(S0(t))) + n_events*eta_max
            #              = sum(eta_i) - sum(d_t * log(S0(t)))  [eta_max cancels]
            return float(np.sum(eta_eff[event_mask]) - np.sum(counts * np.log(risk_at)))

        # ---- Efron ----
        ll = 0.0
        if efron_pre is not None:
            uft, uft_ix, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)

            # Sum of eta for all events (centering cancels out, use original eta)
            all_eta_sum = 0.0
            all_log_denom_sum = 0.0

            for g in range(nuft):
                ix_ev = uft_ix[g]
                d = len(ix_ev)
                if d == 0:
                    continue
                first_idx = (
                    int(first_idx_uft[g])
                    if first_idx_uft is not None
                    else int(np.searchsorted(time, uft[g], side="left"))
                )
                risk_at_t = risk_sum[first_idx]
                sum_events = float(np.sum(exp_eta[ix_ev]))
                all_eta_sum += float(np.sum(eta[ix_ev]))

                # Vectorized log denominator sum
                # Pre-compute k/d values to avoid repeated division
                k_vals = np.arange(d, dtype=np.float64)
                denom = risk_at_t - (k_vals / d) * sum_events
                all_log_denom_sum += float(np.sum(np.log(np.maximum(denom, 1e-300))))

            return float(all_eta_sum - all_log_denom_sum)

        # No precomputation: group event rows by unique failure times (vectorized).
        event_idx = np.flatnonzero(event_mask)
        event_times = time[event_idx]
        uft, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
        first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
        risk_at = risk_sum[first_idx]

        sum_events = np.bincount(inv, weights=exp_eta[event_idx], minlength=len(uft)).astype(np.float64)
        sum_eta_events = np.bincount(inv, weights=eta[event_idx], minlength=len(uft)).astype(np.float64)

        # Vectorized log-likelihood computation
        ll = float(np.sum(sum_eta_events))

        # For each unique failure time, compute sum of log denominators
        max_d = int(np.max(counts)) if len(counts) > 0 else 0
        if max_d > 0:
            # Create k matrix: (n_uft, max_d) where each row has [0/d, 1/d, ..., (d-1)/d]
            # Use broadcasting with careful masking for different d values
            k_matrix = np.arange(max_d, dtype=np.float64) / np.arange(1, max_d + 1, dtype=np.float64)[:, np.newaxis]
            # This is complex; fall back to loop for correctness
            for g in range(len(uft)):
                d = int(counts[g])
                if d == 0:
                    continue
                k = np.arange(d, dtype=np.float64) / d
                denom = risk_at[g] - k * sum_events[g]
                ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))
        else:
            for g in range(len(uft)):
                d = int(counts[g])
                if d == 0:
                    continue
                k = np.arange(d, dtype=np.float64) / d
                denom = risk_at[g] - k * sum_events[g]
                ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))

        return float(ll)
    
    def _solve_newton_delta_gpu(self, hess, grad, cp, eye_cache=None):
        """Newton step delta = inv(hess) @ grad; prefer SPD solve on (-hess) with light jitter."""
        p = int(hess.shape[0])
        try:
            H = -hess
            eps = 1e-11 * (cp.max(cp.abs(cp.diag(H))) + 1.0)
            jitter_eye = eye_cache if eye_cache is not None else cp.eye(p, dtype=cp.float64)
            H = H + eps * jitter_eye
            # Fast path: SPD solve via Cholesky is usually faster than generic solve.
            try:
                L = cp.linalg.cholesky(H)
                y = cp.linalg.solve(L, grad)
                x = cp.linalg.solve(L.T, y)
                return -x
            except Exception:
                return -cp.linalg.solve(H, grad)
        except Exception:
            try:
                return cp.linalg.solve(hess, grad)
            except Exception:
                return cp.linalg.lstsq(hess, grad, rcond=None)[0].flatten()

    def _compute_log_likelihood_gpu(self, beta, X, time, event, efron_pre=None, entry=None, entry_ctx=None):
        """Compute log partial likelihood on GPU."""
        import cupy as cp

        eta = X @ beta
        exp_eta = cp.exp(eta)
        # Entry+breslow path does not consume risk_sum; skip the cumsum to
        # reduce per-evaluation overhead during line-search probes.
        risk_sum = None if entry is not None else cp.cumsum(exp_eta[::-1])[::-1]
        return self._compute_log_likelihood_gpu_from_stats(
            eta, exp_eta, risk_sum, time, event, efron_pre, entry=entry, entry_ctx=entry_ctx
        )

    def _build_entry_ctx_gpu(self, time, event, entry, cp):
        """Build entry-time grouped indexing context for a specific sorted GPU view."""
        event_mask = event == 1
        event_idx = cp.where(event_mask)[0]
        evt_t = cp.asnumpy(time[event_idx])
        if evt_t.size == 0:
            return (
                cp.zeros((0,), dtype=cp.int64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                cp.zeros((0,), dtype=cp.int64),
                cp.zeros((0,), dtype=cp.int64),
                np.zeros((1,), dtype=np.int64),
            )
        uft_np, d_counts = np.unique(evt_t, return_counts=True)
        d_counts = d_counts.astype(np.float64, copy=False)
        entry_order = cp.argsort(entry)
        entry_sorted_np = cp.asnumpy(entry[entry_order])
        time_np = cp.asnumpy(time)
        add_end_np = np.searchsorted(entry_sorted_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_end_np = np.searchsorted(time_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_order = cp.arange(int(time.shape[0]), dtype=cp.int64)
        event_idx = event_idx.astype(cp.int64, copy=False)
        fail_ptr = np.empty(d_counts.shape[0] + 1, dtype=np.int64)
        fail_ptr[0] = 0
        fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
        return (entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr)

    def _compute_log_likelihood_gpu_from_stats(
        self, eta, exp_eta, risk_sum, time, event, efron_pre=None, entry=None, entry_ctx=None
    ):
        """Compute log partial likelihood on GPU with precomputed Efron stats."""
        import cupy as cp

        ll = cp.array(0.0, dtype=cp.float64)
        event_mask = event == 1

        if not cp.any(event_mask):
            return ll

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, _rem_order, event_idx, fail_ptr = self._build_entry_ctx_gpu(
                    time, event, entry, cp
                )
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else cp.where(event_mask)[0]
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None
            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                return cp.array(0.0, dtype=cp.float64)
            if fail_ptr is None:
                fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                fail_ptr[0] = 0
                fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)

            exp_entry = exp_eta[entry_order]
            exp_rem = exp_eta
            add_pref = cp.cumsum(exp_entry, axis=0)
            rem_pref = cp.cumsum(exp_rem, axis=0)
            s0_add = cp.zeros(n_groups, dtype=cp.float64)
            s0_rem = cp.zeros(n_groups, dtype=cp.float64)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = cp.asarray(add_end_np[mask_add] - 1, dtype=cp.int64)
                s0_add[cp.asarray(mask_add)] = add_pref[idx_add]
            if np.any(mask_rem):
                idx_rem = cp.asarray(rem_end_np[mask_rem] - 1, dtype=cp.int64)
                s0_rem[cp.asarray(mask_rem)] = rem_pref[idx_rem]
            s0_vec = cp.maximum(s0_add - s0_rem, 1e-300)
            event_eta = eta[event_idx]

            if self.ties == "breslow":
                d_vec = cp.asarray(d_counts, dtype=cp.float64)
                return cp.sum(event_eta) - cp.sum(d_vec * cp.log(s0_vec))

            ll = cp.sum(event_eta)
            event_exp = exp_eta[event_idx]
            for g in range(n_groups):
                d = int(d_counts[g])
                if d <= 0:
                    continue
                st = int(fail_ptr[g])
                ed = int(fail_ptr[g + 1])
                ef = cp.sum(event_exp[st:ed])
                base = s0_vec[g]
                for k in range(d):
                    denom = cp.maximum(base - (float(k) / float(d)) * ef, 1e-300)
                    ll = ll - cp.log(denom)
            return ll

        if self.ties == 'breslow':
            # Vectorized Breslow using cached failure groups to avoid
            # Python loops and host-device sync in GPU hot path.
            breslow_pre_gpu = getattr(self, "_breslow_pre_gpu", None)
            if (
                breslow_pre_gpu is not None
                and len(breslow_pre_gpu) == 2
                and int(breslow_pre_gpu[0].size) > 0
            ):
                first_idx_uft, counts_uft = breslow_pre_gpu
            else:
                uft, counts_uft = cp.unique(time[event_mask], return_counts=True)
                first_idx_uft = cp.searchsorted(time, uft, side="left")
                counts_uft = counts_uft.astype(cp.int32, copy=False)
            risk_at = risk_sum[first_idx_uft]
            return cp.sum(eta[event_mask]) - cp.sum(
                counts_uft.astype(cp.float64) * cp.log(risk_at)
            )
        
        # Efron: if all groups are singleton failures, Efron == Breslow.
        if getattr(self, "_efron_all_singletons", False):
            ep = efron_pre if efron_pre is not None else getattr(self, "_efron_pre", None)
            if ep is not None:
                _, _, _, _, nuft, first_idx_uft = _unpack_efron_pre6(ep)
                first_idx_uft = cp.asarray(first_idx_uft, dtype=cp.int32)
                counts_uft = cp.ones(int(nuft), dtype=cp.int32)
            else:
                uft, counts_uft = cp.unique(time[event_mask], return_counts=True)
                first_idx_uft = cp.searchsorted(time, uft, side="left")
                counts_uft = counts_uft.astype(cp.int32, copy=False)
            risk_at = risk_sum[first_idx_uft]
            return cp.sum(eta[event_mask]) - cp.sum(
                counts_uft.astype(cp.float64) * cp.log(risk_at)
            )

        # Efron: loop over cached failure groups (see `_cox_efron_cuda.compute_efron_loglik_raw`)
        if efron_pre is not None:
            try:
                csr_gpu = getattr(self, "_efron_pre_csr_gpu", None)
                if csr_gpu is not None:
                    from ._cox_efron_cuda import compute_efron_loglik_raw_csr

                    _, _, _, _, fail_ptr, fail_ind, first_idx_uft, nuft = csr_gpu
                    return compute_efron_loglik_raw_csr(
                        eta,
                        exp_eta,
                        risk_sum,
                        fail_ptr,
                        fail_ind,
                        first_idx_uft,
                        nuft,
                        cupy_module=cp,
                    )
            except Exception:
                pass

            from ._cox_efron_cuda import compute_efron_loglik_raw

            return compute_efron_loglik_raw(
                eta, exp_eta, risk_sum, time, efron_pre, cupy_module=cp
            )

        unique_times = cp.unique(time[event_mask])
        for t in unique_times:
            at_time_t = time == t
            events_at_t = at_time_t & event_mask
            d = int(cp.sum(events_at_t).item())
            
            if d == 0:
                continue
            
            risk_indices = cp.where(time >= t)[0]
            if risk_indices.size == 0:
                continue
            
            first_idx = risk_indices[0]
            risk_at_t = risk_sum[first_idx]
            sum_events = cp.sum(exp_eta[events_at_t])
            
            ll += cp.sum(eta[events_at_t])
            for k in range(d):
                ll -= cp.log(cp.maximum(risk_at_t - (k / d) * sum_events, 1e-300))
        
        return ll
    
    def _compute_gradient_hessian(self, beta, X, time, event, efron_pre=None, entry=None):
        """
        Gradient and Hessian of the log partial likelihood (same sign convention as statsmodels).

        Parameters
        ----------
        efron_pre : optional
            Output of `_efron_unique_failure_indices`; if None and ties='efron', it is recomputed.
            Pass the cached structure from `fit` to avoid O(n) Python work every Newton step.
        """
        n_samples, n_features = X.shape

        # Linear predictor
        eta = X @ beta
        eta_eff = eta
        if entry is not None and self.ties == "breslow":
            eta_eff = eta - np.max(eta)
        exp_eta = np.exp(eta_eff)

        risk_sum = np.cumsum(exp_eta[::-1])[::-1] if entry is None else None
        X_exp_eta = X * exp_eta[:, np.newaxis]
        risk_X_sum = np.cumsum(X_exp_eta[::-1], axis=0)[::-1] if entry is None else None

        if self.ties == 'breslow':
            event_mask = event == 1
            grad = np.zeros(n_features, dtype=np.float64)
            if entry is not None:
                fail_groups = getattr(self, "_entry_fail_groups_np", None)
                add_end_np = getattr(self, "_entry_add_end_np", None)
                rem_end_np = getattr(self, "_entry_rem_end_np", None)
                order_np = getattr(self, "_entry_order_np", None)
                if (
                    fail_groups is None
                    or add_end_np is None
                    or rem_end_np is None
                    or order_np is None
                ):
                    event_idx = np.flatnonzero(event_mask)
                    event_times = time[event_idx]
                    uft_np, inv_np = np.unique(event_times, return_inverse=True)
                    fail_groups = [
                        event_idx[inv_np == g].astype(np.int64, copy=False)
                        for g in range(len(uft_np))
                    ]
                    order_np = np.argsort(np.asarray(entry, dtype=np.float64)).astype(np.int64, copy=False)
                    add_end_np = np.searchsorted(
                        np.asarray(entry, dtype=np.float64)[order_np], uft_np, side="left"
                    ).astype(np.int64, copy=False)
                    rem_end_np = np.searchsorted(time, uft_np, side="left").astype(np.int64, copy=False)

                hess = np.zeros((n_features, n_features), dtype=np.float64)
                s0 = 0.0
                s1 = np.zeros(n_features, dtype=np.float64)
                s2 = np.zeros((n_features, n_features), dtype=np.float64)
                add_ptr = 0
                rem_ptr = 0
                for g, fail_idx in enumerate(fail_groups):
                    add_end = int(add_end_np[g])
                    if add_end > add_ptr:
                        idx_add = order_np[add_ptr:add_end]
                        x_add = X[idx_add]
                        w_add = exp_eta[idx_add]
                        wx_add = x_add * w_add[:, np.newaxis]
                        s0 += float(np.sum(w_add))
                        s1 += np.sum(wx_add, axis=0)
                        s2 += wx_add.T @ x_add
                        add_ptr = add_end
                    rem_end = int(rem_end_np[g])
                    if rem_end > rem_ptr:
                        x_rem = X[rem_ptr:rem_end]
                        w_rem = exp_eta[rem_ptr:rem_end]
                        wx_rem = x_rem * w_rem[:, np.newaxis]
                        s0 -= float(np.sum(w_rem))
                        s1 -= np.sum(wx_rem, axis=0)
                        s2 -= wx_rem.T @ x_rem
                        rem_ptr = rem_end
                    d_t = int(fail_idx.shape[0])
                    if d_t <= 0:
                        continue
                    d_t_f = float(d_t)
                    grad += np.sum(X[fail_idx], axis=0)
                    s0_safe = max(s0, 1e-300)
                    if s0 <= 1e-15:
                        continue
                    ex = s1 / s0_safe
                    grad -= d_t_f * ex
                    hess -= d_t_f * (s2 / s0_safe - np.outer(ex, ex))
                return grad, hess

            first_idx = np.array([], dtype=np.int64)
            counts = np.array([], dtype=np.float64)
            if np.any(event_mask):
                breslow_pre = getattr(self, "_breslow_pre", None)
                if (
                    breslow_pre is not None
                    and len(breslow_pre) == 2
                    and breslow_pre[0].size > 0
                ):
                    first_idx = breslow_pre[0].astype(np.int64, copy=False)
                    counts = breslow_pre[1].astype(np.float64, copy=False)
                else:
                    event_times = time[event_mask]
                    uft, counts_i = np.unique(event_times, return_counts=True)
                    first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
                    counts = counts_i.astype(np.float64)

                sum_X_events = np.sum(X[event_mask], axis=0)
                E_X = risk_X_sum[first_idx] / risk_sum[first_idx][:, np.newaxis]
                grad = sum_X_events - np.sum(E_X * counts[:, np.newaxis], axis=0)

            hess = self._compute_hessian_breslow_fast(
                X, time, event, risk_sum, risk_X_sum, exp_eta, first_idx, counts
            )
        else:
            # Efron: prefer Cython core if available; fall back to Python implementation
            # for environments without compiled extension or unexpected runtime issues.
            # Shift eta by a constant for numerical stability in exp(eta). This does not
            # change Efron gradient/Hessian because terms are scale-invariant.
            eta_efron = eta - np.max(eta)
            if HAS_CYTHON_EFRON and efron_pre is not None:
                try:
                    uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
                    grad, hess = _efron_grad_hess_cython(
                        eta_efron, X, risk_enter, risk_exit, uft_ix, nuft
                    )
                    # Align sign convention with existing CPU Efron backward path.
                    hess = -hess
                    if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
                        raise FloatingPointError("non-finite Cython Efron grad/hess")
                except Exception:
                    from ._cox_efron_cy import efron_grad_hess_python
                    uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
                    grad, hess = efron_grad_hess_python(
                        eta_efron, X, risk_enter, risk_exit, uft_ix, nuft
                    )
                    hess = -hess
                    if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
                        grad, hess = self._compute_gradient_hessian_efron_backward(
                            beta, X, time, event, efron_pre
                        )
            else:
                grad, hess = self._compute_gradient_hessian_efron_backward(
                    beta, X, time, event, efron_pre
                )

        return grad, hess

    def _compute_hessian_breslow_fast(
        self,
        X,
        time,
        event,
        risk_sum,
        risk_X_sum,
        exp_eta,
        first_idx=None,
        counts=None,
    ):
        """Compute Breslow Hessian with an auto-selected CPU strategy."""
        event_mask = event == 1
        if not np.any(event_mask):
            return np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

        # Group tied events by unique failure times to share the same R(t)
        # denominator across all events at time t (Breslow ties).
        if first_idx is None or counts is None or len(first_idx) == 0:
            breslow_pre = getattr(self, "_breslow_pre", None)
            if (
                breslow_pre is not None
                and len(breslow_pre) == 2
                and breslow_pre[0].size > 0
            ):
                first_idx = breslow_pre[0].astype(np.int64, copy=False)
                counts = breslow_pre[1].astype(np.float64, copy=False)
            else:
                event_times = time[event_mask]
                uft, counts_i = np.unique(event_times, return_counts=True)
                first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
                counts = counts_i.astype(np.float64)

        # Two CPU kernels are kept intentionally:
        # 1) Tensor path: higher memory, but can be faster for small p / few groups.
        # 2) Incremental path: lower memory traffic for larger (n, p).
        p = int(X.shape[1])
        n_groups = int(len(first_idx))
        if p <= 24 and n_groups <= 512:
            return self._compute_hessian_breslow_tensor_grouped(
                X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
            )
        return self._compute_hessian_breslow_incremental_grouped(
            X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
        )

    def _compute_hessian_breslow_tensor_grouped(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """Grouped Breslow Hessian using explicit (n, p, p) tensor moments."""
        x2_weighted = np.einsum("ni,nj,n->nij", X, X, exp_eta)
        risk_X2_sum = np.cumsum(x2_weighted[::-1], axis=0)[::-1]
        risk_sum_at = risk_sum[first_idx]
        E_X = risk_X_sum[first_idx] / risk_sum_at[:, np.newaxis]
        E_XX = risk_X2_sum[first_idx] / risk_sum_at[:, np.newaxis, np.newaxis]
        centered = E_XX - np.einsum("ni,nj->nij", E_X, E_X)
        return -np.sum(centered * counts[:, np.newaxis, np.newaxis], axis=0)

    def _compute_hessian_breslow_incremental_grouped(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """Grouped Breslow Hessian with incremental risk-set second moments."""
        # risk_X2 tracks sum_{j in current risk set} exp_eta[j] * x_j x_j^T.
        X_exp = X * exp_eta[:, np.newaxis]
        risk_X2 = X_exp.T @ X

        hess = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        prev_idx = 0
        for g in range(len(first_idx)):
            idx = int(first_idx[g])
            if idx > prev_idx:
                blk = slice(prev_idx, idx)
                # Remove rows that are no longer in risk set.
                risk_X2 -= X_exp[blk].T @ X[blk]
                prev_idx = idx

            rs = float(risk_sum[idx])
            if rs <= 0.0:
                continue
            ex = risk_X_sum[idx] / rs
            exx = risk_X2 / rs
            hess -= counts[g] * (exx - np.outer(ex, ex))

        return hess

    def _compute_hessian_breslow_incremental_grouped_cupy(
        self, X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    ):
        """CuPy Breslow Hessian — vectorized via cumsum of outer products.

        O(n·p²) memory (acceptable on 16GB P100), zero Python loop over groups.
        """
        import cupy as cp

        n, p = int(X.shape[0]), int(X.shape[1])
        nuft = int(first_idx.shape[0])
        if nuft == 0:
            return cp.zeros((p, p), dtype=cp.float64)

        X_exp = X * exp_eta[:, cp.newaxis]
        total = X_exp.T @ X  # (p, p)

        risk_at = risk_sum[first_idx]
        E_X = risk_X_sum[first_idx] / risk_at[:, None]
        sc = counts / risk_at  # (nuft,)


        # Sum weighted risk-set second moments without materializing an
        # O(n * p * p) tensor. Observation i contributes to every prefix
        # whose failure-time start is strictly after i.
        sc_at_start = torch.zeros(
            n_samples, dtype=torch.float64, device=beta.device
        )
        sc_at_start.index_add_(0, first_idx, sc)
        suffix_sc = torch.flip(
            torch.cumsum(torch.flip(sc_at_start, dims=[0]), dim=0),
            dims=[0],
        )
        prefix_weights = suffix_sc - sc_at_start
        weighted_prefix = X_exp.transpose(0, 1) @ (
            X * prefix_weights.unsqueeze(1)
        )

        hess = -torch.sum(sc) * total + weighted_prefix
        hess += torch.einsum(
            "g,gi,gj->ij", weights, E_X_at_uft, E_X_at_uft
        )

        if return_aux:
            return grad, hess, (eta, exp_eta, risk_sum)
        return grad, hess

    def _s2_weighted_update_torch_blocked(self, s2, x, w, block_size, sign=1.0):
        """Blocked update for large slices: s2 += sign * X^T (X * w)."""
        s2_fn = self._get_entry_s2_torch_fn()

        n = int(x.shape[0])
        if n <= 0:
            return s2
        for st in range(0, n, block_size):
            ed = min(st + block_size, n)
            xb = x[st:ed]
            wb = w[st:ed]
            s2 = s2 + sign * s2_fn(xb, wb)
        return s2

    def _get_entry_s2_torch_fn(self):
        """Build/cache torch or torch.compile function for weighted X^T X."""
        fn = getattr(self, "_entry_s2_torch_fn", None)
        if fn is not None:
            return fn
        import torch

        def _s2_core(x, w):
            return x.transpose(0, 1) @ (x * w.unsqueeze(1))

        use_compile = (
            os.environ.get("STATGPU_ENTRY_S2_COMPILE_TORCH", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if use_compile and hasattr(torch, "compile"):
            mode = os.environ.get("STATGPU_ENTRY_S2_COMPILE_MODE", "default")
            try:
                fn = torch.compile(_s2_core, dynamic=True, fullgraph=False, mode=mode)
            except Exception:
                fn = _s2_core
        else:
            fn = _s2_core
        self._entry_s2_torch_fn = fn
        return fn

    def _compute_cindex_torch(self, X, time, event, beta):
        """Compute concordance index (C-index) on Torch."""
        import torch

        # Linear predictor (risk score)
        risk_score = X @ beta

        n = len(time)
        event_mask = (event == 1)

        if torch.sum(event_mask) == 0:
            return torch.tensor(0.5, dtype=torch.float64, device=beta.device)

        # Use chunked vectorized approach for memory efficiency
        event_idx = torch.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return torch.tensor(float("nan"), dtype=torch.float64, device=beta.device)

        concordant = torch.tensor(0, dtype=torch.int64, device=beta.device)
        permissible = torch.tensor(0, dtype=torch.int64, device=beta.device)
        tied_risk = torch.tensor(0, dtype=torch.int64, device=beta.device)

        # Chunk size for memory efficiency (~128 MB per batch matrix)
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time[idx_chunk][:, None]
            risk_i = risk_score[idx_chunk][:, None]
            time_j = time[None, :]
            risk_j = risk_score[None, :]
            event_j = event[None, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))
            # Exclude self-comparisons
            chunk_indices = torch.arange(end - start, device=beta.device)
            perm[chunk_indices, idx_chunk] = False

            concordant += torch.sum(perm & (risk_i > risk_j))
            tied_risk += torch.sum(perm & (risk_i == risk_j))
            permissible += torch.sum(perm)

        if permissible > 0:
            return (concordant.to(torch.float64) + 0.5 * tied_risk.to(torch.float64)) / permissible.to(torch.float64)
        else:
            return torch.tensor(float("nan"), dtype=torch.float64, device=beta.device)

    def _compute_inference_cpu(self, X, time, event, cluster=None):
        """Compute standard errors, z-values, p-values, and confidence intervals."""
        n_features = X.shape[1]

        # Keep inference self-contained (no nested external model fitting),
        # so runtime reflects this implementation directly.
        
        # Compute information matrix (negative Hessian at MLE)
        _, hess = self._compute_gradient_hessian(
            self.coef_, X, time, event, getattr(self, "_efron_pre", None), entry=getattr(self, "_entry", None)
        )
        
        # Bread matrix from observed information.
        try:
            bread = np.linalg.solve(-hess, np.eye(n_features))
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(-hess)

        if self.cov_type == "nonrobust":
            self._var_matrix = bread
        elif self.cov_type == "cluster":
            if cluster is None:
                raise ValueError("cov_type='cluster' requires cluster ids in fit(..., cluster=...)")
            cluster = np.asarray(cluster)
            score_resid = self._compute_robust_score_residuals(X, time, event)
            uniq = np.unique(cluster)
            meat = np.zeros((n_features, n_features), dtype=np.float64)
            for g in uniq:
                u_g = np.sum(score_resid[cluster == g], axis=0)
                meat += np.outer(u_g, u_g)
            self._var_matrix = bread @ meat @ bread
        else:
            score_resid = self._compute_robust_score_residuals(X, time, event)
            meat = score_resid.T @ score_resid
            self._var_matrix = bread @ meat @ bread
            if self.cov_type == "hc1":
                n = X.shape[0]
                k = X.shape[1]
                if n > k:
                    self._var_matrix = self._var_matrix * (n / (n - k))
        
        # Standard errors
        self._bse = np.sqrt(np.maximum(np.diag(self._var_matrix), 0.0))

        # z-values (add epsilon to avoid division by zero)
        self._zvalues = self.coef_ / (self._bse + 1e-30)
        
        # p-values (two-sided)
        self._pvalues = 2 * (1 - stats.norm.cdf(np.abs(self._zvalues)))
        
        # 95% confidence intervals
        alpha = 0.05
        z_crit = stats.norm.ppf(1 - alpha / 2)
        self._conf_int = np.column_stack([
            self.coef_ - z_crit * self._bse,
            self.coef_ + z_crit * self._bse
        ])
        
        # Wald test (global test that all coefficients are 0)
        try:
            var_inv = np.linalg.solve(self._var_matrix, np.eye(n_features))
            self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
        except np.linalg.LinAlgError:
            self._wald_test_stat = np.nan
        self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
        
        # Likelihood ratio test
        self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
        self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
        
        # Score test (Rao's test) - computed at beta = 0
        ep = getattr(self, "_efron_pre", None)
        grad_0, _ = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep, entry=getattr(self, "_entry", None))
        try:
            _, hess_0 = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep, entry=getattr(self, "_entry", None))
            info_0 = -hess_0
            info_0_inv = np.linalg.solve(info_0, np.eye(n_features))
            self._score_test_stat = grad_0 @ info_0_inv @ grad_0
        except:
            self._score_test_stat = np.nan
        self._score_test_pvalue = 1 - stats.chi2.cdf(self._score_test_stat, n_features)

    def _compute_robust_score_residuals(self, X, time, event):
        """
        Per-observation contributions for sandwich (HC0/HC1/cluster).

        When `statsmodels` is available, uses `PHReg.score_residuals`, which
        follows the martingale / leverage construction used by statsmodels for
        cluster-robust covariance (same for both Breslow and Efron partial
        likelihood). This aligns robust SEs with statsmodels much more closely
        than the closed-form Breslow score residual or the fast Efron
        approximation.

        Falls back to `_compute_score_residuals_exact_breslow` (Breslow) or
        `_compute_score_residuals_fast` (Efron) when statsmodels is missing or
        raises.
        """
        sr = self._score_residuals_via_statsmodels_if_available(X, time, event)
        if sr is not None:
            return sr
        if self.ties == "breslow":
            return self._compute_score_residuals_exact_breslow(X, time, event)
        return self._compute_score_residuals_fast(X, time, event)

    def _compute_robust_score_residuals_gpu(self, X, time, event):
        """GPU robust score residuals using event-row approximation."""
        import cupy as cp

        eta = X @ cp.asarray(self.coef_)
        exp_eta = cp.exp(eta)
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1] + 1e-30
        risk_X_sum = cp.cumsum((X * exp_eta[:, cp.newaxis])[::-1], axis=0)[::-1]
        score_residuals = cp.zeros((X.shape[0], X.shape[1]), dtype=cp.float64)
        event_mask = event == 1
        score_residuals[event_mask] = X[event_mask] - risk_X_sum[event_mask] / risk_sum[event_mask, cp.newaxis]
        return score_residuals

    def _score_residuals_via_statsmodels_if_available(
        self, X: np.ndarray, time: np.ndarray, event: np.ndarray
    ):
        """Return statsmodels-style score residuals, or None if unavailable."""
        try:
            import statsmodels.duration.api as smd
        except Exception:
            return None
        try:
            model = smd.PHReg(time, X, status=event, ties=self.ties)
            sr = model.score_residuals(self.coef_)
            if sr.shape != (X.shape[0], X.shape[1]):
                return None
            # Undefined strata / risk-set rows are NaN in statsmodels; drop from meat.
            sr = np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
            return np.asarray(sr, dtype=np.float64)
        except Exception:
            return None

    def _compute_score_residuals_fast(self, X, time, event):
        """
        Fast approximate per-observation score residuals at fitted beta.

        Event-row approximation:
          u_i = x_i - E[X | R(t_i)] for event rows, 0 for censored rows.
        This is substantially faster for larger n.
        """
        n_samples, n_features = X.shape
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1] + 1e-30
        risk_X_sum = np.cumsum((X * exp_eta[:, np.newaxis])[::-1], axis=0)[::-1]
        u = np.zeros((n_samples, n_features), dtype=np.float64)
        # Vectorized: fill only event rows.
        event_mask = event == 1
        u[event_mask] = X[event_mask] - risk_X_sum[event_mask] / risk_sum[event_mask, np.newaxis]
        return u

    def _compute_score_residuals_exact_breslow(self, X, time, event):
        """
        Exact per-observation score residuals for Breslow ties in O(n p).

        u_j = I(event_j) * s_j - exp_eta_j * sum_{i<=j, event_i=1}(s_i / risk_sum_i),
        where s_i = x_i - E[X|R(t_i)].
        """
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1] + 1e-30
        risk_X_sum = np.cumsum((X * exp_eta[:, np.newaxis])[::-1], axis=0)[::-1]
        event_mask = (event == 1).astype(np.float64)
        s = X - (risk_X_sum / risk_sum[:, np.newaxis])
        a = (event_mask[:, np.newaxis] * s) / risk_sum[:, np.newaxis]
        csum_a = np.cumsum(a, axis=0)
        u = event_mask[:, np.newaxis] * s - exp_eta[:, np.newaxis] * csum_a
        return u
    
    def _compute_baseline_hazard(self, X, time, event):
        """Compute Breslow estimator of baseline hazard and survival function."""
        # Get unique event times
        event_mask = event == 1
        if not np.any(event_mask):
            self._unique_times = np.array([])
            self._baseline_hazard = np.array([])
            self._baseline_cumulative_hazard = np.array([])
            return
        
        unique_times = np.unique(time[event_mask])
        self._unique_times = unique_times
        
        # Linear predictor
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        
        # Compute baseline cumulative hazard using Breslow estimator
        cumulative_hazard = np.zeros(len(unique_times))
        
        for i, t in enumerate(unique_times):
            # Events at time t
            d_i = np.sum((time == t) & (event == 1))
            
            # Risk set at time t (all with time >= t)
            risk_set = time >= t
            risk_sum = np.sum(exp_eta[risk_set])
            
            # Breslow estimator contribution
            cumulative_hazard[i] = d_i / risk_sum
        
        # Cumulative sum
        self._baseline_cumulative_hazard = np.cumsum(cumulative_hazard)
        
        # Hazard (discrete)
        self._baseline_hazard = cumulative_hazard

    def _compute_baseline_hazard_gpu(self, X, time, event, beta):
        """Compute Breslow estimator of baseline hazard and survival function on GPU."""
        import cupy as cp

        # Get unique event times
        event_mask = event == 1
        if not cp.any(event_mask):
            self._unique_times = cp.array([])
            self._baseline_hazard = cp.array([])
            self._baseline_cumulative_hazard = cp.array([])
            return

        unique_times = cp.unique(time[event_mask])
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ beta
        exp_eta = cp.exp(eta)

        # Compute baseline cumulative hazard using Breslow estimator (vectorized)
        cumulative_hazard = cp.zeros(len(unique_times))

        # Vectorized computation using searchsorted
        # For each unique time, compute d_i / risk_sum
        for i, t in enumerate(unique_times):
            # Events at time t
            d_i = int(cp.sum((time == t) & (event == 1)))

            # Risk set at time t (all with time >= t)
            risk_set = time >= t
            risk_sum = cp.sum(exp_eta[risk_set])

            # Breslow estimator contribution
            cumulative_hazard[i] = d_i / risk_sum

        # Cumulative sum
        self._baseline_cumulative_hazard = cp.cumsum(cumulative_hazard)

        # Hazard (discrete)
        self._baseline_hazard = cumulative_hazard

        # Transfer to CPU for storage
        self._unique_times = cp.asnumpy(self._unique_times)
        self._baseline_hazard = cp.asnumpy(self._baseline_hazard)
        self._baseline_cumulative_hazard = cp.asnumpy(self._baseline_cumulative_hazard)

    def _compute_baseline_hazard_torch(self, X, time, event, beta):
        """Compute Breslow estimator of baseline hazard and survival function on Torch."""
        import torch

        # Get unique event times
        event_mask = event == 1
        if not torch.any(event_mask):
            self._unique_times = torch.tensor([], dtype=torch.float64, device=beta.device)
            self._baseline_hazard = torch.tensor([], dtype=torch.float64, device=beta.device)
            self._baseline_cumulative_hazard = torch.tensor([], dtype=torch.float64, device=beta.device)
            return

        unique_times = torch.unique(time[event_mask])
        self._unique_times = unique_times

        # Linear predictor
        eta = X @ beta
        exp_eta = torch.exp(eta)

        # Compute baseline cumulative hazard using Breslow estimator (vectorized)
        cumulative_hazard = torch.zeros(len(unique_times), dtype=torch.float64, device=beta.device)

        # Vectorized computation
        for i, t in enumerate(unique_times):
            # Events at time t
            d_i = int(torch.sum((time == t) & (event == 1)))

            # Risk set at time t (all with time >= t)
            risk_set = time >= t
            risk_sum = torch.sum(exp_eta[risk_set])

            # Breslow estimator contribution
            cumulative_hazard[i] = d_i / risk_sum

        # Cumulative sum
        self._baseline_cumulative_hazard = torch.cumsum(cumulative_hazard, dim=0)

        # Hazard (discrete)
        self._baseline_hazard = cumulative_hazard

        # Transfer to CPU for storage
        self._unique_times = self._unique_times.cpu().numpy()
        self._baseline_hazard = self._baseline_hazard.cpu().numpy()
        self._baseline_cumulative_hazard = self._baseline_cumulative_hazard.cpu().numpy()

    def _compute_cindex_gpu(self, X, time, event, beta):
        """Compute concordance index (C-index) on GPU using chunked vectorized approach."""
        import cupy as cp

        # Linear predictor (risk score) on GPU
        risk_score = X @ beta

        n = len(time)
        event_mask = (event == 1)

        if cp.sum(event_mask) == 0:
            return cp.array(0.5, dtype=cp.float64)

        # Use chunked vectorized approach for memory efficiency
        event_idx = cp.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return cp.array(float("nan"), dtype=cp.float64)

        concordant = cp.int64(0)
        permissible = cp.int64(0)
        tied_risk = cp.int64(0)

        # Chunk size for memory efficiency (~128 MB per batch matrix)
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time[idx_chunk][:, None]
            risk_i = risk_score[idx_chunk][:, None]
            time_j = time[None, :]
            risk_j = risk_score[None, :]
            event_j = event[None, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))
            # Exclude self-comparisons
            chunk_indices = cp.arange(end - start, dtype=cp.int64)
            perm[chunk_indices, idx_chunk] = False

            concordant += cp.sum(perm & (risk_i > risk_j))
            tied_risk += cp.sum(perm & (risk_i == risk_j))
            permissible += cp.sum(perm)

        if permissible > 0:
            return (concordant.astype(cp.float64) + 0.5 * tied_risk.astype(cp.float64)) / permissible.astype(cp.float64)
        else:
            return cp.array(float("nan"), dtype=cp.float64)
    
    def _compute_cindex(self):
        """
        Compute concordance index (C-index) using chunked vectorized NumPy.

        Replaces the O(n²) double Python loop with batched boolean matrix ops.
        Chunk size is chosen so each batch matrix stays within ~128 MB.
        """
        if self._X is None or self.coef_ is None:
            self._cindex = None
            return

        risk_score = self._X @ self.coef_
        time = self._time
        event = self._event
        n = len(time)

        event_idx = np.where(event == 1)[0]
        n_events = len(event_idx)

        if n_events == 0:
            self._cindex = np.nan
            return

        concordant = np.int64(0)
        permissible = np.int64(0)
        tied_risk   = np.int64(0)

        # Chunk so each (chunk × n) bool matrix is ≤ 128 MB.
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]          # (c,)

            time_i  = time[idx_chunk, np.newaxis]     # (c, 1)
            risk_i  = risk_score[idx_chunk, np.newaxis]
            time_j  = time[np.newaxis, :]             # (1, n)
            risk_j  = risk_score[np.newaxis, :]
            event_j = event[np.newaxis, :]

            # Permissible pairs: earlier time OR same time with j censored.
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))
            # Exclude self-comparisons.
            perm[np.arange(end - start), idx_chunk] = False

            concordant  += int(np.sum(perm & (risk_i > risk_j)))
            tied_risk   += int(np.sum(perm & (risk_i == risk_j)))
            permissible += int(np.sum(perm))

        if permissible > 0:
            self._cindex = (concordant + 0.5 * tied_risk) / permissible
        else:
            self._cindex = np.nan
    
    def summary(self):
        """Print summary table similar to R's summary(coxph())."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        
        print("=" * 80)
        print("                     Cox Proportional Hazards Model")
        print("=" * 80)
        print(f"Call:")
        print(f"  coxph(formula = Surv(time, event) ~ ., ties = '{self.ties}')")
        print()
        print(f"  n= {self._nobs}, number of events= {int(self._nevents)}")
        print(f"  covariance type= {self.cov_type}")
        print()
        if self.compute_inference and self._bse is not None:
            print(f"{'':<15} {'coef':>10} {'exp(coef)':>12} {'se(coef)':>10} {'z':>10} {'Pr(>|z|)':>10}")
            print("-" * 80)
            
            for i, name in enumerate(self._feature_names):
                print(f"{name:<15} {self.coef_[i]:>10.4f} {self.hazard_ratios_[i]:>12.4f} "
                      f"{self._bse[i]:>10.4f} {self._zvalues[i]:>10.3f} {self._pvalues[i]:>10.4f}")
            
            print("-" * 80)
            print(f"{'':<15} {'exp(coef)':>12} {'exp(-coef)':>12} {'lower .95':>12} {'upper .95':>12}")
            print("-" * 80)
            
            for i, name in enumerate(self._feature_names):
                hr = self.hazard_ratios_[i]
                print(f"{name:<15} {hr:>12.4f} {1/hr:>12.4f} "
                      f"{np.exp(self._conf_int[i, 0]):>12.4f} {np.exp(self._conf_int[i, 1]):>12.4f}")
        else:
            print(f"{'':<15} {'coef':>10} {'exp(coef)':>12}")
            print("-" * 80)
            for i, name in enumerate(self._feature_names):
                print(f"{name:<15} {self.coef_[i]:>10.4f} {self.hazard_ratios_[i]:>12.4f}")
            print("-" * 80)
            print("Inference statistics disabled (compute_inference=False).")
        
        print("=" * 80)
        if self._cindex is None:
            print("Concordance: skipped (compute_cindex=False)")
        else:
            print(f"Concordance: {self._cindex:.3f} (if 0.5-0.7: moderate, 0.7-0.9: strong)")
        if self.compute_inference and self._lr_test_stat is not None:
            print(f"Likelihood ratio test: {self._lr_test_stat:.2f} on {len(self.coef_)} df, p={self._lr_test_pvalue:.4e}")
            print(f"Wald test:            {self._wald_test_stat:.2f} on {len(self.coef_)} df, p={self._wald_test_pvalue:.4e}")
            print(f"Score (logrank) test: {self._score_test_stat:.2f} on {len(self.coef_)} df, p={self._score_test_pvalue:.4e}")
        else:
            print("Likelihood/Wald/Score tests skipped (compute_inference=False).")
        print(f"Number of Newton-Raphson iterations: {self._iterations}")
        print(f"Converged: {self._converged}")
        print("=" * 80)
    
    def predict_hazard_ratio(self, X):
        """
        Predict hazard ratios (exp(X @ coef)).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        
        Returns
        -------
        hazard_ratios : ndarray of shape (n_samples,)
            Predicted hazard ratios.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.exp(X @ self.coef_)
    
    def predict_risk_score(self, X):
        """
        Predict risk scores (X @ coef).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        
        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Predicted risk scores (linear predictor).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_
    
    def predict_survival(self, X, times=None):
        """
        Predict survival function S(t|X) = exp(-H0(t) * exp(X @ coef)).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like, optional
            Times at which to evaluate survival function.
            If None, uses unique event times from training data.
        
        Returns
        -------
        survival : ndarray of shape (n_samples, n_times)
            Predicted survival probabilities.
        times : ndarray
            Times at which survival is evaluated.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if times is None:
            times = self._unique_times
        else:
            times = np.asarray(times)
        
        if len(times) == 0 or self._baseline_cumulative_hazard is None:
            return np.ones((X.shape[0], len(times))), times
        
        # Hazard ratios
        hr = np.exp(X @ self.coef_)
        
        # Survival function: S(t) = exp(-H0(t) * HR)
        survival = np.exp(-self._baseline_cumulative_hazard[np.newaxis, :] * hr[:, np.newaxis])
        
        return survival, times
    
    def predict(self, X):
        """Alias for predict_hazard_ratio."""
        return self.predict_hazard_ratio(X)
    
    def score(self, X, time, event):
        """
        Compute concordance index on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test covariates.
        time : array-like of shape (n_samples,)
            Test event/censoring times.
        event : array-like of shape (n_samples,)
            Test event indicators.

        Returns
        -------
        cindex : float
            Concordance index.
        """
        self._check_is_fitted()

        risk_score = self.predict_risk_score(X)
        time = np.asarray(time)
        event = np.asarray(event)

        n = len(time)
        event_mask = (event == 1)

        if not np.any(event_mask):
            return 0.5

        # Use chunked vectorized approach for memory efficiency
        # Similar to _compute_cindex
        event_idx = np.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return 0.5

        concordant = np.int64(0)
        permissible = np.int64(0)
        tied_risk = np.int64(0)

        # Chunk size: keep each (chunk × n) bool matrix <= 128 MB
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time[idx_chunk, np.newaxis]
            risk_i = risk_score[idx_chunk, np.newaxis]
            time_j = time[np.newaxis, :]
            risk_j = risk_score[np.newaxis, :]
            event_j = event[np.newaxis, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))

            # Exclude self-comparisons
            chunk_indices = np.arange(end - start, dtype=np.int64)
            perm[chunk_indices, idx_chunk] = False

            concordant += int(np.sum(perm & (risk_i > risk_j)))
            tied_risk += int(np.sum(perm & (risk_i == risk_j)))
            permissible += int(np.sum(perm))

        if permissible > 0:
            return (concordant + 0.5 * tied_risk) / permissible
        return np.nan
