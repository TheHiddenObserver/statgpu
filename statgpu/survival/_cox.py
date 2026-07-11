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

        # Cumsum of outer products → prefix at each failure time
        flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n, p * p)
        prefix_flat = cp.cumsum(flat, axis=0)  # (n, p*p)

        # prefix_at_g[g] = prefix_flat[first_idx[g] - 1] if first_idx[g] > 0 else 0
        fi = first_idx.astype(cp.int64)
        prefix_at_g = cp.zeros((nuft, p, p), dtype=cp.float64)
        mask = fi > 0
        if mask.any():
            prefix_at_g[mask] = prefix_flat[fi[mask] - 1].reshape(-1, p, p)

        # risk_X2[g] = total - prefix[g]
        risk_X2 = total[None, :, :] - prefix_at_g  # (nuft, p, p)

        # hess = -sum_g sc[g] * risk_X2[g] + sum_g counts[g] * outer(E_X[g], E_X[g])
        hess = -cp.einsum("g,gij->ij", sc, risk_X2)
        hess += cp.einsum("g,gi,gj->ij", counts, E_X, E_X)

        return hess

    def _compute_hessian_breslow_fused_cupy(self, X, first_idx, counts, exp_eta):
        """Try fused RawKernel Hessian for Breslow; return None on failure."""
        import cupy as cp
        debug_fused = (
            os.environ.get("STATGPU_DEBUG_BRESLOW_FUSED", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        try:
            from ._cox_efron_cuda import compute_breslow_hess_raw
            return compute_breslow_hess_raw(
                X,
                first_idx,
                counts,
                cupy_module=cp,
                exp_eta=exp_eta,
            )
        except Exception as ex:
            if debug_fused:
                print(f"[CUDA Breslow fused fallback] {type(ex).__name__}: {ex}")
            return None

    def _compute_hessian_breslow(self, beta, X, time, event, risk_sum, risk_X_sum, exp_eta):
        """
        Compute Hessian for Breslow approximation.

        Uses an incremental suffix-scan so total cost is O(n·p²) instead of
        the previous O(n_events × n × p²) triple-loop.

        Algorithm:
          1. Compute the full second-moment matrix M = (X * exp_eta).T @ X  -- O(n·p²).
          2. Walk through sorted event positions left-to-right, subtracting the
             contribution of rows that fall *before* the current event (and are
             therefore not in its risk set) from M incrementally.
             Each row is subtracted exactly once, so total subtraction work = O(n·p²).
        """
        n_samples, n_features = X.shape
        hess = np.zeros((n_features, n_features), dtype=np.float64)

        X_exp = X * exp_eta[:, np.newaxis]                  # (n, p)
        risk_X2_sum = X_exp.T @ X                           # (p, p), O(n·p²)

        event_positions = np.where(event)[0]                # sorted ascending
        prev_pos = 0

        for ev_i in event_positions:
            # Remove rows [prev_pos, ev_i) from risk_X2_sum;
            # they have t < t[ev_i] and are no longer in R(t[ev_i]).
            if ev_i > prev_pos:
                blk = slice(prev_pos, ev_i)
                risk_X2_sum -= X_exp[blk].T @ X[blk]       # O(k·p²), k = ev_i - prev_pos
            prev_pos = ev_i  # next event will subtract starting from here

            E_X = risk_X_sum[ev_i] / risk_sum[ev_i]        # (p,)
            E_XX = risk_X2_sum / risk_sum[ev_i]             # (p, p)
            hess -= E_XX - np.outer(E_X, E_X)

        return hess

    def _efron_unique_failure_indices(self, time: np.ndarray, event: np.ndarray):
        """
        Unique failure-time bookkeeping (single stratum), matching statsmodels PHSurvivalTime.
        `time` must be sorted ascending (as in fit).
        """
        ift = np.flatnonzero(event == 1)
        if ift.size == 0:
            return np.array([], dtype=np.float64), [], [], [], 0, np.array([], dtype=np.int32)
        n = time.shape[0]
        ft = time[ift]
        uft = np.unique(ft)
        nuft = int(uft.size)

        # First row index at each unique failure time (sorted time); avoids searchsorted in log-likelihood loops.
        first_idx_uft = np.searchsorted(time, uft, side="left").astype(np.int32)

        # uft_ix: group indices of event rows by unique failure time.
        group_ids = np.searchsorted(uft, ft, side="left").astype(np.int32)  # shape: (n_events,)
        order_ev = np.argsort(group_ids, kind="stable")
        ift_sorted = ift[order_ev]
        group_sorted = group_ids[order_ev]
        counts_ev = np.bincount(group_sorted, minlength=nuft)
        ptr_ev = np.empty(nuft + 1, dtype=np.int32)
        ptr_ev[0] = 0
        ptr_ev[1:] = np.cumsum(counts_ev, dtype=np.int32)
        uft_ix = [ift_sorted[ptr_ev[i] : ptr_ev[i + 1]].tolist() for i in range(nuft)]

        # risk_enter: for each unique failure time i, indices of samples with
        # uft[i-1] <= time < uft[i] (samples entering risk set as we scan backward).
        # For i=0, includes all samples with time >= uft[0].
        j_enter = np.searchsorted(uft, time, side="right").astype(np.int32) - 1
        mask_enter = j_enter >= 0
        idx_enter = np.nonzero(mask_enter)[0]
        j_enter_m = j_enter[mask_enter]
        order_en = np.argsort(j_enter_m, kind="stable")
        idx_enter_sorted = idx_enter[order_en]
        j_enter_sorted = j_enter_m[order_en]
        counts_en = np.bincount(j_enter_sorted, minlength=nuft)
        ptr_en = np.empty(nuft + 1, dtype=np.int32)
        ptr_en[0] = 0
        ptr_en[1:] = np.cumsum(counts_en, dtype=np.int32)
        risk_enter = [
            idx_enter_sorted[ptr_en[i] : ptr_en[i + 1]].tolist() for i in range(nuft)
        ]

        # risk_exit: for backward scan, this is NOT used in the standard Efron algorithm.
        # The original code had a placeholder that put all samples at index 0, which was wrong.
        # For proper backward scan, we don't need risk_exit - we only add samples via risk_enter.
        # Set risk_exit to empty lists for all indices.
        risk_exit = [[] for _ in range(nuft)]

        return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft

    @staticmethod
    def _use_heavy_ties_cpu_fallback() -> bool:
        """Opt-in adaptive CPU fallback for heavy-ties GPU/Torch runs."""
        v = os.environ.get("STATGPU_HEAVY_TIES_CPU_FALLBACK", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _should_cpu_fallback_heavy_ties(self, n_samples, n_features, avg_tie_size):
        """Heuristic: small/medium problems with dense ties are often CPU-faster."""
        if not self._use_heavy_ties_cpu_fallback():
            return False
        if self.ties not in ("efron", "breslow"):
            return False
        if avg_tie_size < 8.0:
            return False
        return int(n_samples) <= 20000 and int(n_features) <= 64

    def _breslow_unique_failure_groups(self, time: np.ndarray, event: np.ndarray):
        """
        Breslow tie groups for sorted time/event.
        Returns (first_idx_uft, counts_uft), both int32 arrays.
        """
        ift = np.flatnonzero(event == 1)
        if ift.size == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        ft = time[ift]
        uft, counts = np.unique(ft, return_counts=True)
        first_idx_uft = np.searchsorted(time, uft, side="left").astype(np.int32)
        return first_idx_uft, counts.astype(np.int32)

    def _compute_gradient_hessian_efron_backward(self, beta, X, time, event, efron_pre=None):
        """
        Efron gradient and Hessian — incremental accumulator backward scan.

        Uses the same algorithm as statsmodels PHReg and the Cython path:
        maintain running xp0/xp1/xp2 accumulators, update incrementally at each
        failure time.  O(nuft·p²) time, O(p²) memory.

        Note: X and time are already sorted by time (caller guarantees this).
        """
        n_features = X.shape[1]
        linpred = X @ beta
        e_linpred = np.exp(linpred)

        # Build Efron precomputed structure if not provided
        if efron_pre is not None:
            uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
        else:
            event_mask = event == 1
            event_idx = np.where(event_mask)[0]
            if len(event_idx) == 0:
                return np.zeros(n_features, dtype=np.float64), np.zeros((n_features, n_features), dtype=np.float64)
            uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft = self._efron_unique_failure_indices(time, event)

        if nuft == 0:
            return np.zeros(n_features, dtype=np.float64), np.zeros((n_features, n_features), dtype=np.float64)

        # first_idx_uft[g] = first row index in sorted data with time == uft[g]
        # Suffix sums with sentinel zero at end so that
        # risk_sum[i] - risk_sum[j] = sum(exp_eta[i:j]) for any i < j.
        n = X.shape[0]
        X_exp = X * e_linpred[:, None]
        risk_sum = np.zeros(n + 1, dtype=np.float64)
        risk_sum[:n] = np.cumsum(e_linpred[::-1])[::-1]
        risk_X_sum = np.zeros((n + 1, n_features), dtype=np.float64)
        risk_X_sum[:n] = np.cumsum(X_exp[::-1], axis=0)[::-1]

        # Dispatch: Numba > Vectorized cumsum > Python incremental
        # Vectorized cumsum: O(n·p²) memory, no Python loop — fast for p <= ~100.
        _VEC_MAX_P = int(os.environ.get("STATGPU_EFRON_VEC_MAX_P", "30"))

        if _HAS_NUMBA_EFRON:
            # Numba JIT — best for all sizes
            fail_ptr = np.zeros(nuft + 1, dtype=np.int64)
            for g in range(nuft):
                fail_ptr[g + 1] = fail_ptr[g] + len(uft_ix[g])
            n_fail = int(fail_ptr[nuft])
            fail_ind = np.empty(n_fail, dtype=np.int64)
            for g in range(nuft):
                ix = uft_ix[g]
                for j in range(len(ix)):
                    fail_ind[fail_ptr[g] + j] = int(ix[j])
            grad, hess = _efron_backward_scan_numba(
                X, e_linpred, risk_sum, risk_X_sum,
                first_idx_uft.astype(np.int64),
                fail_ptr, fail_ind,
                nuft, n, n_features,
            )
        elif n_features <= _VEC_MAX_P:
            # Vectorized cumsum — eliminates Python loop, O(n·p²) memory
            grad, hess = _efron_backward_scan_vectorized(
                X, e_linpred, risk_sum, risk_X_sum,
                first_idx_uft, uft_ix, nuft, n, n_features,
            )
        else:
            # Python incremental — O(p²) memory, Python loop over groups
            grad, hess = _efron_backward_scan_python(
                X, e_linpred, risk_sum, risk_X_sum,
                first_idx_uft, uft_ix, nuft, n, n_features,
            )

        return grad, hess
    
    def _compute_gradient_hessian_gpu(
        self, beta, X, time, event, efron_pre=None, return_aux=False, entry=None, entry_ctx=None
    ):
        """Compute gradient and Hessian on GPU."""
        import cupy as cp
        import time as _time
        
        n_samples, n_features = X.shape
        
        profile_breslow = (
            os.environ.get("STATGPU_PROFILE_BRESLOW_CUDA", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        _t0_all = _time.perf_counter() if profile_breslow else None
        eta = X @ beta
        exp_eta = cp.exp(eta)
        event_mask = event == 1
        
        # Risk sets (entry-aware path uses dynamic masks below).
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1] if entry is None else None
        X_exp_eta = X * exp_eta[:, cp.newaxis]
        risk_X_sum = cp.cumsum(X_exp_eta[::-1], axis=0)[::-1] if entry is None else None
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_pre = _time.perf_counter()
        
        # Efron: when no ties, use Breslow vectorized path.
        if self.ties == "efron" and entry is None:
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
                counts_f = counts_uft.astype(cp.float64)
                grad_pre = getattr(self, "_event_X_sum_gpu", None)
                grad = (
                    grad_pre.copy()
                    if grad_pre is not None and int(grad_pre.shape[0]) == int(n_features)
                    else cp.sum(X[event_mask], axis=0)
                )
                E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
                grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)
                use_fused_breslow = (
                    os.environ.get("STATGPU_BRESLOW_FUSED_CUPY", "0").strip().lower()
                    in ("1", "true", "yes", "on")
                )
                hess = None
                if use_fused_breslow:
                    hess = self._compute_hessian_breslow_fused_cupy(
                        X, first_idx_uft, counts_f, exp_eta
                    )
                if hess is None:
                    hess = self._compute_hessian_breslow_incremental_grouped_cupy(
                        X, risk_sum, risk_X_sum, exp_eta, first_idx_uft, counts_f
                    )
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess
            if efron_pre is None:
                efron_pre = self._efron_unique_failure_indices(
                    cp.asnumpy(time), cp.asnumpy(event)
                )
            out = self._compute_gradient_hessian_efron_backward_gpu(
                beta, X, efron_pre
            )
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out
        
        # Breslow gradient/Hessian (entry-aware path).
        event_mask = event == 1
        grad = cp.zeros(n_features, dtype=cp.float64)

        if not cp.any(event_mask):
            out = (grad, cp.zeros((n_features, n_features), dtype=cp.float64))
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr = self._build_entry_ctx_gpu(
                    time, event, entry, cp
                )
                X_entry = cp.ascontiguousarray(X[entry_order])
                X_rem = cp.ascontiguousarray(X[rem_order])
                grad += cp.sum(X[event_idx], axis=0)
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                X_entry = entry_ctx[4] if len(entry_ctx) > 4 else X[entry_order]
                X_rem = entry_ctx[5] if len(entry_ctx) > 5 else X
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else cp.where(event_mask)[0]
                grad += entry_ctx[7] if len(entry_ctx) > 7 else cp.sum(X[event_mask], axis=0)
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None
            hess = cp.zeros((n_features, n_features), dtype=cp.float64)
            exp_entry = exp_eta[entry_order]
            exp_rem = exp_eta
            wx_entry = X_entry * exp_entry[:, cp.newaxis]
            wx_rem = X_rem * exp_rem[:, cp.newaxis]
            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess
            s0_add_pref = cp.cumsum(exp_entry, axis=0)
            s0_rem_pref = cp.cumsum(exp_rem, axis=0)
            s1_add_pref = cp.cumsum(wx_entry, axis=0)
            s1_rem_pref = cp.cumsum(wx_rem, axis=0)
            s0_add = cp.zeros(n_groups, dtype=cp.float64)
            s0_rem = cp.zeros(n_groups, dtype=cp.float64)
            s1_add = cp.zeros((n_groups, n_features), dtype=cp.float64)
            s1_rem = cp.zeros((n_groups, n_features), dtype=cp.float64)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = cp.asarray(add_end_np[mask_add] - 1, dtype=cp.int64)
                mask_add_cp = cp.asarray(mask_add)
                s0_add[mask_add_cp] = s0_add_pref[idx_add]
                s1_add[mask_add_cp] = s1_add_pref[idx_add]
            if np.any(mask_rem):
                idx_rem = cp.asarray(rem_end_np[mask_rem] - 1, dtype=cp.int64)
                mask_rem_cp = cp.asarray(mask_rem)
                s0_rem[mask_rem_cp] = s0_rem_pref[idx_rem]
                s1_rem[mask_rem_cp] = s1_rem_pref[idx_rem]
            s0_vec = s0_add - s0_rem
            s1_vec = s1_add - s1_rem
            d_vec = cp.asarray(d_counts, dtype=cp.float64)
            s0_safe_vec = cp.maximum(s0_vec, 1e-15)
            use_efron_entry = (self.ties == "efron")
            ex_vec = s1_vec / s0_safe_vec[:, cp.newaxis]
            if not use_efron_entry:
                grad -= cp.sum(d_vec[:, cp.newaxis] * ex_vec, axis=0)
            if use_efron_entry:
                if fail_ptr is None:
                    fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                    fail_ptr[0] = 0
                    fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
                event_exp = exp_eta[event_idx]
                X_fail = X[event_idx]
            add_ptr = 0
            rem_ptr = 0
            s2 = cp.zeros((n_features, n_features), dtype=cp.float64)
            s2_block_size = int(os.environ.get("STATGPU_ENTRY_S2_BLOCK_SIZE", "8192"))
            if s2_block_size <= 0:
                s2_block_size = 10**18
            use_s2_fused = (
                os.environ.get("STATGPU_ENTRY_S2_FUSED_CUPY", "0").strip().lower()
                in ("1", "true", "yes", "on")
            )
            s2_fused_min_rows = int(os.environ.get("STATGPU_ENTRY_S2_FUSED_MIN_ROWS", "512"))
            if s2_fused_min_rows < 1:
                s2_fused_min_rows = 1
            for g in range(n_groups):
                add_end = int(add_end_np[g])
                if add_end > add_ptr:
                    x_add = X_entry[add_ptr:add_end]
                    w_add = exp_entry[add_ptr:add_end]
                    n_add = int(add_end - add_ptr)
                    if use_s2_fused and n_add >= s2_fused_min_rows:
                        s2 = self._s2_weighted_update_cupy_fused(s2, x_add, w_add, sign=1.0)
                    elif n_add <= s2_block_size:
                        s2 = s2 + (x_add.T @ (x_add * w_add[:, cp.newaxis]))
                    else:
                        s2 = self._s2_weighted_update_cupy_blocked(
                            s2, x_add, w_add, s2_block_size, sign=1.0
                        )
                    add_ptr = add_end

                rem_end = int(rem_end_np[g])
                if rem_end > rem_ptr:
                    x_rem = X_rem[rem_ptr:rem_end]
                    w_rem = exp_eta[rem_ptr:rem_end]
                    n_rem = int(rem_end - rem_ptr)
                    if use_s2_fused and n_rem >= s2_fused_min_rows:
                        s2 = self._s2_weighted_update_cupy_fused(s2, x_rem, w_rem, sign=-1.0)
                    elif n_rem <= s2_block_size:
                        s2 = s2 - (x_rem.T @ (x_rem * w_rem[:, cp.newaxis]))
                    else:
                        s2 = self._s2_weighted_update_cupy_blocked(
                            s2, x_rem, w_rem, s2_block_size, sign=-1.0
                        )
                    rem_ptr = rem_end

                d_t_f = float(d_counts[g])
                if d_t_f <= 0:
                    continue
                if use_efron_entry:
                    st = int(fail_ptr[g])
                    ed = int(fail_ptr[g + 1])
                    ef = event_exp[st:ed]
                    xf = X_fail[st:ed]
                    ef_sum = cp.sum(ef)
                    ef_x_sum = cp.sum(xf * ef[:, cp.newaxis], axis=0)
                    ef_x2_sum = (xf.T @ (xf * ef[:, cp.newaxis]))
                    s0_g = cp.maximum(s0_vec[g], 1e-15)
                    s1_g = s1_vec[g]
                    d_i = int(d_t_f)
                    for k in range(d_i):
                        frac = float(k) / float(d_i)
                        denom = cp.maximum(s0_g - frac * ef_sum, 1e-15)
                        s1_k = s1_g - frac * ef_x_sum
                        s2_k = s2 - frac * ef_x2_sum
                        ex_k = s1_k / denom
                        grad -= ex_k
                        hess -= s2_k / denom
                        hess += cp.outer(ex_k, ex_k)
                else:
                    s0_safe = s0_safe_vec[g]
                    hess -= (d_t_f / s0_safe) * s2
            if not use_efron_entry:
                hess += ex_vec.T @ (d_vec[:, cp.newaxis] * ex_vec)
            if return_aux:
                return grad, hess, (eta, exp_eta, risk_sum)
            return grad, hess

        # For Breslow ties, all events at the same failure time share the
        # same risk set R(t); grouping is required for correctness.
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

        counts_f = getattr(self, "_breslow_counts_f_gpu", None)
        if counts_f is None or int(counts_f.shape[0]) != int(counts_uft.shape[0]):
            counts_f = counts_uft.astype(cp.float64)
        grad_pre = getattr(self, "_event_X_sum_gpu", None)
        grad = (
            grad_pre.copy()
            if grad_pre is not None and int(grad_pre.shape[0]) == int(n_features)
            else cp.sum(X[event_mask], axis=0)
        )
        E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
        grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_grad = _time.perf_counter()
        use_fused_breslow = (
            os.environ.get("STATGPU_BRESLOW_FUSED_CUPY", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        hess = None
        if use_fused_breslow:
            hess = self._compute_hessian_breslow_fused_cupy(
                X, first_idx_uft, counts_f, exp_eta
            )
        if hess is None:
            hess = self._compute_hessian_breslow_incremental_grouped_cupy(
                X, risk_sum, risk_X_sum, exp_eta, first_idx_uft, counts_f
            )
        if profile_breslow:
            cp.cuda.Stream.null.synchronize()
            _t_hess = _time.perf_counter()
            print(
                f"[CUDA Breslow profile] pre={(_t_pre - _t0_all):.4f}s "
                f"grad={(_t_grad - _t_pre):.4f}s "
                f"hess={(_t_hess - _t_grad):.4f}s "
                f"total={(_t_hess - _t0_all):.4f}s"
            )
        if return_aux:
            return grad, hess, (eta, exp_eta, risk_sum)
        return grad, hess

    def _s2_weighted_update_cupy_blocked(self, s2, x, w, block_size, sign=1.0):
        """Blocked update for large slices: s2 += sign * X^T (X * w)."""
        import cupy as cp

        n = int(x.shape[0])
        if n <= 0:
            return s2
        for st in range(0, n, block_size):
            ed = min(st + block_size, n)
            xb = x[st:ed]
            wb = w[st:ed]
            s2 = s2 + sign * (xb.T @ (xb * wb[:, cp.newaxis]))
        return s2

    def _get_entry_s2_fused_kernel_cupy(self):
        """Build/cache CuPy RawKernel for fused weighted X^T X update."""
        k = getattr(self, "_entry_s2_fused_kernel_cupy", None)
        if k is not None:
            return k
        import cupy as cp

        src = r"""
        extern "C" __global__
        void entry_s2_outer_f64(const double* x, const double* w, double* out, int n, int p) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            if (i >= p || j >= p) return;
            double acc = 0.0;
            for (int r = 0; r < n; ++r) {
                double wr = w[r];
                double xi = x[(size_t)r * (size_t)p + (size_t)i];
                double xj = x[(size_t)r * (size_t)p + (size_t)j];
                acc += wr * xi * xj;
            }
            out[(size_t)i * (size_t)p + (size_t)j] = acc;
        }
        """
        k = cp.RawKernel(src, "entry_s2_outer_f64")
        self._entry_s2_fused_kernel_cupy = k
        return k

    def _s2_weighted_update_cupy_fused(self, s2, x, w, sign=1.0):
        """CuPy fused kernel update for s2 += sign * X^T (X * w)."""
        import cupy as cp

        n = int(x.shape[0])
        if n <= 0:
            return s2
        x = cp.ascontiguousarray(x, dtype=cp.float64)
        w = cp.ascontiguousarray(w, dtype=cp.float64)
        p = int(x.shape[1])
        out = cp.empty((p, p), dtype=cp.float64)
        threads = (16, 16, 1)
        blocks = ((p + 15) // 16, (p + 15) // 16, 1)
        ker = self._get_entry_s2_fused_kernel_cupy()
        ker(blocks, threads, (x, w, out, np.int32(n), np.int32(p)))
        if sign > 0:
            return s2 + out
        return s2 - out

    def _compute_gradient_hessian_efron_backward_gpu(self, beta, X, efron_pre):
        """CuPy Efron grad/Hessian: prefer single CUDA RawKernel scan, else Python-loop fallback."""
        import cupy as cp

        uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = X.shape[1]
        if nuft == 0:
            return cp.zeros(n_features, dtype=cp.float64), cp.zeros(
                (n_features, n_features), dtype=cp.float64
            )

        n_samples = int(X.shape[0])
        avg_tie = float(n_samples) / max(1.0, float(nuft))
        use_grouped_gemm = (
            os.environ.get("STATGPU_EFRON_GROUPED_GEMM", "1").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if use_grouped_gemm and n_features <= 192 and avg_tie >= 24.0:
            return self._compute_gradient_hessian_efron_grouped_gemm_cupy(
                beta, X, efron_pre
            )

        try:
            from ._cox_efron_cuda import compute_efron_grad_hess_raw

            csr_gpu = getattr(self, "_efron_pre_csr_gpu", None)
            if csr_gpu is not None:
                out = compute_efron_grad_hess_raw(
                    X,
                    beta,
                    efron_pre,
                    efron_csr=csr_gpu,
                    cupy_module=cp,
                )
            else:
                out = compute_efron_grad_hess_raw(X, beta, efron_pre, cupy_module=cp)
            if out is not None:
                return out[0], out[1]
        except Exception:
            pass

        linpred = X @ beta
        linpred = linpred - cp.max(linpred)
        e_linpred = cp.exp(linpred)

        grad = cp.zeros(n_features, dtype=cp.float64)
        hess_inner = cp.zeros((n_features, n_features), dtype=cp.float64)
        xp0 = cp.zeros((), dtype=cp.float64)
        xp1 = cp.zeros(n_features, dtype=cp.float64)
        xp2 = cp.zeros((n_features, n_features), dtype=cp.float64)
        for i in range(nuft)[::-1]:
            ix = risk_enter[i]
            if len(ix) > 0:
                ix = cp.array(ix, dtype=cp.int32)
                elx = e_linpred[ix]
                v = X[ix]
                xp0 = xp0 + elx.sum()
                xp1 = xp1 + (elx[:, None] * v).sum(axis=0)
                xp2 = xp2 + cp.einsum("ij,ik,i->jk", v, v, elx)
            ixf = uft_ix[i]
            if len(ixf) > 0:
                ixf = cp.array(ixf, dtype=cp.int32)
                v = X[ixf]
                elx = e_linpred[ixf]
                xp0f = elx.sum()
                xp1f = (elx[:, None] * v).sum(axis=0)
                xp2f = cp.einsum("ij,ik,i->jk", v, v, elx)
                m = len(ixf)
                J = cp.arange(m, dtype=cp.float64) / max(m, 1)
                c0 = xp0 - J * xp0f
                c0 = cp.maximum(c0, 1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = cp.sum(ak)
                sum_J_c0 = cp.sum(bk)
                sum_aa = cp.sum(ak * ak)
                sum_bb = cp.sum(bk * bk)
                sum_ab = cp.sum(ak * bk)
                grad = grad + v.sum(axis=0)
                grad = grad - (xp1 * sum_inv_c0 - xp1f * sum_J_c0)
                hess_inner = hess_inner + xp2 * sum_inv_c0
                hess_inner = hess_inner - xp2f * sum_J_c0
                hess_inner = hess_inner - (
                    sum_aa * cp.outer(xp1, xp1)
                    + sum_bb * cp.outer(xp1f, xp1f)
                    - sum_ab * (cp.outer(xp1, xp1f) + cp.outer(xp1f, xp1))
                )
            ix = risk_exit[i]
            if len(ix) > 0:
                ix = cp.array(ix, dtype=cp.int32)
                elx = e_linpred[ix]
                v = X[ix]
                xp0 = xp0 - elx.sum()
                xp1 = xp1 - (elx[:, None] * v).sum(axis=0)
                xp2 = xp2 - cp.einsum("ij,ik,i->jk", v, v, elx)

        hess = -hess_inner
        return grad, hess

    def _compute_gradient_hessian_efron_grouped_gemm_cupy(self, beta, X, efron_pre):
        """Exact Efron grad/hess on CuPy via grouped GEMM updates (no p^2 atomics)."""
        import cupy as cp

        _, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = int(X.shape[1])
        linpred = X @ beta
        linpred = linpred - cp.max(linpred)
        e_linpred = cp.exp(linpred)

        grad = cp.zeros(n_features, dtype=cp.float64)
        hess_inner = cp.zeros((n_features, n_features), dtype=cp.float64)
        xp0 = cp.zeros((), dtype=cp.float64)
        xp1 = cp.zeros(n_features, dtype=cp.float64)
        xp2 = cp.zeros((n_features, n_features), dtype=cp.float64)
        j_cache = {}

        for i in range(nuft - 1, -1, -1):
            ix = risk_enter[i]
            if len(ix) > 0:
                idx = cp.asarray(ix, dtype=cp.int32)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 + cp.sum(elx)
                xp1 = xp1 + cp.sum(wv, axis=0)
                xp2 = xp2 + (wv.T @ v)

            ixf = uft_ix[i]
            if len(ixf) > 0:
                idxf = cp.asarray(ixf, dtype=cp.int32)
                v = X[idxf]
                elx = e_linpred[idxf]
                wv = v * elx[:, None]
                xp0f = cp.sum(elx)
                xp1f = cp.sum(wv, axis=0)
                xp2f = wv.T @ v
                m = len(ixf)
                if m not in j_cache:
                    j_cache[m] = cp.arange(m, dtype=cp.float64) / float(max(m, 1))
                J = j_cache[m]
                c0 = cp.maximum(xp0 - J * xp0f, 1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = cp.sum(ak)
                sum_J_c0 = cp.sum(bk)
                sum_aa = cp.sum(ak * ak)
                sum_bb = cp.sum(bk * bk)
                sum_ab = cp.sum(ak * bk)
                grad = grad + cp.sum(v, axis=0)
                grad = grad - (xp1 * sum_inv_c0 - xp1f * sum_J_c0)
                hess_inner = hess_inner + xp2 * sum_inv_c0
                hess_inner = hess_inner - xp2f * sum_J_c0
                hess_inner = hess_inner - (
                    sum_aa * cp.outer(xp1, xp1)
                    + sum_bb * cp.outer(xp1f, xp1f)
                    - sum_ab * (cp.outer(xp1, xp1f) + cp.outer(xp1f, xp1))
                )

            ix = risk_exit[i]
            if len(ix) > 0:
                idx = cp.asarray(ix, dtype=cp.int32)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 - cp.sum(elx)
                xp1 = xp1 - cp.sum(wv, axis=0)
                xp2 = xp2 - (wv.T @ v)

        return grad, -hess_inner

    def _solve_newton_delta_torch(self, hess, grad):
        """Newton step delta = inv(hess) @ grad; prefer SPD solve on (-hess) with light jitter."""
        import torch

        p = int(hess.shape[0])
        try:
            H = -hess
            eps = 1e-11 * (torch.max(torch.abs(torch.diag(H))) + 1.0)
            H = H + eps * torch.eye(p, dtype=torch.float64, device=hess.device)
            return -torch.linalg.solve(H, grad)
        except Exception:
            try:
                return torch.linalg.solve(hess, grad)
            except Exception:
                result = torch.linalg.lstsq(hess, grad)
                return result.solution.flatten()

    def _compute_gradient_hessian_efron_grouped_gemm_torch(self, beta, X, efron_pre):
        """Exact Efron grad/hess on Torch device via grouped GEMM updates."""
        import torch

        _, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = int(X.shape[1])
        linpred = X @ beta
        linpred = linpred - torch.max(linpred)
        e_linpred = torch.exp(linpred)

        grad = torch.zeros(n_features, dtype=torch.float64, device=beta.device)
        hess_inner = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
        xp0 = torch.zeros((), dtype=torch.float64, device=beta.device)
        xp1 = torch.zeros(n_features, dtype=torch.float64, device=beta.device)
        xp2 = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
        j_cache = {}

        for i in range(nuft - 1, -1, -1):
            ix = risk_enter[i]
            if len(ix) > 0:
                idx = torch.as_tensor(ix, dtype=torch.long, device=beta.device)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 + torch.sum(elx)
                xp1 = xp1 + torch.sum(wv, dim=0)
                xp2 = xp2 + (wv.transpose(0, 1) @ v)

            ixf = uft_ix[i]
            if len(ixf) > 0:
                idxf = torch.as_tensor(ixf, dtype=torch.long, device=beta.device)
                v = X[idxf]
                elx = e_linpred[idxf]
                wv = v * elx[:, None]
                xp0f = torch.sum(elx)
                xp1f = torch.sum(wv, dim=0)
                xp2f = wv.transpose(0, 1) @ v
                m = len(ixf)
                if m not in j_cache:
                    j_cache[m] = torch.arange(m, dtype=torch.float64, device=beta.device) / float(max(m, 1))
                J = j_cache[m]
                c0 = torch.clamp(xp0 - J * xp0f, min=1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = torch.sum(ak)
                sum_J_c0 = torch.sum(bk)
                sum_aa = torch.sum(ak * ak)
                sum_bb = torch.sum(bk * bk)
                sum_ab = torch.sum(ak * bk)
                grad = grad + torch.sum(v, dim=0)
                grad = grad - (xp1 * sum_inv_c0 - xp1f * sum_J_c0)
                hess_inner = hess_inner + xp2 * sum_inv_c0
                hess_inner = hess_inner - xp2f * sum_J_c0
                hess_inner = hess_inner - (
                    sum_aa * torch.outer(xp1, xp1)
                    + sum_bb * torch.outer(xp1f, xp1f)
                    - sum_ab * (torch.outer(xp1, xp1f) + torch.outer(xp1f, xp1))
                )

            ix = risk_exit[i]
            if len(ix) > 0:
                idx = torch.as_tensor(ix, dtype=torch.long, device=beta.device)
                v = X[idx]
                elx = e_linpred[idx]
                wv = v * elx[:, None]
                xp0 = xp0 - torch.sum(elx)
                xp1 = xp1 - torch.sum(wv, dim=0)
                xp2 = xp2 - (wv.transpose(0, 1) @ v)

        return grad, -hess_inner

    def _compute_log_likelihood_torch(self, beta, X, time, event, efron_pre=None, entry=None, entry_ctx=None):
        """Compute log partial likelihood on Torch."""
        import torch

        eta = X @ beta
        exp_eta = torch.exp(eta)
        # Entry+breslow path does not consume risk_sum; skip the cumsum to
        # reduce per-evaluation overhead during line-search probes.
        risk_sum = None if entry is not None else torch.cumsum(exp_eta.flip(0), dim=0).flip(0)
        return self._compute_log_likelihood_torch_from_stats(
            eta, exp_eta, risk_sum, time, event, efron_pre, entry=entry, entry_ctx=entry_ctx
        )

    def _build_entry_ctx_torch(self, time, event, entry, device):
        """Build entry-time grouped indexing context for a specific sorted Torch view."""
        import torch

        event_mask = event == 1
        event_idx = torch.where(event_mask)[0]
        evt_t = time[event_idx].detach().cpu().numpy()
        if evt_t.size == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                np.zeros((1,), dtype=np.int64),
            )
        uft_np, d_counts = np.unique(evt_t, return_counts=True)
        d_counts = d_counts.astype(np.float64, copy=False)
        entry_order = torch.argsort(entry, stable=True)
        entry_sorted_np = entry.index_select(0, entry_order).detach().cpu().numpy()
        time_np = time.detach().cpu().numpy()
        add_end_np = np.searchsorted(entry_sorted_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_end_np = np.searchsorted(time_np, uft_np, side="left").astype(np.int64, copy=False)
        rem_order = torch.arange(int(time.shape[0]), dtype=torch.long, device=device)
        event_idx = event_idx.to(torch.long)
        fail_ptr = np.empty(d_counts.shape[0] + 1, dtype=np.int64)
        fail_ptr[0] = 0
        fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
        return (entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr)

    def _compute_log_likelihood_torch_from_stats(
        self, eta, exp_eta, risk_sum, time, event, efron_pre=None, entry=None, entry_ctx=None
    ):
        """Compute log partial likelihood on Torch with precomputed stats."""
        import torch

        ll = torch.tensor(0.0, dtype=torch.float64, device=eta.device)
        event_mask = event == 1

        if not torch.any(event_mask):
            return ll

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, _rem_order, event_idx, fail_ptr = self._build_entry_ctx_torch(
                    time, event, entry, eta.device
                )
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else torch.where(event_mask)[0]
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None

            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                return torch.tensor(0.0, dtype=torch.float64, device=eta.device)
            if fail_ptr is None:
                fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                fail_ptr[0] = 0
                fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)

            exp_entry = exp_eta.index_select(0, entry_order)
            exp_rem = exp_eta
            s0_add_pref = torch.cumsum(exp_entry, dim=0)
            s0_rem_pref = torch.cumsum(exp_rem, dim=0)
            s0_add = torch.zeros(n_groups, dtype=torch.float64, device=eta.device)
            s0_rem = torch.zeros(n_groups, dtype=torch.float64, device=eta.device)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = torch.as_tensor(add_end_np[mask_add] - 1, dtype=torch.long, device=eta.device)
                s0_add[torch.as_tensor(mask_add, dtype=torch.bool, device=eta.device)] = s0_add_pref.index_select(0, idx_add)
            if np.any(mask_rem):
                idx_rem = torch.as_tensor(rem_end_np[mask_rem] - 1, dtype=torch.long, device=eta.device)
                s0_rem[torch.as_tensor(mask_rem, dtype=torch.bool, device=eta.device)] = s0_rem_pref.index_select(0, idx_rem)
            s0_vec = torch.clamp(s0_add - s0_rem, min=1e-300)
            event_eta = eta.index_select(0, event_idx)

            if self.ties == "breslow":
                d_vec = torch.as_tensor(d_counts, dtype=torch.float64, device=eta.device)
                return torch.sum(event_eta) - torch.sum(d_vec * torch.log(s0_vec))

            ll = torch.sum(event_eta)
            event_exp = exp_eta.index_select(0, event_idx)
            for g in range(n_groups):
                d = int(d_counts[g])
                if d <= 0:
                    continue
                st = int(fail_ptr[g])
                ed = int(fail_ptr[g + 1])
                ef = torch.sum(event_exp[st:ed])
                base = s0_vec[g]
                for k in range(d):
                    denom = torch.clamp(base - (float(k) / float(d)) * ef, min=1e-300)
                    ll = ll - torch.log(denom)
            return ll

        if self.ties == "breslow":
            # Vectorized Breslow using cached failure groups
            breslow_pre_torch = getattr(self, "_breslow_pre_torch", None)
            if (
                breslow_pre_torch is not None
                and len(breslow_pre_torch) == 2
                and int(breslow_pre_torch[0].numel()) > 0
            ):
                first_idx_uft, counts_uft = breslow_pre_torch
            else:
                uft, counts_uft = torch.unique(time[event_mask], return_counts=True)
                first_idx_uft = torch.searchsorted(time, uft, side="left")
                counts_uft = counts_uft.to(torch.int32)
            risk_at = risk_sum[first_idx_uft]
            return torch.sum(eta[event_mask]) - torch.sum(
                counts_uft.to(torch.float64) * torch.log(risk_at)
            )

        # Efron: keep computation fully on torch backend.
        if efron_pre is not None:
            needs_exact_ties = not getattr(self, "_efron_all_singletons", False)
            # No-tie Efron equals Breslow; keep computation on torch device.
            if not needs_exact_ties:
                _, _, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
                first_idx_t = torch.as_tensor(first_idx_uft, dtype=torch.int64, device=eta.device)
                counts_t = torch.ones(int(nuft), dtype=torch.float64, device=eta.device)
                risk_at = risk_sum[first_idx_t]
                return torch.sum(eta[event_mask]) - torch.sum(counts_t * torch.log(risk_at))

        # Fallback Efron (loop version)
        unique_times = torch.unique(time[event_mask])
        for t in unique_times:
            at_time_t = time == t
            events_at_t = at_time_t & event_mask
            d = int(torch.sum(events_at_t).item())

            if d == 0:
                continue

            risk_indices = torch.where(time >= t)[0]
            if risk_indices.numel() == 0:
                continue

            first_idx = risk_indices[0]
            risk_at_t = risk_sum[first_idx]
            sum_events = torch.sum(exp_eta[events_at_t])

            ll += torch.sum(eta[events_at_t])
            for k in range(d):
                ll -= torch.log(torch.maximum(risk_at_t - (k / d) * sum_events, torch.tensor(1e-300, dtype=torch.float64, device=eta.device)))

        return ll

    def _compute_gradient_hessian_torch(
        self, beta, X, time, event, efron_pre=None, return_aux=False, entry=None, entry_ctx=None
    ):
        """Fully vectorized gradient/Hessian for Torch - Efron and Breslow."""
        import torch
        n_samples, n_features = X.shape
        eta = X @ beta
        exp_eta = torch.exp(eta)
        rev_idx = torch.arange(n_samples - 1, -1, -1, device=beta.device)
        risk_sum = torch.cumsum(exp_eta[rev_idx], dim=0)[rev_idx] if entry is None else None

        if self.ties == "efron" and efron_pre is not None and entry is None:
            needs_exact_ties = not getattr(self, "_efron_all_singletons", False)
            n_samples = int(X.shape[0])
            avg_tie = float(n_samples) / max(1.0, float(_unpack_efron_pre6(efron_pre)[4]))
            use_grouped_gemm = (
                os.environ.get("STATGPU_EFRON_GROUPED_GEMM", "1").strip().lower()
                in ("1", "true", "yes", "on")
            )
            # For real ties, use exact torch grouped GEMM path only.
            if needs_exact_ties and (
                use_grouped_gemm
                and beta.is_cuda
                and n_features <= 192
                and avg_tie >= 24.0
            ):
                out = self._compute_gradient_hessian_efron_grouped_gemm_torch(
                    beta, X, efron_pre
                )
                if return_aux:
                    return out[0], out[1], (eta, exp_eta, risk_sum)
                return out

            # ---- Triton Efron path ----
            if (
                os.environ.get("STATGPU_EFRON_TRITON", "0").strip().lower()
                in ("1", "true", "yes", "on")
                and beta.is_cuda
                and efron_pre is not None
            ):
                from statgpu.survival._cox_efron_triton import compute_efron_grad_hess_triton
                triton_out = compute_efron_grad_hess_triton(X, beta, efron_pre)
                if triton_out is not None:
                    grad, hess = triton_out
                    if return_aux:
                        return grad, hess, (eta, exp_eta, risk_sum)
                    return grad, hess

        # Reverse cumsum for risk sets (vectorized)
        risk_X_sum = torch.cumsum((X * exp_eta[:, None])[rev_idx], dim=0)[rev_idx] if entry is None else None

        event_mask = event == 1
        if not torch.any(event_mask):
            out = (
                torch.zeros(n_features, dtype=torch.float64, device=beta.device),
                torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device),
            )
            if return_aux:
                return out[0], out[1], (eta, exp_eta, risk_sum)
            return out

        if entry is not None:
            if entry_ctx is None:
                entry_order, d_counts, add_end_np, rem_end_np, rem_order, event_idx, fail_ptr = self._build_entry_ctx_torch(
                    time, event, entry, beta.device
                )
                X_entry = X.index_select(0, entry_order).contiguous()
                X_rem = X.index_select(0, rem_order).contiguous()
                grad = torch.sum(X.index_select(0, event_idx), dim=0)
            else:
                entry_order, d_counts, add_end_np, rem_end_np = entry_ctx[:4]
                X_entry = entry_ctx[4] if len(entry_ctx) > 4 else X.index_select(0, entry_order)
                X_rem = entry_ctx[5] if len(entry_ctx) > 5 else X
                event_idx = entry_ctx[6] if len(entry_ctx) > 6 else torch.where(event_mask)[0]
                grad = entry_ctx[7] if len(entry_ctx) > 7 else torch.sum(X[event_mask], dim=0)
                fail_ptr = entry_ctx[8] if len(entry_ctx) > 8 else None
            hess = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
            exp_entry = exp_eta.index_select(0, entry_order)
            exp_rem = exp_eta
            wx_entry = X_entry * exp_entry.unsqueeze(1)
            wx_rem = X_rem * exp_rem.unsqueeze(1)
            n_groups = int(d_counts.shape[0])
            if n_groups == 0:
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess
            s0_add_pref = torch.cumsum(exp_entry, dim=0)
            s0_rem_pref = torch.cumsum(exp_rem, dim=0)
            s1_add_pref = torch.cumsum(wx_entry, dim=0)
            s1_rem_pref = torch.cumsum(wx_rem, dim=0)
            s0_add = torch.zeros(n_groups, dtype=torch.float64, device=beta.device)
            s0_rem = torch.zeros(n_groups, dtype=torch.float64, device=beta.device)
            s1_add = torch.zeros((n_groups, n_features), dtype=torch.float64, device=beta.device)
            s1_rem = torch.zeros((n_groups, n_features), dtype=torch.float64, device=beta.device)
            mask_add = add_end_np > 0
            mask_rem = rem_end_np > 0
            if np.any(mask_add):
                idx_add = torch.as_tensor(add_end_np[mask_add] - 1, dtype=torch.long, device=beta.device)
                mask_add_t = torch.as_tensor(mask_add, dtype=torch.bool, device=beta.device)
                s0_add[mask_add_t] = s0_add_pref.index_select(0, idx_add)
                s1_add[mask_add_t] = s1_add_pref.index_select(0, idx_add)
            if np.any(mask_rem):
                idx_rem = torch.as_tensor(rem_end_np[mask_rem] - 1, dtype=torch.long, device=beta.device)
                mask_rem_t = torch.as_tensor(mask_rem, dtype=torch.bool, device=beta.device)
                s0_rem[mask_rem_t] = s0_rem_pref.index_select(0, idx_rem)
                s1_rem[mask_rem_t] = s1_rem_pref.index_select(0, idx_rem)
            s0_vec = s0_add - s0_rem
            s1_vec = s1_add - s1_rem
            d_vec = torch.as_tensor(d_counts, dtype=torch.float64, device=beta.device)
            s0_safe_vec = torch.clamp(s0_vec, min=1e-15)
            use_efron_entry = (self.ties == "efron")
            ex_vec = s1_vec / s0_safe_vec.unsqueeze(1)
            if not use_efron_entry:
                grad = grad - torch.sum(d_vec.unsqueeze(1) * ex_vec, dim=0)
            if use_efron_entry:
                if fail_ptr is None:
                    fail_ptr = np.empty(n_groups + 1, dtype=np.int64)
                    fail_ptr[0] = 0
                    fail_ptr[1:] = np.cumsum(d_counts.astype(np.int64), dtype=np.int64)
                event_exp = exp_eta.index_select(0, event_idx)
                X_fail = X.index_select(0, event_idx)
            add_ptr = 0
            rem_ptr = 0
            s2 = torch.zeros((n_features, n_features), dtype=torch.float64, device=beta.device)
            s2_block_size = int(os.environ.get("STATGPU_ENTRY_S2_BLOCK_SIZE", "8192"))
            if s2_block_size <= 0:
                s2_block_size = 10**18
            s2_fn = self._get_entry_s2_torch_fn()
            for g in range(n_groups):
                add_end = int(add_end_np[g])
                if add_end > add_ptr:
                    x_add = X_entry[add_ptr:add_end]
                    w_add = exp_entry[add_ptr:add_end]
                    n_add = int(add_end - add_ptr)
                    if n_add <= s2_block_size:
                        s2 = s2 + s2_fn(x_add, w_add)
                    else:
                        s2 = self._s2_weighted_update_torch_blocked(
                            s2, x_add, w_add, s2_block_size, sign=1.0
                        )
                    add_ptr = add_end

                rem_end = int(rem_end_np[g])
                if rem_end > rem_ptr:
                    x_rem = X_rem[rem_ptr:rem_end]
                    w_rem = exp_eta[rem_ptr:rem_end]
                    n_rem = int(rem_end - rem_ptr)
                    if n_rem <= s2_block_size:
                        s2 = s2 - s2_fn(x_rem, w_rem)
                    else:
                        s2 = self._s2_weighted_update_torch_blocked(
                            s2, x_rem, w_rem, s2_block_size, sign=-1.0
                        )
                    rem_ptr = rem_end

                d_t_f = float(d_counts[g])
                if d_t_f <= 0:
                    continue
                if use_efron_entry:
                    st = int(fail_ptr[g])
                    ed = int(fail_ptr[g + 1])
                    ef = event_exp[st:ed]
                    xf = X_fail[st:ed]
                    ef_sum = torch.sum(ef)
                    ef_x_sum = torch.sum(xf * ef.unsqueeze(1), dim=0)
                    ef_x2_sum = xf.transpose(0, 1) @ (xf * ef.unsqueeze(1))
                    s0_g = torch.clamp(s0_vec[g], min=1e-15)
                    s1_g = s1_vec[g]
                    d_i = int(d_t_f)
                    for k in range(d_i):
                        frac = float(k) / float(d_i)
                        denom = torch.clamp(s0_g - frac * ef_sum, min=1e-15)
                        s1_k = s1_g - frac * ef_x_sum
                        s2_k = s2 - frac * ef_x2_sum
                        ex_k = s1_k / denom
                        grad = grad - ex_k
                        hess = hess - (s2_k / denom)
                        hess = hess + torch.outer(ex_k, ex_k)
                else:
                    s0_safe = s0_safe_vec[g]
                    hess = hess - (d_t_f / s0_safe) * s2
            if not use_efron_entry:
                hess = hess + ex_vec.transpose(0, 1) @ (d_vec.unsqueeze(1) * ex_vec)
            if return_aux:
                return grad, hess, (eta, exp_eta, risk_sum)
            return grad, hess

        # Get event data
        event_times = time[event_mask]

        # Unique failure times with inverse mapping
        uft, unique_inv = torch.unique(event_times, sorted=True, return_inverse=True)
        n_uft = len(uft)
        counts = torch.bincount(unique_inv).to(torch.float64)

        # Get first index of each unique time
        sorted_times, sort_idx = torch.sort(time)
        first_in_sorted = torch.searchsorted(sorted_times, uft, side="left")
        first_idx = sort_idx[first_in_sorted]

        # Risk values at unique times
        risk_at_uft = risk_sum[first_idx]
        risk_X_at_uft = risk_X_sum[first_idx]
        E_X_at_uft = risk_X_at_uft / risk_at_uft[:, None]

        # Sum X and exp(eta) for events at each unique time
        event_indices = event_mask.nonzero(as_tuple=True)[0]
        sum_X_per_uft = torch.zeros((n_uft, n_features), dtype=torch.float64, device=beta.device)
        sum_X_per_uft.index_add_(0, unique_inv, X[event_indices])

        # ============= GRADIENT =============
        if self.ties == "efron":
            # Efron closed-form: (d+1)/2 * E[X|R]
            efron_weight = (counts + 1) / 2.0
            grad = torch.sum(sum_X_per_uft - efron_weight[:, None] * E_X_at_uft, dim=0)
        else:
            # Breslow: d * E[X|R]
            grad = torch.sum(sum_X_per_uft - counts[:, None] * E_X_at_uft, dim=0)

        # Hessian
        # Use incremental risk-set second moments to avoid materializing
        # a (n_samples, n_features, n_features) tensor on GPU (can OOM at 50k x 100).
        X_exp = X * exp_eta[:, None]
        risk_X2 = X_exp.transpose(0, 1) @ X

        # Weight by counts (Breslow) or Efron-adjusted weights
        if self.ties == "efron":
            weights = efron_weight
        else:
            weights = counts

        # ---- Triton Breslow path ----
        if (
            self.ties != "efron"
            and os.environ.get("STATGPU_BRESLOW_TRITON", "0").strip().lower()
            in ("1", "true", "yes", "on")
            and beta.is_cuda
        ):
            from statgpu.survival._cox_efron_triton import compute_breslow_grad_hess_triton
            triton_out = compute_breslow_grad_hess_triton(X, beta, time, event)
            if triton_out is not None:
                grad, hess = triton_out
                if return_aux:
                    return grad, hess, (eta, exp_eta, risk_sum)
                return grad, hess

        # ---- Vectorized Hessian via cumsum of outer products ----
        # hess = -sum_g (counts[g]/s0[g]) * risk_X2[g] + sum_g counts[g] * outer(E_X[g], E_X[g])
        # where risk_X2[g] = total - prefix[g], prefix = cumsum of outer products.
        total = risk_X2  # X_exp.T @ X
        sc = weights / torch.clamp(risk_at_uft, min=1e-300)  # (n_uft,)

        # Cumsum of outer products → prefix at each failure time
        flat = (X_exp[:, :, None] * X[:, None, :]).reshape(n_samples, n_features * n_features)
        prefix_flat = torch.cumsum(flat, dim=0)  # (n, p*p)

        # prefix_at_g[g] = prefix_flat[first_idx[g] - 1] if first_idx[g] > 0 else 0
        prefix_at_g = torch.zeros((n_uft, n_features, n_features),
                                  dtype=torch.float64, device=beta.device)
        mask = first_idx > 0
        if mask.any():
            prefix_at_g[mask] = prefix_flat[first_idx[mask] - 1].reshape(-1, n_features, n_features)

        # risk_X2[g] = total - prefix[g]
        risk_X2_at_g = total.unsqueeze(0) - prefix_at_g  # (n_uft, p, p)

        # hess = -sum_g sc[g] * risk_X2[g] + sum_g weights[g] * outer(E_X[g], E_X[g])
        hess = -torch.einsum("g,gij->ij", sc, risk_X2_at_g)
        hess += torch.einsum("g,gi,gj->ij", weights, E_X_at_uft, E_X_at_uft)

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

    @staticmethod
    def _observed_information(hess):
        """Return a symmetric positive-oriented observed information matrix.

        Breslow kernels return the log-likelihood Hessian, while legacy Efron
        kernels return its negation.  Select the orientation with the larger
        positive spectral mass and keep this compatibility normalization at the
        inference boundary.
        """
        sym = 0.5 * (np.asarray(hess, dtype=np.float64) + np.asarray(hess, dtype=np.float64).T)
        eigvals = np.linalg.eigvalsh(sym)
        positive_mass = float(np.sum(np.clip(eigvals, 0.0, None)))
        negative_mass = float(np.sum(np.clip(-eigvals, 0.0, None)))
        return sym if positive_mass >= negative_mass else -sym

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
        information = self._observed_information(hess)
        try:
            bread = np.linalg.solve(information, np.eye(n_features))
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(information)

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
            info_0 = self._observed_information(hess_0)
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
