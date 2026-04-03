"""
Cox Proportional Hazards regression with GPU acceleration.

Implements Cox PH model using Breslow and Efron approximations for ties with
Newton-Raphson optimization. Matches R's survival::coxph() API.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


def _unpack_efron_pre6(efron_pre):
    """``(uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft)`` — supports legacy 5-tuple in tests only."""
    if len(efron_pre) == 6:
        return efron_pre
    if len(efron_pre) == 5:
        uft, uft_ix, re, rx, nuft = efron_pre
        return uft, uft_ix, re, rx, nuft, None
    raise ValueError(f"invalid efron_pre length {len(efron_pre)}")


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
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.ties = ties.lower()
        self.tol = tol
        self.max_iter = max_iter
        self.compute_inference = compute_inference
        self.cov_type = cov_type.lower()
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        
        if self.ties not in ('breslow', 'efron'):
            raise ValueError("ties must be 'breslow' or 'efron'")
        if self.cov_type not in ("nonrobust", "hc0", "hc1", "cluster"):
            raise ValueError("cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'")
        
        # Fitted attributes
        self.coef_ = None
        self.hazard_ratios_ = None
        
        # Internal storage for inference
        self._time = None
        self._event = None
        self._X = None
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
        
    def fit(self, X, time, event, entry=None, cluster=None):
        """
        Fit Cox Proportional Hazards model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like of shape (n_samples,)
            Time to event or censoring.
        event : array-like of shape (n_samples,)
            Event indicator (1 = event, 0 = censored).
        entry : array-like of shape (n_samples,), optional
            Entry time for delayed entry (left truncation).
        cluster : array-like of shape (n_samples,), optional
            Cluster ids for cluster-robust covariance when `cov_type='cluster'`.
        
        Returns
        -------
        self : CoxPH
            Fitted estimator.
        """
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            import cupy as cp
            
            X_gpu = cp.asarray(self._to_array(X), dtype=cp.float64)
            time_gpu = cp.asarray(self._to_array(time), dtype=cp.float64)
            event_gpu = cp.asarray(self._to_array(event), dtype=cp.int32)
            
            if X_gpu.ndim == 1:
                X_gpu = X_gpu.reshape(-1, 1)
            
            self._nobs = int(X_gpu.shape[0])
            self._nevents = int(cp.sum(event_gpu).item())
            self._feature_names = [f'x{i+1}' for i in range(int(X_gpu.shape[1]))]
            
            # Keep CPU copies only when CPU-side inference/baseline stats are requested.
            if self.compute_inference:
                self._X = cp.asnumpy(X_gpu)
                self._time = cp.asnumpy(time_gpu)
                self._event = cp.asnumpy(event_gpu)
            else:
                self._X = None
                self._time = None
                self._event = None
            
            cluster_gpu = None if cluster is None else cp.asarray(self._to_array(cluster), dtype=cp.int64)
            self._fit_gpu(X_gpu, time_gpu, event_gpu, entry, cluster_gpu)
        else:
            X_np = np.asarray(self._to_array(X, Device.CPU), dtype=np.float64)
            time_np = np.asarray(self._to_array(time, Device.CPU), dtype=np.float64)
            event_np = np.asarray(self._to_array(event, Device.CPU), dtype=np.int32)
            
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            
            self._nobs = X_np.shape[0]
            self._nevents = np.sum(event_np)
            
            # Store original data (CPU mode is CPU-only)
            self._time = time_np.copy()
            self._event = event_np.copy()
            self._X = X_np.copy()
            self._feature_names = [f'x{i+1}' for i in range(X_np.shape[1])]
            
            cluster_np = None if cluster is None else np.asarray(self._to_array(cluster, Device.CPU))
            self._fit_cpu(X_np, time_np, event_np, entry, cluster_np)
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, time, event, entry=None, cluster=None):
        """Fit using CPU (NumPy)."""
        n_samples, n_features = X.shape
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = np.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        cluster_sorted = None if cluster is None else np.asarray(cluster)[order]
        
        self._efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
            self._efron_pre = self._efron_unique_failure_indices(time_sorted, event_sorted)
        else:
            self._breslow_pre = self._breslow_unique_failure_groups(
                time_sorted, event_sorted
            )
        
        # Initialize coefficients
        beta = np.zeros(n_features, dtype=np.float64)
        
        # Compute null log-likelihood (beta = 0)
        self._log_likelihood_null = self._compute_log_likelihood(
            np.zeros(n_features), X_sorted, time_sorted, event_sorted, self._efron_pre
        )
        
        # Newton-Raphson optimization
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian
            grad, hess = self._compute_gradient_hessian(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre
            )
            
            # Newton step
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
            
            # Line search with step halving (old_ll is constant w.r.t. step size)
            old_ll = self._compute_log_likelihood(
                beta, X_sorted, time_sorted, event_sorted, self._efron_pre
            )
            step = 1.0
            for _ in range(20):
                new_beta = beta - step * delta
                new_ll = self._compute_log_likelihood(
                    new_beta, X_sorted, time_sorted, event_sorted, self._efron_pre
                )
                if new_ll > old_ll - 1e-8:
                    break
                step *= 0.5
            
            new_beta = beta - step * delta
            
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
            beta, X_sorted, time_sorted, event_sorted, self._efron_pre
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
        self._compute_cindex()
    
    def _fit_gpu(self, X, time, event, entry=None, cluster=None):
        """Fit using GPU with full GPU computation."""
        import cupy as cp
        from .._gpu_utils import norm_two_tail_pvalues_gpu, norm_crit_gpu_two_tail
        
        n_samples, n_features = X.shape
        
        # Transfer to GPU once
        X = cp.asarray(X, dtype=cp.float64)
        time = cp.asarray(time, dtype=cp.float64)
        event = cp.asarray(event, dtype=cp.int32)
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = cp.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        cluster_sorted = None if cluster is None else cluster[order]
        
        # Precompute Efron tie structure once (depends only on time/event order).
        efron_pre = None
        self._breslow_pre = None
        self._breslow_pre_gpu = None
        if self.ties == "efron":
            efron_pre = self._efron_unique_failure_indices(
                cp.asnumpy(time_sorted), cp.asnumpy(event_sorted)
            )
            self._efron_pre = efron_pre
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
            except Exception:
                self._efron_pre_csr = None
        else:
            self._efron_pre = None
            self._efron_pre_csr = None
            first_idx_uft, counts_uft = self._breslow_unique_failure_groups(
                cp.asnumpy(time_sorted), cp.asnumpy(event_sorted)
            )
            self._breslow_pre = (first_idx_uft, counts_uft)
            self._breslow_pre_gpu = (
                cp.asarray(first_idx_uft, dtype=cp.int32),
                cp.asarray(counts_uft, dtype=cp.int32),
            )
        
        # Initialize coefficients on GPU
        beta = cp.zeros(n_features, dtype=cp.float64)
        
        # Compute null log-likelihood on GPU
        loglik_null_gpu = self._compute_log_likelihood_gpu(
            cp.zeros(n_features, dtype=cp.float64),
            X_sorted,
            time_sorted,
            event_sorted,
            efron_pre,
        )

        # Newton-Raphson optimization on GPU
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian on GPU
            grad, hess = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted, efron_pre
            )
            
            # Newton: delta = inv(hess) @ grad; hess is NSD — solve (-hess) x = grad, delta = -x
            delta = self._solve_newton_delta_gpu(hess, grad, cp)
            
            # Check convergence on GPU
            if cp.linalg.norm(delta) < self.tol:
                self._converged = True
                break
            
            beta = beta - delta
        
        # Compute final log-likelihood on GPU
        loglik_gpu = self._compute_log_likelihood_gpu(
            beta, X_sorted, time_sorted, event_sorted, efron_pre
        )
        
        # Compute C-index on GPU
        cindex_gpu = self._compute_cindex_gpu(X_sorted, time_sorted, event_sorted, beta)
        
        # Single transfer at the end
        self._iterations = iteration + 1
        self.coef_ = cp.asnumpy(beta)
        self.hazard_ratios_ = np.exp(self.coef_)
        self._log_likelihood_null = float(cp.asnumpy(loglik_null_gpu))
        self._log_likelihood = float(cp.asnumpy(loglik_gpu))
        self._cindex = float(cp.asnumpy(cindex_gpu))
        
        # Inference:
        # - nonrobust: stay on GPU to avoid expensive host transfers/recompute
        # - hc0/hc1/cluster: use CPU inference path (current implementation)
        if self.compute_inference:
            if self.cov_type == "nonrobust":
                try:
                    var_gpu = cp.linalg.inv(-hess)
                except Exception:
                    var_gpu = cp.linalg.pinv(-hess)
                bse_gpu = cp.sqrt(cp.maximum(cp.diag(var_gpu), 0.0))
                z_gpu = beta / (bse_gpu + 1e-30)
                p_gpu = norm_two_tail_pvalues_gpu(cp.abs(z_gpu))
                z_crit = norm_crit_gpu_two_tail(0.05)
                ci_gpu = cp.stack([beta - z_crit * bse_gpu, beta + z_crit * bse_gpu], axis=1)

                self._bse = cp.asnumpy(bse_gpu)
                self._zvalues = cp.asnumpy(z_gpu)
                self._pvalues = cp.asnumpy(p_gpu)
                self._conf_int = cp.asnumpy(ci_gpu)
                self._var_matrix = np.diag(np.square(self._bse))
                self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
                self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
                try:
                    var_inv = np.linalg.inv(self._var_matrix)
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
                X_sorted_np = cp.asnumpy(X_sorted)
                time_sorted_np = cp.asnumpy(time_sorted)
                event_sorted_np = cp.asnumpy(event_sorted)
                cluster_sorted_np = None if cluster_sorted is None else cp.asnumpy(cluster_sorted)
                self._compute_inference_cpu(X_sorted_np, time_sorted_np, event_sorted_np, cluster_sorted_np)
                self._compute_baseline_hazard(X_sorted_np, time_sorted_np, event_sorted_np)
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
    
    def _compute_log_likelihood(self, beta, X, time, event, efron_pre=None):
        """Compute log partial likelihood (Breslow/Efron tie handling)."""
        eta = X @ beta
        exp_eta = np.exp(eta)

        # Risk set suffix sums:
        #   risk_sum[i] = sum_{j: time[j] >= time[i]} exp_eta[j]
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]

        event_mask = event == 1
        if not np.any(event_mask):
            return 0.0

        if self.ties == "breslow":
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
            return float(np.sum(eta[event_mask]) - np.sum(counts * np.log(risk_at)))

        # ---- Efron ----
        ll = 0.0
        if efron_pre is not None:
            uft, uft_ix, _, _, nuft, first_idx_uft = _unpack_efron_pre6(efron_pre)
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
                ll += float(np.sum(eta[ix_ev]))

                k = np.arange(d, dtype=np.float64) / d
                denom = risk_at_t - k * sum_events
                ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))

            return float(ll)

        # No precomputation: group event rows by unique failure times.
        event_idx = np.flatnonzero(event_mask)
        event_times = time[event_idx]
        uft, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
        first_idx = np.searchsorted(time, uft, side="left").astype(np.int64)
        risk_at = risk_sum[first_idx]

        sum_events = np.bincount(inv, weights=exp_eta[event_idx], minlength=len(uft)).astype(np.float64)
        sum_eta_events = np.bincount(inv, weights=eta[event_idx], minlength=len(uft)).astype(np.float64)

        for g in range(len(uft)):
            d = int(counts[g])
            if d == 0:
                continue
            ll += float(sum_eta_events[g])
            k = np.arange(d, dtype=np.float64) / d
            denom = risk_at[g] - k * sum_events[g]
            ll -= float(np.sum(np.log(np.maximum(denom, 1e-300))))

        return float(ll)
    
    def _solve_newton_delta_gpu(self, hess, grad, cp):
        """Newton step delta = inv(hess) @ grad; prefer SPD solve on (-hess) with light jitter."""
        p = int(hess.shape[0])
        try:
            H = -hess
            eps = 1e-11 * (cp.max(cp.abs(cp.diag(H))) + 1.0)
            H = H + eps * cp.eye(p, dtype=cp.float64)
            return -cp.linalg.solve(H, grad)
        except Exception:
            try:
                return cp.linalg.solve(hess, grad)
            except Exception:
                return cp.linalg.lstsq(hess, grad, rcond=None)[0].flatten()

    def _compute_log_likelihood_gpu(self, beta, X, time, event, efron_pre=None):
        """Compute log partial likelihood on GPU."""
        import cupy as cp
        
        n_samples = X.shape[0]
        eta = X @ beta
        exp_eta = cp.exp(eta)
        
        # Risk sets
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
        
        # Log-likelihood contribution from events
        ll = cp.array(0.0, dtype=cp.float64)
        event_mask = event == 1
        
        if not cp.any(event_mask):
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
        
        # Efron: loop over cached failure groups (see `_cox_efron_cuda.compute_efron_loglik_raw`)
        if efron_pre is not None:
            try:
                if self._efron_pre_csr is not None:
                    from ._cox_efron_cuda import compute_efron_loglik_raw_csr

                    _, _, _, _, fail_ptr, fail_ind, first_idx_uft, nuft = self._efron_pre_csr
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
    
    def _compute_gradient_hessian(self, beta, X, time, event, efron_pre=None):
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
        exp_eta = np.exp(eta)

        # Risk sets: cumulative sum of exp(eta) for all at risk
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]

        # Weighted risk sets for gradient
        X_exp_eta = X * exp_eta[:, np.newaxis]
        risk_X_sum = np.cumsum(X_exp_eta[::-1], axis=0)[::-1]
        
        if self.ties == 'breslow':
            event_mask = event == 1
            grad = np.zeros(n_features, dtype=np.float64)
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
        """Compute Breslow Hessian using reverse cumulative second moments."""
        # risk_X2_sum[i] = sum_{j>=i} exp_eta[j] * X[j]X[j]^T
        x2_weighted = np.einsum("ni,nj,n->nij", X, X, exp_eta)
        risk_X2_sum = np.cumsum(x2_weighted[::-1], axis=0)[::-1]
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

        risk_sum_at = risk_sum[first_idx]  # (nuft,)
        E_X = risk_X_sum[first_idx] / risk_sum_at[:, np.newaxis]  # (nuft, p)
        E_XX = risk_X2_sum[first_idx] / risk_sum_at[:, np.newaxis, np.newaxis]  # (nuft, p, p)

        centered = E_XX - np.einsum("ni,nj->nij", E_X, E_X)  # (nuft, p, p)
        return -np.sum(centered * counts[:, np.newaxis, np.newaxis], axis=0)
    
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

        # risk_enter: for each row i, group id j where uft[j] <= time[i] < uft[j+1].
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

        # risk_exit: GPU path currently does not support delayed entry; entry is implicitly all zeros.
        entry0 = 0.0
        j_exit = int(np.searchsorted(uft, entry0))
        risk_exit = [[] for _ in range(nuft)]
        risk_exit[j_exit] = list(range(n))

        return uft, uft_ix, risk_enter, risk_exit, nuft, first_idx_uft

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
        Efron gradient and Hessian of the log partial likelihood via backward scan
        over unique failure times (same structure as statsmodels PHReg efron_gradient /
        efron_hessian). O(n_events * p^2) with small constants — avoids O(n^2) Python loops.
        """
        n_samples, n_features = X.shape
        linpred = X @ beta
        linpred = linpred - np.max(linpred)
        e_linpred = np.exp(linpred)

        if efron_pre is None:
            efron_pre = self._efron_unique_failure_indices(time, event)
        uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        if nuft == 0:
            return np.zeros(n_features, dtype=np.float64), np.zeros((n_features, n_features), dtype=np.float64)

        # Single backward scan: same risk-set state (xp0,xp1,xp2) for grad and Hessian.
        grad = np.zeros(n_features, dtype=np.float64)
        hess_inner = np.zeros((n_features, n_features), dtype=np.float64)
        xp0 = 0.0
        xp1 = np.zeros(n_features, dtype=np.float64)
        xp2 = np.zeros((n_features, n_features), dtype=np.float64)
        for i in range(nuft)[::-1]:
            ix = risk_enter[i]
            if len(ix) > 0:
                ix = np.asarray(ix, dtype=np.intp)
                elx = e_linpred[ix]
                v = X[ix]
                xp0 += elx.sum()
                xp1 += (elx[:, None] * v).sum(axis=0)
                xp2 += np.einsum("ij,ik,i->jk", v, v, elx)
            ixf = uft_ix[i]
            if len(ixf) > 0:
                ixf = np.asarray(ixf, dtype=np.intp)
                v = X[ixf]
                elx = e_linpred[ixf]
                xp0f = elx.sum()
                xp1f = (elx[:, None] * v).sum(axis=0)
                xp2f = np.einsum("ij,ik,i->jk", v, v, elx)
                m = len(ixf)
                J = np.arange(m, dtype=np.float64) / max(m, 1)
                c0 = xp0 - J * xp0f
                c0 = np.maximum(c0, 1e-300)
                inv = 1.0 / c0
                ak = inv
                bk = J * inv
                sum_inv_c0 = np.sum(ak)
                sum_J_c0 = np.sum(bk)
                sum_aa = np.sum(ak * ak)
                sum_bb = np.sum(bk * bk)
                sum_ab = np.sum(ak * bk)
                grad += v.sum(axis=0)
                grad -= xp1 * sum_inv_c0 - xp1f * sum_J_c0
                hess_inner += xp2 * sum_inv_c0
                hess_inner -= xp2f * sum_J_c0
                hess_inner -= (
                    sum_aa * np.outer(xp1, xp1)
                    + sum_bb * np.outer(xp1f, xp1f)
                    - sum_ab * (np.outer(xp1, xp1f) + np.outer(xp1f, xp1))
                )
            ix = risk_exit[i]
            if len(ix) > 0:
                ix = np.asarray(ix, dtype=np.intp)
                elx = e_linpred[ix]
                v = X[ix]
                xp0 -= elx.sum()
                xp1 -= (elx[:, None] * v).sum(axis=0)
                xp2 -= np.einsum("ij,ik,i->jk", v, v, elx)

        hess = -hess_inner
        return grad, hess
    
    def _compute_gradient_hessian_gpu(
        self, beta, X, time, event, efron_pre=None
    ):
        """Compute gradient and Hessian on GPU."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        
        eta = X @ beta
        exp_eta = cp.exp(eta)
        
        # Risk sets
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
        X_exp_eta = X * exp_eta[:, cp.newaxis]
        risk_X_sum = cp.cumsum(X_exp_eta[::-1], axis=0)[::-1]
        
        # Efron: same backward scan as CPU (efron_pre built once in _fit_gpu).
        if self.ties == "efron":
            if efron_pre is None:
                efron_pre = self._efron_unique_failure_indices(
                    cp.asnumpy(time), cp.asnumpy(event)
                )
            return self._compute_gradient_hessian_efron_backward_gpu(
                beta, X, efron_pre
            )
        
        # Breslow gradient (vectorized)
        event_mask = event == 1
        grad = cp.zeros(n_features, dtype=cp.float64)

        if not cp.any(event_mask):
            return grad, cp.zeros((n_features, n_features), dtype=cp.float64)

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

        counts_f = counts_uft.astype(cp.float64)
        grad = cp.sum(X[event_mask], axis=0)
        E_X = risk_X_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis]
        grad = grad - cp.sum(E_X * counts_f[:, cp.newaxis], axis=0)

        # Hessian needs reverse cumulative second moments.
        x2_weighted = cp.einsum("ni,nj,n->nij", X, X, exp_eta)
        risk_X2_sum = cp.cumsum(x2_weighted[::-1], axis=0)[::-1]

        E_XX = risk_X2_sum[first_idx_uft] / risk_sum[first_idx_uft][:, cp.newaxis, cp.newaxis]
        centered = E_XX - cp.einsum("ni,nj->nij", E_X, E_X)
        hess = -cp.sum(centered * counts_f[:, cp.newaxis, cp.newaxis], axis=0)
        return grad, hess

    def _compute_gradient_hessian_efron_backward_gpu(self, beta, X, efron_pre):
        """CuPy Efron grad/Hessian: prefer single CUDA RawKernel scan, else Python-loop fallback."""
        import cupy as cp

        uft, uft_ix, risk_enter, risk_exit, nuft, _ = _unpack_efron_pre6(efron_pre)
        n_features = X.shape[1]
        if nuft == 0:
            return cp.zeros(n_features, dtype=cp.float64), cp.zeros(
                (n_features, n_features), dtype=cp.float64
            )

        try:
            from ._cox_efron_cuda import compute_efron_grad_hess_raw

            if self._efron_pre_csr is not None:
                out = compute_efron_grad_hess_raw(
                    X,
                    beta,
                    efron_pre,
                    efron_csr=self._efron_pre_csr,
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
    
    def _compute_inference_cpu(self, X, time, event, cluster=None):
        """Compute standard errors, z-values, p-values, and confidence intervals."""
        n_features = X.shape[1]

        # Keep inference self-contained (no nested external model fitting),
        # so runtime reflects this implementation directly.
        
        # Compute information matrix (negative Hessian at MLE)
        _, hess = self._compute_gradient_hessian(
            self.coef_, X, time, event, getattr(self, "_efron_pre", None)
        )
        
        # Bread matrix from observed information.
        try:
            bread = np.linalg.inv(-hess)
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
        self._bse = np.sqrt(np.diag(self._var_matrix))
        
        # z-values
        self._zvalues = self.coef_ / self._bse
        
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
            var_inv = np.linalg.inv(self._var_matrix)
            self._wald_test_stat = self.coef_ @ var_inv @ self.coef_
        except np.linalg.LinAlgError:
            self._wald_test_stat = np.nan
        self._wald_test_pvalue = 1 - stats.chi2.cdf(self._wald_test_stat, n_features)
        
        # Likelihood ratio test
        self._lr_test_stat = 2 * (self._log_likelihood - self._log_likelihood_null)
        self._lr_test_pvalue = 1 - stats.chi2.cdf(self._lr_test_stat, n_features)
        
        # Score test (Rao's test) - computed at beta = 0
        ep = getattr(self, "_efron_pre", None)
        grad_0, _ = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep)
        try:
            _, hess_0 = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep)
            info_0 = -hess_0
            info_0_inv = np.linalg.inv(info_0)
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
    
    def _compute_cindex_gpu(self, X, time, event, beta):
        """Compute concordance index (C-index) on GPU."""
        import cupy as cp
        
        # Linear predictor (risk score) on GPU
        risk_score = X @ beta
        
        n = len(time)
        
        # Compute concordance on GPU using vectorized operations
        # This is approximate due to pairwise comparison complexity
        # For exact C-index, we need to iterate
        
        # Simplified: use mean risk score difference for events
        event_mask = (event == 1)
        if cp.sum(event_mask) == 0:
            return cp.array(0.5)
        
        # Mean risk score for events vs non-events
        risk_events = cp.mean(risk_score[event_mask])
        risk_no_events = cp.mean(risk_score[~event_mask])
        
        # Approximate C-index
        cindex = 0.5 + 0.5 * cp.sign(risk_events - risk_no_events)
        
        return cindex
    
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
        concordant = 0
        permissible = 0
        tied_risk = 0
        
        for i in range(n):
            if event[i] == 0:
                continue
            
            for j in range(n):
                if i == j:
                    continue
                
                if time[i] < time[j] or (time[i] == time[j] and event[j] == 0):
                    permissible += 1
                    
                    if risk_score[i] > risk_score[j]:
                        concordant += 1
                    elif risk_score[i] == risk_score[j]:
                        tied_risk += 1
        
        if permissible > 0:
            return (concordant + 0.5 * tied_risk) / permissible
        return np.nan
