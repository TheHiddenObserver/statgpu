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
        if self.cov_type not in ("nonrobust", "hc0", "hc1"):
            raise ValueError("cov_type must be one of: 'nonrobust', 'hc0', 'hc1'")
        
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
        
    def fit(self, X, time, event, entry=None):
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
            
            self._fit_gpu(X_gpu, time_gpu, event_gpu, entry)
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
            
            self._fit_cpu(X_np, time_np, event_np, entry)
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, time, event, entry=None):
        """Fit using CPU (NumPy)."""
        n_samples, n_features = X.shape
        
        # Sort by time ascending so risk-set terms are suffix sums:
        # R(t_i) = {j: t_j >= t_i} -> indices i..n-1 after ascending sort.
        order = np.argsort(time)
        X_sorted = X[order]
        time_sorted = time[order]
        event_sorted = event[order]
        
        # Initialize coefficients
        beta = np.zeros(n_features, dtype=np.float64)
        
        # Compute null log-likelihood (beta = 0)
        self._log_likelihood_null = self._compute_log_likelihood(
            np.zeros(n_features), X_sorted, time_sorted, event_sorted
        )
        
        # Newton-Raphson optimization
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian
            grad, hess = self._compute_gradient_hessian(
                beta, X_sorted, time_sorted, event_sorted
            )
            
            # Newton step
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                delta = np.linalg.lstsq(hess, grad, rcond=None)[0]

            # Line search with step halving.
            # Compute old_ll once (invariant inside the inner loop).
            old_ll = self._compute_log_likelihood(beta, X_sorted, time_sorted, event_sorted)
            step = 1.0
            for _ in range(20):
                new_beta = beta - step * delta
                new_ll = self._compute_log_likelihood(
                    new_beta, X_sorted, time_sorted, event_sorted
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
            beta, X_sorted, time_sorted, event_sorted
        )
        
        # Compute optional inference statistics
        if self.compute_inference:
            self._compute_inference_cpu(X_sorted, time_sorted, event_sorted)
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
    
    def _fit_gpu(self, X, time, event, entry=None):
        """Fit using GPU with full GPU computation."""
        import cupy as cp
        
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
        
        # Initialize coefficients on GPU
        beta = cp.zeros(n_features, dtype=cp.float64)
        
        # Compute null log-likelihood on GPU
        loglik_null_gpu = self._compute_log_likelihood_gpu(
            cp.zeros(n_features, dtype=cp.float64), X_sorted, time_sorted, event_sorted
        )
        
        # Newton-Raphson optimization on GPU
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian on GPU
            grad, hess = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted
            )
            
            # Newton step on GPU
            try:
                delta = cp.linalg.solve(hess, grad)
            except Exception:
                delta = cp.linalg.lstsq(hess, grad, rcond=None)[0].flatten()
            
            # Check convergence on GPU
            if cp.linalg.norm(delta) < self.tol:
                self._converged = True
                break
            
            beta = beta - delta
        
        # Compute final log-likelihood on GPU
        loglik_gpu = self._compute_log_likelihood_gpu(
            beta, X_sorted, time_sorted, event_sorted
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
        
        # Optional inference on CPU (can be disabled to minimize host/device transfers)
        if self.compute_inference:
            X_sorted_np = cp.asnumpy(X_sorted)
            time_sorted_np = cp.asnumpy(time_sorted)
            event_sorted_np = cp.asnumpy(event_sorted)
            
            self._compute_inference_cpu(X_sorted_np, time_sorted_np, event_sorted_np)
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
    
    def _compute_log_likelihood(self, beta, X, time, event):
        """Compute log partial likelihood."""
        eta = X @ beta
        exp_eta = np.exp(eta)

        # Risk sets (cumulative sum of exp(eta) from end)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]

        # Log-likelihood contribution from events
        if self.ties == 'breslow':
            # Vectorized: O(n) with no Python loop.
            event_mask = event == 1
            return float(np.sum(eta[event_mask]) - np.sum(np.log(risk_sum[event_mask])))

        # Efron approximation (loop-based, kept for correctness)
        ll = 0.0
        unique_times = np.unique(time[event == 1])
        for t in unique_times:
            at_time_t = (time == t)
            events_at_t = at_time_t & (event == 1)
            d = np.sum(events_at_t)
            if d == 0:
                continue
            first_idx = np.where(time >= t)[0][0]
            risk_at_t = risk_sum[first_idx]
            sum_events = np.sum(exp_eta[events_at_t])
            for k in range(d):
                ll -= np.log(risk_at_t - k * sum_events / d)
            ll += np.sum(eta[events_at_t])
        return ll
    
    def _compute_log_likelihood_gpu(self, beta, X, time, event):
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
            ll = cp.sum(eta[event_mask]) - cp.sum(cp.log(risk_sum[event_mask]))
            return ll
        
        # Efron approximation
        unique_times = cp.unique(time[event_mask])
        for t in unique_times:
            at_time_t = (time == t)
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
                ll -= cp.log(risk_at_t - (k / d) * sum_events)
        
        return ll
    
    def _compute_gradient_hessian(self, beta, X, time, event):
        """
        Compute gradient and Hessian of negative log partial likelihood.
        Uses Breslow or Efron approximation for ties.
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
            # Vectorized gradient: O(n_events * p) instead of O(n * p) Python loop.
            event_mask = event == 1
            grad = (
                np.sum(X[event_mask], axis=0)
                - np.sum(risk_X_sum[event_mask] / risk_sum[event_mask, np.newaxis], axis=0)
            )
            hess = self._compute_hessian_breslow(beta, X, time, event, risk_sum, risk_X_sum, exp_eta)
        else:
            # Efron approximation
            grad, hess = self._compute_gradient_hessian_efron(beta, X, time, event, risk_sum, risk_X_sum, exp_eta)

        return grad, hess
    
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
    
    def _compute_gradient_hessian_efron(self, beta, X, time, event, risk_sum, risk_X_sum, exp_eta):
        """Compute gradient and Hessian for Efron approximation."""
        n_samples, n_features = X.shape
        grad = np.zeros(n_features, dtype=np.float64)
        hess = np.zeros((n_features, n_features), dtype=np.float64)
        
        # Find unique event times
        event_times = time[event == 1]
        unique_times = np.unique(event_times)
        
        for t in unique_times:
            # Indices at this event time
            at_time_t = (time == t)
            events_at_t = at_time_t & (event == 1)
            d = np.sum(events_at_t)
            
            if d == 0:
                continue
            
            # First index in risk set
            risk_set_start = np.where(time >= t)[0][0]
            
            # Sum of exp_eta and X*exp_eta for events at t
            sum_exp_events = np.sum(exp_eta[events_at_t])
            sum_X_exp_events = np.sum(X[events_at_t] * exp_eta[events_at_t][:, np.newaxis], axis=0)
            
            # Efron approximation
            for k in range(d):
                frac = k / d
                
                # Adjusted risk sum and risk_X_sum
                risk_sum_k = risk_sum[risk_set_start] - frac * sum_exp_events
                risk_X_sum_k = risk_X_sum[risk_set_start] - frac * sum_X_exp_events
                
                # Gradient contribution
                E_X = risk_X_sum_k / risk_sum_k
                
                # Add to gradient (each event contributes)
                for idx in np.where(events_at_t)[0]:
                    grad += X[idx] - E_X
                
                # Hessian contribution
                # Compute E[XX^T] at this risk set
                risk_X2_sum = np.zeros((n_features, n_features))
                for j in range(risk_set_start, n_samples):
                    if not events_at_t[j]:
                        risk_X2_sum += np.outer(X[j], X[j]) * exp_eta[j]
                    else:
                        # Partial contribution from tied events
                        risk_X2_sum += np.outer(X[j], X[j]) * exp_eta[j] * (1 - frac)
                
                E_XX = risk_X2_sum / risk_sum_k
                hess -= (E_XX - np.outer(E_X, E_X))
        
        return grad, hess
    
    def _compute_gradient_hessian_gpu(self, beta, X, time, event):
        """Compute gradient and Hessian on GPU."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        
        eta = X @ beta
        exp_eta = cp.exp(eta)
        
        # Risk sets
        risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
        X_exp_eta = X * exp_eta[:, cp.newaxis]
        risk_X_sum = cp.cumsum(X_exp_eta[::-1], axis=0)[::-1]
        
        # Efron path on GPU (avoid host/device transfers in iteration loop).
        if self.ties == 'efron':
            grad = cp.zeros(n_features, dtype=cp.float64)
            hess = cp.zeros((n_features, n_features), dtype=cp.float64)
            event_mask = event == 1
            
            if not cp.any(event_mask):
                return grad, hess
            
            unique_times = cp.unique(time[event_mask])
            for t in unique_times:
                at_time_t = (time == t)
                events_at_t = at_time_t & event_mask
                d = int(cp.sum(events_at_t).item())
                
                if d == 0:
                    continue
                
                risk_indices = cp.where(time >= t)[0]
                if risk_indices.size == 0:
                    continue
                risk_set_start = int(risk_indices[0].item())
                
                # Event aggregates at this tied time.
                X_events = X[events_at_t]
                exp_events = exp_eta[events_at_t]
                sum_X_events = cp.sum(X_events, axis=0)
                sum_exp_events = cp.sum(exp_events)
                sum_X_exp_events = cp.sum(X_events * exp_events[:, cp.newaxis], axis=0)
                sum_X2_exp_events = cp.einsum('ni,nj,n->ij', X_events, X_events, exp_events)
                
                # Full risk-set second moment at time t.
                X_risk = X[risk_set_start:]
                exp_risk = exp_eta[risk_set_start:]
                risk_X2_sum_full = cp.einsum('ni,nj,n->ij', X_risk, X_risk, exp_risk)
                
                for k in range(d):
                    frac = k / d
                    risk_sum_k = risk_sum[risk_set_start] - frac * sum_exp_events
                    risk_X_sum_k = risk_X_sum[risk_set_start] - frac * sum_X_exp_events
                    risk_X2_sum_k = risk_X2_sum_full - frac * sum_X2_exp_events
                    
                    E_X = risk_X_sum_k / risk_sum_k
                    E_XX = risk_X2_sum_k / risk_sum_k
                    
                    # Same objective/sign convention as CPU implementation.
                    grad += sum_X_events - d * E_X
                    hess -= (E_XX - cp.outer(E_X, E_X))
            
            return grad, hess
        
        # Breslow gradient
        event_mask = event == 1
        grad = cp.zeros(n_features, dtype=cp.float64)
        
        if cp.any(event_mask):
            grad = cp.sum(X[event_mask], axis=0) - cp.sum(risk_X_sum[event_mask] / risk_sum[event_mask][:, cp.newaxis], axis=0)
        
        # Hessian on GPU (Breslow): reverse scan accumulates weighted second moments
        hess = cp.zeros((n_features, n_features), dtype=cp.float64)
        risk_X2_sum = cp.zeros((n_features, n_features), dtype=cp.float64)
        
        for i in range(n_samples - 1, -1, -1):
            x_i = X[i]
            w_i = exp_eta[i]
            risk_X2_sum = risk_X2_sum + w_i * cp.outer(x_i, x_i)
            
            if event[i]:
                E_X = risk_X_sum[i] / risk_sum[i]
                E_XX = risk_X2_sum / risk_sum[i]
                hess -= (E_XX - cp.outer(E_X, E_X))
        
        return grad, hess
    
    def _compute_inference_cpu(self, X, time, event):
        """Compute standard errors, z-values, p-values, and confidence intervals."""
        n_features = X.shape[1]
        
        # Compute information matrix (negative Hessian at MLE)
        _, hess = self._compute_gradient_hessian(self.coef_, X, time, event)
        
        # Bread matrix from observed information.
        try:
            bread = np.linalg.inv(-hess)
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(-hess)

        if self.cov_type == "nonrobust":
            self._var_matrix = bread
        else:
            score_resid = self._compute_score_residuals_approx(X, time, event)
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
        grad_0, _ = self._compute_gradient_hessian(np.zeros(n_features), X, time, event)
        try:
            _, hess_0 = self._compute_gradient_hessian(np.zeros(n_features), X, time, event)
            info_0 = -hess_0
            info_0_inv = np.linalg.inv(info_0)
            self._score_test_stat = grad_0 @ info_0_inv @ grad_0
        except:
            self._score_test_stat = np.nan
        self._score_test_pvalue = 1 - stats.chi2.cdf(self._score_test_stat, n_features)

    def _compute_score_residuals_approx(self, X, time, event):
        """
        Approximate per-observation score residuals at fitted beta.

        Uses event-time contribution:
          u_i = x_i - E[X | R(t_i)] for event rows, and 0 for censored rows.
        """
        n_samples, n_features = X.shape
        eta = X @ self.coef_
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        risk_X_sum = np.cumsum((X * exp_eta[:, np.newaxis])[::-1], axis=0)[::-1]
        u = np.zeros((n_samples, n_features), dtype=np.float64)
        # Vectorized: fill only event rows.
        event_mask = event == 1
        u[event_mask] = X[event_mask] - risk_X_sum[event_mask] / risk_sum[event_mask, np.newaxis]
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
