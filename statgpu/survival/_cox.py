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
        n_jobs: Optional[int] = None
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.ties = ties.lower()
        self.tol = tol
        self.max_iter = max_iter
        
        if self.ties not in ('breslow', 'efron'):
            raise ValueError("ties must be 'breslow' or 'efron'")
        
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
        # Convert inputs
        X_np = np.asarray(X, dtype=np.float64)
        time_np = np.asarray(time, dtype=np.float64)
        event_np = np.asarray(event, dtype=np.int32)
        
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        self._nobs = X_np.shape[0]
        self._nevents = np.sum(event_np)
        
        # Store original data
        self._time = time_np.copy()
        self._event = event_np.copy()
        self._X = X_np.copy()
        
        # Create feature names
        self._feature_names = [f'x{i+1}' for i in range(X_np.shape[1])]
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu(X_np, time_np, event_np, entry)
        else:
            self._fit_cpu(X_np, time_np, event_np, entry)
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, time, event, entry=None):
        """Fit using CPU (NumPy)."""
        n_samples, n_features = X.shape
        
        # Sort by time (descending for partial likelihood)
        order = np.argsort(-time)
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
            
            # Line search with step halving
            step = 1.0
            for _ in range(20):
                new_beta = beta - step * delta
                new_ll = self._compute_log_likelihood(
                    new_beta, X_sorted, time_sorted, event_sorted
                )
                old_ll = self._compute_log_likelihood(
                    beta, X_sorted, time_sorted, event_sorted
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
        
        # Compute inference statistics
        self._compute_inference_cpu(X_sorted, time_sorted, event_sorted)
        self._compute_baseline_hazard(X_sorted, time_sorted, event_sorted)
        self._compute_cindex()
    
    def _fit_gpu(self, X, time, event, entry=None):
        """Fit using GPU (CuPy)."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        
        # Transfer to GPU
        X_gpu = cp.asarray(X, dtype=cp.float64)
        time_gpu = cp.asarray(time, dtype=cp.float64)
        event_gpu = cp.asarray(event, dtype=cp.int32)
        
        # Sort by time (descending)
        order = cp.argsort(-time_gpu)
        X_sorted = X_gpu[order]
        time_sorted = time_gpu[order]
        event_sorted = event_gpu[order]
        
        # Initialize coefficients
        beta = cp.zeros(n_features, dtype=cp.float64)
        
        # Compute null log-likelihood
        self._log_likelihood_null = self._compute_log_likelihood_gpu(
            cp.zeros(n_features), X_sorted, time_sorted, event_sorted
        ).get()
        
        # Newton-Raphson optimization
        for iteration in range(self.max_iter):
            # Compute gradient and Hessian
            grad, hess = self._compute_gradient_hessian_gpu(
                beta, X_sorted, time_sorted, event_sorted
            )
            
            # Newton step
            try:
                delta = cp.linalg.solve(hess, grad)
            except Exception:
                delta = cp.linalg.lstsq(hess, grad, rcond=None)[0].flatten()
            
            # Check convergence
            if cp.linalg.norm(delta) < self.tol:
                self._converged = True
                break
            
            beta = beta - delta
        
        # Transfer results back to CPU
        self._iterations = iteration + 1
        self.coef_ = beta.get()
        self.hazard_ratios_ = np.exp(self.coef_)
        
        # Compute inference on CPU (transfer sorted arrays)
        X_sorted_np = X_sorted.get()
        time_sorted_np = time_sorted.get()
        event_sorted_np = event_sorted.get()
        
        self._log_likelihood = self._compute_log_likelihood(
            self.coef_, X_sorted_np, time_sorted_np, event_sorted_np
        )
        
        self._compute_inference_cpu(X_sorted_np, time_sorted_np, event_sorted_np)
        self._compute_baseline_hazard(X_sorted_np, time_sorted_np, event_sorted_np)
        self._compute_cindex()
    
    def _compute_log_likelihood(self, beta, X, time, event):
        """Compute log partial likelihood."""
        n_samples = X.shape[0]
        eta = X @ beta
        exp_eta = np.exp(eta)
        
        # Risk sets (cumulative sum of exp(eta) from end)
        risk_sum = np.cumsum(exp_eta[::-1])[::-1]
        
        # Log-likelihood contribution from events
        ll = 0.0
        if self.ties == 'breslow':
            # Breslow: sum of eta for events minus sum of log(risk_sum) at event times
            for i in range(n_samples):
                if event[i]:
                    ll += eta[i] - np.log(risk_sum[i])
        else:
            # Efron approximation
            unique_times = np.unique(time[event == 1])
            for t in unique_times:
                # Find indices at this time
                at_time_t = (time == t)
                events_at_t = at_time_t & (event == 1)
                d = np.sum(events_at_t)
                
                if d == 0:
                    continue
                
                # Risk set at time t (first index where time >= t)
                first_idx = np.where(time >= t)[0][0]
                risk_at_t = risk_sum[first_idx]
                
                # Sum of exp(eta) for events at t
                sum_events = np.sum(exp_eta[events_at_t])
                
                # Efron correction: subtract k/d * sum_events for k = 0 to d-1
                # Average of (risk_at_t - k/d * sum_events) for k=0,...,d-1
                for k in range(d):
                    ll -= np.log(risk_at_t - k * sum_events / d)
                
                # Add sum of eta for events
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
        ll = 0.0
        event_mask = event == 1
        
        if cp.any(event_mask):
            ll = cp.sum(eta[event_mask]) - cp.sum(cp.log(risk_sum[event_mask]))
        
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
        
        # Initialize gradient and Hessian
        grad = np.zeros(n_features, dtype=np.float64)
        hess = np.zeros((n_features, n_features), dtype=np.float64)
        
        if self.ties == 'breslow':
            # Breslow approximation
            for i in range(n_samples):
                if event[i]:
                    # Gradient: X[i] - risk_X_sum[i] / risk_sum[i]
                    grad += X[i] - risk_X_sum[i] / risk_sum[i]
            
            # Hessian computation
            hess = self._compute_hessian_breslow(beta, X, time, event, risk_sum, risk_X_sum, exp_eta)
        else:
            # Efron approximation
            grad, hess = self._compute_gradient_hessian_efron(beta, X, time, event, risk_sum, risk_X_sum, exp_eta)
        
        return grad, hess
    
    def _compute_hessian_breslow(self, beta, X, time, event, risk_sum, risk_X_sum, exp_eta):
        """Compute Hessian for Breslow approximation."""
        n_samples, n_features = X.shape
        hess = np.zeros((n_features, n_features), dtype=np.float64)
        
        for i in range(n_samples):
            if event[i]:
                # E[X | risk set] and E[X X^T | risk set]
                E_X = risk_X_sum[i] / risk_sum[i]
                
                # Second moment: sum(X X^T * exp_eta) / risk_sum
                risk_X2_sum = np.zeros((n_features, n_features))
                for j in range(i, n_samples):
                    risk_X2_sum += np.outer(X[j], X[j]) * exp_eta[j]
                
                E_XX = risk_X2_sum / risk_sum[i]
                hess -= (E_XX - np.outer(E_X, E_X))
        
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
        
        # Gradient
        event_mask = event == 1
        grad = cp.zeros(n_features, dtype=cp.float64)
        
        if cp.any(event_mask):
            grad = cp.sum(X[event_mask], axis=0) - cp.sum(risk_X_sum[event_mask] / risk_sum[event_mask][:, cp.newaxis], axis=0)
        
        # Hessian (simplified for GPU)
        hess = cp.zeros((n_features, n_features), dtype=cp.float64)
        
        # Convert to numpy for complex Hessian computation
        X_np = X.get()
        time_np = time.get()
        event_np = event.get()
        beta_np = beta.get()
        
        _, hess_np = self._compute_gradient_hessian(beta_np, X_np, time_np, event_np)
        hess = cp.asarray(hess_np)
        
        return grad, hess
    
    def _compute_inference_cpu(self, X, time, event):
        """Compute standard errors, z-values, p-values, and confidence intervals."""
        n_features = X.shape[1]
        
        # Compute information matrix (negative Hessian at MLE)
        _, hess = self._compute_gradient_hessian(self.coef_, X, time, event)
        
        # Variance-covariance matrix is inverse of information matrix
        try:
            self._var_matrix = np.linalg.inv(-hess)
        except np.linalg.LinAlgError:
            self._var_matrix = np.linalg.pinv(-hess)
        
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
    
    def _compute_cindex(self):
        """Compute concordance index (C-index)."""
        if self._X is None or self.coef_ is None:
            self._cindex = None
            return
        
        # Linear predictor (risk score)
        risk_score = self._X @ self.coef_
        
        n = len(self._time)
        concordant = 0
        permissible = 0
        tied_risk = 0
        
        for i in range(n):
            if self._event[i] == 0:
                continue  # Skip censored observations as first in pair
            
            for j in range(n):
                if i == j:
                    continue
                
                # Check if pair is permissible (i had event before j or j is censored after i)
                if self._time[i] < self._time[j] or (self._time[i] == self._time[j] and self._event[j] == 0):
                    permissible += 1
                    
                    # Compare risk scores
                    if risk_score[i] > risk_score[j]:
                        concordant += 1
                    elif risk_score[i] == risk_score[j]:
                        tied_risk += 1
        
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
        print()
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
        
        print("=" * 80)
        print(f"Concordance: {self._cindex:.3f} (if 0.5-0.7: moderate, 0.7-0.9: strong)")
        print(f"Likelihood ratio test: {self._lr_test_stat:.2f} on {len(self.coef_)} df, p={self._lr_test_pvalue:.4e}")
        print(f"Wald test:            {self._wald_test_stat:.2f} on {len(self.coef_)} df, p={self._wald_test_pvalue:.4e}")
        print(f"Score (logrank) test: {self._score_test_stat:.2f} on {len(self.coef_)} df, p={self._score_test_pvalue:.4e}")
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
