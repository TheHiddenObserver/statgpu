"""
Cox Proportional Hazards regression with GPU acceleration.

Implements Cox PH model using Breslow approximation for ties with
Newton-Raphson optimization. Matches R's survival::coxph() API.
"""

from typing import Optional, Union, Tuple, Dict, Any
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
            
            # Line search
            step = 1.0
            new_beta = beta - step * delta
            
            # Check convergence
            if np.linalg.norm(delta) < self.tol:
                self._converged = True
                break
            
            beta = new_beta
        
        self._iterations = iteration + 1
        self.coef_ = beta
        self.hazard_ratios_ = np.exp(beta)
        
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
        
        self._compute_inference_cpu(X_sorted_np, time_sorted_np, event_sorted_np)
        self._compute_baseline_hazard(X_sorted_np, time_sorted_np, event_sorted_np)
        self._compute_cindex()
    
    def _compute_gradient_hessian(self, beta, X, time, event):
        """
        Compute gradient and Hessian of negative log partial likelihood.
        Uses Breslow approximation for ties.
        """
        n_samples, n_features = X.shape
        
        # Linear predictor
        eta = X @ beta
        exp_eta = np.exp(eta)
        
        # Risk sets: cumulative sum of exp(eta) for all at risk
        risk_sum = np.cumsum(exp_eta)
        
        # Find unique event times and handle ties
        unique_times