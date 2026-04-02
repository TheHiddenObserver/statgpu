"""
Logistic regression with full statistical inference and GPU support.
Uses IRLS (Iteratively Reweighted Least Squares) algorithm.
"""

from typing import Optional, Union, Tuple
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class LogisticRegression(BaseEstimator):
    """
    Logistic regression with GPU acceleration and full statistical inference.
    
    Uses IRLS (Iteratively Reweighted Least Squares) algorithm with
    optional L2 regularization.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization.
    max_iter : int, default=100
        Maximum number of iterations for IRLS.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    n_jobs : int, optional
        Number of parallel jobs for CPU computation.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(
        self,
        fit_intercept: bool = True,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.compute_inference = compute_inference
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        
        # Internal storage for inference
        self._X_design = None
        self._y = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._bse = None
        self._zvalues = None
        self._pvalues = None
        self._conf_int = None
        self._loglik = None
        self._loglik_null = None
        
    def _sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit logistic regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (0 or 1).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        
        Returns
        -------
        self : object
        """
        self._y = np.asarray(y).astype(float)
        
        X_arr = self._to_array(X)
        y_arr = self._to_array(y).astype(float)
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        if self.compute_inference:
            self._compute_inference()
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU with IRLS."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Add intercept if needed
        if self.fit_intercept:
            self._X_design = np.column_stack([np.ones(n_samples, dtype=X.dtype), X])
        else:
            self._X_design = X.copy()
        
        # Initialize parameters
        params = np.zeros(self._X_design.shape[1])
        
        # Regularization parameter (lambda = 1 / (2*C))
        alpha = 1.0 / (2.0 * self.C) if self.C > 0 else 0.0
        
        # IRLS iteration
        for iteration in range(self.max_iter):
            params_old = params.copy()
            
            # Predicted probabilities
            eta = self._X_design @ params
            p = self._sigmoid(eta)
            
            # Weights for WLS
            W = p * (1 - p)
            W = np.clip(W, 1e-8, 1 - 1e-8)  # Avoid numerical issues
            
            if sample_weight is not None:
                W = W * np.asarray(sample_weight)
            
            # Working response
            z = eta + (y - p) / W
            
            # Weighted least squares
            # (X'WX + alpha*I) * params = X'Wz
            XtWX = self._X_design.T @ (self._X_design * W[:, np.newaxis])
            
            # Add L2 regularization (don't regularize intercept)
            if alpha > 0:
                reg_diag = np.full(XtWX.shape[0], alpha)
                if self.fit_intercept:
                    reg_diag[0] = 0.0  # Don't regularize intercept
                XtWX += np.diag(reg_diag)
            
            Xtz = self._X_design.T @ (W * z)
            
            try:
                params = np.linalg.solve(XtWX, Xtz)
            except np.linalg.LinAlgError:
                params = np.linalg.lstsq(XtWX, Xtz, rcond=None)[0]
            
            # Check convergence
            if np.linalg.norm(params - params_old) < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        self._params = params
        
        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params.copy()
        
        # Degrees of freedom
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
    
    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU with IRLS."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Add intercept if needed
        if self.fit_intercept:
            X_design = cp.column_stack([cp.ones(n_samples, dtype=X.dtype), X])
        else:
            X_design = X
        
        # Initialize parameters
        params = cp.zeros(X_design.shape[1])
        
        # Regularization parameter
        alpha = 1.0 / (2.0 * self.C) if self.C > 0 else 0.0
        
        # IRLS iteration
        for iteration in range(self.max_iter):
            params_old = params.copy()
            
            # Predicted probabilities
            eta = X_design @ params
            p = 1 / (1 + cp.exp(-cp.clip(eta, -500, 500)))
            
            # Weights for WLS
            W = p * (1 - p)
            W = cp.clip(W, 1e-8, 1 - 1e-8)
            
            if sample_weight is not None:
                W = W * cp.asarray(sample_weight)
            
            # Working response
            z = eta + (y - p) / W
            
            # Weighted least squares
            XtWX = X_design.T @ (X_design * W[:, cp.newaxis])
            
            # Add L2 regularization
            if alpha > 0:
                reg_diag = cp.full(XtWX.shape[0], alpha)
                if self.fit_intercept:
                    reg_diag[0] = 0.0
                XtWX += cp.diag(reg_diag)
            
            Xtz = X_design.T @ (W * z)
            
            try:
                params = cp.linalg.solve(XtWX, Xtz)
            except Exception:
                params = cp.linalg.lstsq(XtWX, Xtz)[0]
            
            # Check convergence
            if cp.linalg.norm(params - params_old) < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        
        # Compute log-likelihood on GPU
        eta = X_design @ params
        p = 1 / (1 + cp.exp(-cp.clip(eta, -500, 500)))
        loglik = cp.sum(y * cp.log(p + 1e-10) + (1 - y) * cp.log(1 - p + 1e-10))
        
        # Compute accuracy on GPU
        y_pred = (p > 0.5).astype(cp.int32)
        accuracy = cp.mean(y_pred == y)
        
        # Store GPU results temporarily
        self._loglik_gpu = loglik
        self._accuracy_gpu = accuracy
        
        # Single transfer at the end
        params_np = params.get()
        X_design_np = X_design.get()
        
        self._X_design = X_design_np
        self._params = params_np
        
        if self.fit_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()
        
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._loglik = float(self._loglik_gpu.get())
        self._accuracy = float(self._accuracy_gpu.get())
    
    def _compute_inference(self):
        """Compute standard errors, z-stats, p-values, and confidence intervals."""
        if self._X_design is None or self._params is None:
            return
        
        # Predicted probabilities
        eta = self._X_design @ self._params
        p = self._sigmoid(eta)
        
        # Compute Hessian (information matrix)
        W = p * (1 - p)
        W = np.clip(W, 1e-8, 1 - 1e-8)
        
        XtWX = self._X_design.T @ (self._X_design * W[:, np.newaxis])
        
        # Add regularization to Hessian
        alpha = 1.0 / (2.0 * self.C) if self.C > 0 else 0.0
        if alpha > 0:
            reg_diag = np.full(XtWX.shape[0], alpha)
            if self.fit_intercept:
                reg_diag[0] = 0.0
            XtWX += np.diag(reg_diag)
        
        try:
            cov_params = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            cov_params = np.linalg.pinv(XtWX)
        
        # Standard errors
        self._bse = np.sqrt(np.diag(cov_params))
        
        # z-values (asymptotic normal)
        self._zvalues = self._params / self._bse
        
        # p-values (two-tailed)
        self._pvalues = 2 * (1 - stats.norm.cdf(np.abs(self._zvalues)))
        
        # 95% confidence intervals
        alpha = 0.05
        z_crit = stats.norm.ppf(1 - alpha/2)
        self._conf_int = np.column_stack([
            self._params - z_crit * self._bse,
            self._params + z_crit * self._bse
        ])
        
        # Log-likelihood
        eps = 1e-15  # Avoid log(0)
        p_clipped = np.clip(p, eps, 1 - eps)
        self._loglik = np.sum(self._y * np.log(p_clipped) + (1 - self._y) * np.log(1 - p_clipped))
        
        # Null log-likelihood (intercept-only model)
        y_mean = np.mean(self._y)
        y_mean = np.clip(y_mean, eps, 1 - eps)
        self._loglik_null = np.sum(self._y * np.log(y_mean) + (1 - self._y) * np.log(1 - y_mean))
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        ndarray of shape (n_samples, 2)
            Returns the probability of the samples for each class.
        """
        self._check_is_fitted()
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        
        # Linear predictor
        eta = X @ self.coef_ + self.intercept_
        
        # Sigmoid
        p1 = self._sigmoid(eta)
        
        # Return probabilities for both classes
        return np.column_stack([1 - p1, p1])
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def score(self, X, y):
        """
        Return mean accuracy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        float
            Mean accuracy.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        return np.mean(y_pred == y)
    
    @property
    def loglikelihood(self):
        """Log-likelihood of the fitted model."""
        return self._loglik
    
    @property
    def loglikelihood_null(self):
        """Log-likelihood of the null model."""
        return self._loglik_null
    
    @property
    def aic(self):
        """Akaike Information Criterion."""
        if self._loglik is None:
            return None
        k = len(self._params)
        return -2 * self._loglik + 2 * k
    
    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self._loglik is None or self._nobs is None:
            return None
        k = len(self._params)
        return -2 * self._loglik + k * np.log(self._nobs)
    
    @property
    def pseudo_rsquared(self):
        """
        Pseudo R-squared (McFadden's).
        
        Measures the improvement of the full model over the null model.
        """
        if self._loglik is None or self._loglik_null is None:
            return None
        if self._loglik_null == 0:
            return 0.0
        return 1 - (self._loglik / self._loglik_null)
    
    @property
    def accuracy(self):
        """Classification accuracy on training data."""
        if self._y is None or not self._fitted:
            return None
        y_pred = self.predict(self._X_design[:, 1:] if self.fit_intercept else self._X_design)
        return np.mean(y_pred == self._y)
    
    @property
    def precision(self):
        """Precision on training data."""
        if self._y is None or not self._fitted:
            return None
        y_pred = self.predict(self._X_design[:, 1:] if self.fit_intercept else self._X_design)
        tp = np.sum((y_pred == 1) & (self._y == 1))
        fp = np.sum((y_pred == 1) & (self._y == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @property
    def recall(self):
        """Recall on training data."""
        if self._y is None or not self._fitted:
            return None
        y_pred = self.predict(self._X_design[:, 1:] if self.fit_intercept else self._X_design)
        tp = np.sum((y_pred == 1) & (self._y == 1))
        fn = np.sum((y_pred == 0) & (self._y == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @property
    def f1(self):
        """F1 score on training data."""
        prec = self.precision
        rec = self.recall
        if prec is None or rec is None:
            return None
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
    
    def summary(self):
        """Print summary table similar to statsmodels/R."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if self._bse is None or self._pvalues is None or self._conf_int is None:
            raise RuntimeError(
                "compute_inference=False: inference statistics are not available. "
                "Re-fit with compute_inference=True (default) to use summary()."
            )
        
        # Build feature names
        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]
        
        print("=" * 80)
        print("                           Logistic Regression Results")
        print("=" * 80)
        print(f"No. Observations:           {self._nobs:>15}")
        print(f"Degrees of Freedom:         {self._df_resid:>15}")
        print(f"Iterations:                 {self.n_iter_:>15}")
        print(f"Log-Likelihood:             {self.loglikelihood:>15.4f}")
        print(f"Log-Likelihood (Null):      {self.loglikelihood_null:>15.4f}")
        print(f"Pseudo R-squared:           {self.pseudo_rsquared:>15.4f}")
        print(f"AIC:                        {self.aic:>15.4f}")
        print(f"BIC:                        {self.bic:>15.4f}")
        print(f"Accuracy:                   {self.accuracy:>15.4f}")
        print(f"Precision:                  {self.precision:>15.4f}")
        print(f"Recall:                     {self.recall:>15.4f}")
        print(f"F1 Score:                   {self.f1:>15.4f}")
        print("-" * 80)
        print(f"{'':<15} {'coef':>12} {'std err':>12} {'z':>10} {'P>|z|':>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)
        
        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{self._zvalues[i]:>10.3f} {self._pvalues[i]:>10.4f} "
                  f"{self._conf_int[i, 0]:>12.4f} {self._conf_int[i, 1]:>12.4f}")
        
        print("=" * 80)
