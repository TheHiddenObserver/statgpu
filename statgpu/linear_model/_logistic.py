"""
Logistic regression with full statistical inference and GPU support.
Uses IRLS (Iteratively Reweighted Least Squares) algorithm.
"""

from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device
from ..evaluation import (
    binary_average_precision_score,
    binary_precision_recall_curve,
    binary_roc_auc_score,
    binary_roc_curve,
    evaluate_binary_classification,
)


def _require_cupy(context: str):
    """Import CuPy or raise a clear ImportError when it is unavailable.

    Parameters
    ----------
    context : str
        Short description of the caller (used in the error message).

    Returns
    -------
    module
        The ``cupy`` module.

    Raises
    ------
    ImportError
        If CuPy is not installed, with a message that explains how to
        install it and why it is required here.
    """
    try:
        import cupy as cp
        return cp
    except ImportError as exc:
        raise ImportError(
            f"{context} requires CuPy for GPU computation, but CuPy is not "
            "installed. Install CuPy matching your CUDA version, e.g.: "
            "`pip install cupy-cuda12x` (CUDA 12.x) or "
            "`pip install cupy-cuda11x` (CUDA 11.x)."
        ) from exc


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
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.compute_inference = compute_inference
        self.cov_type = cov_type.lower()
        if self.cov_type not in ("nonrobust", "hc0", "hc1"):
            raise ValueError("cov_type must be one of: 'nonrobust', 'hc0', 'hc1'")
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
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
        self._train_pred_cache = None

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        self._train_pred_cache = None
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        
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
        self._y = self._to_numpy(y).astype(float)
        self._train_pred_cache = None
        
        X_arr = self._to_array(X)
        y_arr = self._to_array(y).astype(float)
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        if self.compute_inference and device != Device.CUDA:
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
        from .._gpu_utils import norm_two_tail_pvalues_gpu, norm_crit_gpu_two_tail
        
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

        if self.compute_inference:
            # Bread: inverse Hessian, H = X'WX (+ ridge)
            W_inf = p * (1 - p)
            W_inf = cp.clip(W_inf, 1e-8, 1 - 1e-8)
            H = X_design.T @ (X_design * W_inf[:, cp.newaxis])
            if alpha > 0:
                reg_diag_inf = cp.full(H.shape[0], alpha)
                if self.fit_intercept:
                    reg_diag_inf[0] = 0.0
                H += cp.diag(reg_diag_inf)
            try:
                eye = cp.eye(H.shape[0], dtype=H.dtype)
                bread = cp.linalg.solve(H, eye)
            except Exception:
                bread = cp.linalg.pinv(H)

            if self.cov_type == "nonrobust":
                cov_params = bread
            else:
                # Sandwich robust covariance for GLM/logit:
                # meat = X' diag((y-p)^2) X
                resid_score = y - p
                r2 = cp.square(resid_score)
                Xw = X_design * r2[:, cp.newaxis]
                meat = X_design.T @ Xw
                cov_params = bread @ meat @ bread
                if self.cov_type == "hc1":
                    n = X_design.shape[0]
                    k = X_design.shape[1]
                    if n > k:
                        cov_params = cov_params * (n / (n - k))

            bse_gpu = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
            zvalues_gpu = params / (bse_gpu + 1e-30)
            pvalues_gpu = norm_two_tail_pvalues_gpu(cp.abs(zvalues_gpu))
            z_crit = norm_crit_gpu_two_tail(0.05)
            conf_int_gpu = cp.stack(
                [params - z_crit * bse_gpu, params + z_crit * bse_gpu], axis=1
            )

            self._bse = bse_gpu.get()
            self._zvalues = zvalues_gpu.get()
            self._pvalues = pvalues_gpu.get()
            self._conf_int = conf_int_gpu.get()
        
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
        self._loglik = float(cp.asnumpy(self._loglik_gpu))
        self._accuracy = float(cp.asnumpy(self._accuracy_gpu))
        y_mean = cp.mean(y)
        y_mean = cp.clip(y_mean, 1e-15, 1 - 1e-15)
        self._loglik_null = float(
            cp.asnumpy(cp.sum(y * cp.log(y_mean) + (1 - y) * cp.log(1 - y_mean)))
        )

        # Release large temporary GPU tensors early.
        try:
            del X_design
        except Exception:
            pass
        try:
            del XtWX
        except Exception:
            pass
        try:
            del Xtz
        except Exception:
            pass
        try:
            del params
        except Exception:
            pass
        try:
            del W
        except Exception:
            pass
        try:
            del z
        except Exception:
            pass
        try:
            del eta
        except Exception:
            pass
        try:
            del p
        except Exception:
            pass
        self._cleanup_cuda_memory()
    
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
            bread = np.linalg.solve(XtWX, np.eye(XtWX.shape[0]))
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(XtWX)

        if self.cov_type == "nonrobust":
            cov_params = bread
        else:
            resid_score = self._y - p
            r2 = np.square(resid_score)
            Xw = self._X_design * r2[:, np.newaxis]
            meat = self._X_design.T @ Xw
            cov_params = bread @ meat @ bread
            if self.cov_type == "hc1":
                n = self._X_design.shape[0]
                k = self._X_design.shape[1]
                if n > k:
                    cov_params = cov_params * (n / (n - k))
        
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

    def _train_classification_table(self):
        """Training-set classification table on current device."""
        if self._y is None or not self._fitted:
            return None

        X_train = self._X_design[:, 1:] if self.fit_intercept else self._X_design
        if self._get_compute_device() == Device.CUDA:
            import cupy as cp

            y_true = cp.asarray(self._to_array(self._y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X_train))[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=0.5,
                include_curves=False,
                backend="cupy",
            )
            return out["classification_table"]

        y_score = self._to_numpy(self.predict_proba(X_train))[:, 1]
        out = evaluate_binary_classification(
            self._y,
            y_score,
            threshold=0.5,
            include_curves=False,
            backend="numpy",
        )
        return out["classification_table"]

    @staticmethod
    def _to_python_float(value):
        """Convert scalar-like values (including CuPy scalars) to float."""
        if value is None:
            return float("nan")
        try:
            import cupy as cp

            if isinstance(value, cp.ndarray):
                return float(value.item())
            if type(value).__module__.startswith("cupy"):
                return float(value.item())
        except Exception:
            pass
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass
        return float(value)
    
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
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            X_gpu = cp.asarray(self._to_array(X, Device.CUDA))
            coef_gpu = cp.asarray(self.coef_)
            intercept_gpu = cp.asarray(self.intercept_, dtype=coef_gpu.dtype)
            eta = X_gpu @ coef_gpu + intercept_gpu
            p1 = 1.0 / (1.0 + cp.exp(-cp.clip(eta, -500, 500)))
            return cp.column_stack([1 - p1, p1])
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        eta = X @ self.coef_ + self.intercept_
        p1 = self._sigmoid(eta)
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

    def predict_with_threshold(self, X, threshold: float = 0.5):
        """
        Predict class labels using a custom probability threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        threshold : float, default=0.5
            Probability threshold for positive class assignment.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
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
        y_pred = self._to_numpy(self.predict(X))
        y = self._to_numpy(y)
        return np.mean(y_pred == y)

    def confusion_matrix(self, X, y, threshold: float = 0.5) -> np.ndarray:
        """Compute binary confusion matrix on a dataset."""
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("confusion_matrix")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="cupy",
            )
            return out["confusion_matrix"]

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        out = evaluate_binary_classification(
            y_true,
            y_score,
            threshold=threshold,
            include_curves=False,
            backend="numpy",
        )
        return out["confusion_matrix"]

    def classification_table(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        """Return a compact classification table on a dataset."""
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("classification_table")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="cupy",
            )
            return out["classification_table"]

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        out = evaluate_binary_classification(
            y_true,
            y_score,
            threshold=threshold,
            include_curves=False,
            backend="numpy",
        )
        return out["classification_table"]

    def roc_curve(self, X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve arrays (fpr, tpr, thresholds)."""
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("roc_curve")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            return binary_roc_curve(y_true, y_score, backend="cupy")

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        return binary_roc_curve(y_true, y_score, backend="numpy")

    def roc_auc_score(self, X, y) -> float:
        """Compute ROC-AUC on a dataset."""
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("roc_auc_score")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            return binary_roc_auc_score(y_true, y_score, backend="cupy")

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        return binary_roc_auc_score(y_true, y_score, backend="numpy")

    def precision_recall_curve(self, X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision-recall arrays (precision, recall, thresholds)."""
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("precision_recall_curve")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            return binary_precision_recall_curve(y_true, y_score, backend="cupy")

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        return binary_precision_recall_curve(y_true, y_score, backend="numpy")

    def average_precision_score(self, X, y) -> float:
        """Compute average precision on a dataset."""
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("average_precision_score")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            return binary_average_precision_score(y_true, y_score, backend="cupy")

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        return binary_average_precision_score(y_true, y_score, backend="numpy")

    def evaluate_classification(
        self,
        X,
        y,
        threshold: float = 0.5,
        include_curves: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute classification metrics in one pass from a single probability call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,)
            Binary true labels encoded as 0/1.
        threshold : float, default=0.5
            Probability threshold used for hard predictions.
        include_curves : bool, default=True
            Whether to include full ROC/PR curve arrays in the output.

        Returns
        -------
        dict
            A dictionary with batched metrics. On CUDA device, arrays/scalars
            are GPU-backed (CuPy) except ``threshold``.
        """
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")

        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("evaluate_classification")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            return evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=include_curves,
                backend="cupy",
            )

        y_true = self._to_numpy(y).reshape(-1)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        return evaluate_binary_classification(
            y_true,
            y_score,
            threshold=threshold,
            include_curves=include_curves,
            backend="numpy",
        )

    def plot_roc_curve(self, X, y, ax=None, label: Optional[str] = None):
        """
        Plot ROC curve with matplotlib and return the axes object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_roc_curve(). "
                "Install it with: pip install matplotlib"
            ) from exc

        fpr, tpr, _ = self.roc_curve(X, y)
        auc = self.roc_auc_score(X, y)
        fpr_plot = self._to_numpy(fpr)
        tpr_plot = self._to_numpy(tpr)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        line_label = label if label is not None else f"ROC (AUC={self._to_python_float(auc):.3f})"
        ax.plot(fpr_plot, tpr_plot, label=line_label)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        return ax

    def plot_precision_recall_curve(self, X, y, ax=None, label: Optional[str] = None):
        """
        Plot precision-recall curve with matplotlib and return the axes object.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_precision_recall_curve(). "
                "Install it with: pip install matplotlib"
            ) from exc

        precision, recall, _ = self.precision_recall_curve(X, y)
        ap = self.average_precision_score(X, y)
        precision_plot = self._to_numpy(precision)
        recall_plot = self._to_numpy(recall)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        line_label = label if label is not None else f"PR (AP={self._to_python_float(ap):.3f})"
        ax.plot(recall_plot, precision_plot, label=line_label)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        return ax
    
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
        table = self._train_classification_table()
        if table is None:
            return None
        return table["accuracy"]
    
    @property
    def precision(self):
        """Precision on training data."""
        table = self._train_classification_table()
        if table is None:
            return None
        return table["precision"]
    
    @property
    def recall(self):
        """Recall on training data."""
        table = self._train_classification_table()
        if table is None:
            return None
        return table["recall"]
    
    @property
    def f1(self):
        """F1 score on training data."""
        table = self._train_classification_table()
        if table is None:
            return None
        return table["f1"]

    @property
    def auc(self):
        """ROC-AUC on training data."""
        if self._y is None or not self._fitted:
            return None
        X_train = self._X_design[:, 1:] if self.fit_intercept else self._X_design
        if self._get_compute_device() == Device.CUDA:
            import cupy as cp

            y_true = cp.asarray(self._to_array(self._y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X_train))[:, 1]
            try:
                return binary_roc_auc_score(y_true, y_score, backend="cupy")
            except ValueError:
                return None
        y_score = self._to_numpy(self.predict_proba(X_train))[:, 1]
        try:
            return binary_roc_auc_score(self._y, y_score, backend="numpy")
        except ValueError:
            return None

    @property
    def average_precision(self):
        """Average precision on training data."""
        if self._y is None or not self._fitted:
            return None
        X_train = self._X_design[:, 1:] if self.fit_intercept else self._X_design
        if self._get_compute_device() == Device.CUDA:
            import cupy as cp

            y_true = cp.asarray(self._to_array(self._y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X_train))[:, 1]
            try:
                return binary_average_precision_score(y_true, y_score, backend="cupy")
            except ValueError:
                return None
        y_score = self._to_numpy(self.predict_proba(X_train))[:, 1]
        try:
            return binary_average_precision_score(self._y, y_score, backend="numpy")
        except ValueError:
            return None
    
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
        print(f"Covariance Type:            {self.cov_type:>15}")
        print(f"Log-Likelihood:             {self.loglikelihood:>15.4f}")
        print(f"Log-Likelihood (Null):      {self.loglikelihood_null:>15.4f}")
        print(f"Pseudo R-squared:           {self.pseudo_rsquared:>15.4f}")
        print(f"AIC:                        {self.aic:>15.4f}")
        print(f"BIC:                        {self.bic:>15.4f}")
        print(f"Accuracy:                   {self._to_python_float(self.accuracy):>15.4f}")
        print(f"Precision:                  {self._to_python_float(self.precision):>15.4f}")
        print(f"Recall:                     {self._to_python_float(self.recall):>15.4f}")
        print(f"F1 Score:                   {self._to_python_float(self.f1):>15.4f}")
        auc = self.auc
        auc_display = self._to_python_float(auc)
        print(f"ROC-AUC:                    {auc_display:>15.4f}")
        ap = self.average_precision
        ap_display = self._to_python_float(ap)
        print(f"Avg Precision:              {ap_display:>15.4f}")
        print("-" * 80)
        print(f"{'':<15} {'coef':>12} {'std err':>12} {'z':>10} {'P>|z|':>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)
        
        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{self._zvalues[i]:>10.3f} {self._pvalues[i]:>10.4f} "
                  f"{self._conf_int[i, 0]:>12.4f} {self._conf_int[i, 1]:>12.4f}")
        
        print("=" * 80)
