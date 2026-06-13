"""
Logistic regression with full statistical inference and GPU support.
Uses IRLS (Iteratively Reweighted Least Squares) algorithm.
"""

from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from scipy import stats

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _get_torch_device_str
from statgpu.metrics import (
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
        hac_maxlags: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.compute_inference = compute_inference
        self.cov_type = cov_type.lower()
        if self.cov_type not in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
            raise ValueError(
                "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'"
            )
        if hac_maxlags is not None and int(hac_maxlags) < 0:
            raise ValueError("hac_maxlags must be a non-negative integer or None")
        self.hac_maxlags = None if hac_maxlags is None else int(hac_maxlags)
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
        self._train_eval_cache = None

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        self._train_pred_cache = None
        self._train_eval_cache = None
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _resolve_hac_maxlags(self, n_obs: int) -> int:
        """Resolve HAC lag count with a Newey-West style default rule."""
        if n_obs <= 1:
            return 0
        if self.hac_maxlags is None:
            maxlags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
        else:
            maxlags = int(self.hac_maxlags)
        return max(0, min(maxlags, n_obs - 1))

    def _hac_meat_numpy(self, scores: np.ndarray) -> np.ndarray:
        """Bartlett-kernel HAC meat from per-observation score matrix."""
        n_obs = int(scores.shape[0])
        meat = scores.T @ scores
        maxlags = self._resolve_hac_maxlags(n_obs)
        if maxlags == 0:
            return meat
        for lag in range(1, maxlags + 1):
            weight = 1.0 - (lag / (maxlags + 1.0))
            gamma = scores[lag:].T @ scores[:-lag]
            meat = meat + weight * (gamma + gamma.T)
        return meat

    def _hac_meat_cupy(self, scores):
        """CuPy Bartlett-kernel HAC meat from per-observation score matrix."""
        cp = _require_cupy("_hac_meat_cupy")

        n_obs = int(scores.shape[0])
        meat = scores.T @ scores
        maxlags = self._resolve_hac_maxlags(n_obs)
        if maxlags == 0:
            return meat
        for lag in range(1, maxlags + 1):
            weight = 1.0 - (lag / (maxlags + 1.0))
            gamma = scores[lag:].T @ scores[:-lag]
            meat = meat + weight * (gamma + gamma.T)
        return meat
        
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
        self._train_eval_cache = None

        # Get backend - support explicit torch backend selection
        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        X_arr = self._to_array(X, backend=backend_name)
        # Handle dtype conversion based on backend
        if backend_name == "torch":
            import torch
            y_arr = self._to_array(y, backend=backend_name)
            if y_arr.dtype != torch.float64:
                y_arr = y_arr.to(torch.float64)
        elif backend_name == "cupy":
            import cupy as cp
            y_arr = self._to_array(y, backend=backend_name).astype(cp.float64)
        else:
            y_arr = self._to_array(y, backend=backend_name).astype(float)

        device = self._get_compute_device()

        # Route to appropriate backend
        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, sample_weight)
        elif backend_name == "cupy":
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        if self.compute_inference and device == Device.CPU:
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
        from statgpu.inference._distributions_backend import norm
        
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
                resid_score = y - p
                scores = X_design * resid_score[:, cp.newaxis]

                if self.cov_type == "hac":
                    meat = self._hac_meat_cupy(scores)
                else:
                    if self.cov_type in ("hc2", "hc3"):
                        leverage = W_inf * cp.einsum("ij,jk,ik->i", X_design, bread, X_design)
                        leverage = cp.clip(leverage, 0.0, 1.0 - 1e-12)
                        if self.cov_type == "hc2":
                            scores = scores / cp.sqrt(1.0 - leverage)[:, cp.newaxis]
                        else:
                            scores = scores / (1.0 - leverage)[:, cp.newaxis]
                    meat = scores.T @ scores

                cov_params = bread @ meat @ bread
                if self.cov_type == "hc1":
                    n = X_design.shape[0]
                    k = X_design.shape[1]
                    if n > k:
                        cov_params = cov_params * (n / (n - k))

            bse_gpu = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
            zvalues_gpu = params / (bse_gpu + 1e-30)
            pvalues_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(zvalues_gpu)))
            z_crit = norm.ppf(0.975)
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

    def _cleanup_torch_memory(self):
        """Best-effort Torch CUDA memory cleanup."""
        self._train_pred_cache = None
        self._train_eval_cache = None
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with IRLS."""
        import torch
        from statgpu.inference._distributions_backend import norm

        # Note: Device.TORCH.value is 'torch', but Torch expects 'cuda' or 'cpu'.
        torch_device = _get_torch_device_str()

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Ensure Torch tensors on GPU
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch_device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(torch_device)
        if y.dtype != torch.float64:
            y = y.to(torch.float64)
        if X.dtype != torch.float64:
            X = X.to(torch.float64)

        # Add intercept if needed
        if self.fit_intercept:
            X_design = torch.cat([torch.ones(n_samples, 1, dtype=torch.float64, device=torch_device), X], dim=1)
        else:
            X_design = X

        # Initialize parameters
        params = torch.zeros(X_design.shape[1], dtype=torch.float64, device=torch_device)

        # Regularization parameter (lambda = 1 / (2*C))
        alpha = 1.0 / (2.0 * self.C) if self.C > 0 else 0.0

        # IRLS iteration
        iteration = 0
        for iteration in range(self.max_iter):
            params_old = params.clone()

            # Predicted probabilities
            eta = X_design @ params
            p = 1 / (1 + torch.exp(-torch.clamp(eta, -500, 500)))

            # Weights for WLS
            W = p * (1 - p)
            W = torch.clamp(W, 1e-8, 1 - 1e-8)

            if sample_weight is not None:
                if not isinstance(sample_weight, torch.Tensor):
                    sample_weight_torch = torch.from_numpy(sample_weight).to(torch_device)
                else:
                    sample_weight_torch = sample_weight.to(torch_device)
                if sample_weight_torch.dtype != torch.float64:
                    sample_weight_torch = sample_weight_torch.to(torch.float64)
                W = W * sample_weight_torch

            # Working response
            z = eta + (y - p) / W

            # Weighted least squares
            XtWX = X_design.T @ (X_design * W[:, None])

            # Add L2 regularization
            if alpha > 0:
                reg_diag = torch.full((XtWX.shape[0],), alpha, dtype=torch.float64, device=torch_device)
                if self.fit_intercept:
                    reg_diag[0] = 0.0
                XtWX += torch.diag(reg_diag)

            Xtz = X_design.T @ (W * z)

            try:
                params = torch.linalg.solve(XtWX, Xtz)
            except Exception:
                params = torch.linalg.lstsq(XtWX, Xtz)[0]

            # Check convergence
            if torch.linalg.norm(params - params_old) < self.tol:
                break

        self.n_iter_ = iteration + 1

        # Compute log-likelihood on GPU
        eta = X_design @ params
        p = 1 / (1 + torch.exp(-torch.clamp(eta, -500, 500)))
        loglik = torch.sum(y * torch.log(p + 1e-10) + (1 - y) * torch.log(1 - p + 1e-10))

        # Compute accuracy on GPU
        y_pred = (p > 0.5).to(torch.int32)
        y_true = y.to(torch.int32).reshape(y_pred.shape)
        accuracy = torch.mean((y_pred == y_true).to(torch.float64))

        # Store GPU results temporarily
        self._loglik_gpu = loglik
        self._accuracy_gpu = accuracy

        if self.compute_inference:
            # Bread: inverse Hessian, H = X'WX (+ ridge)
            W_inf = p * (1 - p)
            W_inf = torch.clamp(W_inf, 1e-8, 1 - 1e-8)
            H = X_design.T @ (X_design * W_inf[:, None])
            if alpha > 0:
                reg_diag_inf = torch.full((H.shape[0],), alpha, dtype=torch.float64, device=torch_device)
                if self.fit_intercept:
                    reg_diag_inf[0] = 0.0
                H += torch.diag(reg_diag_inf)
            try:
                eye = torch.eye(H.shape[0], dtype=H.dtype, device=torch_device)
                bread = torch.linalg.solve(H, eye)
            except Exception:
                bread = torch.linalg.pinv(H)

            if self.cov_type == "nonrobust":
                cov_params = bread
            else:
                resid_score = y - p
                scores = X_design * resid_score[:, None]

                if self.cov_type == "hac":
                    meat = self._hac_meat_torch(scores)
                else:
                    if self.cov_type in ("hc2", "hc3"):
                        leverage = W_inf * torch.einsum("ij,jk,ik->i", X_design, bread, X_design)
                        leverage = torch.clamp(leverage, 0.0, 1.0 - 1e-12)
                        if self.cov_type == "hc2":
                            scores = scores / torch.sqrt(1.0 - leverage)[:, None]
                        else:
                            scores = scores / (1.0 - leverage)[:, None]
                    meat = scores.T @ scores

                cov_params = bread @ meat @ bread
                if self.cov_type == "hc1":
                    n = X_design.shape[0]
                    k = X_design.shape[1]
                    if n > k:
                        cov_params = cov_params * (n / (n - k))

            bse_gpu = torch.sqrt(torch.clamp(torch.diag(cov_params), 0.0))
            zvalues_gpu = params / (bse_gpu + 1e-30)
            pvalues_gpu = torch.minimum(torch.tensor(1.0, device=torch_device), 2.0 * norm.sf(torch.abs(zvalues_gpu), device=torch_device))
            z_crit = norm.ppf(0.975, device=torch_device)
            conf_int_gpu = torch.stack(
                [params - z_crit * bse_gpu, params + z_crit * bse_gpu], dim=1
            )

            self._bse = bse_gpu.cpu().numpy()
            self._zvalues = zvalues_gpu.cpu().numpy()
            self._pvalues = pvalues_gpu.cpu().numpy()
            self._conf_int = conf_int_gpu.cpu().numpy()

        # Single transfer at the end
        params_np = params.cpu().numpy()
        X_design_np = X_design.cpu().numpy()

        self._X_design = X_design_np
        self._params = params_np

        if self.fit_intercept:
            self.intercept_ = float(params_np[0])
            self.coef_ = params_np[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params_np.copy()

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._loglik = float(self._loglik_gpu.cpu().numpy())
        self._accuracy = float(self._accuracy_gpu.cpu().numpy())
        y_mean = torch.mean(y)
        y_mean = torch.clamp(y_mean, 1e-15, 1 - 1e-15)
        self._loglik_null = float(torch.sum(y * torch.log(y_mean) + (1 - y) * torch.log(1 - y_mean)).cpu().numpy())

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
        self._cleanup_torch_memory()

    def _hac_meat_torch(self, scores):
        """Torch Bartlett-kernel HAC meat from per-observation score matrix."""
        import torch

        n_obs = int(scores.shape[0])
        meat = scores.T @ scores
        maxlags = self._resolve_hac_maxlags(n_obs)
        if maxlags == 0:
            return meat
        for lag in range(1, maxlags + 1):
            weight = 1.0 - (lag / (maxlags + 1.0))
            gamma = scores[lag:].T @ scores[:-lag]
            meat = meat + weight * (gamma + gamma.T)
        return meat
    
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

            scores = self._X_design * resid_score[:, np.newaxis]
            if self.cov_type == "hac":
                meat = self._hac_meat_numpy(scores)
            else:
                if self.cov_type in ("hc2", "hc3"):
                    leverage = W * np.einsum("ij,jk,ik->i", self._X_design, bread, self._X_design)
                    leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)
                    if self.cov_type == "hc2":
                        scores = scores / np.sqrt(1.0 - leverage)[:, np.newaxis]
                    else:
                        scores = scores / (1.0 - leverage)[:, np.newaxis]
                meat = scores.T @ scores

            cov_params = bread @ meat @ bread
            if self.cov_type == "hc1":
                n = self._X_design.shape[0]
                k = self._X_design.shape[1]
                if n > k:
                    cov_params = cov_params * (n / (n - k))
        
        # Standard errors
        self._bse = np.sqrt(np.maximum(np.diag(cov_params), 0.0))

        # z-values (asymptotic normal, add epsilon to avoid division by zero)
        self._zvalues = self._params / (self._bse + 1e-30)
        
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
        """Training-set classification table on current device.

        Results are cached in ``_train_eval_cache`` so that multiple
        properties (accuracy, precision, recall, f1, auc, average_precision)
        sharing the same training data only trigger a single forward pass.
        """
        if self._y is None or not self._fitted:
            return None

        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("classification_table")

        X_train = self._X_design[:, 1:] if self.fit_intercept else self._X_design
        device = self._get_compute_device()
        if device == Device.CUDA:
            cp = _require_cupy("_train_classification_table")

            y_true = cp.asarray(self._to_array(self._y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X_train))[:, 1]
            self._train_eval_cache = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=0.5,
                include_curves=False,
                backend="cupy",
            )
            return self._train_eval_cache["classification_table"]
        if device == Device.TORCH:
            import torch

            y_true = self._to_array(self._y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X_train)[:, 1]
            if not isinstance(y_score, torch.Tensor):
                y_score = torch.as_tensor(y_score, dtype=torch.float64, device=y_true.device)
            self._train_eval_cache = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=0.5,
                include_curves=False,
                backend="torch",
            )
            return self._train_eval_cache["classification_table"]

        y_score = self._to_numpy(self.predict_proba(X_train))[:, 1]
        self._train_eval_cache = evaluate_binary_classification(
            self._y,
            y_score,
            threshold=0.5,
            include_curves=False,
            backend="numpy",
        )
        return self._train_eval_cache["classification_table"]

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
        if device == Device.TORCH:
            import torch

            X_torch = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef_torch = torch.as_tensor(self.coef_, dtype=X_torch.dtype, device=X_torch.device)
            intercept_torch = torch.as_tensor(
                self.intercept_, dtype=X_torch.dtype, device=X_torch.device
            )
            eta = X_torch @ coef_torch + intercept_torch
            p1 = 1.0 / (1.0 + torch.exp(-torch.clamp(eta, -500, 500)))
            return torch.column_stack([1 - p1, p1])
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
        if hasattr(proba, "to") and hasattr(proba, "dtype"):
            return (proba[:, 1] >= 0.5).to(dtype=proba.dtype)
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
        if hasattr(proba, "to") and hasattr(proba, "dtype"):
            return (proba[:, 1] >= threshold).to(dtype=proba.dtype)
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
        y_pred = self.predict(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            yb = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            return float(cp.mean(y_pred.reshape(-1) == yb).item())
        if device == Device.TORCH:
            import torch

            yb = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            return float(torch.mean((y_pred.reshape(-1) == yb).to(torch.float64)).item())
        y_pred = self._to_numpy(y_pred)
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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="torch",
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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="torch",
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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            return binary_roc_curve(y_true, y_score, backend="torch")

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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            return binary_roc_auc_score(y_true, y_score, backend="torch")

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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            return binary_precision_recall_curve(y_true, y_score, backend="torch")

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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            return binary_average_precision_score(y_true, y_score, backend="torch")

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
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            return evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=include_curves,
                backend="torch",
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
        # Use cached eval result if available (populated by _train_classification_table)
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("roc_auc")
        # Trigger cache population via _train_classification_table
        self._train_classification_table()
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("roc_auc")
        return None

    @property
    def average_precision(self):
        """Average precision on training data."""
        if self._y is None or not self._fitted:
            return None
        # Use cached eval result if available (populated by _train_classification_table)
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("average_precision")
        # Trigger cache population via _train_classification_table
        self._train_classification_table()
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("average_precision")
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
