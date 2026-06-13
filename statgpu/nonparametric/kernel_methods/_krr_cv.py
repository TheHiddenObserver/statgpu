"""
Cross-validated Kernel Ridge Regression with GPU acceleration.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy, _torch_dev, xp_zeros, xp_astype

from ._kernels import pairwise_kernels
from ._krr import KernelRidge


def _kfold_indices(n_samples: int, n_splits: int, random_state: Optional[int] = None):
    """Generate K-fold train/test index arrays."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int64)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop
    return folds


class KernelRidgeCV(BaseEstimator):
    r"""Cross-validated Kernel Ridge Regression.

    Efficiently searches over a grid of regularization parameters using
    eigendecomposition of the kernel matrix.  For a fixed kernel matrix
    :math:`K = Q \Lambda Q^\top`, the solution for any :math:`\alpha` is:

    .. math::
        \boldsymbol{\alpha}(\lambda)
            = Q \, \text{diag}\!\left(\frac{1}{\lambda_i + \lambda}\right)
              Q^\top \mathbf{y}

    This avoids re-solving the linear system for every alpha value.

    Parameters
    ----------
    alphas : array-like, optional
        Regularization strengths to try.  If ``None``, a log-spaced grid
        of 100 values is generated automatically.
    cv : int, default=5
        Number of cross-validation folds.
    kernel : str or callable, default='rbf'
        Kernel metric name or callable.
    gamma : float, optional
        Kernel coefficient.  Defaults to ``1 / n_features``.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1
        Independent term for polynomial and sigmoid kernels.
    kernel_params : dict, optional
        Additional parameters for the kernel function.
    random_state : int, optional
        Random state for fold generation.
    device : str or Device, default='auto'
        Computation device.
    n_jobs : int, optional
        Not used; kept for API compatibility.

    Attributes
    ----------
    alpha_ : float
        Best regularization parameter found by cross-validation.
    best_score_ : float
        Best mean R^2 score across folds.
    cv_results_ : dict
        Detailed cross-validation results.
    estimator_ : KernelRidge
        Fitted KernelRidge model with the best alpha.
    dual_coef_ : ndarray
        Dual coefficients of the fitted model (shortcut).
    X_fit_ : ndarray
        Training data (shortcut).
    """

    def __init__(
        self,
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        kernel: Union[str, callable] = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alphas = alphas
        self.cv = cv
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.random_state = random_state

        # Fitted attributes
        self.alpha_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.estimator_ = None
        self.dual_coef_ = None
        self.X_fit_ = None

    def _get_kernel_params(self):
        """Collect kernel-specific parameters."""
        params = {}
        if self.kernel_params is not None:
            params.update(self.kernel_params)
        k = str(self.kernel).strip().lower() if isinstance(self.kernel, str) else ""
        if k in ("rbf", "gaussian", "polynomial", "poly", "laplacian", "sigmoid"):
            if self.gamma is not None:
                params["gamma"] = self.gamma
        if k in ("polynomial", "poly", "sigmoid"):
            if self.degree != 3 and k in ("polynomial", "poly"):
                params["degree"] = self.degree
            if self.coef0 != 1:
                params["coef0"] = self.coef0
        return params

    def _generate_alpha_grid(self, eigvals):
        """Generate log-spaced alpha grid based on eigenvalue range.

        Parameters
        ----------
        eigvals : ndarray
            Eigenvalues of the kernel matrix.

        Returns
        -------
        alphas : ndarray
            Log-spaced alpha values.
        """
        eig_np = _to_numpy(eigvals).ravel()
        lambda_max = float(np.max(eig_np))
        lambda_min = float(np.min(eig_np[eig_np > 1e-12])) if np.any(eig_np > 1e-12) else 1e-6

        alpha_max = max(lambda_max * 10.0, 1.0)
        alpha_min = max(lambda_min * 1e-3, 1e-8)

        if alpha_min >= alpha_max:
            alpha_min = alpha_max * 1e-4

        return np.logspace(
            np.log10(alpha_min),
            np.log10(alpha_max),
            num=100,
            dtype=np.float64,
        )

    def fit(self, X, y):
        """Fit KernelRidgeCV model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self
        """
        # Resolve backend
        backend = self._get_backend()
        xp = backend.xp

        X_arr = xp_astype(self._to_array(X), xp.float64, xp)
        y_arr = xp_astype(self._to_array(y), xp.float64, xp)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]
        n_targets = y_arr.shape[1]

        # Compute full kernel matrix once
        kernel_params = self._get_kernel_params()
        K = pairwise_kernels(X_arr, X_arr, metric=self.kernel, xp=xp, **kernel_params)

        # Eigendecompose: K = Q @ diag(eigvals) @ Q.T
        eigvals, Q = xp.linalg.eigh(K)

        # Generate alpha grid if not provided
        alphas_np = self.alphas
        if alphas_np is None:
            alphas_np = self._generate_alpha_grid(eigvals)
        else:
            alphas_np = np.asarray(alphas_np, dtype=np.float64).ravel()
        n_alphas = alphas_np.shape[0]

        # Project y into eigenbasis once: Q_T @ y
        Q_T = Q.T  # eigh returns real eigenvectors for symmetric K
        Qt_y = Q_T @ y_arr  # (n_samples, n_targets)

        # K-fold CV
        n_folds = int(self.cv)
        folds = _kfold_indices(n_samples, n_folds, random_state=self.random_state)

        # mse_table: (n_alphas, n_folds, n_targets)
        mse_table = xp_zeros((n_alphas, n_folds, n_targets), xp.float64, xp, X_arr)

        # Detect torch backend for GPU-accelerated CV
        _is_torch = hasattr(K, 'device') and hasattr(K, 'is_cuda') and not hasattr(K, 'get')

        for fi, (train_idx, test_idx) in enumerate(folds):
            n_train = len(train_idx)
            n_test = len(test_idx)

            K_train = K[train_idx][:, train_idx]
            y_train = y_arr[train_idx]
            y_test = y_arr[test_idx]
            K_test = K[test_idx][:, train_idx]

            if _is_torch and K.is_cuda:
                import torch
                # Full GPU path: eigendecomposition + batched alpha sweep
                fold_eigvals, fold_Q = torch.linalg.eigh(K_train)
                Qt_y_fold = fold_Q.T @ y_train  # (n_train, n_targets)

                # Vectorized alpha sweep: (n_alphas, n_train)
                alphas_t = torch.asarray(alphas_np, dtype=torch.float64, device=K.device)
                inv_diag = 1.0 / (fold_eigvals[None, :] + alphas_t[:, None])  # (n_alphas, n_train)

                # dual_coefs: (n_alphas, n_train, n_targets)
                weighted = inv_diag[:, :, None] * Qt_y_fold[None, :, :]  # (n_alphas, n_train, n_targets)
                dual_coefs = fold_Q @ weighted  # (n_alphas, n_train, n_targets)

                # Predict: K_test @ dual_coefs -> (n_alphas, n_test, n_targets)
                y_pred = torch.matmul(K_test.unsqueeze(0), dual_coefs)  # (n_alphas, n_test, n_targets)
                residuals = y_pred - y_test[None, :, :]
                mse_vals = torch.mean(residuals ** 2, dim=1)  # (n_alphas, n_targets)

                mse_table[:, fi, :] = mse_vals
            else:
                # NumPy and CuPy path: vectorized alpha sweep on device
                fold_eigvals, fold_Q = xp.linalg.eigh(K_train)
                Qt_y_fold = fold_Q.T @ y_train  # (n_train, n_targets)

                # Vectorized alpha sweep: (n_alphas, n_train)
                alphas_dev = xp.asarray(alphas_np, dtype=xp.float64)
                inv_diag = 1.0 / (fold_eigvals[None, :] + alphas_dev[:, None])  # (n_alphas, n_train)

                # dual_coefs: (n_alphas, n_train, n_targets)
                weighted = inv_diag[:, :, None] * Qt_y_fold[None, :, :]  # (a, n, t)

                # Q @ weighted for each alpha: (a, m, t) = sum_n Q[m,n] * weighted[a,n,t]
                dual_coefs = xp.einsum('mn,ant->amt', fold_Q, weighted)  # (a, m, t)

                # Predict: K_test @ dual_coefs for each alpha
                # K_test[t,m] @ dual_coefs[a,m,tgt] -> y_pred[a,t,tgt]
                if n_targets == 1:
                    # Faster path for single target
                    y_pred = (K_test @ dual_coefs[:, :, 0].T).T  # (a, n_test)
                    residuals = y_pred - y_test.ravel()[None, :]
                    mse_vals = xp.mean(residuals ** 2, axis=1, keepdims=True)  # (a, 1)
                else:
                    y_pred = xp.einsum('tm,amk->atk', K_test, dual_coefs)  # (a, n_test, n_targets)
                    residuals = y_pred - y_test[None, :, :]
                    mse_vals = xp.mean(residuals ** 2, axis=1)  # (a, n_targets)

                mse_table[:, fi, :] = mse_vals

        # Mean MSE across folds: (n_alphas, n_targets)
        mean_mse = xp.mean(mse_table, axis=1)

        # For single target, select best alpha by mean MSE
        if n_targets == 1:
            mean_mse_1d = mean_mse[:, 0]
            best_idx = int(xp.argmin(mean_mse_1d).item())
        else:
            # Average across targets for selection
            mean_mse_avg = xp.mean(mean_mse, axis=1)
            best_idx = int(xp.argmin(mean_mse_avg).item())

        self.alpha_ = float(alphas_np[best_idx])

        # Compute mean R^2 across folds for best alpha
        mean_mse_best = float(mean_mse[best_idx, 0].item()) if n_targets == 1 else float(xp.mean(mean_mse[best_idx]).item())
        y_var = float(xp.var(y_arr).item())
        self.best_score_ = 1.0 - mean_mse_best / y_var if y_var > 0 else 0.0

        # Build cv_results_
        self.cv_results_ = {
            "alphas": alphas_np,
            "mean_mse": _to_numpy(mean_mse),
            "mse_table": _to_numpy(mse_table),
            "best_alpha": self.alpha_,
            "best_score": self.best_score_,
        }

        # Refit with best alpha
        self.estimator_ = KernelRidge(
            alpha=self.alpha_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            device=self.device,
        )
        self.estimator_.fit(X, y)

        # Shortcut attributes
        self.dual_coef_ = self.estimator_.dual_coef_
        self.X_fit_ = self.estimator_.X_fit_
        self._xp = self.estimator_._xp

        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the best Kernel Ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples_test, n_features)

        Returns
        -------
        y_pred : ndarray
        """
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def score(self, X, y):
        """Return R^2 score using the best Kernel Ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        score : float
        """
        self._check_is_fitted()
        return self.estimator_.score(X, y)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update({
            "alphas": self.alphas,
            "cv": self.cv,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "kernel_params": self.kernel_params,
            "random_state": self.random_state,
        })
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        super().set_params(**params)
        for key in ("alphas", "cv", "kernel", "gamma", "degree", "coef0",
                     "kernel_params", "random_state"):
            if key in params:
                setattr(self, key, params[key])
        return self
