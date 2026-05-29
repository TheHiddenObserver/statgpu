"""
Kernel Ridge Regression with GPU acceleration.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _to_numpy, _torch_dev, xp_eye, xp_astype

from ._kernels import pairwise_kernels


class KernelRidge(BaseEstimator):
    r"""Kernel Ridge Regression.

    Solves the dual problem:

    .. math::
        \boldsymbol{\alpha} = (K + \alpha I)^{-1} \mathbf{y}

    where :math:`K` is the kernel matrix of the training data.  Predictions
    are computed as :math:`\hat{y} = K_{\text{test}} \boldsymbol{\alpha}`.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    kernel : str or callable, default='rbf'
        Kernel metric name (``'rbf'``, ``'linear'``, ``'polynomial'``,
        ``'laplacian'``, ``'sigmoid'``, ``'cosine'``) or a callable.
    gamma : float, optional
        Kernel coefficient for rbf, polynomial, laplacian, sigmoid.
        Defaults to ``1 / n_features``.
    degree : int, default=3
        Degree for the polynomial kernel.
    coef0 : float, default=1
        Independent term for polynomial and sigmoid kernels.
    kernel_params : dict, optional
        Additional parameters passed to the kernel function.
    device : str or Device, default='auto'
        Computation device.
    n_jobs : int, optional
        Not used; kept for API compatibility.

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Dual coefficients in the kernel space.
    X_fit_ : ndarray of shape (n_samples, n_features)
        Training data stored for prediction.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: Union[str, callable] = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

        # Fitted attributes
        self.dual_coef_ = None
        self.X_fit_ = None
        self._xp = None
        self._backend = None

    def _get_kernel_params(self):
        """Collect kernel-specific parameters."""
        params = {}
        if self.kernel_params is not None:
            params.update(self.kernel_params)
        # Only pass gamma/degree/coef0 for kernels that use them
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

    def fit(self, X, y, sample_weight=None):
        """Fit Kernel Ridge Regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        sample_weight : ignored
            Not used; kept for API compatibility.

        Returns
        -------
        self
        """
        # Resolve backend and convert arrays
        self._backend = self._get_backend()
        xp = self._backend.xp
        self._xp = xp

        X_arr = self._to_array(X)
        y_arr = xp_astype(self._to_array(y), xp.float64, xp)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        n_samples = X_arr.shape[0]
        alpha = float(self.alpha)

        # Compute kernel matrix
        kernel_params = self._get_kernel_params()
        K = pairwise_kernels(X_arr, X_arr, metric=self.kernel, xp=xp, **kernel_params)

        # Regularize: K + alpha * I
        K_reg = K + alpha * xp_eye(n_samples, K.dtype, xp, K)

        # Solve (K + alpha I) * dual_coef = y
        self.dual_coef_ = xp.linalg.solve(K_reg, y_arr)

        self.X_fit_ = X_arr
        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the kernel ridge model.

        Parameters
        ----------
        X : array-like of shape (n_samples_test, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples_test,) or (n_samples_test, n_targets)
        """
        self._check_is_fitted()
        xp = self._xp

        X_arr = self._to_array(X)
        kernel_params = self._get_kernel_params()
        K_test = pairwise_kernels(X_arr, self.X_fit_, metric=self.kernel, xp=xp, **kernel_params)

        y_pred = K_test @ self.dual_coef_
        # Flatten if single target
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        return y_pred

    def score(self, X, y):
        """Return the coefficient of determination R^2.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        score : float
            R^2 score.
        """
        self._check_is_fitted()
        xp = self._xp

        y_pred = self.predict(X)
        y_arr = xp_astype(self._to_array(y), xp.float64, xp)
        if y_arr.ndim == 1:
            y_arr = y_arr.ravel()

        ss_res = xp.sum((y_arr - y_pred) ** 2)
        ss_tot = xp.sum((y_arr - xp.mean(y_arr)) ** 2)

        ss_res_val = float(ss_res.item()) if hasattr(ss_res, "item") else float(ss_res)
        ss_tot_val = float(ss_tot.item()) if hasattr(ss_tot, "item") else float(ss_tot)

        if ss_tot_val == 0.0:
            return 0.0
        return 1.0 - ss_res_val / ss_tot_val

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update({
            "alpha": self.alpha,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "degree": self.degree,
            "coef0": self.coef0,
            "kernel_params": self.kernel_params,
        })
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        super().set_params(**params)
        for key in ("alpha", "kernel", "gamma", "degree", "coef0", "kernel_params"):
            if key in params:
                setattr(self, key, params[key])
        return self
