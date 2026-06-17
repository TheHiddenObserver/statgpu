"""Graphical Lasso for sparse inverse covariance estimation with GPU support."""

from __future__ import annotations

__all__ = ["GraphicalLasso", "GraphicalLassoCV"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _get_xp, _to_float_scalar, _to_numpy, xp_zeros, xp_eye

from statgpu.covariance._empirical import (
    EmpiricalCovariance,
    _detect_backend,
    _stable_inv,
)


class GraphicalLasso(EmpiricalCovariance):
    """
    Sparse inverse covariance estimation via the graphical lasso.

    Estimates a sparse precision matrix (inverse covariance) by solving:

        maximize  log(det(theta)) - trace(S * theta) - alpha * ||theta||_1

    using the graphical lasso algorithm (Friedman, Hastie & Tibshirani, 2008).

    Parameters
    ----------
    alpha : float, default=0.01
        Regularization parameter for the L1 penalty.
    max_iter : int, default=100
        Maximum number of outer iterations.
    tol : float, default=1e-4
        Convergence tolerance on the dual gap.
    assume_centered : bool, default=False
        If True, data is assumed to be already centered.
    device : str or Device, default='auto'
        Computation device.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Estimated sparse precision matrix.
    location_ : ndarray of shape (n_features,)
        Estimated mean.
    n_iter_ : int
        Number of iterations performed.
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.

    References
    ----------
    Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
    covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-4,
        assume_centered: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(assume_centered=assume_centered, device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """Fit the graphical lasso model to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        # Work in numpy for the iterative block-coordinate descent
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n, p = X_np.shape
        if n < 2:
            raise ValueError(f"Need at least 2 samples, got {n}")

        if not self.assume_centered:
            location_np = X_np.mean(axis=0)
            X_np = X_np - location_np
        else:
            location_np = np.zeros(p)

        # Sample covariance
        S = X_np.T @ X_np / float(n)

        # Graphical lasso: block coordinate descent
        # Initialize W = S + alpha * I (ensures positive definiteness)
        W = S.copy()
        np.fill_diagonal(W, W.diagonal() + self.alpha)
        theta = np.linalg.inv(W)

        for iteration in range(self.max_iter):
            W_old = W.copy()

            for j in range(p):
                # Partition: solve the L1-regularized regression for feature j
                # Using the cyclical coordinate descent approach
                not_j = list(range(j)) + list(range(j + 1, p))
                S_11 = S[np.ix_(not_j, not_j)]
                W_11 = W[np.ix_(not_j, not_j)]

                # Solve: beta = argmin ||S_11^{1/2} beta - s_12||^2 + alpha * ||beta||_1
                # Using coordinate descent
                s_12 = S[not_j, j]
                beta = theta[not_j, j] * W[j, j]  # warm start

                for _ in range(100):  # inner CD iterations
                    beta_old = beta.copy()
                    for k_idx in range(p - 1):
                        # Partial residual
                        residual = s_12[k_idx] - S_11[k_idx, :].dot(beta) + S_11[k_idx, k_idx] * beta[k_idx]
                        # Soft thresholding
                        beta[k_idx] = _soft_threshold(residual / S_11[k_idx, k_idx], self.alpha / S_11[k_idx, k_idx])
                    if np.max(np.abs(beta - beta_old)) < 1e-6:
                        break

                # Update W and theta for feature j
                theta_j = np.zeros(p)
                theta_j[not_j] = beta
                w_j = S_11 @ beta
                theta_j[j] = 1.0 / (S[j, j] + self.alpha + beta.dot(w_j))
                w_12 = w_j * theta_j[j]

                W[j, not_j] = w_12
                W[not_j, j] = w_12
                W[j, j] = S[j, j] + self.alpha
                theta[j, :] = theta_j
                theta[:, j] = theta_j

            # Check convergence via dual gap
            gap = np.abs(np.trace(W @ theta) - p)
            if gap < self.tol:
                break
            if np.max(np.abs(W - W_old)) < self.tol:
                break

        # Convert to target backend
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        _ref = None
        if backend_name == "torch":
            import torch
            _ref = torch.empty(0, dtype=torch.float64, device="cuda")
        kw = {"device": _ref.device} if _ref else {}

        self.covariance_ = xp.asarray(W, dtype=xp.float64, **kw)
        self.precision_ = xp.asarray(theta, dtype=xp.float64, **kw)
        self.location_ = xp.asarray(location_np, dtype=xp.float64, **kw)
        self.n_iter_ = iteration + 1
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["alpha"] = self.alpha
        params["max_iter"] = self.max_iter
        params["tol"] = self.tol
        return params

    def set_params(self, **params):
        for key in ["alpha", "max_iter", "tol"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self


class GraphicalLassoCV(EmpiricalCovariance):
    """
    Graphical Lasso with cross-validated regularization parameter.

    Selects the best ``alpha`` from a grid by maximizing the log-likelihood
    on held-out folds.

    Parameters
    ----------
    alphas : int or array-like, default=4
        If int, number of alpha values to try (log-spaced from 0.01 to 1).
        If array-like, the specific alpha values to try.
    cv : int, default=5
        Number of cross-validation folds.
    max_iter : int, default=100
        Maximum number of GLasso iterations per alpha.
    tol : float, default=1e-4
        Convergence tolerance.
    assume_centered : bool, default=False
        If True, data is assumed to be already centered.
    device : str or Device, default='auto'
        Computation device.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix (at best alpha).
    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix (at best alpha).
    location_ : ndarray of shape (n_features,)
        Estimated mean.
    alpha_ : float
        Best alpha selected by cross-validation.
    cv_results_ : list of dict
        Results for each alpha value tried.
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.
    """

    def __init__(
        self,
        alphas: Union[int, list] = 4,
        cv: int = 5,
        max_iter: int = 100,
        tol: float = 1e-4,
        assume_centered: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(assume_centered=assume_centered, device=device, n_jobs=n_jobs)
        self.alphas = alphas
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """Fit the graphical lasso model with cross-validated alpha.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n, p = X_np.shape

        # Build alpha grid
        if isinstance(self.alphas, int):
            alpha_grid = np.logspace(-2, 0, self.alphas)
        else:
            alpha_grid = np.asarray(self.alphas, dtype=np.float64)

        # K-fold CV
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        fold_size = n // self.cv
        folds = []
        for i in range(self.cv):
            start = i * fold_size
            end = start + fold_size if i < self.cv - 1 else n
            folds.append(indices[start:end])

        cv_results = []
        best_score = -np.inf
        best_alpha = alpha_grid[0]

        for alpha in alpha_grid:
            scores = []
            for k in range(self.cv):
                test_idx = folds[k]
                train_idx = np.concatenate([folds[j] for j in range(self.cv) if j != k])

                X_train = X_np[train_idx]
                X_test = X_np[test_idx]

                # Fit GLasso on train
                gl = GraphicalLasso(
                    alpha=alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    assume_centered=self.assume_centered,
                    device="cpu",
                )
                gl.fit(X_train)

                # Score on test: log-likelihood
                n_test = X_test.shape[0]
                loc = np.asarray(gl.location_)
                prec = np.asarray(gl.precision_)
                cov = np.asarray(gl.covariance_)

                X_centered = X_test - loc
                mahal = np.sum(X_centered @ prec * X_centered, axis=1)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    scores.append(-np.inf)
                    continue
                ll = -0.5 * (p * np.log(2 * np.pi) + logdet + mahal.mean())
                scores.append(ll)

            mean_score = np.mean(scores)
            cv_results.append({"alpha": alpha, "mean_score": mean_score, "scores": scores})

            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Fit final model with best alpha
        gl_final = GraphicalLasso(
            alpha=best_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            assume_centered=self.assume_centered,
            device="cpu",
        )
        gl_final.fit(X_np if self.assume_centered else X_np - X_np.mean(axis=0))

        # Convert to target backend
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        _ref = None
        if backend_name == "torch":
            import torch
            _ref = torch.empty(0, dtype=torch.float64, device="cuda")
        kw = {"device": _ref.device} if _ref else {}

        self.covariance_ = xp.asarray(np.asarray(gl_final.covariance_), dtype=xp.float64, **kw)
        self.precision_ = xp.asarray(np.asarray(gl_final.precision_), dtype=xp.float64, **kw)
        self.location_ = xp.asarray(np.asarray(gl_final.location_), dtype=xp.float64, **kw)
        self.alpha_ = best_alpha
        self.cv_results_ = cv_results
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["alphas"] = self.alphas
        params["cv"] = self.cv
        params["max_iter"] = self.max_iter
        params["tol"] = self.tol
        return params

    def set_params(self, **params):
        for key in ["alphas", "cv", "max_iter", "tol"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self


def _soft_threshold(x, threshold):
    """Soft thresholding operator."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    return 0.0
