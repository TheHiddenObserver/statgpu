"""Graphical Lasso for sparse inverse covariance estimation with GPU support."""

from __future__ import annotations

__all__ = ["GraphicalLasso", "GraphicalLassoCV"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _get_xp, _to_numpy

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
        """Fit graphical lasso by covariance block coordinate descent."""
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if isinstance(self.max_iter, bool) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.tol) or float(self.tol) <= 0:
            raise ValueError("tol must be finite and positive")

        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if X_np.ndim != 2 or X_np.shape[0] < 2 or X_np.shape[1] < 1:
            raise ValueError("X must be a non-empty 2D array with at least 2 samples")
        if not np.all(np.isfinite(X_np)):
            raise ValueError("X contains NaN or infinite values")

        n, p = X_np.shape
        if self.assume_centered:
            location_np = np.zeros(p, dtype=np.float64)
            X_centered = X_np
        else:
            location_np = X_np.mean(axis=0)
            X_centered = X_np - location_np
        empirical = X_centered.T @ X_centered / float(n)

        if alpha == 0.0 or p == 1:
            covariance = empirical.copy()
            precision = np.linalg.pinv(covariance)
            self.n_iter_ = 1
        else:
            covariance = empirical.copy()
            np.fill_diagonal(covariance, np.diag(empirical))
            inner_tol = min(1e-8, float(self.tol) * 0.1)
            beta_cache = [np.zeros(p - 1, dtype=np.float64) for _ in range(p)]
            self.n_iter_ = 0

            for outer in range(int(self.max_iter)):
                previous = covariance.copy()
                self.n_iter_ = outer + 1
                for j in range(p):
                    mask = np.arange(p) != j
                    W11 = covariance[np.ix_(mask, mask)]
                    s12 = empirical[mask, j]
                    beta = beta_cache[j].copy()

                    for _ in range(1000):
                        beta_old = beta.copy()
                        for coordinate in range(p - 1):
                            diagonal = W11[coordinate, coordinate]
                            if diagonal <= 0:
                                raise ValueError("GraphicalLasso encountered a non-positive covariance diagonal")
                            partial = s12[coordinate] - W11[coordinate] @ beta + diagonal * beta[coordinate]
                            beta[coordinate] = _soft_threshold(partial, alpha) / diagonal
                        if np.max(np.abs(beta - beta_old)) <= inner_tol:
                            break

                    beta_cache[j] = beta
                    w12 = W11 @ beta
                    covariance[mask, j] = w12
                    covariance[j, mask] = w12
                    covariance[j, j] = empirical[j, j]

                if np.max(np.abs(covariance - previous)) <= float(self.tol):
                    break

            covariance = 0.5 * (covariance + covariance.T)
            precision = np.linalg.pinv(covariance)
            precision = 0.5 * (precision + precision.T)

        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        _ref = None
        if backend_name == "torch":
            import torch
            device = self._get_compute_device()
            target = "cuda" if device.value in ("torch", "cuda") else "cpu"
            _ref = torch.empty(0, dtype=torch.float64, device=target)
        kwargs = {"device": _ref.device} if _ref is not None else {}

        self.covariance_ = xp.asarray(covariance, dtype=xp.float64, **kwargs)
        self.precision_ = xp.asarray(precision, dtype=xp.float64, **kwargs)
        self.location_ = xp.asarray(location_np, dtype=xp.float64, **kwargs)
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
        random_state: Optional[int] = None,
        assume_centered: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(assume_centered=assume_centered, device=device, n_jobs=n_jobs)
        self.alphas = alphas
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

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
        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        n, p = X_np.shape
        if n < 2 or p < 1 or not np.all(np.isfinite(X_np)):
            raise ValueError("X must be a finite 2D array with at least 2 samples")
        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)):
            raise ValueError("cv must be an integer")
        if int(self.cv) < 2 or int(self.cv) > n:
            raise ValueError("cv must satisfy 2 <= cv <= n_samples")

        if isinstance(self.alphas, (int, np.integer)) and not isinstance(self.alphas, bool):
            if int(self.alphas) < 1:
                raise ValueError("alphas must be a positive integer or a non-empty array")
            alpha_grid = np.logspace(-2, 0, int(self.alphas))
        else:
            alpha_grid = np.asarray(self.alphas, dtype=np.float64).ravel()
        if alpha_grid.size == 0 or not np.all(np.isfinite(alpha_grid)) or np.any(alpha_grid < 0):
            raise ValueError("alphas must be finite, non-negative, and non-empty")

        # K-fold CV
        rng = np.random.RandomState(self.random_state)
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
        gl_final.fit(X_np)  # GraphicalLasso handles centering internally

        # Convert to target backend
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        _ref = None
        if backend_name == "torch":
            import torch
            _dev = self._get_compute_device()
            _cuda_dev = "cuda" if _dev.value in ("torch", "cuda") else "cpu"
            _ref = torch.empty(0, dtype=torch.float64, device=_cuda_dev)
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
    """Soft thresholding operator (vectorized)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)
