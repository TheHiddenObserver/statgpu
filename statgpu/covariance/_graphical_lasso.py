"""Graphical Lasso for sparse inverse covariance estimation with native backend execution."""

from __future__ import annotations

__all__ = ["GraphicalLasso", "GraphicalLassoCV"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _get_xp, _to_float_scalar, xp_asarray, xp_zeros
from statgpu.covariance._empirical import EmpiricalCovariance, _detect_backend, _stable_inv


def _copy_array(x):
    return x.clone() if hasattr(x, "clone") else x.copy()


def _finite_all(x, xp) -> bool:
    return bool(_to_float_scalar(xp.all(xp.isfinite(x))))


def _index_array(indices, xp, ref):
    return xp_asarray(
        np.asarray(indices, dtype=np.int64), dtype=xp.int64, xp=xp, ref_arr=ref
    )


def _soft_threshold(x, threshold, xp):
    return xp.sign(x) * xp.maximum(xp.abs(x) - threshold, xp.zeros_like(x))


class GraphicalLasso(EmpiricalCovariance):
    """Sparse inverse covariance estimation via graphical lasso.

    The block-coordinate descent is executed on the selected NumPy, CuPy, or
    Torch backend. Only scalar convergence diagnostics are synchronized.
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

    def _prepare_input(self, X):
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        ref = None
        if backend_name == "torch":
            import torch

            if isinstance(X, torch.Tensor):
                ref = X
            else:
                dev = self._get_compute_device()
                target = "cuda" if dev.value in ("torch", "cuda") else "cpu"
                ref = torch.empty(0, dtype=torch.float64, device=target)
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=ref)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or int(X_arr.shape[0]) < 2 or int(X_arr.shape[1]) < 1:
            raise ValueError("X must be a non-empty 2D array with at least 2 samples")
        if not _finite_all(X_arr, xp):
            raise ValueError("X contains NaN or infinite values")
        return backend_name, xp, X_arr

    def fit(self, X, y=None):
        """Fit graphical lasso by covariance block coordinate descent."""
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, (int, np.integer))
            or int(self.max_iter) < 1
        ):
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(float(self.tol)) or float(self.tol) <= 0:
            raise ValueError("tol must be finite and positive")

        backend_name, xp, X_arr = self._prepare_input(X)
        n, p = int(X_arr.shape[0]), int(X_arr.shape[1])
        if self.assume_centered:
            location = xp_zeros(p, xp.float64, xp, X_arr)
            centered = X_arr
        else:
            location = xp.mean(X_arr, axis=0)
            centered = X_arr - location
        empirical = centered.T @ centered / float(n)

        if alpha == 0.0 or p == 1:
            covariance = _copy_array(empirical)
            precision = xp.linalg.pinv(covariance)
            self.n_iter_ = 1
        else:
            covariance = _copy_array(empirical)
            inner_tol = min(1e-8, float(self.tol) * 0.1)
            beta_cache = [xp_zeros(p - 1, xp.float64, xp, X_arr) for _ in range(p)]
            self.n_iter_ = 0

            for outer in range(int(self.max_iter)):
                previous = _copy_array(covariance)
                self.n_iter_ = outer + 1

                for j in range(p):
                    indices = [i for i in range(p) if i != j]
                    idx = _index_array(indices, xp, X_arr)
                    W11 = covariance[idx][:, idx]
                    s12 = empirical[idx, j]
                    beta = _copy_array(beta_cache[j])

                    for _ in range(1000):
                        beta_old = _copy_array(beta)
                        for coordinate in range(p - 1):
                            diagonal = W11[coordinate, coordinate]
                            if _to_float_scalar(diagonal) <= 0.0:
                                raise ValueError(
                                    "GraphicalLasso encountered a non-positive covariance diagonal"
                                )
                            partial = (
                                s12[coordinate]
                                - W11[coordinate] @ beta
                                + diagonal * beta[coordinate]
                            )
                            beta[coordinate] = _soft_threshold(partial, alpha, xp) / diagonal
                        delta = _to_float_scalar(xp.max(xp.abs(beta - beta_old)))
                        if delta <= inner_tol:
                            break

                    beta_cache[j] = beta
                    w12 = W11 @ beta
                    covariance[idx, j] = w12
                    covariance[j, idx] = w12
                    covariance[j, j] = empirical[j, j]

                outer_delta = _to_float_scalar(xp.max(xp.abs(covariance - previous)))
                if outer_delta <= float(self.tol):
                    break

            covariance = 0.5 * (covariance + covariance.T)
            precision = _stable_inv(covariance, xp, backend_name)
            precision = 0.5 * (precision + precision.T)

        self.covariance_ = covariance
        self.precision_ = precision
        self.location_ = location
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        return params

    def set_params(self, **params):
        for key in ["alpha", "max_iter", "tol"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self


class GraphicalLassoCV(EmpiricalCovariance):
    """Graphical Lasso with backend-native cross-validation."""

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
        probe = GraphicalLasso(
            alpha=0.0,
            max_iter=self.max_iter,
            tol=self.tol,
            assume_centered=self.assume_centered,
            device=self.device,
        )
        backend_name, xp, X_arr = probe._prepare_input(X)
        n, p = int(X_arr.shape[0]), int(X_arr.shape[1])

        if (
            isinstance(self.cv, bool)
            or not isinstance(self.cv, (int, np.integer))
            or int(self.cv) < 2
            or int(self.cv) > n
        ):
            raise ValueError("cv must satisfy 2 <= cv <= n_samples")

        if isinstance(self.alphas, (int, np.integer)) and not isinstance(self.alphas, bool):
            if int(self.alphas) < 1:
                raise ValueError("alphas must be a positive integer or a non-empty array")
            alpha_grid = np.logspace(-2, 0, int(self.alphas))
        else:
            alpha_grid = np.asarray(self.alphas, dtype=np.float64).ravel()
        if alpha_grid.size == 0 or not np.all(np.isfinite(alpha_grid)) or np.any(alpha_grid < 0):
            raise ValueError("alphas must be finite, non-negative, and non-empty")

        rng = np.random.RandomState(self.random_state)
        folds = np.array_split(rng.permutation(n), int(self.cv))
        cv_results = []
        best_score = -np.inf
        best_alpha = float(alpha_grid[0])

        for alpha in alpha_grid:
            scores = []
            for fold_index in range(int(self.cv)):
                test_np = folds[fold_index]
                train_np = np.concatenate(
                    [folds[j] for j in range(int(self.cv)) if j != fold_index]
                )
                train_idx = _index_array(train_np, xp, X_arr)
                test_idx = _index_array(test_np, xp, X_arr)
                X_train = X_arr[train_idx]
                X_test = X_arr[test_idx]

                model = GraphicalLasso(
                    alpha=float(alpha),
                    max_iter=self.max_iter,
                    tol=self.tol,
                    assume_centered=self.assume_centered,
                    device=self.device,
                ).fit(X_train)
                scores.append(float(model.score(X_test)))

            mean_score = float(np.mean(scores))
            cv_results.append(
                {"alpha": float(alpha), "mean_score": mean_score, "scores": scores}
            )
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = float(alpha)

        final = GraphicalLasso(
            alpha=best_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            assume_centered=self.assume_centered,
            device=self.device,
        ).fit(X_arr)

        self.covariance_ = final.covariance_
        self.precision_ = final.precision_
        self.location_ = final.location_
        self.alpha_ = best_alpha
        self.cv_results_ = cv_results
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            alphas=self.alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        return params

    def set_params(self, **params):
        for key in ["alphas", "cv", "max_iter", "tol", "random_state"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
