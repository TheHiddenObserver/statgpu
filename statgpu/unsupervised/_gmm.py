"""Gaussian mixture models."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._kmeans import KMeans
from statgpu.unsupervised._utils import check_2d_array, reject_sparse, scalar_to_float


class GaussianMixture(BaseEstimator):
    """Diagonal-covariance Gaussian mixture model fitted with EM."""

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "diag",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def _validate_params(self, n_samples: int):
        if not isinstance(self.n_components, (int, np.integer)) or int(self.n_components) < 1:
            raise ValueError("n_components must be a positive integer")
        if int(self.n_components) > n_samples:
            raise ValueError("n_components must be less than or equal to n_samples")
        if self.covariance_type != "diag":
            raise NotImplementedError("GaussianMixture v1 only supports covariance_type='diag'")
        if self.init_params not in ("kmeans", "random"):
            raise ValueError("init_params must be one of: 'kmeans', 'random'")
        if float(self.tol) < 0.0:
            raise ValueError("tol must be non-negative")
        if float(self.reg_covar) < 0.0:
            raise ValueError("reg_covar must be non-negative")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(self.n_init, (int, np.integer)) or int(self.n_init) < 1:
            raise ValueError("n_init must be a positive integer")

    def _estimate_log_gaussian_prob(self, backend, X, means, covariances):
        n_features = X.shape[1]
        precisions = 1.0 / covariances
        log_det = backend.sum(backend.log(covariances), axis=1)
        x2 = backend.matmul(X * X, precisions.T)
        cross = backend.matmul(X, (means * precisions).T)
        mean2 = backend.sum(means * means * precisions, axis=1)
        quad = x2 - 2.0 * cross + backend.expand_dims(mean2, 0)
        return -0.5 * (float(n_features) * np.log(2.0 * np.pi) + backend.expand_dims(log_det, 0) + quad)

    def _estimate_weighted_log_prob(self, backend, X, weights, means, covariances):
        return self._estimate_log_gaussian_prob(backend, X, means, covariances) + backend.expand_dims(backend.log(weights), 0)

    def _e_step(self, backend, X, weights, means, covariances):
        weighted_log_prob = self._estimate_weighted_log_prob(backend, X, weights, means, covariances)
        log_prob_norm = backend.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - backend.expand_dims(log_prob_norm, 1)
        return scalar_to_float(backend.mean(log_prob_norm)), backend.exp(log_resp)

    def _m_step(self, backend, X, resp):
        n_samples = X.shape[0]
        nk = backend.sum(resp, axis=0) + 10.0 * np.finfo(np.float64).eps
        weights = nk / float(n_samples)
        means = backend.matmul(resp.T, X) / backend.expand_dims(nk, 1)
        second_moment = backend.matmul(resp.T, X * X) / backend.expand_dims(nk, 1)
        covariances = backend.maximum(second_moment - means * means, float(self.reg_covar))
        return weights, means, covariances

    def _initialize(self, backend, X, seed):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(seed)
        if self.init_params == "kmeans":
            km = KMeans(
                n_clusters=int(self.n_components),
                n_init=1,
                max_iter=min(50, int(self.max_iter)),
                random_state=seed,
                device=self.device,
            ).fit(X)
            means = km.cluster_centers_
        else:
            indices = rng.choice(n_samples, size=int(self.n_components), replace=False)
            means = X[backend.asarray(indices, dtype=backend.int64)]
        weights = backend.full((int(self.n_components),), 1.0 / float(self.n_components), dtype=backend.float64)
        global_var = backend.mean((X - backend.mean(X, axis=0)) ** 2, axis=0) + float(self.reg_covar)
        covariances = backend.ones((int(self.n_components), n_features), dtype=backend.float64) * global_var
        return weights, means, covariances

    def fit(self, X, y=None):
        reject_sparse(X, "GaussianMixture")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(int(self.n_init)):
            seed = None if self.random_state is None else int(rng.integers(0, np.iinfo(np.int32).max))
            weights, means, covariances = self._initialize(backend, X_arr, seed)
            lower_bound = -np.inf
            converged = False
            n_iter = 0
            for n_iter in range(1, int(self.max_iter) + 1):
                prev_lower_bound = lower_bound
                lower_bound, resp = self._e_step(backend, X_arr, weights, means, covariances)
                weights, means, covariances = self._m_step(backend, X_arr, resp)
                if abs(lower_bound - prev_lower_bound) < float(self.tol):
                    converged = True
                    break
            if best is None or lower_bound > best[0]:
                best = (lower_bound, converged, n_iter, weights, means, covariances)

        lower_bound, converged, n_iter, weights, means, covariances = best
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = 1.0 / backend.sqrt(covariances)
        self.converged_ = bool(converged)
        self.n_iter_ = int(n_iter)
        self.lower_bound_ = float(lower_bound)
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def score_samples(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        return backend.logsumexp(
            self._estimate_weighted_log_prob(backend, X_arr, self.weights_, self.means_, self.covariances_),
            axis=1,
        )

    def predict_proba(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        _, resp = self._e_step(backend, X_arr, self.weights_, self.means_, self.covariances_)
        return resp

    def predict(self, X):
        backend = self._get_backend()
        return backend.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).predict(X)

    def score(self, X, y=None):
        backend = self._get_backend()
        return scalar_to_float(backend.mean(self.score_samples(X)))

    def _n_parameters(self):
        return int(self.n_components) * self.n_features_in_ * 2 + int(self.n_components) - 1

    def bic(self, X):
        return -2.0 * float(self.score(X)) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])

    def aic(self, X):
        return -2.0 * float(self.score(X)) * X.shape[0] + 2.0 * self._n_parameters()

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "tol": self.tol,
                "reg_covar": self.reg_covar,
                "max_iter": self.max_iter,
                "n_init": self.n_init,
                "init_params": self.init_params,
                "random_state": self.random_state,
            }
        )
        return params
