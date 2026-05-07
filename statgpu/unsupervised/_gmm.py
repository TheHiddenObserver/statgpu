"""Gaussian mixture models."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._kmeans import KMeans
from statgpu.unsupervised._utils import check_2d_array, reject_sparse, scalar_to_float


class GaussianMixture(BaseEstimator):
    """Gaussian mixture model fitted with log-domain EM."""

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
        if self.covariance_type not in ("diag", "spherical", "tied", "full"):
            raise ValueError("covariance_type must be one of: 'diag', 'spherical', 'tied', 'full'")
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

    def _linalg_inv(self, backend, matrix):
        return backend.xp.linalg.inv(matrix)

    def _linalg_logdet(self, backend, matrix):
        sign, logabsdet = backend.xp.linalg.slogdet(matrix)
        if scalar_to_float(sign) <= 0.0:
            raise ValueError("covariance matrix must be positive definite")
        return logabsdet

    def _linalg_cholesky(self, backend, matrix):
        return backend.xp.linalg.cholesky(matrix)

    def _eye(self, backend, n_features: int):
        if hasattr(backend, "eye"):
            return backend.eye(n_features, dtype=backend.float64)
        return backend.asarray(np.eye(n_features), dtype=backend.float64)

    def _estimate_log_gaussian_prob(self, backend, X, means, covariances):
        n_features = X.shape[1]
        log_2pi = float(n_features) * np.log(2.0 * np.pi)
        if self.covariance_type == "diag":
            precisions = 1.0 / covariances
            log_det = backend.sum(backend.log(covariances), axis=1)
            x2 = backend.matmul(X * X, precisions.T)
            cross = backend.matmul(X, (means * precisions).T)
            mean2 = backend.sum(means * means * precisions, axis=1)
            quad = x2 - 2.0 * cross + backend.expand_dims(mean2, 0)
            return -0.5 * (log_2pi + backend.expand_dims(log_det, 0) + quad)

        if self.covariance_type == "spherical":
            precisions = 1.0 / covariances
            log_det = float(n_features) * backend.log(covariances)
            diff = backend.expand_dims(X, 1) - backend.expand_dims(means, 0)
            quad = backend.sum(diff * diff, axis=2) * backend.expand_dims(precisions, 0)
            return -0.5 * (log_2pi + backend.expand_dims(log_det, 0) + quad)

        log_probs = []
        if self.covariance_type == "tied":
            precision = self._linalg_inv(backend, covariances)
            log_det = self._linalg_logdet(backend, covariances)
            for k in range(int(self.n_components)):
                diff = X - means[k]
                quad = backend.sum(backend.matmul(diff, precision) * diff, axis=1)
                log_probs.append(-0.5 * (log_2pi + log_det + quad))
            return backend.stack(log_probs, axis=1)

        for k in range(int(self.n_components)):
            precision = self._linalg_inv(backend, covariances[k])
            log_det = self._linalg_logdet(backend, covariances[k])
            diff = X - means[k]
            quad = backend.sum(backend.matmul(diff, precision) * diff, axis=1)
            log_probs.append(-0.5 * (log_2pi + log_det + quad))
        return backend.stack(log_probs, axis=1)

    def _estimate_weighted_log_prob(self, backend, X, weights, means, covariances):
        return self._estimate_log_gaussian_prob(backend, X, means, covariances) + backend.expand_dims(backend.log(weights), 0)

    def _e_step(self, backend, X, weights, means, covariances):
        weighted_log_prob = self._estimate_weighted_log_prob(backend, X, weights, means, covariances)
        log_prob_norm = backend.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - backend.expand_dims(log_prob_norm, 1)
        return scalar_to_float(backend.mean(log_prob_norm)), backend.exp(log_resp)

    def _m_step(self, backend, X, resp):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        nk = backend.sum(resp, axis=0) + 10.0 * np.finfo(np.float64).eps
        weights = nk / float(n_samples)
        means = backend.matmul(resp.T, X) / backend.expand_dims(nk, 1)
        if self.covariance_type in ("diag", "spherical"):
            second_moment = backend.matmul(resp.T, X * X) / backend.expand_dims(nk, 1)
            diag_covariances = backend.maximum(second_moment - means * means, float(self.reg_covar))
            if self.covariance_type == "diag":
                return weights, means, diag_covariances
            spherical_covariances = backend.maximum(backend.mean(diag_covariances, axis=1), float(self.reg_covar))
            return weights, means, spherical_covariances

        eye = self._eye(backend, n_features)
        if self.covariance_type == "tied":
            covariance = backend.zeros((n_features, n_features), dtype=backend.float64)
            for k in range(int(self.n_components)):
                diff = X - means[k]
                weighted = diff * backend.expand_dims(resp[:, k], 1)
                covariance = covariance + backend.matmul(weighted.T, diff)
            covariance = covariance / float(n_samples) + float(self.reg_covar) * eye
            return weights, means, covariance

        covariances = []
        for k in range(int(self.n_components)):
            diff = X - means[k]
            weighted = diff * backend.expand_dims(resp[:, k], 1)
            covariance = backend.matmul(weighted.T, diff) / nk[k] + float(self.reg_covar) * eye
            covariances.append(covariance)
        covariances = backend.stack(covariances, axis=0)
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
        centered = X - backend.mean(X, axis=0)
        global_var = backend.mean(centered * centered, axis=0) + float(self.reg_covar)
        if self.covariance_type == "diag":
            covariances = backend.ones((int(self.n_components), n_features), dtype=backend.float64) * global_var
        elif self.covariance_type == "spherical":
            covariances = backend.full(
                (int(self.n_components),),
                scalar_to_float(backend.mean(global_var)),
                dtype=backend.float64,
            )
        else:
            global_covariance = backend.matmul(centered.T, centered) / float(n_samples)
            global_covariance = global_covariance + float(self.reg_covar) * self._eye(backend, n_features)
            if self.covariance_type == "tied":
                covariances = global_covariance
            else:
                covariances = backend.stack([backend.copy(global_covariance) for _ in range(int(self.n_components))], axis=0)
        return weights, means, covariances

    def _estimate_precisions_cholesky(self, backend, covariances):
        if self.covariance_type in ("diag", "spherical"):
            return 1.0 / backend.sqrt(covariances)
        if self.covariance_type == "tied":
            return self._linalg_cholesky(backend, self._linalg_inv(backend, covariances))
        return backend.stack(
            [self._linalg_cholesky(backend, self._linalg_inv(backend, covariances[k])) for k in range(int(self.n_components))],
            axis=0,
        )

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
        self.precisions_cholesky_ = self._estimate_precisions_cholesky(backend, covariances)
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
        n_components = int(self.n_components)
        n_features = int(self.n_features_in_)
        mean_params = n_components * n_features
        weight_params = n_components - 1
        if self.covariance_type == "diag":
            covariance_params = n_components * n_features
        elif self.covariance_type == "spherical":
            covariance_params = n_components
        elif self.covariance_type == "tied":
            covariance_params = n_features * (n_features + 1) // 2
        else:
            covariance_params = n_components * n_features * (n_features + 1) // 2
        return mean_params + covariance_params + weight_params

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
