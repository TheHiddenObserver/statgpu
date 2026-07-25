"""Minimum Covariance Determinant estimator with native backend C-steps."""

from __future__ import annotations

__all__ = ["MinCovDet"]

from typing import Optional, Union

import numpy as np
from scipy.stats import chi2 as _chi2

from statgpu._config import Device
from statgpu.backends import _get_xp, _to_float_scalar, xp_asarray, xp_zeros
from statgpu.covariance._empirical import EmpiricalCovariance, _detect_backend, _stable_inv


def _consistency_factor(p, alpha):
    q_alpha = _chi2.ppf(alpha, df=p)
    return alpha / _chi2.cdf(q_alpha, df=p + 2)


def _copy_array(x):
    return x.clone() if hasattr(x, "clone") else x.copy()


def _index_array(indices, xp, ref):
    return xp_asarray(
        np.asarray(indices, dtype=np.int64), dtype=xp.int64, xp=xp, ref_arr=ref
    )


def _finite_all(x, xp) -> bool:
    return bool(_to_float_scalar(xp.all(xp.isfinite(x))))


def _fast_logdet(cov, xp):
    sign, logdet = xp.linalg.slogdet(cov)
    if _to_float_scalar(sign) <= 0.0:
        return float("-inf")
    value = _to_float_scalar(logdet)
    return value if np.isfinite(value) else float("-inf")


class MinCovDet(EmpiricalCovariance):
    """Minimum Covariance Determinant robust covariance estimator.

    Random subset indices and chi-square cutoffs are generated on the CPU, but
    covariance, inverse, distance, ordering, C-step, and reweighting operations
    stay on the selected NumPy, CuPy, or Torch backend.
    """

    def __init__(
        self,
        support_fraction: Optional[float] = None,
        random_state: Optional[int] = None,
        assume_centered: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(assume_centered=assume_centered, device=device, n_jobs=n_jobs)
        self.support_fraction = support_fraction
        self.random_state = random_state

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
            raise ValueError("X must be a finite 2D array with at least 2 samples")
        if not _finite_all(X_arr, xp):
            raise ValueError("X contains NaN or infinite values")
        return backend_name, xp, X_arr

    def fit(self, X, y=None):
        backend_name, xp, X_arr = self._prepare_input(X)
        n, p = int(X_arr.shape[0]), int(X_arr.shape[1])
        if n <= p:
            raise ValueError(
                "MinCovDet requires n_samples > n_features for a nonsingular support covariance"
            )

        if self.support_fraction is not None:
            fraction = float(self.support_fraction)
            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("support_fraction must be finite and in (0, 1]")
            h = int(np.ceil(fraction * n))
            h = min(max(h, p + 1), n)
        else:
            h = min(int(np.ceil(0.5 * (n + p + 1))), n)

        rng = np.random.RandomState(self.random_state)
        n_trials = 30 if n <= 500 else 50
        best_subset = self._fast_mcd(X_arr, h, rng, xp, backend_name, n_trials)

        X_sub = X_arr[best_subset]
        if self.assume_centered:
            raw_location = xp_zeros(p, xp.float64, xp, X_arr)
            raw_centered = X_sub
        else:
            raw_location = xp.mean(X_sub, axis=0)
            raw_centered = X_sub - raw_location
        raw_cov = raw_centered.T @ raw_centered / float(h)

        c_raw = _consistency_factor(p, h / n)
        raw_cov_corrected = raw_cov * c_raw
        raw_cov_inv = _stable_inv(raw_cov_corrected, xp, backend_name)
        centered_all = X_arr if self.assume_centered else X_arr - raw_location
        mahal_raw = xp.sum((centered_all @ raw_cov_inv) * centered_all, axis=1)

        threshold = float(_chi2.ppf(0.975, p))
        support = mahal_raw <= threshold
        n_support = int(_to_float_scalar(xp.sum(support)))
        c_reweight = _consistency_factor(p, 0.975)

        if n_support < p + 1:
            support_mask = (
                xp.zeros(n, dtype=xp.bool, device=X_arr.device)
                if backend_name == "torch"
                else xp.zeros(n, dtype=xp.bool_)
            )
            support_mask[best_subset] = True
            final_location = raw_location
            final_cov = raw_cov_corrected
            dist_final = mahal_raw
        else:
            X_support = X_arr[support]
            if self.assume_centered:
                final_location = xp_zeros(p, xp.float64, xp, X_arr)
                final_centered = X_support
            else:
                final_location = xp.mean(X_support, axis=0)
                final_centered = X_support - final_location
            final_cov = (final_centered.T @ final_centered / float(n_support)) * c_reweight
            support_mask = support
            final_inv = _stable_inv(final_cov, xp, backend_name)
            centered_final = X_arr if self.assume_centered else X_arr - final_location
            dist_final = xp.sum((centered_final @ final_inv) * centered_final, axis=1)

        self.covariance_ = final_cov
        self.location_ = final_location
        self.precision_ = _stable_inv(final_cov, xp, backend_name)
        self.support_ = support_mask
        self.raw_covariance_ = raw_cov
        self.raw_location_ = raw_location
        self.dist_ = dist_final
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

    def _fast_mcd(self, X, h, rng, xp, backend_name, n_trials):
        n = int(X.shape[0])
        candidates = []
        for _ in range(int(n_trials)):
            subset = _index_array(rng.choice(n, size=h, replace=False), xp, X)
            logdet, refined = self._c_step(
                X, subset, h, max_iter=2, xp=xp, backend_name=backend_name
            )
            candidates.append((logdet, refined))
        candidates.sort(key=lambda item: item[0])

        best_logdet = np.inf
        best_subset = None
        for _, subset in candidates[: min(10, len(candidates))]:
            logdet, refined = self._c_step(
                X, subset, h, max_iter=30, xp=xp, backend_name=backend_name
            )
            if logdet < best_logdet:
                best_logdet = logdet
                best_subset = refined
        if best_subset is None:
            raise ValueError(
                "MinCovDet could not find a positive-definite support covariance"
            )
        return best_subset

    def _c_step(self, X, subset, h, max_iter, xp, backend_name):
        best_logdet = np.inf
        best_subset = _copy_array(subset)

        for _ in range(int(max_iter)):
            X_sub = X[subset]
            if self.assume_centered:
                loc = xp_zeros(int(X.shape[1]), xp.float64, xp, X)
                centered = X_sub
            else:
                loc = xp.mean(X_sub, axis=0)
                centered = X_sub - loc
            cov = centered.T @ centered / float(h)
            logdet = _fast_logdet(cov, xp)
            if logdet == float("-inf") or logdet >= best_logdet:
                break

            best_logdet = logdet
            best_subset = _copy_array(subset)
            cov_inv = _stable_inv(cov, xp, backend_name)
            centered_all = X if self.assume_centered else X - loc
            mahal = xp.sum((centered_all @ cov_inv) * centered_all, axis=1)
            new_subset = xp.argsort(mahal)[:h]
            if bool(_to_float_scalar(xp.all(new_subset == subset))):
                break
            subset = new_subset

        return best_logdet, best_subset

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        return params

    def set_params(self, **params):
        for key in ["support_fraction", "random_state"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
