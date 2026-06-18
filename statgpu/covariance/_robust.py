"""Minimum Covariance Determinant (MCD) robust covariance estimator with GPU support."""

from __future__ import annotations

__all__ = ["MinCovDet"]

from typing import Optional, Union

import numpy as np
from scipy.stats import chi2 as _chi2

from statgpu._config import Device
from statgpu.backends import _get_xp, _to_numpy

from statgpu.covariance._empirical import (
    EmpiricalCovariance,
    _detect_backend,
    _stable_inv,
)


def _consistency_factor(p, alpha):
    """Compute the consistency correction factor for the MCD estimator.

    From Croux & Haesbroeck (1999).

    Parameters
    ----------
    p : int
        Number of features.
    alpha : float
        Fraction of observations in the support (h/n).

    Returns
    -------
    c : float
        Consistency correction factor.
    """
    q_alpha = _chi2.ppf(alpha, df=p)
    c_alpha = alpha / _chi2.cdf(q_alpha, df=p + 2)
    return c_alpha


def _fast_logdet(cov):
    """Compute log(det(cov)) using Cholesky for numerical stability."""
    try:
        L = np.linalg.cholesky(cov)
        return 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        return -np.inf


class MinCovDet(EmpiricalCovariance):
    """
    Minimum Covariance Determinant (MCD) robust covariance estimator.

    Finds the subset of ``h`` observations (out of ``n``) whose covariance
    matrix has the smallest determinant, yielding a robust estimate of
    location and scatter that is resistant to outliers.

    Uses the FAST-MCD algorithm of Rousseeuw & Van Driessen (1999) with
    multi-stage C-steps for refinement, consistency correction factors,
    and reweighting.

    Parameters
    ----------
    support_fraction : float or None, default=None
        Fraction of observations to use for computing the MCD.
        Default: ``ceil(0.5 * (n + p + 1)) / n``.
    random_state : int or None, default=None
        Random seed for initial subset selection.
    assume_centered : bool, default=False
        If True, data is assumed to be already centered.
    device : str or Device, default='auto'
        Computation device.
    n_jobs : int or None, default=None
        Number of parallel jobs (reserved for future use).

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Robust covariance estimate (reweighted and consistency-corrected).
    location_ : ndarray of shape (n_features,)
        Robust location estimate.
    precision_ : ndarray of shape (n_features, n_features)
        Precision matrix (inverse covariance).
    support_ : ndarray of shape (n_samples,) of bool
        Boolean mask indicating which observations are in the support set.
    raw_covariance_ : ndarray
        Raw covariance estimate before reweighting.
    raw_location_ : ndarray
        Raw location estimate before reweighting.
    dist_ : ndarray of shape (n_samples,)
        Mahalanobis distances of the training observations.
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.

    References
    ----------
    Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the
    minimum covariance determinant estimator. *Technometrics*, 41(3), 212-223.
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

    def fit(self, X, y=None):
        """Fit the MCD covariance model to *X*.

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
        if n < 2:
            raise ValueError(f"Need at least 2 samples, got {n}")

        # Determine h (support size) -- use ceil like sklearn
        if self.support_fraction is not None:
            h = int(np.ceil(self.support_fraction * n))
            h = max(h, p + 1)
            h = min(h, n)
        else:
            h = min(int(np.ceil(0.5 * (n + p + 1))), n)

        rng = np.random.RandomState(self.random_state)

        # ---- Multi-stage FAST-MCD ----
        if n <= 500:
            # Small dataset: direct random starts
            best_subset = self._fast_mcd_small(X_np, h, rng)
        else:
            # Large dataset: 3-stage algorithm
            best_subset = self._fast_mcd_large(X_np, h, rng)

        # Raw estimates from best subset
        X_sub = X_np[best_subset]
        raw_location = X_sub.mean(axis=0)
        raw_cov = (X_sub - raw_location).T @ (X_sub - raw_location) / float(h)

        # Consistency correction factor for raw estimate
        alpha_raw = h / n
        c_raw = _consistency_factor(p, alpha_raw)
        raw_cov_corrected = raw_cov * c_raw

        # Reweighting: use CORRECTED distances
        raw_cov_inv = np.linalg.pinv(raw_cov_corrected)
        X_centered = X_np - raw_location
        mahal_raw = np.sum(X_centered @ raw_cov_inv * X_centered, axis=1)

        # Second consistency factor for reweighting (alpha = 0.975)
        c_reweight = _consistency_factor(p, 0.975)

        # Reweighted support: chi2 threshold at 0.975
        threshold = _chi2.ppf(0.975, p)
        support = mahal_raw <= threshold
        n_support = support.sum()

        if n_support < p + 1:
            # Fallback to raw estimates
            support_mask = np.zeros(n, dtype=bool)
            support_mask[best_subset] = True
            final_location = raw_location
            final_cov = raw_cov_corrected
            dist_final = mahal_raw
        else:
            X_support = X_np[support]
            final_location = X_support.mean(axis=0)
            final_cov_emp = (X_support - final_location).T @ (X_support - final_location) / float(n_support)
            final_cov = final_cov_emp * c_reweight
            support_mask = support

            # Final Mahalanobis distances with reweighted covariance
            final_cov_inv = np.linalg.pinv(final_cov)
            X_centered_final = X_np - final_location
            dist_final = np.sum(X_centered_final @ final_cov_inv * X_centered_final, axis=1)

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
        cov_arr = xp.asarray(final_cov, dtype=xp.float64, **kw)
        loc_arr = xp.asarray(final_location, dtype=xp.float64, **kw)
        raw_cov_arr = xp.asarray(raw_cov, dtype=xp.float64, **kw)
        raw_loc_arr = xp.asarray(raw_location, dtype=xp.float64, **kw)

        precision = _stable_inv(cov_arr, xp, backend_name)

        self.covariance_ = cov_arr
        self.location_ = loc_arr
        self.precision_ = precision
        self.support_ = support_mask
        self.raw_covariance_ = raw_cov_arr
        self.raw_location_ = raw_loc_arr
        self.dist_ = xp.asarray(dist_final, dtype=xp.float64, **kw)
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

    def _fast_mcd_small(self, X, h, rng):
        """FAST-MCD for n <= 500: 30 trials, 2 initial C-steps, keep top 10, full C-steps."""
        n = X.shape[0]
        n_trials = 30

        # Stage 1: 30 trials with 2 C-step iterations each
        candidates = []
        for _ in range(n_trials):
            subset = rng.choice(n, size=h, replace=False)
            logdet, subset = self._c_step(X, subset, h, max_iter=2)
            candidates.append((logdet, subset))

        # Keep top 10
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:10]

        # Stage 2: full C-steps from top candidates
        best_logdet = np.inf
        best_subset = None
        for _, subset in top_candidates:
            logdet, subset = self._c_step(X, subset, h, max_iter=30)
            if logdet < best_logdet:
                best_logdet = logdet
                best_subset = subset

        return best_subset

    def _fast_mcd_large(self, X, h, rng):
        """FAST-MCD for n > 500: 3-stage algorithm (Rousseeuw & Van Driessen 1999)."""
        n, p = X.shape

        # Stage 1: split into subsets of ~300, run 500 total trials
        subset_size = min(300, n)
        n_subsets = max(1, n // subset_size)
        n_trials_per_subset = max(10, 500 // n_subsets)

        # Compute initial robust location from the median of each variable
        # (fast, no iteration needed)
        initial_loc = np.median(X, axis=0)
        initial_cov = np.cov(X, rowvar=False, bias=True)

        all_candidates = []
        for s in range(n_subsets):
            start = s * subset_size
            end = min(start + subset_size, n)
            X_sub = X[start:end]
            n_sub = X_sub.shape[0]
            h_sub = min(h, n_sub)

            for _ in range(n_trials_per_subset):
                subset = rng.choice(n_sub, size=h_sub, replace=False)
                logdet, subset_local = self._c_step(X_sub, subset, h_sub, max_iter=2)
                # Map local indices to global
                subset_global = subset_local + start
                all_candidates.append((logdet, subset_global))

        # Keep top 10
        all_candidates.sort(key=lambda x: x[0])
        top_candidates = all_candidates[:10]

        # Stage 2: pool and run full C-steps
        best_logdet = np.inf
        best_subset = None
        for _, subset in top_candidates:
            logdet, subset = self._c_step(X, subset, h, max_iter=30)
            if logdet < best_logdet:
                best_logdet = logdet
                best_subset = subset

        return best_subset

    @staticmethod
    def _c_step(X, subset, h, max_iter=30):
        """Perform C-steps: recompute covariance from subset, select h
        observations with smallest Mahalanobis distances.

        Returns (logdet, subset) where logdet is the log-determinant of
        the covariance (for numerical stability).
        """
        n = X.shape[0]
        best_logdet = np.inf
        best_subset = subset.copy()

        for _ in range(max_iter):
            X_sub = X[subset]
            loc = X_sub.mean(axis=0)
            cov = (X_sub - loc).T @ (X_sub - loc) / float(h)

            # Use logdet for numerical stability
            logdet = _fast_logdet(cov)
            if logdet == -np.inf:
                break

            # Stop if determinant stopped improving
            if logdet >= best_logdet:
                break

            best_logdet = logdet
            best_subset = subset.copy()

            try:
                cov_inv = np.linalg.pinv(cov)
            except np.linalg.LinAlgError:
                break

            X_centered = X - loc
            mahal = np.sum(X_centered @ cov_inv * X_centered, axis=1)
            # Use argpartition for O(n) selection
            new_subset = np.argpartition(mahal, h - 1)[:h]
            new_subset.sort()

            if np.array_equal(new_subset, subset):
                break
            subset = new_subset

        return best_logdet, best_subset

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["support_fraction"] = self.support_fraction
        params["random_state"] = self.random_state
        return params

    def set_params(self, **params):
        for key in ["support_fraction", "random_state"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
