"""
CoxPHCV: Cross-validated Cox Proportional Hazards regression.

Implements K-fold cross-validation to select the optimal penalty (L2 regularization)
parameter for Cox PH models.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
from collections import OrderedDict
import hashlib
import numpy as np

from .._config import Device
from .._cv_base import CVEstimatorBase
from ._cox import CoxPH


# =============================================================================
# CV Cache
# =============================================================================

_COXPH_CV_CACHE_MAXSIZE = int(64)
_COXPH_CV_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _coxcv_cache_get(cache_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get cached CoxPH CV results."""
    if cache_key is None:
        return None
    val = _COXPH_CV_CACHE.get(cache_key)
    if val is not None:
        _COXPH_CV_CACHE.move_to_end(cache_key)
    return val


def _coxcv_cache_put(cache_key: Optional[str], value: Dict[str, Any]) -> None:
    """Put cached CoxPH CV results."""
    if cache_key is None:
        return
    _COXPH_CV_CACHE[cache_key] = value
    _COXPH_CV_CACHE.move_to_end(cache_key)
    while len(_COXPH_CV_CACHE) > _COXPH_CV_CACHE_MAXSIZE:
        _COXPH_CV_CACHE.popitem(last=False)


def _make_coxph_cv_auto_cache_key(
    X_shape: Tuple[int, ...],
    time_shape: Tuple[int, ...],
    event_shape: Tuple[int, ...],
    penalties: Optional[np.ndarray],
    n_penalties: int,
    penalty_min_ratio: float,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    ties: str,
    use_gpu: bool,
    max_iter: int,
    tol: float,
) -> str:
    """Generate automatic cache key for CoxPH CV."""
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(X_shape, dtype=np.int64).tobytes())
    h.update(np.asarray(time_shape, dtype=np.int64).tobytes())
    h.update(np.asarray(event_shape, dtype=np.int64).tobytes())
    if penalties is not None:
        h.update(np.asarray(penalties, dtype=np.float64).tobytes())
    h.update(str(n_penalties).encode("utf-8"))
    h.update(str(penalty_min_ratio).encode("utf-8"))
    h.update(str(folds).encode("utf-8"))
    h.update(str(ties).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    h.update(str(max_iter).encode("utf-8"))
    h.update(str(tol).encode("utf-8"))
    return h.hexdigest()


# =============================================================================
# K-fold helpers
# =============================================================================

def _kfold_indices(n_samples: int, n_splits: int, random_state: Optional[int] = None):
    """Generate K-fold train/test indices."""
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


def _folds_are_complements(folds, n_samples: int) -> bool:
    """Check if folds are complementary."""
    test_indices = np.concatenate([f[1] for f in folds])
    if len(test_indices) != n_samples:
        return False
    return np.array_equal(np.sort(test_indices), np.arange(n_samples))


# =============================================================================
# Penalty grid generation
# =============================================================================

def _default_coxph_penalty_grid(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_penalties: int = 100,
    penalty_min_ratio: float = 1e-3,
) -> np.ndarray:
    """
    Generate default penalty grid for CoxPHCV.

    Penalty values are log-spaced, similar to alpha grid in RidgeCV.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_features).
    time : ndarray
        Survival times.
    event : ndarray
        Event indicators (1=event, 0=censored).
    n_penalties : int
        Number of penalty values.
    penalty_min_ratio : float
        Minimum penalty as ratio of max penalty.

    Returns
    -------
    penalties : ndarray
        Log-spaced penalty values.
    """
    n_samples, n_features = X.shape
    n_events = int(np.sum(event))

    if n_events == 0:
        # No events - return simple grid
        return np.geomspace(1e-3, 1, n_penalties)

    # Estimate penalty_max from data variance
    # Larger variance -> larger potential penalty
    X_var = np.var(X, axis=0)
    penalty_max = np.max(X_var) * n_events * 0.1

    # Ensure penalty_max is positive and reasonable
    penalty_max = max(penalty_max, 1.0)
    penalty_min = penalty_min_ratio * penalty_max

    penalties = np.geomspace(penalty_max, penalty_min, n_penalties)
    return penalties.astype(np.float64)


# =============================================================================
# Partial likelihood computation for CV evaluation
# =============================================================================

def _compute_partial_likelihood(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    coef: np.ndarray,
    ties: str = 'breslow',
) -> float:
    """
    Compute log partial likelihood for given coefficients.

    This is used for CV evaluation on held-out test folds.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_features).
    time : ndarray
        Survival times.
    event : ndarray
        Event indicators.
    coef : ndarray
        Coefficient values.
    ties : str
        'breslow' or 'efron'.

    Returns
    -------
    log_pl : float
        Log partial likelihood value.
    """
    if coef is None or np.all(coef == 0):
        return 0.0

    risk_scores = X @ coef
    exp_risk = np.exp(risk_scores)

    # Sort by time ascending for efficient suffix-sum risk sets
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    risk_sorted = risk_scores[order]
    exp_risk_sorted = exp_risk[order]

    n = len(time)
    log_pl = 0.0

    if ties == 'breslow':
        # Breslow method - vectorized using cumulative sum
        # Risk set R(t_i) = {j: t_j >= t_i} -> suffix sum after ascending sort
        risk_set_sum = np.cumsum(exp_risk_sorted[::-1])[::-1]

        # Only event rows contribute
        event_mask = event_sorted == 1
        if np.any(event_mask):
            log_pl = np.sum(risk_sorted[event_mask]) - np.sum(np.log(risk_set_sum[event_mask] + 1e-300))

    elif ties == 'efron':
        # Efron method - vectorized by unique failure times
        event_mask = event_sorted == 1
        if not np.any(event_mask):
            return 0.0

        event_idx = np.where(event_mask)[0]
        event_times = time_sorted[event_idx]

        # Group by unique event times
        unique_times, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)

        # Precompute suffix sums for risk sets
        risk_set_sum = np.cumsum(exp_risk_sorted[::-1])[::-1]

        for g, t in enumerate(unique_times):
            d = counts[g]
            if d == 0:
                continue

            # Find first index at this time
            first_idx = np.searchsorted(time_sorted, t, side='left')
            risk_at_t = risk_set_sum[first_idx]

            # Get event rows for this time
            event_rows = event_idx[inv == g]
            sum_risk = np.sum(risk_sorted[event_rows])
            sum_exp_risk = np.sum(exp_risk_sorted[event_rows])

            # Efron correction
            k = np.arange(d, dtype=np.float64) / d
            denom = risk_at_t - k * sum_exp_risk
            log_pl += sum_risk - np.sum(np.log(np.maximum(denom, 1e-300)))

    return float(log_pl)


# =============================================================================
# CV main function
# =============================================================================

def _select_coxph_penalty_cv(
    X,
    time,
    event,
    *,
    penalties=None,
    n_penalties: int = 100,
    penalty_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    ties: str = "breslow",
    device: Union[str, Device] = Device.CPU,
    max_iter: int = 100,
    tol: float = 1e-9,
    return_details: bool = False,
    cache_key: Optional[str] = None,
):
    """
    Select penalty for CoxPH via K-fold cross-validation.

    For each fold:
    1. Split data into train/test
    2. Fit CoxPH on train for each penalty
    3. Evaluate partial likelihood on test

    Returns the penalty with maximum mean partial likelihood.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_features).
    time : ndarray
        Survival times (n_samples,).
    event : ndarray
        Event indicators (n_samples,).
    penalties : ndarray or None
        Penalty values to try. If None, generates grid.
    n_penalties : int
        Number of penalties (if penalties is None).
    penalty_min_ratio : float
        Minimum penalty ratio.
    cv_folds : int
        Number of CV folds.
    cv_splits : list or None
        Pre-computed CV splits.
    random_state : int or None
        Random seed.
    ties : str
        'breslow' or 'efron'.
    device : str or Device
        Computation device.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    return_details : bool
        Whether to return full CV details.
    cache_key : str or None
        Cache key.

    Returns
    -------
    best_penalty : float
    details : dict (if return_details=True)
    """
    device_name = str(device).lower() if not isinstance(device, Device) else device.value
    use_gpu = device_name in (Device.CUDA.value, Device.TORCH.value)

    # Convert to numpy arrays
    X_np = np.asarray(X, dtype=np.float64)
    time_np = np.asarray(time, dtype=np.float64)
    event_np = np.asarray(event, dtype=np.int32)

    n_samples = X_np.shape[0]

    # Generate penalty grid
    if penalties is None:
        penalties = _default_coxph_penalty_grid(X_np, time_np, event_np, n_penalties, penalty_min_ratio)
    else:
        penalties = np.asarray(penalties, dtype=np.float64)
        penalties = penalties[np.isfinite(penalties)]
        penalties = penalties[penalties >= 0]
        if penalties.size == 0:
            penalties = _default_coxph_penalty_grid(X_np, time_np, event_np, n_penalties, penalty_min_ratio)

    n_penalties_actual = len(penalties)

    # Handle degenerate cases
    if n_samples < 4 or cv_folds < 2:
        if not return_details:
            return float(penalties[0])
        return {
            "penalty": float(penalties[0]),
            "penalties": penalties.astype(np.float64),
            "pl_path": np.full((n_penalties_actual, 1), np.nan, dtype=np.float64),
            "mean_pl": np.full(n_penalties_actual, np.nan, dtype=np.float64),
            "best_pl": np.nan,
        }

    # Generate CV folds
    if cv_splits is not None:
        folds = cv_splits
    else:
        folds = _kfold_indices(n_samples, cv_folds, random_state)

    folds_are_complements_flag = _folds_are_complements(folds, n_samples)
    n_folds = len(folds)

    # Cache handling
    cache_key_eff = cache_key
    if cache_key_eff is None and _COXPH_CV_CACHE_MAXSIZE > 0:
        cache_key_eff = _make_coxph_cv_auto_cache_key(
            X_shape=X_np.shape,
            time_shape=time_np.shape,
            event_shape=event_np.shape,
            penalties=penalties,
            n_penalties=n_penalties,
            penalty_min_ratio=penalty_min_ratio,
            folds=folds,
            ties=ties,
            use_gpu=use_gpu,
            max_iter=max_iter,
            tol=tol,
        )

    cached_result = _coxcv_cache_get(cache_key_eff)
    if cached_result is not None:
        if return_details:
            return cached_result["penalty"], cached_result
        return cached_result["penalty"]

    # Storage for partial likelihoods: (n_penalties, n_folds)
    pl_path = np.full((n_penalties_actual, n_folds), np.nan, dtype=np.float64)

    # CV loop
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        time_train, time_test = time_np[train_idx], time_np[test_idx]
        event_train, event_test = event_np[train_idx], event_np[test_idx]

        # Check if fold has events
        n_events_train = int(np.sum(event_train))
        n_events_test = int(np.sum(event_test))

        if n_events_train == 0 or n_events_test == 0:
            continue  # Skip this fold - no events to compute PL

        for penalty_idx, penalty in enumerate(penalties):
            # Fit CoxPH on train
            model = CoxPH(
                ties=ties,
                max_iter=max_iter,
                tol=tol,
                device=device,
                compute_inference=False,
                penalty=penalty,
            )

            try:
                model.fit(X_train, time_train, event_train)

                # Check convergence
                if not model._converged:
                    continue

                # Evaluate partial likelihood on test
                pl_test = _compute_partial_likelihood(
                    X_test, time_test, event_test,
                    model.coef_, ties=ties
                )

                pl_path[penalty_idx, fold_idx] = pl_test

            except Exception:
                # Convergence failure or other error, leave as NaN
                continue

    # Compute mean partial likelihood across folds
    mean_pl = np.nanmean(pl_path, axis=1)

    # Find best penalty (maximum partial likelihood)
    if np.any(np.isfinite(mean_pl)):
        best_idx = np.nanargmax(mean_pl)
        best_penalty = float(penalties[best_idx])
        best_pl = float(mean_pl[best_idx])
    else:
        # No valid CV results - use first penalty
        best_penalty = float(penalties[0])
        best_pl = np.nan

    # Prepare details
    details = {
        "penalty": best_penalty,
        "penalties": penalties.astype(np.float64),
        "pl_path": pl_path.astype(np.float64),
        "mean_pl": mean_pl.astype(np.float64),
        "best_pl": best_pl,
        "n_folds": n_folds,
    }

    # Cache result
    if _COXPH_CV_CACHE_MAXSIZE > 0:
        _coxcv_cache_put(cache_key_eff, details)

    if return_details:
        return best_penalty, details

    return best_penalty


# =============================================================================
# CoxPHCV Class
# =============================================================================

class CoxPHCV(CVEstimatorBase):
    """
    Cross-validated Cox Proportional Hazards regression.

    This class implements K-fold cross-validation to select the optimal
    penalty (L2 regularization) parameter for Cox PH models.

    Parameters
    ----------
    penalties : array-like or None
        Penalty values to try. If None, generates n_penalties values.
    n_penalties : int, default=100
        Number of penalty values (if penalties is None).
    penalty_min_ratio : float, default=1e-3
        Minimum penalty as ratio of max penalty.
    cv : int, default=5
        Number of CV folds.
    ties : str, default='breslow'
        Method for handling ties: 'breslow' or 'efron'.
    tol : float, default=1e-9
        Convergence tolerance.
    max_iter : int, default=100
        Maximum iterations.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    compute_inference : bool, default=True
        Whether to compute standard errors after fitting.
    cov_type : str, default='nonrobust'
        Covariance estimator.
    gpu_memory_cleanup : bool, default=False
        Whether to free GPU memory after fitting.
    random_state : int or None
        Random seed for CV splits.

    Attributes
    ----------
    penalty_ : float
        Selected penalty value.
    penalties_ : ndarray
        All penalty values tested.
    cv_results_ : dict
        CV results including partial_likelihood_path.
    best_score_ : float
        Best (maximum) partial likelihood across CV folds.
    coef_ : ndarray
        Coefficients of the final model.
    hazard_ratios_ : ndarray
        exp(coef) = hazard ratios.
    estimator_ : CoxPH
        The fitted CoxPH with selected penalty.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.survival import CoxPHCV
    >>> X = np.random.randn(1000, 20)
    >>> time = np.random.exponential(scale=100, size=1000)
    >>> event = np.random.binomial(1, 0.7, size=1000)
    >>> model = CoxPHCV(cv=5, device='cuda')
    >>> model.fit(X, time, event)
    >>> print(f"Selected penalty: {model.penalty_:.4f}")
    >>> print(f"Best CV score: {model.best_score_:.4f}")
    """

    def __init__(
        self,
        penalties=None,
        n_penalties: int = 100,
        penalty_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        ties: str = "breslow",
        tol: float = 1e-9,
        max_iter: int = 100,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.penalties = penalties
        self.n_penalties = int(n_penalties)
        self.penalty_min_ratio = float(penalty_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.ties = str(ties)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.compute_inference = bool(compute_inference)
        self.cov_type = str(cov_type)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

        # Output attributes (initialized to None)
        self.penalty_ = None
        self.penalties_ = None
        self.cv_results_ = None
        self.best_score_ = None
        self.coef_ = None
        self.hazard_ratios_ = None
        self.estimator_ = None

    def _fit_cv(self, X, time, event, entry=None, cluster=None):
        """
        Fit CoxPH with K-fold cross-validation.

        Parameters
        ----------
        X : array-like
            Design matrix.
        time : array-like
            Survival times.
        event : array-like
            Event indicators.
        entry : array-like, optional
            Entry times (not yet supported).
        cluster : array-like, optional
            Cluster ids (not yet supported).

        Returns
        -------
        self
        """
        if entry is not None:
            raise NotImplementedError("Delayed entry is not yet supported in CoxPHCV.")
        if cluster is not None:
            raise NotImplementedError("Cluster-robust covariance is not yet supported in CoxPHCV.")

        device_name = self._get_compute_device().value
        use_gpu = device_name == Device.CUDA.value

        # Normalize penalties to list
        if isinstance(self.penalties, (list, tuple, np.ndarray)):
            penalties = np.asarray(self.penalties, dtype=np.float64)
        else:
            penalties = None

        # Perform CV to find best penalty
        best_penalty, details = _select_coxph_penalty_cv(
            X, time, event,
            penalties=penalties,
            n_penalties=self.n_penalties,
            penalty_min_ratio=self.penalty_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            ties=self.ties,
            device=device_name,
            max_iter=self.max_iter,
            tol=self.tol,
            return_details=True,
        )

        # Store CV results
        self.penalty_ = float(best_penalty)
        self.penalties_ = np.asarray(details["penalties"], dtype=np.float64)

        pl_path = np.asarray(details["pl_path"], dtype=np.float64)
        mean_pl = np.asarray(details["mean_pl"], dtype=np.float64)

        self.cv_results_ = {
            "pl_path": pl_path,
            "mean_pl": mean_pl,
        }
        self.best_score_ = float(details["best_pl"])

        # Fit final model on full data with best penalty
        final_model = CoxPH(
            ties=self.ties,
            tol=self.tol,
            max_iter=self.max_iter,
            device=self.device,
            n_jobs=self.n_jobs,
            compute_inference=self.compute_inference,
            cov_type=self.cov_type,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
            penalty=self.penalty_,
        )
        final_model.fit(X, time, event)

        self.estimator_ = final_model
        self.coef_ = final_model.coef_.copy()
        self.hazard_ratios_ = final_model.hazard_ratios_.copy()

        return self

    def fit(self, X, time, event, entry=None, cluster=None):
        """
        Fit CoxPH model with cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like of shape (n_samples,)
            Time to event or censoring.
        event : array-like of shape (n_samples,)
            Event indicator (1 = event, 0 = censored).
        entry : array-like, optional
            Entry time for delayed entry (not yet supported).
        cluster : array-like, optional
            Cluster ids (not yet supported).

        Returns
        -------
        self : CoxPHCV
        """
        return self._fit_cv(X, time, event, entry=entry, cluster=cluster)

    def predict(self, X):
        """
        Predict risk scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        risk_scores : ndarray
            Risk scores (linear predictor).
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_

    def score(self, X, time, event):
        """
        Return C-index (concordance index).

        Parameters
        ----------
        X : array-like
            Covariate matrix.
        time : array-like
            Survival times.
        event : array-like
            Event indicators.

        Returns
        -------
        c_index : float
            C-index (0.5 = random, 1.0 = perfect).
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        time_arr = np.asarray(time, dtype=np.float64)
        event_arr = np.asarray(event, dtype=np.int32)

        # Compute risk scores
        risk_scores = X_arr @ self.coef_

        n = len(time_arr)
        event_mask = (event_arr == 1)

        if not np.any(event_mask):
            return 0.5

        # Use chunked vectorized approach for memory efficiency
        # Similar to _compute_cindex in _cox.py
        event_idx = np.where(event_mask)[0]
        n_events = len(event_idx)

        if n_events == 0:
            return 0.5

        concordant = np.int64(0)
        permissible = np.int64(0)
        tied_risk = np.int64(0)

        # Chunk size: keep each (chunk × n) bool matrix <= 128 MB
        chunk_size = max(1, min(n_events, int(128e6 / max(n, 1))))

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            idx_chunk = event_idx[start:end]

            time_i = time_arr[idx_chunk, np.newaxis]
            risk_i = risk_scores[idx_chunk, np.newaxis]
            time_j = time_arr[np.newaxis, :]
            risk_j = risk_scores[np.newaxis, :]
            event_j = event_arr[np.newaxis, :]

            # Permissible pairs: earlier time OR same time with j censored
            perm = (time_i < time_j) | ((time_i == time_j) & (event_j == 0))

            # Exclude self-comparisons
            chunk_indices = np.arange(end - start, dtype=np.int64)
            perm[chunk_indices, idx_chunk] = False

            concordant += int(np.sum(perm & (risk_i > risk_j)))
            tied_risk += int(np.sum(perm & (risk_i == risk_j)))
            permissible += int(np.sum(perm))

        if permissible == 0:
            return 0.5

        return (concordant + 0.5 * tied_risk) / permissible

    def summary(self):
        """Return summary of the fitted model."""
        if self.estimator_ is None:
            raise RuntimeError("No fitted estimator available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(f"{self.estimator_.__class__.__name__} does not implement summary().")
        return self.estimator_.summary()
