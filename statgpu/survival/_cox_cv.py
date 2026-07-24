"""
CoxPHCV: Cross-validated Cox Proportional Hazards regression.

Implements K-fold cross-validation to select the optimal penalty (L2 regularization)
parameter for Cox PH models.
"""

from typing import Optional, Union, Tuple, Dict, Any, List
from collections import OrderedDict
import hashlib
import os
import numpy as np

from statgpu._config import Device
from statgpu.backends import _get_torch_device_str
from statgpu.cross_validation._base import CVEstimatorBase
from statgpu.survival._cox import CoxPH


# =============================================================================
# CV Cache
# =============================================================================

_COXPH_CV_CACHE_MAXSIZE = int(64)
_COXPH_CV_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _env_flag(name: str, default: bool = False) -> bool:
    """Safely parse boolean env var."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_int(
    name: str,
    default: int,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Safely parse integer env var with optional bounds."""
    raw = os.environ.get(name)
    try:
        val = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        val = int(default)
    if min_value is not None:
        val = max(min_value, val)
    if max_value is not None:
        val = min(max_value, val)
    return val


def _env_float(name: str, default: float, *, min_value: Optional[float] = None) -> float:
    """Safely parse float env var with optional lower bound."""
    raw = os.environ.get(name)
    try:
        val = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        val = float(default)
    if min_value is not None:
        val = max(min_value, val)
    return val


def _hash_optional_array(h: "hashlib._blake2.blake2b", tag: str, arr: Optional[np.ndarray]) -> None:
    """Hash optional array content for cache-key disambiguation."""
    if arr is None:
        h.update(f"{tag}:none".encode("utf-8"))
        return
    h.update(tag.encode("utf-8"))
    arr_np = np.asarray(arr)
    h.update(np.asarray(arr_np.shape, dtype=np.int64).tobytes())
    h.update(str(arr_np.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr_np).tobytes())


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


def _sample_hash(h, arr, max_rows=50):
    """Hash a sampled subset of an array for cache key generation."""
    arr_np = np.asarray(arr, dtype=np.float64).ravel()
    n = arr_np.shape[0]
    if n <= max_rows:
        h.update(arr_np.tobytes())
    else:
        # Sample first, middle, and last rows
        indices = np.concatenate([np.arange(max_rows//2), np.arange(n-max_rows//2, n)])
        h.update(arr_np[indices].tobytes())


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
    fit_device: str,
    cv_cuda_torch_bridge: bool,
    entry: Optional[np.ndarray],
    cluster: Optional[np.ndarray],
    two_stage_enabled: bool,
    halving_enabled: bool,
    coarse_n: int,
    window: int,
    halving_topk: int,
    fast_iter: int,
    fast_tol: float,
    max_iter: int,
    tol: float,
    X_data=None,
    time_data=None,
    event_data=None,
) -> str:
    """
    Generate automatic cache key for CoxPH CV.

    Includes structural inputs (shapes/grid/folds), execution-path settings
    (fit device/bridge/two-stage/halving), and optional delayed-entry or
    clustering arrays to avoid stale collisions across distinct CV runs.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(X_shape, dtype=np.int64).tobytes())
    h.update(np.asarray(time_shape, dtype=np.int64).tobytes())
    h.update(np.asarray(event_shape, dtype=np.int64).tobytes())
    # Include sampled data content to avoid collisions across datasets with same shape
    if X_data is not None:
        _sample_hash(h, X_data, max_rows=50)
    if time_data is not None:
        _sample_hash(h, time_data, max_rows=50)
    if event_data is not None:
        _sample_hash(h, event_data, max_rows=50)
    if penalties is not None:
        h.update(np.asarray(penalties, dtype=np.float64).tobytes())
    h.update(str(n_penalties).encode("utf-8"))
    h.update(str(penalty_min_ratio).encode("utf-8"))
    h.update(str(folds).encode("utf-8"))
    h.update(str(ties).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    h.update(str(fit_device).encode("utf-8"))
    h.update(str(cv_cuda_torch_bridge).encode("utf-8"))
    _hash_optional_array(h, "entry", entry)
    _hash_optional_array(h, "cluster", cluster)
    h.update(str(two_stage_enabled).encode("utf-8"))
    h.update(str(halving_enabled).encode("utf-8"))
    h.update(str(coarse_n).encode("utf-8"))
    h.update(str(window).encode("utf-8"))
    h.update(str(halving_topk).encode("utf-8"))
    h.update(str(fast_iter).encode("utf-8"))
    h.update(str(fast_tol).encode("utf-8"))
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
    entry: Optional[np.ndarray] = None,
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
    entry : ndarray or None
        Delayed-entry times (left truncation). If None, assumes entry=0 for all samples.
    ties : str
        'breslow' or 'efron'.

    Returns
    -------
    log_pl : float
        Log partial likelihood value.
    """
    n = len(time)
    if coef is None or np.all(coef == 0):
        # Null model: compute log partial likelihood at beta=0
        # L(0) = sum_events[-log(|R(t_i)|)] where |R(t_i)| = n - i (sorted)
        order = np.argsort(time)
        event_sorted = event[order]
        # Risk set size at sorted position i is (n - i)
        risk_set_sizes = n - np.arange(n)
        event_mask = event_sorted.astype(bool)
        null_ll = -np.sum(np.log(risk_set_sizes[event_mask].astype(float)))
        return null_ll

    risk_scores = X @ coef
    exp_risk = np.exp(risk_scores)

    # Fast path (no delayed-entry): keep vectorized suffix-sum implementation.
    if entry is None:
        order = np.argsort(time)
        time_sorted = time[order]
        event_sorted = event[order]
        risk_sorted = risk_scores[order]
        exp_risk_sorted = exp_risk[order]
        log_pl = 0.0
        if ties == 'breslow':
            risk_set_sum = np.cumsum(exp_risk_sorted[::-1])[::-1]
            event_mask = event_sorted == 1
            if np.any(event_mask):
                log_pl = np.sum(risk_sorted[event_mask]) - np.sum(np.log(risk_set_sum[event_mask] + 1e-300))
        elif ties == 'efron':
            event_mask = event_sorted == 1
            if not np.any(event_mask):
                return 0.0
            event_idx = np.where(event_mask)[0]
            event_times = time_sorted[event_idx]
            unique_times, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
            risk_set_sum = np.cumsum(exp_risk_sorted[::-1])[::-1]
            for g, t in enumerate(unique_times):
                d = counts[g]
                if d == 0:
                    continue
                first_idx = np.searchsorted(time_sorted, t, side='left')
                risk_at_t = risk_set_sum[first_idx]
                event_rows = event_idx[inv == g]
                sum_risk = np.sum(risk_sorted[event_rows])
                sum_exp_risk = np.sum(exp_risk_sorted[event_rows])
                k = np.arange(d, dtype=np.float64) / d
                denom = risk_at_t - k * sum_exp_risk
                log_pl += sum_risk - np.sum(np.log(np.maximum(denom, 1e-300)))
        return float(log_pl)

    entry_arr = np.asarray(entry, dtype=np.float64)
    # Delayed-entry path
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]
    entry_sorted = entry_arr[order]
    risk_sorted = risk_scores[order]
    exp_risk_sorted = exp_risk[order]

    log_pl = 0.0

    # With delayed entry, risk set is:
    #   R(t) = {j: entry_j <= t <= time_j}
    # We compute denominators directly per unique event time for correctness.
    event_mask = event_sorted == 1
    if not np.any(event_mask):
        return 0.0
    event_idx = np.where(event_mask)[0]
    event_times = time_sorted[event_idx]

    if ties == 'breslow':
        unique_times, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
        for g, t in enumerate(unique_times):
            d = counts[g]
            if d == 0:
                continue
            events_at_t = event_idx[inv == g]
            risk_mask = (entry_sorted <= t) & (time_sorted >= t)
            risk_at_t = np.sum(exp_risk_sorted[risk_mask])
            sum_risk = np.sum(risk_sorted[events_at_t])
            log_pl += sum_risk - d * np.log(max(risk_at_t, 1e-300))

    elif ties == 'efron':
        # Efron method by unique failure times
        unique_times, inv, counts = np.unique(event_times, return_inverse=True, return_counts=True)
        for g, t in enumerate(unique_times):
            d = counts[g]
            if d == 0:
                continue
            event_rows = event_idx[inv == g]
            risk_mask = (entry_sorted <= t) & (time_sorted >= t)
            risk_at_t = np.sum(exp_risk_sorted[risk_mask])
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
    entry=None,
    cluster=None,
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
    entry : ndarray or None
        Delayed-entry times.
    cluster : ndarray or None
        Cluster ids (used in model fitting; scoring remains partial likelihood).
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
    # Optional CV bridge for CUDA: many medium-size CV workloads are faster with
    # torch backend while preserving the same CoxPHCV public API.
    cv_cuda_torch_bridge = os.environ.get(
        "STATGPU_COXPHCV_CUDA_TORCH_BRIDGE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")

    # Convert to numpy arrays
    X_np = np.asarray(X, dtype=np.float64)
    time_np = np.asarray(time, dtype=np.float64)
    event_np = np.asarray(event, dtype=np.int32)
    entry_np = None if entry is None else np.asarray(entry, dtype=np.float64)
    cluster_np = None if cluster is None else np.asarray(cluster)

    n_samples = X_np.shape[0]
    n_features = X_np.shape[1]
    fit_device = device_name
    if (
        cv_cuda_torch_bridge
        and device_name == Device.CUDA.value
        and n_samples >= 1500
        and n_features >= 40
    ):
        fit_device = Device.TORCH.value

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
    if (
        entry_np is not None
        and fit_device == Device.CPU.value
        and np.any(penalties > 0.0)
    ):
        raise NotImplementedError(
            'CPU delayed-entry CoxPHCV cannot evaluate nonzero penalties; '
            'use device=cuda/device=torch or pass penalties=[0.0].'
        )

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

    # Keep exhaustive full-grid CV as the default behavior. Two-stage is opt-in.
    two_stage_enabled = (
        _env_flag("STATGPU_COXPHCV_TWO_STAGE", False)  # default=False: opt-in
        and device_name == Device.CUDA.value
        and n_penalties_actual >= 8
    )
    halving_enabled = (
        _env_flag("STATGPU_COXPHCV_SUCCESSIVE_HALVING", False)
        and device_name == Device.CUDA.value
        and n_penalties_actual >= 8
    )
    coarse_n = _env_int(
        "STATGPU_COXPHCV_TWO_STAGE_COARSE",
        6,
        min_value=3,
        max_value=n_penalties_actual,
    )
    window = _env_int("STATGPU_COXPHCV_TWO_STAGE_WINDOW", 2, min_value=1)
    halving_topk = _env_int(
        "STATGPU_COXPHCV_HALVING_TOPK",
        3,
        min_value=1,
        max_value=n_penalties_actual,
    )
    fast_iter = _env_int(
        "STATGPU_COXPHCV_HALVING_FAST_ITER",
        30,
        min_value=5,
        max_value=max_iter,
    )
    fast_tol = _env_float("STATGPU_COXPHCV_HALVING_FAST_TOL", 1e-6, min_value=tol)

    # Cache handling
    cache_key_eff = cache_key
    if cache_key_eff is None and _COXPH_CV_CACHE_MAXSIZE > 0:
        cache_key_eff = _make_coxph_cv_auto_cache_key(
            X_shape=X_np.shape,
            time_shape=time_np.shape,
            event_shape=event_np.shape,
            X_data=X_np,
            time_data=time_np,
            event_data=event_np,
            penalties=penalties,
            n_penalties=n_penalties,
            penalty_min_ratio=penalty_min_ratio,
            folds=folds,
            ties=ties,
            use_gpu=use_gpu,
            fit_device=fit_device,
            cv_cuda_torch_bridge=cv_cuda_torch_bridge,
            entry=entry_np,
            cluster=cluster_np,
            two_stage_enabled=two_stage_enabled,
            halving_enabled=halving_enabled,
            coarse_n=coarse_n,
            window=window,
            halving_topk=halving_topk,
            fast_iter=fast_iter,
            fast_tol=fast_tol,
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

    def _evaluate_penalty_indices(
        penalty_indices: np.ndarray,
        *,
        fit_max_iter: int,
        fit_tol: float,
    ) -> None:
        if penalty_indices.size == 0:
            return
        penalty_indices = np.unique(np.asarray(penalty_indices, dtype=np.int64))
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            time_train, time_test = time_np[train_idx], time_np[test_idx]
            event_train, event_test = event_np[train_idx], event_np[test_idx]
            entry_train = None if entry_np is None else entry_np[train_idx]
            entry_test = None if entry_np is None else entry_np[test_idx]
            cluster_train = None if cluster_np is None else cluster_np[train_idx]
            X_fit = X_train
            time_fit = time_train
            event_fit = event_train
            entry_fit = entry_train
            cluster_fit = cluster_train

            # Reduce repeated host->device conversions by preparing one fold
            # tensor/array per backend and reusing it across penalties.
            if fit_device == Device.CUDA.value:
                try:
                    import cupy as cp
                    X_fit = cp.asarray(X_train, dtype=cp.float64)
                    time_fit = cp.asarray(time_train, dtype=cp.float64)
                    event_fit = cp.asarray(event_train, dtype=cp.int32)
                    entry_fit = None if entry_train is None else cp.asarray(entry_train, dtype=cp.float64)
                    cluster_fit = None if cluster_train is None else cp.asarray(cluster_train, dtype=cp.int64)
                except Exception:
                    X_fit = X_train
                    time_fit = time_train
                    event_fit = event_train
                    entry_fit = entry_train
                    cluster_fit = cluster_train
            elif fit_device == Device.TORCH.value:
                try:
                    import torch
                    torch_device = _get_torch_device_str()
                    X_fit = torch.as_tensor(X_train, dtype=torch.float64, device=torch_device)
                    time_fit = torch.as_tensor(time_train, dtype=torch.float64, device=torch_device)
                    event_fit = torch.as_tensor(event_train, dtype=torch.int32, device=torch_device)
                    entry_fit = None if entry_train is None else torch.as_tensor(
                        entry_train, dtype=torch.float64, device=torch_device
                    )
                    cluster_fit = None if cluster_train is None else torch.as_tensor(
                        cluster_train, dtype=torch.int64, device=torch_device
                    )
                except Exception:
                    X_fit = X_train
                    time_fit = time_train
                    event_fit = event_train
                    entry_fit = entry_train
                    cluster_fit = cluster_train

            n_events_train = int(np.sum(event_train))
            n_events_test = int(np.sum(event_test))
            if n_events_train == 0 or n_events_test == 0:
                continue

            prev_coef = None
            for penalty_idx in penalty_indices:
                if np.isfinite(pl_path[penalty_idx, fold_idx]):
                    continue
                penalty = penalties[penalty_idx]
                model = CoxPH(
                    ties=ties,
                    max_iter=fit_max_iter,
                    tol=fit_tol,
                    device=fit_device,
                    compute_inference=False,
                    penalty=penalty,
                )
                try:
                    model.fit(
                        X_fit,
                        time_fit,
                        event_fit,
                        entry=entry_fit,
                        cluster=cluster_fit,
                        init_coef=prev_coef,
                    )
                    if not model._converged:
                        continue
                    prev_coef = np.asarray(model.coef_, dtype=np.float64).copy()
                    pl_test = _compute_partial_likelihood(
                        X_test, time_test, event_test, model.coef_, entry=entry_test, ties=ties
                    )
                    pl_path[penalty_idx, fold_idx] = pl_test
                except Exception:
                    continue

    if two_stage_enabled:
        stage1_idx = np.unique(
            np.linspace(0, n_penalties_actual - 1, num=coarse_n, dtype=np.int64)
        )
        _evaluate_penalty_indices(
            stage1_idx,
            fit_max_iter=(fast_iter if halving_enabled else max_iter),
            fit_tol=(fast_tol if halving_enabled else tol),
        )
        stage1_mean = np.nanmean(pl_path[stage1_idx, :], axis=1)
        if np.any(np.isfinite(stage1_mean)):
            stage1_best = int(stage1_idx[int(np.nanargmax(stage1_mean))])
        else:
            stage1_best = int(stage1_idx[len(stage1_idx) // 2])
        lo = max(0, stage1_best - window)
        hi = min(n_penalties_actual - 1, stage1_best + window)
        stage2_idx = np.arange(lo, hi + 1, dtype=np.int64)
        _evaluate_penalty_indices(
            stage2_idx,
            fit_max_iter=(fast_iter if halving_enabled else max_iter),
            fit_tol=(fast_tol if halving_enabled else tol),
        )
        if halving_enabled:
            stage2_mean = np.full(stage2_idx.shape[0], np.nan, dtype=np.float64)
            stage2_valid = np.any(np.isfinite(pl_path[stage2_idx, :]), axis=1)
            if np.any(stage2_valid):
                stage2_mean[stage2_valid] = np.nanmean(pl_path[stage2_idx[stage2_valid], :], axis=1)
                order = np.argsort(np.nan_to_num(stage2_mean, nan=-np.inf))[::-1]
                top_idx = stage2_idx[order[: min(halving_topk, len(stage2_idx))]]
                # Re-evaluate top candidates with full precision and overwrite.
                pl_path[top_idx, :] = np.nan
                _evaluate_penalty_indices(top_idx, fit_max_iter=max_iter, fit_tol=tol)
    else:
        full_idx = np.arange(n_penalties_actual, dtype=np.int64)
        if halving_enabled:
            _evaluate_penalty_indices(full_idx, fit_max_iter=fast_iter, fit_tol=fast_tol)
            full_mean = np.full(full_idx.shape[0], np.nan, dtype=np.float64)
            full_valid = np.any(np.isfinite(pl_path[full_idx, :]), axis=1)
            if np.any(full_valid):
                full_mean[full_valid] = np.nanmean(pl_path[full_idx[full_valid], :], axis=1)
                order = np.argsort(np.nan_to_num(full_mean, nan=-np.inf))[::-1]
                top_idx = full_idx[order[:halving_topk]]
                pl_path[top_idx, :] = np.nan
                _evaluate_penalty_indices(top_idx, fit_max_iter=max_iter, fit_tol=tol)
        else:
            _evaluate_penalty_indices(full_idx, fit_max_iter=max_iter, fit_tol=tol)

    # Safety fallback: if no penalty has any finite fold score, evaluate full grid once.
    has_any_valid = np.any(np.isfinite(pl_path), axis=1)
    if not np.any(has_any_valid):
        _evaluate_penalty_indices(
            np.arange(n_penalties_actual, dtype=np.int64),
            fit_max_iter=max_iter,
            fit_tol=tol,
        )

    # Compute mean partial likelihood across folds
    mean_pl = np.full(n_penalties_actual, np.nan, dtype=np.float64)
    valid_rows = np.any(np.isfinite(pl_path), axis=1)
    if np.any(valid_rows):
        mean_pl[valid_rows] = np.nanmean(pl_path[valid_rows], axis=1)

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
    inference_mode : {'strict', 'approx'}, default='strict'
        Robust-inference policy forwarded to the final CoxPH estimator.
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
        inference_mode: str = "strict",
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
        self.inference_mode = str(inference_mode).lower()
        if self.inference_mode not in ('strict', 'approx'):
            raise ValueError('inference_mode must be strict or approx')
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

        # Output attributes (initialized to None)
        self.penalty_ = None
        self.penalties_ = None
        self.cv_results_ = None
        self.best_score_ = None
        self.coef_ = None
        self.hazard_ratios_ = None
        self.estimator_ = None
        self.converged_ = False
        self.termination_reason_ = None
        self.n_iter_ = 0
        self.final_kkt_inf_ = None
        self.final_kkt_normalized_ = None
        self.inference_method_ = None
        self.inference_backend_ = None
        self.inference_approximate_ = False
        self.inference_fallback_reason_ = None
        self.full_host_transfer_performed_ = False

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        """Best-effort Torch CUDA cache cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    def __del__(self):
        try:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()
        except Exception:
            pass

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
            Entry times (delayed entry).
        cluster : array-like, optional
            Cluster ids.

        Returns
        -------
        self
        """
        device_name = self._get_compute_device().value
        X_shape = getattr(X, 'shape', None)
        if X_shape is None or len(X_shape) != 2:
            raise ValueError('X must be a two-dimensional array')
        n_samples, n_features = (int(X_shape[0]), int(X_shape[1]))
        if entry is not None and self.compute_inference and self.cov_type.lower() != 'nonrobust':
            raise NotImplementedError(
                'Robust/cluster covariance with delayed entry is not implemented. '
                'Use cov_type=nonrobust or compute_inference=False when entry is provided.'
            )
        cv_cuda_torch_bridge = os.environ.get(
            "STATGPU_COXPHCV_CUDA_TORCH_BRIDGE", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        fit_device_name = device_name
        if (
            cv_cuda_torch_bridge
            and device_name == Device.CUDA.value
            and n_samples >= 1500
            and n_features >= 40
        ):
            fit_device_name = Device.TORCH.value

        # Normalize penalties to list
        if isinstance(self.penalties, (list, tuple, np.ndarray)):
            penalties = np.asarray(self.penalties, dtype=np.float64)
        else:
            penalties = None

        # Perform CV to find best penalty
        best_penalty, details = _select_coxph_penalty_cv(
            X, time, event,
            entry=entry,
            cluster=cluster,
            penalties=penalties,
            n_penalties=self.n_penalties,
            penalty_min_ratio=self.penalty_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            ties=self.ties,
            device=fit_device_name,
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
            device=fit_device_name,
            n_jobs=self.n_jobs,
            compute_inference=self.compute_inference,
            cov_type=self.cov_type,
            inference_mode=self.inference_mode,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
            penalty=self.penalty_,
        )
        final_model.fit(X, time, event, entry=entry, cluster=cluster)

        self.estimator_ = final_model
        self.coef_ = final_model.coef_.copy()
        self.hazard_ratios_ = final_model.hazard_ratios_.copy()
        for attribute in (
            'converged_', 'termination_reason_', 'n_iter_', 'final_kkt_inf_',
            'final_kkt_normalized_', 'inference_method_', 'inference_backend_',
            'inference_approximate_', 'inference_fallback_reason_',
            'full_host_transfer_performed_',
        ):
            setattr(self, attribute, getattr(final_model, attribute))
        self._cleanup_cuda_memory()
        self._cleanup_torch_memory()

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
            Entry time for delayed entry.
        cluster : array-like, optional
            Cluster ids.

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
        risk_scores : backend-native array
            Risk scores (linear predictor) on the estimator backend.
        """
        if self.estimator_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.estimator_.predict_risk_score(X)

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
        if self.estimator_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.estimator_.score(X, time, event)

    def summary(self):
        """Return summary of the fitted model."""
        if self.estimator_ is None:
            raise RuntimeError("No fitted estimator available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(f"{self.estimator_.__class__.__name__} does not implement summary().")
        return self.estimator_.summary()
