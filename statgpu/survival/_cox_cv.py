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
from statgpu.backends import _to_numpy
from statgpu.cross_validation._base import CVEstimatorBase
from statgpu.survival._cox import CoxPH
from statgpu.survival._risk_sets import cox_counting_process_objective


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
    if arr_np.dtype.hasobject or arr_np.dtype.kind in {"U", "S"}:
        h.update(repr(arr_np.tolist()).encode("utf-8"))
    else:
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
    """Hash complete numeric array content for cache-key correctness.

    CV is much more expensive than hashing its inputs.  Sampling only the
    first/last rows allowed mutations in the middle of a same-shaped dataset
    to reuse a stale penalty path and diagnostics.
    """
    arr_np = np.asarray(arr, dtype=np.float64).ravel()
    h.update(np.asarray(arr_np.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(arr_np).tobytes())


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
    entry: Optional[np.ndarray],
    cluster: Optional[np.ndarray],
    strata: Optional[np.ndarray],
    subject_id: Optional[np.ndarray],
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
    (fit device/two-stage/halving), and optional delayed-entry or
    clustering, stratification, or subject arrays to avoid stale collisions
    across distinct CV runs.
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
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        h.update(np.asarray([fold_idx], dtype=np.int64).tobytes())
        for tag, indices in (("train", train_idx), ("test", test_idx)):
            index_arr = np.ascontiguousarray(indices, dtype=np.int64)
            h.update(tag.encode("utf-8"))
            h.update(np.asarray(index_arr.shape, dtype=np.int64).tobytes())
            h.update(index_arr.tobytes())
    h.update(str(ties).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    h.update(str(fit_device).encode("utf-8"))
    _hash_optional_array(h, "entry", entry)
    _hash_optional_array(h, "cluster", cluster)
    _hash_optional_array(h, "strata", strata)
    _hash_optional_array(h, "subject_id", subject_id)
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


def _group_kfold_indices(
    subject_id: np.ndarray,
    n_splits: int,
    random_state: Optional[int] = None,
):
    """Generate folds without placing one subject in train and test."""
    subject_arr = np.asarray(subject_id).reshape(-1)
    _, subject_codes = np.unique(subject_arr, return_inverse=True)
    n_subjects = int(np.max(subject_codes)) + 1 if subject_codes.size else 0
    if n_splits < 2:
        raise ValueError("cv_folds must be at least 2")
    if n_splits > n_subjects:
        raise ValueError(
            "cv_folds cannot exceed the number of unique subject_id values"
        )

    rng = np.random.RandomState(random_state)
    shuffled_subjects = rng.permutation(n_subjects)
    subject_sizes = np.bincount(subject_codes, minlength=n_subjects)
    # Greedily balance row counts while retaining randomized tie-breaking.
    ordered_subjects = shuffled_subjects[
        np.argsort(-subject_sizes[shuffled_subjects], kind="stable")
    ]
    fold_subjects = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    for subject_code in ordered_subjects:
        fold_idx = int(np.argmin(fold_sizes))
        fold_subjects[fold_idx].append(int(subject_code))
        fold_sizes[fold_idx] += int(subject_sizes[subject_code])

    indices = np.arange(subject_arr.shape[0], dtype=np.int64)
    folds = []
    for test_subjects in fold_subjects:
        test_mask = np.isin(subject_codes, test_subjects)
        folds.append((indices[~test_mask], indices[test_mask]))
    return folds


def _folds_are_complements(folds, n_samples: int) -> bool:
    """Check that test folds partition rows and train is each complement."""
    all_indices = np.arange(n_samples, dtype=np.int64)
    for train_idx, test_idx in folds:
        expected_train = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        if not np.array_equal(np.sort(train_idx), expected_train):
            return False
    test_indices = np.concatenate([f[1] for f in folds])
    if len(test_indices) != n_samples:
        return False
    return np.array_equal(np.sort(test_indices), np.arange(n_samples))


def _unpack_survival_target(time, event, *, entry=None, start=None):
    """Accept either separate arrays or sklearn-style two/three-column y."""
    if event is not None:
        return time, event, entry, start

    y = np.asarray(_to_numpy(time), dtype=np.float64)
    if y.ndim != 2 or y.shape[1] not in (2, 3):
        raise ValueError(
            "When event is omitted, y must have columns [time, event] or "
            "[start, stop, event]."
        )
    if y.shape[1] == 2:
        return y[:, 0], y[:, 1], entry, start
    if entry is not None or start is not None:
        raise ValueError(
            "Do not pass entry/start separately when y already has "
            "[start, stop, event] columns."
        )
    return y[:, 1], y[:, 2], None, y[:, 0]


def _validate_cv_splits(folds, n_samples: int) -> None:
    """Reject malformed, overlapping, or out-of-bounds train/test folds."""
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        for name, values in (("train", train_idx), ("test", test_idx)):
            if values.ndim != 1:
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices must be 1-dimensional"
                )
            if values.size == 0:
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices must not be empty"
                )
            if np.unique(values).size != values.size:
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices contain duplicates"
                )
            if np.any(values < 0) or np.any(values >= n_samples):
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices are out of bounds"
                )
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(
                f"cv_splits fold {fold_idx} train and test indices must be disjoint"
            )


def _coerce_cv_indices(values, *, fold_idx: int, name: str) -> np.ndarray:
    """Validate custom fold indices before converting them to ``int64``."""
    if isinstance(values, (list, tuple)):
        object_values = np.asarray(values, dtype=object)
        if object_values.ndim == 1 and any(
            isinstance(value, (bool, np.bool_)) for value in object_values
        ):
            raise ValueError(
                f"cv_splits fold {fold_idx} {name} indices must contain integers, "
                "not booleans"
            )
    try:
        values_np = np.asarray(_to_numpy(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cv_splits fold {fold_idx} {name} indices must contain integers"
        ) from exc

    if values_np.ndim != 1:
        raise ValueError(
            f"cv_splits fold {fold_idx} {name} indices must be 1-dimensional"
        )

    kind = values_np.dtype.kind
    if kind == "b":
        raise ValueError(
            f"cv_splits fold {fold_idx} {name} indices must contain integers, "
            "not booleans"
        )
    if kind in {"i", "u"}:
        if kind == "u" and np.any(values_np > np.iinfo(np.int64).max):
            raise ValueError(
                f"cv_splits fold {fold_idx} {name} indices exceed the int64 range"
            )
        return values_np.astype(np.int64, copy=False)
    if kind == "f":
        valid = (
            np.all(np.isfinite(values_np))
            and np.all(values_np == np.floor(values_np))
            and np.all(values_np >= -(2**63))
            and np.all(values_np < 2**63)
        )
        if valid:
            return values_np.astype(np.int64)
        raise ValueError(
            f"cv_splits fold {fold_idx} {name} indices must contain integers"
        )
    if kind == "O":
        int64_info = np.iinfo(np.int64)
        normalized = []
        for value in values_np.tolist():
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices must contain "
                    "integers, not booleans"
                )
            if isinstance(value, (int, np.integer)):
                integer = int(value)
            elif (
                isinstance(value, (float, np.floating))
                and np.isfinite(value)
                and float(value).is_integer()
            ):
                integer = int(value)
            else:
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices must contain integers"
                )
            if integer < int64_info.min or integer > int64_info.max:
                raise ValueError(
                    f"cv_splits fold {fold_idx} {name} indices exceed the int64 range"
                )
            normalized.append(integer)
        return np.asarray(normalized, dtype=np.int64)

    raise ValueError(f"cv_splits fold {fold_idx} {name} indices must contain integers")


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
    strata: Optional[np.ndarray] = None,
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
        Delayed-entry/counting-process start times. Rows are at risk on
        ``(entry, time]``. If None, assumes zero for all samples.
    strata : ndarray or None
        Stratum labels. Each stratum contributes an independent risk set.
    ties : str
        'breslow', 'efron', or 'exact'.

    Returns
    -------
    log_pl : float
        Log partial likelihood value.
    """
    ties = str(ties).lower()
    if ties not in {"breslow", "efron", "exact"}:
        raise ValueError("ties must be 'breslow', 'efron', or 'exact'")

    X_arr = np.asarray(X, dtype=np.float64)
    time_arr = np.asarray(time, dtype=np.float64).reshape(-1)
    event_raw = np.asarray(event, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")
    n_samples = X_arr.shape[0]
    if time_arr.shape[0] != n_samples or event_raw.shape[0] != n_samples:
        raise ValueError("time and event must have shape (n_samples,)")
    if not np.all(np.isfinite(X_arr)) or not np.all(np.isfinite(time_arr)):
        raise ValueError("X and time must contain only finite values")
    if not np.all(np.isfinite(event_raw)) or np.any(
        (event_raw != 0) & (event_raw != 1)
    ):
        raise ValueError("event must contain only 0/1 finite values")
    event_arr = event_raw.astype(np.int32)
    if coef is None:
        coef_arr = np.zeros(X_arr.shape[1], dtype=np.float64)
    else:
        coef_arr = np.asarray(coef, dtype=np.float64).reshape(-1)
    if coef_arr.shape[0] != X_arr.shape[1]:
        raise ValueError("coef must have shape (n_features,)")
    start_arr = None
    if entry is not None:
        start_arr = np.asarray(entry, dtype=np.float64).reshape(-1)
        if start_arr.shape[0] != n_samples:
            raise ValueError("entry must have shape (n_samples,)")
        if not np.all(np.isfinite(start_arr)):
            raise ValueError("entry must contain only finite values")
        if np.any(start_arr < 0) or np.any(start_arr >= time_arr):
            raise ValueError("each row must satisfy 0 <= entry < time")
    elif np.any(time_arr <= 0):
        raise ValueError("time must be positive when entry is not provided")

    if strata is None:
        strata_codes = np.zeros(n_samples, dtype=np.int64)
    else:
        strata_arr = np.asarray(strata).reshape(-1)
        if strata_arr.shape[0] != n_samples:
            raise ValueError("strata must have shape (n_samples,)")
        _, strata_codes = np.unique(strata_arr, return_inverse=True)
        strata_codes = strata_codes.astype(np.int64, copy=False)

    if not np.any(event_arr == 1):
        return 0.0

    # Exact tied likelihood uses an elementary-symmetric partition DP. Reuse
    # the single mathematical reference for that uncommon path; Breslow and
    # Efron below intentionally compute log-likelihood only, avoiding the O(p²)
    # score/information work in every held-out CV evaluation.
    if ties == "exact":
        result = cox_counting_process_objective(
            coef_arr,
            X_arr,
            time_arr,
            event_arr,
            start=np.zeros_like(time_arr) if start_arr is None else start_arr,
            strata=strata_codes,
            ties=ties,
            compute_derivatives=False,
        )
        return float(result["log_likelihood"])

    total_loglik = 0.0
    risk_scores = X_arr @ coef_arr
    for stratum_code in np.unique(strata_codes):
        stratum_mask = strata_codes == stratum_code
        if not np.any((event_arr == 1) & stratum_mask):
            continue
        order = np.argsort(time_arr[stratum_mask], kind="mergesort")
        time_sorted = time_arr[stratum_mask][order]
        event_sorted = event_arr[stratum_mask][order]
        risk_sorted = risk_scores[stratum_mask][order]
        start_sorted = (
            None if start_arr is None else start_arr[stratum_mask][order]
        )

        event_idx = np.flatnonzero(event_sorted == 1)
        event_times = time_sorted[event_idx]
        unique_times, counts = np.unique(event_times, return_counts=True)
        group_ends = np.cumsum(counts, dtype=np.int64)
        group_starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), group_ends[:-1]]
        )
        log_risk_suffix = None
        if start_sorted is None:
            log_risk_suffix = np.logaddexp.accumulate(
                risk_sorted[::-1]
            )[::-1]

        for group_idx, failure_time in enumerate(unique_times):
            n_failures = int(counts[group_idx])
            event_rows = event_idx[
                group_starts[group_idx] : group_ends[group_idx]
            ]
            sum_event_risk = float(np.sum(risk_sorted[event_rows]))

            if start_sorted is None:
                first_risk_idx = int(
                    np.searchsorted(time_sorted, failure_time, side="left")
                )
                log_risk_sum = float(log_risk_suffix[first_risk_idx])
                denominator_shift = log_risk_sum
                scaled_risk_sum = 1.0
            else:
                risk_mask = (start_sorted < failure_time) & (
                    time_sorted >= failure_time
                )
                if not np.any(risk_mask):
                    raise FloatingPointError(
                        "empty Cox risk set at an observed failure time"
                    )
                risk_at_time = risk_sorted[risk_mask]
                denominator_shift = float(np.max(risk_at_time))
                scaled_risk_sum = float(
                    np.sum(np.exp(risk_at_time - denominator_shift))
                )
                log_risk_sum = denominator_shift + np.log(scaled_risk_sum)

            if ties == "breslow":
                total_loglik += (
                    sum_event_risk - n_failures * log_risk_sum
                )
                continue

            scaled_failure_sum = float(
                np.sum(np.exp(risk_sorted[event_rows] - denominator_shift))
            )
            fractions = np.arange(n_failures, dtype=np.float64) / n_failures
            scaled_denominators = (
                scaled_risk_sum - fractions * scaled_failure_sum
            )
            if np.any(scaled_denominators <= 0):
                raise FloatingPointError(
                    "non-positive Cox risk-set denominator"
                )
            total_loglik += sum_event_risk - float(
                np.sum(
                    denominator_shift + np.log(scaled_denominators)
                )
            )

    return float(total_loglik)


# =============================================================================
# CV main function
# =============================================================================

def _select_coxph_penalty_cv(
    X,
    time,
    event,
    entry=None,
    cluster=None,
    start=None,
    strata=None,
    subject_id=None,
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
        Delayed-entry times. Mutually exclusive with ``start``.
    cluster : ndarray or None
        Cluster ids (used in model fitting; scoring remains partial likelihood).
    start : ndarray or None
        Counting-process start times; rows are at risk on ``(start, time]``.
    strata : ndarray or None
        Stratum labels defining independent risk sets.
    subject_id : ndarray or None
        Subject identifiers. Automatically generated folds keep all rows from
        one subject together.
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
        'breslow', 'efron', or 'exact'.
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
    device_name = (
        str(device).lower() if not isinstance(device, Device) else device.value
    )
    if device_name not in {member.value for member in Device}:
        raise ValueError("device must be 'cpu', 'cuda', 'torch', or 'auto'")
    use_gpu = device_name in (Device.CUDA.value, Device.TORCH.value)
    fit_device = device_name

    if entry is not None and start is not None:
        raise ValueError("pass only one of entry and start")
    entry_supplied = entry is not None
    start_supplied = start is not None
    start_values = entry if entry_supplied else start

    # Fold construction and diagnostics are orchestrated on the host. Explicit
    # GPU modes convert each fold once, then keep both candidate fitting and
    # held-out partial-likelihood scoring on the requested backend.
    X_np = np.asarray(_to_numpy(X), dtype=np.float64)
    time_np = np.asarray(_to_numpy(time), dtype=np.float64).reshape(-1)
    event_raw_np = np.asarray(_to_numpy(event), dtype=np.float64).reshape(-1)
    entry_np = (
        None
        if start_values is None
        else np.asarray(_to_numpy(start_values), dtype=np.float64).reshape(-1)
    )
    cluster_np = None if cluster is None else np.asarray(_to_numpy(cluster)).reshape(-1)
    strata_np = None if strata is None else np.asarray(_to_numpy(strata)).reshape(-1)
    subject_np = (
        None
        if subject_id is None
        else np.asarray(_to_numpy(subject_id)).reshape(-1)
    )

    if X_np.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")
    n_samples = X_np.shape[0]
    if time_np.shape[0] != n_samples or event_raw_np.shape[0] != n_samples:
        raise ValueError("time and event must have shape (n_samples,)")
    if not np.all(np.isfinite(X_np)) or not np.all(np.isfinite(time_np)):
        raise ValueError("X and time must contain only finite values")
    if not np.all(np.isfinite(event_raw_np)) or np.any(
        (event_raw_np != 0) & (event_raw_np != 1)
    ):
        raise ValueError("event must contain only 0/1 finite values")
    event_np = event_raw_np.astype(np.int32)
    if entry_np is not None and entry_np.shape[0] != n_samples:
        raise ValueError("entry must have shape (n_samples,)")
    if entry_np is not None and not np.all(np.isfinite(entry_np)):
        raise ValueError("entry must contain only finite values")
    if cluster_np is not None and cluster_np.shape[0] != n_samples:
        raise ValueError("cluster must have shape (n_samples,)")
    if strata_np is not None and strata_np.shape[0] != n_samples:
        raise ValueError("strata must have shape (n_samples,)")
    if subject_np is not None and subject_np.shape[0] != n_samples:
        raise ValueError("subject_id must have shape (n_samples,)")
    strata_codes_np = None
    if strata_np is not None:
        _, strata_codes_np = np.unique(strata_np, return_inverse=True)
        strata_codes_np = strata_codes_np.astype(np.int64, copy=False)

    # Generate penalty grid
    if penalties is None:
        penalties = _default_coxph_penalty_grid(X_np, time_np, event_np, n_penalties, penalty_min_ratio)
    else:
        penalties = np.asarray(penalties, dtype=np.float64)
        if penalties.ndim != 1 or penalties.size == 0:
            raise ValueError("penalties must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(penalties)):
            raise ValueError("penalties must contain only finite values")
        if np.any(penalties < 0):
            raise ValueError("penalties must be non-negative")

    n_penalties_actual = len(penalties)

    # Generate CV folds
    if cv_splits is not None:
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            folds.append(
                (
                    _coerce_cv_indices(train_idx, fold_idx=fold_idx, name="train"),
                    _coerce_cv_indices(test_idx, fold_idx=fold_idx, name="test"),
                )
            )
    else:
        if subject_np is None:
            if cv_folds < 2:
                raise ValueError("cv_folds must be at least 2")
            if cv_folds > n_samples:
                raise ValueError("cv_folds cannot exceed n_samples")
            folds = _kfold_indices(n_samples, cv_folds, random_state)
        else:
            folds = _group_kfold_indices(subject_np, cv_folds, random_state)

    if not folds:
        raise ValueError("cv_splits must contain at least one fold")
    _validate_cv_splits(folds, n_samples)

    if subject_np is not None:
        _, subject_codes = np.unique(subject_np, return_inverse=True)
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            shared_subjects = np.intersect1d(
                subject_codes[train_idx], subject_codes[test_idx]
            )
            if shared_subjects.size:
                raise ValueError(
                    "cv_splits must keep every subject_id wholly within train "
                    f"or test; fold {fold_idx} contains subject leakage"
                )

    folds_are_complements_flag = _folds_are_complements(folds, n_samples)
    n_folds = len(folds)
    train_event_counts = np.asarray(
        [int(np.sum(event_np[train_idx])) for train_idx, _ in folds],
        dtype=np.int64,
    )
    test_event_counts = np.asarray(
        [int(np.sum(event_np[test_idx])) for _, test_idx in folds],
        dtype=np.int64,
    )
    fold_valid = (train_event_counts > 0) & (test_event_counts > 0)
    n_effective_folds = int(np.sum(fold_valid))
    if n_effective_folds == 0:
        raise RuntimeError(
            "CoxPHCV could not evaluate any fold: each fold needs at least "
            "one event in both its training and held-out partitions."
        )

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
            entry=entry_np,
            cluster=cluster_np,
            strata=strata_np,
            subject_id=subject_np,
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

    # Per-candidate/per-fold diagnostics. Candidates are compared only when
    # they have a finite score on every data-valid fold, guaranteeing identical
    # effective fold counts for penalty selection.
    pl_path = np.full((n_penalties_actual, n_folds), np.nan, dtype=np.float64)
    converged_path = np.zeros((n_penalties_actual, n_folds), dtype=bool)
    attempted_path = np.zeros((n_penalties_actual, n_folds), dtype=bool)
    iterations_path = np.full(
        (n_penalties_actual, n_folds), -1, dtype=np.int64
    )
    failure_path = np.full(
        (n_penalties_actual, n_folds), "not_evaluated", dtype=object
    )
    failure_path[:, ~fold_valid] = "fold_has_no_train_or_test_events"

    def _reset_penalty_indices(penalty_indices: np.ndarray) -> None:
        penalty_indices = np.unique(
            np.asarray(penalty_indices, dtype=np.int64)
        )
        if penalty_indices.size == 0:
            return
        active = np.ix_(penalty_indices, np.flatnonzero(fold_valid))
        pl_path[active] = np.nan
        converged_path[active] = False
        attempted_path[active] = False
        iterations_path[active] = -1
        failure_path[active] = "not_evaluated"

    def _complete_candidate_mask() -> np.ndarray:
        eligible = np.isfinite(pl_path[:, fold_valid]) & converged_path[:, fold_valid]
        return np.all(eligible, axis=1)

    def _complete_candidate_means(
        penalty_indices: np.ndarray,
    ) -> np.ndarray:
        penalty_indices = np.asarray(penalty_indices, dtype=np.int64)
        means = np.full(penalty_indices.shape[0], np.nan, dtype=np.float64)
        complete = _complete_candidate_mask()[penalty_indices]
        if np.any(complete):
            means[complete] = np.mean(
                pl_path[penalty_indices[complete]][:, fold_valid], axis=1
            )
        return means

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
            if not fold_valid[fold_idx]:
                continue
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            time_train, time_test = time_np[train_idx], time_np[test_idx]
            event_train, event_test = event_np[train_idx], event_np[test_idx]
            entry_train = None if entry_np is None else entry_np[train_idx]
            entry_test = None if entry_np is None else entry_np[test_idx]
            cluster_train = None if cluster_np is None else cluster_np[train_idx]
            strata_train = None if strata_np is None else strata_np[train_idx]
            strata_test = None if strata_np is None else strata_np[test_idx]
            strata_test_codes = (
                None
                if strata_codes_np is None
                else strata_codes_np[test_idx]
            )
            subject_train = None if subject_np is None else subject_np[train_idx]
            X_fit = X_train
            time_fit = time_train
            event_fit = event_train
            entry_fit = entry_train
            cluster_fit = cluster_train
            X_score = X_test
            time_score = time_test
            event_score = event_test
            entry_score = entry_test
            strata_score = strata_test_codes

            # Prepare one fold per explicit backend and reuse it across the
            # penalty path. Import/conversion failures propagate: explicit GPU
            # requests never fall back to NumPy or switch GPU frameworks.
            if fit_device == Device.CUDA.value:
                import cupy as cp

                X_fit = cp.asarray(X_train, dtype=cp.float64)
                time_fit = cp.asarray(time_train, dtype=cp.float64)
                event_fit = cp.asarray(event_train, dtype=cp.int32)
                entry_fit = (
                    None
                    if entry_train is None
                    else cp.asarray(entry_train, dtype=cp.float64)
                )
                X_score = cp.asarray(X_test, dtype=cp.float64)
                time_score = cp.asarray(time_test, dtype=cp.float64)
                event_score = cp.asarray(event_test, dtype=cp.int32)
                entry_score = (
                    None
                    if entry_test is None
                    else cp.asarray(entry_test, dtype=cp.float64)
                )
                strata_score = (
                    None
                    if strata_test_codes is None
                    else cp.asarray(strata_test_codes, dtype=cp.int64)
                )
            elif fit_device == Device.TORCH.value:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "device='torch' requires torch.cuda.is_available() "
                        "to be True; no Torch CPU fallback is performed."
                    )
                torch_device = "cuda"
                X_fit = torch.as_tensor(
                    X_train, dtype=torch.float64, device=torch_device
                )
                time_fit = torch.as_tensor(
                    time_train, dtype=torch.float64, device=torch_device
                )
                event_fit = torch.as_tensor(
                    event_train, dtype=torch.int32, device=torch_device
                )
                entry_fit = (
                    None
                    if entry_train is None
                    else torch.as_tensor(
                        entry_train, dtype=torch.float64, device=torch_device
                    )
                )
                X_score = torch.as_tensor(
                    X_test, dtype=torch.float64, device=torch_device
                )
                time_score = torch.as_tensor(
                    time_test, dtype=torch.float64, device=torch_device
                )
                event_score = torch.as_tensor(
                    event_test, dtype=torch.int32, device=torch_device
                )
                entry_score = (
                    None
                    if entry_test is None
                    else torch.as_tensor(
                        entry_test,
                        dtype=torch.float64,
                        device=torch_device,
                    )
                )
                strata_score = (
                    None
                    if strata_test_codes is None
                    else torch.as_tensor(
                        strata_test_codes,
                        dtype=torch.int64,
                        device=torch_device,
                    )
                )

            prev_coef = None
            for penalty_idx in penalty_indices:
                if attempted_path[penalty_idx, fold_idx]:
                    continue
                penalty = penalties[penalty_idx]
                model = CoxPH(
                    ties=ties,
                    max_iter=fit_max_iter,
                    tol=fit_tol,
                    device=fit_device,
                    compute_inference=False,
                    compute_cindex=False,
                    penalty=penalty,
                )
                attempted_path[penalty_idx, fold_idx] = True
                try:
                    model.fit(
                        X_fit,
                        time_fit,
                        event_fit,
                        entry=entry_fit if entry_supplied else None,
                        cluster=cluster_fit,
                        init_coef=prev_coef,
                        start=entry_fit if start_supplied else None,
                        strata=strata_train,
                        subject_id=subject_train,
                    )
                except Exception as exc:
                    failure_path[penalty_idx, fold_idx] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise

                converged_path[penalty_idx, fold_idx] = bool(
                    getattr(model, "_converged", False)
                )
                iterations_path[penalty_idx, fold_idx] = int(
                    getattr(model, "_iterations", -1)
                )
                coef_np = np.asarray(_to_numpy(model.coef_), dtype=np.float64)
                if not np.all(np.isfinite(coef_np)):
                    failure_path[penalty_idx, fold_idx] = (
                        "non_finite_coefficients"
                    )
                    continue
                if converged_path[penalty_idx, fold_idx]:
                    prev_coef = coef_np.copy()
                if fit_device == Device.CPU.value:
                    pl_test = _compute_partial_likelihood(
                        X_test,
                        time_test,
                        event_test,
                        coef_np,
                        entry=entry_test,
                        strata=strata_test,
                        ties=ties,
                    )
                else:
                    if fit_device == Device.CUDA.value:
                        coef_score = cp.asarray(coef_np, dtype=cp.float64)
                    else:
                        coef_score = torch.as_tensor(
                            coef_np,
                            dtype=torch.float64,
                            device=torch_device,
                        )
                    score_result = cox_counting_process_objective(
                        coef_score,
                        X_score,
                        time_score,
                        event_score,
                        start=entry_score,
                        strata=strata_score,
                        ties=ties,
                        compute_derivatives=False,
                    )
                    pl_test = float(
                        np.asarray(
                            _to_numpy(score_result["log_likelihood"])
                        )
                    )
                if not np.isfinite(pl_test):
                    failure_path[penalty_idx, fold_idx] = (
                        "non_finite_partial_likelihood"
                    )
                    continue
                pl_path[penalty_idx, fold_idx] = float(pl_test)
                failure_path[penalty_idx, fold_idx] = (
                    None
                    if converged_path[penalty_idx, fold_idx]
                    else "did_not_converge"
                )

    if two_stage_enabled:
        stage1_idx = np.unique(
            np.linspace(0, n_penalties_actual - 1, num=coarse_n, dtype=np.int64)
        )
        _evaluate_penalty_indices(
            stage1_idx,
            fit_max_iter=(fast_iter if halving_enabled else max_iter),
            fit_tol=(fast_tol if halving_enabled else tol),
        )
        stage1_mean = _complete_candidate_means(stage1_idx)
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
            stage2_mean = _complete_candidate_means(stage2_idx)
            stage2_valid = np.isfinite(stage2_mean)
            if np.any(stage2_valid):
                order = np.argsort(np.nan_to_num(stage2_mean, nan=-np.inf))[::-1]
                top_idx = stage2_idx[order[: min(halving_topk, len(stage2_idx))]]
                # Only the full-precision finalists remain eligible. Retaining
                # fast-pass scores would compare different optimization
                # tolerances despite equal fold counts.
                screened_out = np.setdiff1d(
                    np.arange(n_penalties_actual, dtype=np.int64), top_idx
                )
                _reset_penalty_indices(screened_out)
                _reset_penalty_indices(top_idx)
                _evaluate_penalty_indices(top_idx, fit_max_iter=max_iter, fit_tol=tol)
    else:
        full_idx = np.arange(n_penalties_actual, dtype=np.int64)
        if halving_enabled:
            _evaluate_penalty_indices(full_idx, fit_max_iter=fast_iter, fit_tol=fast_tol)
            full_mean = _complete_candidate_means(full_idx)
            full_valid = np.isfinite(full_mean)
            if np.any(full_valid):
                order = np.argsort(np.nan_to_num(full_mean, nan=-np.inf))[::-1]
                top_idx = full_idx[order[:halving_topk]]
                screened_out = np.setdiff1d(full_idx, top_idx)
                _reset_penalty_indices(screened_out)
                _reset_penalty_indices(top_idx)
                _evaluate_penalty_indices(top_idx, fit_max_iter=max_iter, fit_tol=tol)
        else:
            _evaluate_penalty_indices(full_idx, fit_max_iter=max_iter, fit_tol=tol)

    # A staged/fast pass may leave candidates unevaluated or incomplete. If no
    # complete candidate exists, give every candidate one full-precision pass
    # before declaring CV failure.
    candidate_complete = _complete_candidate_mask()
    if not np.any(candidate_complete):
        all_indices = np.arange(n_penalties_actual, dtype=np.int64)
        _reset_penalty_indices(all_indices)
        _evaluate_penalty_indices(
            all_indices, fit_max_iter=max_iter, fit_tol=tol
        )
        candidate_complete = _complete_candidate_mask()

    if not np.any(candidate_complete):
        effective_fold_counts = np.sum(
            np.isfinite(pl_path[:, fold_valid])
            & converged_path[:, fold_valid],
            axis=1,
        ).astype(np.int64)
        raise RuntimeError(
            "All CoxPHCV penalty candidates failed to converge with finite scores "
            f"on the same {n_effective_folds} effective folds; observed fold "
            f"counts were {effective_fold_counts.tolist()}."
        )

    # Compute mean partial likelihood across folds
    mean_pl = np.full(n_penalties_actual, np.nan, dtype=np.float64)
    mean_pl[candidate_complete] = np.mean(
        pl_path[candidate_complete][:, fold_valid], axis=1
    )
    effective_fold_counts = np.sum(
        np.isfinite(pl_path[:, fold_valid])
        & converged_path[:, fold_valid],
        axis=1,
    ).astype(np.int64)

    # Find best penalty (maximum partial likelihood)
    best_idx = int(np.nanargmax(mean_pl))
    best_penalty = float(penalties[best_idx])
    best_pl = float(mean_pl[best_idx])

    fold_indices = [
        (train_idx.copy(), test_idx.copy()) for train_idx, test_idx in folds
    ]
    fold_metadata = [
        {
            "fold": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_events_train": int(train_event_counts[fold_idx]),
            "n_events_test": int(test_event_counts[fold_idx]),
            "valid": bool(fold_valid[fold_idx]),
        }
        for fold_idx, (train_idx, test_idx) in enumerate(folds)
    ]
    if subject_np is not None:
        for metadata, (train_idx, test_idx) in zip(fold_metadata, folds):
            metadata["n_subjects_train"] = int(
                np.unique(subject_np[train_idx]).shape[0]
            )
            metadata["n_subjects_test"] = int(
                np.unique(subject_np[test_idx]).shape[0]
            )

    # Prepare details
    details = {
        "penalty": best_penalty,
        "penalties": penalties.astype(np.float64),
        "pl_path": pl_path.astype(np.float64),
        "mean_pl": mean_pl.astype(np.float64),
        "best_pl": best_pl,
        "n_folds": n_folds,
        "fold": np.arange(n_folds, dtype=np.int64),
        "fold_indices": fold_indices,
        "fold_metadata": fold_metadata,
        "fold_valid": fold_valid.copy(),
        "folds_are_complements": bool(folds_are_complements_flag),
        "converged_path": converged_path.copy(),
        "convergence": converged_path.copy(),
        "attempted_path": attempted_path.copy(),
        "iterations_path": iterations_path.copy(),
        "failure_path": failure_path.copy(),
        "effective_fold_counts": effective_fold_counts,
        "effective_n_folds": n_effective_folds,
        "candidate_complete": candidate_complete.copy(),
        "effective_device": fit_device,
        "scoring_device": fit_device,
        "orchestration_device": "cpu",
        "grouped_by_subject": subject_np is not None,
        "uses_start": entry_np is not None,
        "uses_strata": strata_np is not None,
        "uses_subject_id": subject_np is not None,
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
        Method for handling ties: 'breslow', 'efron', or 'exact'.
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
        Whether to free backend caches after public prediction/scoring calls
        and when the estimator is destroyed. Fit-time caches are retained.
    random_state : int or None
        Random seed for CV splits.

    Attributes
    ----------
    penalty_ : float
        Selected penalty value.
    penalties_ : ndarray
        All penalty values tested.
    cv_results_ : dict
        CV scores plus fold indices, convergence/failure diagnostics, effective
        fold counts, and the effective device.
    best_score_ : float
        Best (maximum) partial likelihood across CV folds.
    coef_ : ndarray
        Coefficients of the final model.
    hazard_ratios_ : ndarray
        exp(coef) = hazard ratios.
    estimator_ : CoxPH
        The fitted CoxPH with selected penalty.
    effective_device_ : str
        Backend used for both CV candidate fits and the final refit.

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

    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        """Expose modern sklearn tags for packed survival responses."""
        try:
            from sklearn.utils._tags import RegressorTags, Tags, TargetTags
        except ImportError:  # scikit-learn < 1.6
            return {"requires_y": True, "multioutput": True}

        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(
                required=True,
                one_d_labels=False,
                two_d_labels=True,
                multi_output=True,
                single_output=False,
            ),
            regressor_tags=RegressorTags(),
        )

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
        # Preserve public constructor objects exactly for sklearn.clone().
        # Normalization for computation happens at fit time.
        self.penalties = penalties
        self.n_penalties = n_penalties
        self.penalty_min_ratio = penalty_min_ratio
        self.cv = cv
        self.cv_splits = cv_splits
        self.ties = ties
        self.tol = tol
        self.max_iter = max_iter
        self.compute_inference = compute_inference
        self.cov_type = cov_type
        self.gpu_memory_cleanup = gpu_memory_cleanup

        ties_name = str(ties).lower()
        cov_type_name = str(cov_type).lower()
        if ties_name not in {"breslow", "efron", "exact"}:
            raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
        if cov_type_name not in {"nonrobust", "hc0", "hc1", "cluster"}:
            raise ValueError(
                "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'cluster'"
            )

        # Output attributes (initialized to None)
        self.penalty_ = None
        self.penalties_ = None
        self.cv_results_ = None
        self.best_score_ = None
        self.coef_ = None
        self.hazard_ratios_ = None
        self.estimator_ = None
        self.effective_device_ = None

    def _reset_fit_state(self):
        """Remove every fitted/CV artifact before a new public fit attempt."""
        self._fitted = False
        self.penalty_ = None
        self.penalties_ = None
        self.cv_results_ = None
        self.best_score_ = None
        self.coef_ = None
        self.hazard_ratios_ = None
        self.estimator_ = None
        self.effective_device_ = None

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

    def _fit_cv(
        self,
        X,
        time,
        event,
        entry=None,
        cluster=None,
        *,
        start=None,
        strata=None,
        subject_id=None,
    ):
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
        start : array-like, optional
            Counting-process start times. Mutually exclusive with ``entry``.
        strata : array-like, optional
            Stratum labels defining independent risk sets.
        subject_id : array-like, optional
            Subject identifiers used for grouped folds and counting-process
            concordance.

        Returns
        -------
        self
        """
        device_name = self._get_compute_device().value
        fit_device_name = device_name
        ties_name = str(self.ties).lower()
        cov_type_name = str(self.cov_type).lower()
        n_penalties = int(self.n_penalties)
        penalty_min_ratio = float(self.penalty_min_ratio)
        cv_folds = int(self.cv)
        max_iter = int(self.max_iter)
        tol = float(self.tol)

        penalties = (
            None
            if self.penalties is None
            else np.asarray(_to_numpy(self.penalties), dtype=np.float64)
        )

        # Perform CV to find best penalty
        best_penalty, details = _select_coxph_penalty_cv(
            X, time, event,
            entry=entry,
            cluster=cluster,
            start=start,
            strata=strata,
            subject_id=subject_id,
            penalties=penalties,
            n_penalties=n_penalties,
            penalty_min_ratio=penalty_min_ratio,
            cv_folds=cv_folds,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            ties=ties_name,
            device=fit_device_name,
            max_iter=max_iter,
            tol=tol,
            return_details=True,
        )

        # Store CV results
        self.penalty_ = float(best_penalty)
        self.penalties_ = np.asarray(details["penalties"], dtype=np.float64)

        pl_path = np.asarray(details["pl_path"], dtype=np.float64)
        mean_pl = np.asarray(details["mean_pl"], dtype=np.float64)

        self.cv_results_ = {}
        for key, value in details.items():
            if key == "penalty":
                continue
            if isinstance(value, np.ndarray):
                self.cv_results_[key] = value.copy()
            elif key == "fold_indices":
                self.cv_results_[key] = [
                    (train_idx.copy(), test_idx.copy())
                    for train_idx, test_idx in value
                ]
            elif key == "fold_metadata":
                self.cv_results_[key] = [dict(item) for item in value]
            else:
                self.cv_results_[key] = value
        # Preserve normalized arrays even if a custom selector supplied lists.
        self.cv_results_["pl_path"] = pl_path
        self.cv_results_["mean_pl"] = mean_pl
        self.best_score_ = float(details["best_pl"])
        self.effective_device_ = str(
            details.get("effective_device", fit_device_name)
        )

        # Fit final model on full data with best penalty
        final_model = CoxPH(
            ties=ties_name,
            tol=tol,
            max_iter=max_iter,
            device=fit_device_name,
            n_jobs=self.n_jobs,
            compute_inference=bool(self.compute_inference),
            cov_type=cov_type_name,
            gpu_memory_cleanup=bool(self.gpu_memory_cleanup),
            penalty=self.penalty_,
        )
        final_model.fit(
            X,
            time,
            event,
            entry=entry,
            cluster=cluster,
            start=start,
            strata=strata,
            subject_id=subject_id,
        )

        self.estimator_ = final_model
        self.coef_ = np.asarray(_to_numpy(final_model.coef_)).copy()
        self.hazard_ratios_ = np.asarray(
            _to_numpy(final_model.hazard_ratios_)
        ).copy()
        self._fitted = True

        return self

    def fit(
        self,
        X,
        time,
        event=None,
        entry=None,
        cluster=None,
        *,
        start=None,
        strata=None,
        subject_id=None,
    ):
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
        start : array-like, optional
            Counting-process start times. Mutually exclusive with ``entry``.
        strata : array-like, optional
            Stratum labels defining independent risk sets.
        subject_id : array-like, optional
            Subject identifiers. All rows from one subject remain in the same
            automatically generated CV fold.

        Returns
        -------
        self : CoxPHCV
        """
        self._reset_fit_state()
        try:
            time, event, entry, start = _unpack_survival_target(
                time, event, entry=entry, start=start
            )
            return self._fit_cv(
                X,
                time,
                event,
                entry=entry,
                cluster=cluster,
                start=start,
                strata=strata,
                subject_id=subject_id,
            )
        except Exception:
            # A failed refit must never leave the previous estimator, or a
            # partially updated CV result, observable through public methods.
            self._reset_fit_state()
            raise

    def predict(self, X):
        """
        Predict hazard ratios through the final refitted ``CoxPH``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.

        Returns
        -------
        hazard_ratios : ndarray
            ``exp(X @ coef_)`` from the selected/refitted estimator.
        """
        try:
            if self.estimator_ is None:
                raise ValueError("Model not fitted. Call fit() first.")
            return self.estimator_.predict(
                np.asarray(_to_numpy(X), dtype=np.float64)
            )
        finally:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()

    def predict_risk_score(self, X):
        """Predict the linear risk score ``X @ coef_``."""
        try:
            if self.estimator_ is None:
                raise ValueError("Model not fitted. Call fit() first.")
            return self.estimator_.predict_risk_score(
                np.asarray(_to_numpy(X), dtype=np.float64)
            )
        finally:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()

    def predict_hazard_ratio(self, X):
        """Predict hazard ratios through the final refitted estimator."""
        return self.predict(X)

    def predict_survival(self, X, times=None, strata=None):
        """Predict survival curves through the final refitted estimator."""
        try:
            if self.estimator_ is None:
                raise ValueError("Model not fitted. Call fit() first.")
            return self.estimator_.predict_survival(
                np.asarray(_to_numpy(X), dtype=np.float64),
                times=times,
                strata=strata,
            )
        finally:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()

    def score(self, X, time, event=None, start=None, strata=None, subject_id=None):
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
        start : array-like, optional
            Counting-process start times.
        strata : array-like, optional
            Stratum labels for stratified concordance.
        subject_id : array-like, optional
            Subject identifiers; within-subject row pairs are excluded.

        Returns
        -------
        c_index : float
            C-index (0.5 = random, 1.0 = perfect).
        """
        try:
            if self.estimator_ is None:
                raise ValueError("Model not fitted. Call fit() first.")
            time, event, _, start = _unpack_survival_target(
                time, event, start=start
            )
            return float(
                self.estimator_.score(
                    np.asarray(_to_numpy(X), dtype=np.float64),
                    np.asarray(_to_numpy(time), dtype=np.float64),
                    np.asarray(_to_numpy(event), dtype=np.float64),
                    start=(
                        None
                        if start is None
                        else np.asarray(_to_numpy(start), dtype=np.float64)
                    ),
                    strata=(
                        None if strata is None else np.asarray(_to_numpy(strata))
                    ),
                    subject_id=(
                        None
                        if subject_id is None
                        else np.asarray(_to_numpy(subject_id))
                    ),
                )
            )
        finally:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()

    def summary(self):
        """Return summary of the fitted model."""
        if self.estimator_ is None:
            raise RuntimeError("No fitted estimator available.")
        if not hasattr(self.estimator_, "summary"):
            raise RuntimeError(f"{self.estimator_.__class__.__name__} does not implement summary().")
        return self.estimator_.summary()
