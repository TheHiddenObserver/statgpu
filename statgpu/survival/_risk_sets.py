"""Backend-native Cox counting-process risk-set primitives.

This module is the correctness reference for delayed entry, start/stop data,
and stratified Cox models.  It deliberately keeps the statistical definition
in one place for NumPy, CuPy, and Torch.  Specialized no-entry kernels may be
faster, but they must agree with these primitives.

The counting-process convention matches R's ``Surv(start, stop, event)``:
rows are at risk on the half-open interval ``(start, stop]``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _backend_name(value: Any) -> str:
    module = type(value).__module__
    if module.startswith("cupy"):
        return "cupy"
    if module.startswith("torch"):
        return "torch"
    return "numpy"


def _array_namespace(value: Any):
    name = _backend_name(value)
    if name == "cupy":
        import cupy as cp

        return name, cp
    if name == "torch":
        import torch

        return name, torch
    return name, np


def _scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _zeros(backend: str, xp: Any, shape: Tuple[int, ...], like: Any):
    if backend == "torch":
        return xp.zeros(shape, dtype=like.dtype, device=like.device)
    return xp.zeros(shape, dtype=like.dtype)


def _eye(backend: str, xp: Any, n: int, like: Any):
    if backend == "torch":
        return xp.eye(n, dtype=like.dtype, device=like.device)
    return xp.eye(n, dtype=like.dtype)


def _unique_sorted(values: Any, backend: str, xp: Any):
    if backend == "torch":
        return xp.unique(values, sorted=True)
    return xp.unique(values)


def _nonzero(mask: Any, backend: str, xp: Any):
    if backend == "torch":
        return xp.nonzero(mask, as_tuple=False).reshape(-1)
    return xp.nonzero(mask)[0]


def _outer(a: Any, b: Any, backend: str, xp: Any):
    if backend == "torch":
        return xp.outer(a, b)
    return xp.outer(a, b)


def _sum(value: Any, backend: str, xp: Any, axis=None):
    if backend == "torch":
        if axis is None:
            return xp.sum(value)
        return xp.sum(value, dim=axis)
    return xp.sum(value, axis=axis)


def _max(value: Any, backend: str, xp: Any):
    if backend == "torch":
        return xp.max(value)
    return xp.max(value)


def _log(value: Any, xp: Any):
    return xp.log(value)


def _exp(value: Any, xp: Any):
    return xp.exp(value)


def _exp_finite_float64(value: Any, backend: str, xp: Any):
    """Exponentiate a log quantity without overflow warnings or infinities."""
    upper = float(np.log(np.finfo(np.float64).max))
    if backend == "torch":
        return xp.exp(xp.clamp(value, max=upper))
    return xp.exp(xp.minimum(value, upper))


def _as_backend_array(value: Any, backend: str, xp: Any, like: Any, *, integer=False):
    if backend == "torch":
        dtype = xp.int64 if integer else like.dtype
        return xp.as_tensor(value, dtype=dtype, device=like.device)
    dtype = xp.int64 if integer else like.dtype
    return xp.asarray(value, dtype=dtype)


def _as_float(mask: Any, backend: str, like: Any):
    if backend == "torch":
        return mask.to(dtype=like.dtype)
    return mask.astype(like.dtype, copy=False)


def _nonnegative_env_int(name: str, default: int, maximum: int) -> int:
    """Read a bounded non-negative integer without import-time fragility."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        value = int(default)
    return min(max(0, value), int(maximum))


def _torch_channelwise_scan_limits() -> Tuple[int, int]:
    """Return the row/channel bounds for the Torch Exact split-scan path."""
    min_rows = _nonnegative_env_int(
        "STATGPU_TORCH_EXACT_SCAN_MIN_ROWS", 2048, 10_000_000
    )
    max_channels = _nonnegative_env_int(
        "STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS", 64, 4096
    )
    return min_rows, max_channels


def _torch_channelwise_scan_strategy(value: Any, xp: Any) -> str:
    """Resolve native/channelwise scanning with a conservative auto gate."""
    strategy = os.environ.get(
        "STATGPU_TORCH_EXACT_SCAN_STRATEGY", "auto"
    ).strip().lower()
    if strategy not in {"auto", "native", "channelwise"}:
        strategy = "auto"
    if strategy != "auto":
        return strategy

    # The optimization is evidenced on Torch 2.0 and Pascal/P100. Later Torch
    # releases and newer GPU architectures use native cumsum until benchmarked.
    version = str(getattr(xp, "__version__", "")).split("+")[0]
    try:
        capability = tuple(xp.cuda.get_device_capability(value.device))
    except Exception:
        return "native"
    return "channelwise" if version.startswith("2.0.") and capability == (6, 0) else "native"


def _cumsum_axis0(value: Any, backend: str, xp: Any, *, allow_channelwise: bool = True):
    """Cumulative sum over rows with a bounded Torch CUDA channel split.

    Torch 2.0 CUDA has a severe long-scan penalty for small multi-dimensional
    tensors when the sample axis is scanned directly.  For sufficiently long
    arrays with a bounded number of trailing channels, make those channels
    contiguous and run the efficient one-dimensional scan per channel.  Small,
    wide, CPU, NumPy, and CuPy inputs retain their native single-call path.
    """
    if backend != "torch":
        return xp.cumsum(value, axis=0)
    if value.ndim <= 1 or not bool(value.is_cuda):
        return xp.cumsum(value, dim=0)

    n_rows = int(value.shape[0])
    n_channels = math.prod(int(size) for size in value.shape[1:])
    min_rows, max_channels = _torch_channelwise_scan_limits()
    strategy = _torch_channelwise_scan_strategy(value, xp)
    if (
        not allow_channelwise
        or strategy == "native"
        or n_rows < min_rows
        or max_channels == 0
        or n_channels > max_channels
    ):
        return xp.cumsum(value, dim=0)

    channel_major = value.reshape(n_rows, n_channels).transpose(0, 1).contiguous()
    scanned = xp.stack(
        [xp.cumsum(channel_major[channel], dim=0) for channel in range(n_channels)],
        dim=1,
    )
    return scanned.reshape(value.shape)


def _center_within_strata(X: Any, strata: Any, backend: str, xp: Any):
    """Center covariates by stratum on their existing backend."""
    if backend == "torch":
        unique, inverse = xp.unique(
            strata, sorted=True, return_inverse=True
        )
        sums = _zeros(backend, xp, (int(unique.shape[0]), int(X.shape[1])), X)
        sums.index_add_(0, inverse, X)
        counts = xp.bincount(inverse, minlength=int(unique.shape[0])).to(
            dtype=X.dtype
        )
    else:
        unique, inverse = xp.unique(strata, return_inverse=True)
        sums = _zeros(backend, xp, (int(unique.shape[0]), int(X.shape[1])), X)
        xp.add.at(sums, inverse, X)
        counts = xp.bincount(inverse, minlength=int(unique.shape[0])).astype(
            X.dtype, copy=False
        )
    means = sums / counts.reshape(-1, 1)
    return X - means[inverse]


def _segment_codes(starts: Any, backend: str, xp: Any):
    """Return zero-based segment codes from a boolean start mask."""
    if backend == "torch":
        return xp.cumsum(starts.to(dtype=xp.int64), dim=0) - 1
    return xp.cumsum(starts.astype(xp.int64, copy=False), axis=0) - 1


def _segmented_cumsum_axis0(
    value: Any,
    segment_codes: Any,
    segment_starts: Any,
    backend: str,
    xp: Any,
    *,
    allow_channelwise: bool = True,
):
    """Cumulative sum over contiguous segments without a Python segment loop."""
    cumulative = _cumsum_axis0(
        value, backend, xp, allow_channelwise=allow_channelwise
    )
    n_segments = int(segment_starts.shape[0])
    offsets = _zeros(
        backend, xp, (n_segments, *tuple(value.shape[1:])), value
    )
    if n_segments > 1:
        offsets[1:] = cumulative[segment_starts[1:] - 1]
    return cumulative - offsets[segment_codes]


def _group_sum_axis0(
    value: Any, group_codes: Any, n_groups: int, backend: str, xp: Any
):
    """Sum rows by integer group code on the active backend."""
    output = _zeros(
        backend, xp, (n_groups, *tuple(value.shape[1:])), value
    )
    if backend == "torch":
        output.index_add_(0, group_codes, value)
    else:
        xp.add.at(output, group_codes, value)
    return output


def _group_max_1d(
    value: Any, group_codes: Any, n_groups: int, backend: str, xp: Any
):
    """Maximum of a vector by integer group code on the active backend."""
    if backend == "torch":
        output = xp.full(
            (n_groups,), -float("inf"), dtype=value.dtype, device=value.device
        )
        output.scatter_reduce_(
            0, group_codes, value, reduce="amax", include_self=True
        )
    else:
        output = xp.full((n_groups,), -float("inf"), dtype=value.dtype)
        xp.maximum.at(output, group_codes, value)
    return output


def _batched_group_objective(
    eta: Any,
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
    *,
    ties: str,
    score_residuals: bool,
    compute_derivatives: bool,
) -> Dict[str, Any]:
    """Vectorized Breslow/Efron objective for counting-process risk sets.

    Failure times are processed in bounded dense batches.  This replaces one
    Python/device-kernel launch sequence per failure time with matrix products
    and batched second moments while capping the temporary risk-mask size.
    The exact-ties path deliberately remains on its elementary-symmetric DP.
    """
    backend, xp = _array_namespace(X)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    loglik = _zeros(backend, xp, (), X)
    score = _zeros(backend, xp, (n_features,), X) if compute_derivatives else None
    information = (
        _zeros(backend, xp, (n_features, n_features), X)
        if compute_derivatives
        else None
    )
    residuals = (
        _zeros(backend, xp, (n_samples, n_features), X) if score_residuals else None
    )
    # Cap the combined dense batch workspace. Derivative evaluation creates two
    # ``batch x p x p`` second-moment tensors in addition to several risk-set
    # views, while log-likelihood-only evaluation creates no p-squared tensor.
    # Accounting for both terms prevents wide models from exhausting GPU memory.
    max_batch_entries = 2_000_000
    for stratum in _unique_sorted(strata, backend, xp):
        stratum_idx = _nonzero(strata == stratum, backend, xp)
        Xs = X[stratum_idx] if compute_derivatives else None
        stops = stop[stratum_idx]
        starts = start[stratum_idx]
        events = event[stratum_idx]
        etas = eta[stratum_idx]
        failure_times = _unique_sorted(stops[events == 1], backend, xp)
        n_groups = int(failure_times.shape[0])
        if n_groups == 0:
            continue

        n_stratum = int(stratum_idx.shape[0])
        entries_per_group = 4 * max(n_stratum, 1)
        if compute_derivatives:
            entries_per_group += 2 * max(n_features * n_features, 1)
        batch_size = max(
            1, min(n_groups, max_batch_entries // max(entries_per_group, 1))
        )
        residual_stratum = (
            _zeros(backend, xp, (n_stratum, n_features), X)
            if residuals is not None
            else None
        )

        for batch_start in range(0, n_groups, batch_size):
            times = failure_times[batch_start : batch_start + batch_size]
            risk_mask = (starts.reshape(1, -1) < times.reshape(-1, 1)) & (
                stops.reshape(1, -1) >= times.reshape(-1, 1)
            )
            fail_mask = (events.reshape(1, -1) == 1) & (
                stops.reshape(1, -1) == times.reshape(-1, 1)
            )
            risk_float = _as_float(risk_mask, backend, X)
            fail_float = _as_float(fail_mask, backend, X)
            # A failure-time-specific shift is essential: a stratum-wide
            # extreme linear predictor may already have left later risk sets.
            # Masked entries use -inf and therefore cannot determine the max.
            masked_eta = xp.where(
                risk_mask,
                etas.reshape(1, -1),
                xp.full_like(risk_float, -float("inf")),
            )
            if backend == "torch":
                eta_shift = xp.max(masked_eta, dim=1).values
            else:
                eta_shift = xp.max(masked_eta, axis=1)
            shifted_eta = xp.where(
                risk_mask,
                etas.reshape(1, -1) - eta_shift.reshape(-1, 1),
                xp.full_like(risk_float, -float("inf")),
            )
            group_weights = _exp(shifted_eta, xp)
            weighted_risk = group_weights
            weighted_fail = fail_float * group_weights

            d = _sum(fail_float, backend, xp, axis=1)
            s0 = _sum(weighted_risk, backend, xp, axis=1)
            e0 = _sum(weighted_fail, backend, xp, axis=1)
            if _scalar_bool(_sum(s0 <= 0, backend, xp) > 0):
                raise FloatingPointError("non-positive Cox risk-set denominator")
            loglik = loglik + _sum(fail_float @ etas, backend, xp)
            if compute_derivatives:
                s1 = weighted_risk @ Xs
                e1 = weighted_fail @ Xs
                s2 = xp.einsum("bn,ni,nj->bij", weighted_risk, Xs, Xs)
                e2 = xp.einsum("bn,ni,nj->bij", weighted_fail, Xs, Xs)
                score = score + _sum(fail_float @ Xs, backend, xp, axis=0)

            if residual_stratum is not None:
                # Conventional counting-process martingale score residuals,
                # matching statsmodels PHReg.score_residuals.  The sandwich
                # meat uses a Breslow hazard increment even when the partial
                # likelihood bread uses Efron ties.
                xbar = s1 / s0.reshape(-1, 1)
                event_count = _sum(fail_float, backend, xp, axis=0)
                hazard_weight = weighted_risk * (d / s0).reshape(-1, 1)
                hazard_count = _sum(hazard_weight, backend, xp, axis=0)
                residual_stratum = residual_stratum + (
                    Xs * event_count.reshape(-1, 1)
                    - fail_float.T @ xbar
                    - Xs * hazard_count.reshape(-1, 1)
                    + hazard_weight.T @ xbar
                )

            if ties == "breslow":
                loglik = loglik - _sum(d * (_log(s0, xp) + eta_shift), backend, xp)
                if compute_derivatives:
                    mean = s1 / s0.reshape(-1, 1)
                    score = score - _sum(d.reshape(-1, 1) * mean, backend, xp, axis=0)
                    covariance = s2 / s0.reshape(-1, 1, 1) - xp.einsum(
                        "bi,bj->bij", mean, mean
                    )
                    information = information + _sum(
                        d.reshape(-1, 1, 1) * covariance,
                        backend,
                        xp,
                        axis=0,
                    )
                continue

            max_ties = _scalar_int(_max(d, backend, xp))
            for substep in range(max_ties):
                active = d > float(substep)
                # Every row is an observed failure group, so d >= 1.  The
                # active mask makes inactive groups algebraically zero.
                frac = float(substep) / d
                denom = s0 - frac * e0
                if _scalar_bool(_sum(active & (denom <= 0), backend, xp) > 0):
                    raise FloatingPointError("non-positive Cox risk-set denominator")
                active_float = _as_float(active, backend, X)
                safe_denom = xp.where(active, denom, xp.ones_like(denom))
                loglik = loglik - _sum(
                    active_float * (_log(safe_denom, xp) + eta_shift),
                    backend,
                    xp,
                )
                if compute_derivatives:
                    a1 = s1 - frac.reshape(-1, 1) * e1
                    a2 = s2 - frac.reshape(-1, 1, 1) * e2
                    mean = a1 / safe_denom.reshape(-1, 1)
                    score = score - _sum(
                        active_float.reshape(-1, 1) * mean,
                        backend,
                        xp,
                        axis=0,
                    )
                    covariance = a2 / safe_denom.reshape(-1, 1, 1) - xp.einsum(
                        "bi,bj->bij", mean, mean
                    )
                    information = information + _sum(
                        active_float.reshape(-1, 1, 1) * covariance,
                        backend,
                        xp,
                        axis=0,
                    )
        if residuals is not None:
            residuals[stratum_idx] = residual_stratum

    result = {"log_likelihood": loglik}
    if compute_derivatives:
        result["score"] = score
        result["information"] = 0.5 * (information + information.T)
    if residuals is not None:
        result["score_residuals"] = residuals
    return result


def _numpy_group_objective(
    eta: np.ndarray,
    X: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    start: np.ndarray,
    strata: np.ndarray,
    *,
    ties: str,
    score_residuals: bool,
    compute_derivatives: bool,
) -> Dict[str, Any]:
    """BLAS-oriented NumPy reference without dense group-by-row tensors."""
    n_samples, n_features = X.shape
    loglik = 0.0
    score = np.zeros(n_features, dtype=X.dtype) if compute_derivatives else None
    information = (
        np.zeros((n_features, n_features), dtype=X.dtype)
        if compute_derivatives
        else None
    )
    residuals = (
        np.zeros((n_samples, n_features), dtype=X.dtype) if score_residuals else None
    )
    for stratum in np.unique(strata):
        stratum_mask = strata == stratum
        event_mask_s = stratum_mask & (event == 1)
        for failure_time in np.unique(stop[event_mask_s]):
            fail_mask = event_mask_s & (stop == failure_time)
            risk_mask = stratum_mask & (start < failure_time) & (stop >= failure_time)
            fail_idx = np.flatnonzero(fail_mask)
            risk_idx = np.flatnonzero(risk_mask)
            d = int(fail_idx.size)
            if d == 0:
                continue
            if risk_idx.size == 0:
                raise FloatingPointError(
                    "empty Cox risk set at an observed failure time"
                )
            eta_shift = float(np.max(eta[risk_idx]))
            w_risk = np.exp(eta[risk_idx] - eta_shift)
            w_fail = np.exp(eta[fail_idx] - eta_shift)
            s0 = float(np.sum(w_risk))
            e0 = float(np.sum(w_fail))

            loglik += float(np.sum(eta[fail_idx]))
            if compute_derivatives:
                X_risk = X[risk_idx]
                X_fail = X[fail_idx]
                s1 = X_risk.T @ w_risk
                s2 = (X_risk * w_risk[:, None]).T @ X_risk
                e1 = X_fail.T @ w_fail
                e2 = (X_fail * w_fail[:, None]).T @ X_fail
                score += np.sum(X_fail, axis=0)
            if residuals is not None:
                xbar = s1 / s0
                residuals[risk_idx] -= (X_risk - xbar) * (w_risk * d / s0)[:, None]
                residuals[fail_idx] += X_fail - xbar

            if ties == "breslow":
                loglik -= d * (np.log(s0) + eta_shift)
                if compute_derivatives:
                    mean = s1 / s0
                    score -= d * mean
                    information += d * (s2 / s0 - np.outer(mean, mean))
                continue

            for substep in range(d):
                frac = float(substep) / float(d)
                denom = s0 - frac * e0
                if denom <= 0:
                    raise FloatingPointError("non-positive Cox risk-set denominator")
                loglik -= np.log(denom) + eta_shift
                if compute_derivatives:
                    a1 = s1 - frac * e1
                    a2 = s2 - frac * e2
                    mean = a1 / denom
                    score -= mean
                    information += a2 / denom - np.outer(mean, mean)

    result: Dict[str, Any] = {"log_likelihood": np.asarray(loglik, dtype=X.dtype)}
    if compute_derivatives:
        result["score"] = score
        result["information"] = 0.5 * (information + information.T)
    if residuals is not None:
        result["score_residuals"] = residuals
    return result


def _nested_exact_group_objective(
    eta: Any,
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
    *,
    score_residuals: bool,
    compute_derivatives: bool,
):
    """Exact objective for nested one-stratum right-censored risk sets.

    Sorting rows by decreasing stop time turns every risk set into a prefix.
    The size-k elementary-symmetric state for every prefix can then be formed
    with one cumulative sum, so the DP costs ``O(n * max_ties)`` instead of
    carrying a separate state for every failure group.  A conservative numeric
    gate retains the normalized log-space reference for extreme predictors or
    combinatorial counts that are unsafe in ordinary float64 arithmetic.
    """
    if score_residuals:
        return None
    backend, xp = _array_namespace(X)
    if _scalar_bool(_sum(start != 0, backend, xp) > 0):
        return None

    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if backend == "torch":
        order = xp.argsort(stop, descending=True, stable=True)
        order = order[xp.argsort(strata[order], stable=True)]
    else:
        order = xp.lexsort(xp.stack((-stop, strata), axis=0))
    sorted_stop = stop[order]
    sorted_strata = strata[order]
    sorted_eta = eta[order]
    sorted_X = X[order]
    sorted_event_mask = event[order] == 1
    event_rows = _nonzero(sorted_event_mask, backend, xp)
    if int(event_rows.shape[0]) == 0:
        return None

    stratum_starts_mask = xp.zeros_like(sorted_strata, dtype=xp.bool_ if backend != "torch" else xp.bool)
    stratum_starts_mask[0] = True
    stratum_starts_mask[1:] = sorted_strata[1:] != sorted_strata[:-1]
    stratum_codes = _segment_codes(stratum_starts_mask, backend, xp)
    stratum_starts = _nonzero(stratum_starts_mask, backend, xp)
    n_strata = int(stratum_starts.shape[0])

    event_stops = sorted_stop[event_rows]
    event_stratum_codes = stratum_codes[event_rows]
    failure_starts_mask = xp.zeros_like(
        event_stratum_codes, dtype=xp.bool_ if backend != "torch" else xp.bool
    )
    failure_starts_mask[0] = True
    failure_starts_mask[1:] = (
        (event_stratum_codes[1:] != event_stratum_codes[:-1])
        | (event_stops[1:] != event_stops[:-1])
    )
    event_group_codes = _segment_codes(failure_starts_mask, backend, xp)
    failure_starts = _nonzero(failure_starts_mask, backend, xp)
    n_groups = int(failure_starts.shape[0])
    if n_groups == 0:
        return None
    integer_counts = xp.bincount(event_group_codes, minlength=n_groups)

    max_ties = _scalar_int(_max(integer_counts, backend, xp))
    eta_min = xp.min(eta)
    eta_range = float((_max(eta, backend, xp) - eta_min).item())
    max_abs_x = float(_max(xp.abs(X), backend, xp).item())
    log_combinations = (
        math.lgamma(n_samples + 1)
        - math.lgamma(max_ties + 1)
        - math.lgamma(n_samples - max_ties + 1)
    )
    # Global scaling makes every row weight <= 1.  These bounds prevent either
    # early-prefix underflow or raw polynomial/moment overflow.  Shapes outside
    # the safe region keep the fully normalized log-space implementation.
    if (
        max_ties * eta_range > 500.0
        or log_combinations > 600.0
        or log_combinations + 2.0 * math.log(max(max_abs_x, 1.0)) > 600.0
    ):
        return None

    state_width = 1
    if compute_derivatives:
        state_width += n_features + n_features * n_features
    itemsize = X.element_size() if backend == "torch" else int(X.dtype.itemsize)
    n_events = int(event_rows.shape[0])
    event_state_width = 2 + (2 * n_features if compute_derivatives else 0)
    base_estimated_bytes = itemsize * (
        12 * n_samples * state_width
        + 4 * n_events * event_state_width
        + 4 * n_groups * state_width
    )
    max_bytes = _nonnegative_env_int(
        "STATGPU_EXACT_NESTED_MAX_BYTES",
        512 * 1024 * 1024,
        1 << 50,
    )
    if max_bytes == 0 or base_estimated_bytes > max_bytes:
        return None

    allow_torch_channelwise = True
    if backend == "torch":
        min_scan_rows, max_scan_channels = _torch_channelwise_scan_limits()
        eligible_channels = [
            channels
            for channels in (n_features, n_features * n_features)
            if 0 < channels <= max_scan_channels
        ]
        split_scan_extra_bytes = 0
        if n_samples >= min_scan_rows and eligible_channels:
            # Contiguous channel-major input, per-channel outputs, and stacked
            # row-major output can coexist at the scan boundary. If those extras
            # do not fit, keep the nested DP but use Torch's native scan instead
            # of falling back to the much more expensive general Exact path.
            split_scan_extra_bytes = itemsize * 3 * max(eligible_channels) * n_samples
        allow_torch_channelwise = (
            base_estimated_bytes + split_scan_extra_bytes <= max_bytes
        )

    eta_shift_by_stratum = _group_max_1d(
        sorted_eta, stratum_codes, n_strata, backend, xp
    )
    weights = _exp(
        sorted_eta - eta_shift_by_stratum[stratum_codes], xp
    )

    stop_block_starts_mask = xp.zeros_like(stratum_starts_mask)
    stop_block_starts_mask[0] = True
    stop_block_starts_mask[1:] = (
        (sorted_strata[1:] != sorted_strata[:-1])
        | (sorted_stop[1:] != sorted_stop[:-1])
    )
    stop_block_codes = _segment_codes(stop_block_starts_mask, backend, xp)
    stop_block_ends_mask = xp.zeros_like(stratum_starts_mask)
    stop_block_ends_mask[-1] = True
    stop_block_ends_mask[:-1] = stop_block_starts_mask[1:]
    stop_block_ends = _nonzero(stop_block_ends_mask, backend, xp)
    stop_block_risk_counts = (
        stop_block_ends
        - stratum_starts[stratum_codes[stop_block_ends]]
        + 1
    )
    risk_counts_by_row = stop_block_risk_counts[stop_block_codes]
    failure_event_positions = event_rows[failure_starts]
    risk_counts = risk_counts_by_row[failure_event_positions]
    if _scalar_bool(_sum(risk_counts < integer_counts, backend, xp) > 0):
        raise FloatingPointError("exact failure count exceeds its Cox risk set")

    sorted_event_eta = sorted_eta[sorted_event_mask]
    failure_eta = _group_sum_axis0(
        sorted_event_eta,
        event_group_codes,
        n_groups,
        backend,
        xp,
    )
    failure_group_stratum_codes = event_stratum_codes[failure_starts]
    eta_shift = eta_shift_by_stratum[failure_group_stratum_codes]
    failure_X = None
    if compute_derivatives:
        sorted_event_X = sorted_X[sorted_event_mask]
        failure_X = _group_sum_axis0(
            sorted_event_X,
            event_group_codes,
            n_groups,
            backend,
            xp,
        )
    counts = _as_float(integer_counts, backend, X)
    partition = _zeros(backend, xp, (n_groups,), X)
    exact_mean = (
        _zeros(backend, xp, (n_groups, n_features), X) if compute_derivatives else None
    )
    exact_second = (
        _zeros(backend, xp, (n_groups, n_features, n_features), X)
        if compute_derivatives
        else None
    )

    previous_z = xp.ones_like(sorted_eta)
    previous_first = (
        _zeros(backend, xp, (n_samples, n_features), X) if compute_derivatives else None
    )
    previous_second = (
        _zeros(backend, xp, (n_samples, n_features, n_features), X)
        if compute_derivatives
        else None
    )
    row_outer = (
        xp.einsum("ni,nj->nij", sorted_X, sorted_X) if compute_derivatives else None
    )
    zero_z = _zeros(backend, xp, (1,), X)
    zero_first = (
        _zeros(backend, xp, (1, n_features), X) if compute_derivatives else None
    )
    zero_second = (
        _zeros(backend, xp, (1, n_features, n_features), X)
        if compute_derivatives
        else None
    )

    for subset_size in range(1, max_ties + 1):
        if subset_size == 1:
            base_z = previous_z
            base_first = previous_first
            base_second = previous_second
        elif backend == "torch":
            base_z = xp.cat((zero_z, previous_z[:-1]), dim=0)
            if compute_derivatives:
                base_first = xp.cat((zero_first, previous_first[:-1]), dim=0)
                base_second = xp.cat((zero_second, previous_second[:-1]), dim=0)
        else:
            base_z = xp.concatenate((zero_z, previous_z[:-1]), axis=0)
            if compute_derivatives:
                base_first = xp.concatenate((zero_first, previous_first[:-1]), axis=0)
                base_second = xp.concatenate(
                    (zero_second, previous_second[:-1]), axis=0
                )
        if subset_size > 1:
            base_z = xp.where(stratum_starts_mask, 0.0, base_z)
            if compute_derivatives:
                base_first = xp.where(
                    stratum_starts_mask.reshape(-1, 1), 0.0, base_first
                )
                base_second = xp.where(
                    stratum_starts_mask.reshape(-1, 1, 1),
                    0.0,
                    base_second,
                )

        contribution_z = weights * base_z
        current_z = _segmented_cumsum_axis0(
            contribution_z,
            stratum_codes,
            stratum_starts,
            backend,
            xp,
            allow_channelwise=False,
        )
        if compute_derivatives:
            contribution_first = weights.reshape(-1, 1) * (
                base_first + base_z.reshape(-1, 1) * sorted_X
            )
            cross = base_first.reshape(n_samples, n_features, 1) * sorted_X.reshape(
                n_samples, 1, n_features
            ) + sorted_X.reshape(n_samples, n_features, 1) * base_first.reshape(
                n_samples, 1, n_features
            )
            contribution_second = weights.reshape(-1, 1, 1) * (
                base_second + cross + base_z.reshape(-1, 1, 1) * row_outer
            )
            current_first = _segmented_cumsum_axis0(
                contribution_first,
                stratum_codes,
                stratum_starts,
                backend,
                xp,
                allow_channelwise=allow_torch_channelwise,
            )
            current_second = _segmented_cumsum_axis0(
                contribution_second,
                stratum_codes,
                stratum_starts,
                backend,
                xp,
                allow_channelwise=allow_torch_channelwise,
            )

        selected = integer_counts == subset_size
        if _scalar_bool(_sum(selected, backend, xp) > 0):
            group_idx = _nonzero(selected, backend, xp)
            prefix_idx = (
                stratum_starts[
                    failure_group_stratum_codes[group_idx]
                ]
                + risk_counts[group_idx]
                - 1
            )
            selected_z = current_z[prefix_idx]
            partition[group_idx] = selected_z
            if compute_derivatives:
                exact_mean[group_idx] = current_first[prefix_idx] / selected_z.reshape(
                    -1, 1
                )
                exact_second[group_idx] = current_second[
                    prefix_idx
                ] / selected_z.reshape(-1, 1, 1)

        previous_z = current_z
        if compute_derivatives:
            previous_first = current_first
            previous_second = current_second

    if _scalar_bool(_sum((partition <= 0) | ~xp.isfinite(partition), backend, xp) > 0):
        return None
    loglik = _sum(
        failure_eta - _log(partition, xp) - counts * eta_shift,
        backend,
        xp,
    )
    result: Dict[str, Any] = {"log_likelihood": loglik}
    if compute_derivatives:
        if _scalar_bool(
            _sum(~xp.isfinite(exact_mean), backend, xp)
            + _sum(~xp.isfinite(exact_second), backend, xp)
            > 0
        ):
            return None
        score = _sum(failure_X - exact_mean, backend, xp, axis=0)
        covariance = exact_second - xp.einsum("gi,gj->gij", exact_mean, exact_mean)
        information = _sum(covariance, backend, xp, axis=0)
        result["score"] = score
        result["information"] = 0.5 * (information + information.T)
    return result


def _batched_exact_states(
    X: Any,
    log_weights: Any,
    risk_mask: Any,
    counts: Any,
    backend: str,
    xp: Any,
    *,
    compute_derivatives: bool,
):
    """Evaluate every exact failure group in one row-wise DP scan.

    The elementary-symmetric recurrence is independent across failure groups
    but sequential across risk rows.  Keeping a leading group dimension reduces
    the launch-bound loop from ``sum(risk_set_sizes)`` iterations to ``n_rows``
    while preserving the same normalized log-space moments.
    """
    n_groups, n_rows = int(risk_mask.shape[0]), int(risk_mask.shape[1])
    n_features = int(X.shape[1])
    max_ties = _scalar_int(_max(counts, backend, xp))
    if backend == "torch":
        counts_int = counts.to(dtype=xp.int64)
        subset_sizes = xp.arange(
            1, max_ties + 1, dtype=xp.int64, device=X.device
        ).reshape(1, -1)
    else:
        counts_int = counts.astype(xp.int64, copy=False)
        subset_sizes = xp.arange(1, max_ties + 1, dtype=xp.int64).reshape(1, -1)

    log_z = _zeros(backend, xp, (n_groups, max_ties + 1), X)
    log_z[:, 1:] = -float("inf")
    mean = (
        _zeros(backend, xp, (n_groups, max_ties + 1, n_features), X)
        if compute_derivatives
        else None
    )
    second = (
        _zeros(
            backend,
            xp,
            (n_groups, max_ties + 1, n_features, n_features),
            X,
        )
        if compute_derivatives
        else None
    )
    processed = xp.zeros_like(counts_int)

    for row in range(n_rows):
        active = risk_mask[:, row]
        if backend == "torch":
            processed_next = processed + active.to(dtype=xp.int64)
        else:
            processed_next = processed + active.astype(xp.int64, copy=False)
        valid = (
            active.reshape(-1, 1)
            & (subset_sizes <= counts_int.reshape(-1, 1))
            & (subset_sizes <= processed_next.reshape(-1, 1))
        )

        old_log_z = log_z[:, 1:]
        previous_log_z = log_z[:, :-1]
        added_log_z = log_weights[:, row].reshape(-1, 1) + previous_log_z
        new_log_z = xp.logaddexp(old_log_z, added_log_z)
        safe_new_log_z = xp.where(valid, new_log_z, 0.0)
        old_weight = xp.where(valid, _exp(old_log_z - safe_new_log_z, xp), 0.0)
        added_weight = xp.where(valid, _exp(added_log_z - safe_new_log_z, xp), 0.0)

        if compute_derivatives:
            old_mean = mean[:, 1:]
            previous_mean = mean[:, :-1]
            old_second = second[:, 1:]
            previous_second = second[:, :-1]
            x = X[row].reshape(1, 1, n_features)
            added_mean = previous_mean + x
            outer_x = x.reshape(1, 1, n_features, 1) * x.reshape(1, 1, 1, n_features)
            cross = previous_mean.reshape(
                n_groups, max_ties, n_features, 1
            ) * x.reshape(1, 1, 1, n_features) + x.reshape(
                1, 1, n_features, 1
            ) * previous_mean.reshape(
                n_groups, max_ties, 1, n_features
            )
            added_second = previous_second + cross + outer_x
            new_mean = (
                old_weight.reshape(n_groups, max_ties, 1) * old_mean
                + added_weight.reshape(n_groups, max_ties, 1) * added_mean
            )
            new_second = (
                old_weight.reshape(n_groups, max_ties, 1, 1) * old_second
                + added_weight.reshape(n_groups, max_ties, 1, 1) * added_second
            )
            mean[:, 1:] = xp.where(
                valid.reshape(n_groups, max_ties, 1), new_mean, old_mean
            )
            second[:, 1:] = xp.where(
                valid.reshape(n_groups, max_ties, 1, 1), new_second, old_second
            )

        log_z[:, 1:] = xp.where(valid, new_log_z, old_log_z)
        processed = processed_next

    if backend == "torch":
        group_index = xp.arange(n_groups, dtype=xp.int64, device=X.device)
    else:
        group_index = xp.arange(n_groups, dtype=xp.int64)
    partition = log_z[group_index, counts_int]
    if not compute_derivatives:
        return partition, None, None
    return (
        partition,
        mean[group_index, counts_int],
        second[group_index, counts_int],
    )


def _batched_exact_group_objective(
    eta: Any,
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
    *,
    score_residuals: bool,
    compute_derivatives: bool,
):
    """Batched Exact objective for backend-native multi-stratum workloads.

    Return ``None`` when the estimated dense workspace exceeds the configured
    ceiling so the memory-bounded per-group reference path remains available.
    """
    backend, xp = _array_namespace(X)
    n_strata = int(_unique_sorted(strata, backend, xp).shape[0])
    if n_strata > 1 and (backend == "numpy" or n_strata < 8):
        return None
    event_rows = _nonzero(event == 1, backend, xp)
    if int(event_rows.shape[0]) == 0:
        return None
    event_times = stop[event_rows]
    event_strata = strata[event_rows]
    if backend == "torch":
        event_order = xp.argsort(event_times, descending=True, stable=True)
        event_order = event_order[
            xp.argsort(event_strata[event_order], stable=True)
        ]
    else:
        event_order = xp.lexsort(
            xp.stack((-event_times, event_strata), axis=0)
        )
    grouped_times = event_times[event_order]
    grouped_strata = event_strata[event_order]
    failure_starts_mask = xp.zeros_like(
        grouped_strata, dtype=xp.bool_ if backend != "torch" else xp.bool
    )
    failure_starts_mask[0] = True
    failure_starts_mask[1:] = (
        (grouped_strata[1:] != grouped_strata[:-1])
        | (grouped_times[1:] != grouped_times[:-1])
    )
    event_group_codes = _segment_codes(failure_starts_mask, backend, xp)
    failure_starts = _nonzero(failure_starts_mask, backend, xp)
    n_groups = int(failure_starts.shape[0])
    if n_groups == 0:
        return None
    integer_counts = xp.bincount(event_group_codes, minlength=n_groups)
    failure_times = grouped_times[failure_starts]
    failure_strata = grouped_strata[failure_starts]
    counts = _as_float(integer_counts, backend, X)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    max_ties = _scalar_int(_max(integer_counts, backend, xp))
    state_width = 1
    if compute_derivatives:
        state_width += n_features + n_features * n_features
    itemsize = X.element_size() if backend == "torch" else int(X.dtype.itemsize)
    estimated_bytes = itemsize * (
        4 * n_groups * n_samples + 12 * n_groups * (max_ties + 1) * state_width
    )
    max_bytes = _nonnegative_env_int(
        "STATGPU_EXACT_BATCH_MAX_BYTES",
        512 * 1024 * 1024,
        1 << 50,
    )
    if max_bytes == 0 or estimated_bytes > max_bytes:
        return None

    same_stratum = strata.reshape(1, -1) == failure_strata.reshape(-1, 1)
    risk_mask = same_stratum & (
        start.reshape(1, -1) < failure_times.reshape(-1, 1)
    ) & (stop.reshape(1, -1) >= failure_times.reshape(-1, 1))
    fail_mask = (event.reshape(1, -1) == 1) & (
        stop.reshape(1, -1) == failure_times.reshape(-1, 1)
    ) & same_stratum
    risk_float = _as_float(risk_mask, backend, X)
    fail_float = _as_float(fail_mask, backend, X)
    risk_counts = _sum(risk_float, backend, xp, axis=1)
    if _scalar_bool(_sum(risk_counts <= 0, backend, xp) > 0):
        raise FloatingPointError("empty Cox risk set at an observed failure time")

    masked_eta = xp.where(
        risk_mask,
        eta.reshape(1, -1),
        xp.full_like(risk_float, -float("inf")),
    )
    if backend == "torch":
        eta_shift = xp.max(masked_eta, dim=1).values
    else:
        eta_shift = xp.max(masked_eta, axis=1)
    log_weights = xp.where(
        risk_mask,
        eta.reshape(1, -1) - eta_shift.reshape(-1, 1),
        xp.zeros_like(risk_float),
    )
    partition, exact_mean, exact_second = _batched_exact_states(
        X,
        log_weights,
        risk_mask,
        counts,
        backend,
        xp,
        compute_derivatives=compute_derivatives,
    )
    if _scalar_bool(_sum(~xp.isfinite(partition), backend, xp) > 0):
        raise FloatingPointError("non-finite exact Cox tie log-partition")

    loglik = _sum(
        fail_float @ eta - partition - counts * eta_shift,
        backend,
        xp,
    )
    result: Dict[str, Any] = {"log_likelihood": loglik}
    if compute_derivatives:
        score = _sum(fail_float @ X - exact_mean, backend, xp, axis=0)
        covariance = exact_second - xp.einsum("gi,gj->gij", exact_mean, exact_mean)
        information = _sum(covariance, backend, xp, axis=0)
        result["score"] = score
        result["information"] = 0.5 * (information + information.T)
    if score_residuals:
        event_count = _sum(fail_float, backend, xp, axis=0)
        allocation = exact_mean / risk_counts.reshape(-1, 1)
        result["score_residuals"] = (
            X * event_count.reshape(-1, 1) - risk_float.T @ allocation
        )
    return result


def _reference_exact_group_objective(
    eta: Any,
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
    *,
    score_residuals: bool,
    compute_derivatives: bool,
) -> Dict[str, Any]:
    """Evaluate Exact ties with the bounded reference loop."""
    backend, xp = _array_namespace(X)
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    loglik = _zeros(backend, xp, (), X)
    score = _zeros(backend, xp, (n_features,), X) if compute_derivatives else None
    information = (
        _zeros(backend, xp, (n_features, n_features), X)
        if compute_derivatives
        else None
    )
    residuals = (
        _zeros(backend, xp, (n_samples, n_features), X)
        if score_residuals
        else None
    )

    for stratum in _unique_sorted(strata, backend, xp):
        stratum_mask = strata == stratum
        event_mask_s = stratum_mask & (event == 1)
        failure_times = _unique_sorted(stop[event_mask_s], backend, xp)
        for failure_time in failure_times:
            fail_mask = event_mask_s & (stop == failure_time)
            risk_mask = stratum_mask & (start < failure_time) & (stop >= failure_time)
            fail_idx = _nonzero(fail_mask, backend, xp)
            risk_idx = _nonzero(risk_mask, backend, xp)
            d = int(fail_idx.shape[0])
            if d == 0:
                continue
            if int(risk_idx.shape[0]) == 0:
                raise FloatingPointError(
                    "empty Cox risk set at an observed failure time"
                )

            eta_shift = _max(eta[risk_idx], backend, xp)
            log_w_risk = eta[risk_idx] - eta_shift
            loglik = loglik + _sum(eta[fail_idx], backend, xp)
            if compute_derivatives:
                X_risk = X[risk_idx]
                X_fail = X[fail_idx]
                score = score + _sum(X_fail, backend, xp, axis=0)
                if residuals is not None:
                    residuals[fail_idx] = residuals[fail_idx] + X_fail
                (
                    log_partition,
                    exact_mean,
                    exact_second,
                ) = _exact_tie_log_partition_moments(
                    X_risk, log_w_risk, d, backend, xp
                )
            else:
                log_partition = _exact_tie_log_partition(
                    log_w_risk, d, backend, xp
                )
            if _scalar_bool(~xp.isfinite(log_partition)):
                raise FloatingPointError("non-finite exact Cox tie log-partition")
            loglik = loglik - (log_partition + float(d) * eta_shift)
            if compute_derivatives:
                score = score - exact_mean
                information = information + (
                    exact_second - _outer(exact_mean, exact_mean, backend, xp)
                )
                if residuals is not None:
                    allocation = exact_mean / float(risk_idx.shape[0])
                    residuals[risk_idx] = residuals[risk_idx] - allocation

    result = {"log_likelihood": loglik}
    if compute_derivatives:
        result["score"] = score
        result["information"] = 0.5 * (information + information.T)
    if residuals is not None:
        result["score_residuals"] = residuals
    return result


def _stratified_exact_group_objective(
    eta: Any,
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
    *,
    score_residuals: bool,
    compute_derivatives: bool,
):
    """Compose optimized one-stratum Exact objectives across strata.

    Exact partial likelihoods are additive across strata. Reusing a bounded
    fast path once per stratum avoids the launch-bound stratum-by-failure-time
    reference loop without weakening its numerical or memory safety fallback.
    """
    if score_residuals:
        return None
    backend, xp = _array_namespace(X)
    unique_strata = _unique_sorted(strata, backend, xp)
    if int(unique_strata.shape[0]) <= 1:
        return None

    n_features = int(X.shape[1])
    loglik = _zeros(backend, xp, (), X)
    score = _zeros(backend, xp, (n_features,), X) if compute_derivatives else None
    information = (
        _zeros(backend, xp, (n_features, n_features), X)
        if compute_derivatives
        else None
    )
    for stratum in unique_strata:
        rows = _nonzero(strata == stratum, backend, xp)
        event_s = event[rows]
        if _scalar_int(_sum(event_s == 1, backend, xp)) == 0:
            continue
        X_s = X[rows]
        stop_s = stop[rows]
        start_s = start[rows]
        strata_s = strata[rows]
        eta_s = eta[rows]
        result = _nested_exact_group_objective(
            eta_s,
            X_s,
            stop_s,
            event_s,
            start_s,
            strata_s,
            score_residuals=False,
            compute_derivatives=compute_derivatives,
        )
        if result is None:
            result = _batched_exact_group_objective(
                eta_s,
                X_s,
                stop_s,
                event_s,
                start_s,
                strata_s,
                score_residuals=False,
                compute_derivatives=compute_derivatives,
            )
        if result is None:
            result = _reference_exact_group_objective(
                eta_s,
                X_s,
                stop_s,
                event_s,
                start_s,
                strata_s,
                score_residuals=False,
                compute_derivatives=compute_derivatives,
            )
        loglik = loglik + result["log_likelihood"]
        if compute_derivatives:
            score = score + result["score"]
            information = information + result["information"]

    combined: Dict[str, Any] = {"log_likelihood": loglik}
    if compute_derivatives:
        combined["score"] = score
        combined["information"] = 0.5 * (information + information.T)
    return combined


def _exact_tie_log_partition_moments(
    X_risk: Any,
    log_w_risk: Any,
    d: int,
    backend: str,
    xp: Any,
):
    """Stable elementary-symmetric DP for an exact tied-event group.

    Returns ``(log_Z, E[S], E[S S'])`` for the weighted distribution over all
    size-``d`` subsets, where ``S`` is the subset covariate sum. Maintaining
    normalized moments and ``log_Z`` avoids overflow from combinatorial counts
    such as ``choose(1100, 550)``.

    All subset sizes for one risk-set row are updated as one backend operation.
    Snapshotting the previous row preserves the descending-DP dependency while
    avoiding one GPU kernel-launch sequence per ``(row, subset_size)`` pair.
    """
    n_risk, n_features = int(X_risk.shape[0]), int(X_risk.shape[1])
    if d > n_risk:
        raise ValueError("number of tied events cannot exceed the risk-set size")
    log_z = _zeros(backend, xp, (d + 1,), X_risk)
    log_z[1:] = -float("inf")
    mean = _zeros(backend, xp, (d + 1, n_features), X_risk)
    second = _zeros(backend, xp, (d + 1, n_features, n_features), X_risk)

    def snapshot(value: Any):
        return value.clone() if backend == "torch" else value.copy()

    for row in range(n_risk):
        upper = min(d, row + 1)
        x = X_risk[row]
        log_weight = log_w_risk[row]

        old_log_z = snapshot(log_z[1 : upper + 1])
        previous_log_z = snapshot(log_z[:upper])
        added_log_z = log_weight + previous_log_z
        new_log_z = xp.logaddexp(old_log_z, added_log_z)
        old_weight = _exp(old_log_z - new_log_z, xp)
        added_weight = _exp(added_log_z - new_log_z, xp)

        old_mean = snapshot(mean[1 : upper + 1])
        previous_mean = snapshot(mean[:upper])
        old_second = snapshot(second[1 : upper + 1])
        previous_second = snapshot(second[:upper])
        added_mean = previous_mean + x.reshape(1, -1)
        outer_x = _outer(x, x, backend, xp).reshape(1, n_features, n_features)
        cross = previous_mean.reshape(upper, n_features, 1) * x.reshape(
            1, 1, n_features
        ) + x.reshape(1, n_features, 1) * previous_mean.reshape(upper, 1, n_features)
        added_second = previous_second + cross + outer_x

        mean[1 : upper + 1] = (
            old_weight.reshape(-1, 1) * old_mean
            + added_weight.reshape(-1, 1) * added_mean
        )
        second[1 : upper + 1] = (
            old_weight.reshape(-1, 1, 1) * old_second
            + added_weight.reshape(-1, 1, 1) * added_second
        )
        log_z[1 : upper + 1] = new_log_z
    return log_z[d], mean[d], second[d]


def _exact_tie_log_partition(
    log_w_risk: Any,
    d: int,
    backend: str,
    xp: Any,
):
    """Return only the exact-tie log partition without p-squared moments."""
    n_risk = int(log_w_risk.shape[0])
    if d > n_risk:
        raise ValueError("number of tied events cannot exceed the risk-set size")
    log_z = _zeros(backend, xp, (d + 1,), log_w_risk)
    log_z[1:] = -float("inf")

    def snapshot(value: Any):
        return value.clone() if backend == "torch" else value.copy()

    for row in range(n_risk):
        upper = min(d, row + 1)
        old_log_z = snapshot(log_z[1 : upper + 1])
        previous_log_z = snapshot(log_z[:upper])
        log_z[1 : upper + 1] = xp.logaddexp(old_log_z, log_w_risk[row] + previous_log_z)
    return log_z[d]


def _validate_counting_process_inputs(
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
) -> None:
    if getattr(X, "ndim", None) != 2:
        raise ValueError("X must be a 2-dimensional array")
    n = int(X.shape[0])
    for name, value in (
        ("stop", stop),
        ("event", event),
        ("start", start),
        ("strata", strata),
    ):
        if getattr(value, "ndim", None) != 1 or int(value.shape[0]) != n:
            raise ValueError(f"{name} must have shape (n_samples,)")

    backend, xp = _array_namespace(X)
    for name, value in (("X", X), ("stop", stop), ("event", event), ("start", start)):
        if _scalar_bool(_sum(~xp.isfinite(value), backend, xp) > 0):
            raise ValueError(f"{name} must contain only finite values")
    if _scalar_bool(_sum((event != 0) & (event != 1), backend, xp) > 0):
        raise ValueError("event must contain only 0/1 values")
    if _scalar_bool(_sum(start < 0, backend, xp) > 0):
        raise ValueError("start times must be non-negative")
    if _scalar_bool(_sum(stop <= start, backend, xp) > 0):
        raise ValueError("each row must satisfy start < stop")
    if _scalar_int(_sum(event, backend, xp)) == 0:
        raise ValueError("at least one observed event is required")


def prepare_counting_process_inputs(
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Normalize counting-process arrays without changing their backend."""
    backend, xp = _array_namespace(X)
    if backend == "torch":
        X = X.to(dtype=xp.float64)
        stop = xp.as_tensor(stop, dtype=X.dtype, device=X.device)
        # Validate in floating point before converting to integer so values
        # such as 0.5 or 1.9 cannot be silently truncated into valid events.
        event = xp.as_tensor(event, dtype=X.dtype, device=X.device)
        start = (
            xp.zeros_like(stop)
            if start is None
            else xp.as_tensor(start, dtype=X.dtype, device=X.device)
        )
        if strata is None:
            strata = xp.zeros(stop.shape[0], dtype=xp.int64, device=X.device)
        else:
            strata_raw = xp.as_tensor(strata, device=X.device)
            if strata_raw.ndim != 1 or int(strata_raw.shape[0]) != int(stop.shape[0]):
                raise ValueError("strata must have shape (n_samples,)")
            if strata_raw.is_complex():
                raise ValueError("strata must contain integer-valued labels")
            if strata_raw.is_floating_point():
                invalid = ~xp.isfinite(strata_raw) | (strata_raw != xp.round(strata_raw))
                if _scalar_bool(xp.any(invalid)):
                    raise ValueError("strata must contain finite integer-valued labels")
            strata = strata_raw.to(dtype=xp.int64)
    else:
        X = xp.asarray(X, dtype=xp.float64)
        stop = xp.asarray(stop, dtype=xp.float64)
        event = xp.asarray(event, dtype=xp.float64)
        start = (
            xp.zeros_like(stop)
            if start is None
            else xp.asarray(start, dtype=xp.float64)
        )
        if strata is None:
            strata = xp.zeros(stop.shape[0], dtype=xp.int64)
        else:
            strata_raw = xp.asarray(strata)
            if strata_raw.ndim != 1 or int(strata_raw.shape[0]) != int(stop.shape[0]):
                raise ValueError("strata must have shape (n_samples,)")
            kind = strata_raw.dtype.kind
            if kind not in "biuf":
                raise ValueError("strata must contain numeric integer-valued labels")
            if kind == "f":
                invalid = ~xp.isfinite(strata_raw) | (strata_raw != xp.rint(strata_raw))
                if _scalar_bool(xp.any(invalid)):
                    raise ValueError("strata must contain finite integer-valued labels")
            strata = strata_raw.astype(xp.int64, copy=False)
    _validate_counting_process_inputs(X, stop, event, start, strata)
    event = event.to(dtype=xp.int64) if backend == "torch" else event.astype(xp.int64)
    return X, stop, event, start, strata


def cox_counting_process_objective(
    beta: Any,
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
    ties: str = "efron",
    score_residuals: bool = False,
    compute_derivatives: bool = True,
) -> Dict[str, Any]:
    """Evaluate Cox partial log-likelihood, score, and information.

    Parameters use the counting-process convention ``(start, stop]``.  By
    default, the returned ``information`` is the positive-oriented observed
    information, i.e. ``-d2 loglik / d beta2``.  Set
    ``compute_derivatives=False`` for a log-likelihood-only result that avoids
    score vectors, p-by-p information matrices, and batched p-by-p moments.
    ``score_residuals`` returns conventional Breslow martingale score residuals
    for robust covariance estimation, including when the likelihood uses Efron
    ties.
    """
    ties = str(ties).lower()
    if ties not in {"breslow", "efron", "exact"}:
        raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
    compute_derivatives = bool(compute_derivatives)
    if score_residuals and not compute_derivatives:
        raise ValueError("score_residuals requires compute_derivatives=True")

    X, stop, event, start, strata = prepare_counting_process_inputs(
        X, stop, event, start=start, strata=strata
    )
    backend, xp = _array_namespace(X)
    beta = _as_backend_array(beta, backend, xp, X).reshape(-1)
    n_features = int(X.shape[1])
    if int(beta.shape[0]) != n_features:
        raise ValueError("beta must have shape (n_features,)")

    # A stratified Cox likelihood is invariant to an independent constant
    # covariate shift inside each stratum.  Center within strata before forming
    # raw second moments to prevent catastrophic cancellation for data such as
    # ``X_g = z_g +/- 1e10`` while preserving the exact objective.
    X_centered = _center_within_strata(X, strata, backend, xp)
    eta = X_centered @ beta
    if ties == "exact":
        nested_exact = _nested_exact_group_objective(
            eta,
            X_centered,
            stop,
            event,
            start,
            strata,
            score_residuals=score_residuals,
            compute_derivatives=compute_derivatives,
        )
        if nested_exact is not None:
            return nested_exact
        batched_exact = _batched_exact_group_objective(
            eta,
            X_centered,
            stop,
            event,
            start,
            strata,
            score_residuals=score_residuals,
            compute_derivatives=compute_derivatives,
        )
        if batched_exact is not None:
            return batched_exact
        stratified_exact = _stratified_exact_group_objective(
            eta,
            X_centered,
            stop,
            event,
            start,
            strata,
            score_residuals=score_residuals,
            compute_derivatives=compute_derivatives,
        )
        if stratified_exact is not None:
            return stratified_exact
    if ties != "exact":
        if backend == "numpy":
            return _numpy_group_objective(
                eta,
                X_centered,
                stop,
                event,
                start,
                strata,
                ties=ties,
                score_residuals=score_residuals,
                compute_derivatives=compute_derivatives,
            )
        return _batched_group_objective(
            eta,
            X_centered,
            stop,
            event,
            start,
            strata,
            ties=ties,
            score_residuals=score_residuals,
            compute_derivatives=compute_derivatives,
        )

    return _reference_exact_group_objective(
        eta,
        X_centered,
        stop,
        event,
        start,
        strata,
        score_residuals=score_residuals,
        compute_derivatives=compute_derivatives,
    )


def cox_baseline_hazard(
    beta: Any,
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
    ties: str = "efron",
) -> Dict[int, Dict[str, Any]]:
    """Compute stratum-specific baseline hazard increments on the input backend."""
    ties = str(ties).lower()
    if ties not in {"breslow", "efron", "exact"}:
        raise ValueError("ties must be 'breslow', 'efron', or 'exact'")
    X, stop, event, start, strata = prepare_counting_process_inputs(
        X, stop, event, start=start, strata=strata
    )
    backend, xp = _array_namespace(X)
    beta = _as_backend_array(beta, backend, xp, X).reshape(-1)
    output: Dict[int, Dict[str, Any]] = {}

    for stratum in _unique_sorted(strata, backend, xp):
        stratum_mask = strata == stratum
        X_s = X[stratum_mask]
        stop_s = stop[stratum_mask]
        event_s = event[stratum_mask]
        start_s = start[stratum_mask]
        n_stratum = int(X_s.shape[0])
        x_reference = _sum(X_s, backend, xp, axis=0) / float(n_stratum)
        eta = (X_s - x_reference.reshape(1, -1)) @ beta
        reference_linear_predictor = x_reference @ beta
        event_mask = event_s == 1
        event_times = stop_s[event_mask]
        if backend == "torch":
            failure_times, event_counts = xp.unique(
                event_times, sorted=True, return_counts=True
            )
        else:
            failure_times, event_counts = xp.unique(event_times, return_counts=True)
        n_groups = int(failure_times.shape[0])

        if n_groups == 0:
            empty = _zeros(backend, xp, (0,), X)
            output[_scalar_int(stratum)] = {
                "time": failure_times,
                "hazard": empty,
                "cumulative_hazard": empty,
                "log_hazard": empty,
                "log_cumulative_hazard": empty,
                "log_hazard_centered": empty,
                "log_cumulative_hazard_centered": empty,
                "x_reference": x_reference,
            }
            continue

        # Ordinary right-censored risk sets are nested.  A descending stop-time
        # order turns every denominator into a prefix, so one stable log-prefix
        # scan replaces the previous full risk-mask scan per failure group.
        use_prefix = not _scalar_bool(_sum(start_s != 0, backend, xp) > 0)
        if use_prefix and backend == "cupy":
            # CuPy does not implement logaddexp.accumulate.  A global shift is
            # safe in the normal predictor range; extreme ranges retain the
            # stable per-group path instead of underflowing an early prefix.
            eta_range = float((_max(eta, backend, xp) - xp.min(eta)).item())
            use_prefix = eta_range <= 500.0
        if use_prefix:
            if backend == "torch":
                order = xp.argsort(stop_s, descending=True)
            else:
                order = xp.argsort(-stop_s)
            sorted_stop = stop_s[order]
            sorted_eta = eta[order]
            if backend == "torch":
                log_risk_prefix = xp.logcumsumexp(sorted_eta, dim=0)
                risk_counts = xp.searchsorted(
                    -sorted_stop, -failure_times, right=True
                ).to(dtype=xp.int64)
            else:
                if backend == "cupy":
                    eta_shift = _max(sorted_eta, backend, xp)
                    log_risk_prefix = (
                        _log(xp.cumsum(_exp(sorted_eta - eta_shift, xp)), xp)
                        + eta_shift
                    )
                else:
                    log_risk_prefix = xp.logaddexp.accumulate(sorted_eta)
                risk_counts = xp.searchsorted(
                    -sorted_stop, -failure_times, side="right"
                ).astype(xp.int64, copy=False)
            if _scalar_bool(_sum(risk_counts < event_counts, backend, xp) > 0):
                raise FloatingPointError(
                    "baseline failure count exceeds its Cox risk set"
                )
            counts = _as_float(event_counts, backend, X)
            log_increments_centered = (
                _log(counts, xp) - log_risk_prefix[risk_counts - 1]
            )
            log_increments = log_increments_centered - reference_linear_predictor
            if backend == "torch":
                log_cumulative_centered = xp.logcumsumexp(
                    log_increments_centered, dim=0
                )
            elif backend == "cupy":
                increment_shift = _max(log_increments_centered, backend, xp)
                log_cumulative_centered = (
                    _log(
                        xp.cumsum(_exp(log_increments_centered - increment_shift, xp)),
                        xp,
                    )
                    + increment_shift
                )
            else:
                log_cumulative_centered = xp.logaddexp.accumulate(
                    log_increments_centered
                )
            log_cumulative = log_cumulative_centered - reference_linear_predictor
            output[_scalar_int(stratum)] = {
                "time": failure_times,
                "hazard": _exp_finite_float64(log_increments, backend, xp),
                "cumulative_hazard": _exp_finite_float64(log_cumulative, backend, xp),
                "log_hazard": log_increments,
                "log_cumulative_hazard": log_cumulative,
                "log_hazard_centered": log_increments_centered,
                "log_cumulative_hazard_centered": log_cumulative_centered,
                "x_reference": x_reference,
            }
            continue

        # Delayed entry breaks nested prefixes.  Retain the stable, backend-
        # native per-group implementation for the general counting-process case.
        increments = _zeros(backend, xp, (n_groups,), X)
        log_increments = _zeros(backend, xp, (n_groups,), X)
        log_cumulative = _zeros(backend, xp, (n_groups,), X)
        log_increments_centered = _zeros(backend, xp, (n_groups,), X)
        log_cumulative_centered = _zeros(backend, xp, (n_groups,), X)
        running_log_cumulative = _zeros(backend, xp, (), X)
        running_log_cumulative[...] = -float("inf")
        running_log_cumulative_centered = _zeros(backend, xp, (), X)
        running_log_cumulative_centered[...] = -float("inf")
        for group_idx, failure_time in enumerate(failure_times):
            fail_mask = event_mask & (stop_s == failure_time)
            risk_mask = (start_s < failure_time) & (stop_s >= failure_time)
            d = _scalar_int(_sum(fail_mask, backend, xp))
            eta_shift = _max(eta[risk_mask], backend, xp)
            s0 = _sum(_exp(eta[risk_mask] - eta_shift, xp), backend, xp)
            # Use the conventional Breslow baseline after Breslow, Efron, or
            # Exact coefficient estimation.  Tie handling affects beta, not
            # this baseline convention.
            log_increment_centered = float(np.log(float(d))) - eta_shift - _log(s0, xp)
            log_increment = log_increment_centered - reference_linear_predictor
            increments[group_idx] = _exp_finite_float64(log_increment, backend, xp)
            log_increments[group_idx] = log_increment
            log_increments_centered[group_idx] = log_increment_centered
            running_log_cumulative = xp.logaddexp(running_log_cumulative, log_increment)
            running_log_cumulative_centered = xp.logaddexp(
                running_log_cumulative_centered, log_increment_centered
            )
            log_cumulative[group_idx] = running_log_cumulative
            log_cumulative_centered[group_idx] = running_log_cumulative_centered

        output[_scalar_int(stratum)] = {
            "time": failure_times,
            "hazard": increments,
            "cumulative_hazard": _exp_finite_float64(log_cumulative, backend, xp),
            "log_hazard": log_increments,
            "log_cumulative_hazard": log_cumulative,
            "log_hazard_centered": log_increments_centered,
            "log_cumulative_hazard_centered": log_cumulative_centered,
            "x_reference": x_reference,
        }
    return output


def step_evaluate(times: Any, knots: Any, values: Any, *, left_value: float = 0.0):
    """Evaluate a right-continuous step function without changing backend."""
    backend, xp = _array_namespace(knots)
    times = _as_backend_array(times, backend, xp, knots)
    if int(knots.shape[0]) == 0:
        if backend == "torch":
            return xp.full(
                times.shape, left_value, dtype=values.dtype, device=values.device
            )
        return xp.full(times.shape, left_value, dtype=values.dtype)
    if backend == "torch":
        idx = xp.searchsorted(knots, times, right=True) - 1
        out = xp.full(times.shape, left_value, dtype=values.dtype, device=values.device)
    else:
        idx = xp.searchsorted(knots, times, side="right") - 1
        out = xp.full(times.shape, left_value, dtype=values.dtype)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def counting_process_concordance(
    beta: Any,
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
    subject_id: Optional[Any] = None,
):
    """Harrell-style concordance for right-censored counting-process rows.

    At each observed failure, the event row is compared with rows that are
    still at risk strictly beyond that failure time, plus censored rows ending
    at the same time.  Rows belonging to the same subject are never compared.
    """
    X, stop, event, start, strata = prepare_counting_process_inputs(
        X, stop, event, start=start, strata=strata
    )
    backend, xp = _array_namespace(X)
    beta = _as_backend_array(beta, backend, xp, X).reshape(-1)
    if subject_id is None:
        if backend == "torch":
            subject_id = xp.arange(X.shape[0], dtype=xp.int64, device=X.device)
        else:
            subject_id = xp.arange(X.shape[0], dtype=xp.int64)
    else:
        subject_id = _as_backend_array(
            subject_id, backend, xp, X, integer=True
        ).reshape(-1)
        if int(subject_id.shape[0]) != int(X.shape[0]):
            raise ValueError("subject_id must have shape (n_samples,)")

    X_centered = _center_within_strata(X, strata, backend, xp)
    risk_score = X_centered @ beta
    concordant = _zeros(backend, xp, (), X)
    tied = _zeros(backend, xp, (), X)
    permissible = _zeros(backend, xp, (), X)
    event_rows = _nonzero(event == 1, backend, xp)
    n_events = int(event_rows.shape[0])
    max_pair_entries = 2_000_000
    batch_size = max(1, min(n_events, max_pair_entries // max(int(X.shape[0]), 1)))
    for batch_start in range(0, n_events, batch_size):
        rows = event_rows[batch_start : batch_start + batch_size]
        failure_time = stop[rows].reshape(-1, 1)
        comparison = (
            (strata.reshape(1, -1) == strata[rows].reshape(-1, 1))
            & (start.reshape(1, -1) < failure_time)
            & (
                (stop.reshape(1, -1) > failure_time)
                | ((stop.reshape(1, -1) == failure_time) & (event.reshape(1, -1) == 0))
            )
            & (subject_id.reshape(1, -1) != subject_id[rows].reshape(-1, 1))
        )
        risk_i = risk_score[rows].reshape(-1, 1)
        risk_j = risk_score.reshape(1, -1)
        permissible = permissible + _sum(comparison, backend, xp)
        concordant = concordant + _sum(comparison & (risk_i > risk_j), backend, xp)
        tied = tied + _sum(comparison & (risk_i == risk_j), backend, xp)
    if _scalar_bool(permissible == 0):
        if backend == "torch":
            return xp.as_tensor(0.5, dtype=X.dtype, device=X.device)
        return xp.asarray(0.5, dtype=X.dtype)
    return (concordant + 0.5 * tied) / permissible
