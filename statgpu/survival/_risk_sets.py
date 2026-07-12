"""Backend-native Cox counting-process risk-set primitives.

This module is the correctness reference for delayed entry, start/stop data,
and stratified Cox models.  It deliberately keeps the statistical definition
in one place for NumPy, CuPy, and Torch.  Specialized no-entry kernels may be
faster, but they must agree with these primitives.

The counting-process convention matches R's ``Surv(start, stop, event)``:
rows are at risk on the half-open interval ``(start, stop]``.
"""

from __future__ import annotations

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


def _center_within_strata(X: Any, strata: Any, backend: str, xp: Any):
    """Center covariates by stratum on their existing backend."""
    centered = _zeros(backend, xp, tuple(X.shape), X)
    for stratum in _unique_sorted(strata, backend, xp):
        rows = strata == stratum
        n_rows = _scalar_int(_sum(rows, backend, xp))
        reference = _sum(X[rows], backend, xp, axis=0) / float(n_rows)
        centered[rows] = X[rows] - reference.reshape(1, -1)
    return centered


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


def _exact_tie_log_partition_moments(
    X_risk: Any,
    log_w_risk: Any,
    d: int,
    backend: str,
    xp: Any,
):
    """Stable elementary-symmetric DP for an exact tied-event group.

    Returns ``(log_Z, E[S], E[S S'])`` for the weighted distribution over all
    size-``d`` subsets, where ``S`` is the subset covariate sum.  Maintaining
    normalized moments and ``log_Z`` avoids overflow from combinatorial counts
    such as ``choose(1100, 550)``.  Descending subset-size updates ensure each
    risk-set row is used at most once.
    """
    n_risk, n_features = int(X_risk.shape[0]), int(X_risk.shape[1])
    if d > n_risk:
        raise ValueError("number of tied events cannot exceed the risk-set size")
    log_z = _zeros(backend, xp, (d + 1,), X_risk)
    log_z[1:] = -float("inf")
    mean = _zeros(backend, xp, (d + 1, n_features), X_risk)
    second = _zeros(backend, xp, (d + 1, n_features, n_features), X_risk)
    for row in range(n_risk):
        x = X_risk[row]
        log_weight = log_w_risk[row]
        outer_x = _outer(x, x, backend, xp)
        for subset_size in range(min(d, row + 1), 0, -1):
            old_log_z = log_z[subset_size]
            added_log_z = log_weight + log_z[subset_size - 1]
            new_log_z = xp.logaddexp(old_log_z, added_log_z)
            old_weight = _exp(old_log_z - new_log_z, xp)
            added_weight = _exp(added_log_z - new_log_z, xp)
            previous_mean = mean[subset_size - 1]
            added_mean = previous_mean + x
            added_second = (
                second[subset_size - 1]
                + _outer(previous_mean, x, backend, xp)
                + _outer(x, previous_mean, backend, xp)
                + outer_x
            )
            mean[subset_size] = (
                old_weight * mean[subset_size] + added_weight * added_mean
            )
            second[subset_size] = (
                old_weight * second[subset_size] + added_weight * added_second
            )
            log_z[subset_size] = new_log_z
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
    for row in range(n_risk):
        log_weight = log_w_risk[row]
        for subset_size in range(min(d, row + 1), 0, -1):
            log_z[subset_size] = xp.logaddexp(
                log_z[subset_size], log_weight + log_z[subset_size - 1]
            )
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
        strata = (
            xp.zeros(stop.shape[0], dtype=xp.int64, device=X.device)
            if strata is None
            else xp.as_tensor(strata, dtype=xp.int64, device=X.device)
        )
    else:
        X = xp.asarray(X, dtype=xp.float64)
        stop = xp.asarray(stop, dtype=xp.float64)
        event = xp.asarray(event, dtype=xp.float64)
        start = (
            xp.zeros_like(stop)
            if start is None
            else xp.asarray(start, dtype=xp.float64)
        )
        strata = (
            xp.zeros(stop.shape[0], dtype=xp.int64)
            if strata is None
            else xp.asarray(strata, dtype=xp.int64)
        )
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
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if int(beta.shape[0]) != n_features:
        raise ValueError("beta must have shape (n_features,)")

    # A stratified Cox likelihood is invariant to an independent constant
    # covariate shift inside each stratum.  Center within strata before forming
    # raw second moments to prevent catastrophic cancellation for data such as
    # ``X_g = z_g +/- 1e10`` while preserving the exact objective.
    X_centered = _center_within_strata(X, strata, backend, xp)
    eta = X_centered @ beta
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

    unique_strata = _unique_sorted(strata, backend, xp)
    for stratum in unique_strata:
        stratum_mask = strata == stratum
        event_mask_s = stratum_mask & (event == 1)
        failure_times = _unique_sorted(stop[event_mask_s], backend, xp)
        if int(failure_times.shape[0]) == 0:
            continue

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
                X_risk = X_centered[risk_idx]
                X_fail = X_centered[fail_idx]
                score = score + _sum(X_fail, backend, xp, axis=0)
                if residuals is not None:
                    residuals[fail_idx] = residuals[fail_idx] + X_fail
                (
                    log_partition,
                    exact_mean,
                    exact_second,
                ) = _exact_tie_log_partition_moments(X_risk, log_w_risk, d, backend, xp)
            else:
                log_partition = _exact_tie_log_partition(log_w_risk, d, backend, xp)
            if _scalar_bool(~xp.isfinite(log_partition)):
                raise FloatingPointError("non-finite exact Cox tie log-partition")
            loglik = loglik - (log_partition + float(d) * eta_shift)
            if compute_derivatives:
                score = score - exact_mean
                information = information + (
                    exact_second - _outer(exact_mean, exact_mean, backend, xp)
                )
                if residuals is not None:
                    # The exact score is additive but individual conditional
                    # inclusion probabilities require another DP pass.  Preserve
                    # the exact row-sum contract with an equal risk-set allocation.
                    # Cluster-robust inference for exact ties is rejected by the
                    # estimator until exact inclusion probabilities are exposed.
                    allocation = exact_mean / float(risk_idx.shape[0])
                    residuals[risk_idx] = residuals[risk_idx] - allocation

    result = {"log_likelihood": loglik}
    if compute_derivatives:
        result["score"] = score
        result["information"] = 0.5 * (information + information.T)
    if residuals is not None:
        result["score_residuals"] = residuals
    return result


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
        n_stratum = _scalar_int(_sum(stratum_mask, backend, xp))
        x_reference = _sum(X[stratum_mask], backend, xp, axis=0) / float(n_stratum)
        eta = (X - x_reference.reshape(1, -1)) @ beta
        reference_linear_predictor = x_reference @ beta
        event_mask_s = stratum_mask & (event == 1)
        failure_times = _unique_sorted(stop[event_mask_s], backend, xp)
        increments = _zeros(backend, xp, (int(failure_times.shape[0]),), X)
        log_increments = _zeros(backend, xp, (int(failure_times.shape[0]),), X)
        log_cumulative = _zeros(backend, xp, (int(failure_times.shape[0]),), X)
        log_increments_centered = _zeros(backend, xp, (int(failure_times.shape[0]),), X)
        log_cumulative_centered = _zeros(backend, xp, (int(failure_times.shape[0]),), X)
        if int(failure_times.shape[0]) == 0:
            output[_scalar_int(stratum)] = {
                "time": failure_times,
                "hazard": increments,
                "cumulative_hazard": increments,
                "log_hazard": log_increments,
                "log_cumulative_hazard": log_cumulative,
                "log_hazard_centered": log_increments_centered,
                "log_cumulative_hazard_centered": log_cumulative_centered,
                "x_reference": x_reference,
            }
            continue

        running_log_cumulative = _zeros(backend, xp, (), X)
        running_log_cumulative[...] = -float("inf")
        running_log_cumulative_centered = _zeros(backend, xp, (), X)
        running_log_cumulative_centered[...] = -float("inf")
        for group_idx, failure_time in enumerate(failure_times):
            fail_mask = event_mask_s & (stop == failure_time)
            risk_mask = stratum_mask & (start < failure_time) & (stop >= failure_time)
            d = _scalar_int(_sum(fail_mask, backend, xp))
            eta_shift = _max(eta[risk_mask], backend, xp)
            s0 = _sum(_exp(eta[risk_mask] - eta_shift, xp), backend, xp)
            # Use the conventional Breslow baseline after Breslow, Efron, or
            # Exact coefficient estimation.  This matches the legacy CoxPH
            # prediction path and common external APIs; tie handling affects
            # beta, not this baseline convention.
            log_increment_centered = float(np.log(float(d))) - eta_shift - _log(s0, xp)
            log_increment = log_increment_centered - reference_linear_predictor
            increment = _exp_finite_float64(log_increment, backend, xp)
            increments[group_idx] = increment
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
