"""Backend-native public scoring adapter for :class:`CoxPH`.

The numerical Cox implementation is intentionally kept in ``_cox.py``.  This
module owns the public packed-target boundary so a CuPy or Torch target is not
materialized on the host merely because ``event`` was supplied inside a two- or
three-column survival array.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar
from statgpu.backends._array_ops import _sync_scalars
from statgpu.backends._utils import _require_real_array
from statgpu.survival._concordance import (
    MAX_CONCORDANCE_PAIR_ENTRIES,
    concordance_tile_shape,
)


_MAX_CONCORDANCE_PAIR_ENTRIES = MAX_CONCORDANCE_PAIR_ENTRIES
_concordance_tile_shape = concordance_tile_shape


def score(
    self,
    X,
    time,
    event=None,
    start=None,
    strata=None,
    subject_id=None,
):
    """Compute a backend-native Harrell-style concordance index."""
    self._check_is_fitted()
    _require_real_array(X, "X")
    _require_real_array(start, "start")
    if event is None:
        _require_real_array(time, "packed survival target")
    else:
        _require_real_array(time, "time")
        _require_real_array(event, "event")
    X_arr, backend, coef = self._prepare_prediction_X(X)
    xp = backend.xp
    n_samples = int(X_arr.shape[0])

    if event is None:
        target = backend.asarray(time, dtype=backend.float64)
        if target.ndim != 2 or int(target.shape[1]) not in (2, 3):
            raise ValueError(
                "packed survival targets require [time, event] or "
                "[start, stop, event]"
            )
        if int(target.shape[0]) != n_samples:
            raise ValueError(
                "X and packed survival target must contain the same number of rows"
            )
        if int(target.shape[1]) == 2:
            time, event = target[:, 0], target[:, 1]
        else:
            if start is not None:
                raise ValueError(
                    "start is already present in the packed survival target"
                )
            start, time, event = target[:, 0], target[:, 1], target[:, 2]

    time_arr = backend.asarray(time, dtype=backend.float64)
    event_raw = backend.asarray(event, dtype=backend.float64)
    if time_arr.ndim != 1:
        raise ValueError("time must have shape (n_samples,)")
    if int(time_arr.shape[0]) != n_samples:
        raise ValueError("X, time, and event must contain the same number of rows")
    if event_raw.ndim != 1 or int(event_raw.shape[0]) != n_samples:
        raise ValueError("event must have shape (n_samples,)")
    if not bool(_to_float_scalar(xp.all(xp.isfinite(time_arr)))) or bool(
        _to_float_scalar(xp.any(time_arr <= 0))
    ):
        raise ValueError("time must contain only positive finite values")
    if not bool(_to_float_scalar(xp.all(xp.isfinite(event_raw)))) or bool(
        _to_float_scalar(xp.any((event_raw != 0) & (event_raw != 1)))
    ):
        raise ValueError("event must contain only 0/1 finite values")
    event_arr = backend.asarray(event_raw, dtype=backend.int64)

    use_counting = (
        self._strata is not None
        or self._is_counting_process
        or start is not None
        or strata is not None
        or subject_id is not None
    )
    if use_counting:
        from statgpu.survival._risk_sets import counting_process_concordance

        fitted_n_strata = (
            1
            if self._strata is None
            else int(
                np.unique(np.asarray(self._to_numpy(self._strata))).shape[0]
            )
        )
        strata_codes = self._encode_prediction_strata(
            strata,
            n_samples=n_samples,
            backend=backend,
            context="scoring",
            required=fitted_n_strata > 1,
        )
        subject_codes, _ = self._encode_group_labels(
            subject_id, n_samples, "subject_id", return_labels=False
        )
        start_arr = (
            None
            if start is None
            else backend.asarray(start, dtype=backend.float64)
        )
        value = counting_process_concordance(
            coef,
            X_arr,
            time_arr,
            event_arr,
            start=start_arr,
            strata=strata_codes,
            subject_id=subject_codes,
        )
        return float(_to_float_scalar(value))

    risk_score = X_arr @ coef
    event_idx = xp.where(event_arr == 1)[0]
    n_events = int(event_idx.shape[0])
    if n_events == 0:
        return 0.5

    concordant = backend.zeros((), dtype=backend.float64)
    tied_risk = backend.zeros((), dtype=backend.float64)
    permissible = backend.zeros((), dtype=backend.float64)
    event_tile, sample_tile = _concordance_tile_shape(n_events, n_samples)
    for batch_start in range(0, n_events, event_tile):
        batch_end = min(batch_start + event_tile, n_events)
        idx = event_idx[batch_start:batch_end]
        time_i = time_arr[idx, None]
        risk_i = risk_score[idx, None]
        for sample_start in range(0, n_samples, sample_tile):
            sample_end = min(sample_start + sample_tile, n_samples)
            time_j = time_arr[None, sample_start:sample_end]
            risk_j = risk_score[None, sample_start:sample_end]
            event_j = event_arr[None, sample_start:sample_end]
            sample_idx = backend.arange(
                sample_start, sample_end, dtype=backend.int64
            )
            perm = (
                (time_i < time_j)
                | ((time_i == time_j) & (event_j == 0))
            ) & (idx[:, None] != sample_idx[None, :])
            concordant = concordant + xp.sum(perm & (risk_i > risk_j))
            tied_risk = tied_risk + xp.sum(perm & (risk_i == risk_j))
            permissible = permissible + xp.sum(perm)

    concordant, tied_risk, permissible = _sync_scalars(
        concordant,
        tied_risk,
        permissible,
        backend=backend.name,
    )
    if permissible <= 0:
        return 0.5
    return float((concordant + 0.5 * tied_risk) / permissible)


__all__ = ["score"]
