"""Backend-native public scoring adapter for :class:`CoxPH`.

The numerical Cox implementation is intentionally kept in ``_cox.py``.  This
module owns the public packed-target boundary so a CuPy or Torch target is not
materialized on the host merely because ``event`` was supplied inside a two- or
three-column survival array.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar


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

        if strata is None:
            fitted_n_strata = (
                1
                if self._strata is None
                else int(
                    np.unique(np.asarray(self._to_numpy(self._strata))).shape[0]
                )
            )
            if fitted_n_strata > 1:
                raise ValueError(
                    "strata is required when scoring a stratified CoxPH fit"
                )
            strata_codes = None
        elif self._strata_labels is not None:
            mapping = {
                value: idx
                for idx, value in enumerate(self._strata_labels.tolist())
            }
            try:
                codes = np.asarray(
                    [
                        mapping[value]
                        for value in np.asarray(self._to_numpy(strata)).tolist()
                    ],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise ValueError(
                    f"unknown scoring stratum: {exc.args[0]!r}"
                ) from exc
            strata_codes = backend.asarray(codes, dtype=backend.int64)
        else:
            strata_codes, _ = self._encode_group_labels(
                strata, n_samples, "strata"
            )
        subject_codes, _ = self._encode_group_labels(
            subject_id, n_samples, "subject_id"
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

    concordant = permissible = tied_risk = 0.0
    chunk_size = max(1, min(n_events, int(128e6 / max(n_samples, 1))))
    for batch_start in range(0, n_events, chunk_size):
        batch_end = min(batch_start + chunk_size, n_events)
        idx = event_idx[batch_start:batch_end]
        time_i = time_arr[idx, None]
        risk_i = risk_score[idx, None]
        perm = (time_i < time_arr[None, :]) | (
            (time_i == time_arr[None, :]) & (event_arr[None, :] == 0)
        )
        rows = backend.arange(batch_end - batch_start, dtype=backend.int64)
        perm[rows, idx] = False
        concordant += _to_float_scalar(
            xp.sum(perm & (risk_i > risk_score[None, :]))
        )
        tied_risk += _to_float_scalar(
            xp.sum(perm & (risk_i == risk_score[None, :]))
        )
        permissible += _to_float_scalar(xp.sum(perm))

    if permissible <= 0:
        return float("nan")
    return float((concordant + 0.5 * tied_risk) / permissible)


def install(CoxPH):
    """Install the reviewed public score boundary on the existing class object."""
    score.__name__ = "score"
    score.__qualname__ = "CoxPH.score"
    score.__module__ = CoxPH.__module__
    CoxPH.score = score
    return CoxPH


__all__ = ["install", "score"]
