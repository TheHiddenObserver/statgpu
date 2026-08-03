"""Order-invariant public penalty-grid boundary for :mod:`._cox_cv`."""

from __future__ import annotations

from functools import wraps

import numpy as np

from statgpu.cross_validation._grid_validation import coerce_real_numeric_grid

from . import _cox_cv as _module


_ORIGINAL_SELECT_COXPH_PENALTY_CV = _module._select_coxph_penalty_cv
_CANDIDATE_AXIS_KEYS = (
    "pl_path",
    "mean_pl",
    "converged_path",
    "convergence",
    "attempted_path",
    "iterations_path",
    "failure_path",
    "effective_fold_counts",
    "candidate_complete",
)


def _descending_penalty_order(penalties: np.ndarray) -> np.ndarray:
    """Return a stable strongest-to-weakest regularization order."""
    return np.argsort(-np.asarray(penalties, dtype=np.float64), kind="stable")


def _restore_original_candidate_order(value, descending_order):
    """Map a candidate-axis result from rank order back to caller order."""
    array = np.asarray(value)
    if array.ndim < 1 or int(array.shape[0]) != int(len(descending_order)):
        return value
    restored = np.empty_like(array)
    restored[descending_order, ...] = array
    return restored


def _prefer_stronger_near_tie(details, sorted_penalties):
    """Resolve numerically tied CV scores independently of input ordering."""
    mean_pl = np.asarray(details.get("mean_pl"), dtype=np.float64)
    complete = np.asarray(
        details.get("candidate_complete", np.isfinite(mean_pl)), dtype=bool
    )
    eligible = complete & np.isfinite(mean_pl)
    if mean_pl.ndim != 1 or mean_pl.size != sorted_penalties.size or not np.any(eligible):
        return float(details["penalty"])

    best = float(np.max(mean_pl[eligible]))
    tolerance = max(1e-12, abs(best) * 1e-10)
    candidates = np.flatnonzero(eligible & (mean_pl >= best - tolerance))
    selected = int(candidates[np.argmax(sorted_penalties[candidates])])
    details["penalty"] = float(sorted_penalties[selected])
    details["best_pl"] = float(mean_pl[selected])
    return float(sorted_penalties[selected])


@wraps(_ORIGINAL_SELECT_COXPH_PENALTY_CV)
def _select_coxph_penalty_cv_order_invariant(*args, **kwargs):
    """Run continuation and staged screening in penalty-rank order.

    Public diagnostics retain the exact user-supplied grid order. Internally,
    every custom grid is stably sorted from strongest to weakest penalty before
    warm starts, coarse screening, halving, and neighborhood refinement. This
    makes a permutation of the same grid an equivalent CV problem.
    """
    supplied = kwargs.get("penalties")
    if supplied is None:
        return _ORIGINAL_SELECT_COXPH_PENALTY_CV(*args, **kwargs)

    original_penalties = coerce_real_numeric_grid(supplied, name="penalties")
    descending_order = _descending_penalty_order(original_penalties)
    sorted_penalties = original_penalties[descending_order]
    forwarded = dict(kwargs)
    forwarded["penalties"] = sorted_penalties

    result = _ORIGINAL_SELECT_COXPH_PENALTY_CV(*args, **forwarded)
    if not bool(forwarded.get("return_details", False)):
        return result

    best_penalty, details = result
    details = dict(details)
    best_penalty = _prefer_stronger_near_tie(details, sorted_penalties)
    for key in _CANDIDATE_AXIS_KEYS:
        if key in details:
            details[key] = _restore_original_candidate_order(
                details[key], descending_order
            )
    details["penalties"] = original_penalties.copy()
    details["penalty_evaluation_order"] = sorted_penalties.copy()
    details["penalty_input_order_preserved"] = True
    details["penalty"] = float(best_penalty)
    return float(best_penalty), details


_module._select_coxph_penalty_cv = _select_coxph_penalty_cv_order_invariant


__all__ = [
    "_descending_penalty_order",
    "_restore_original_candidate_order",
    "_select_coxph_penalty_cv_order_invariant",
]
