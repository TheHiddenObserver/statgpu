"""Correctness-first safety boundary for experimental CoxPHCV screening.

The historical two-stage and successive-halving branches are activated only
through environment variables and can remove candidates before full-precision
evaluation. Until their candidate-ranking semantics are independently proven
and covered on both CUDA backends, every requested screening run is converted
into one ordinary exhaustive full-precision selector invocation.
"""

from __future__ import annotations

from functools import wraps
import os
import threading
import warnings

import numpy as np

from . import _cox_cv as _module


_TWO_STAGE_ENV = "STATGPU_COXPHCV_TWO_STAGE"
_HALVING_ENV = "STATGPU_COXPHCV_SUCCESSIVE_HALVING"
_STAGED_ENV_NAMES = frozenset({_TWO_STAGE_ENV, _HALVING_ENV})
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_ORIGINAL_SELECT_COXPH_PENALTY_CV = _module._select_coxph_penalty_cv
_STAGED_FALLBACK_LOCK = threading.RLock()


def _raw_staged_request_present():
    """Check the process environment without consulting a patched reader."""
    return any(
        str(os.environ.get(name, "")).strip().lower() in _TRUTHY_ENV_VALUES
        for name in _STAGED_ENV_NAMES
    )


def _requested_staged_controls():
    """Return the two experimental screening requests from the public env."""
    return (
        bool(_module._env_flag(_TWO_STAGE_ENV, False)),
        bool(_module._env_flag(_HALVING_ENV, False)),
    )


def _annotate_exhaustive_fallback(
    details,
    *,
    two_stage_requested,
    halving_requested,
):
    """Publish the requested-vs-effective screening contract."""
    annotated = dict(details)
    penalties = np.asarray(annotated.get("penalties", ()), dtype=np.float64)
    n_candidates = int(penalties.size)
    annotated.update(
        {
            "two_stage_requested": bool(two_stage_requested),
            "two_stage_enabled": False,
            "successive_halving_requested": bool(halving_requested),
            "successive_halving_enabled": False,
            "staged_execution_mode": "exhaustive_safety_fallback",
            "staged_safety_strategy": "single_pass_exhaustive",
            "staged_fallback_reason": (
                "experimental screening is disabled until deterministic "
                "candidate ranking and three-backend evidence are complete"
            ),
            "fast_pass_candidate_mask": np.zeros(n_candidates, dtype=bool),
            "full_precision_candidate_mask": np.ones(n_candidates, dtype=bool),
            "screened_out_candidate_mask": np.zeros(n_candidates, dtype=bool),
        }
    )
    return annotated


@wraps(_ORIGINAL_SELECT_COXPH_PENALTY_CV)
def _select_coxph_penalty_cv_with_staged_safety(*args, **kwargs):
    """Run one exhaustive full-precision selector when screening is requested."""
    # Ordinary exhaustive calls do not pay a global serialization cost. When
    # either process-wide switch is truthy, every selector enters the lock
    # before temporarily replacing the module-level staged flag reader.
    if not _raw_staged_request_present():
        return _ORIGINAL_SELECT_COXPH_PENALTY_CV(*args, **kwargs)

    requested_details = bool(kwargs.get("return_details", False))
    with _STAGED_FALLBACK_LOCK:
        two_stage_requested, halving_requested = _requested_staged_controls()
        if not (two_stage_requested or halving_requested):
            return _ORIGINAL_SELECT_COXPH_PENALTY_CV(*args, **kwargs)

        original_env_flag = _module._env_flag

        def exhaustive_env_flag(name, default=False):
            if name in _STAGED_ENV_NAMES:
                return False
            return original_env_flag(name, default)

        warnings.warn(
            "CoxPHCV two-stage/successive-halving screening is temporarily "
            "disabled for correctness; exhaustive full-precision CV over all "
            "candidates is used.",
            RuntimeWarning,
            stacklevel=2,
        )
        _module._env_flag = exhaustive_env_flag
        try:
            if requested_details:
                best_penalty, details = _ORIGINAL_SELECT_COXPH_PENALTY_CV(
                    *args, **kwargs
                )
            else:
                forwarded = dict(kwargs)
                forwarded["return_details"] = True
                best_penalty, details = _ORIGINAL_SELECT_COXPH_PENALTY_CV(
                    *args, **forwarded
                )
        finally:
            _module._env_flag = original_env_flag

    details = _annotate_exhaustive_fallback(
        details,
        two_stage_requested=two_stage_requested,
        halving_requested=halving_requested,
    )
    if requested_details:
        return float(best_penalty), details
    return float(best_penalty)


_module._select_coxph_penalty_cv = _select_coxph_penalty_cv_with_staged_safety

_STAGED_DOC = """

    Experimental screening safety
    -----------------------------
    The environment-controlled two-stage and successive-halving optimizations
    currently fall back to one exhaustive full-precision CV pass over all
    candidates on every backend. A ``RuntimeWarning`` is emitted and
    ``cv_results_`` records the requested and effective modes plus candidate
    masks. This prevents approximate screening from silently changing the
    selected penalty or doubling the complete CuPy candidate path.
"""
if _STAGED_DOC.strip() not in (_module.CoxPHCV.__doc__ or ""):
    _module.CoxPHCV.__doc__ = (_module.CoxPHCV.__doc__ or "") + _STAGED_DOC


__all__ = [
    "_annotate_exhaustive_fallback",
    "_raw_staged_request_present",
    "_select_coxph_penalty_cv_with_staged_safety",
]
