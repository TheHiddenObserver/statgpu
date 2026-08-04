"""Correctness-first safety boundary for experimental CoxPHCV screening.

The historical two-stage and successive-halving branches are activated only
through environment variables and can remove candidates before full-precision
evaluation. Until their candidate-ranking semantics are independently proven
and covered on both CUDA backends, every requested screening run is converted
into an all-candidate full-precision run.
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
_COARSE_ENV = "STATGPU_COXPHCV_TWO_STAGE_COARSE"
_WINDOW_ENV = "STATGPU_COXPHCV_TWO_STAGE_WINDOW"
_TOPK_ENV = "STATGPU_COXPHCV_HALVING_TOPK"
_FAST_ITER_ENV = "STATGPU_COXPHCV_HALVING_FAST_ITER"
_FAST_TOL_ENV = "STATGPU_COXPHCV_HALVING_FAST_TOL"
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


def _explicit_cupy_request(kwargs):
    """Return whether the selector was explicitly routed to the CuPy backend."""
    device = kwargs.get("device", "cpu")
    device_name = getattr(device, "value", device)
    return str(device_name).lower() in {"cuda", "cupy"}


def _candidate_count(kwargs):
    """Read the candidate count without copying a backend array to the host."""
    penalties = kwargs.get("penalties")
    if penalties is not None:
        shape = getattr(penalties, "shape", None)
        if shape is not None and len(shape) == 1:
            return int(shape[0])
        try:
            return int(len(penalties))
        except TypeError:
            pass
    return int(kwargs.get("n_penalties", 100))


def _annotate_exhaustive_fallback(
    details,
    *,
    two_stage_requested,
    halving_requested,
    fallback_strategy,
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
            "staged_safety_strategy": str(fallback_strategy),
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
    """Run all candidates at full precision when screening is requested."""
    # Ordinary exhaustive calls do not pay a global serialization cost. When
    # either process-wide switch is truthy, every selector first enters the
    # lock before reading or temporarily replacing module-level env readers.
    if not _raw_staged_request_present():
        return _ORIGINAL_SELECT_COXPH_PENALTY_CV(*args, **kwargs)

    requested_details = bool(kwargs.get("return_details", False))
    with _STAGED_FALLBACK_LOCK:
        two_stage_requested, halving_requested = _requested_staged_controls()
        if not (two_stage_requested or halving_requested):
            return _ORIGINAL_SELECT_COXPH_PENALTY_CV(*args, **kwargs)

        original_env_flag = _module._env_flag
        original_env_int = _module._env_int
        original_env_float = _module._env_float
        explicit_cupy = _explicit_cupy_request(kwargs)
        n_candidates = _candidate_count(kwargs)
        max_iter = int(kwargs.get("max_iter", 100))
        tol = float(kwargs.get("tol", 1e-9))

        def exhaustive_env_flag(name, default=False):
            if not explicit_cupy and name in _STAGED_ENV_NAMES:
                return False
            return original_env_flag(name, default)

        def full_candidate_env_int(
            name,
            default,
            *,
            min_value=None,
            max_value=None,
        ):
            if explicit_cupy and name in {_COARSE_ENV, _WINDOW_ENV, _TOPK_ENV}:
                return n_candidates
            if explicit_cupy and name == _FAST_ITER_ENV:
                return max_iter
            return original_env_int(
                name,
                default,
                min_value=min_value,
                max_value=max_value,
            )

        def full_precision_env_float(name, default, *, min_value=None):
            if explicit_cupy and name == _FAST_TOL_ENV:
                return tol
            return original_env_float(
                name,
                default,
                min_value=min_value,
            )

        warnings.warn(
            "CoxPHCV two-stage/successive-halving screening is temporarily "
            "disabled for correctness; all candidates are evaluated at full "
            "precision.",
            RuntimeWarning,
            stacklevel=2,
        )
        _module._env_flag = exhaustive_env_flag
        _module._env_int = full_candidate_env_int
        _module._env_float = full_precision_env_float
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
            _module._env_int = original_env_int
            _module._env_float = original_env_float

    details = _annotate_exhaustive_fallback(
        details,
        two_stage_requested=two_stage_requested,
        halving_requested=halving_requested,
        fallback_strategy=(
            "full_candidate_staged_machinery"
            if explicit_cupy
            else "single_pass_exhaustive"
        ),
    )
    if requested_details:
        return float(best_penalty), details
    return float(best_penalty)


_module._select_coxph_penalty_cv = _select_coxph_penalty_cv_with_staged_safety

_STAGED_DOC = """

    Experimental screening safety
    -----------------------------
    The environment-controlled two-stage and successive-halving optimizations
    currently evaluate every candidate at full precision on every backend. A
    ``RuntimeWarning`` is emitted and ``cv_results_`` records the requested and
    effective modes plus candidate masks. This prevents approximate screening
    from silently changing the selected penalty. Explicit CuPy runs retain the
    staged fold-workspace machinery with all candidate sets expanded to the
    complete grid; CPU and Torch use a single exhaustive pass.
"""
if _STAGED_DOC.strip() not in (_module.CoxPHCV.__doc__ or ""):
    _module.CoxPHCV.__doc__ = (_module.CoxPHCV.__doc__ or "") + _STAGED_DOC


__all__ = [
    "_annotate_exhaustive_fallback",
    "_candidate_count",
    "_explicit_cupy_request",
    "_raw_staged_request_present",
    "_select_coxph_penalty_cv_with_staged_safety",
]
