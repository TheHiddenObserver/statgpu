"""GPU-accelerated effect size measures for ANOVA.

Provides :func:`cohens_f` and :func:`partial_eta_squared`.
"""

from __future__ import annotations

__all__ = ["cohens_f", "partial_eta_squared"]

from typing import Any

import numpy as np

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar


def cohens_f(
    *groups: Any,
    backend: str = "auto",
    dtype: Any = None,
) -> float:
    """Compute Cohen's f effect size from group data.

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays, one per group.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.
    dtype : dtype or None, default=None
        Float dtype for computation.

    Returns
    -------
    float
        Cohen's f = sqrt(eta_squared / (1 - eta_squared)).

    Notes
    -----
    Cohen's f is related to eta-squared by:
        f = sqrt(eta^2 / (1 - eta^2))

    Small: 0.10, Medium: 0.25, Large: 0.40 (Cohen 1988).
    """
    from statgpu.anova._oneway import f_oneway
    result = f_oneway(*groups, backend=backend, dtype=dtype)
    eta2 = result.eta_squared
    if eta2 >= 1.0:
        return float("inf")
    if np.isnan(eta2):
        return float("nan")
    return float(np.sqrt(eta2 / (1 - eta2)))


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> float:
    """Compute partial eta-squared from sum of squares.

    Parameters
    ----------
    ss_effect : float
        Sum of squares for the effect of interest.
    ss_error : float
        Sum of squares for the error term.

    Returns
    -------
    float
        Partial eta-squared = ss_effect / (ss_effect + ss_error).

    Notes
    -----
    Partial eta-squared is defined as:
        eta_p^2 = SS_effect / (SS_effect + SS_error)

    This is equivalent to eta-squared in one-way ANOVA but differs
    in multi-factor designs where SS_error is the residual SS.
    """
    try:
        ss_effect = float(ss_effect)
        ss_error = float(ss_error)
    except (TypeError, ValueError) as exc:
        raise TypeError("ss_effect and ss_error must be real scalars") from exc
    if not np.isfinite(ss_effect) or not np.isfinite(ss_error):
        raise ValueError("ss_effect and ss_error must be finite")
    if ss_effect < 0.0 or ss_error < 0.0:
        raise ValueError("sum-of-squares inputs must be non-negative")
    total = ss_effect + ss_error
    if total == 0.0:
        return float("nan")
    return float(ss_effect / total)
