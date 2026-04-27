"""
Formula term helpers and custom evaluation environments.

Patsy natively supports R-style formula terms:

- ``C(var)`` — treat as categorical (one-hot encoding)
- ``np.func(var)`` — apply numpy function (e.g. ``np.log(x)``)
- ``x1:x2`` — interaction only
- ``x1*x2`` — main effects + interaction
- ``x1 + x2`` — additive
- ``x1 + x2 - 1`` — additive without intercept
- ``np.log(x)`` — transformations

This module provides helper functions for constructing custom
patsy evaluation environments, needed for model-specific syntax
like ``Surv(time, event)`` in Cox PH models.
"""

from typing import Dict, Any, Optional

import numpy as np


def _surv(time, event):
    """Survival function for patsy formula parsing.

    Mimics R's survival::Surv() function for use in patsy formulas::

        "Surv(time, event) ~ x1 + x2"

    Parameters
    ----------
    time : array-like
        Survival/follow-up times.
    event : array-like
        Event indicator (1 = event occurred, 0 = censored).

    Returns
    -------
    result : ndarray of shape (n, 2)
        Column 0: time, Column 1: event.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()

    if len(time) != len(event):
        raise ValueError(
            f"time ({len(time)} elements) and event ({len(event)} elements) "
            "must have the same length."
        )

    return np.column_stack([time, event])


def make_surv_env() -> Dict[str, Any]:
    """Create a patsy evaluation environment with ``Surv`` function.

    Returns
    -------
    env : dict
        Custom functions for patsy's ``EvalEnvironment``.

    Examples
    --------
    >>> from statgpu.core.formula._terms import make_surv_env
    >>> import patsy
    >>> env = make_surv_env()
    >>> # Then pass env to patsy.dmatrices or dmatrix
    """
    return {"Surv": _surv}
