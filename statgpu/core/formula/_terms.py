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

from typing import Any, Dict

import numpy as np


def _surv(*args):
    """Survival function for patsy formula parsing.

    Mimics R's survival::Surv() function for use in patsy formulas::

        "Surv(time, event) ~ x1 + x2"
        "Surv(start, stop, event) ~ x1 + x2"

    Parameters
    ----------
    *args : tuple of array-like
        Either ``(time, event)`` for right-censored data or
        ``(start, stop, event)`` for counting-process data.  Counting-process
        rows follow the R convention ``(start, stop]``.

    Returns
    -------
    result : ndarray of shape (n, 2) or (n, 3)
        ``[time, event]`` or ``[start, stop, event]``.
    """
    if len(args) not in (2, 3):
        raise TypeError(
            "Surv expects Surv(time, event) or Surv(start, stop, event)"
        )
    columns = [np.asarray(value, dtype=np.float64).ravel() for value in args]
    lengths = {len(value) for value in columns}
    if len(lengths) != 1:
        raise ValueError("all Surv arguments must have the same length")
    if len(columns) == 3:
        start, stop, event = columns
        if np.any(start < 0) or np.any(stop <= start):
            raise ValueError("Surv(start, stop, event) requires 0 <= start < stop")
    else:
        _, event = columns
    if np.any((event != 0) & (event != 1)):
        raise ValueError("Surv event must contain only 0/1 values")
    return np.column_stack(columns)


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
    return {"Surv": _surv, "np": np}
