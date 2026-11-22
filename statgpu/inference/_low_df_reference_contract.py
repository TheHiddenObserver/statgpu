"""Compatibility helpers for class-owned low-degree Student-t inference.

The regularized-incomplete-beta representation used by the generic Student-t
implementation can square an extreme statistic before evaluating the tail. For
df=1/2 that intermediate can overflow even when the final two-sided p-value is
representable. Reuse the stable reference formulas already maintained by
``two_sided_reference_inference`` for those two degrees of freedom so every
consumer, including exact/precomputed CuPy and Torch Gaussian inference, shares
the same numerical semantics.
"""

from __future__ import annotations

import numpy as np

from ._distributions_backend import TDistributionBase
from ._reference_distribution import two_sided_reference_inference


_LOW_DF = frozenset({1.0, 2.0})


def _backend_context(distribution, statistic=None):
    """Return backend namespace and concrete Torch device for a distribution."""
    sf = distribution._sf
    cp = getattr(sf, "_cp", None)
    if cp is not None:
        return "cupy", cp, None

    torch = getattr(sf, "_torch", None)
    if torch is not None:
        device = getattr(statistic, "device", None)
        if device is None:
            device = getattr(sf, "_device", None)
        return "torch", torch, None if device is None else str(device)

    return "numpy", np, None


def _statistic_abs_on_backend(statistic, *, backend, xp, device):
    """Normalize a statistic to float64 without changing accelerator ownership."""
    if backend == "torch":
        value = xp.as_tensor(statistic, dtype=xp.float64, device=device)
    else:
        value = xp.asarray(statistic, dtype=xp.float64)
    return xp.abs(value)


def _reference_scalar(*, backend, xp, device):
    """Create a scalar device reference for critical-value-only calls."""
    if backend == "torch":
        return xp.zeros((), dtype=xp.float64, device=device)
    return xp.asarray(0.0, dtype=xp.float64)


def _install_low_df_reference_contract() -> None:
    """No-op: ``TDistributionBase`` owns the stable low-df boundary directly."""
