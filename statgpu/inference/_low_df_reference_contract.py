"""Stable low-degree Student-t two-sided inference contract.

The regularized-incomplete-beta representation used by the generic Student-t
implementation can square an extreme statistic before evaluating the tail. For
df=1/2 that intermediate can overflow even when the final two-sided p-value is
representable. Reuse the stable reference formulas already maintained by
``two_sided_reference_inference`` for those two degrees of freedom so every
consumer, including exact/precomputed CuPy and Torch Gaussian inference, shares
the same numerical semantics.
"""

from __future__ import annotations

import functools

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
    """Route df=1/2 Student-t p-values and critical values through stable formulas."""
    current_pvalue = TDistributionBase.two_sided_pvalue
    current_critical = TDistributionBase.two_sided_critical_value
    if getattr(current_pvalue, "_statgpu_low_df_reference", False):
        return

    @functools.wraps(current_pvalue)
    def _two_sided_pvalue_stable(self, stat_abs, df):
        df_f = float(df)
        if df_f not in _LOW_DF:
            return current_pvalue(self, stat_abs, df)

        backend, xp, device = _backend_context(self, stat_abs)
        statistic_abs = _statistic_abs_on_backend(
            stat_abs,
            backend=backend,
            xp=xp,
            device=device,
        )
        pvalues, _ = two_sided_reference_inference(
            statistic_abs,
            distribution="t",
            alpha=0.05,
            backend=backend,
            xp=xp,
            df=df_f,
            device=device,
        )
        return pvalues

    @functools.wraps(current_critical)
    def _two_sided_critical_value_stable(
        self,
        alpha,
        df,
        *,
        max_bisect_steps=60,
    ):
        df_f = float(df)
        if df_f not in _LOW_DF:
            return current_critical(
                self,
                alpha,
                df,
                max_bisect_steps=max_bisect_steps,
            )

        backend, xp, device = _backend_context(self)
        statistic_ref = _reference_scalar(
            backend=backend,
            xp=xp,
            device=device,
        )
        _, critical = two_sided_reference_inference(
            statistic_ref,
            distribution="t",
            alpha=alpha,
            backend=backend,
            xp=xp,
            df=df_f,
            device=device,
        )
        return critical

    _two_sided_pvalue_stable._statgpu_low_df_reference = True
    _two_sided_pvalue_stable._statgpu_original = current_pvalue
    _two_sided_critical_value_stable._statgpu_low_df_reference = True
    _two_sided_critical_value_stable._statgpu_original = current_critical
    TDistributionBase.two_sided_pvalue = _two_sided_pvalue_stable
    TDistributionBase.two_sided_critical_value = _two_sided_critical_value_stable


_install_low_df_reference_contract()
