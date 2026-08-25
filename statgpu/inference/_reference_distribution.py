"""Backend-native reference-distribution inference helpers.

This module owns the small amount of policy needed by estimators that already
have a test statistic and need two-sided p-values plus a confidence-interval
critical value.  It deliberately delegates general distribution numerics to the
registered inference backends and keeps exact low-degree Student-t identities in
one place so model implementations do not grow backend-specific workarounds.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import xp_asarray
from statgpu.inference._distributions_backend import get_distribution


def two_sided_reference_inference(
    statistic_abs,
    *,
    distribution: str,
    alpha: float,
    backend: str,
    xp,
    df=None,
    device=None,
):
    """Return backend-native two-sided p-values and a critical value.

    Parameters
    ----------
    statistic_abs : array-like
        Absolute test statistic already resident on ``backend``.
    distribution : {"normal", "t"}
        Reference distribution.
    alpha : float
        Two-sided confidence-interval significance level.
    backend : {"numpy", "cupy", "torch"}
        Registered inference backend.
    xp : module
        Active numerical namespace corresponding to ``backend``.
    df : float, optional
        Student-t degrees of freedom.
    device : str, optional
        Concrete Torch device when ``backend="torch"``.

    Notes
    -----
    General normal and Student-t calculations delegate to
    :func:`get_distribution`. Student-t df=1 and df=2 use exact identities.
    The df=2 identity is important for maintained Torch versions without native
    ``betainc``: it preserves high precision without falling back to CPU/SciPy.
    """
    name = str(distribution).lower()
    alpha_f = float(alpha)
    if not np.isfinite(alpha_f) or not 0.0 < alpha_f < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    statistic_abs = xp_asarray(
        statistic_abs,
        dtype=xp.float64,
        xp=xp,
        ref_arr=statistic_abs,
    )

    if name in {"normal", "norm", "z"}:
        dist = get_distribution("norm", backend=backend, device=device)
        return (
            dist.two_sided_pvalue(statistic_abs),
            dist.two_sided_critical_value(alpha_f),
        )

    if name not in {"t", "student-t", "student_t"}:
        raise ValueError("distribution must be 'normal' or 't'")
    if df is None or not np.isfinite(float(df)) or float(df) <= 0.0:
        raise ValueError("Student-t inference requires finite positive df")
    df_f = float(df)

    if df_f == 1.0:
        # Student-t(1) is exactly Cauchy.  For x >= 0,
        #   P(|T_1| >= x) = 1 - (1 - 2 atan(x) / pi) = 2 atan(1 / x) / pi,
        # where atan(x) + atan(1/x) = pi/2 for x > 0 makes the second form
        # well conditioned for large statistics: the survival-function form
        # cancels subtractively and collapses to zero already around |t| ~ 1e15
        # although the tail remains representable.
        inv = xp_asarray(
            1.0,
            dtype=xp.float64,
            xp=xp,
            ref_arr=statistic_abs,
        ) / xp.maximum(statistic_abs, np.finfo(np.float64).tiny)
        # ``arctan`` (not the NumPy-2-only ``atan`` alias) so NumPy<2 CI
        # environments resolve the same backend call.
        pvalues = 2.0 * xp.arctan(inv) / np.pi
        dist = get_distribution("cauchy", backend=backend, device=device)
        return pvalues, dist.isf(alpha_f / 2.0)

    if df_f == 2.0:
        # For T_2 and x >= 0,
        #   P(|T_2| >= x) = 1 - x / sqrt(x^2 + 2).
        # Evaluate the algebraically equivalent rationalized form below to avoid
        # catastrophic cancellation when x is large:
        #   2 / {sqrt(x^2+2) [sqrt(x^2+2) + x]}.
        # ``hypot`` keeps sqrt(x^2 + 2) finite whenever the mathematical root is
        # representable, and the sequential divisions avoid overflowing the
        # denominator product while a subnormal but nonzero tail is still
        # representable in float64.
        sqrt_two = xp_asarray(
            np.sqrt(2.0),
            dtype=xp.float64,
            xp=xp,
            ref_arr=statistic_abs,
        )
        root = xp.hypot(statistic_abs, sqrt_two)
        pvalues = (2.0 / root) / (root + statistic_abs)
        alpha_dev = xp_asarray(
            alpha_f,
            dtype=xp.float64,
            xp=xp,
            ref_arr=statistic_abs,
        )
        critical = (
            sqrt_two
            * (1.0 - alpha_dev)
            / xp.sqrt(alpha_dev * (2.0 - alpha_dev))
        )
        return pvalues, critical

    dist = get_distribution("t", backend=backend, device=device)
    return (
        dist.two_sided_pvalue(statistic_abs, df_f),
        dist.two_sided_critical_value(alpha_f, df_f),
    )
