"""GPU-accelerated two-way ANOVA.

Provides :func:`f_twoway` for two-factor analysis of variance with optional
interaction term, backend-agnostic across NumPy, CuPy, and PyTorch.
"""

from __future__ import annotations

__all__ = ["f_twoway", "TwoWayAnovaResult"]

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TwoWayAnovaResult:
    """Result of a two-way ANOVA.

    Attributes
    ----------
    factor_a_statistic : float
        F-statistic for factor A.
    factor_a_pvalue : float
        P-value for factor A.
    factor_a_df : int
        Degrees of freedom for factor A (a - 1).
    factor_a_eta_squared : float
        Eta-squared for factor A.

    factor_b_statistic : float
        F-statistic for factor B.
    factor_b_pvalue : float
        P-value for factor B.
    factor_b_df : int
        Degrees of freedom for factor B (b - 1).
    factor_b_eta_squared : float
        Eta-squared for factor B.

    interaction_statistic : float or None
        F-statistic for interaction (None if interaction=False).
    interaction_pvalue : float or None
        P-value for interaction (None if interaction=False).
    interaction_df : int or None
        Degrees of freedom for interaction (None if interaction=False).
    interaction_eta_squared : float or None
        Eta-squared for interaction (None if interaction=False).

    df_within : int
        Residual degrees of freedom.
    ss_within : float
        Residual sum of squares.
    """

    factor_a_statistic: float
    factor_a_pvalue: float
    factor_a_df: int
    factor_a_eta_squared: float

    factor_b_statistic: float
    factor_b_pvalue: float
    factor_b_df: int
    factor_b_eta_squared: float

    interaction_statistic: Optional[float]
    interaction_pvalue: Optional[float]
    interaction_df: Optional[int]
    interaction_eta_squared: Optional[float]

    df_within: int
    ss_within: float


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def f_twoway(
    data: Any,
    interaction: bool = True,
    backend: str = "auto",
    dtype: Any = None,
) -> TwoWayAnovaResult:
    """Perform a two-way ANOVA.

    Parameters
    ----------
    data : array-like of shape (a, b) or list of lists
        Cell means or raw data grouped by factor A (rows) and factor B
        (columns).  Each element ``data[i][j]`` should be an array of
        observations in cell (i, j), or a scalar cell mean with known
        cell sizes.

        For balanced designs, ``data`` can be a 2-D array of shape
        ``(n_a, n_b)`` where each element is an array of observations.

    interaction : bool, default=True
        If True, include the interaction term (full model).
        If False, fit an additive model (no interaction).

    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.

    dtype : dtype or None, default=None
        Float dtype for computation.  ``None`` uses ``float64``.

    Returns
    -------
    TwoWayAnovaResult
        Dataclass with F-statistics, p-values, dfs, and eta-squared for
        each factor and (optionally) the interaction.

    Raises
    ------
    ValueError
        If the data structure is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.anova import f_twoway
    >>> # 2x3 balanced design, 5 obs per cell
    >>> data = [[np.random.randn(5) for _ in range(3)] for _ in range(2)]
    >>> result = f_twoway(data, interaction=True)
    """
    # Resolve backend -- use numpy for initial parsing, then switch
    resolved = _resolve_backend(backend)
    xp = _get_xp(resolved)
    float_dtype = dtype if dtype is not None else xp.float64

    # Parse data into cell arrays
    cells, n_a, n_b, cell_sizes, cell_sums, cell_ss = _parse_cells(
        data, xp, float_dtype
    )

    N = sum(cell_sizes.values())
    grand_total = sum(cell_sums.values())
    grand_mean = grand_total / N

    # --- Sum of Squares decomposition ---
    # SSA (factor A)
    ss_a = 0.0
    for i in range(n_a):
        row_total = sum(cell_sums.get((i, j), 0.0) for j in range(n_b))
        row_n = sum(cell_sizes.get((i, j), 0) for j in range(n_b))
        if row_n > 0:
            row_mean = row_total / row_n
            ss_a += row_n * (row_mean - grand_mean) ** 2

    # SSB (factor B)
    ss_b = 0.0
    for j in range(n_b):
        col_total = sum(cell_sums.get((i, j), 0.0) for i in range(n_a))
        col_n = sum(cell_sizes.get((i, j), 0) for i in range(n_a))
        if col_n > 0:
            col_mean = col_total / col_n
            ss_b += col_n * (col_mean - grand_mean) ** 2

    # SSW (within / residual)
    ssw = 0.0
    for (i, j), vals in cells.items():
        cell_mean = cell_sums[(i, j)] / cell_sizes[(i, j)]
        ssw += cell_ss[(i, j)] - cell_sizes[(i, j)] * cell_mean ** 2

    # SST = sum(x^2) - N * grand_mean^2
    total_ss_raw = sum(cell_ss[(i, j)] for (i, j) in cells)
    sst = total_ss_raw - N * grand_mean ** 2
    # SS_AB = SST - SSA - SSB - SSW (clamp to 0 for floating-point rounding)
    ss_ab = max(sst - ss_a - ss_b - ssw, 0.0)

    # Degrees of freedom
    df_a = n_a - 1
    df_b = n_b - 1
    if interaction:
        df_ab = df_a * df_b
    else:
        df_ab = 0
        ss_ab = 0.0
    df_w = N - (df_a + df_b + df_ab + 1)

    if df_w <= 0:
        raise ValueError(
            f"Not enough observations for the model. "
            f"N={int(N)}, df_within={df_w}. Need more observations."
        )

    # Mean squares
    ms_a = ss_a / df_a if df_a > 0 else 0.0
    ms_b = ss_b / df_b if df_b > 0 else 0.0
    ms_ab = ss_ab / df_ab if df_ab > 0 else 0.0
    ms_w = ssw / df_w

    # F-statistics
    f_a = ms_a / ms_w if ms_w > 0 else float("inf")
    f_b = ms_b / ms_w if ms_w > 0 else float("inf")
    f_ab = ms_ab / ms_w if ms_w > 0 and df_ab > 0 else None

    # P-values from F distribution
    from statgpu.inference._distributions_backend import get_distribution

    f_dist = get_distribution("f", backend=resolved)

    p_a = _to_float_scalar(f_dist.sf(f_a, df_a, df_w))
    p_b = _to_float_scalar(f_dist.sf(f_b, df_b, df_w))
    p_ab = _to_float_scalar(f_dist.sf(f_ab, df_ab, df_w)) if f_ab is not None else None

    # Eta-squared: use appropriate denominator
    # For interaction model: ss_a + ss_b + ss_ab + ssw
    # For additive model: ss_a + ss_b + ssw (exclude interaction SS)
    if interaction:
        sst_denom = ss_a + ss_b + ss_ab + ssw
    else:
        sst_denom = ss_a + ss_b + ssw
    eta_a = ss_a / sst_denom if sst_denom > 0 else float("nan")
    eta_b = ss_b / sst_denom if sst_denom > 0 else float("nan")
    eta_ab = ss_ab / sst_denom if sst_denom > 0 and interaction else None

    return TwoWayAnovaResult(
        factor_a_statistic=f_a,
        factor_a_pvalue=p_a,
        factor_a_df=df_a,
        factor_a_eta_squared=eta_a,
        factor_b_statistic=f_b,
        factor_b_pvalue=p_b,
        factor_b_df=df_b,
        factor_b_eta_squared=eta_b,
        interaction_statistic=f_ab,
        interaction_pvalue=p_ab,
        interaction_df=df_ab if interaction else None,
        interaction_eta_squared=eta_ab,
        df_within=df_w,
        ss_within=ssw,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cells(data, xp, float_dtype):
    """Parse the data argument into a dict of cell arrays.

    Returns
    -------
    cells : dict (i, j) -> 1-D xp array
    n_a, n_b : int
    cell_sizes : dict (i, j) -> int
    cell_sums : dict (i, j) -> float
    cell_ss : dict (i, j) -> float (sum of squares within cell)
    """
    cells = {}
    cell_sizes = {}
    cell_sums = {}
    cell_ss = {}

    # data is expected to be a list of lists (or array of arrays)
    data_list = list(data) if not isinstance(data, list) else data

    n_a = len(data_list)
    if n_a < 1:
        raise ValueError("data must have at least 1 row (factor A level)")

    n_b = 0
    for i, row in enumerate(data_list):
        if not isinstance(row, (list, tuple)):
            row = [row]
        if i == 0:
            n_b = len(row)
        elif len(row) != n_b:
            raise ValueError(f"All rows must have the same number of columns. "
                             f"Row 0 has {n_b}, row {i} has {len(row)}.")
        for j, cell_data in enumerate(row):
            arr = np.asarray(cell_data, dtype=float_dtype).ravel()
            if arr.size == 0:
                raise ValueError(f"Cell ({i}, {j}) is empty.")
            cells[(i, j)] = arr
            cell_sizes[(i, j)] = arr.size
            cell_sums[(i, j)] = float(arr.sum())
            cell_ss[(i, j)] = float((arr ** 2).sum())

    return cells, n_a, n_b, cell_sizes, cell_sums, cell_ss
