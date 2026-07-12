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
    """Perform a balanced two-way ANOVA.

    Each cell must contain the same number of observations.  Unbalanced
    designs require an explicit sums-of-squares convention (Type I/II/III),
    which this API does not expose, so they are rejected rather than silently
    applying the orthogonal balanced-design decomposition.
    """
    resolved = _resolve_backend(backend)
    xp = _get_xp(resolved)
    float_dtype = dtype if dtype is not None else xp.float64

    _, n_a, n_b, cell_arrays, cell_sizes_arr, _, _ = _parse_cells_vectorized(
        data, xp, float_dtype
    )
    if n_a < 2 or n_b < 2:
        raise ValueError("two-way ANOVA requires at least 2 levels for each factor")

    cell_sizes = np.asarray(_to_numpy(cell_sizes_arr), dtype=np.int64)
    if cell_sizes.size != n_a * n_b or np.any(cell_sizes != cell_sizes[0]):
        raise ValueError(
            "f_twoway currently requires a balanced design with equal cell sizes; "
            "unbalanced designs need an explicit Type I/II/III sums-of-squares choice"
        )
    n_cell = int(cell_sizes[0])
    if n_cell < 1:
        raise ValueError("each factor cell must contain at least one observation")

    cube = xp.stack(cell_arrays, axis=0).reshape(n_a, n_b, n_cell)
    cell_means = xp.mean(cube, axis=2)
    row_means = xp.mean(cell_means, axis=1)
    col_means = xp.mean(cell_means, axis=0)
    grand_mean = xp.mean(cell_means)

    ss_a = _to_float_scalar(
        float(n_b * n_cell) * xp.sum((row_means - grand_mean) ** 2)
    )
    ss_b = _to_float_scalar(
        float(n_a * n_cell) * xp.sum((col_means - grand_mean) ** 2)
    )
    interaction_effect = (
        cell_means - row_means[:, None] - col_means[None, :] + grand_mean
    )
    ss_ab_full = _to_float_scalar(
        float(n_cell) * xp.sum(interaction_effect ** 2)
    )
    ss_within_cells = _to_float_scalar(
        xp.sum((cube - cell_means[:, :, None]) ** 2)
    )

    df_a = n_a - 1
    df_b = n_b - 1
    df_ab_full = df_a * df_b
    n_total = n_a * n_b * n_cell

    if interaction:
        ss_ab = ss_ab_full
        df_ab = df_ab_full
        ss_error = ss_within_cells
        df_error = n_total - n_a * n_b
    else:
        ss_ab = 0.0
        df_ab = 0
        # Omitting the interaction makes its variation part of the additive
        # model residual.  Keeping only within-cell SSE inflates both main
        # effect F statistics.
        ss_error = ss_within_cells + ss_ab_full
        df_error = n_total - (1 + df_a + df_b)

    if df_error <= 0:
        raise ValueError(
            f"Not enough observations for the requested model: N={n_total}, "
            f"df_within={df_error}"
        )

    from statgpu.inference._distributions_backend import get_distribution

    f_dist = get_distribution("f", backend=resolved)
    ms_error = ss_error / df_error

    def _effect_test(ss_effect, df_effect):
        ms_effect = ss_effect / df_effect
        if ms_error == 0.0:
            if ms_effect == 0.0:
                return float("nan"), float("nan")
            return float("inf"), 0.0
        statistic = ms_effect / ms_error
        return statistic, _to_float_scalar(f_dist.sf(statistic, df_effect, df_error))

    f_a, p_a = _effect_test(ss_a, df_a)
    f_b, p_b = _effect_test(ss_b, df_b)
    if interaction:
        f_ab, p_ab = _effect_test(ss_ab, df_ab)
    else:
        f_ab = p_ab = None

    total_ss = ss_a + ss_b + ss_ab_full + ss_within_cells
    eta_a = ss_a / total_ss if total_ss > 0 else float("nan")
    eta_b = ss_b / total_ss if total_ss > 0 else float("nan")
    eta_ab = ss_ab_full / total_ss if total_ss > 0 and interaction else None

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
        df_within=df_error,
        ss_within=ss_error,
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
            arr = xp.asarray(cell_data, dtype=float_dtype).ravel()
            n_obs = int(arr.numel()) if hasattr(arr, 'numel') else int(arr.size)
            if n_obs == 0:
                raise ValueError(f"Cell ({i}, {j}) is empty.")
            cells[(i, j)] = arr
            cell_sizes[(i, j)] = n_obs
            cell_sums[(i, j)] = float(arr.sum())
            cell_ss[(i, j)] = float((arr ** 2).sum())

    return cells, n_a, n_b, cell_sizes, cell_sums, cell_ss


def _parse_cells_vectorized(data, xp, float_dtype):
    """Parse data into cell arrays without per-cell GPU sync.

    Returns
    -------
    cells : dict (i, j) -> 1-D xp array
    n_a, n_b : int
    cell_arrays : list of 1-D arrays (flat, in row-major order)
    cell_sizes_arr : 1-D xp array of cell sizes
    a_labels, b_labels : not used (kept for API compat)
    """
    cells = {}
    cell_arrays = []
    cell_sizes_list = []

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
            arr = xp.asarray(cell_data, dtype=float_dtype).ravel()
            n_obs = int(arr.numel()) if hasattr(arr, 'numel') else int(arr.size)
            if n_obs == 0:
                raise ValueError(f"Cell ({i}, {j}) is empty.")
            cells[(i, j)] = arr
            cell_arrays.append(arr)
            cell_sizes_list.append(n_obs)

    if hasattr(xp, 'tensor'):  # torch
        # Infer device from first cell array to avoid CPU/CUDA mismatch
        _device = cell_arrays[0].device if cell_arrays else 'cpu'
        cell_sizes_arr = xp.tensor(cell_sizes_list, dtype=float_dtype, device=_device)
    else:
        cell_sizes_arr = xp.array(cell_sizes_list, dtype=float_dtype)
    return cells, n_a, n_b, cell_arrays, cell_sizes_arr, None, None
