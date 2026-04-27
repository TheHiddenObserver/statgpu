"""
Design matrix building utilities.

Provides convenience function for one-shot formula evaluation.
"""

from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd

from ._parser import FormulaParser


def parse_formula(
    formula: str,
    data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """One-shot convenience function for formula parsing.

    Parameters
    ----------
    formula : str
        R-style formula string, e.g. ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        DataFrame containing the referenced columns.

    Returns
    -------
    y : ndarray
        Response variable(s).
    X : ndarray
        Predictor design matrix.
    design_info : patsy.DesignInfo
        Metadata for the predictor design.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"y": [1, 2, 3], "x": [4, 5, 6]})
    >>> y, X, info = parse_formula("y ~ x", df)
    """
    parser = FormulaParser(formula)
    return parser.eval(data)


def parse_formula_safe(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Safe formula parsing that falls back to raw arrays.

    Used by model ``fit()`` methods to support both formula and array interfaces.

    Parameters
    ----------
    formula : str or None
        R-style formula string. If ``None``, ``X`` and ``y`` are used directly.
    data : pd.DataFrame or None
        DataFrame for formula parsing. Required when ``formula`` is given.
    X : ndarray or None
        Raw predictor matrix (used when ``formula`` is ``None``).
    y : ndarray or None
        Raw response vector (used when ``formula`` is ``None``).

    Returns
    -------
    y : ndarray
        Response variable(s).
    X : ndarray
        Predictor design matrix.
    design_info : patsy.DesignInfo or None
        Design metadata (``None`` when raw arrays are used).

    Raises
    ------
    ValueError
        If both formula and arrays are ``None``, or if formula is given without data.
    """
    if formula is not None:
        if data is None:
            raise ValueError(
                "formula was provided but data (DataFrame) is None. "
                "When using formula, pass data=your_dataframe."
            )
        return parse_formula(formula, data)

    if X is None or y is None:
        raise ValueError(
            "Either formula+data or X+y must be provided. "
            "Got formula=None and incomplete array input."
        )

    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y, np.asarray(X), None
