"""Formula interface support for panel data models.

Provides a shared helper for integrating patsy-style formulas with
panel model fit/predict/summary methods.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _parse_formula_panel(formula, data):
    """Parse formula+data for panel models.

    Returns (y, X, design_info) where y and X are numpy arrays and
    design_info is the patsy DesignInfo for later prediction use.
    """
    from statgpu.core.formula import FormulaParser
    parser = FormulaParser(formula)
    return parser.eval(data)


def _prepare_formula_fit(formula, data, X, y, model_has_intercept=True):
    """Handle formula vs array input for panel models.

    Parameters
    ----------
    formula : str or None
        R-style formula (e.g. "y ~ x1 + x2").
    data : DataFrame or None
        Data for formula parsing.
    X : array-like or None
        Predictor matrix (used when formula is None).
    y : array-like or None
        Response vector (used when formula is None).
    model_has_intercept : bool
        Whether the model adds its own intercept. If True and the formula
        includes an intercept, the intercept column is stripped from X.

    Returns
    -------
    y_arr : ndarray
    X_arr : ndarray
    design_info : object or None
    feature_names : list or None
    formula_has_intercept : bool or None
    """
    if formula is not None:
        if data is None:
            raise ValueError(
                "formula was provided but data is None. "
                "Pass data=your_dataframe when using formula."
            )
        y_arr, X_arr, design_info = _parse_formula_panel(formula, data)
        formula_column_names = list(design_info.column_names)
        has_intercept = "Intercept" in formula_column_names
        feature_names = [n for n in formula_column_names if n != "Intercept"]

        if has_intercept and model_has_intercept:
            # Strip intercept — model will add its own
            intercept_idx = formula_column_names.index("Intercept")
            X_arr = np.delete(X_arr, intercept_idx, axis=1)
        elif has_intercept and not model_has_intercept:
            # Model doesn't add intercept, keep it from formula
            pass

        return y_arr, X_arr, design_info, feature_names, has_intercept
    else:
        if X is None or y is None:
            raise ValueError("Either formula+data or X+y must be provided.")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        X_arr = np.asarray(X, dtype=np.float64)
        return y_arr, X_arr, None, None, None


def _formula_predict(X, design_info, formula_has_intercept, model_has_intercept):
    """Prepare X for prediction when model was trained with a formula.

    Parameters
    ----------
    X : array-like or DataFrame
        New data for prediction.
    design_info : object or None
        patsy DesignInfo from training.
    formula_has_intercept : bool or None
        Whether the formula included an intercept.
    model_has_intercept : bool
        Whether the model adds its own intercept.

    Returns
    -------
    X_arr : ndarray
        Design matrix ready for prediction.
    """
    if design_info is not None and hasattr(X, 'columns'):
        # X is a DataFrame — use formula to build design matrix
        from statgpu.core.formula import FormulaParser
        parser = FormulaParser()
        parser._design_info = design_info
        parser.formula = None
        X_arr = parser.transform(X)

        col_names = list(design_info.column_names)
        if formula_has_intercept and model_has_intercept and "Intercept" in col_names:
            # Strip the formula intercept — model adds its own
            intercept_idx = col_names.index("Intercept")
            X_arr = np.delete(X_arr, intercept_idx, axis=1)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
    return X_arr


def _get_feature_names(feature_names, n_features, prefix="x"):
    """Get feature names for summary display.

    Parameters
    ----------
    feature_names : list or None
        Names from formula (may include intercept).
    n_features : int
        Number of features (excluding intercept).
    prefix : str
        Prefix for auto-generated names.

    Returns
    -------
    names : list of str
    """
    if feature_names is not None:
        return list(feature_names)
    return [f"{prefix}{i}" for i in range(n_features)]
