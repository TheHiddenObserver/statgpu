"""Formula interface support for panel data models.

Supports three formula styles:

1. **fixest pipe syntax**: ``"y ~ x1 + x2 | entity + time"``
2. **linearmodels tokens**: ``"y ~ x1 + EntityEffects + TimeEffects"``
3. **Standard R formula**: ``"y ~ x1 + x2"``

Plus backward-compatible array interface: ``fit(X, y, entity_ids=...)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Panel formula parsing (fixest pipe syntax + linearmodels tokens)
# ---------------------------------------------------------------------------

# linearmodels-style magic tokens
_PANEL_TOKENS = frozenset({"EntityEffects", "TimeEffects", "FixedEffects"})


def _split_panel_formula(formula: str) -> Tuple[str, List[str]]:
    """Split a fixest-style panel formula on ``|``.

    Parameters
    ----------
    formula : str
        Formula string, e.g. ``"y ~ x1 + x2 | entity + time"``.

    Returns
    -------
    main_formula : str
        The left side of ``|``, e.g. ``"y ~ x1 + x2"``.
    fe_vars : list of str
        Fixed effect variable names from the right side of ``|``,
        e.g. ``["entity", "time"]``.  Empty list if no ``|``.

    Examples
    --------
    >>> _split_panel_formula("y ~ x1 + x2 | entity + time")
    ('y ~ x1 + x2', ['entity', 'time'])
    >>> _split_panel_formula("y ~ x1 + x2")
    ('y ~ x1 + x2', [])
    """
    # Find the top-level | (not inside parentheses)
    depth = 0
    pipe_pos = -1
    for i, ch in enumerate(formula):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '|' and depth == 0:
            pipe_pos = i
            break

    if pipe_pos < 0:
        return formula.strip(), []

    main = formula[:pipe_pos].strip()
    rhs = formula[pipe_pos + 1:].strip()

    # Parse RHS: "+ separated variable names
    fe_vars = [v.strip() for v in rhs.split('+') if v.strip()]
    for v in fe_vars:
        if not v.isidentifier():
            raise ValueError(
                f"Invalid fixed effect variable name '{v}' in formula RHS. "
                f"Only simple variable names are supported (no transformations)."
            )

    return main, fe_vars


def _strip_panel_tokens(formula: str) -> Tuple[str, bool, bool]:
    """Detect and strip linearmodels-style tokens from a formula.

    Parameters
    ----------
    formula : str
        Formula string, e.g. ``"y ~ x1 + EntityEffects + TimeEffects"``.

    Returns
    -------
    clean_formula : str
        Formula with tokens removed.
    entity_effects : bool
        True if ``EntityEffects`` or ``FixedEffects`` token was present.
    time_effects : bool
        True if ``TimeEffects`` token was present.
    """
    entity_effects = False
    time_effects = False

    clean = formula
    for token in _PANEL_TOKENS:
        if token in clean:
            if token in ("EntityEffects", "FixedEffects"):
                entity_effects = True
            elif token == "TimeEffects":
                time_effects = True
            # Remove the token and surrounding + signs
            clean = clean.replace(f"+ {token}", "").replace(f"+{token}", "")
            clean = clean.replace(f"{token} +", "").replace(f"{token}+", "")
            clean = clean.replace(token, "")

    # Clean up whitespace
    clean = ' '.join(clean.split())

    # Validate that formula has at least one predictor after token removal
    if '~' in clean:
        rhs = clean.split('~', 1)[1].strip()
        if not rhs or rhs in ('+', '-', '*', '/'):
            raise ValueError(
                f"Formula has no predictors after removing panel tokens. "
                f"Original: '{formula}', cleaned: '{clean}'"
            )

    return clean, entity_effects, time_effects


def parse_panel_formula(formula, data):
    """Parse a panel formula (fixest pipe or linearmodels tokens).

    Parameters
    ----------
    formula : str
        Formula string supporting fixest pipe syntax and linearmodels tokens.
    data : DataFrame
        Data containing all variables referenced in the formula.

    Returns
    -------
    y : ndarray
        Response vector.
    X : ndarray
        Design matrix (without FE columns).
    design_info : object
        patsy DesignInfo for prediction.
    entity_ids : ndarray or None
        Entity identifiers if specified via ``|``.
    time_ids : ndarray or None
        Time identifiers if specified via ``|``.
    entity_effects : bool
        Whether entity effects are requested.
    time_effects : bool
        Whether time effects are requested.
    feature_names : list of str
        Names of regressor columns.
    """
    # Step 1: Strip linearmodels-style tokens
    clean_formula, token_entity, token_time = _strip_panel_tokens(formula)

    # Step 2: Split on | (fixest syntax)
    main_formula, fe_vars = _split_panel_formula(clean_formula)

    # Merge token-based and pipe-based FE specifications
    entity_effects = token_entity
    time_effects = token_time

    entity_ids = None
    time_ids = None

    if fe_vars:
        # Map FE variables to entity/time
        # Convention: first FE var = entity, second = time (if present)
        if len(fe_vars) >= 1:
            entity_effects = True
            if fe_vars[0] in data.columns:
                entity_ids = data[fe_vars[0]].values
        if len(fe_vars) >= 2:
            time_effects = True
            if fe_vars[1] in data.columns:
                time_ids = data[fe_vars[1]].values
        if len(fe_vars) > 2:
            # For >2 FE vars, we still extract entity and time
            # but warn that high-dim FE is not yet supported
            import warnings
            warnings.warn(
                f"Formula has {len(fe_vars)} fixed effect variables. "
                f"Only the first two are used as entity/time effects. "
                f"High-dimensional FE (>2) is not yet supported.",
                UserWarning,
                stacklevel=3,
            )

    # Step 3: Parse the main formula with patsy
    from statgpu.core.formula import FormulaParser
    parser = FormulaParser(main_formula)
    y_arr, X_arr, design_info = parser.eval(data)

    formula_column_names = list(design_info.column_names)
    has_intercept = "Intercept" in formula_column_names
    feature_names = [n for n in formula_column_names if n != "Intercept"]

    return (
        y_arr, X_arr, design_info,
        entity_ids, time_ids,
        entity_effects, time_effects,
        feature_names, has_intercept,
    )


# ---------------------------------------------------------------------------
# Standard formula helpers (backward compatible)
# ---------------------------------------------------------------------------

def _parse_formula_panel(formula, data):
    """Parse formula+data for panel models (legacy, no pipe support).

    Returns (y, X, design_info).
    """
    from statgpu.core.formula import FormulaParser
    parser = FormulaParser(formula)
    return parser.eval(data)


def _prepare_formula_fit(formula, data, X, y, model_has_intercept=True,
                         support_pipe=False, entity_effects_attr=None,
                         time_effects_attr=None):
    """Handle formula vs array input for panel models.

    Parameters
    ----------
    formula : str or None
        R-style formula (e.g. "y ~ x1 + x2" or "y ~ x1 + x2 | entity + time").
    data : DataFrame or None
        Data for formula parsing.
    X : array-like or None
        Predictor matrix (used when formula is None).
    y : array-like or None
        Response vector (used when formula is None).
    model_has_intercept : bool
        Whether the model adds its own intercept.
    support_pipe : bool
        If True, parse ``|`` as fixest-style fixed effects.
    entity_effects_attr : str or None
        If set, store entity_effects flag under this attribute name.
    time_effects_attr : str or None
        If set, store time_effects flag under this attribute name.

    Returns
    -------
    y_arr : ndarray
    X_arr : ndarray
    design_info : object or None
    feature_names : list or None
    formula_has_intercept : bool or None
    entity_ids : ndarray or None
    time_ids : ndarray or None
    entity_effects : bool
    time_effects : bool
    """
    if formula is not None:
        if data is None:
            raise ValueError(
                "formula was provided but data is None. "
                "Pass data=your_dataframe when using formula."
            )

        if support_pipe:
            (y_arr, X_arr, design_info,
             entity_ids, time_ids,
             entity_effects, time_effects,
             feature_names, has_intercept) = parse_panel_formula(formula, data)
            # For linearmodels tokens, try to extract entity/time from DataFrame
            if entity_effects and entity_ids is None and hasattr(data, 'columns'):
                if 'entity' in data.columns:
                    entity_ids = data['entity'].values
            if time_effects and time_ids is None and hasattr(data, 'columns'):
                if 'time' in data.columns:
                    time_ids = data['time'].values
        else:
            y_arr, X_arr, design_info = _parse_formula_panel(formula, data)
            entity_ids, time_ids = None, None
            entity_effects, time_effects = False, False
            formula_column_names = list(design_info.column_names)
            has_intercept = "Intercept" in formula_column_names
            feature_names = [n for n in formula_column_names if n != "Intercept"]

        # Strip intercept if present — let model handle it
        if has_intercept:
            intercept_idx = list(design_info.column_names).index("Intercept")
            X_arr = np.delete(X_arr, intercept_idx, axis=1)
            feature_names = [n for n in feature_names if n != "Intercept"]

        return (y_arr, X_arr, design_info, feature_names, has_intercept,
                entity_ids, time_ids, entity_effects, time_effects)
    else:
        if X is None or y is None:
            raise ValueError("Either formula+data or X+y must be provided.")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        X_arr = np.asarray(X, dtype=np.float64)
        return (y_arr, X_arr, None, None, None,
                None, None, False, False)


def _formula_predict(X, design_info, formula_has_intercept, model_has_intercept):
    """Prepare X for prediction when model was trained with a formula.

    The intercept is always stripped from the prediction matrix if the
    formula included one, because _prepare_formula_fit strips it during
    training regardless of model_has_intercept.  The model adds its own
    intercept if needed.
    """
    if design_info is not None and hasattr(X, 'columns'):
        import patsy
        X_arr = patsy.build_design_matrices([design_info], X)[0]

        # Always strip intercept if formula had one — it was stripped during fit
        col_names = list(design_info.column_names)
        if formula_has_intercept and "Intercept" in col_names:
            intercept_idx = col_names.index("Intercept")
            X_arr = np.delete(X_arr, intercept_idx, axis=1)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
    return X_arr


def _get_feature_names(feature_names, n_features, prefix="x"):
    """Get feature names for summary display."""
    if feature_names is not None:
        return list(feature_names)
    return [f"{prefix}{i}" for i in range(n_features)]
