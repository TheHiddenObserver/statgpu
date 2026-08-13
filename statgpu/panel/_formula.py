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


def _top_level_formula_rhs_start(formula: str):
    """Return the index after the top-level formula separator, if present."""
    depth = 0
    quote = None
    escaped = False
    for i, ch in enumerate(formula):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth = max(depth - 1, 0)
            continue
        if depth == 0 and ch == "~":
            return i + 1
    return None


def _top_level_panel_token_spans(formula: str, token: str):
    """Return top-level RHS spans where ``token`` is a complete identifier."""
    rhs_start = _top_level_formula_rhs_start(formula)
    if rhs_start is None:
        return []

    i = rhs_start
    depth = 0
    quote = None
    escaped = False
    spans = []
    while i < len(formula):
        ch = formula[i]
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            i += 1
            continue
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
            depth = max(depth - 1, 0)
            i += 1
            continue
        if depth == 0 and formula.startswith(token, i):
            left = formula[i - 1] if i > 0 else ""
            j = i + len(token)
            right = formula[j] if j < len(formula) else ""
            left_ok = not (left.isalnum() or left == "_")
            right_ok = not (right.isalnum() or right == "_")
            if left_ok and right_ok:
                spans.append((i, j))
                i = j
                continue
        i += 1
    return spans


def _strip_panel_tokens(formula: str) -> Tuple[str, bool, bool]:
    """Detect top-level linearmodels effect tokens without substring rewriting.

    Magic tokens are recognized only as complete identifiers at the top level
    of the main formula RHS. Identifiers such as ``EntityEffectsScore`` and
    occurrences inside transforms such as ``C(EntityEffects)`` remain ordinary
    Patsy expressions rather than being rewritten by string substitution.
    """
    entity_effects = False
    time_effects = False
    clean = formula

    for token in _PANEL_TOKENS:
        spans = _top_level_panel_token_spans(clean, token)
        if not spans:
            continue
        if token in ("EntityEffects", "FixedEffects"):
            entity_effects = True
        elif token == "TimeEffects":
            time_effects = True

        for token_start, token_end in reversed(spans):
            left = token_start - 1
            while left >= 0 and clean[left].isspace():
                left -= 1
            right = token_end
            while right < len(clean) and clean[right].isspace():
                right += 1
            if left >= 0 and clean[left] == "+":
                clean = clean[:left] + clean[token_end:]
            elif right < len(clean) and clean[right] == "+":
                clean = clean[:token_start] + clean[right + 1:]
            else:
                clean = clean[:token_start] + clean[token_end:]

    clean = clean.strip()
    rhs_start = _top_level_formula_rhs_start(clean)
    if rhs_start is not None:
        rhs = clean[rhs_start:].strip()
        if not rhs or rhs in ("+", "-", "*", "/"):
            raise ValueError(
                "Formula has no predictors after removing panel tokens. "
                f"Original: {formula!r}, cleaned: {clean!r}"
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
    # Step 1: isolate the fixest pipe before interpreting magic tokens.
    # Tokens apply only to the main Patsy formula, never to FE variable names.
    main_formula, fe_vars = _split_panel_formula(formula)

    # Step 2: detect linearmodels-style tokens in the main formula only.
    clean_formula, token_entity, token_time = _strip_panel_tokens(main_formula)
    if fe_vars and (token_entity or token_time):
        raise ValueError(
            "Panel formulas cannot combine linearmodels-style effect tokens "
            "with fixest pipe fixed effects; choose one fixed-effect syntax"
        )
    main_formula = clean_formula

    # Merge the selected FE specification into the fit request.
    entity_effects = token_entity
    time_effects = token_time

    entity_ids = None
    time_ids = None

    if len(fe_vars) > 2:
        raise ValueError(
            "Panel formula fixed effects support at most two variables "
            "(entity and time); high-dimensional FE (>2) is not supported"
        )

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

    # Step 3: Parse the main formula with patsy
    from statgpu.core.formula import FormulaParser
    parser = FormulaParser(main_formula)
    y_arr, X_arr, design_info = parser.eval(data)
    setattr(design_info, "_statgpu_row_positions", np.asarray(parser._row_positions, dtype=np.int64))

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
    y_arr, X_arr, design_info = parser.eval(data)
    setattr(design_info, "_statgpu_row_positions", np.asarray(parser._row_positions, dtype=np.int64))
    return y_arr, X_arr, design_info


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
            entity_ids = _align_formula_side_array(entity_ids, design_info, len(y_arr), "entity_ids")
            time_ids = _align_formula_side_array(time_ids, design_info, len(y_arr), "time_ids")
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
        # Preserve NumPy/CuPy/Torch arrays.  The estimator resolves dtype/device
        # after this formula-only boundary; converting here would force GPU
        # array input through host NumPy.
        return (y, X, None, None, None,
                None, None, False, False)


def _ordered_categorical_array(values):
    """Return an ordered categorical array-like without importing pandas."""
    candidate = getattr(values, "array", values)
    dtype = getattr(candidate, "dtype", None)
    if (
        getattr(dtype, "categories", None) is not None
        and bool(getattr(dtype, "ordered", False))
        and getattr(candidate, "codes", None) is not None
    ):
        return candidate
    return None


def _align_formula_side_array(values, design_info, expected_n=None, name="array"):
    """Align an observation-level side array with rows retained by Patsy."""
    if values is None:
        return None

    categorical = _ordered_categorical_array(values)
    if categorical is None:
        arr = np.asarray(values)
        if arr.ndim == 0:
            raise ValueError(f"{name} must be observation-level")
        n_values = int(arr.shape[0])
    else:
        arr = None
        n_values = int(len(categorical))

    positions = getattr(design_info, "_statgpu_row_positions", None)
    if positions is None:
        if expected_n is not None and n_values != expected_n:
            raise ValueError(f"{name} must have {expected_n} observations")
        return categorical if categorical is not None else arr

    positions = np.asarray(positions, dtype=np.int64)
    if n_values == positions.shape[0]:
        return categorical if categorical is not None else arr
    if positions.size and n_values > int(positions.max()):
        if categorical is not None:
            return categorical.take(positions)
        return arr[positions]
    if positions.size == 0 and n_values == 0:
        return categorical if categorical is not None else arr
    raise ValueError(f"{name} has {n_values} observations and cannot be aligned to the {positions.shape[0]} rows retained by the formula")


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
        # Preserve NumPy/CuPy/Torch input. The estimator performs
        # backend-aware dtype/device conversion downstream.
        X_arr = X
    return X_arr


def _get_feature_names(feature_names, n_features, prefix="x"):
    """Get feature names for summary display."""
    if feature_names is not None:
        return list(feature_names)
    return [f"{prefix}{i}" for i in range(n_features)]
