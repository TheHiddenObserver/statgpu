"""Data profiler: ingestion, type inference, imputation, encoding."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._config import AgentConfig, MISSING_STRINGS, TRUE_STRINGS, FALSE_STRINGS


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_STRINGS:
        return True
    return False


def _float_or_nan(value) -> float:
    if _is_missing(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _is_numeric_column(values: Sequence[Any]) -> bool:
    nonmissing = [v for v in values if not _is_missing(v)]
    if not nonmissing:
        return False
    for value in nonmissing:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def table_columns(data: Any) -> List[str]:
    """Extract column names from various data formats."""
    if isinstance(data, Mapping):
        return list(data.keys())
    if hasattr(data, "columns"):
        return [str(col) for col in list(data.columns)]
    if isinstance(data, np.ndarray) and data.dtype.names:
        return list(data.dtype.names)
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):
        return list(data[0].keys())
    raise ValueError(
        "data must be a pandas-like DataFrame, mapping of columns, structured array, "
        "or list of row dictionaries."
    )


def table_column(data: Any, name: str) -> List[Any]:
    """Extract a single column from various data formats."""
    if isinstance(data, Mapping):
        if name not in data:
            raise KeyError(f"Column '{name}' not found.")
        return list(data[name])
    if hasattr(data, "columns"):
        if name not in list(data.columns):
            raise KeyError(f"Column '{name}' not found.")
        col = data[name]
        return list(col.to_numpy() if hasattr(col, "to_numpy") else col)
    if isinstance(data, np.ndarray) and data.dtype.names:
        if name not in data.dtype.names:
            raise KeyError(f"Column '{name}' not found.")
        return list(data[name])
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):
        return [row.get(name) for row in data]
    raise ValueError("Unsupported data object.")


def prepare_target_array(y: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    """Prepare target array, encoding binary categorical targets."""
    arr_obj = list(y) if not isinstance(y, np.ndarray) else y
    values = np.asarray(arr_obj)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("Target array is empty.")
    if _is_numeric_column(list(values)):
        return np.array([_float_or_nan(v) for v in values], dtype=float), {}

    labels = ["__missing__" if _is_missing(v) else str(v).strip() for v in values]
    unique = sorted({label for label in labels if label != "__missing__"})
    if len(unique) != 2:
        raise ValueError(
            "Only numeric targets or binary categorical targets are supported."
        )
    mapping = {unique[0]: 0.0, unique[1]: 1.0}
    encoded = np.array(
        [np.nan if label == "__missing__" else mapping[label] for label in labels],
        dtype=float,
    )
    return encoded, mapping


def prepare_numeric_vector(values: Any, label: str) -> np.ndarray:
    """Prepare a numeric vector from various input formats."""
    if values is None:
        raise ValueError(f"{label} is required.")
    arr = np.array([_float_or_nan(v) for v in list(values)], dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def prepare_event_vector(values: Any) -> np.ndarray:
    """Prepare a binary event vector."""
    out = []
    for value in list(values):
        if _is_missing(value):
            out.append(np.nan)
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in TRUE_STRINGS:
                out.append(1.0)
            elif normalized in FALSE_STRINGS:
                out.append(0.0)
            else:
                out.append(_float_or_nan(value))
        else:
            out.append(_float_or_nan(value))
    arr = np.asarray(out, dtype=float)
    return np.where(arr > 0, 1.0, np.where(arr == 0, 0.0, np.nan))


def sanitize_numeric_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, int, List[str]]:
    """Impute non-finite values with column medians."""
    X = np.asarray(X, dtype=float)
    notes: List[str] = []
    imputed = 0
    for idx, name in enumerate(feature_names):
        col = X[:, idx]
        finite = np.isfinite(col)
        if np.all(finite):
            continue
        if not np.any(finite):
            raise ValueError(f"Feature '{name}' has no finite values.")
        fill = float(np.nanmedian(col[finite]))
        missing = int(np.sum(~finite))
        X[~finite, idx] = fill
        imputed += missing
    if imputed:
        notes.append(f"Imputed {imputed} non-finite feature values with medians.")
    return X, imputed, notes


def encode_features(
    raw_feature_columns: Dict[str, List[Any]],
    mask: np.ndarray,
    max_categories: int,
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]], int, List[str]]:
    """Encode features: numeric imputation + categorical one-hot encoding."""
    encoded_parts: List[np.ndarray] = []
    encoded_names: List[str] = []
    encoded_features: Dict[str, List[str]] = {}
    notes: List[str] = []
    imputed = 0

    for name, values in raw_feature_columns.items():
        filtered = [values[i] for i in np.flatnonzero(mask)]
        if _is_numeric_column(filtered):
            arr = np.array([_float_or_nan(v) for v in filtered], dtype=float)
            finite = np.isfinite(arr)
            if not np.any(finite):
                notes.append(f"Skipped feature '{name}' because all values are missing.")
                continue
            fill_value = float(np.nanmedian(arr[finite]))
            missing = int(np.sum(~finite))
            if missing:
                arr[~finite] = fill_value
                imputed += missing
            encoded_parts.append(arr.reshape(-1, 1))
            encoded_names.append(name)
        else:
            values_norm = [
                "__missing__" if _is_missing(v) else str(v).strip()
                for v in filtered
            ]
            counts = Counter(values_norm)
            categories = [cat for cat, _ in counts.most_common()]
            if len(categories) <= 1:
                notes.append(f"Skipped categorical feature '{name}' with one level.")
                continue
            if len(categories) > max_categories:
                kept = categories[:max_categories]
                notes.append(
                    f"Collapsed categorical feature '{name}' to top "
                    f"{max_categories} levels."
                )
            else:
                kept = categories
            reference = kept[0]
            dummy_categories = kept[1:]
            matrix = np.zeros((len(values_norm), len(dummy_categories)), dtype=float)
            for row_idx, value in enumerate(values_norm):
                if value in dummy_categories:
                    matrix[row_idx, dummy_categories.index(value)] = 1.0
            encoded_parts.append(matrix)
            names = [f"{name}[{cat}]" for cat in dummy_categories]
            encoded_names.extend(names)
            encoded_features[name] = [reference] + dummy_categories

    if not encoded_parts:
        raise ValueError("No usable feature columns after preprocessing.")
    X_arr = np.column_stack(encoded_parts)
    return X_arr, encoded_names, encoded_features, imputed, notes


def prepare_table(
    data: Any,
    target: Optional[str],
    time: Any,
    event: Any,
    feature_columns: Optional[Sequence[str]],
    config: AgentConfig,
) -> "PreparedData":
    """Prepare data from table-like input (CSV, DataFrame, dict, list-of-dicts)."""
    from ._analysis import PreparedData

    columns = table_columns(data)
    if not columns:
        raise ValueError("data has no columns.")

    time_col = time if isinstance(time, str) else None
    event_col = event if isinstance(event, str) else None
    excluded = {name for name in (target, time_col, event_col) if name is not None}
    if feature_columns is None:
        features = [name for name in columns if name not in excluded]
    else:
        features = list(feature_columns)
    if not features:
        raise ValueError("No feature columns were selected.")

    raw_feature_columns = {name: table_column(data, name) for name in features}
    n = len(next(iter(raw_feature_columns.values())))

    y_arr = None
    y_mapping: Dict[str, float] = {}
    if target is not None:
        y_arr, y_mapping = prepare_target_array(table_column(data, target))

    time_arr = None
    if time is not None:
        time_arr = prepare_numeric_vector(
            table_column(data, time) if isinstance(time, str) else time,
            "time",
        )
    event_arr = None
    if event is not None:
        event_arr = prepare_event_vector(
            table_column(data, event) if isinstance(event, str) else event
        )

    mask = np.ones(n, dtype=bool)
    for arr in (y_arr, time_arr, event_arr):
        if arr is not None:
            if arr.shape[0] != n:
                raise ValueError("Response/survival column length must match data rows.")
            mask &= np.isfinite(arr)
    dropped = int(n - np.sum(mask))

    X_arr, encoded_names, encoded_features, imputed, notes = encode_features(
        raw_feature_columns, mask, config.max_categories
    )

    if y_arr is not None:
        y_arr = y_arr[mask]
    if time_arr is not None:
        time_arr = time_arr[mask]
    if event_arr is not None:
        event_arr = event_arr[mask]
    if dropped:
        notes.append(f"Dropped {dropped} rows with missing response/survival fields.")

    X_arr, imputed_more, matrix_notes = sanitize_numeric_matrix(X_arr, encoded_names)
    imputed += imputed_more
    notes.extend(matrix_notes)

    return PreparedData(
        X=X_arr,
        y=y_arr,
        time=time_arr,
        event=event_arr,
        feature_names=encoded_names,
        target_name=target,
        y_mapping=y_mapping,
        dropped_rows=dropped,
        imputed_values=imputed,
        encoded_features=encoded_features,
        notes=notes,
    )


def prepare_array(
    X: Any,
    y: Any,
    time: Any,
    event: Any,
    feature_names: Optional[Sequence[str]],
    target: Optional[str],
) -> "PreparedData":
    """Prepare data from numpy array input."""
    from ._analysis import PreparedData

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")
    names = list(feature_names) if feature_names is not None else [
        f"x{i}" for i in range(X_arr.shape[1])
    ]
    if len(names) != X_arr.shape[1]:
        raise ValueError("feature_names length must match X.shape[1].")
    X_arr, imputed, notes = sanitize_numeric_matrix(X_arr, names)

    y_arr, y_mapping = prepare_target_array(y) if y is not None else (None, {})
    time_arr = prepare_numeric_vector(time, "time") if time is not None else None
    event_arr = prepare_event_vector(event) if event is not None else None

    n = X_arr.shape[0]
    for label, arr in (("y", y_arr), ("time", time_arr), ("event", event_arr)):
        if arr is not None and arr.shape[0] != n:
            raise ValueError(f"{label} length must match X.shape[0].")

    mask = np.ones(n, dtype=bool)
    for arr in (y_arr, time_arr, event_arr):
        if arr is not None:
            mask &= np.isfinite(arr)
    dropped = int(n - np.sum(mask))
    if dropped:
        X_arr = X_arr[mask]
        if y_arr is not None:
            y_arr = y_arr[mask]
        if time_arr is not None:
            time_arr = time_arr[mask]
        if event_arr is not None:
            event_arr = event_arr[mask]
        notes.append(f"Dropped {dropped} rows with missing response/survival fields.")

    return PreparedData(
        X=X_arr,
        y=y_arr,
        time=time_arr,
        event=event_arr,
        feature_names=names,
        target_name=target,
        y_mapping=y_mapping,
        dropped_rows=dropped,
        imputed_values=imputed,
        encoded_features={},
        notes=notes,
    )
