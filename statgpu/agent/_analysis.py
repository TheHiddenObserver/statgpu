"""Automatic statistical analysis agent built on top of statgpu estimators."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from statgpu.linear_model import LinearRegression, LogisticRegression, PoissonRegression, Ridge
from statgpu.survival import CoxPH
from statgpu.unsupervised import KMeans, PCA


MISSING_STRINGS = {"", "na", "nan", "null", "none", "missing"}
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "event", "case", "dead"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "censored", "control", "alive"}


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


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


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NA"
    if abs(numeric) >= 1000 or (abs(numeric) < 0.001 and numeric != 0.0):
        return f"{numeric:.3e}"
    return f"{numeric:.4f}"


def _safe_call(func, *args, **kwargs):
    try:
        value = func(*args, **kwargs)
    except Exception:
        return None
    try:
        if value is None or not np.isfinite(float(value)):
            return None
    except (TypeError, ValueError):
        return value
    return value


@dataclass
class DataProfile:
    """Data profile produced by the profiler stage."""

    n_samples: int
    n_features: int
    task_type: str
    feature_names: List[str]
    target_name: Optional[str] = None
    device: str = "auto"
    dropped_rows: int = 0
    imputed_values: int = 0
    encoded_features: Dict[str, List[str]] = field(default_factory=dict)
    target_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(
            {
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "task_type": self.task_type,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "device": self.device,
                "dropped_rows": self.dropped_rows,
                "imputed_values": self.imputed_values,
                "encoded_features": self.encoded_features,
                "target_summary": self.target_summary,
                "notes": self.notes,
            }
        )


@dataclass
class AnalysisPlan:
    """Planned agent stages and statgpu methods."""

    task_type: str
    agents: List[str]
    methods: List[str]
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready(
            {
                "task_type": self.task_type,
                "agents": self.agents,
                "methods": self.methods,
                "rationale": self.rationale,
            }
        )


@dataclass
class ModelResult:
    """Single fitted model result."""

    name: str
    task_type: str
    estimator: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    coefficients: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self, include_estimator: bool = False) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "task_type": self.task_type,
            "metrics": self.metrics,
            "coefficients": self.coefficients,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "error": self.error,
        }
        if include_estimator:
            out["estimator"] = self.estimator
        return _json_ready(out)


@dataclass
class AnalysisResult:
    """Complete automatic analysis result."""

    profile: DataProfile
    plan: AnalysisPlan
    models: List[ModelResult]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self, include_estimators: bool = False) -> Dict[str, Any]:
        return _json_ready(
            {
                "profile": self.profile.to_dict(),
                "plan": self.plan.to_dict(),
                "models": [
                    model.to_dict(include_estimator=include_estimators)
                    for model in self.models
                ],
                "warnings": self.warnings,
                "recommendations": self.recommendations,
            }
        )

    def to_markdown(self, max_terms: int = 12) -> str:
        lines = [
            "# statgpu Automatic Analysis Report",
            "",
            "## Data Profile",
            f"- Task: `{self.profile.task_type}`",
            f"- Samples: {self.profile.n_samples}",
            f"- Features: {self.profile.n_features}",
            f"- Target: {self.profile.target_name or 'None'}",
            f"- Device: `{self.profile.device}`",
        ]
        if self.profile.dropped_rows:
            lines.append(f"- Dropped rows: {self.profile.dropped_rows}")
        if self.profile.imputed_values:
            lines.append(f"- Imputed feature values: {self.profile.imputed_values}")
        if self.profile.target_summary:
            summary = ", ".join(
                f"{key}={_format_metric(value)}"
                for key, value in self.profile.target_summary.items()
            )
            lines.append(f"- Target summary: {summary}")

        lines.extend(["", "## Agent Plan"])
        lines.append("- Agents: " + ", ".join(f"`{name}`" for name in self.plan.agents))
        lines.append("- Methods: " + ", ".join(f"`{name}`" for name in self.plan.methods))
        for reason in self.plan.rationale:
            lines.append(f"- {reason}")

        lines.extend(["", "## Results"])
        for model in self.models:
            lines.append(f"### {model.name}")
            if model.error:
                lines.append(f"- Error: {model.error}")
                continue
            if model.metrics:
                metric_text = ", ".join(
                    f"{key}={_format_metric(value)}"
                    for key, value in model.metrics.items()
                )
                lines.append(f"- Metrics: {metric_text}")
            if model.diagnostics:
                diag_text = ", ".join(
                    f"{key}={_format_metric(value)}"
                    for key, value in model.diagnostics.items()
                    if isinstance(value, (str, int, float, np.integer, np.floating))
                    or value is None
                )
                if diag_text:
                    lines.append(f"- Diagnostics: {diag_text}")
            if model.coefficients:
                lines.append("")
                lines.append("| term | estimate | std_error | statistic | p_value | interval |")
                lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
                for row in model.coefficients[:max_terms]:
                    interval = "NA"
                    if row.get("ci_low") is not None and row.get("ci_high") is not None:
                        interval = (
                            f"[{_format_metric(row.get('ci_low'))}, "
                            f"{_format_metric(row.get('ci_high'))}]"
                        )
                    lines.append(
                        "| {term} | {estimate} | {std_error} | {statistic} | {p_value} | {interval} |".format(
                            term=row.get("term", ""),
                            estimate=_format_metric(row.get("estimate")),
                            std_error=_format_metric(row.get("std_error")),
                            statistic=_format_metric(row.get("statistic")),
                            p_value=_format_metric(row.get("p_value")),
                            interval=interval,
                        )
                    )
            for warning in model.warnings:
                lines.append(f"- Warning: {warning}")

        if self.warnings:
            lines.extend(["", "## Validation Warnings"])
            for warning in self.warnings:
                lines.append(f"- {warning}")

        if self.recommendations:
            lines.extend(["", "## Recommended Next Checks"])
            for recommendation in self.recommendations:
                lines.append(f"- {recommendation}")
        return "\n".join(lines) + "\n"

    def save_markdown(self, path: str, max_terms: int = 12) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_markdown(max_terms=max_terms))


@dataclass
class _PreparedData:
    X: np.ndarray
    y: Optional[np.ndarray]
    time: Optional[np.ndarray]
    event: Optional[np.ndarray]
    feature_names: List[str]
    target_name: Optional[str]
    y_mapping: Dict[str, float]
    dropped_rows: int
    imputed_values: int
    encoded_features: Dict[str, List[str]]
    notes: List[str]


class StatGPUAnalysisAgent:
    """Agentic automatic statistical analysis powered by statgpu.

    The agent follows a deterministic profiler -> planner -> runner ->
    validator -> reporter loop. It does not call a remote LLM; the design is
    meant to be reproducible and easy to audit inside statistical pipelines.
    """

    def __init__(
        self,
        device: str = "auto",
        cov_type: str = "hc3",
        random_state: Optional[int] = 0,
        max_categories: int = 20,
        include_regularized: bool = True,
        include_unsupervised_diagnostics: bool = True,
        gpu_memory_cleanup: bool = False,
    ):
        self.device = device
        self.cov_type = cov_type
        self.random_state = random_state
        self.max_categories = int(max_categories)
        self.include_regularized = bool(include_regularized)
        self.include_unsupervised_diagnostics = bool(include_unsupervised_diagnostics)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

    def analyze_csv(
        self,
        path: str,
        target: Optional[str] = None,
        task: str = "auto",
        time: Optional[str] = None,
        event: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> AnalysisResult:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("CSV file has no data rows.")
        return self.analyze(
            data=rows,
            target=target,
            task=task,
            time=time,
            event=event,
            feature_columns=feature_columns,
        )

    def analyze(
        self,
        data: Any = None,
        X: Any = None,
        y: Any = None,
        target: Optional[str] = None,
        task: str = "auto",
        time: Any = None,
        event: Any = None,
        feature_names: Optional[Sequence[str]] = None,
        feature_columns: Optional[Sequence[str]] = None,
    ) -> AnalysisResult:
        prepared = self._prepare_data(
            data=data,
            X=X,
            y=y,
            target=target,
            time=time,
            event=event,
            feature_names=feature_names,
            feature_columns=feature_columns,
        )
        task_type = self._infer_task(prepared, task)
        profile = self._build_profile(prepared, task_type)
        plan = self._build_plan(task_type, prepared)

        models: List[ModelResult] = []
        if task_type == "regression":
            models.extend(self._run_regression(prepared))
        elif task_type == "binary_classification":
            models.extend(self._run_binary_classification(prepared))
        elif task_type == "poisson":
            models.extend(self._run_poisson(prepared))
        elif task_type == "survival":
            models.extend(self._run_survival(prepared))
        elif task_type == "unsupervised":
            models.extend(self._run_unsupervised(prepared))
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if (
            self.include_unsupervised_diagnostics
            and task_type != "unsupervised"
            and prepared.X.shape[0] >= 3
            and prepared.X.shape[1] >= 2
        ):
            models.extend(self._run_pca_diagnostic(prepared))

        warnings = self._validate(prepared, task_type, models)
        recommendations = self._recommend(prepared, task_type, warnings, models)
        return AnalysisResult(
            profile=profile,
            plan=plan,
            models=models,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _prepare_data(
        self,
        data: Any,
        X: Any,
        y: Any,
        target: Optional[str],
        time: Any,
        event: Any,
        feature_names: Optional[Sequence[str]],
        feature_columns: Optional[Sequence[str]],
    ) -> _PreparedData:
        if data is not None:
            return self._prepare_table(
                data=data,
                target=target,
                time=time,
                event=event,
                feature_columns=feature_columns,
            )
        if X is None:
            raise ValueError("Pass either data=... or X=....")

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
        X_arr, imputed, notes = self._sanitize_numeric_matrix(X_arr, names)

        y_arr, y_mapping = self._prepare_target_array(y) if y is not None else (None, {})
        time_arr = self._prepare_numeric_vector(time, "time") if time is not None else None
        event_arr = self._prepare_event_vector(event) if event is not None else None

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

        return _PreparedData(
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

    def _prepare_table(
        self,
        data: Any,
        target: Optional[str],
        time: Any,
        event: Any,
        feature_columns: Optional[Sequence[str]],
    ) -> _PreparedData:
        columns = self._table_columns(data)
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

        raw_feature_columns = {name: self._table_column(data, name) for name in features}
        n = len(next(iter(raw_feature_columns.values())))

        y_arr = None
        y_mapping: Dict[str, float] = {}
        if target is not None:
            y_arr, y_mapping = self._prepare_target_array(self._table_column(data, target))

        time_arr = None
        if time is not None:
            time_arr = self._prepare_numeric_vector(
                self._table_column(data, time) if isinstance(time, str) else time,
                "time",
            )
        event_arr = None
        if event is not None:
            event_arr = self._prepare_event_vector(
                self._table_column(data, event) if isinstance(event, str) else event
            )

        mask = np.ones(n, dtype=bool)
        for arr in (y_arr, time_arr, event_arr):
            if arr is not None:
                if arr.shape[0] != n:
                    raise ValueError("Response/survival column length must match data rows.")
                mask &= np.isfinite(arr)
        dropped = int(n - np.sum(mask))

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
                if len(categories) > self.max_categories:
                    kept = categories[: self.max_categories]
                    notes.append(
                        f"Collapsed categorical feature '{name}' to top "
                        f"{self.max_categories} levels."
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
        if y_arr is not None:
            y_arr = y_arr[mask]
        if time_arr is not None:
            time_arr = time_arr[mask]
        if event_arr is not None:
            event_arr = event_arr[mask]
        if dropped:
            notes.append(f"Dropped {dropped} rows with missing response/survival fields.")

        X_arr, imputed_more, matrix_notes = self._sanitize_numeric_matrix(X_arr, encoded_names)
        imputed += imputed_more
        notes.extend(matrix_notes)

        return _PreparedData(
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

    def _table_columns(self, data: Any) -> List[str]:
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

    def _table_column(self, data: Any, name: str) -> List[Any]:
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

    def _prepare_target_array(self, y: Any) -> Tuple[np.ndarray, Dict[str, float]]:
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

    def _prepare_numeric_vector(self, values: Any, label: str) -> np.ndarray:
        if values is None:
            raise ValueError(f"{label} is required.")
        arr = np.array([_float_or_nan(v) for v in list(values)], dtype=float)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    def _prepare_event_vector(self, values: Any) -> np.ndarray:
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

    def _sanitize_numeric_matrix(
        self,
        X: np.ndarray,
        feature_names: Sequence[str],
    ) -> Tuple[np.ndarray, int, List[str]]:
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

    def _infer_task(self, prepared: _PreparedData, requested: str) -> str:
        aliases = {
            "auto": "auto",
            "regression": "regression",
            "linear": "regression",
            "classification": "binary_classification",
            "binary": "binary_classification",
            "binary_classification": "binary_classification",
            "poisson": "poisson",
            "count": "poisson",
            "survival": "survival",
            "cox": "survival",
            "unsupervised": "unsupervised",
            "clustering": "unsupervised",
        }
        key = requested.lower()
        if key not in aliases:
            raise ValueError(f"Unknown task='{requested}'.")
        task = aliases[key]
        if task != "auto":
            return task
        if prepared.time is not None and prepared.event is not None:
            return "survival"
        if prepared.y is None:
            return "unsupervised"

        y = prepared.y[np.isfinite(prepared.y)]
        unique = np.unique(y)
        if unique.size == 2:
            return "binary_classification"
        if (
            unique.size > 2
            and np.all(y >= 0)
            and np.allclose(y, np.round(y), atol=1e-8)
            and unique.size <= max(20, int(0.5 * y.size))
        ):
            return "poisson"
        return "regression"

    def _build_profile(self, prepared: _PreparedData, task_type: str) -> DataProfile:
        target_summary: Dict[str, Any] = {}
        if prepared.y is not None:
            y = prepared.y[np.isfinite(prepared.y)]
            target_summary = {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "unique": int(np.unique(y).size),
            }
        elif prepared.time is not None and prepared.event is not None:
            target_summary = {
                "events": int(np.sum(prepared.event == 1)),
                "censored": int(np.sum(prepared.event == 0)),
                "time_min": float(np.min(prepared.time)),
                "time_max": float(np.max(prepared.time)),
            }
        return DataProfile(
            n_samples=int(prepared.X.shape[0]),
            n_features=int(prepared.X.shape[1]),
            task_type=task_type,
            feature_names=list(prepared.feature_names),
            target_name=prepared.target_name,
            device=self.device,
            dropped_rows=prepared.dropped_rows,
            imputed_values=prepared.imputed_values,
            encoded_features=prepared.encoded_features,
            target_summary=target_summary,
            notes=list(prepared.notes),
        )

    def _build_plan(self, task_type: str, prepared: _PreparedData) -> AnalysisPlan:
        agents = [
            "profiler",
            "planner",
            "statgpu_runner",
            "self_validator",
            "reporter",
        ]
        methods: List[str]
        if task_type == "regression":
            methods = ["LinearRegression"]
            if self.include_regularized:
                methods.append("Ridge")
        elif task_type == "binary_classification":
            methods = ["LogisticRegression"]
        elif task_type == "poisson":
            methods = ["PoissonRegression"]
        elif task_type == "survival":
            methods = ["CoxPH"]
        else:
            methods = ["PCA", "KMeans"]
        if (
            self.include_unsupervised_diagnostics
            and task_type != "unsupervised"
            and prepared.X.shape[1] >= 2
        ):
            methods.append("PCA(diagnostic)")

        rationale = [
            "The profiler selected the task from the target and survival fields.",
            "The runner uses statgpu estimators with the configured device policy.",
            "The validator checks missingness, rank, conditioning, class balance, and fit outputs.",
        ]
        if self.cov_type:
            rationale.append(f"Inference models use cov_type='{self.cov_type}' where supported.")
        return AnalysisPlan(task_type=task_type, agents=agents, methods=methods, rationale=rationale)

    def _run_regression(self, prepared: _PreparedData) -> List[ModelResult]:
        assert prepared.y is not None
        results: List[ModelResult] = []
        model = LinearRegression(
            device=self.device,
            compute_inference=True,
            cov_type=self.cov_type,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
        )
        results.append(
            self._fit_supervised_model(
                "LinearRegression",
                model,
                prepared,
                fit_args=(prepared.X, prepared.y),
                score_args=(prepared.X, prepared.y),
            )
        )

        if self.include_regularized:
            ridge = Ridge(
                alpha=1.0,
                device=self.device,
                compute_inference=prepared.X.shape[0] > prepared.X.shape[1] + 1,
                cov_type=self.cov_type,
                gpu_memory_cleanup=self.gpu_memory_cleanup,
            )
            results.append(
                self._fit_supervised_model(
                    "Ridge(alpha=1.0)",
                    ridge,
                    prepared,
                    fit_args=(prepared.X, prepared.y),
                    score_args=(prepared.X, prepared.y),
                )
            )
        return results

    def _run_binary_classification(self, prepared: _PreparedData) -> List[ModelResult]:
        assert prepared.y is not None
        y = self._coerce_binary_y(prepared.y)
        attempts = [
            ("LogisticRegression", 1e10),
            ("LogisticRegression(C=1.0 self-correction)", 1.0),
        ]
        for idx, (name, c_value) in enumerate(attempts):
            model = LogisticRegression(
                C=c_value,
                max_iter=200,
                device=self.device,
                compute_inference=True,
                cov_type=self.cov_type,
                gpu_memory_cleanup=self.gpu_memory_cleanup,
            )
            result = self._fit_supervised_model(
                name,
                model,
                prepared,
                fit_args=(prepared.X, y),
                score_args=(prepared.X, y),
            )
            if result.error is None:
                if idx == 1:
                    result.warnings.append(
                        "Unregularized logistic fit failed; reran with C=1.0."
                    )
                return [result]
        return [result]

    def _run_poisson(self, prepared: _PreparedData) -> List[ModelResult]:
        assert prepared.y is not None
        y = np.asarray(prepared.y, dtype=float)
        model = PoissonRegression(
            device=self.device,
            max_iter=200,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
        )
        return [
            self._fit_supervised_model(
                "PoissonRegression",
                model,
                prepared,
                fit_args=(prepared.X, y),
                score_args=(prepared.X, y),
            )
        ]

    def _run_survival(self, prepared: _PreparedData) -> List[ModelResult]:
        if prepared.time is None or prepared.event is None:
            raise ValueError("Survival analysis requires time and event.")
        model = CoxPH(
            ties="efron",
            device=self.device,
            compute_inference=True,
            compute_cindex=True,
            cov_type="nonrobust" if self.cov_type not in ("hc0", "hc1") else self.cov_type,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
        )
        result = self._fit_survival_model(model, prepared)
        return [result]

    def _run_unsupervised(self, prepared: _PreparedData) -> List[ModelResult]:
        results = self._run_pca_diagnostic(prepared)
        Xs = self._standardize(prepared.X)
        n_clusters = min(8, max(2, int(round(math.sqrt(max(2, prepared.X.shape[0]) / 2.0)))))
        if prepared.X.shape[0] >= n_clusters:
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                device=self.device,
            )
            try:
                model.fit(Xs)
                labels = _to_numpy(model.labels_).astype(int)
                counts = np.bincount(labels, minlength=n_clusters)
                results.append(
                    ModelResult(
                        name=f"KMeans(k={n_clusters})",
                        task_type="unsupervised",
                        estimator=model,
                        metrics={
                            "inertia": float(_to_numpy(model.inertia_)),
                            "n_iter": int(_to_numpy(model.n_iter_)),
                        },
                        diagnostics={"cluster_sizes": counts.tolist()},
                    )
                )
            except Exception as exc:
                results.append(
                    ModelResult(
                        name=f"KMeans(k={n_clusters})",
                        task_type="unsupervised",
                        error=str(exc),
                    )
                )
        return results

    def _run_pca_diagnostic(self, prepared: _PreparedData) -> List[ModelResult]:
        n_components = min(5, prepared.X.shape[0] - 1, prepared.X.shape[1])
        if n_components < 1:
            return []
        Xs = self._standardize(prepared.X)
        model = PCA(
            n_components=n_components,
            random_state=self.random_state,
            device=self.device,
        )
        try:
            model.fit(Xs)
            ratio = _to_numpy(model.explained_variance_ratio_).astype(float)
            metrics = {
                "components": int(n_components),
                "explained_variance_ratio_sum": float(np.sum(ratio)),
            }
            diagnostics = {
                "explained_variance_ratio": ratio.tolist(),
                "top_loading_pc1": self._top_pca_loading(model, prepared.feature_names),
            }
            return [
                ModelResult(
                    name="PCA(diagnostic)",
                    task_type="unsupervised",
                    estimator=model,
                    metrics=metrics,
                    diagnostics=diagnostics,
                )
            ]
        except Exception as exc:
            return [
                ModelResult(
                    name="PCA(diagnostic)",
                    task_type="unsupervised",
                    error=str(exc),
                )
            ]

    def _fit_supervised_model(
        self,
        name: str,
        model: Any,
        prepared: _PreparedData,
        fit_args: Tuple[Any, ...],
        score_args: Tuple[Any, ...],
    ) -> ModelResult:
        try:
            model.fit(*fit_args)
            metrics = self._supervised_metrics(model, name, score_args)
            coefficients = self._extract_coefficients(model, prepared.feature_names, name)
            diagnostics = self._model_diagnostics(model)
            return ModelResult(
                name=name,
                task_type=self._model_task_name(name),
                estimator=model,
                metrics=metrics,
                coefficients=coefficients,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return ModelResult(name=name, task_type=self._model_task_name(name), error=str(exc))

    def _fit_survival_model(self, model: CoxPH, prepared: _PreparedData) -> ModelResult:
        try:
            model.fit(prepared.X, prepared.time, prepared.event)
            metrics = {
                "c_index": getattr(model, "_cindex", None),
                "log_likelihood": getattr(model, "_log_likelihood", None),
            }
            coefficients = self._extract_coefficients(model, prepared.feature_names, "CoxPH")
            diagnostics = {
                "events": int(np.sum(prepared.event == 1)),
                "censored": int(np.sum(prepared.event == 0)),
                "converged": bool(getattr(model, "_converged", False)),
                "iterations": int(getattr(model, "_iterations", 0)),
            }
            return ModelResult(
                name="CoxPH",
                task_type="survival",
                estimator=model,
                metrics=metrics,
                coefficients=coefficients,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return ModelResult(name="CoxPH", task_type="survival", error=str(exc))

    def _supervised_metrics(
        self,
        model: Any,
        name: str,
        score_args: Tuple[Any, ...],
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        score = _safe_call(model.score, *score_args) if hasattr(model, "score") else None
        if score is not None:
            if "Logistic" in name:
                metrics["accuracy"] = score
            else:
                metrics["score"] = score
        if hasattr(model, "aic"):
            aic = _safe_call(model.aic)
            if aic is not None:
                metrics["aic"] = aic
        if hasattr(model, "bic"):
            bic = _safe_call(model.bic)
            if bic is not None:
                metrics["bic"] = bic
        if "Logistic" in name:
            X, y = score_args
            auc = _safe_call(model.roc_auc_score, X, y) if hasattr(model, "roc_auc_score") else None
            if auc is not None:
                metrics["roc_auc"] = auc
            table = _safe_call(model.classification_table, X, y) if hasattr(model, "classification_table") else None
            if isinstance(table, dict):
                for key in ("precision", "recall", "f1"):
                    if key in table:
                        metrics[key] = table[key]
        if "Poisson" in name and hasattr(model, "predict"):
            X, y = score_args
            pred = _safe_call(model.predict, X)
            if pred is not None:
                pred_arr = np.clip(np.asarray(_to_numpy(pred), dtype=float), 1e-12, None)
                y_arr = np.asarray(y, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.where(y_arr == 0, 0.0, y_arr * np.log(y_arr / pred_arr))
                deviance = 2.0 * np.mean(term - (y_arr - pred_arr))
                if np.isfinite(deviance):
                    metrics["mean_poisson_deviance"] = float(deviance)
        return metrics

    def _extract_coefficients(
        self,
        model: Any,
        feature_names: Sequence[str],
        model_name: str,
    ) -> List[Dict[str, Any]]:
        coef = getattr(model, "coef_", None)
        if coef is None:
            return []
        coef_arr = np.asarray(coef, dtype=float)
        if coef_arr.ndim != 1:
            return []

        has_intercept = bool(getattr(model, "fit_intercept", False)) and not isinstance(model, CoxPH)
        names = list(feature_names)
        estimates = coef_arr
        if has_intercept:
            names = ["Intercept"] + names
            estimates = np.concatenate([[float(getattr(model, "intercept_", 0.0))], coef_arr])

        bse = self._aligned_array(getattr(model, "_bse", None), len(estimates))
        pvalues = self._aligned_array(getattr(model, "_pvalues", None), len(estimates))
        conf = getattr(model, "_conf_int", None)
        conf_arr = np.asarray(conf, dtype=float) if conf is not None else None
        zvalues = getattr(model, "_zvalues", None)
        tvalues = getattr(model, "_tvalues", None)
        stat = self._aligned_array(zvalues if zvalues is not None else tvalues, len(estimates))

        rows: List[Dict[str, Any]] = []
        for idx, estimate in enumerate(estimates):
            row: Dict[str, Any] = {
                "term": names[idx] if idx < len(names) else f"x{idx}",
                "estimate": float(estimate),
                "std_error": None if bse is None else float(bse[idx]),
                "statistic": None if stat is None else float(stat[idx]),
                "p_value": None if pvalues is None else float(pvalues[idx]),
                "ci_low": None,
                "ci_high": None,
            }
            if conf_arr is not None and conf_arr.ndim == 2 and idx < conf_arr.shape[0]:
                row["ci_low"] = float(conf_arr[idx, 0])
                row["ci_high"] = float(conf_arr[idx, 1])
            if "Logistic" in model_name:
                row["odds_ratio"] = float(np.exp(np.clip(estimate, -700, 700)))
            if isinstance(model, CoxPH):
                row["hazard_ratio"] = float(np.exp(np.clip(estimate, -700, 700)))
            rows.append(row)

        def sort_key(row):
            p_value = row.get("p_value")
            if p_value is None or not np.isfinite(p_value):
                return (1, -abs(row["estimate"]))
            return (0, p_value)

        intercept = [row for row in rows if row["term"] == "Intercept"]
        non_intercept = [row for row in rows if row["term"] != "Intercept"]
        non_intercept.sort(key=sort_key)
        return intercept + non_intercept

    def _aligned_array(self, value: Any, expected_len: int) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != expected_len:
            return None
        return arr

    def _model_diagnostics(self, model: Any) -> Dict[str, Any]:
        diagnostics = {}
        for attr, key in (
            ("n_iter_", "n_iter"),
            ("_df_resid", "df_resid"),
            ("_nobs", "nobs"),
            ("_loglik", "log_likelihood"),
        ):
            if hasattr(model, attr):
                value = getattr(model, attr)
                if value is not None:
                    diagnostics[key] = value
        return diagnostics

    def _model_task_name(self, model_name: str) -> str:
        if "Logistic" in model_name:
            return "binary_classification"
        if "Poisson" in model_name:
            return "poisson"
        return "regression"

    def _coerce_binary_y(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        unique = np.unique(values[np.isfinite(values)])
        if unique.size != 2:
            raise ValueError("Binary classification requires exactly two target values.")
        return (values == unique.max()).astype(float)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return (X - mean) / scale

    def _top_pca_loading(self, model: PCA, feature_names: Sequence[str]) -> Optional[str]:
        components = _to_numpy(model.components_)
        if components.size == 0:
            return None
        idx = int(np.argmax(np.abs(components[0])))
        return feature_names[idx] if idx < len(feature_names) else f"x{idx}"

    def _validate(
        self,
        prepared: _PreparedData,
        task_type: str,
        models: Sequence[ModelResult],
    ) -> List[str]:
        warnings = list(prepared.notes)
        n, p = prepared.X.shape
        if n < 30:
            warnings.append("Sample size is below 30; inference may be unstable.")
        if p + 1 >= n and task_type in ("regression", "binary_classification", "poisson", "survival"):
            warnings.append("Feature count is close to or greater than sample size.")
        if p <= 500 and n >= 2:
            rank = int(np.linalg.matrix_rank(prepared.X))
            if rank < p:
                warnings.append(f"Design matrix rank is {rank} < {p}; coefficients may be non-identifiable.")
            try:
                condition = float(np.linalg.cond(prepared.X))
                if np.isfinite(condition) and condition > 1e8:
                    warnings.append(f"Design matrix condition number is high ({condition:.2e}).")
            except Exception:
                pass
        if task_type == "binary_classification" and prepared.y is not None:
            y = self._coerce_binary_y(prepared.y)
            pos_rate = float(np.mean(y))
            if pos_rate < 0.1 or pos_rate > 0.9:
                warnings.append(f"Binary target is imbalanced (positive rate={pos_rate:.3f}).")
        if task_type == "survival" and prepared.event is not None:
            events = int(np.sum(prepared.event == 1))
            if events < max(10, p):
                warnings.append("Number of observed events is low relative to model size.")

        if all(model.error for model in models):
            warnings.append("All planned model fits failed.")
        for model in models:
            if model.error:
                warnings.append(f"{model.name} failed: {model.error}")
            if model.coefficients and all(row.get("p_value") is None for row in model.coefficients):
                if model.task_type in ("regression", "binary_classification", "survival"):
                    warnings.append(f"{model.name} did not return coefficient p-values.")
        warnings.append("Reported predictive metrics are training-set diagnostics, not holdout estimates.")
        return self._deduplicate(warnings)

    def _recommend(
        self,
        prepared: _PreparedData,
        task_type: str,
        warnings: Sequence[str],
        models: Sequence[ModelResult],
    ) -> List[str]:
        recs = ["Confirm the automatically inferred task and target definition before using results."]
        if task_type in ("regression", "binary_classification", "poisson"):
            recs.append("Run a held-out or cross-validated evaluation for predictive claims.")
        if any("condition number" in warning or "rank" in warning for warning in warnings):
            recs.append("Inspect collinearity and consider removing or combining redundant features.")
        if any("imbalanced" in warning for warning in warnings):
            recs.append("Report threshold-specific classification metrics and class prevalence.")
        if prepared.encoded_features:
            recs.append("Review one-hot encoded categorical features and their reference levels.")
        if any(model.name.startswith("PCA") and not model.error for model in models):
            recs.append("Use PCA diagnostics to identify dominant feature groups or batch effects.")
        return self._deduplicate(recs)

    def _deduplicate(self, values: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for value in values:
            if value and value not in seen:
                out.append(value)
                seen.add(value)
        return out
