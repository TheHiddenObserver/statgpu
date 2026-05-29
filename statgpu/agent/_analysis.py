"""Automatic statistical analysis agent built on top of statgpu estimators."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._config import AgentConfig
from ._memory import MemoryStore
from ._profiler import prepare_table, prepare_array
from ._planner import infer_task, build_plan, MethodPruner
from ._runner import (
    run_unsupervised,
    run_pca_diagnostic,
)
from ._validator import validate, recommend, run_diagnostics
from ._reporter import to_markdown, save_markdown as _save_markdown, save_json as _save_json, save_notebook as _save_notebook
from ._model_comparison import ModelComparator
from ._cross_validation import AgentCrossValidator, CVResult


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
        return _json_ready({
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
        })


@dataclass
class AnalysisPlan:
    """Planned agent stages and statgpu methods."""

    task_type: str
    agents: List[str]
    methods: List[str]
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _json_ready({
            "task_type": self.task_type,
            "agents": self.agents,
            "methods": self.methods,
            "rationale": self.rationale,
        })


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
    cv_results: Optional[CVResult] = None
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)

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
        if self.cv_results is not None:
            out["cv_results"] = self.cv_results.to_dict()
        if self.feature_importance:
            out["feature_importance"] = self.feature_importance
        return _json_ready(out)


@dataclass
class AnalysisResult:
    """Complete automatic analysis result."""

    profile: DataProfile
    plan: AnalysisPlan
    models: List[ModelResult]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_trace: List[Dict[str, Any]] = field(default_factory=list)
    comparison: Any = None  # Optional[ModelComparison]

    def to_dict(self, include_estimators: bool = False) -> Dict[str, Any]:
        out = _json_ready({
            "profile": self.profile.to_dict(),
            "plan": self.plan.to_dict(),
            "models": [
                model.to_dict(include_estimator=include_estimators)
                for model in self.models
            ],
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        })
        if self.validation_trace:
            out["validation_trace"] = _json_ready(self.validation_trace)
        if self.comparison is not None:
            out["comparison"] = self.comparison.to_dict()
        return out

    def to_markdown(self, max_terms: int = 12) -> str:
        return to_markdown(self, max_terms=max_terms)

    def save_markdown(self, path: str, max_terms: int = 12) -> None:
        _save_markdown(self, path, max_terms=max_terms)

    def save_json(self, path: str, include_estimators: bool = False) -> None:
        _save_json(self, path, include_estimators=include_estimators)

    def save_notebook(self, data_source: str, output_path: str) -> None:
        _save_notebook(self, data_source, output_path)


@dataclass
class PreparedData:
    """Internal data container for prepared data."""

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


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class StatGPUAnalysisAgent:
    """Agentic automatic statistical analysis powered by statgpu.

    The agent follows a deterministic profiler -> planner -> runner ->
    validator -> reporter loop with proactive pruning and self-correction.
    It does not call a remote LLM; the design is meant to be reproducible
    and easy to audit inside statistical pipelines.
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
        cv_folds: int = 5,
        multiple_testing_method: str = "none",
        alpha: float = 0.05,
        memory_path: Optional[str] = None,
    ):
        self.config = AgentConfig(
            device=device,
            cov_type=cov_type,
            random_state=random_state,
            max_categories=int(max_categories),
            include_regularized=bool(include_regularized),
            include_unsupervised_diagnostics=bool(include_unsupervised_diagnostics),
            gpu_memory_cleanup=bool(gpu_memory_cleanup),
            cv_folds=int(cv_folds),
            multiple_testing_method=multiple_testing_method,
            alpha=float(alpha),
        )
        self._memory_store = MemoryStore(memory_path) if memory_path else None

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
        # Stage 1: Profile
        if data is not None:
            prepared = prepare_table(data, target, time, event, feature_columns, self.config)
        elif X is not None:
            prepared = prepare_array(X, y, time, event, feature_names, target)
        else:
            raise ValueError("Pass either data=... or X=....")

        # Stage 2: Plan (with pruning)
        task_type = infer_task(prepared, task)
        profile = self._build_profile(prepared, task_type)

        available_methods = MethodPruner().prune(prepared, task_type, self.config)
        plan = build_plan(task_type, prepared, available_methods, self.config)

        # Stage 3: Run models using pruned method list
        models = self._run_with_methods(prepared, task_type, available_methods)

        # Self-correction loop (GeneAgent pattern)
        trace: List[dict] = []
        for round_num in range(self.config.max_correction_rounds):
            issues = self._diagnose(models, prepared, task_type)
            if not issues:
                break
            correction = self._plan_correction(issues, prepared, task_type)
            trace.append({
                "round": round_num,
                "issues": issues,
                "correction": correction,
                "methods_before": [m.name for m in models],
            })
            # Re-run with corrected methods
            corrected_methods = correction.get("methods", available_methods)
            if corrected_methods and corrected_methods != available_methods:
                available_methods = corrected_methods
                models = self._run_with_methods(prepared, task_type, available_methods)
            else:
                break  # No actionable correction possible

        # Stage 4: Cross-validation (if enabled)
        if self.config.cv_folds > 0 and task_type != "unsupervised":
            from ._cross_validation import AgentCrossValidator
            from ._validator import _coerce_binary_y
            cv = AgentCrossValidator(
                n_folds=self.config.cv_folds,
                random_state=self.config.random_state,
            )
            # Prepare CV target (coerce binary y)
            cv_y = prepared.y
            if task_type == "binary_classification" and prepared.y is not None:
                try:
                    cv_y = _coerce_binary_y(prepared.y)
                except Exception:
                    cv_y = prepared.y

            for model in models:
                if model.error is not None or model.estimator is None:
                    continue
                try:
                    est = model.estimator
                    est_class = est.__class__
                    est_params = est.get_params()
                    est_params.pop('device', None)
                    factory = lambda c=est_class, p=dict(est_params): c(**p)

                    if task_type == "survival" and hasattr(prepared, 'time') and prepared.time is not None:
                        model.cv_results = cv.evaluate_survival(
                            factory, prepared.X, prepared.time, prepared.event,
                        )
                    elif cv_y is not None:
                        model.cv_results = cv.evaluate_supervised(
                            factory, prepared.X, cv_y, task_type,
                        )
                except Exception:
                    pass  # CV is best-effort

        # Stage 5: Validate
        warnings = validate(prepared, task_type, models, self.config)
        recommendations = recommend(prepared, task_type, warnings, models)

        # Model comparison
        comparator = ModelComparator()
        comparison = comparator.compare(models, task_type)

        # Record to memory
        successful = [m.name for m in models if m.error is None]
        failed = [f"{m.name}: {m.error}" for m in models if m.error is not None]
        if self._memory_store is not None:
            self._memory_store.record_analysis(
                n_samples=prepared.X.shape[0],
                n_features=prepared.X.shape[1],
                task_type=task_type,
                successful=successful,
                failed=failed,
            )

        return AnalysisResult(
            profile=profile,
            plan=plan,
            models=models,
            warnings=warnings,
            recommendations=recommendations,
            validation_trace=trace,
            comparison=comparison,
        )

    def _run_with_methods(self, prepared, task_type: str, methods: List[str]) -> List[ModelResult]:
        """Run models using specified methods."""
        from ._runner import (
            fit_supervised_model, fit_survival_model,
            run_pca_diagnostic, run_unsupervised,
        )
        from ._planner import MethodRegistry

        models: List[ModelResult] = []

        # For unsupervised, delegate to the existing runner
        if task_type == "unsupervised":
            return run_unsupervised(prepared, self.config)

        # Fit each method from the pruned list
        for method_name in methods:
            # Look up in registry
            entry = None
            for m in MethodRegistry.get_methods(task_type):
                if m["name"] == method_name:
                    entry = m
                    break

            if entry is not None:
                try:
                    # Pass config to factory (new signature supports cfg= parameter)
                    import inspect
                    sig = inspect.signature(entry["factory"])
                    if "cfg" in sig.parameters:
                        estimator = entry["factory"](cfg=self.config)
                    else:
                        estimator = entry["factory"]()

                    # Apply config parameters to estimator
                    if hasattr(estimator, 'device'):
                        from statgpu._config import Device
                        device_val = self.config.device
                        if isinstance(device_val, str):
                            try:
                                device_val = Device(device_val)
                            except ValueError:
                                pass
                        estimator.device = device_val
                    if hasattr(estimator, 'cov_type') and self.config.cov_type:
                        # CoxPH only supports nonrobust/hc0/hc1/cluster
                        from statgpu.survival import CoxPH
                        if isinstance(estimator, CoxPH):
                            safe_cov_types = {"nonrobust", "hc0", "hc1", "cluster"}
                            estimator.cov_type = self.config.cov_type if self.config.cov_type in safe_cov_types else "nonrobust"
                        else:
                            estimator.cov_type = self.config.cov_type
                    if hasattr(estimator, 'compute_inference'):
                        estimator.compute_inference = True
                    if hasattr(estimator, 'gpu_memory_cleanup'):
                        estimator.gpu_memory_cleanup = self.config.gpu_memory_cleanup

                    if task_type == "survival":
                        result = fit_survival_model(estimator, prepared, self.config)
                    else:
                        if prepared.y is None:
                            continue
                        # Binary classification: coerce y to 0/1
                        if task_type == "binary_classification":
                            from ._validator import _coerce_binary_y
                            y = _coerce_binary_y(prepared.y)
                        else:
                            y = prepared.y
                        result = fit_supervised_model(
                            method_name, estimator, prepared,
                            fit_args=(prepared.X, y),
                            score_args=(prepared.X, y),
                            task_type=task_type, config=self.config,
                        )
                    models.append(result)
                except Exception as exc:
                    models.append(ModelResult(name=method_name, task_type=task_type, error=str(exc)))
            elif "PCA" in method_name:
                models.extend(run_pca_diagnostic(prepared, self.config))
            else:
                # Unknown method - report error instead of silent skip
                models.append(ModelResult(
                    name=method_name, task_type=task_type,
                    error=f"Method '{method_name}' not found in registry",
                ))

        # Always run PCA diagnostic for supervised tasks if not already included
        has_pca = any("PCA" in m.name for m in models)
        if (
            self.config.include_unsupervised_diagnostics
            and task_type != "unsupervised"
            and not has_pca
            and prepared.X.shape[0] >= 3
            and prepared.X.shape[1] >= 2
        ):
            models.extend(run_pca_diagnostic(prepared, self.config))

        return models

    def _build_profile(self, prepared: PreparedData, task_type: str) -> DataProfile:
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
            device=self.config.device,
            dropped_rows=prepared.dropped_rows,
            imputed_values=prepared.imputed_values,
            encoded_features=prepared.encoded_features,
            target_summary=target_summary,
            notes=list(prepared.notes),
        )

    def _diagnose(self, models, prepared, task_type: str) -> List[str]:
        """Diagnose issues with fitted models."""
        issues = []
        for model in models:
            if model.error:
                err = str(model.error).lower()
                if "singular" in err or "rank" in err:
                    issues.append("rank_deficient")
                elif "separation" in err:
                    issues.append("separation")
                elif "convergence" in err:
                    issues.append("non_convergence")
        return issues

    def _plan_correction(self, issues: List[str], prepared,
                         task_type: str) -> dict:
        """Plan correction based on diagnosed issues."""
        n, p = prepared.X.shape
        if "rank_deficient" in issues:
            return {"action": "switch_to_ridge", "methods": ["Ridge(alpha=1.0)"]}
        if "separation" in issues:
            return {"action": "already_corrected", "methods": []}
        return {"action": "unknown", "methods": []}

    def save_pipeline(self, path: str, result: AnalysisResult) -> None:
        """Save fitted pipeline (config + estimator states) for later reuse.

        Loaded estimators can be used for predict() but not fit().

        SECURITY WARNING: pickle files can execute arbitrary code on load.
        Only load pipeline files from trusted sources.
        """
        import pickle

        serializable_models = []
        for m in result.models:
            model_data = {
                "name": m.name,
                "task_type": m.task_type,
                "metrics": m.metrics,
                "coefficients": m.coefficients,
                "diagnostics": m.diagnostics,
                "feature_importance": m.feature_importance,
                "warnings": m.warnings,
                "error": m.error,
            }
            if m.estimator is not None:
                try:
                    model_data["estimator"] = m.estimator
                except Exception:
                    pass
            serializable_models.append(model_data)

        state = {
            "config": {
                "device": self.config.device,
                "cov_type": self.config.cov_type,
                "cv_folds": self.config.cv_folds,
                "multiple_testing_method": self.config.multiple_testing_method,
            },
            "profile": result.profile.to_dict(),
            "plan": result.plan.to_dict(),
            "models": serializable_models,
            "warnings": result.warnings,
            "recommendations": result.recommendations,
            "validation_trace": result.validation_trace,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load_pipeline(path: str) -> dict:
        """Load a previously saved pipeline state.

        Returns a dict with keys: config, profile, plan, models, warnings,
        recommendations, validation_trace. Loaded estimators can be used
        for predict() but not fit().

        SECURITY WARNING: pickle files can execute arbitrary code on load.
        Only load pipeline files from trusted sources.
        """
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# Alias for backward compatibility
AutoAnalysisAgent = StatGPUAnalysisAgent
