"""Validation, multiple testing correction, and diagnostics."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ._config import AgentConfig


class ValidationRuleRegistry:
    """Pluggable registry of validation rules."""

    _rules: Dict[str, List[Callable]] = {}

    @classmethod
    def register(cls, task_type: str, rule: Callable):
        """Register a validation rule for a task type."""
        cls._rules.setdefault(task_type, []).append(rule)

    @classmethod
    def run_custom_rules(cls, task_type: str, prepared, models) -> List[str]:
        """Run custom validation rules and return warnings."""
        warnings = []
        for rule in cls._rules.get(task_type, []):
            try:
                warnings.extend(rule(prepared, models))
            except Exception:
                pass
        return warnings


def _coerce_binary_y(y: np.ndarray) -> np.ndarray:
    """Coerce target to binary 0/1.

    Maps the larger unique value to 1.0 and the smaller to 0.0.
    Assumes target has already been numeric-encoded by prepare_target_array()
    (string targets like "alive"/"dead" are encoded as 0.0/1.0 upstream).
    """
    values = np.asarray(y, dtype=float)
    unique = np.unique(values[np.isfinite(values)])
    if unique.size != 2:
        raise ValueError("Binary classification requires exactly two target values.")
    return (values == unique.max()).astype(float)


def validate(
    prepared,
    task_type: str,
    models: Sequence,
    config: AgentConfig,
) -> List[str]:
    """Run validation checks and return warnings."""
    warnings = list(prepared.notes)
    n, p = prepared.X.shape

    if n < config.min_sample_size_warn:
        warnings.append(f"Sample size is below {config.min_sample_size_warn}; inference may be unstable.")

    if p + 1 >= n and task_type in ("regression", "binary_classification", "poisson", "survival"):
        warnings.append("Feature count is close to or greater than sample size.")

    if p <= 500 and n >= 2:
        rank = int(np.linalg.matrix_rank(prepared.X))
        if rank < p:
            warnings.append(f"Design matrix rank is {rank} < {p}; coefficients may be non-identifiable.")
        try:
            condition = float(np.linalg.cond(prepared.X))
            if np.isfinite(condition) and condition > config.condition_number_threshold:
                warnings.append(f"Design matrix condition number is high ({condition:.2e}).")
        except Exception:
            pass

    if task_type == "binary_classification" and prepared.y is not None:
        y = _coerce_binary_y(prepared.y)
        pos_rate = float(np.mean(y))
        if pos_rate < config.imbalance_low_threshold or pos_rate > config.imbalance_high_threshold:
            warnings.append(f"Binary target is imbalanced (positive rate={pos_rate:.3f}).")

    if task_type == "survival" and prepared.event is not None:
        events = int(np.sum(prepared.event == 1))
        if events < config.min_events_per_feature * p:
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

    # Suggest multiple testing correction when many features
    if (
        config.multiple_testing_method == "none"
        and p >= 10
        and task_type in ("regression", "binary_classification", "survival")
    ):
        warnings.append(
            f"Feature count is {p}. Consider --multiple-testing bh for exploratory analysis "
            "or --multiple-testing holm for confirmatory analysis."
        )

    # Run custom validation rules
    custom_warnings = ValidationRuleRegistry.run_custom_rules(task_type, prepared, models)
    warnings.extend(custom_warnings)

    return _deduplicate(warnings)


def apply_multiple_testing_correction(
    coefficients: List[Dict[str, Any]],
    method: str = "none",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Add adjusted p-values when method != 'none'. Always preserve raw p-values."""
    if method == "none":
        return coefficients

    non_intercept_indices = [
        i for i, r in enumerate(coefficients)
        if r["term"] != "Intercept" and r.get("p_value") is not None
    ]
    raw_pvalues = np.array([coefficients[i]["p_value"] for i in non_intercept_indices])

    if raw_pvalues.size < 2:
        return coefficients  # Need at least 2 tests for correction

    try:
        from statgpu.inference import adjust_pvalues
        reject, pval_adj = adjust_pvalues(raw_pvalues, method=method, alpha=alpha)
        for idx, (coef_idx, rej, padj) in enumerate(
            zip(non_intercept_indices, reject, pval_adj)
        ):
            coefficients[coef_idx]["adj_p_value"] = float(padj)
            coefficients[coef_idx]["rejected"] = bool(rej)
    except Exception:
        pass  # If correction fails, leave raw p-values unchanged

    return coefficients


def run_diagnostics(model, prepared) -> Dict[str, Any]:
    """Extract regression diagnostics from fitted model."""
    diagnostics: Dict[str, Any] = {}

    # Basic diagnostics
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

    # VIF, Cook's distance, leverage (only for models with residual info)
    if hasattr(model, "_resid") and model._resid is not None:
        try:
            from statgpu.diagnostics import RegressionDiagnostics
            diag = RegressionDiagnostics(model)
            vif_values = diag.vif()
            cooks = diag.cooks_distance
            leverage = diag.leverage

            diagnostics["vif_max"] = float(np.max(vif_values))
            diagnostics["vif_high_features"] = [
                prepared.feature_names[i]
                for i, v in enumerate(vif_values)
                if i < len(prepared.feature_names) and v > 10
            ]
            diagnostics["influential_points"] = int(np.sum(cooks > 1.0))
            n_params = len(getattr(model, "_params", []))
            if n_params > 0:
                diagnostics["high_leverage_points"] = int(np.sum(
                    leverage > 2 * n_params / len(leverage)
                ))
        except Exception:
            pass  # Diagnostics are best-effort

    return diagnostics


def recommend(
    prepared,
    task_type: str,
    warnings: Sequence[str],
    models: Sequence,
) -> List[str]:
    """Generate recommendations based on analysis results."""
    recs = ["Confirm the automatically inferred task and target definition before using results."]
    if task_type in ("regression", "binary_classification", "poisson"):
        recs.append("Run a held-out or cross-validated evaluation for predictive claims.")
    if any("condition number" in warning or "rank" in warning for warning in warnings):
        recs.append("Inspect collinearity and consider removing or combining redundant features.")
    if any("imbalanced" in warning for warning in warnings):
        recs.append("Report threshold-specific classification metrics and class prevalence.")
    if prepared.encoded_features:
        recs.append("Review one-hot encoded categorical features and their reference levels.")
    if any(getattr(m, "name", "").startswith("PCA") and not m.error for m in models):
        recs.append("Use PCA diagnostics to identify dominant feature groups or batch effects.")
    return _deduplicate(recs)


def _deduplicate(values: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out
