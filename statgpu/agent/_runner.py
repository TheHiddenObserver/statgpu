"""Model fitting, self-correction, and multi-candidate competition."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._config import AgentConfig
from ._planner import MethodRegistry, MethodPruner


def compute_feature_importance(model, X: np.ndarray, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Compute feature importance via standardized coefficient magnitude."""
    coef = getattr(model, 'coef_', None)
    if coef is None:
        return []
    coef_arr = np.asarray(coef, dtype=float).ravel()
    if coef_arr.size == 0:
        return []

    X_std = np.std(np.asarray(X, dtype=float), axis=0)
    X_std = np.where(X_std > 0, X_std, 1.0)
    n = min(len(coef_arr), len(X_std), len(feature_names))
    std_coef = np.abs(coef_arr[:n] * X_std[:n])
    total = std_coef.sum()
    if total > 0:
        std_coef = std_coef / total

    return sorted(
        [{"feature": feature_names[i], "importance": float(std_coef[i])}
         for i in range(n)],
        key=lambda x: -x["importance"],
    )


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


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


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (X - mean) / scale


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_supervised_model(
    name: str,
    model: Any,
    prepared,
    fit_args: Tuple[Any, ...],
    score_args: Tuple[Any, ...],
    task_type: str,
    config: AgentConfig,
) -> "ModelResult":
    """Fit a supervised model and extract results."""
    from ._analysis import ModelResult

    try:
        model.fit(*fit_args)
        metrics = _supervised_metrics(model, name, score_args)
        coefficients = extract_coefficients(model, prepared.feature_names, name)
        diagnostics = _model_diagnostics(model)

        # Apply multiple testing correction
        from ._validator import apply_multiple_testing_correction
        coefficients = apply_multiple_testing_correction(
            coefficients, config.multiple_testing_method, config.alpha
        )

        feature_importance = compute_feature_importance(model, prepared.X, prepared.feature_names)

        return ModelResult(
            name=name,
            task_type=task_type,
            estimator=model,
            metrics=metrics,
            coefficients=coefficients,
            diagnostics=diagnostics,
            feature_importance=feature_importance,
        )
    except Exception as exc:
        return ModelResult(name=name, task_type=task_type, error=str(exc))


def fit_survival_model(model, prepared, config: AgentConfig) -> "ModelResult":
    """Fit a survival model and extract results."""
    from ._analysis import ModelResult

    try:
        model.fit(prepared.X, prepared.time, prepared.event)
        metrics = {
            "c_index": getattr(model, "_cindex", None),
            "log_likelihood": getattr(model, "_log_likelihood", None),
        }
        coefficients = extract_coefficients(model, prepared.feature_names, "CoxPH")

        from ._validator import apply_multiple_testing_correction
        coefficients = apply_multiple_testing_correction(
            coefficients, config.multiple_testing_method, config.alpha
        )

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


def extract_coefficients(model, feature_names: Sequence[str], model_name: str) -> List[Dict[str, Any]]:
    """Extract coefficient table from fitted model."""
    from statgpu.survival import CoxPH

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

    bse = _aligned_array(getattr(model, "_bse", None), len(estimates))
    pvalues = _aligned_array(getattr(model, "_pvalues", None), len(estimates))
    conf = getattr(model, "_conf_int", None)
    conf_arr = np.asarray(conf, dtype=float) if conf is not None else None
    zvalues = getattr(model, "_zvalues", None)
    tvalues = getattr(model, "_tvalues", None)
    stat = _aligned_array(zvalues if zvalues is not None else tvalues, len(estimates))

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


def _aligned_array(value: Any, expected_len: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        return None
    return arr


def _supervised_metrics(model: Any, name: str, score_args: Tuple[Any, ...]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    score = _safe_call(model.score, *score_args) if hasattr(model, "score") else None
    if score is not None:
        if "Logistic" in name:
            metrics["accuracy"] = score
        else:
            metrics["score"] = score
    if hasattr(model, "aic"):
        try:
            aic_val = model.aic
            if aic_val is not None and np.isfinite(float(aic_val)):
                metrics["aic"] = float(aic_val)
        except Exception:
            pass
    if hasattr(model, "bic"):
        try:
            bic_val = model.bic
            if bic_val is not None and np.isfinite(float(bic_val)):
                metrics["bic"] = float(bic_val)
        except Exception:
            pass
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


def _model_diagnostics(model: Any) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Task-specific runners
# ---------------------------------------------------------------------------

def run_regression(prepared, config: AgentConfig) -> List["ModelResult"]:
    """Fit regression models."""
    from statgpu.linear_model import LinearRegression, Ridge
    from ._analysis import ModelResult

    assert prepared.y is not None
    results: List[ModelResult] = []
    device = config.device

    model = LinearRegression(
        device=device,
        compute_inference=True,
        cov_type=config.cov_type,
        gpu_memory_cleanup=config.gpu_memory_cleanup,
    )
    results.append(
        fit_supervised_model(
            "LinearRegression", model, prepared,
            fit_args=(prepared.X, prepared.y),
            score_args=(prepared.X, prepared.y),
            task_type="regression", config=config,
        )
    )

    if config.include_regularized:
        ridge = Ridge(
            alpha=1.0,
            device=device,
            compute_inference=prepared.X.shape[0] > prepared.X.shape[1] + 1,
            cov_type=config.cov_type,
            gpu_memory_cleanup=config.gpu_memory_cleanup,
        )
        results.append(
            fit_supervised_model(
                "Ridge(alpha=1.0)", ridge, prepared,
                fit_args=(prepared.X, prepared.y),
                score_args=(prepared.X, prepared.y),
                task_type="regression", config=config,
            )
        )
    return results


def run_binary_classification(prepared, config: AgentConfig) -> List["ModelResult"]:
    """Fit binary classification models with self-correction."""
    from statgpu.linear_model import LogisticRegression
    from ._analysis import ModelResult
    from ._validator import _coerce_binary_y

    assert prepared.y is not None
    y = _coerce_binary_y(prepared.y)
    device = config.device

    attempts = [
        ("LogisticRegression", 1e10),
        ("LogisticRegression(C=1.0 self-correction)", 1.0),
    ]
    result = None
    for idx, (name, c_value) in enumerate(attempts):
        model = LogisticRegression(
            C=c_value,
            max_iter=200,
            device=device,
            compute_inference=True,
            cov_type=config.cov_type,
            gpu_memory_cleanup=config.gpu_memory_cleanup,
        )
        result = fit_supervised_model(
            name, model, prepared,
            fit_args=(prepared.X, y),
            score_args=(prepared.X, y),
            task_type="binary_classification", config=config,
        )
        if result.error is None:
            if idx == 1:
                result.warnings.append("Unregularized logistic fit failed; reran with C=1.0.")
            return [result]
    return [result]


def run_poisson(prepared, config: AgentConfig) -> List["ModelResult"]:
    """Fit Poisson regression model."""
    from statgpu.linear_model import PoissonRegression
    from ._analysis import ModelResult

    assert prepared.y is not None
    y = np.asarray(prepared.y, dtype=float)
    model = PoissonRegression(
        device=config.device,
        max_iter=200,
        gpu_memory_cleanup=config.gpu_memory_cleanup,
    )
    return [
        fit_supervised_model(
            "PoissonRegression", model, prepared,
            fit_args=(prepared.X, y),
            score_args=(prepared.X, y),
            task_type="poisson", config=config,
        )
    ]


def run_survival(prepared, config: AgentConfig) -> List["ModelResult"]:
    """Fit survival model."""
    from statgpu.survival import CoxPH

    if prepared.time is None or prepared.event is None:
        raise ValueError("Survival analysis requires time and event.")
    model = CoxPH(
        ties="efron",
        device=config.device,
        compute_inference=True,
        compute_cindex=True,
        cov_type="nonrobust" if config.cov_type not in ("hc0", "hc1") else config.cov_type,
        gpu_memory_cleanup=config.gpu_memory_cleanup,
    )
    return [fit_survival_model(model, prepared, config)]


def _fit_kmeans(Xs: np.ndarray, k: int, prepared, config: AgentConfig) -> "ModelResult":
    """Fit KMeans with k clusters."""
    from statgpu.unsupervised import KMeans
    from ._analysis import ModelResult

    model = KMeans(n_clusters=k, random_state=config.random_state, device=config.device)
    try:
        model.fit(Xs)
        labels = _to_numpy(model.labels_).astype(int)
        counts = np.bincount(labels, minlength=k)
        return ModelResult(
            name=f"KMeans(k={k})",
            task_type="unsupervised",
            estimator=model,
            metrics={
                "inertia": float(_to_numpy(model.inertia_)),
                "n_iter": int(_to_numpy(model.n_iter_)),
            },
            diagnostics={"cluster_sizes": counts.tolist()},
        )
    except Exception as exc:
        return ModelResult(name=f"KMeans(k={k})", task_type="unsupervised", error=str(exc))


def _fit_gmm(Xs: np.ndarray, prepared, config: AgentConfig) -> "ModelResult":
    """Fit GaussianMixture with BIC model selection."""
    from statgpu.unsupervised import GaussianMixture
    from ._analysis import ModelResult

    n = Xs.shape[0]
    max_k = min(8, n // 5)  # At least 5 samples per component
    if max_k < 2:
        return ModelResult(name="GaussianMixture", task_type="unsupervised",
                           error="Too few samples for GMM")

    best_bic = float("inf")
    best_model = None
    best_k = 2
    bic_values = {}

    for k in range(2, max_k + 1):
        try:
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=config.random_state,
                device=config.device,
            )
            model.fit(Xs)
            bic = float(_to_numpy(model.bic(Xs)))
            bic_values[k] = bic
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_k = k
        except Exception:
            continue

    if best_model is None:
        return ModelResult(name="GaussianMixture", task_type="unsupervised",
                           error="All GMM fits failed")

    labels = _to_numpy(best_model.predict(Xs)).astype(int)
    counts = np.bincount(labels, minlength=best_k)
    return ModelResult(
        name=f"GaussianMixture(k={best_k},bic)",
        task_type="unsupervised",
        estimator=best_model,
        metrics={
            "bic": best_bic,
            "n_components": best_k,
        },
        diagnostics={
            "cluster_sizes": counts.tolist(),
            "bic_values": {str(k): v for k, v in bic_values.items()},
        },
    )


def run_unsupervised(prepared, config: AgentConfig) -> List["ModelResult"]:
    """Fit unsupervised models: PCA + multi-k KMeans + GaussianMixture."""
    results = run_pca_diagnostic(prepared, config)
    Xs = _standardize(prepared.X)
    n = prepared.X.shape[0]

    # KMeans with multiple k values
    for k in [2, 3, 5]:
        if n >= k:
            results.append(_fit_kmeans(Xs, k, prepared, config))

    # GaussianMixture with BIC model selection
    if n >= 10:
        results.append(_fit_gmm(Xs, prepared, config))

    return results


def run_pca_diagnostic(prepared, config: AgentConfig) -> List["ModelResult"]:
    """Run PCA diagnostic analysis."""
    from statgpu.unsupervised import PCA
    from ._analysis import ModelResult

    n_components = min(5, prepared.X.shape[0] - 1, prepared.X.shape[1])
    if n_components < 1:
        return []
    Xs = _standardize(prepared.X)
    model = PCA(
        n_components=n_components,
        random_state=config.random_state,
        device=config.device,
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
            "top_loading_pc1": _top_pca_loading(model, prepared.feature_names),
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
            ModelResult(name="PCA(diagnostic)", task_type="unsupervised", error=str(exc))
        ]


def _top_pca_loading(model, feature_names: Sequence[str]) -> Optional[str]:
    components = _to_numpy(model.components_)
    if components.size == 0:
        return None
    idx = int(np.argmax(np.abs(components[0])))
    return feature_names[idx] if idx < len(feature_names) else f"x{idx}"
