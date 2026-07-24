"""Strict numerical validation for the PR79 accuracy evidence pipeline.

The helpers in this module deliberately reject missing or non-finite values.
An accuracy gate must never turn NaN/Inf, a missing reference, or a missing
contract field into a zero error or a skipped check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np


DEFAULT_THRESHOLDS = {
    "coef_max_abs": 1e-7,
    "coef_rel_l2": 1e-6,
    "prediction_rel": 1e-7,
    "objective_rel": 1e-8,
    "hessian_rel_fro": 1e-5,
    "covariance_rel_fro": 1e-5,
    "bse_rel": 1e-5,
    "baseline_hazard_max_abs": 1e-6,
    "final_state": 1e-5,
}


class NumericalValidationError(ValueError):
    """Raised when numerical evidence is missing, malformed, or non-finite."""


def _finite_array(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NumericalValidationError(f"{name} is not numeric") from exc
    if array.size == 0 and not allow_empty:
        raise NumericalValidationError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise NumericalValidationError(f"{name} contains NaN or Inf")
    return array


def _same_shape(actual: np.ndarray, reference: np.ndarray, name: str) -> None:
    if actual.shape != reference.shape:
        raise NumericalValidationError(
            f"{name} shape mismatch: {actual.shape} != {reference.shape}"
        )


def _relative_array_error(actual: Any, reference: Any, name: str) -> float:
    actual_array = _finite_array(actual, f"actual {name}")
    reference_array = _finite_array(reference, f"reference {name}")
    _same_shape(actual_array, reference_array, name)
    error = np.linalg.norm(actual_array - reference_array)
    denominator = max(1.0, float(np.linalg.norm(reference_array)))
    value = float(error / denominator)
    if not np.isfinite(value):
        raise NumericalValidationError(f"{name} error is non-finite")
    return value


def coef_max_abs_error(coef: np.ndarray, ref: np.ndarray) -> float:
    coef_array = _finite_array(coef, "actual coefficient")
    ref_array = _finite_array(ref, "reference coefficient")
    _same_shape(coef_array, ref_array, "coefficient")
    return float(np.max(np.abs(coef_array - ref_array)))


def coef_rel_l2_error(coef: np.ndarray, ref: np.ndarray) -> float:
    return _relative_array_error(coef, ref, "coefficient")


def prediction_rel_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return _relative_array_error(
        _finite_array(pred, "actual prediction").reshape(-1),
        _finite_array(ref, "reference prediction").reshape(-1),
        "prediction",
    )


def objective_rel_error(value: float, ref: float) -> float:
    actual = float(_finite_array(value, "actual objective").reshape(-1)[0])
    reference = float(_finite_array(ref, "reference objective").reshape(-1)[0])
    return abs(actual - reference) / (1.0 + abs(reference))


def bse_rel_error(bse: np.ndarray, ref: np.ndarray) -> float:
    actual = _finite_array(bse, "actual BSE")
    reference = _finite_array(ref, "reference BSE")
    _same_shape(actual, reference, "BSE")
    scale = np.maximum(np.abs(reference), 1e-30)
    error = np.abs(actual - reference) / scale
    if not np.isfinite(error).all():
        raise NumericalValidationError("BSE relative error is non-finite")
    return float(np.max(error))


def covariance_rel_fro_error(cov: np.ndarray, ref: np.ndarray) -> float:
    actual = _finite_array(cov, "actual covariance")
    reference = _finite_array(ref, "reference covariance")
    _same_shape(actual, reference, "covariance")
    if actual.ndim != 2 or actual.shape[0] != actual.shape[1]:
        raise NumericalValidationError("covariance must be square")
    return _relative_array_error(actual, reference, "covariance")


def _case_inputs(case: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = case.get("inputs", case)
    if not isinstance(inputs, Mapping):
        raise NumericalValidationError("case inputs are missing")
    return inputs


def recompute_cox_final_state(
    run: Mapping[str, Any], case: Mapping[str, Any]
) -> Dict[str, Any]:
    """Independently recompute Cox final-beta likelihood and derivatives.

    The implementation uses direct risk-set sums.  It is intentionally
    backend-neutral and does not call a fitted estimator's private kernels.
    Efron and Breslow ties and delayed entry are handled from raw case data.
    """
    inputs = _case_inputs(case)
    results = run.get("results")
    if not isinstance(results, Mapping):
        raise NumericalValidationError("Cox results are missing")

    X = _finite_array(inputs.get("X"), "Cox X")
    time = _finite_array(inputs.get("time"), "Cox time").reshape(-1)
    event = _finite_array(inputs.get("event"), "Cox event").reshape(-1)
    beta = _finite_array(results.get("coef_"), "stored final beta").reshape(-1)
    entry_value = inputs.get("entry")
    entry = None
    if entry_value is not None:
        entry = _finite_array(entry_value, "Cox entry").reshape(-1)

    if X.ndim != 2 or X.shape[0] != time.size or X.shape[0] != event.size:
        raise NumericalValidationError("Cox input shapes are inconsistent")
    if beta.size != X.shape[1]:
        raise NumericalValidationError("stored beta has the wrong feature count")
    if entry is not None and entry.size != time.size:
        raise NumericalValidationError("Cox entry has the wrong length")
    if not np.isin(event, (0.0, 1.0)).all():
        raise NumericalValidationError("Cox event must contain only 0/1")
    if entry is not None and np.any(entry > time):
        raise NumericalValidationError("Cox entry cannot exceed observed time")

    parameters = run.get("parameters", {})
    ties = str(parameters.get("ties", case.get("parameters", {}).get("ties", "efron")))
    if ties not in {"efron", "breslow"}:
        raise NumericalValidationError(f"unsupported Cox ties method: {ties}")
    penalty = float(parameters.get("penalty", 0.0))
    if not np.isfinite(penalty) or penalty < 0.0:
        raise NumericalValidationError("Cox penalty must be finite and non-negative")

    eta = X @ beta
    shift = float(np.max(eta))
    exp_eta = np.exp(eta - shift)
    p = X.shape[1]
    log_likelihood = 0.0
    gradient = np.zeros(p, dtype=np.float64)
    hessian = np.zeros((p, p), dtype=np.float64)
    event_times = np.unique(time[event == 1.0])
    if event_times.size == 0:
        raise NumericalValidationError("Cox case has no observed events")

    for failure_time in event_times:
        failed = np.flatnonzero((time == failure_time) & (event == 1.0))
        at_risk = time >= failure_time
        if entry is not None:
            at_risk &= entry <= failure_time
        risk_index = np.flatnonzero(at_risk)
        if risk_index.size == 0:
            raise NumericalValidationError("Cox event has an empty risk set")

        risk_weight = exp_eta[risk_index]
        risk_x = X[risk_index]
        s0 = float(np.sum(risk_weight))
        s1 = np.sum(risk_x * risk_weight[:, None], axis=0)
        s2 = (risk_x * risk_weight[:, None]).T @ risk_x
        failed_weight = exp_eta[failed]
        failed_x = X[failed]
        e0 = float(np.sum(failed_weight))
        e1 = np.sum(failed_x * failed_weight[:, None], axis=0)
        e2 = (failed_x * failed_weight[:, None]).T @ failed_x
        d = int(failed.size)

        log_likelihood += float(np.sum(eta[failed]))
        gradient += np.sum(failed_x, axis=0)
        fractions = (0.0,) if ties == "breslow" else tuple(k / d for k in range(d))
        multiplier = d if ties == "breslow" else 1
        for fraction in fractions:
            denominator = s0 - fraction * e0
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise NumericalValidationError("Cox risk denominator is invalid")
            moment1 = s1 - fraction * e1
            moment2 = s2 - fraction * e2
            mean = moment1 / denominator
            log_likelihood -= multiplier * (np.log(denominator) + shift)
            gradient -= multiplier * mean
            hessian -= multiplier * (
                moment2 / denominator - np.outer(mean, mean)
            )

    penalized_objective = log_likelihood - penalty * float(beta @ beta)
    penalized_gradient = gradient - 2.0 * penalty * beta
    penalized_hessian = hessian - 2.0 * penalty * np.eye(p)
    information = -penalized_hessian
    covariance = np.linalg.pinv(information, hermitian=True)
    covariance = 0.5 * (covariance + covariance.T)
    bse = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    kkt_inf = float(np.linalg.norm(penalized_gradient, ord=np.inf))
    kkt_normalized = kkt_inf / (
        1.0
        + float(np.linalg.norm(gradient, ord=np.inf))
        + 2.0 * penalty * float(np.linalg.norm(beta, ord=np.inf))
    )

    for name, value in {
        "log_likelihood": log_likelihood,
        "penalized_objective": penalized_objective,
        "gradient": gradient,
        "hessian": hessian,
        "covariance": covariance,
        "bse": bse,
        "kkt_inf": kkt_inf,
        "kkt_normalized": kkt_normalized,
    }.items():
        _finite_array(value, f"recomputed Cox {name}")

    return {
        "log_likelihood": float(log_likelihood),
        "penalized_objective": float(penalized_objective),
        "gradient": gradient,
        "hessian": hessian,
        "penalized_hessian": penalized_hessian,
        "covariance": covariance,
        "bse": bse,
        "kkt_inf": kkt_inf,
        "kkt_normalized": kkt_normalized,
    }


def _contract_check(name: str, value: float, threshold: float) -> Dict[str, Any]:
    if not np.isfinite(value):
        raise NumericalValidationError(f"{name} produced a non-finite error")
    return {
        "check": name,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(value <= threshold),
    }


def validate_cox_final_state(
    run: Mapping[str, Any], case: Mapping[str, Any], threshold: float = 1e-5
) -> Dict[str, Any]:
    recomputed = recompute_cox_final_state(run, case)
    results = run["results"]
    required = (
        "_log_likelihood",
        "_penalized_objective",
        "_final_kkt_inf",
        "_final_kkt_normalized",
        "_var_matrix",
        "_bse",
    )
    missing = [name for name in required if name not in results or results[name] is None]
    if missing:
        raise NumericalValidationError(
            "Cox final-state fields are missing: " + ", ".join(missing)
        )

    checks = [
        _contract_check(
            "cox_log_likelihood_final",
            objective_rel_error(results["_log_likelihood"], recomputed["log_likelihood"]),
            threshold,
        ),
        _contract_check(
            "cox_penalized_objective_final",
            objective_rel_error(
                results["_penalized_objective"], recomputed["penalized_objective"]
            ),
            threshold,
        ),
        _contract_check(
            "cox_kkt_inf_final",
            objective_rel_error(results["_final_kkt_inf"], recomputed["kkt_inf"]),
            threshold,
        ),
        _contract_check(
            "cox_kkt_normalized_final",
            objective_rel_error(
                results["_final_kkt_normalized"], recomputed["kkt_normalized"]
            ),
            threshold,
        ),
        _contract_check(
            "cox_kkt_stationarity",
            float(recomputed["kkt_normalized"]),
            threshold,
        ),
        _contract_check(
            "cox_hessian_symmetry",
            _relative_array_error(
                recomputed["hessian"], recomputed["hessian"].T, "Cox Hessian symmetry"
            ),
            threshold,
        ),
        _contract_check(
            "cox_covariance_final",
            covariance_rel_fro_error(results["_var_matrix"], recomputed["covariance"]),
            threshold,
        ),
        _contract_check(
            "cox_bse_final",
            bse_rel_error(results["_bse"], recomputed["bse"]),
            threshold,
        ),
    ]
    if "_final_hessian" in results and results["_final_hessian"] is not None:
        checks.append(
            _contract_check(
                "cox_hessian_final",
                _relative_array_error(
                    results["_final_hessian"], recomputed["hessian"], "Cox Hessian"
                ),
                threshold,
            )
        )
    passed = all(check["passed"] for check in checks)
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "checks": checks,
        "recomputed": {
            "log_likelihood": recomputed["log_likelihood"],
            "penalized_objective": recomputed["penalized_objective"],
            "kkt_inf": recomputed["kkt_inf"],
            "kkt_normalized": recomputed["kkt_normalized"],
            "hessian_frobenius": float(np.linalg.norm(recomputed["hessian"])),
            "information_min_eigenvalue": float(
                np.min(np.linalg.eigvalsh(-recomputed["penalized_hessian"]))
            ),
        },
    }


def validate_least_squares_final_state(
    run: Mapping[str, Any], case: Mapping[str, Any], threshold: float = 1e-7
) -> Dict[str, Any]:
    """Verify the explicit final-beta contract for linear/panel estimators."""
    inputs = _case_inputs(case)
    results = run.get("results")
    if not isinstance(results, Mapping):
        raise NumericalValidationError("least-squares results are missing")
    X = _finite_array(inputs.get("X"), "least-squares X")
    y = _finite_array(inputs.get("y"), "least-squares y").reshape(-1)
    coef = _finite_array(results.get("coef_"), "least-squares coefficient").reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise NumericalValidationError("least-squares input shapes are inconsistent")
    if run.get("model_id") == "PooledOLS" and coef.size == X.shape[1] + 1:
        intercept = float(coef[0])
        slope = coef[1:]
    else:
        if coef.size != X.shape[1]:
            raise NumericalValidationError("stored coefficient has the wrong feature count")
        intercept_value = results.get("intercept_", 0.0)
        intercept_array = _finite_array(
            intercept_value, "least-squares intercept"
        ).reshape(-1)
        if intercept_array.size != 1:
            raise NumericalValidationError("least-squares intercept must be scalar")
        intercept = float(intercept_array[0])
        slope = coef
    prediction = X @ slope + intercept
    residual_sse = float(np.sum((y - prediction) ** 2))
    stored_prediction = results.get("predictions")
    stored_sse = results.get("residual_sum_squares")
    if stored_prediction is None or stored_sse is None:
        raise NumericalValidationError(
            "least-squares predictions/residual_sum_squares contract is missing"
        )
    checks = [
        _contract_check(
            "least_squares_prediction_final",
            prediction_rel_error(stored_prediction, prediction),
            threshold,
        ),
        _contract_check(
            "least_squares_objective_final",
            objective_rel_error(stored_sse, residual_sse),
            threshold,
        ),
    ]
    rank_deficient = bool(results.get("_rank_deficient"))
    if not rank_deficient and results.get("_var_matrix") is not None and results.get("_bse") is not None:
        covariance = _finite_array(results["_var_matrix"], "stored covariance")
        bse = _finite_array(results["_bse"], "stored BSE")
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise NumericalValidationError("stored covariance must be square")
        expected_bse = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        checks.append(
            _contract_check(
                "least_squares_covariance_bse_final",
                bse_rel_error(bse, expected_bse),
                threshold,
            )
        )
    elif not rank_deficient and results.get("_var_matrix") is not None:
        raise NumericalValidationError("stored covariance is present without BSE")
    elif not rank_deficient and results.get("_bse") is not None:
        _finite_array(results["_bse"], "stored BSE")
    passed = all(check["passed"] for check in checks)
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "checks": checks,
        "recomputed": {"residual_sum_squares": residual_sse},
    }


def validate_run_final_state(
    run: Mapping[str, Any], case: Mapping[str, Any], threshold: float = 1e-5
) -> Dict[str, Any]:
    if run.get("status") != "success":
        raise NumericalValidationError("cannot validate a failed raw run")
    model_id = run.get("model_id")
    if model_id == "CoxPH":
        return validate_cox_final_state(run, case, threshold)
    if model_id in {"LinearRegression", "PooledOLS"}:
        return validate_least_squares_final_state(run, case, threshold)
    raise NumericalValidationError(
        f"no explicit final-state contract for model {model_id!r}"
    )


def validate_final_state_consistency(
    runs: List[Dict[str, Any]],
    cases: Optional[Mapping[str, Mapping[str, Any]]] = None,
    threshold: float = 1e-5,
) -> Dict[str, Any]:
    """Validate every run; missing case evidence is an explicit failure."""
    case_map = cases or {}
    checks: List[Dict[str, Any]] = []
    for run in runs:
        run_key = str(run.get("run_key", "<missing-run-key>"))
        try:
            case = case_map[run["case_id"]]
            result = validate_run_final_state(run, case, threshold)
            for check in result["checks"]:
                checks.append({"run": run_key, **check})
        except (KeyError, NumericalValidationError, ValueError, np.linalg.LinAlgError) as exc:
            checks.append({
                "run": run_key,
                "check": "final_state_contract",
                "value": None,
                "threshold": threshold,
                "passed": False,
                "reason": str(exc),
            })
    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    return {
        "status": "pass" if failed == 0 else "fail",
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "checks": checks,
    }


def _run_identity(run: Mapping[str, Any]) -> tuple:
    parameters = dict(run.get("parameters", {}))
    parameters.pop("backend", None)
    return (
        run.get("case_id"),
        run.get("model_id"),
        parameters.get("iteration"),
        tuple(sorted((key, repr(value)) for key, value in parameters.items())),
    )


def validate_backend_parity(
    runs: List[Dict[str, Any]],
    reference_backend: str = "numpy",
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Validate backend parity without skip-on-error false greens."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    references = {
        _run_identity(run): run
        for run in runs
        if run.get("backend") == reference_backend
    }
    checks: List[Dict[str, Any]] = []
    reclassified: List[Dict[str, Any]] = []
    for run in runs:
        if run.get("backend") == reference_backend:
            continue
        run_key = str(run.get("run_key", "<missing-run-key>"))
        reference = references.get(_run_identity(run))
        if reference is None:
            checks.append({
                "run": run_key,
                "check": "reference_present",
                "value": None,
                "threshold": 0.0,
                "passed": False,
                "reason": "missing reference run",
            })
            continue
        if run.get("status") != "success" or reference.get("status") != "success":
            checks.append({
                "run": run_key,
                "check": "successful_pair",
                "value": None,
                "threshold": 0.0,
                "passed": False,
                "reason": "backend or reference run failed",
            })
            continue
        actual = run.get("results", {})
        expected = reference.get("results", {})
        rank_deficient = bool(run.get("parameters", {}).get("rank_deficient"))
        metric_specs = [
            ("coef_max_abs", "coef_", coef_max_abs_error),
            ("prediction_rel", "predictions", prediction_rel_error),
            ("bse_rel", "_bse", bse_rel_error),
            ("covariance_rel_fro", "_var_matrix", covariance_rel_fro_error),
        ]
        if "_log_likelihood" in actual or "_log_likelihood" in expected:
            metric_specs.append(("objective_rel", "_log_likelihood", objective_rel_error))
        for metric, field, function in metric_specs:
            try:
                if field not in actual or field not in expected:
                    raise NumericalValidationError(f"missing parity field {field}")
                value = float(function(actual[field], expected[field]))
                item = {
                    "run": run_key,
                    "check": metric,
                    "value": value,
                    "threshold": limits[metric],
                    "passed": value <= limits[metric],
                }
                if rank_deficient and field in {"coef_", "_bse", "_var_matrix"}:
                    item.update({
                        "classification": "not_comparable",
                        "reason": "rank-deficient coefficient space is not identifiable",
                    })
                    reclassified.append(item)
                else:
                    checks.append(item)
            except (NumericalValidationError, ValueError, TypeError) as exc:
                checks.append({
                    "run": run_key,
                    "check": metric,
                    "value": None,
                    "threshold": limits[metric],
                    "passed": False,
                    "reason": str(exc),
                })
    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    return {
        "status": "pass" if failed == 0 else "fail",
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "reclassified": len(reclassified),
        "reclassified_items": reclassified,
        "checks": checks,
    }


def _bse_threshold_from_condition(cond: float, base: float) -> float:
    """Compatibility helper retained for callers outside the canonical gate."""
    condition = float(_finite_array(cond, "condition number").reshape(-1)[0])
    if condition < 1e6:
        return base
    if condition < 1e9:
        return max(base, 1e-4)
    if condition < 1e12:
        return max(base, 1e-3)
    return max(base, 1e-2)
