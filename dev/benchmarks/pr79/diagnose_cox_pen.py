#!/usr/bin/env python3
"""Reproducible penalized CoxPH fixed-beta and fitted-model parity checks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time as time_module
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dev.benchmarks.pr79.generators.survival import generate_coxph_penalized

DEFAULT_N = 100
DEFAULT_P = 8
DEFAULT_SEED = 42
DEFAULT_TIES = "efron"
DEFAULT_PENALTY = 0.1
DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 30
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "configs"
    / "expected_accuracy_manifest.json"
)


def load_physical_gpu_matrix(manifest_path=DEFAULT_MANIFEST):
    """Load the auditable full Cox physical-GPU matrix contract."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return manifest["configurations"]["full"]["cox_physical_gpu_matrix"]


def expand_physical_gpu_matrix(matrix=None):
    """Expand every declared Cox GPU axis into deterministic case records."""
    matrix = load_physical_gpu_matrix() if matrix is None else matrix
    axes = matrix["axes"]
    axis_names = tuple(axes)
    cases = []
    for backend in matrix["physical_gpu_backends"]:
        for values in itertools.product(*(axes[name] for name in axis_names)):
            parameters = dict(zip(axis_names, values))
            parameters["cov_type"] = (
                "hc0"
                if parameters["compute_inference"] and not parameters["entry"]
                else "nonrobust"
            )
            identity = {"backend": backend, **parameters}
            encoded = json.dumps(
                identity, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            cases.append({
                "case_id": "cox-gpu-" + hashlib.sha256(encoded).hexdigest()[:16],
                **identity,
                "thresholds": dict(matrix["thresholds"]),
            })
    return cases


def prepare_physical_gpu_case(case, *, n=DEFAULT_N, p=DEFAULT_P, seed=DEFAULT_SEED):
    """Build one matrix case, including ties, delayed entry, and row order."""
    X, time_values, event_values, beta = generate_coxph_penalized(
        n, p, seed, penalty=float(case["penalty"])
    )
    order = np.argsort(time_values, kind="stable")
    tie_pattern = case["tie_pattern"]
    group_size = {"no_ties": 1, "small_ties": 3, "heavy_ties": 12}[tie_pattern]
    transformed_time = np.empty_like(time_values, dtype=np.float64)
    transformed_time[order] = (
        np.arange(n, dtype=np.int64) // group_size + 1
    ).astype(np.float64)
    time_values = transformed_time

    entry = None
    if bool(case["entry"]):
        rng = np.random.default_rng(seed + 101)
        entry = time_values * rng.uniform(0.0, 0.75, size=n)

    if case["row_order"] == "permuted":
        permutation = np.random.default_rng(seed + 211).permutation(n)
        X = X[permutation]
        time_values = time_values[permutation]
        event_values = event_values[permutation]
        if entry is not None:
            entry = entry[permutation]
    elif case["row_order"] != "canonical":
        raise ValueError("row_order must be canonical or permuted")

    return {
        "X": np.asarray(X, dtype=np.float64),
        "time": np.asarray(time_values, dtype=np.float64),
        "event": np.asarray(event_values, dtype=np.int32),
        "entry": None if entry is None else np.asarray(entry, dtype=np.float64),
        "fixed_beta": np.asarray(beta, dtype=np.float64),
    }


def stable_sort_risk_set_inputs(
    X, time_values, event_values, *, entry=None, cluster=None
):
    """Stable-sort every row-aligned risk-set input by ascending time."""
    X_arr = np.asarray(X, dtype=np.float64)
    time_arr = np.asarray(time_values, dtype=np.float64)
    event_arr = np.asarray(event_values, dtype=np.int32)
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    n_samples = X_arr.shape[0]
    if time_arr.shape != (n_samples,) or event_arr.shape != (n_samples,):
        raise ValueError("time and event must have shape (n_samples,)")
    entry_arr = None if entry is None else np.asarray(entry, dtype=np.float64)
    cluster_arr = None if cluster is None else np.asarray(cluster)
    if entry_arr is not None and entry_arr.shape != (n_samples,):
        raise ValueError("entry must have shape (n_samples,)")
    if cluster_arr is not None and cluster_arr.shape != (n_samples,):
        raise ValueError("cluster must have shape (n_samples,)")
    order = np.argsort(time_arr, kind="stable")
    return {
        "order": order,
        "X": np.ascontiguousarray(X_arr[order]),
        "time": np.ascontiguousarray(time_arr[order]),
        "event": np.ascontiguousarray(event_arr[order]),
        "entry": None if entry_arr is None else np.ascontiguousarray(entry_arr[order]),
        "cluster": (
            None if cluster_arr is None else np.ascontiguousarray(cluster_arr[order])
        ),
    }


def _new_model(
    backend,
    *,
    compute_inference,
    penalty,
    ties,
    tol,
    max_iter,
    inference_mode="strict",
    cov_type="nonrobust",
):
    from statgpu.survival import CoxPH

    kwargs = dict(
        ties=ties,
        penalty=penalty,
        compute_inference=compute_inference,
        compute_cindex=False,
        tol=tol,
        max_iter=max_iter,
        inference_mode=inference_mode,
        cov_type=cov_type,
    )
    if backend == "numpy":
        kwargs["device"] = "cpu"
    elif backend == "cupy":
        kwargs["device"] = "cuda"
    elif backend == "torch":
        kwargs["device"] = "torch"
    else:
        raise ValueError("backend must be one of: numpy, cupy, torch")
    return CoxPH(**kwargs)


def _require_backend(backend):
    if backend == "numpy":
        return
    if backend == "cupy":
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("CuPy is installed but no CUDA device is available")
        return
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is not available")
        return
    raise ValueError("backend must be one of: numpy, cupy, torch")


def _backend_arrays(backend, X, time_values, event_values, beta):
    if backend == "numpy":
        return X, time_values, event_values, beta
    if backend == "cupy":
        import cupy as cp

        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(time_values, dtype=cp.float64),
            cp.asarray(event_values, dtype=cp.int32),
            cp.asarray(beta, dtype=cp.float64),
        )
    if backend == "torch":
        import torch

        device = torch.device("cuda")
        return (
            torch.as_tensor(X, dtype=torch.float64, device=device),
            torch.as_tensor(time_values, dtype=torch.float64, device=device),
            torch.as_tensor(event_values, dtype=torch.int32, device=device),
            torch.as_tensor(beta, dtype=torch.float64, device=device),
        )
    raise ValueError("backend must be one of: numpy, cupy, torch")


def _to_numpy(backend, value):
    if backend == "numpy":
        return np.asarray(value)
    if backend == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    if backend == "torch":
        return value.detach().cpu().numpy()
    raise ValueError("backend must be one of: numpy, cupy, torch")


def _to_float(backend, value):
    if backend == "numpy":
        return float(value)
    if backend == "cupy":
        import cupy as cp

        return float(cp.asnumpy(value))
    if backend == "torch":
        return float(value.detach().cpu().item())
    raise ValueError("backend must be one of: numpy, cupy, torch")


def _canonical_loglik_hessian(model, raw_hessian):
    """Normalize legacy kernel signs to the mathematical LL Hessian."""
    raw = np.asarray(raw_hessian, dtype=np.float64)
    raw_sym = 0.5 * (raw + raw.T)
    canonical = -np.asarray(model._observed_information(raw_sym), dtype=np.float64)
    if np.allclose(raw_sym, canonical, rtol=1e-10, atol=1e-10):
        orientation = "log_likelihood_hessian"
    elif np.allclose(raw_sym, -canonical, rtol=1e-10, atol=1e-10):
        orientation = "observed_information"
    else:
        orientation = "mixed_or_indefinite"
    return canonical, orientation


def _covariance_from_hessian(unpenalized_hessian, penalty):
    p = int(unpenalized_hessian.shape[0])
    penalized_hessian = np.asarray(
        unpenalized_hessian, dtype=np.float64
    ) - 2.0 * penalty * np.eye(p, dtype=np.float64)
    information = -penalized_hessian
    try:
        covariance = np.linalg.solve(information, np.eye(p, dtype=np.float64))
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
    covariance = 0.5 * (covariance + covariance.T)
    bse = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return penalized_hessian, covariance, bse


def evaluate_fixed_beta(
    backend,
    *,
    beta,
    X,
    time_values,
    event_values,
    penalty,
    ties,
    tol=DEFAULT_TOL,
    max_iter=DEFAULT_MAX_ITER,
    entry=None,
    inference_mode="strict",
    cov_type="nonrobust",
):
    """Evaluate all required fixed-beta quantities on one backend."""
    _require_backend(backend)
    sorted_inputs = stable_sort_risk_set_inputs(
        X, time_values, event_values, entry=entry
    )
    X = np.asarray(sorted_inputs["X"], dtype=np.float64)
    time_values = np.asarray(sorted_inputs["time"], dtype=np.float64)
    event_values = np.asarray(sorted_inputs["event"], dtype=np.int32)
    entry = sorted_inputs["entry"]
    model = _new_model(
        backend,
        compute_inference=False,
        penalty=penalty,
        ties=ties,
        tol=tol,
        max_iter=max_iter,
        inference_mode=inference_mode,
        cov_type=cov_type,
    )
    efron_pre = (
        model._efron_unique_failure_indices(time_values, event_values)
        if ties == "efron"
        else None
    )
    X_b, time_b, event_b, beta_b = _backend_arrays(
        backend, X, time_values, event_values, beta
    )
    if entry is None or backend == "numpy":
        entry_b = entry
    elif backend == "cupy":
        import cupy as cp

        entry_b = cp.asarray(entry, dtype=cp.float64)
    else:
        import torch

        entry_b = torch.as_tensor(entry, dtype=torch.float64, device="cuda")

    if backend == "numpy":
        # Independent reference: these calls must never be replaced by a GPU
        # helper or by cached values from a fitted model.
        gradient_raw, hessian_raw = model._compute_gradient_hessian(
            beta_b, X_b, time_b, event_b, efron_pre, entry=entry_b
        )
        log_likelihood_raw = model._compute_log_likelihood(
            beta_b, X_b, time_b, event_b, efron_pre, entry=entry_b
        )
    elif backend == "cupy":
        gradient_raw, hessian_raw, _ = model._compute_gradient_hessian_gpu(
            beta_b, X_b, time_b, event_b, efron_pre, return_aux=True, entry=entry_b
        )
        log_likelihood_raw = model._compute_log_likelihood_gpu(
            beta_b, X_b, time_b, event_b, efron_pre, entry=entry_b
        )
    else:
        gradient_raw, hessian_raw, _ = model._compute_gradient_hessian_torch(
            beta_b, X_b, time_b, event_b, efron_pre, return_aux=True, entry=entry_b
        )
        log_likelihood_raw = model._compute_log_likelihood_torch(
            beta_b, X_b, time_b, event_b, efron_pre, entry=entry_b
        )

    gradient = np.asarray(_to_numpy(backend, gradient_raw), dtype=np.float64)
    raw_hessian = np.asarray(_to_numpy(backend, hessian_raw), dtype=np.float64)
    unpen_hessian, orientation = _canonical_loglik_hessian(model, raw_hessian)
    pen_hessian, covariance, bse = _covariance_from_hessian(unpen_hessian, penalty)
    log_likelihood = _to_float(backend, log_likelihood_raw)
    beta_np = np.asarray(beta, dtype=np.float64)
    penalized_objective = log_likelihood - penalty * float(np.dot(beta_np, beta_np))
    return {
        "beta": beta_np.copy(),
        "unpenalized_log_likelihood": log_likelihood,
        "penalized_objective": penalized_objective,
        "gradient": gradient,
        "penalized_gradient": gradient - 2.0 * penalty * beta_np,
        "raw_unpenalized_hessian": 0.5 * (raw_hessian + raw_hessian.T),
        "raw_hessian_orientation": orientation,
        "unpenalized_hessian": unpen_hessian,
        "penalized_hessian": pen_hessian,
        "covariance": covariance,
        "bse": bse,
        "information_condition_number": float(np.linalg.cond(-pen_hessian)),
    }


def _fit_backend(
    backend,
    *,
    X,
    time_values,
    event_values,
    penalty,
    ties,
    tol,
    max_iter,
    entry=None,
    compute_inference=True,
    inference_mode="strict",
    cov_type="nonrobust",
):
    _require_backend(backend)
    sorted_inputs = stable_sort_risk_set_inputs(
        X, time_values, event_values, entry=entry
    )
    X = np.asarray(sorted_inputs["X"], dtype=np.float64)
    time_values = np.asarray(sorted_inputs["time"], dtype=np.float64)
    event_values = np.asarray(sorted_inputs["event"], dtype=np.int32)
    entry = sorted_inputs["entry"]
    model = _new_model(
        backend,
        compute_inference=compute_inference,
        penalty=penalty,
        ties=ties,
        tol=tol,
        max_iter=max_iter,
        inference_mode=inference_mode,
        cov_type=cov_type,
    )
    if backend == "numpy":
        X_fit, time_fit, event_fit, entry_fit = X, time_values, event_values, entry
    else:
        X_fit, time_fit, event_fit, _ = _backend_arrays(
            backend,
            X,
            time_values,
            event_values,
            np.zeros(X.shape[1], dtype=np.float64),
        )
        if entry is None:
            entry_fit = None
        elif backend == "cupy":
            import cupy as cp

            entry_fit = cp.asarray(entry, dtype=cp.float64)
        else:
            import torch

            entry_fit = torch.as_tensor(entry, dtype=torch.float64, device="cuda")
    model.fit(X_fit, time=time_fit, event=event_fit, entry=entry_fit)

    coefficients = np.asarray(model.coef_, dtype=np.float64)
    at_solution = evaluate_fixed_beta(
        backend,
        beta=coefficients,
        X=X,
        time_values=time_values,
        event_values=event_values,
        penalty=penalty,
        ties=ties,
        tol=tol,
        max_iter=max_iter,
        entry=entry,
        inference_mode=inference_mode,
        cov_type=cov_type,
    )
    gradient = np.asarray(at_solution["gradient"], dtype=np.float64)
    pen_gradient = gradient - 2.0 * penalty * coefficients
    kkt_inf = float(np.linalg.norm(pen_gradient, ord=np.inf))
    kkt_normalized = kkt_inf / (
        1.0
        + float(np.linalg.norm(gradient, ord=np.inf))
        + 2.0 * penalty * float(np.linalg.norm(coefficients, ord=np.inf))
    )
    covariance = (
        None
        if getattr(model, "_var_matrix", None) is None
        else np.asarray(model._var_matrix, dtype=np.float64)
    )
    bse = (
        None
        if getattr(model, "_bse", None) is None
        else np.asarray(model._bse, dtype=np.float64)
    )
    return {
        "coefficients": coefficients,
        "unpenalized_log_likelihood": float(at_solution["unpenalized_log_likelihood"]),
        "reported_unpenalized_log_likelihood": float(model._log_likelihood),
        "penalized_objective": float(at_solution["penalized_objective"]),
        "reported_penalized_objective": float(model._penalized_objective),
        "final_kkt_inf": kkt_inf,
        "final_kkt_normalized": kkt_normalized,
        "reported_final_kkt_inf": getattr(model, "_final_kkt_inf", None),
        "reported_final_kkt_normalized": getattr(model, "_final_kkt_normalized", None),
        "converged": bool(model._converged),
        "termination_reason": model._termination_reason,
        "iterations": int(model._iterations),
        "objective_history": list(getattr(model, "_objective_history", [])),
        "compute_inference": bool(compute_inference),
        "inference_mode": inference_mode,
        "cov_type": cov_type,
        "inference_method": getattr(model, "inference_method_", None),
        "inference_backend": getattr(model, "inference_backend_", None),
        "inference_approximate": getattr(model, "inference_approximate_", False),
        "inference_fallback_reason": getattr(
            model, "inference_fallback_reason_", None
        ),
        "covariance": covariance,
        "bse": bse,
        "fixed_beta_covariance_at_solution": at_solution["covariance"],
        "fixed_beta_bse_at_solution": at_solution["bse"],
    }


def _max_abs_difference(left, right):
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    return float(np.max(np.abs(left_arr - right_arr)))


def _max_relative_difference(left, right):
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    scale = np.maximum(np.abs(left_arr), 1e-12)
    return float(np.max(np.abs(left_arr - right_arr) / scale))


def _relative_l2_difference(left, right):
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    return float(
        np.linalg.norm(left_arr - right_arr)
        / max(1.0, float(np.linalg.norm(right_arr)))
    )


def _scalar_relative_difference(left, right):
    left_value = float(left)
    right_value = float(right)
    return abs(left_value - right_value) / (1.0 + abs(right_value))


def _maximum_objective_decrease(history):
    values = np.asarray(history, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.maximum(values[:-1] - values[1:], 0.0).max())


def _all_finite(*values):
    for value in values:
        if value is None:
            return False
        try:
            if not bool(np.all(np.isfinite(np.asarray(value, dtype=np.float64)))):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _add_metric_check(checks, name, value, tolerance, *, backend=None, reference=None):
    passed = bool(np.isfinite(value) and value <= tolerance)
    check = {
        "name": name,
        "value": value,
        "tolerance": tolerance,
        "status": "pass" if passed else "fail",
    }
    if backend is not None:
        check["backend"] = backend
    if reference is not None:
        check["reference"] = reference
    checks.append(check)


def _add_boolean_check(
    checks, name, passed, *, backend=None, actual=None, expected=None
):
    check = {"name": name, "status": "pass" if bool(passed) else "fail"}
    if backend is not None:
        check["backend"] = backend
    if actual is not None:
        check["actual"] = actual
    if expected is not None:
        check["expected"] = expected
    checks.append(check)


def _add_numpy_self_checks(checks, fixed, fitted, *, max_iter):
    _add_boolean_check(
        checks,
        "fixed_beta_required_quantities_are_finite",
        _all_finite(
            fixed["unpenalized_log_likelihood"],
            fixed["penalized_objective"],
            fixed["gradient"],
            fixed["unpenalized_hessian"],
            fixed["penalized_hessian"],
            fixed["covariance"],
            fixed["bse"],
        ),
        backend="numpy",
    )
    _add_metric_check(
        checks,
        "fixed_beta_covariance_symmetry",
        _max_abs_difference(fixed["covariance"], np.asarray(fixed["covariance"]).T),
        1e-10,
        backend="numpy",
    )
    _add_boolean_check(
        checks,
        "fixed_beta_covariance_has_positive_diagonal",
        bool(np.all(np.diag(np.asarray(fixed["covariance"])) > 0.0)),
        backend="numpy",
    )
    _add_boolean_check(
        checks,
        "fitted_required_quantities_are_finite",
        _all_finite(
            fitted["coefficients"],
            fitted["unpenalized_log_likelihood"],
            fitted["penalized_objective"],
            fitted["final_kkt_inf"],
            fitted["final_kkt_normalized"],
            fitted["reported_final_kkt_inf"],
            fitted["reported_final_kkt_normalized"],
            fitted["bse"],
        ),
        backend="numpy",
    )
    _add_metric_check(
        checks,
        "fitted_log_likelihood_is_final_beta_value",
        abs(
            fitted["unpenalized_log_likelihood"]
            - fitted["reported_unpenalized_log_likelihood"]
        ),
        1e-10,
        backend="numpy",
    )
    _add_metric_check(
        checks,
        "fitted_penalized_objective_is_final_beta_value",
        abs(fitted["penalized_objective"] - fitted["reported_penalized_objective"]),
        1e-10,
        backend="numpy",
    )
    _add_metric_check(
        checks,
        "fitted_reported_kkt_matches_recomputed_kkt",
        abs(fitted["final_kkt_inf"] - fitted["reported_final_kkt_inf"]),
        1e-8,
        backend="numpy",
    )
    _add_metric_check(
        checks,
        "fitted_final_normalized_kkt",
        fitted["final_kkt_normalized"],
        1e-7,
        backend="numpy",
    )
    _add_boolean_check(
        checks,
        "fitted_converged",
        fitted["converged"] is True,
        backend="numpy",
        actual=fitted["converged"],
        expected=True,
    )
    _add_boolean_check(
        checks,
        "fitted_termination_reason",
        fitted["termination_reason"] == "kkt_converged",
        backend="numpy",
        actual=fitted["termination_reason"],
        expected="kkt_converged",
    )
    _add_boolean_check(
        checks,
        "fitted_iteration_count_is_valid",
        0 <= fitted["iterations"] <= max_iter,
        backend="numpy",
        actual=fitted["iterations"],
        expected="0..{}".format(max_iter),
    )
    _add_metric_check(
        checks,
        "fitted_bse_matches_final_beta_hessian",
        _max_relative_difference(fitted["fixed_beta_bse_at_solution"], fitted["bse"]),
        1e-8,
        backend="numpy",
    )
    _add_metric_check(
        checks,
        "fitted_objective_maximum_decrease",
        _maximum_objective_decrease(fitted["objective_history"]),
        1e-10,
        backend="numpy",
    )


def _add_backend_parity_checks(checks, backend, fixed_ref, fixed, fitted_ref, fitted):
    fixed_metrics = (
        ("unpenalized_log_likelihood", 1e-9, "abs"),
        ("penalized_objective", 1e-9, "abs"),
        ("gradient", 1e-8, "abs"),
        ("unpenalized_hessian", 1e-7, "abs"),
        ("penalized_hessian", 1e-7, "abs"),
        ("covariance", 1e-7, "relative"),
        ("bse", 1e-7, "relative"),
    )
    for metric, tolerance, mode in fixed_metrics:
        fn = _max_relative_difference if mode == "relative" else _max_abs_difference
        _add_metric_check(
            checks,
            "fixed_beta_{}_parity".format(metric),
            fn(fixed_ref[metric], fixed[metric]),
            tolerance,
            backend=backend,
            reference="numpy",
        )

    fitted_metrics = (
        ("coefficients", 1e-6, "relative_l2"),
        ("unpenalized_log_likelihood", 1e-9, "scalar_relative"),
        ("penalized_objective", 1e-9, "scalar_relative"),
        ("final_kkt_normalized", 1e-7, "abs"),
        ("bse", 1e-5, "relative"),
    )
    for metric, tolerance, mode in fitted_metrics:
        if mode == "relative":
            fn = _max_relative_difference
        elif mode == "relative_l2":
            fn = _relative_l2_difference
        elif mode == "scalar_relative":
            fn = _scalar_relative_difference
        else:
            fn = _max_abs_difference
        _add_metric_check(
            checks,
            "fitted_{}_parity".format(metric),
            fn(fitted_ref[metric], fitted[metric]),
            tolerance,
            backend=backend,
            reference="numpy",
        )
    _add_metric_check(
        checks,
        "fitted_final_normalized_kkt",
        float(fitted["final_kkt_normalized"]),
        1e-7,
        backend=backend,
    )
    _add_boolean_check(
        checks,
        "fitted_convergence_parity",
        fitted["converged"] is True and fitted["converged"] == fitted_ref["converged"],
        backend=backend,
        actual=fitted["converged"],
        expected=fitted_ref["converged"],
    )
    _add_boolean_check(
        checks,
        "fitted_termination_reason_parity",
        fitted["termination_reason"]
        == fitted_ref["termination_reason"]
        == "kkt_converged",
        backend=backend,
        actual=fitted["termination_reason"],
        expected=fitted_ref["termination_reason"],
    )
    _add_metric_check(
        checks,
        "fitted_iterations_parity",
        float(abs(fitted["iterations"] - fitted_ref["iterations"])),
        1.0,
        backend=backend,
        reference="numpy",
    )
    _add_metric_check(
        checks,
        "fitted_bse_matches_own_final_beta_hessian",
        _max_relative_difference(fitted["fixed_beta_bse_at_solution"], fitted["bse"]),
        1e-8,
        backend=backend,
    )
    _add_metric_check(
        checks,
        "fitted_objective_maximum_decrease",
        _maximum_objective_decrease(fitted["objective_history"]),
        1e-10,
        backend=backend,
    )


def _synchronize_backend(backend):
    if backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch

        torch.cuda.synchronize()


def _time_backend(
    backend,
    *,
    X,
    time_values,
    event_values,
    penalty,
    ties,
    tol,
    max_iter,
    warmup,
    repeats,
):
    _require_backend(backend)
    sorted_inputs = stable_sort_risk_set_inputs(X, time_values, event_values)
    X = np.asarray(sorted_inputs["X"], dtype=np.float64)
    time_values = np.asarray(sorted_inputs["time"], dtype=np.float64)
    event_values = np.asarray(sorted_inputs["event"], dtype=np.int32)
    samples = []
    for index in range(warmup + repeats):
        model = _new_model(
            backend,
            compute_inference=False,
            penalty=penalty,
            ties=ties,
            tol=tol,
            max_iter=max_iter,
        )
        _synchronize_backend(backend)
        started = time_module.perf_counter()
        model.fit(X, time=time_values, event=event_values)
        _synchronize_backend(backend)
        elapsed = time_module.perf_counter() - started
        if index >= warmup:
            samples.append(float(elapsed))
    return {
        "warmup": warmup,
        "repeats": repeats,
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


def _validated_code_provenance():
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        code_sha = done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        code_sha = "unknown"
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(_PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = bool(done.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    digest = hashlib.sha256()
    for relative in (
        "statgpu/survival/_cox.py",
        "dev/benchmarks/pr79/generators/survival.py",
        "dev/benchmarks/pr79/diagnose_cox_pen.py",
        "dev/benchmarks/pr79/torch_parity.py",
    ):
        path = _PROJECT_ROOT / relative
        if path.exists():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return {
        "validated_code_sha": code_sha,
        "validated_worktree_dirty": dirty,
        "validated_source_sha256": digest.hexdigest(),
    }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _start_gpu_memory_tracking(backend):
    if backend == "cupy":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        return None
    if backend == "torch":
        import torch

        torch.cuda.reset_peak_memory_stats()
        return None
    return None


def _peak_gpu_memory_bytes(backend):
    if backend == "cupy":
        import cupy as cp

        return int(cp.get_default_memory_pool().total_bytes())
    if backend == "torch":
        import torch

        return int(torch.cuda.max_memory_allocated())
    return 0


def run_physical_gpu_matrix_case(case, *, tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER):
    """Execute one expanded matrix case against its independent NumPy reference."""
    backend = case["backend"]
    if backend not in {"cupy", "torch"}:
        raise ValueError("physical GPU matrix cases require cupy or torch")
    data = prepare_physical_gpu_case(case)
    thresholds = case["thresholds"]
    penalty = float(case["penalty"])
    ties = case["ties"]
    inference_mode = case["inference_mode"]
    compute_inference = bool(case["compute_inference"])
    common = {
        "X": data["X"],
        "time_values": data["time"],
        "event_values": data["event"],
        "entry": data["entry"],
        "penalty": penalty,
        "ties": ties,
        "tol": tol,
        "max_iter": max_iter,
        "inference_mode": inference_mode,
        "cov_type": case["cov_type"],
    }
    report = {
        "matrix_schema_version": "pr79-cox-gpu-case-result-1.0",
        "case": dict(case),
        "checks": [],
        "errors": [],
        "results": {},
        "status": "error",
    }
    try:
        _require_backend(backend)
        _start_gpu_memory_tracking(backend)
        fixed_reference = evaluate_fixed_beta(
            "numpy", beta=data["fixed_beta"], **common
        )
        fixed_gpu = evaluate_fixed_beta(
            backend, beta=data["fixed_beta"], **common
        )
        fitted_gpu = _fit_backend(
            backend, compute_inference=compute_inference, **common
        )
        fitted_reference = None
        if data["entry"] is None or penalty == 0.0:
            fitted_reference = _fit_backend(
                "numpy", compute_inference=compute_inference, **common
            )

        checks = report["checks"]
        fixed_specs = (
            ("unpenalized_log_likelihood", "scalar", thresholds["unpenalized_log_likelihood_rel_error"]),
            ("penalized_objective", "scalar", thresholds["penalized_objective_rel_error"]),
            ("gradient", "relative_l2", thresholds["hessian_rel_fro_error"]),
            ("unpenalized_hessian", "relative_l2", thresholds["hessian_rel_fro_error"]),
        )
        for metric, mode, threshold in fixed_specs:
            function = (
                _scalar_relative_difference
                if mode == "scalar"
                else _relative_l2_difference
            )
            _add_metric_check(
                checks,
                "fixed_beta_{}_parity".format(metric),
                function(fixed_gpu[metric], fixed_reference[metric]),
                threshold,
                backend=backend,
                reference="numpy",
            )
        _add_metric_check(
            checks,
            "fitted_final_normalized_kkt",
            fitted_gpu["final_kkt_normalized"],
            thresholds["normalized_final_kkt"],
            backend=backend,
        )
        _add_metric_check(
            checks,
            "fitted_log_likelihood_is_final_beta_value",
            _scalar_relative_difference(
                fitted_gpu["reported_unpenalized_log_likelihood"],
                fitted_gpu["unpenalized_log_likelihood"],
            ),
            thresholds["unpenalized_log_likelihood_rel_error"],
            backend=backend,
        )
        _add_metric_check(
            checks,
            "fitted_penalized_objective_is_final_beta_value",
            _scalar_relative_difference(
                fitted_gpu["reported_penalized_objective"],
                fitted_gpu["penalized_objective"],
            ),
            thresholds["penalized_objective_rel_error"],
            backend=backend,
        )
        _add_metric_check(
            checks,
            "fitted_objective_maximum_decrease",
            _maximum_objective_decrease(fitted_gpu["objective_history"]),
            thresholds["objective_decrease_tolerance"],
            backend=backend,
        )
        _add_boolean_check(
            checks,
            "fitted_converged_with_kkt_reason",
            fitted_gpu["converged"]
            and fitted_gpu["termination_reason"] == "kkt_converged",
            backend=backend,
            actual=fitted_gpu["termination_reason"],
            expected="kkt_converged",
        )
        if compute_inference and case["cov_type"] == "nonrobust":
            _add_metric_check(
                checks,
                "fitted_bse_matches_own_final_beta_hessian",
                _max_relative_difference(
                    fitted_gpu["fixed_beta_bse_at_solution"], fitted_gpu["bse"]
                ),
                thresholds["bse_rel_error"],
                backend=backend,
            )
        if compute_inference and case["cov_type"] != "nonrobust":
            _add_boolean_check(
                checks,
                "fitted_inference_mode_provenance",
                fitted_gpu["inference_method"] is not None
                and fitted_gpu["inference_backend"] is not None
                and fitted_gpu["inference_approximate"]
                is (inference_mode == "approx"),
                backend=backend,
                actual={
                    "method": fitted_gpu["inference_method"],
                    "backend": fitted_gpu["inference_backend"],
                    "approximate": fitted_gpu["inference_approximate"],
                },
                expected={"approximate": inference_mode == "approx"},
            )
        if fitted_reference is not None:
            _add_metric_check(
                checks,
                "fitted_coefficients_parity",
                _relative_l2_difference(
                    fitted_gpu["coefficients"], fitted_reference["coefficients"]
                ),
                thresholds["coefficient_rel_l2_error"],
                backend=backend,
                reference="numpy",
            )
            _add_metric_check(
                checks,
                "fitted_unpenalized_log_likelihood_parity",
                _scalar_relative_difference(
                    fitted_gpu["unpenalized_log_likelihood"],
                    fitted_reference["unpenalized_log_likelihood"],
                ),
                thresholds["unpenalized_log_likelihood_rel_error"],
                backend=backend,
                reference="numpy",
            )
            _add_metric_check(
                checks,
                "fitted_penalized_objective_parity",
                _scalar_relative_difference(
                    fitted_gpu["penalized_objective"],
                    fitted_reference["penalized_objective"],
                ),
                thresholds["penalized_objective_rel_error"],
                backend=backend,
                reference="numpy",
            )
            if compute_inference:
                _add_metric_check(
                    checks,
                    "fitted_bse_parity",
                    _max_relative_difference(
                        fitted_gpu["bse"], fitted_reference["bse"]
                    ),
                    thresholds["bse_rel_error"],
                    backend=backend,
                    reference="numpy",
                )
        else:
            report["numpy_fit_contract"] = (
                "CPU delayed-entry CoxPH with penalty is explicitly unsupported"
            )
        _synchronize_backend(backend)
        report["peak_gpu_memory_bytes"] = _peak_gpu_memory_bytes(backend)
        report["results"] = {
            "fixed_numpy": fixed_reference,
            "fixed_gpu": fixed_gpu,
            "fitted_numpy": fitted_reference,
            "fitted_gpu": fitted_gpu,
        }
        report["status"] = (
            "pass"
            if checks and all(check["status"] == "pass" for check in checks)
            else "fail"
        )
    except Exception as exc:
        report["errors"].append({
            "type": type(exc).__name__,
            "message": str(exc),
        })
        report["status"] = "error"
    return _json_safe(report)


def run_physical_gpu_matrix(cases):
    """Execute selected matrix cases without skipping backend failures."""
    results = [run_physical_gpu_matrix_case(case) for case in cases]
    by_pair = {}
    for result in results:
        case = result["case"]
        key = tuple(
            (name, json.dumps(value, sort_keys=True))
            for name, value in sorted(case.items())
            if name not in {"case_id", "row_order", "thresholds"}
        )
        by_pair.setdefault(key, {})[case["row_order"]] = result
    permutation_checks = []
    for pair in by_pair.values():
        if set(pair) != {"canonical", "permuted"}:
            continue
        canonical = pair["canonical"]
        permuted = pair["permuted"]
        thresholds = canonical["case"]["thresholds"]
        if canonical["status"] != "pass" or permuted["status"] != "pass":
            permutation_checks.append({
                "case_id": canonical["case"]["case_id"],
                "paired_case_id": permuted["case"]["case_id"],
                "status": "fail",
                "reason": "canonical or permuted case did not pass",
            })
            continue
        canonical_fit = canonical["results"]["fitted_gpu"]
        permuted_fit = permuted["results"]["fitted_gpu"]
        metrics = {
            "coefficient_rel_l2_error": _relative_l2_difference(
                permuted_fit["coefficients"], canonical_fit["coefficients"]
            ),
            "unpenalized_log_likelihood_rel_error": _scalar_relative_difference(
                permuted_fit["unpenalized_log_likelihood"],
                canonical_fit["unpenalized_log_likelihood"],
            ),
            "penalized_objective_rel_error": _scalar_relative_difference(
                permuted_fit["penalized_objective"],
                canonical_fit["penalized_objective"],
            ),
        }
        passed = all(value <= thresholds[name] for name, value in metrics.items())
        permutation_checks.append({
            "case_id": canonical["case"]["case_id"],
            "paired_case_id": permuted["case"]["case_id"],
            "metrics": metrics,
            "status": "pass" if passed else "fail",
        })
    all_cases_passed = bool(results) and all(
        result["status"] == "pass" for result in results
    )
    all_permutations_passed = all(
        check["status"] == "pass" for check in permutation_checks
    )
    return {
        "matrix_schema_version": "pr79-cox-gpu-matrix-results-1.0",
        "case_count": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "failed": sum(result["status"] != "pass" for result in results),
        "permutation_checks": permutation_checks,
        "status": "pass" if all_cases_passed and all_permutations_passed else "fail",
        "cases": results,
    }


def build_report(
    *,
    backend="all",
    include_timing=True,
    timing_warmup=1,
    timing_repeats=3,
    tol=DEFAULT_TOL,
    max_iter=DEFAULT_MAX_ITER,
):
    """Build the complete machine-readable diagnostic report."""
    if backend not in {"all", "numpy", "cupy", "torch"}:
        raise ValueError("backend must be one of: all, numpy, cupy, torch")
    if timing_warmup < 0 or timing_repeats < 1:
        raise ValueError("timing_warmup must be >= 0 and timing_repeats >= 1")

    X_raw, time_raw, event_raw, beta_fixed = generate_coxph_penalized(
        DEFAULT_N, DEFAULT_P, DEFAULT_SEED, penalty=DEFAULT_PENALTY
    )
    sorted_inputs = stable_sort_risk_set_inputs(X_raw, time_raw, event_raw)
    X = np.asarray(sorted_inputs["X"], dtype=np.float64)
    time_values = np.asarray(sorted_inputs["time"], dtype=np.float64)
    event_values = np.asarray(sorted_inputs["event"], dtype=np.int32)
    requested = (
        ["numpy", "cupy", "torch"]
        if backend == "all"
        else ["numpy"] if backend == "numpy" else ["numpy", backend]
    )
    report = {
        **_validated_code_provenance(),
        "case": {
            "n": int(X.shape[0]),
            "p": int(X.shape[1]),
            "ties": DEFAULT_TIES,
            "penalty": DEFAULT_PENALTY,
            "dtype": "float64",
            "seed": DEFAULT_SEED,
            "risk_set_order": "ascending_time_stable",
            "fixed_beta": np.asarray(beta_fixed, dtype=np.float64),
        },
        "requested_backend": backend,
        "fixed_beta": {},
        "fitted": {},
        "checks": [],
        "timing": {"enabled": bool(include_timing), "backends": {}},
        "errors": [],
        "status": "error",
    }

    for current in requested:
        try:
            report["fixed_beta"][current] = evaluate_fixed_beta(
                current,
                beta=np.asarray(beta_fixed, dtype=np.float64),
                X=X,
                time_values=time_values,
                event_values=event_values,
                penalty=DEFAULT_PENALTY,
                ties=DEFAULT_TIES,
                tol=tol,
                max_iter=max_iter,
            )
        except Exception as exc:
            report["errors"].append(
                {
                    "backend": current,
                    "stage": "fixed_beta",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        try:
            report["fitted"][current] = _fit_backend(
                current,
                X=X,
                time_values=time_values,
                event_values=event_values,
                penalty=DEFAULT_PENALTY,
                ties=DEFAULT_TIES,
                tol=tol,
                max_iter=max_iter,
            )
        except Exception as exc:
            report["errors"].append(
                {
                    "backend": current,
                    "stage": "fitted",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    if "numpy" in report["fixed_beta"] and "numpy" in report["fitted"]:
        _add_numpy_self_checks(
            report["checks"],
            report["fixed_beta"]["numpy"],
            report["fitted"]["numpy"],
            max_iter=max_iter,
        )
        for current in requested:
            if current == "numpy":
                continue
            if current in report["fixed_beta"] and current in report["fitted"]:
                _add_backend_parity_checks(
                    report["checks"],
                    current,
                    report["fixed_beta"]["numpy"],
                    report["fixed_beta"][current],
                    report["fitted"]["numpy"],
                    report["fitted"][current],
                )

    if include_timing:
        for current in requested:
            if current not in report["fitted"]:
                continue
            try:
                report["timing"]["backends"][current] = _time_backend(
                    current,
                    X=X,
                    time_values=time_values,
                    event_values=event_values,
                    penalty=DEFAULT_PENALTY,
                    ties=DEFAULT_TIES,
                    tol=tol,
                    max_iter=max_iter,
                    warmup=timing_warmup,
                    repeats=timing_repeats,
                )
            except Exception as exc:
                report["errors"].append(
                    {
                        "backend": current,
                        "stage": "timing",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        numpy_timing = report["timing"]["backends"].get("numpy")
        if numpy_timing is not None:
            numpy_median = float(numpy_timing["median_seconds"])
            for timing_result in report["timing"]["backends"].values():
                timing_result["speedup_vs_numpy"] = numpy_median / float(
                    timing_result["median_seconds"]
                )

    failed_check = any(check["status"] != "pass" for check in report["checks"])
    missing = any(
        current not in report["fixed_beta"] or current not in report["fitted"]
        for current in requested
    )
    if report["errors"] or missing:
        report["status"] = "error"
    elif not report["checks"] or failed_check:
        report["status"] = "fail"
    else:
        report["status"] = "pass"
    return _json_safe(report)


def _parser(default_output):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-full-matrix",
        action="store_true",
        help="Print the expanded physical-GPU Cox matrix contract and exit.",
    )
    parser.add_argument(
        "--matrix-case-id",
        help="Execute exactly one expanded physical-GPU matrix case by case_id.",
    )
    parser.add_argument(
        "--run-full-matrix",
        action="store_true",
        help="Execute every physical-GPU matrix case (intentionally expensive).",
    )
    parser.add_argument(
        "--matrix-backend",
        choices=("all", "cupy", "torch"),
        default="all",
        help="Filter --run-full-matrix to one physical GPU backend.",
    )
    parser.add_argument(
        "--backend",
        choices=("all", "numpy", "cupy", "torch"),
        default="all",
        help="Backend to validate; GPU selections also run the NumPy reference.",
    )
    parser.add_argument(
        "--no-timing", action="store_true", help="Run correctness checks only."
    )
    parser.add_argument("--timing-warmup", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="JSON artifact path; use a single dash to disable output.",
    )
    return parser


def cli_main(argv=None, *, default_output=None):
    if default_output is None:
        default_output = (
            _PROJECT_ROOT / "results" / "pr79" / "accuracy" / "cox_pen_diagnostics.json"
        )
    args = _parser(default_output).parse_args(argv)
    matrix_action_count = sum(bool(value) for value in (
        args.print_full_matrix, args.matrix_case_id, args.run_full_matrix
    ))
    if matrix_action_count > 1:
        raise SystemExit(
            "choose only one of --print-full-matrix, --matrix-case-id, "
            "or --run-full-matrix"
        )
    if matrix_action_count:
        matrix = load_physical_gpu_matrix()
        cases = expand_physical_gpu_matrix(matrix)
        if args.print_full_matrix:
            result = {
                "matrix_schema_version": matrix["matrix_schema_version"],
                "case_count": len(cases),
                "cases": cases,
            }
            exit_code = 0
        else:
            if args.matrix_case_id:
                selected = [
                    case for case in cases
                    if case["case_id"] == args.matrix_case_id
                ]
                if not selected:
                    raise SystemExit(
                        "unknown physical-GPU matrix case_id: {}".format(
                            args.matrix_case_id
                        )
                    )
            else:
                selected = [
                    case for case in cases
                    if args.matrix_backend == "all"
                    or case["backend"] == args.matrix_backend
                ]
            result = run_physical_gpu_matrix(selected)
            exit_code = 0 if result["status"] == "pass" else 1
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if str(args.output) != "-":
            output_path = args.output
            if not output_path.is_absolute():
                output_path = _PROJECT_ROOT / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return exit_code
    report = build_report(
        backend=args.backend,
        include_timing=not args.no_timing,
        timing_warmup=args.timing_warmup,
        timing_repeats=args.timing_repeats,
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if str(args.output) != "-":
        output_path = args.output
        if not output_path.is_absolute():
            output_path = _PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "pass" else 1


def main(argv=None):
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
