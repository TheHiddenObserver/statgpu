"""Newton solver for stratified/start-stop Cox counting-process models."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numbers
import numpy as np

from ._risk_sets import (
    _array_namespace,
    _as_backend_array,
    _eye,
    _scalar_bool,
    cox_baseline_hazard,
    cox_counting_process_objective,
    prepare_counting_process_inputs,
)


def _norm(value: Any, backend: str, xp: Any):
    if backend == "torch":
        return xp.linalg.vector_norm(value)
    return xp.linalg.norm(value)


def _solve(information: Any, score: Any, backend: str, xp: Any):
    try:
        return xp.linalg.solve(information, score)
    except Exception as exc:
        # Stay on the selected backend.  A least-squares solve is a numerical
        # fallback, not a device fallback.
        try:
            if backend == "torch":
                return xp.linalg.lstsq(information, score.unsqueeze(1)).solution[:, 0]
            return xp.linalg.lstsq(information, score, rcond=None)[0]
        except Exception:
            raise RuntimeError("Cox observed information is singular") from exc


def fit_counting_process_cox(
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
    ties: str = "efron",
    penalty: float = 0.0,
    tol: float = 1e-9,
    max_iter: int = 100,
    init_coef: Optional[Any] = None,
    compute_baseline: bool = True,
    compute_score_residuals: bool = True,
) -> Dict[str, Any]:
    """Fit a Cox model using a backend-native damped Newton method.

    The optimized objective is ``log_partial_likelihood - penalty * ||beta||²``.
    Every rejected Newton step is handled by backtracking; an iteration never
    silently accepts a step that decreases the penalized objective.
    """
    X, stop, event, start, strata = prepare_counting_process_inputs(
        X, stop, event, start=start, strata=strata
    )
    backend, xp = _array_namespace(X)
    n_features = int(X.shape[1])
    if init_coef is None:
        beta = _as_backend_array([0.0] * n_features, backend, xp, X)
    else:
        beta = _as_backend_array(init_coef, backend, xp, X).reshape(-1)
        if int(beta.shape[0]) != n_features:
            raise ValueError("init_coef must have shape (n_features,)")
        if not _scalar_bool(xp.all(xp.isfinite(beta))):
            raise ValueError("init_coef must contain only finite values")

    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be a finite non-negative number")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, numbers.Integral
    ) or int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number")

    identity = _eye(backend, xp, n_features, X)
    converged = False
    iterations = 0
    stop_reason = "max_iter"
    objective_history = []

    current = cox_counting_process_objective(
        beta, X, stop, event, start=start, strata=strata, ties=ties
    )
    current_penalized = current["log_likelihood"] - penalty * (beta @ beta)
    objective_history.append(current_penalized)

    for iteration in range(max_iter):
        iterations = iteration + 1
        penalized_score = current["score"] - 2.0 * penalty * beta
        penalized_information = current["information"] + 2.0 * penalty * identity
        delta = _solve(penalized_information, penalized_score, backend, xp)
        delta_norm = _norm(delta, backend, xp)
        if _scalar_bool(delta_norm <= tol * (1.0 + _norm(beta, backend, xp))):
            converged = True
            stop_reason = "newton_step"
            break

        step = 1.0
        accepted = False
        candidate = None
        candidate_penalized = None
        for _ in range(30):
            candidate_beta = beta + step * delta
            trial = cox_counting_process_objective(
                candidate_beta,
                X,
                stop,
                event,
                start=start,
                strata=strata,
                ties=ties,
            )
            trial_penalized = trial["log_likelihood"] - penalty * (
                candidate_beta @ candidate_beta
            )
            # Armijo ascent with a tiny absolute cushion for floating error.
            directional = penalized_score @ delta
            threshold = current_penalized + 1e-4 * step * directional - 1e-12
            if _scalar_bool(trial_penalized >= threshold):
                accepted = True
                candidate = (candidate_beta, trial)
                candidate_penalized = trial_penalized
                break
            step *= 0.5

        if not accepted:
            stop_reason = "line_search_failed"
            raise RuntimeError(
                "Cox Newton line search failed to find an improving step"
            )

        beta, current = candidate
        current_penalized = candidate_penalized
        objective_history.append(current_penalized)
        if _scalar_bool(step * delta_norm <= tol * (1.0 + _norm(beta, backend, xp))):
            converged = True
            stop_reason = "newton_step"
            break

    final = cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
        score_residuals=bool(compute_score_residuals),
    )
    final_penalized_score = final["score"] - 2.0 * penalty * beta
    if not converged and _scalar_bool(
        _norm(final_penalized_score, backend, xp)
        <= 10.0 * tol * (1.0 + _norm(beta, backend, xp))
    ):
        converged = True
        stop_reason = "score_norm"

    null_beta = beta * 0.0
    null_result = cox_counting_process_objective(
        null_beta, X, stop, event, start=start, strata=strata, ties=ties
    )
    baseline = (
        cox_baseline_hazard(beta, X, stop, event, start=start, strata=strata, ties=ties)
        if compute_baseline
        else None
    )
    return {
        "coef": beta,
        "log_likelihood": final["log_likelihood"],
        "penalized_log_likelihood": final["log_likelihood"] - penalty * (beta @ beta),
        "null_log_likelihood": null_result["log_likelihood"],
        "score": final["score"],
        "penalized_score": final_penalized_score,
        "information": final["information"],
        "score_residuals": final.get("score_residuals"),
        "baseline": baseline,
        "iterations": iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "objective_history": objective_history,
    }
