"""Newton solver for stratified/start-stop Cox counting-process models."""

from __future__ import annotations

from dataclasses import dataclass
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


_DEVICE_ERROR_MARKERS = (
    "out of memory",
    "cuda",
    "cublas",
    "cusolver",
    "illegal memory",
    "device mismatch",
    "device-side",
    "hip error",
    "driver error",
)
_SINGULAR_ERROR_MARKERS = (
    "singular",
    "not invertible",
    "not positive definite",
    "not positive-definite",
    "rank deficient",
    "rank-deficient",
    "ill-conditioned",
)


@dataclass(frozen=True)
class _PreparedRightCensoredCox:
    """Reusable sorted loss state for one ordinary right-censored dataset."""

    loss: Any
    X_sorted: Any
    source_X: Any
    source_stop: Any
    source_event: Any
    ties: str
    backend: str
    device: str
    n_samples: int
    n_features: int
    full_target_host_transfer_performed: bool

    def matches_sources(self, X: Any, stop: Any, event: Any, ties: str) -> bool:
        """Require identity matches so private CV state cannot fit other data."""
        return bool(
            X is self.source_X
            and stop is self.source_stop
            and event is self.source_event
            and str(ties).lower() == self.ties
        )

    def matches_content(
        self, X: Any, stop: Any, event: Any, ties: str
    ) -> bool:
        """Verify current inputs against the immutable preprocessed state.

        Identity alone cannot detect an in-place mutation after preparation.
        Compare the current arrays with the cached centered/sorted arrays on
        their existing backend and transfer only the final boolean result.
        """
        ties = str(ties).lower()
        backend, xp = _array_namespace(X)
        device = str(getattr(X, "device", "cpu"))
        shape = getattr(X, "shape", ())
        if (
            ties != self.ties
            or backend != self.backend
            or device != self.device
            or len(shape) not in (1, 2)
            or int(shape[0]) != self.n_samples
        ):
            return False

        X_arr = _as_backend_array(
            X, backend, xp, self.X_sorted, name="X"
        )
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        stop_arr = _as_backend_array(
            stop, backend, xp, self.X_sorted, name="stop"
        ).reshape(-1)
        event_arr = _as_backend_array(
            event, backend, xp, self.X_sorted, name="event"
        ).reshape(-1)
        if (
            tuple(X_arr.shape) != (self.n_samples, self.n_features)
            or int(stop_arr.shape[0]) != self.n_samples
            or int(event_arr.shape[0]) != self.n_samples
        ):
            return False

        loss = self.loss
        order = getattr(loss, "_order", None)
        x_reference = getattr(loss, "_x_reference", None)
        cached_time = getattr(loss, "_time_sorted", None)
        cached_event = getattr(loss, "_event_sorted", None)
        if (
            order is None
            or x_reference is None
            or cached_time is None
            or cached_event is None
            or getattr(loss, "_X_sorted", None) is not self.X_sorted
        ):
            return False

        current_X_sorted = (
            X_arr - x_reference.reshape(1, -1)
        )[order]
        matches = (
            xp.all(current_X_sorted == self.X_sorted)
            & xp.all(stop_arr[order] == cached_time)
            & xp.all(event_arr[order] == cached_event)
        )
        return _scalar_bool(matches)


def prepare_right_censored_cox_fast_path(
    X: Any,
    stop: Any,
    event: Any,
    *,
    ties: str,
) -> _PreparedRightCensoredCox:
    """Build once-per-dataset right-censored sorting/grouping metadata."""
    ties = str(ties).lower()
    if ties not in {"breslow", "efron"}:
        raise ValueError("right-censored preparation supports Breslow/Efron ties")
    backend, _ = _array_namespace(X)
    from statgpu.losses import CoxPartialLikelihoodLoss

    loss = CoxPartialLikelihoodLoss(ties=ties)
    X_sorted, _ = loss.preprocess(X, {"time": stop, "event": event})
    device = str(getattr(X_sorted, "device", "cpu"))
    target_was_device_resident = backend == "cupy" or (
        backend == "torch"
        and str(getattr(getattr(X_sorted, "device", None), "type", "cpu"))
        != "cpu"
    )
    return _PreparedRightCensoredCox(
        loss=loss,
        X_sorted=X_sorted,
        source_X=X,
        source_stop=stop,
        source_event=event,
        ties=ties,
        backend=backend,
        device=device,
        n_samples=int(X_sorted.shape[0]),
        n_features=int(X_sorted.shape[1]),
        full_target_host_transfer_performed=target_was_device_resident,
    )


def _is_singular_linalg_error(exc: BaseException) -> bool:
    """Identify numerical singularity without swallowing device/runtime errors."""
    message = str(exc).lower()
    if any(marker in message for marker in _DEVICE_ERROR_MARKERS):
        return False
    return any(marker in message for marker in _SINGULAR_ERROR_MARKERS)


def _solve(information: Any, score: Any, backend: str, xp: Any):
    try:
        return xp.linalg.solve(information, score)
    except Exception as exc:
        if not _is_singular_linalg_error(exc):
            raise
        # Stay on the selected backend.  A least-squares solve is a numerical
        # fallback, not a device fallback.
        try:
            if backend == "torch":
                return xp.linalg.lstsq(information, score.unsqueeze(1)).solution[:, 0]
            return xp.linalg.lstsq(information, score, rcond=None)[0]
        except Exception as fallback_exc:
            if not _is_singular_linalg_error(fallback_exc):
                raise
            raise RuntimeError(
                f"{backend} Cox observed information is singular"
            ) from fallback_exc


def _score_test_statistic(score: Any, information: Any, backend: str, xp: Any):
    """Return the null-score quadratic form or an explicit singular reason."""
    try:
        delta = xp.linalg.solve(information, score)
    except Exception as exc:
        if not _is_singular_linalg_error(exc):
            raise
        return None, f"{backend} null information is singular: {exc}"
    return score @ delta, None


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
    right_censored_fast_path: bool = False,
    right_censored_prepared: Optional[_PreparedRightCensoredCox] = None,
    _inputs_prepared: bool = False,
) -> Dict[str, Any]:
    """Fit a Cox model using a backend-native damped Newton method.

    The optimized objective is ``log_partial_likelihood - penalty * ||beta||²``.
    Every rejected Newton step is handled by backtracking; an iteration never
    silently accepts a step that decreases the penalized objective.
    """
    if not _inputs_prepared:
        X, stop, event, start, strata = prepare_counting_process_inputs(
            X, stop, event, start=start, strata=strata
        )
    backend, xp = _array_namespace(X)
    n_features = int(X.shape[1])
    if init_coef is None:
        beta = _as_backend_array([0.0] * n_features, backend, xp, X)
    else:
        beta = _as_backend_array(
            init_coef, backend, xp, X, name="init_coef"
        ).reshape(-1)
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
    fast_loss = None
    fast_X = None
    prepared_created_here = False
    if right_censored_fast_path:
        if ties not in {"breslow", "efron"}:
            raise ValueError(
                "right_censored_fast_path supports only Breslow/Efron ties"
            )
        if compute_score_residuals:
            raise ValueError(
                "right_censored_fast_path does not compute score residuals"
            )
        ordinary_inputs = xp.all(start == 0) & xp.all(strata == strata[0])
        if not _scalar_bool(ordinary_inputs):
            raise ValueError(
                "right_censored_fast_path requires all-zero start times and "
                "a single stratum"
            )
        if right_censored_prepared is None:
            right_censored_prepared = prepare_right_censored_cox_fast_path(
                X, stop, event, ties=ties
            )
            prepared_created_here = True
        if not isinstance(
            right_censored_prepared, _PreparedRightCensoredCox
        ) or (
            right_censored_prepared.ties != ties
            or right_censored_prepared.backend != backend
            or right_censored_prepared.device
            != str(getattr(X, "device", "cpu"))
            or right_censored_prepared.n_samples != int(X.shape[0])
            or right_censored_prepared.n_features != n_features
            or (
                not prepared_created_here
                and not right_censored_prepared.matches_content(
                    X, stop, event, ties
                )
            )
        ):
            raise ValueError(
                "prepared right-censored metadata does not match fit backend, "
                "device, ties, dataset shape, or dataset contents"
            )
        fast_loss = right_censored_prepared.loss
        fast_X = right_censored_prepared.X_sorted
    elif right_censored_prepared is not None:
        raise ValueError(
            "prepared right-censored metadata requires right_censored_fast_path"
        )

    def evaluate(coef):
        if fast_loss is None:
            return cox_counting_process_objective(
                coef, X, stop, event, start=start, strata=strata, ties=ties
            )
        eta = fast_X @ coef
        loglik, score, hessian = fast_loss._objective_from_eta_backend(
            eta,
            fast_X,
            xp,
            ties,
            compute_information=True,
        )
        return {
            "log_likelihood": loglik,
            "score": score,
            "information": -hessian,
        }

    converged = False
    iterations = 0
    stop_reason = "max_iter"
    objective_history = []

    current = evaluate(beta)
    initial_null = current if init_coef is None else None
    current_penalized = current["log_likelihood"] - penalty * (beta @ beta)
    objective_history.append(current_penalized)

    for iteration in range(max_iter):
        iterations = iteration + 1
        penalized_score = current["score"] - 2.0 * penalty * beta
        score_inf = xp.max(xp.abs(penalized_score))
        raw_score_inf = xp.max(xp.abs(current["score"]))
        beta_inf = xp.max(xp.abs(beta))
        kkt_normalized = score_inf / (
            1.0 + raw_score_inf + 2.0 * penalty * beta_inf
        )
        if _scalar_bool(kkt_normalized <= tol):
            converged = True
            stop_reason = "kkt_converged"
            break
        penalized_information = current["information"] + 2.0 * penalty * identity
        delta = _solve(penalized_information, penalized_score, backend, xp)
        directional = penalized_score @ delta
        if _scalar_bool((~xp.isfinite(directional)) | (directional <= 0.0)):
            # At a saturated but finite predictor, the observed information can
            # round to zero while the score remains informative.  A normalized
            # score direction lets backtracking leave that boundary safely.
            delta = penalized_score / (1.0 + xp.max(xp.abs(penalized_score)))

        step = 1.0
        accepted = False
        candidate = None
        candidate_penalized = None
        for _ in range(30):
            candidate_beta = beta + step * delta
            trial = evaluate(candidate_beta)
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
            # Return the last accepted iterate instead of turning an ordinary
            # numerical non-convergence into an estimator-level exception.  The
            # caller receives converged=False and can decide whether to exclude
            # the candidate (for example, in CoxPHCV) while programming, input,
            # import, and device errors still propagate normally.
            stop_reason = "line_search_failed"
            break

        beta, current = candidate
        current_penalized = candidate_penalized
        objective_history.append(current_penalized)
        # A small Newton step alone is not a convergence certificate.  The
        # next iteration evaluates the normalized KKT residual at the accepted
        # coefficient vector.

    final = (
        cox_counting_process_objective(
            beta,
            X,
            stop,
            event,
            start=start,
            strata=strata,
            ties=ties,
            score_residuals=True,
        )
        if compute_score_residuals
        else current
    )
    final_penalized_score = final["score"] - 2.0 * penalty * beta
    final_score_inf = xp.max(xp.abs(final_penalized_score))
    final_raw_score_inf = xp.max(xp.abs(final["score"]))
    final_beta_inf = xp.max(xp.abs(beta))
    final_kkt_normalized = final_score_inf / (
        1.0 + final_raw_score_inf + 2.0 * penalty * final_beta_inf
    )
    if _scalar_bool(final_kkt_normalized <= tol):
        converged = True
        stop_reason = "kkt_converged"

    if initial_null is None:
        null_beta = beta * 0.0
        null_result = evaluate(null_beta)
    else:
        null_result = initial_null
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
        "null_score": null_result["score"],
        "null_information": null_result["information"],
        "score": final["score"],
        "penalized_score": final_penalized_score,
        "information": final["information"],
        "score_residuals": final.get("score_residuals"),
        "baseline": baseline,
        "iterations": iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "objective_history": objective_history,
        "full_target_host_transfer_performed": bool(
            right_censored_prepared is not None
            and right_censored_prepared.full_target_host_transfer_performed
        ),
    }


__all__ = []
