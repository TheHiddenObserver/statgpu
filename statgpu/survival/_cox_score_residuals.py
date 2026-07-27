"""Case-wise Cox score residuals consistent with the selected tie method."""

from __future__ import annotations

from typing import Any, Optional

from ._risk_sets import (
    _array_namespace,
    _as_backend_array,
    _as_float,
    _center_within_strata,
    _exp,
    _max,
    _nonzero,
    _scalar_bool,
    _sum,
    _unique_sorted,
    _zeros,
    prepare_counting_process_inputs,
)


def cox_score_residuals(
    beta: Any,
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
    ties: str = "efron",
):
    """Return backend-native case-wise score contributions.

    For Breslow ties this is the conventional counting-process martingale
    score residual.  For an Efron tied group of size ``d``, each substep uses
    the same adjusted risk weights and mean as the Efron score,

    ``a_jk = w_j (1 - k/d * I[j in D]) / (S0 - k/d * E0)``.

    The event contribution is split evenly over the ``d`` substeps and the
    adjusted-risk contribution is centered at the corresponding Efron mean.
    Consequently the residuals satisfy ``residuals.sum(0) == score`` up to
    floating-point summation error, including with delayed entry and strata.
    """
    ties = str(ties).lower()
    if ties not in {"breslow", "efron"}:
        raise NotImplementedError(
            "case-wise score residuals are implemented for Breslow/Efron ties"
        )

    X, stop, event, start, strata = prepare_counting_process_inputs(
        X, stop, event, start=start, strata=strata
    )
    backend, xp = _array_namespace(X)
    beta = _as_backend_array(beta, backend, xp, X, name="beta").reshape(-1)
    if int(beta.shape[0]) != int(X.shape[1]):
        raise ValueError("beta must have shape (n_features,)")

    X_centered = _center_within_strata(X, strata, backend, xp)
    eta = X_centered @ beta
    residuals = _zeros(backend, xp, tuple(X_centered.shape), X_centered)

    for stratum in _unique_sorted(strata, backend, xp):
        stratum_idx = _nonzero(strata == stratum, backend, xp)
        Xs = X_centered[stratum_idx]
        stops = stop[stratum_idx]
        starts = start[stratum_idx]
        events = event[stratum_idx]
        etas = eta[stratum_idx]
        residual_stratum = _zeros(backend, xp, tuple(Xs.shape), Xs)
        failure_times = _unique_sorted(stops[events == 1], backend, xp)

        for failure_time in failure_times:
            fail_mask = (events == 1) & (stops == failure_time)
            risk_mask = (starts < failure_time) & (stops >= failure_time)
            fail_idx = _nonzero(fail_mask, backend, xp)
            risk_idx = _nonzero(risk_mask, backend, xp)
            d = int(fail_idx.shape[0])
            if d == 0:
                continue
            if int(risk_idx.shape[0]) == 0:
                raise FloatingPointError(
                    "empty Cox risk set at an observed failure time"
                )

            X_fail = Xs[fail_idx]
            X_risk = Xs[risk_idx]
            eta_shift = _max(etas[risk_idx], backend, xp)
            risk_weights = _exp(etas[risk_idx] - eta_shift, xp)
            s0 = _sum(risk_weights, backend, xp)
            if _scalar_bool(s0 <= 0):
                raise FloatingPointError("non-positive Cox risk-set denominator")
            s1 = risk_weights @ X_risk

            if ties == "breslow":
                mean = s1 / s0
                residual_stratum[fail_idx] = (
                    residual_stratum[fail_idx] + X_fail - mean
                )
                hazard_weight = risk_weights * (float(d) / s0)
                residual_stratum[risk_idx] = residual_stratum[risk_idx] - (
                    X_risk - mean
                ) * hazard_weight.reshape(-1, 1)
                continue

            fail_in_risk = _as_float(
                (events[risk_idx] == 1) & (stops[risk_idx] == failure_time),
                backend,
                Xs,
            )
            failure_weights = risk_weights * fail_in_risk
            e0 = _sum(failure_weights, backend, xp)
            e1 = failure_weights @ X_risk

            for substep in range(d):
                fraction = float(substep) / float(d)
                denominator = s0 - fraction * e0
                if _scalar_bool(denominator <= 0):
                    raise FloatingPointError(
                        "non-positive Cox risk-set denominator"
                    )
                mean = (s1 - fraction * e1) / denominator

                # Split the tied event numerator evenly across Efron substeps.
                residual_stratum[fail_idx] = residual_stratum[fail_idx] + (
                    X_fail - mean
                ) / float(d)

                # The risk contribution uses exactly the adjusted denominator
                # weights whose weighted mean appears in the Efron score.
                adjusted_weights = risk_weights * (
                    1.0 - fraction * fail_in_risk
                )
                normalized = adjusted_weights / denominator
                residual_stratum[risk_idx] = residual_stratum[risk_idx] - (
                    X_risk - mean
                ) * normalized.reshape(-1, 1)

        residuals[stratum_idx] = residual_stratum

    return residuals


__all__ = ["cox_score_residuals"]
