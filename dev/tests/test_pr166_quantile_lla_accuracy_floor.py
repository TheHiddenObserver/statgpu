"""Documented accuracy floor of the scalar Quantile FISTA-LLA inner solve.

The PR166 scalar LLA physical gate certifies convergence/refresh behavior and
CPU/GPU trajectory parity; it does not certify objective optimality. The
fixed-step inner FISTA used by ``fista_lla_path`` can stall above the convex
weighted-L1 optimum while still reporting convergence.

These hosted contracts make that gap explicit and regression-bounded:

- the independent HiGHS LP oracle and the backtracking ``fista_solver``
  control path are exact, so the measured gap isolates the LLA inner solver;
- the current gap is bounded from above by a regression ceiling;
- reaching the target accuracy is ``xfail(strict=True)`` until the inner
  solver is upgraded (follow-up task); once it passes, the strict marker
  forces this module to be updated and the oracle promoted into the physical
  validator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
from statgpu.penalties._adaptive_l1 import AdaptiveL1Penalty
from statgpu.solvers import fista_lla_path, fista_solver
from statgpu.solvers._convergence import ConvergenceWarning


Q = 0.35
SCAD_ALPHA = 0.04
SCAD_A = 3.7
L1_COEFFS = np.asarray([0.05], dtype=np.float64)
CONTROL_MAX_ITER = 20000
CONTROL_TOL = 1e-10
LLA_MAX_ITER = 5000
LLA_MAX_STEPS = 3
LLA_TOL = 1e-10
CURRENT_GAP_CEILING = 5e-3
TARGET_GAP = 1e-8


def _data():
    rng = np.random.default_rng(4242)
    x = np.linspace(0.5, 1.5, 48).reshape(-1, 1)
    y = (1.0 + 0.8 * x[:, 0] + rng.laplace(scale=0.3, size=48)).astype(
        np.float64
    )
    weights = np.linspace(0.4, 1.7, 48, dtype=np.float64)
    rng.shuffle(weights)
    return x, y, weights


def _weighted_quantile_objective(x, y, weights, beta, l1_coeffs):
    residual = y - np.asarray(x, dtype=np.float64) @ np.asarray(
        beta, dtype=np.float64
    ).reshape(-1)
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(np.average(pinball, weights=np.asarray(weights, dtype=np.float64)))
    return fit + float(np.sum(np.asarray(l1_coeffs) * np.abs(beta)))


def _weighted_quantile_l1_lp_reference(x, y, weights, l1_coeffs):
    """Solve ``mean pinball + sum_j c_j |b_j|`` with SciPy HiGHS."""
    from scipy.optimize import linprog

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    l1_coeffs = np.asarray(l1_coeffs, dtype=np.float64).reshape(-1)
    n, p = x.shape
    up, um = 2 * p, 2 * p + n
    objective = np.zeros(2 * p + 2 * n, dtype=np.float64)
    objective[:p] = l1_coeffs
    objective[p : 2 * p] = l1_coeffs
    normalized_weight = weights / float(np.sum(weights))
    objective[up : up + n] = Q * normalized_weight
    objective[um:] = (1.0 - Q) * normalized_weight
    A_eq = np.zeros((n, 2 * p + 2 * n), dtype=np.float64)
    A_eq[:, :p] = x
    A_eq[:, p : 2 * p] = -x
    A_eq[:, up : up + n] = np.eye(n, dtype=np.float64)
    A_eq[:, um:] = -np.eye(n, dtype=np.float64)
    result = linprog(
        objective,
        A_eq=A_eq,
        b_eq=y,
        bounds=[(0.0, None)] * (2 * p + 2 * n),
        method="highs",
    )
    if not bool(result.success):
        raise AssertionError(
            "weighted Quantile-L1 LP reference failed: "
            f"status={result.status}, message={result.message!r}"
        )
    beta = result.x[:p] - result.x[p : 2 * p]
    reconstructed = _weighted_quantile_objective(
        x, y, weights, beta, l1_coeffs
    )
    if abs(reconstructed - float(result.fun)) > 1e-10:
        raise AssertionError(
            "weighted Quantile-L1 LP objective reconstruction drifted: "
            f"{reconstructed:.16g} vs solver {float(result.fun):.16g}"
        )
    return beta, float(result.fun)


class _ConstantWeightSCAD(SCADPenalty):
    """SCAD stand-in whose LLA weights do not move with the iterate."""

    def __init__(self, weights):
        super().__init__(alpha=SCAD_ALPHA, a=SCAD_A)
        self._fixed = np.asarray(weights, dtype=np.float64)

    def lla_weights(self, coef):
        return self._fixed.copy()


def _lla_inner_solution(x, y, weights, l1_coeffs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        coef, _, _ = fista_lla_path(
            QuantileLoss(Q),
            _ConstantWeightSCAD(l1_coeffs),
            x,
            y,
            alpha_path=np.asarray([SCAD_ALPHA]),
            max_lla_per_step=LLA_MAX_STEPS,
            max_iter=LLA_MAX_ITER,
            lla_tol=LLA_TOL,
            tol=LLA_TOL,
            fit_intercept=False,
            sample_weight=weights,
        )
    return np.asarray(coef, dtype=np.float64).reshape(-1)


def _control_solution(x, y, weights, l1_coeffs):
    penalty = AdaptiveL1Penalty(
        alpha=1.0,
        normalize=False,
        weights=tuple(float(value) for value in l1_coeffs),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        coef, _ = fista_solver(
            QuantileLoss(Q),
            penalty,
            x,
            y,
            max_iter=CONTROL_MAX_ITER,
            tol=CONTROL_TOL,
            sample_weight=weights,
        )
    return np.asarray(coef, dtype=np.float64).reshape(-1)


def test_weighted_quantile_l1_oracle_and_control_path_are_exact():
    pytest.importorskip("scipy.optimize")
    x, y, weights = _data()
    _, lp_value = _weighted_quantile_l1_lp_reference(
        x, y, weights, L1_COEFFS
    )
    control = _control_solution(x, y, weights, L1_COEFFS)
    control_gap = (
        _weighted_quantile_objective(x, y, weights, control, L1_COEFFS)
        - lp_value
    )
    assert control_gap <= 1e-10, control_gap


def test_quantile_lla_inner_solver_accuracy_floor_is_bounded():
    pytest.importorskip("scipy.optimize")
    x, y, weights = _data()
    _, lp_value = _weighted_quantile_l1_lp_reference(
        x, y, weights, L1_COEFFS
    )
    lla = _lla_inner_solution(x, y, weights, L1_COEFFS)
    gap = (
        _weighted_quantile_objective(x, y, weights, lla, L1_COEFFS)
        - lp_value
    )
    assert 0.0 <= gap <= CURRENT_GAP_CEILING, gap


@pytest.mark.xfail(
    strict=True,
    reason=(
        "documented: the fixed-step Quantile FISTA-LLA inner solve stalls "
        "~3e-4 above the LP optimum; remove this marker and promote the LP "
        "oracle into the physical validator once the inner solver is upgraded"
    ),
)
def test_quantile_lla_inner_solver_reaches_target_accuracy():
    pytest.importorskip("scipy.optimize")
    x, y, weights = _data()
    _, lp_value = _weighted_quantile_l1_lp_reference(
        x, y, weights, L1_COEFFS
    )
    lla = _lla_inner_solution(x, y, weights, L1_COEFFS)
    gap = (
        _weighted_quantile_objective(x, y, weights, lla, L1_COEFFS)
        - lp_value
    )
    assert gap <= TARGET_GAP, gap
