"""Issue #169 contracts: scalar Quantile LLA accuracy and route delegation.

The public scalar Quantile SCAD/MCP low-level route delegates to the maintained
dedicated Proximal IRLS-LLA engine, which reaches an independent LP fixed-point
oracle on a nontrivial converged fixture. Warm-started and path-reporting calls
keep the historical fused FISTA-LLA engine, and group penalties keep the group
surrogate route.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_lla_path
from statgpu.solvers._convergence import ConvergenceWarning


Q = 0.35
ALPHA = 0.5
SCAD_A = 3.7
TARGET_GAP = 1e-8
TARGET_MAX_ITER = 20000
TARGET_MAX_LLA = 8
TARGET_TOL = 1e-10


def _fixture():
    rng = np.random.default_rng(4242)
    x = np.linspace(0.5, 1.5, 48).reshape(-1, 1)
    y = (1.0 + 0.8 * x[:, 0] + rng.laplace(scale=0.3, size=48)).astype(
        np.float64
    )
    weights = np.linspace(0.4, 1.7, 48, dtype=np.float64)
    rng.shuffle(weights)
    return x, y, weights


def _scad_lla_weights(beta):
    abs_beta = np.abs(beta)
    weights = np.full_like(abs_beta, ALPHA)
    interior = (abs_beta > ALPHA) & (abs_beta <= SCAD_A * ALPHA)
    weights[interior] = (
        SCAD_A * ALPHA - abs_beta[interior]
    ) / (SCAD_A - 1.0)
    weights[abs_beta > SCAD_A * ALPHA] = 0.0
    return weights


def _weighted_quantile_objective(x, y, weights, beta, l1_coeffs):
    residual = y - np.asarray(x, dtype=np.float64) @ np.asarray(
        beta, dtype=np.float64
    ).reshape(-1)
    pinball = np.where(residual >= 0.0, Q * residual, (Q - 1.0) * residual)
    fit = float(
        np.average(pinball, weights=np.asarray(weights, dtype=np.float64))
    )
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


def test_scalar_quantile_lla_delegates_to_dedicated_proximal_solver(
    monkeypatch,
):
    import statgpu.solvers._quantile_proximal_public_contract as public_mod

    calls = []

    def recording_solver(loss, penalty, X, y, alpha_path, **kwargs):
        calls.append((loss, penalty, alpha_path, kwargs))
        return np.zeros(int(np.asarray(X).shape[1])), 0.0, 1

    monkeypatch.setattr(
        public_mod, "proximal_irls_quantile_solver", recording_solver
    )

    x, y, weights = _fixture()
    fista_lla_path(
        QuantileLoss(Q),
        SCADPenalty(alpha=ALPHA, a=SCAD_A),
        x,
        y,
        alpha_path=[ALPHA],
        max_lla_per_step=2,
        max_iter=30,
        tol=1e-8,
        lla_tol=1e-8,
        fit_intercept=False,
        sample_weight=weights,
    )

    assert len(calls) == 1
    _, penalty, alpha_path, kwargs = calls[0]
    assert penalty.name == "scad"
    assert list(alpha_path) == [ALPHA]
    assert kwargs["sample_weight"] is weights


def test_scalar_quantile_warm_start_and_path_keep_fused_engine(monkeypatch):
    import statgpu.solvers._quantile_proximal_public_contract as public_mod

    def forbidden_solver(*args, **kwargs):
        raise AssertionError(
            "dedicated solver must not run for warm-started/path-reporting calls"
        )

    monkeypatch.setattr(
        public_mod, "proximal_irls_quantile_solver", forbidden_solver
    )

    x, y, _ = _fixture()
    penalty = SCADPenalty(alpha=ALPHA, a=SCAD_A)
    warm = fista_lla_path(
        QuantileLoss(Q),
        penalty,
        x,
        y,
        alpha_path=[ALPHA],
        max_lla_per_step=1,
        max_iter=5,
        tol=0.5,
        lla_tol=0.5,
        fit_intercept=False,
        init_coef=np.zeros(1, dtype=np.float64),
    )
    assert np.all(np.isfinite(np.asarray(warm[0])))
    assert int(warm[2]) >= 1

    path_result = fista_lla_path(
        QuantileLoss(Q),
        penalty,
        x,
        y,
        alpha_path=[ALPHA],
        max_lla_per_step=1,
        max_iter=5,
        tol=0.5,
        lla_tol=0.5,
        fit_intercept=False,
        return_path=True,
    )
    assert len(path_result) == 4
    assert np.all(np.isfinite(np.asarray(path_result[0])))


def test_scalar_quantile_custom_factory_keeps_fused_engine(monkeypatch):
    import statgpu.solvers._fista_lla_group_contract as contract
    import statgpu.solvers._quantile_proximal_public_contract as public_mod

    def forbidden_solver(*args, **kwargs):
        raise AssertionError(
            "dedicated solver must not run when a custom factory is supplied"
        )

    monkeypatch.setattr(
        public_mod, "proximal_irls_quantile_solver", forbidden_solver
    )

    captured = {}

    def fake_base(loss, penalty, X, y, alpha_path, **kwargs):
        captured["factory"] = kwargs.get("lla_penalty_factory")
        return np.zeros(int(np.asarray(X).shape[1])), 0.0, 1

    monkeypatch.setattr(contract, "_base_fista_lla_path", fake_base)

    x, y, _ = _fixture()
    factory = lambda derivatives: None
    result = fista_lla_path(
        QuantileLoss(Q),
        SCADPenalty(alpha=ALPHA, a=SCAD_A),
        x,
        y,
        alpha_path=[ALPHA],
        max_lla_per_step=1,
        max_iter=5,
        tol=0.5,
        lla_tol=0.5,
        fit_intercept=False,
        lla_penalty_factory=factory,
    )

    assert result[2] == 1
    assert captured["factory"] is factory


def test_scalar_quantile_lla_reaches_lp_fixed_point():
    pytest.importorskip("scipy.optimize")
    x, y, weights = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, intercept, n_iter = fista_lla_path(
            QuantileLoss(Q),
            SCADPenalty(alpha=ALPHA, a=SCAD_A),
            x,
            y,
            alpha_path=[ALPHA],
            max_lla_per_step=TARGET_MAX_LLA,
            max_iter=TARGET_MAX_ITER,
            tol=TARGET_TOL,
            lla_tol=TARGET_TOL,
            fit_intercept=False,
            sample_weight=weights,
        )

    beta = np.asarray(coef, dtype=np.float64).reshape(-1)
    l1_coeffs = _scad_lla_weights(beta)
    assert np.any(l1_coeffs > 0.0)
    abs_beta = np.abs(beta)
    assert np.any((abs_beta > ALPHA) & (abs_beta <= SCAD_A * ALPHA))
    beta_lp, lp_value = _weighted_quantile_l1_lp_reference(
        x, y, weights, l1_coeffs
    )
    gap = (
        _weighted_quantile_objective(x, y, weights, beta, l1_coeffs)
        - lp_value
    )
    assert gap <= TARGET_GAP, gap
    assert np.max(np.abs(beta - beta_lp)) <= 1e-6
    assert np.isfinite(float(intercept))
    assert int(n_iter) >= 1
