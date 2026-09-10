"""Independent positive-reference checks for standardized node-wise precision."""

import numpy as np
import pytest

from statgpu.linear_model.penalized._nodewise_precision import (
    NODEWISE_KKT_TOL,
    build_nodewise_precision_numpy,
)


def _soft_threshold(value: float, alpha: float) -> float:
    return float(np.sign(value) * max(abs(value) - alpha, 0.0))


def test_explicit_alpha_matches_closed_form_two_feature_reference():
    """Check p=2 against the analytic one-coordinate Lasso solution.

    With two standardized features every node-wise nuisance problem has one
    coefficient, so its Lasso solution is exactly soft-threshold(correlation,
    alpha).  This reference does not call statgpu's path solver.
    """
    rng = np.random.default_rng(314159)
    latent = rng.normal(size=120)
    X = np.column_stack(
        [
            1.7 * latent + 0.35 * rng.normal(size=latent.size),
            0.6 * latent + 0.55 * rng.normal(size=latent.size),
        ]
    )
    X -= X.mean(axis=0)
    n = int(X.shape[0])
    alpha = 0.08

    sigma = X.T @ X / float(n)
    d = np.sqrt(np.diag(sigma))
    sigma_z = sigma / d[:, None] / d[None, :]
    corr = float(sigma_z[0, 1])
    gamma = _soft_threshold(corr, alpha)
    tau = float(
        1.0
        - 2.0 * corr * gamma
        + gamma * gamma
        + alpha * abs(gamma)
    )
    theta_ref = np.asarray(
        [[1.0, -gamma], [-gamma, 1.0]], dtype=np.float64
    ) / tau
    M_ref = theta_ref / d[:, None] / d[None, :]

    M, resolved, meta = build_nodewise_precision_numpy(
        X,
        requested_alpha=alpha,
        effective_n=float(n),
        weighted=False,
    )

    assert resolved == pytest.approx(alpha)
    assert meta["nodewise_alpha_source"] == "user"
    assert meta["nodewise_max_kkt_residual"] <= NODEWISE_KKT_TOL
    np.testing.assert_allclose(M, M_ref, rtol=2e-7, atol=2e-9)

    # Positive KKT/M Sigma diagnostic on the standardized scale.  The diagonal
    # is one, while each off-diagonal residual is bounded by alpha / tau_j.
    theta = d[:, None] * M * d[None, :]
    certificate = theta @ sigma_z
    np.testing.assert_allclose(np.diag(certificate), np.ones(2), rtol=2e-7, atol=2e-9)
    for j in range(2):
        k = 1 - j
        assert abs(float(certificate[j, k])) <= (
            alpha * float(theta[j, j]) + 5e-7
        )
