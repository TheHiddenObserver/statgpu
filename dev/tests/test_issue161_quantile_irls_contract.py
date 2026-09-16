"""Issue #161 regressions for the low-level Quantile IRLS penalty contract."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedQuantileRegression
from statgpu.losses import QuantileLoss
from statgpu.penalties import ElasticNetPenalty, L1Penalty, L2Penalty, SCADPenalty


def _data(seed=16101, n=96):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = np.column_stack([x, np.ones(n)]).astype(np.float64)
    y = (0.8 * x + 0.35 + rng.laplace(scale=0.12, size=n)).astype(np.float64)
    return X, y


@pytest.mark.parametrize(
    "penalty",
    [
        ElasticNetPenalty(alpha=0.2, l1_ratio=0.4),
        L1Penalty(alpha=0.2),
        SCADPenalty(alpha=0.2),
    ],
)
def test_direct_quantile_irls_rejects_non_smooth_penalties_before_linear_solve(
    monkeypatch, penalty
):
    X, y = _data()

    def forbidden(*args, **kwargs):
        raise AssertionError("linear solve must not run for unsupported IRLS penalty")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)

    with pytest.raises(ValueError, match="supports only L2 or no penalty"):
        QuantileLoss(quantile=0.5).irls(
            X,
            y,
            penalty=penalty,
            fit_intercept=True,
        )


def test_direct_quantile_irls_rejects_unknown_penalty_objects_fail_closed():
    X, y = _data(seed=16102)

    class UnknownPenalty:
        alpha = 0.1

    with pytest.raises(ValueError, match="supports only L2 or no penalty"):
        QuantileLoss(quantile=0.5).irls(X, y, penalty=UnknownPenalty())


@pytest.mark.parametrize("penalty", [None, L2Penalty(alpha=0.05)])
def test_direct_quantile_irls_keeps_supported_none_and_l2_routes(penalty):
    X, y = _data(seed=16103)
    coef, n_iter = QuantileLoss(quantile=0.5).irls(
        X,
        y,
        penalty=penalty,
        fit_intercept=True,
        max_iter=200,
        tol=1e-9,
    )

    coef = np.asarray(coef, dtype=np.float64)
    assert coef.shape == (X.shape[1],)
    assert np.all(np.isfinite(coef))
    assert 1 <= n_iter <= 200


def test_weighted_l2_quantile_irls_is_invariant_to_global_weight_rescaling():
    X, y = _data(seed=16104)
    weights = np.linspace(0.4, 1.8, X.shape[0], dtype=np.float64)
    loss = QuantileLoss(quantile=0.35)
    penalty = L2Penalty(alpha=0.03)

    base, _ = loss.irls(
        X,
        y,
        penalty=penalty,
        fit_intercept=True,
        sample_weight=weights,
        max_iter=250,
        tol=1e-10,
    )
    scaled, _ = loss.irls(
        X,
        y,
        penalty=penalty,
        fit_intercept=True,
        sample_weight=7.25 * weights,
        max_iter=250,
        tol=1e-10,
    )

    np.testing.assert_allclose(
        np.asarray(base), np.asarray(scaled), rtol=2e-10, atol=2e-11
    )


def test_public_quantile_estimator_keeps_irls_elasticnet_fail_closed():
    X, y = _data(seed=16105)
    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty="elasticnet",
        alpha=0.1,
        l1_ratio=0.4,
        solver="irls",
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="solver='irls' only supports smooth L2 or no-penalty objectives",
    ):
        model.fit(X[:, :1], y)


def test_direct_l2_quantile_irls_preserves_torch_cpu_backend_when_available(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    X_np, y_np = _data(seed=16106, n=48)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)

    real_ones = torch.ones
    penalty_diag_devices = []

    def tracked_ones(*args, **kwargs):
        if args and args[0] == X.shape[1] and kwargs.get("dtype") == torch.float64:
            penalty_diag_devices.append(kwargs.get("device"))
        return real_ones(*args, **kwargs)

    monkeypatch.setattr(torch, "ones", tracked_ones)

    coef, n_iter = QuantileLoss(quantile=0.5).irls(
        X,
        y,
        penalty=L2Penalty(alpha=0.02),
        fit_intercept=True,
        max_iter=120,
        tol=1e-8,
    )

    assert penalty_diag_devices
    assert all(device == X.device for device in penalty_diag_devices)
    assert isinstance(coef, torch.Tensor)
    assert coef.device.type == "cpu"
    assert coef.dtype == torch.float64
    assert bool(torch.all(torch.isfinite(coef)).item())
    assert 1 <= n_iter <= 120
