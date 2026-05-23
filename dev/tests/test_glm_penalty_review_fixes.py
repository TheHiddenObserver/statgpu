import numpy as np
import pytest

from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
from statgpu.glm_core._solver import (
    _fused_glm_value_and_gradient,
    admm_solver,
    fista_bb_solver,
    fista_solver,
    lbfgs_solver,
    newton_solver,
)
from statgpu.glm_core._squared import SquaredErrorLoss
from statgpu.glm_core._tweedie import TweedieLoss
from statgpu.linear_model import GeneralizedLinearModel, PenalizedLinearRegression
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import (
    GroupMCPPenalty,
    GroupSCADPenalty,
    L2Penalty,
    MCPPenalty,
    SCADPenalty,
)


def test_negative_binomial_loss_alpha_matches_finite_difference_gradient():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(40, 4))
    y = rng.poisson(lam=2.0, size=40).astype(float)
    coef = rng.normal(scale=0.15, size=4)
    loss = NegativeBinomialLoss(alpha=0.35)

    analytical = loss.gradient(X, y, coef)
    numerical = np.empty_like(coef)
    eps = 1e-6
    for j in range(coef.size):
        step = np.zeros_like(coef)
        step[j] = eps
        numerical[j] = (
            loss.value(X, y, coef + step) - loss.value(X, y, coef - step)
        ) / (2.0 * eps)

    assert np.allclose(analytical, numerical, rtol=1e-4, atol=1e-5)


def test_fused_negative_binomial_uses_loss_alpha():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(32, 3))
    y = rng.poisson(lam=1.7, size=32).astype(float)
    coef = rng.normal(scale=0.2, size=3)
    loss = NegativeBinomialLoss(alpha=0.25)

    fused_value, fused_grad = _fused_glm_value_and_gradient(loss, X, y, coef)

    assert np.allclose(fused_value, loss.value(X, y, coef))
    assert np.allclose(fused_grad, loss.gradient(X, y, coef))


def test_fused_tweedie_uses_loss_power():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(32, 3))
    y = np.exp(rng.normal(scale=0.3, size=32))
    coef = rng.normal(scale=0.2, size=3)
    loss = TweedieLoss(power=1.25)

    fused_value, fused_grad = _fused_glm_value_and_gradient(loss, X, y, coef)

    assert np.allclose(fused_value, loss.value(X, y, coef))
    assert np.allclose(fused_grad, loss.gradient(X, y, coef))


@pytest.mark.parametrize(
    "loss, loss_kwargs",
    [
        ("poisson", {}),
        ("gamma", {}),
        ("inverse_gaussian", {}),
        ("negative_binomial", {"alpha": 0.4}),
        ("tweedie", {"power": 1.3}),
    ],
)
def test_penalized_glm_predict_uses_mean_scale_for_positive_families(loss, loss_kwargs):
    X = np.array([[0.2, -0.1], [1.0, 0.5], [-0.4, 0.7]])
    coef = np.array([0.8, -0.35])
    intercept = 0.15
    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        penalty="l2",
        fit_intercept=True,
        device="cpu",
        loss_kwargs=loss_kwargs,
    )
    model.coef_ = coef
    model.intercept_ = intercept

    pred = model.predict(X)
    expected = np.exp(X @ coef + intercept)

    assert np.all(pred > 0.0)
    assert np.allclose(pred, expected)


def test_general_glm_irls_l2_scaling_matches_penalized_glm():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(80, 5))
    beta = rng.normal(size=5)
    y = X @ beta + 0.3 + rng.normal(scale=0.2, size=80)
    C = 0.7
    alpha = 1.0 / (2.0 * C)

    glm = GeneralizedLinearModel(
        family="gaussian",
        fit_intercept=True,
        C=C,
        solver="irls",
        device="cpu",
        max_iter=100,
        tol=1e-10,
    ).fit(X, y)
    penalized = PenalizedLinearRegression(
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        solver="irls",
        device="cpu",
        max_iter=100,
        tol=1e-10,
    ).fit(X, y)

    assert np.allclose(glm.coef_, penalized.coef_, rtol=1e-6, atol=1e-6)
    assert np.allclose(glm.intercept_, penalized.intercept_, rtol=1e-6, atol=1e-6)


def test_fista_uniform_sample_weight_is_noop_and_nonuniform_raises():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.05)

    coef_unweighted, _ = fista_solver(loss, penalty, X, y, max_iter=80, tol=1e-10)
    coef_uniform, _ = fista_solver(
        loss, penalty, X, y, max_iter=80, tol=1e-10,
        sample_weight=np.full(X.shape[0], 3.0),
    )

    assert np.allclose(coef_uniform, coef_unweighted)
    with pytest.raises(ValueError, match="non-uniform sample_weight"):
        fista_solver(loss, penalty, X, y, sample_weight=np.linspace(0.5, 1.5, X.shape[0]))


@pytest.mark.parametrize("solver", [fista_bb_solver, newton_solver, admm_solver, lbfgs_solver])
def test_non_irls_solvers_reject_nonuniform_sample_weight(solver):
    rng = np.random.default_rng(15)
    X = rng.normal(size=(24, 3))
    y = rng.normal(size=24)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.1)

    with pytest.raises(ValueError, match="non-uniform sample_weight"):
        solver(loss, penalty, X, y, sample_weight=np.linspace(1.0, 2.0, X.shape[0]))


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: SCADPenalty(**kw),
        lambda **kw: MCPPenalty(**kw),
        lambda **kw: GroupSCADPenalty(groups=[[0, 1]], **kw),
        lambda **kw: GroupMCPPenalty(groups=[[0, 1]], **kw),
    ],
)
def test_scad_mcp_penalties_validate_parameters(factory):
    with pytest.raises(ValueError):
        factory(alpha=0.0)
    if "SCAD" in factory().__class__.__name__:
        with pytest.raises(ValueError):
            factory(a=2.0)
    else:
        with pytest.raises(ValueError):
            factory(gamma=1.0)


def _configured_pglm(loss, penalty, device="auto", alpha=1.0):
    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        solver="auto",
        device=device,
    )
    model._loss = model._resolve_loss()
    model._penalty = model._resolve_penalty()
    return model


def test_auto_backend_routing_prefers_torch_for_large_nb_l2(monkeypatch):
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: True),
    )
    model = _configured_pglm("negative_binomial", "l2")
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "torch"


def test_auto_backend_routing_uses_cpu_for_large_sparse_gaussian():
    model = _configured_pglm("squared_error", "l1")
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "numpy"


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_auto_backend_routing_uses_cpu_for_large_gaussian_exact(alpha):
    model = _configured_pglm("squared_error", "l2", alpha=alpha)
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "numpy"


@pytest.mark.parametrize(
    "loss,penalty",
    [
        ("logistic", "l1"),
        ("logistic", "elasticnet"),
        ("gamma", "l2"),
        ("tweedie", "l1"),
        ("tweedie", "elasticnet"),
    ],
)
def test_auto_backend_routing_uses_cpu_for_large_guarded_slow_paths(monkeypatch, loss, penalty):
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: True),
    )
    model = _configured_pglm(loss, penalty)
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "numpy"


def test_auto_backend_routing_does_not_change_explicit_cuda(monkeypatch):
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: True),
    )
    model = _configured_pglm("negative_binomial", "l2", device="cuda")
    X = np.zeros((5000, 500))

    assert model._auto_backend_override("cupy", X) == "cupy"


def test_predict_uses_selected_backend_after_auto_routing():
    model = _configured_pglm("poisson", "l2")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    model._selected_backend_name = "numpy"

    pred = model.predict(np.ones((3, 2)))

    assert isinstance(pred, np.ndarray)
    assert np.allclose(pred, np.exp(np.ones((3, 2)) @ model.coef_ + model.intercept_))


def test_predict_auto_falls_back_to_numpy_when_selected_gpu_backend_unavailable(monkeypatch):
    model = _configured_pglm("poisson", "l2", device="auto")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    model._selected_backend_name = "cupy"
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_cupy_available",
        staticmethod(lambda: False),
    )
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: False),
    )

    pred = model.predict(np.ones((3, 2)))

    assert isinstance(pred, np.ndarray)
    assert np.allclose(pred, np.exp(np.ones((3, 2)) @ model.coef_ + model.intercept_))


def test_predict_explicit_cuda_raises_when_gpu_backend_unavailable(monkeypatch):
    model = _configured_pglm("poisson", "l2", device="cuda")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_cupy_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="device='cuda'"):
        model.predict(np.ones((3, 2)))


def test_predict_explicit_torch_raises_when_gpu_backend_unavailable(monkeypatch):
    model = _configured_pglm("poisson", "l2", device="torch")
    model.coef_ = np.array([0.2, -0.1])
    model.intercept_ = 0.3
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_torch_cuda_available",
        staticmethod(lambda: False),
    )

    with pytest.raises(RuntimeError, match="device='torch'"):
        model.predict(np.ones((3, 2)))
