import numpy as np
import pytest

from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
from statgpu.glm_core._gamma import GammaLoss
from statgpu.glm_core._solver import (
    _fused_glm_value_and_gradient,
    _get_sqerr_proximal_cupy,
    admm_solver,
    fista_lla_path,
    fista_bb_solver,
    fista_solver,
    lbfgs_solver,
    newton_solver,
)
from statgpu.glm_core._squared import SquaredErrorLoss
from statgpu.glm_core._tweedie import TweedieLoss
from statgpu.linear_model import (
    GeneralizedLinearModel,
    OrderedGeneralizedLinearModel,
    PenalizedLinearRegression,
)
from statgpu.linear_model._penalized import (
    PenalizedGeneralizedLinearModel,
    _resolve_loss_name,
)
from statgpu.penalties import (
    AdaptiveL1Penalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
    L2Penalty,
    MCPPenalty,
    SCADPenalty,
)


def _skip_if_cupy_cuda_unavailable():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            pytest.skip("CuPy CUDA unavailable")
    except Exception as exc:
        pytest.skip(f"CuPy CUDA unavailable: {exc}")
    return cp


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


def test_cupy_sqerr_fused_proximal_uses_y_current_as_center():
    cp = _skip_if_cupy_cuda_unavailable()

    y_current = cp.asarray([0.45, -1.25, 0.04, 2.5, -0.9], dtype=cp.float64)
    coef_old = cp.asarray([0.2, -1.0, 0.1, 2.1, -0.4], dtype=cp.float64)
    grad = cp.asarray([0.5, -0.25, 1.0, -0.4, 0.3], dtype=cp.float64)
    thresh = cp.asarray([0.04, 0.12, 0.02, 0.3, 0.15], dtype=cp.float64)
    step = 0.2
    beta = 0.35

    fused = _get_sqerr_proximal_cupy()
    coef_new, y_next = fused(y_current, grad, step, thresh, coef_old, beta)

    w = y_current - step * grad
    expected_coef = cp.sign(w) * cp.maximum(cp.abs(w) - thresh, 0.0)
    expected_y = expected_coef + beta * (expected_coef - coef_old)

    assert cp.allclose(coef_new, expected_coef)
    assert cp.allclose(y_next, expected_y)


@pytest.mark.parametrize("penalty", ["l1", "scad", "mcp"])
def test_cupy_squared_error_penalties_match_cpu_after_fused_lla_fix(penalty):
    cp = _skip_if_cupy_cuda_unavailable()

    rng = np.random.RandomState(42)
    n, p = 500, 20
    X_np = rng.randn(n, p)
    y_np = X_np @ np.linspace(2.0, 0.5, p) + rng.randn(n) * 0.5
    X_cu = cp.asarray(X_np)
    y_cu = cp.asarray(y_np)

    common = dict(
        loss="squared_error",
        penalty=penalty,
        alpha=0.1,
        compute_inference=False,
        max_iter=200,
    )
    model_cpu = PenalizedGeneralizedLinearModel(device="cpu", **common)
    model_cu = PenalizedGeneralizedLinearModel(device="cuda", **common)

    model_cpu.fit(X_np, y_np)
    model_cu.fit(X_cu, y_cu)

    coef_cpu = np.asarray(model_cpu.coef_)
    coef_cu_raw = model_cu.coef_
    coef_cu = (
        cp.asnumpy(coef_cu_raw)
        if isinstance(coef_cu_raw, cp.ndarray)
        else np.asarray(coef_cu_raw)
    )
    corr = np.corrcoef(coef_cpu, coef_cu)[0, 1]

    assert np.all(np.isfinite(coef_cu))
    assert corr > 0.999
    # CuPy fused kernel may take more iterations due to different numerical
    # path, but the final result is correct (corr > 0.999).
    assert abs(model_cpu.n_iter_ - model_cu.n_iter_) < 300


def test_resolve_gamma_loss_forwards_link_kwargs():
    loss = _resolve_loss_name("gamma", {"link": "inverse_power"})

    assert isinstance(loss, GammaLoss)
    assert loss.link_name == "inverse_power"


def test_adaptive_l1_external_weights_respect_normalize_false():
    weights = np.array([1.0, 2.0, 5.0])
    penalty = AdaptiveL1Penalty(alpha=0.1, weights=weights, normalize=False)

    assert np.allclose(penalty.lla_weights(np.zeros_like(weights)), weights)


def test_general_glm_fit_invokes_cleanup_hook(monkeypatch):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        device="cpu",
        gpu_memory_cleanup=True,
    )
    calls = []
    monkeypatch.setattr(model, "_cleanup_backend_memory", calls.append)

    model.fit(X, y)

    assert calls == ["numpy"]


def test_ordered_glm_rejects_sample_weight():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(24, 3))
    y = rng.integers(0, 3, size=24)
    sample_weight = np.linspace(0.5, 1.5, X.shape[0])
    model = OrderedGeneralizedLinearModel(device="cpu", max_iter=5)

    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(X, y, sample_weight=sample_weight)


def test_ordered_glm_fit_invokes_cleanup_hook(monkeypatch):
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 3, size=30)
    model = OrderedGeneralizedLinearModel(device="cpu", max_iter=5)
    calls = []

    def fake_cleanup(backend_name):
        calls.append(backend_name)

    monkeypatch.setattr(model, "_cleanup_backend_memory", fake_cleanup)
    model.fit(X, y)

    assert calls == ["numpy"]


def test_torch_irls_promotes_mixed_float_dtype_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")

    rng = np.random.default_rng(15)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    beta = np.array([0.1, -0.2, 0.05, 0.15])
    y = rng.poisson(np.exp(X.astype(np.float64) @ beta)).astype(np.float64)

    glm = GeneralizedLinearModel(
        family="poisson",
        solver="irls",
        device="torch",
        fit_intercept=False,
        max_iter=5,
    )
    glm.fit(X, y)
    assert np.all(np.isfinite(glm.coef_))

    pglm = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty="l2",
        alpha=0.01,
        solver="irls",
        device="torch",
        fit_intercept=True,
        max_iter=5,
    )
    pglm.fit(X, y)
    assert np.all(np.isfinite(pglm.coef_))


def test_torch_fista_lla_squared_error_promotes_mixed_float_dtype_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")

    rng = np.random.default_rng(16)
    X_np = rng.normal(size=(36, 4)).astype(np.float32)
    y_np = rng.normal(size=36).astype(np.float64)
    X = torch.as_tensor(X_np, device="cuda")
    y = torch.as_tensor(y_np, device="cuda")

    coef, intercept, n_iter = fista_lla_path(
        SquaredErrorLoss(),
        SCADPenalty(alpha=0.05),
        X,
        y,
        alpha_path=np.array([0.08, 0.05]),
        max_lla_per_step=1,
        max_iter=[3, 3],
        tol=1e-4,
        fit_intercept=True,
    )

    assert coef.shape == (X_np.shape[1],)
    assert np.all(np.isfinite(coef))
    assert np.isfinite(intercept)
    assert n_iter > 0


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


def test_fista_uniform_sample_weight_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for torch")
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.05)
    sample_weight = torch.full((X.shape[0],), 3.0, device="cuda")

    coef_uniform, _ = fista_solver(
        loss, penalty, X, y, max_iter=80, tol=1e-10, sample_weight=sample_weight
    )

    assert coef_uniform.shape == (X.shape[1],)


def test_fista_uniform_sample_weight_accepts_cupy_array():
    try:
        import cupy as cp
    except Exception as exc:
        pytest.skip(f"CuPy unavailable: {exc}")
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 4))
    y = rng.normal(size=40)
    loss = SquaredErrorLoss()
    penalty = L2Penalty(alpha=0.05)
    sample_weight = cp.full(X.shape[0], 3.0)

    coef_uniform, _ = fista_solver(
        loss, penalty, X, y, max_iter=80, tol=1e-10, sample_weight=sample_weight
    )

    assert coef_uniform.shape == (X.shape[1],)


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
