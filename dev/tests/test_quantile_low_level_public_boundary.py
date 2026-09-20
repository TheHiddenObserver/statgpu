"""Fresh-review regressions for public low-level Quantile boundaries."""

from __future__ import annotations

import importlib
import inspect
import warnings

import types

import numpy as np
import pytest

from statgpu import solvers
from statgpu.solvers import _fista as _fista_mod
import statgpu.solvers._quantile_solver_guard as _solver_guard
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.losses import QuantileLoss
import statgpu.losses._quantile as _quantile_loss_mod
from statgpu.penalties import GroupSCADPenalty, L1Penalty, L2Penalty, MCPPenalty, SCADPenalty
from statgpu.glm_core._squared import SquaredErrorLoss
from statgpu.backends._array_ops import (
    _max_eigval_power,
    _psd_spectral_upper_bound,
)
import statgpu.backends._array_ops as _array_ops_mod
import statgpu.backends._utils as _backend_utils
from statgpu.solvers._fista import _weighted_gram_lipschitz
import statgpu.losses._quantile_irls_validation_contract as _irls_contract
import statgpu.solvers._quantile_proximal_public_contract as _prox_contract
import statgpu.solvers._quantile_solver_guard as _solver_guard
from statgpu.solvers import _proximal_irls_quantile as _prox_kernel


def _data(seed=16701):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 3))
    y = 0.2 + X @ np.array([0.5, -0.25, 0.1])
    y = y + rng.laplace(scale=0.08, size=X.shape[0])
    return X, y


def test_xp_asarray_cupy_targets_reference_device(monkeypatch):
    class _FakeDevice:
        id = 3

    class _FakeRef:
        device = _FakeDevice()

    fake_cupy = types.SimpleNamespace(__name__="cupy")

    captured = {}

    def fake_cupy_asarray_on_device(value, target_device, dtype=None):
        captured["value"] = value
        captured["target_device"] = target_device
        captured["dtype"] = dtype
        return "aligned"

    monkeypatch.setattr(_array_ops_mod, "_xp", lambda ref: fake_cupy)
    monkeypatch.setattr(
        _backend_utils,
        "_cupy_asarray_on_device",
        fake_cupy_asarray_on_device,
    )

    source = np.asarray([1.0, 2.0], dtype=np.float64)
    result = _array_ops_mod._xp_asarray(source, np.float64, _FakeRef())

    assert result == "aligned"
    assert captured["value"] is source
    assert captured["target_device"] == 3
    assert captured["dtype"] is np.float64


def test_to_backend_cupy_targets_reference_device(monkeypatch):
    class _FakeDevice:
        id = 5

    class _FakeRef:
        __module__ = "cupy._core.core"
        device = _FakeDevice()
        dtype = np.dtype("float64")

    captured = {}

    def fake_cupy_asarray_on_device(value, target_device, dtype=None):
        captured["value"] = value
        captured["target_device"] = target_device
        captured["dtype"] = dtype
        return "aligned"

    monkeypatch.setattr(_array_ops_mod, "_resolve_backend", lambda *args: "cupy")
    monkeypatch.setattr(
        _backend_utils,
        "_cupy_asarray_on_device",
        fake_cupy_asarray_on_device,
    )

    fake_cupy = type(
        "_FakeCupyModule",
        (),
        {"float64": np.float64, "asarray": staticmethod(lambda *a, **k: "raw")},
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)

    source = np.asarray([1.0, 2.0], dtype=np.float64)
    result = _array_ops_mod._to_backend(
        source,
        backend="auto",
        ref_tensor=_FakeRef(),
        dtype=np.float64,
    )

    assert result == "aligned"
    assert captured["value"] is source
    assert captured["target_device"] == 5
    assert captured["dtype"] is np.float64


def test_solver_zeros_cupy_targets_reference_device(monkeypatch):
    captured = {"active_device": None, "zeros_device": None}

    class _DeviceContext:
        def __init__(self, device_id):
            self.device_id = int(device_id)
            self.previous = None

        def __enter__(self):
            self.previous = captured["active_device"]
            captured["active_device"] = self.device_id
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["active_device"] = self.previous

    def fake_zeros(n, dtype=None):
        captured["zeros_device"] = captured["active_device"]
        return np.zeros(n, dtype=dtype)

    fake_cupy = types.SimpleNamespace(
        float64=np.float64,
        zeros=fake_zeros,
        cuda=types.SimpleNamespace(Device=_DeviceContext),
    )

    class _FakeDevice:
        id = 7

    class _FakeRef:
        __module__ = "cupy._core.core"
        device = _FakeDevice()
        dtype = np.dtype("float64")

    monkeypatch.setattr(_array_ops_mod, "_resolve_backend", lambda *args: "cupy")
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)

    result = _array_ops_mod._zeros(3, "cupy", ref_tensor=_FakeRef())

    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float64))
    assert captured["zeros_device"] == 7


def test_cupy_solver_primitives_follow_operand_device(monkeypatch):
    state = {
        "current": None,
        "entered": [],
        "seed_device": None,
        "sync_targets": [],
    }

    class FakeDeviceContext:
        def __init__(self, device_id):
            self.device_id = int(device_id)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device_id
            state["entered"].append(self.device_id)
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    def fake_arange(*args, dtype=None):
        state["seed_device"] = state["current"]
        return np.arange(*args, dtype=dtype)

    fake_cupy = types.SimpleNamespace(
        __name__="cupy",
        float64=np.float64,
        abs=np.abs,
        arange=fake_arange,
        asnumpy=np.asarray,
        dot=np.dot,
        maximum=np.maximum,
        ones_like=np.ones_like,
        sqrt=np.sqrt,
        stack=np.stack,
        sum=np.sum,
        where=np.where,
        cuda=types.SimpleNamespace(Device=FakeDeviceContext),
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", fake_cupy)

    grad = np.asarray([2.0, -1.0], dtype=np.float64)
    coef = np.asarray([0.5, -0.25], dtype=np.float64)
    clipped = _array_ops_mod._clip_grad_on_device(grad, coef, "cupy")
    assert np.all(np.isfinite(clipped))

    class FakeDevice:
        id = 9

    class FakeScalar:
        __module__ = "cupy._core.core"

        def __init__(self, value):
            self.value = float(value)
            self.device = FakeDevice()

    def fake_align(value, target_device, dtype=None):
        state["sync_targets"].append(int(target_device))
        payload = value.value if isinstance(value, FakeScalar) else value
        return np.asarray(payload, dtype=dtype)

    monkeypatch.setattr(
        _backend_utils,
        "_cupy_asarray_on_device",
        fake_align,
    )
    monkeypatch.setattr(_array_ops_mod, "_resolve_backend", lambda *a: "cupy")
    synced = _array_ops_mod._sync_scalars(
        FakeScalar(1.25),
        FakeScalar(-0.5),
        backend="cupy",
    )
    assert synced == pytest.approx((1.25, -0.5))
    assert state["sync_targets"] == [9, 9]

    class FakeMatrix:
        __module__ = "cupy._core.core"

        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)
            self.shape = self.values.shape
            self.dtype = self.values.dtype
            self.device = FakeDevice()

        def __matmul__(self, other):
            return self.values @ other

    monkeypatch.setattr(_array_ops_mod, "_xp", lambda value: fake_cupy)
    estimate = _array_ops_mod._max_eigval_power(
        FakeMatrix([[2.0, 0.0], [0.0, 1.0]]),
        n_iter=3,
    )
    assert np.isfinite(estimate)
    assert state["seed_device"] == 9


def test_safe_psd_spectral_bound_handles_empty_gram():
    empty = np.empty((0, 0), dtype=np.float64)
    assert _psd_spectral_upper_bound(empty) == 0.0


def test_safe_psd_spectral_bound_closes_power_seed_orthogonality_gap():
    gram = np.array(
        [[8.2, -3.6], [-3.6, 2.8]],
        dtype=np.float64,
    )
    exact = float(np.linalg.eigvalsh(gram)[-1])
    approximate = float(_max_eigval_power(gram))
    safe = float(_psd_spectral_upper_bound(gram))

    assert exact == pytest.approx(10.0, rel=0.0, abs=1e-12)
    assert approximate == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert safe >= exact
    assert _weighted_gram_lipschitz(gram) == pytest.approx(safe)


def test_safe_psd_spectral_bound_preserves_torch_backend():
    torch = pytest.importorskip("torch")
    gram = torch.tensor(
        [[8.2, -3.6], [-3.6, 2.8]],
        dtype=torch.float64,
    )
    safe = _psd_spectral_upper_bound(gram)
    exact = float(torch.linalg.eigvalsh(gram)[-1])

    assert safe >= exact
    assert _weighted_gram_lipschitz(gram) == pytest.approx(safe)


def test_quantile_lipschitz_uses_safe_psd_upper_bound():
    gram = np.array(
        [[8.2, -3.6], [-3.6, 2.8]],
        dtype=np.float64,
    )
    eigvals, eigvecs = np.linalg.eigh(gram)
    sqrt_gram = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    X = np.sqrt(2.0) * sqrt_gram
    loss = QuantileLoss(quantile=0.5)

    observed = float(loss.lipschitz(X, np.zeros(2)))
    exact_scaled = 0.5 * float(np.linalg.eigvalsh(gram)[-1])

    assert observed >= exact_scaled


def test_public_quantile_fista_integer_design_preserves_fractional_inputs():
    X_int = np.asarray(
        [[1, 0, 2], [0, 1, -1], [2, 1, 0], [-1, 2, 1]],
        dtype=np.int64,
    )
    X_float = X_int.astype(np.float64)
    y = np.asarray([0.25, -0.4, 1.15, 0.6], dtype=np.float64)
    weights = np.asarray([0.25, 0.75, 1.25, 1.75], dtype=np.float64)
    loss = QuantileLoss(quantile=0.35)
    penalty = L2Penalty(alpha=0.0)

    def solve(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return solvers.fista_solver(
                loss,
                penalty,
                X,
                y,
                max_iter=12,
                tol=1e-10,
                sample_weight=weights,
            )

    coef_int, n_iter_int = solve(X_int)
    coef_float, n_iter_float = solve(X_float)
    np.testing.assert_allclose(coef_int, coef_float, rtol=0.0, atol=1e-14)
    assert n_iter_int == n_iter_float


def test_public_quantile_fista_accepts_numpy_response_with_torch_design():
    torch = pytest.importorskip("torch")
    X = torch.eye(3, dtype=torch.float64)
    y = np.asarray([0.4, -0.2, 0.7], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        coef, n_iter = solvers.fista_solver(
            QuantileLoss(quantile=0.35),
            L2Penalty(alpha=0.0),
            X,
            y,
            max_iter=3,
            tol=1e-8,
        )

    assert torch.is_tensor(coef)
    assert coef.device == X.device
    assert bool(torch.all(torch.isfinite(coef)).item())
    assert n_iter >= 1


def test_fista_rejects_unverified_trial_after_backtracking_exhaustion():
    class AlwaysRejectingLoss:
        name = "always_reject"
        _is_quadratic = False
        _lipschitz_static = True

        def preprocess(self, X, y):
            return X, y

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            return 1.0

        def fused_value_and_gradient(
            self, X, y, coef, sample_weight=None
        ):
            # At the current point the surrogate starts from zero objective
            # with a nonzero gradient.
            return np.asarray(0.0), np.ones_like(coef)

        def value(self, X, y, coef, sample_weight=None):
            # Every proposed trial is deliberately above the quadratic bound,
            # so none of the 20 backtracking attempts may be accepted.
            return np.asarray(1.0)

    X = np.eye(2, dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    penalty = L2Penalty(alpha=0.0)

    with pytest.warns(
        ConvergenceWarning,
        match="line search failed to find an acceptable proximal step",
    ):
        coef, n_iter = _fista_mod.fista_solver(
            AlwaysRejectingLoss(),
            penalty,
            X,
            y,
            max_iter=5,
            tol=1e-12,
        )

    np.testing.assert_array_equal(coef, np.zeros(2, dtype=np.float64))
    assert n_iter == 1


def test_quantile_value_stays_on_torch_backend_and_matches_numpy():
    torch = pytest.importorskip("torch")
    loss = QuantileLoss(quantile=0.3)
    X_np = np.array(
        [[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]],
        dtype=np.float64,
    )
    y_np = np.array([0.2, -0.1, 0.5], dtype=np.float64)
    coef_np = np.array([0.1, -0.2], dtype=np.float64)
    weights_np = np.array([0.5, 1.0, 1.5], dtype=np.float64)

    expected = loss.value(
        X_np,
        y_np,
        coef_np,
        sample_weight=weights_np,
    )
    observed = loss.value(
        torch.as_tensor(X_np),
        torch.as_tensor(y_np),
        torch.as_tensor(coef_np),
        sample_weight=torch.as_tensor(weights_np),
    )

    assert isinstance(expected, float)
    assert torch.is_tensor(observed)
    assert observed.device.type == "cpu"
    assert float(observed.item()) == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_quantile_weighted_gradient_stays_on_torch_backend_and_matches_numpy():
    torch = pytest.importorskip("torch")
    loss = QuantileLoss(quantile=0.3)
    X_np = np.array(
        [[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]],
        dtype=np.float64,
    )
    y_np = np.array([0.2, -0.1, 0.5], dtype=np.float64)
    coef_np = np.array([0.1, -0.2], dtype=np.float64)
    weights_np = np.array([0.5, 1.0, 1.5], dtype=np.float64)

    expected = loss.gradient(
        X_np,
        y_np,
        coef_np,
        sample_weight=weights_np,
    )
    observed = loss.gradient(
        torch.as_tensor(X_np),
        torch.as_tensor(y_np),
        torch.as_tensor(coef_np),
        sample_weight=torch.as_tensor(weights_np),
    )

    assert torch.is_tensor(observed)
    assert observed.device.type == "cpu"
    np.testing.assert_allclose(
        observed.detach().numpy(),
        expected,
        rtol=0.0,
        atol=1e-15,
    )


def test_quantile_fused_value_stays_on_torch_backend():
    torch = pytest.importorskip("torch")
    loss = QuantileLoss(quantile=0.3)
    X = torch.tensor(
        [[1.0, -0.5], [0.2, 0.7], [-0.3, 0.4]],
        dtype=torch.float64,
    )
    y = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64)
    coef = torch.tensor([0.1, -0.2], dtype=torch.float64)
    weights = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)

    value, grad = loss.fused_value_and_gradient(
        X,
        y,
        coef,
        sample_weight=weights,
    )

    assert torch.is_tensor(value)
    assert value.ndim == 0
    assert value.device == X.device
    assert value.dtype == X.dtype
    assert torch.is_tensor(grad)
    assert grad.device == X.device
    assert grad.dtype == X.dtype

    eta = X @ coef
    residual = y - eta
    per_sample = torch.where(
        residual >= 0.0,
        0.3 * residual,
        (0.3 - 1.0) * residual,
    )
    expected_value = torch.sum(weights * per_sample) / torch.sum(weights)
    expected_grad = X.T @ (
        weights
        * torch.where(
            residual < 0.0,
            torch.tensor(0.7, dtype=X.dtype),
            torch.tensor(-0.3, dtype=X.dtype),
        )
    ) / torch.sum(weights)

    torch.testing.assert_close(value, expected_value)
    torch.testing.assert_close(grad, expected_grad)


def test_quantile_response_validation_preserves_torch_backend():
    torch = pytest.importorskip("torch")
    loss = QuantileLoss(quantile=0.3)
    response = torch.tensor(
        [[-0.2], [0.1], [0.4]],
        dtype=torch.float64,
    )

    validated = loss.validate_response(response)

    assert torch.is_tensor(validated)
    assert validated.device == response.device
    assert validated.dtype == response.dtype
    assert tuple(validated.shape) == (3,)
    torch.testing.assert_close(validated, response.reshape(-1))


def test_async_quantile_fista_cv_mode_can_converge_on_torch_cpu():
    torch = pytest.importorskip("torch")

    X = torch.zeros((8, 1), dtype=torch.float64)
    y = torch.zeros(8, dtype=torch.float64)
    loss = QuantileLoss(quantile=0.5)
    penalty = L1Penalty(alpha=0.1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        coef, n_iter = solvers.fista_solver(
            loss,
            penalty,
            X,
            y,
            max_iter=5,
            tol=1e-10,
            cv_mode=True,
        )

    assert n_iter == 2
    assert torch.is_tensor(coef)
    torch.testing.assert_close(coef, torch.zeros_like(coef))
    assert not any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    )


def test_weighted_fista_uses_spectral_gram_scale_not_max_diagonal():
    gram = np.asarray(
        [[1.0, 0.99], [0.99, 1.0]],
        dtype=np.float64,
    )

    observed = _fista_mod._weighted_gram_lipschitz(gram)
    max_diagonal = float(np.max(np.diag(gram)))
    exact = float(np.linalg.eigvalsh(gram)[-1])

    assert observed == pytest.approx(exact, rel=2e-2, abs=1e-8)
    assert observed > max_diagonal * 1.9


def test_quantile_lipschitz_reflects_in_place_design_mutation():
    X = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=np.float64,
    )
    coef = np.zeros(X.shape[1], dtype=np.float64)
    loss = QuantileLoss(quantile=0.3)

    before = float(loss.lipschitz(X, coef))
    X *= 5.0
    after = float(loss.lipschitz(X, coef))

    assert after > before * 10.0


def test_quantile_lipschitz_is_declared_static_within_one_fista_solve():
    assert QuantileLoss._lipschitz_static is True


def _invalid_weights(n):
    negative = np.ones(n, dtype=np.float64)
    negative[2] = -0.25
    nonfinite = np.ones(n, dtype=np.float64)
    nonfinite[3] = np.nan
    return [
        pytest.param(np.ones(n - 1), id="wrong-length"),
        pytest.param(negative, id="negative"),
        pytest.param(nonfinite, id="nonfinite"),
        pytest.param(np.zeros(n), id="zero-sum"),
    ]


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_direct_quantile_irls_rejects_invalid_xy_shapes_before_numerics(
    X_transform, y_transform, message
):
    X, y = _data(seed=16713)
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match=message):
        loss.irls(X_transform(X), y_transform(y), max_iter=3)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_direct_quantile_irls_rejects_nonfinite_xy_before_numerics(target, message):
    X, y = _data(seed=16718)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(quantile=0.3)
    with pytest.raises(ValueError, match=message):
        loss.irls(X_bad, y_bad, max_iter=3)


def test_public_quantile_lbfgs_normalizes_python_array_like_inputs(monkeypatch):
    X, y = _data(seed=16731)
    captured = {}

    def fake_lbfgs(loss, penalty, X_arg, y_arg, *args, **kwargs):
        captured["X"] = X_arg
        captured["y"] = y_arg
        return np.zeros(X_arg.shape[1], dtype=np.float64), 1

    monkeypatch.setattr(_solver_guard, "_lbfgs_solver", fake_lbfgs)

    coef, n_iter = solvers.lbfgs_solver(
        QuantileLoss(quantile=0.3),
        None,
        X.tolist(),
        y.tolist(),
        max_iter=3,
    )

    assert isinstance(captured["X"], np.ndarray)
    assert isinstance(captured["y"], np.ndarray)
    assert captured["X"].dtype == np.float64
    assert captured["y"].dtype == np.float64
    np.testing.assert_array_equal(coef, np.zeros(X.shape[1]))
    assert n_iter == 1


def test_public_quantile_lbfgs_aligns_numpy_response_to_torch_design(monkeypatch):
    torch = pytest.importorskip("torch")
    X = torch.eye(3, dtype=torch.float64)
    y = np.asarray([0.4, -0.2, 0.7], dtype=np.float64)
    captured = {}

    def fake_lbfgs(loss, penalty, X_arg, y_arg, *args, **kwargs):
        captured["X"] = X_arg
        captured["y"] = y_arg
        return torch.zeros(X_arg.shape[1], dtype=X_arg.dtype, device=X_arg.device), 1

    monkeypatch.setattr(_solver_guard, "_lbfgs_solver", fake_lbfgs)

    coef, n_iter = solvers.lbfgs_solver(
        QuantileLoss(quantile=0.35),
        None,
        X,
        y,
        max_iter=3,
    )

    assert torch.is_tensor(captured["X"])
    assert torch.is_tensor(captured["y"])
    assert captured["X"].device == X.device
    assert captured["y"].device == X.device
    assert captured["y"].dtype == X.dtype
    assert torch.is_tensor(coef)
    assert n_iter == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": True}, "max_iter must be a positive integer"),
        ({"max_iter": 0}, "max_iter must be a positive integer"),
        ({"tol": True}, "tol must be a finite positive number"),
        ({"tol": "1e-4"}, "tol must be a finite positive number"),
        ({"tol": np.nan}, "tol must be a finite positive number"),
        ({"cv_mode": "False"}, "cv_mode must be boolean"),
        ({"lipschitz_L": True}, "lipschitz_L must be None or a finite positive number"),
        ({"lipschitz_L": 0.0}, "lipschitz_L must be None or a finite positive number"),
        ({"lipschitz_L": np.nan}, "lipschitz_L must be None or a finite positive number"),
    ],
)
def test_public_quantile_fista_rejects_invalid_controls_before_loss_work(
    monkeypatch, kwargs, message
):
    X, y = _data(seed=16732)
    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kw):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match=message):
        solvers.fista_solver(
            loss,
            L2Penalty(alpha=0.04),
            X,
            y,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": True}, "max_iter must be a positive integer"),
        ({"max_iter": 0}, "max_iter must be a positive integer"),
        ({"tol": True}, "tol must be a finite positive number"),
        ({"tol": "1e-4"}, "tol must be a finite positive number"),
        ({"tol": np.nan}, "tol must be a finite positive number"),
        ({"history_size": True}, "history_size must be a positive integer"),
        ({"history_size": 0}, "history_size must be a positive integer"),
        ({"history_size": -2}, "history_size must be a positive integer"),
    ],
)
def test_public_quantile_lbfgs_rejects_invalid_stopping_controls_before_loss_work(
    monkeypatch, kwargs, message
):
    X, y = _data(seed=16734)
    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kw):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match=message):
        solvers.lbfgs_solver(
            loss,
            None,
            X,
            y,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("solver_name", "penalty"),
    [
        ("fista_solver", L2Penalty(alpha=0.04)),
        ("lbfgs_solver", None),
    ],
)
@pytest.mark.parametrize(
    ("init_coef", "message"),
    [
        (
            np.zeros((3, 1), dtype=np.float64),
            "init_coef must be one-dimensional",
        ),
        (
            np.zeros(2, dtype=np.float64),
            "init_coef must have length n_features",
        ),
        (
            np.asarray([0.0, np.nan, 0.0], dtype=np.float64),
            "init_coef must contain finite values",
        ),
        (
            np.asarray([True, False, True], dtype=bool),
            "init_coef must contain real numeric values",
        ),
        (
            np.asarray([1.0j, 0.0j, 0.0j], dtype=np.complex128),
            "init_coef must contain real numeric values",
        ),
    ],
)
def test_public_quantile_first_order_rejects_invalid_warm_start_before_loss_work(
    monkeypatch, solver_name, penalty, init_coef, message
):
    X, y = _data(seed=16733)
    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match=message):
        getattr(solvers, solver_name)(
            loss,
            penalty,
            X,
            y,
            init_coef=init_coef,
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("solver_name", "penalty"),
    [
        ("fista_solver", L2Penalty(alpha=0.04)),
        ("lbfgs_solver", None),
    ],
)
@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_public_quantile_fista_lbfgs_reject_nonfinite_xy_before_loss_work(
    monkeypatch, solver_name, penalty, target, message
):
    X, y = _data(seed=16719)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match=message):
        getattr(solvers, solver_name)(
            loss,
            penalty,
            X_bad,
            y_bad,
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("solver_name", "penalty"),
    [
        ("fista_solver", L2Penalty(alpha=0.04)),
        ("lbfgs_solver", None),
    ],
)
@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_public_quantile_fista_lbfgs_reject_invalid_xy_before_loss_work(
    monkeypatch, solver_name, penalty, X_transform, y_transform, message
):
    X, y = _data(seed=16715)
    loss = QuantileLoss(quantile=0.3)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    solver_fn = getattr(solvers, solver_name)

    with pytest.raises(ValueError, match=message):
        solver_fn(
            loss,
            penalty,
            X_transform(X),
            y_transform(y),
            max_iter=3,
        )


@pytest.mark.parametrize(
    "solver_name",
    ["newton_solver", "proximal_newton_solver", "lbfgs_b_solver"],
)
def test_public_smooth_second_order_solvers_reject_quantile_before_loss_work(
    monkeypatch, solver_name
):
    X, y = _data(seed=16717)
    loss = QuantileLoss(quantile=0.3)
    penalty = L2Penalty(alpha=0.04)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported Quantile solver must fail before loss work")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    with pytest.raises(ValueError, match="does not support Quantile loss"):
        getattr(solvers, solver_name)(loss, penalty, X, y, max_iter=3)


def test_public_quantile_cd_solver_is_fail_closed_compatibility_symbol(monkeypatch):
    X, y = _data(seed=16716)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.04)

    def forbidden(*args, **kwargs):
        raise AssertionError("retired Quantile CD kernel must not execute")

    import statgpu.solvers._quantile_solver_guard as guard_mod

    monkeypatch.setattr(guard_mod, "_quantile_cd_solver", forbidden)
    with pytest.raises(NotImplementedError, match="compatibility symbol"):
        solvers.quantile_cd_solver(
            loss,
            penalty,
            X,
            y,
            sample_weight=np.linspace(0.5, 1.5, X.shape[0]),
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_public_proximal_quantile_rejects_nonfinite_xy_before_path_work(
    monkeypatch, target, message
):
    X, y = _data(seed=16720)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )
    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X_bad,
            y_bad,
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
        )


def test_direct_quantile_irls_uses_reference_aware_ridge_allocations(
    monkeypatch,
):
    X, y = _data(seed=16720)
    loss = QuantileLoss(quantile=0.3)
    penalty = L2Penalty(alpha=0.1)
    calls = []

    original_eye = _quantile_loss_mod.xp_eye
    original_ones = _quantile_loss_mod.xp_ones

    def recording_eye(n, dtype, xp, ref_arr=None):
        calls.append(("eye", ref_arr))
        return original_eye(n, dtype, xp, ref_arr=ref_arr)

    def recording_ones(n, dtype, xp, ref_arr=None):
        calls.append(("ones", ref_arr))
        return original_ones(n, dtype, xp, ref_arr=ref_arr)

    monkeypatch.setattr(_quantile_loss_mod, "xp_eye", recording_eye)
    monkeypatch.setattr(_quantile_loss_mod, "xp_ones", recording_ones)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        loss.irls(
            X,
            y,
            penalty=penalty,
            max_iter=1,
            fit_intercept=True,
        )

    assert [name for name, _ in calls] == ["eye", "ones"]
    assert all(ref is not None and ref.shape == X.shape for _, ref in calls)


def test_direct_quantile_irls_budget_exhaustion_is_observable():
    from statgpu.solvers._convergence import ConvergenceWarning

    X, y = _data(seed=16721)
    loss = QuantileLoss(quantile=0.3)

    with pytest.warns(ConvergenceWarning, match="did not converge within 1 iterations"):
        coef, n_iter = loss.irls(
            X,
            y,
            max_iter=1,
            tol=1e-16,
        )

    assert n_iter == 1
    assert np.all(np.isfinite(np.asarray(coef)))


@pytest.mark.parametrize(
    ("init_coef", "message"),
    [
        (np.zeros((2, 1), dtype=np.float64), "init_coef must be one-dimensional"),
        (np.zeros(1, dtype=np.float64), "init_coef must have length n_features"),
        (np.asarray([0.0, np.nan, 0.0]), "init_coef must contain finite values"),
        (
            np.asarray([True, False, True], dtype=bool),
            "init_coef must contain real numeric values",
        ),
        (
            np.asarray([0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            "init_coef must contain real numeric values",
        ),
    ],
)
def test_direct_quantile_irls_rejects_invalid_init_coef(init_coef, message):
    X, y = _data(seed=16722)
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match=message):
        loss.irls(
            X,
            y,
            init_coef=init_coef,
            max_iter=3,
        )


def test_direct_quantile_irls_never_penalizes_intercept_only_coordinate():
    loss = QuantileLoss(quantile=0.4)
    X = np.ones((7, 1), dtype=np.float64)
    y = np.asarray([-1.0, -0.4, 0.1, 0.3, 0.8, 1.2, 2.0], dtype=np.float64)
    init = np.asarray([0.25], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        unpenalized, _ = loss.irls(
            X,
            y,
            penalty=None,
            max_iter=1,
            tol=1e-20,
            init_coef=init,
            fit_intercept=True,
        )
        penalized, _ = loss.irls(
            X,
            y,
            penalty=L2Penalty(alpha=100.0),
            max_iter=1,
            tol=1e-20,
            init_coef=init,
            fit_intercept=True,
        )

    np.testing.assert_array_equal(penalized, unpenalized)


@pytest.mark.parametrize("alpha", [np.nan, -0.1, True, "0.1"])
def test_direct_quantile_irls_rejects_invalid_mutated_l2_alpha(alpha):
    X, y = _data(seed=16723)
    loss = QuantileLoss(quantile=0.3)
    penalty = L2Penalty(alpha=0.1)
    penalty.alpha = alpha

    with pytest.raises(ValueError, match="L2 penalty alpha must be"):
        loss.irls(
            X,
            y,
            penalty=penalty,
            max_iter=3,
        )


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_direct_quantile_irls_rejects_invalid_weights_before_numerics(sample_weight):
    X, y = _data()
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match="sample_weight"):
        loss.irls(X, y, sample_weight=sample_weight, max_iter=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter must be a positive integer"),
        ({"max_iter": True}, "max_iter must be a positive integer"),
        ({"tol": 0.0}, "tol must be a finite positive number"),
        ({"tol": True}, "tol must be a finite positive number"),
        ({"tol": "1e-6"}, "tol must be a finite positive number"),
        ({"eps": 0.0}, "eps must be a finite positive number"),
        ({"fit_intercept": "False"}, "fit_intercept must be boolean"),
    ],
)
def test_direct_quantile_irls_rejects_invalid_stopping_controls(kwargs, message):
    X, y = _data(seed=16705)
    loss = QuantileLoss(quantile=0.3)

    with pytest.raises(ValueError, match=message):
        loss.irls(X, y, **kwargs)


def test_direct_quantile_irls_validation_preserves_array_like_design_input():
    X, y = _data(seed=16704)
    loss = QuantileLoss(quantile=0.3)

    coef, n_iter = loss.irls(
        X.tolist(),
        y.tolist(),
        sample_weight=np.ones(X.shape[0]),
        max_iter=2,
    )

    assert np.asarray(coef).shape == (X.shape[1],)
    assert np.all(np.isfinite(np.asarray(coef)))
    assert 1 <= n_iter <= 2


@pytest.mark.parametrize(
    ("loss_factory", "penalty_factory", "message"),
    [
        (
            lambda: SquaredErrorLoss(),
            lambda: SCADPenalty(alpha=0.05),
            "requires QuantileLoss",
        ),
        (
            lambda: QuantileLoss(quantile=0.3),
            lambda: L2Penalty(alpha=0.05),
            "requires scalar SCAD or MCP",
        ),
        (
            lambda: QuantileLoss(quantile=0.3),
            lambda: GroupSCADPenalty(
                alpha=0.05,
                a=3.7,
                groups=[[0, 1], [2]],
            ),
            "requires scalar SCAD or MCP",
        ),
    ],
)
def test_public_proximal_quantile_solver_rejects_wrong_objective_before_path_work(
    monkeypatch, loss_factory, penalty_factory, message
):
    X, y = _data(seed=16711)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss_factory(),
            penalty_factory(),
            X,
            y,
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_public_proximal_quantile_rejects_invalid_xy_shapes_before_path_work(
    monkeypatch, X_transform, y_transform, message
):
    X, y = _data(seed=16714)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X_transform(X),
            y_transform(y),
            alpha_path=np.array([0.08, 0.05]),
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("factory", "attr", "bad_value", "message"),
    [
        (
            lambda: SCADPenalty(alpha=0.05, a=3.7),
            "a",
            np.nan,
            "SCAD penalty a must be",
        ),
        (
            lambda: SCADPenalty(alpha=0.05, a=3.7),
            "a",
            2.0,
            "SCAD penalty a must be",
        ),
        (
            lambda: MCPPenalty(alpha=0.05, gamma=3.0),
            "gamma",
            np.nan,
            "MCP penalty gamma must be",
        ),
        (
            lambda: MCPPenalty(alpha=0.05, gamma=3.0),
            "gamma",
            1.0,
            "MCP penalty gamma must be",
        ),
    ],
)
def test_public_proximal_quantile_rejects_mutated_penalty_shape_before_path_work(
    monkeypatch, factory, attr, bad_value, message
):
    X, y = _data(seed=16724)
    loss = QuantileLoss(quantile=0.3)
    penalty = factory()
    setattr(penalty, attr, bad_value)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("invalid penalty shape must fail before path work")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )
    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.08, 0.05]),
            max_iter=3,
        )


@pytest.mark.parametrize("sample_weight", _invalid_weights(24))
def test_public_proximal_quantile_solver_rejects_invalid_weights(sample_weight):
    X, y = _data(seed=16702)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    for solver_fn in (
        solvers.proximal_irls_quantile_solver,
        _prox_kernel.proximal_irls_quantile_solver,
    ):
        with pytest.raises(ValueError, match="sample_weight"):
            solver_fn(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.08, 0.05]),
                max_iter=3,
                sample_weight=sample_weight,
            )


@pytest.mark.parametrize(
    ("alpha_path", "message"),
    [
        ([], "alpha_path must be a non-empty"),
        (np.array([[0.08, 0.05]]), "one-dimensional"),
        ([0.08, 0.0], "finite positive"),
        ([0.08, np.nan], "finite positive"),
        ([0.08, True], "finite positive"),
        ([0.05, 0.08], "non-increasing"),
    ],
)
def test_public_proximal_quantile_rejects_invalid_alpha_path_before_path_work(
    monkeypatch, alpha_path, message
):
    X, y = _data(seed=16712)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    def forbidden_path(*args, **kwargs):
        raise AssertionError("continuation/backend work must not start")

    monkeypatch.setattr(
        _prox_contract,
        "resolve_auto_quantile_continuation_path",
        forbidden_path,
    )

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=alpha_path,
            max_iter=3,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 0}, "max_iter must be a positive integer or sequence"),
        ({"max_iter": True}, "max_iter must be a positive integer or sequence"),
        ({"max_iter": [2, 0]}, "max_iter sequence must contain only positive integers"),
        ({"max_iter": [2]}, "max_iter sequence must have one positive integer per alpha_path step"),
        ({"max_lla_per_step": 0}, "max_lla_per_step must be a positive integer"),
        ({"max_lla_per_step": True}, "max_lla_per_step must be a positive integer"),
        ({"tol": 0.0}, "tol must be a finite positive number"),
        ({"tol": "1e-6"}, "tol must be a finite positive number"),
        ({"lla_tol": False}, "lla_tol must be a finite positive number"),
        ({"fit_intercept": "False"}, "fit_intercept must be boolean"),
    ],
)
def test_public_proximal_quantile_rejects_invalid_stopping_controls(kwargs, message):
    X, y = _data(seed=16706)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    with pytest.raises(ValueError, match=message):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.array([0.08, 0.05]),
            **kwargs,
        )


def test_public_proximal_quantile_flat_target_remains_valid_when_lla_weights_stay_zero(
    monkeypatch,
):
    X, y = _data(seed=16709)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    monkeypatch.setattr(
        _prox_kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.zeros(
            p, dtype=coef.dtype
        ),
    )

    def converged_flat_irls(
        X_arg,
        y_arg,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        point = np.full(X_arg.shape[1], 2.0, dtype=np.float64)
        return point, 1

    monkeypatch.setattr(loss, "irls", converged_flat_irls)

    token = _prox_kernel._STRICT_CV_TARGET.set(True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coef, intercept, n_iter = solvers.proximal_irls_quantile_solver(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.05]),
                max_lla_per_step=1,
                max_iter=2,
                tol=1e-12,
                lla_tol=1e-12,
                fit_intercept=False,
            )
    finally:
        _prox_kernel._STRICT_CV_TARGET.reset(token)

    assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert n_iter == 1
    np.testing.assert_array_equal(coef, np.full(X.shape[1], 2.0))
    assert intercept == 0.0


def test_public_proximal_quantile_flat_boundary_probe_does_not_leak_internal_warning(
    monkeypatch,
):
    X, y = _data(seed=16710)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)
    from statgpu.solvers import _proximal_irls_quantile as kernel

    monkeypatch.setattr(
        kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.zeros(
            p, dtype=coef.dtype
        ),
    )
    calls = {"irls": 0}

    def exhausted_flat_irls(
        X_arg,
        y_arg,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        calls["irls"] += 1
        point = np.asarray(init_coef, dtype=np.float64)
        if calls["irls"] == 1:
            return point.copy(), int(max_iter)
        warnings.warn(
            "diagnostic IRLS exhausted its one-step budget",
            ConvergenceWarning,
            stacklevel=2,
        )
        return point + 0.1, int(max_iter)

    monkeypatch.setattr(loss, "irls", exhausted_flat_irls)

    token = kernel._STRICT_CV_TARGET.set(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with pytest.raises(
                FloatingPointError,
                match="target reached max_iter=2 before IRLS convergence",
            ):
                solvers.proximal_irls_quantile_solver(
                    loss,
                    penalty,
                    X,
                    y,
                    alpha_path=np.array([0.05]),
                    max_lla_per_step=1,
                    max_iter=2,
                    tol=1e-12,
                    lla_tol=1e-12,
                    fit_intercept=False,
                )
    finally:
        kernel._STRICT_CV_TARGET.reset(token)

    assert calls["irls"] == 2


def test_public_proximal_quantile_active_target_exhaustion_warns(monkeypatch):
    X, y = _data(seed=16707)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)
    from statgpu.solvers import _proximal_irls_quantile as kernel

    monkeypatch.setattr(
        kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.ones(p, dtype=coef.dtype),
    )

    def drift(X_arg, X_sq_arg, y_arg, w_arg, beta_arg, thresh_arg, p_arg, eps_arg, xp, backend):
        return beta_arg + 0.1

    monkeypatch.setattr(kernel, "_parallel_majorization_step", drift)

    with pytest.warns(ConvergenceWarning, match="target reached max_iter=2"):
        solvers.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.array([0.05]),
            max_lla_per_step=1,
            max_iter=2,
            tol=1e-12,
            lla_tol=1.0,
        )


def test_public_proximal_quantile_strict_target_exhaustion_raises(monkeypatch):
    X, y = _data(seed=16708)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)
    from statgpu.solvers import _proximal_irls_quantile as kernel

    monkeypatch.setattr(
        kernel,
        "_compute_lla_weights",
        lambda penalty_arg, coef, p, xp, backend: xp.ones(p, dtype=coef.dtype),
    )
    monkeypatch.setattr(
        kernel,
        "_parallel_majorization_step",
        lambda X_arg, X_sq_arg, y_arg, w_arg, beta_arg, thresh_arg, p_arg, eps_arg, xp, backend: beta_arg + 0.1,
    )

    token = kernel._STRICT_CV_TARGET.set(True)
    try:
        with pytest.raises(FloatingPointError, match="target reached max_iter=2"):
            solvers.proximal_irls_quantile_solver(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.array([0.05]),
                max_lla_per_step=1,
                max_iter=2,
                tol=1e-12,
                lla_tol=1.0,
            )
    finally:
        kernel._STRICT_CV_TARGET.reset(token)


def test_public_proximal_quantile_none_budget_has_defined_default():
    X, y = _data(seed=16703)
    loss = QuantileLoss(quantile=0.3)
    penalty = SCADPenalty(alpha=0.05)

    coef, intercept, n_iter = solvers.proximal_irls_quantile_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path=np.array([0.05]),
        max_lla_per_step=1,
        sample_weight=np.ones(X.shape[0]),
    )

    assert np.all(np.isfinite(coef))
    assert np.isfinite(intercept)
    assert 0 <= n_iter <= 100


def test_public_proximal_quantile_wrapper_preserves_introspection_and_alias():
    public = solvers.proximal_irls_quantile_solver
    historical = _prox_kernel.proximal_irls_quantile_solver
    kernel = getattr(public, "__wrapped__", None)

    assert historical is public
    assert kernel is not None
    assert inspect.signature(public) == inspect.signature(kernel)
    doc = " ".join((inspect.getdoc(public) or "").split())
    assert "None`` uses 100 iterations per step" in doc
    assert "Non-empty one-dimensional continuation path" in doc
    assert "finite positive values in non-increasing order" in doc


def test_quantile_solver_guard_reload_preserves_public_aliases_and_signatures(
    monkeypatch,
):
    import statgpu.solvers._lbfgs as _lbfgs_module

    reloaded = importlib.reload(_solver_guard)
    assert solvers.fista_solver is reloaded.fista_solver
    assert solvers.lbfgs_solver is reloaded.lbfgs_solver
    assert _lbfgs_module.lbfgs_solver is reloaded.lbfgs_solver

    fista_params = tuple(inspect.signature(solvers.fista_solver).parameters)
    lbfgs_params = tuple(inspect.signature(solvers.lbfgs_solver).parameters)
    assert fista_params == (
        "loss", "penalty", "X", "y", "max_iter", "tol", "init_coef",
        "sample_weight", "lipschitz_L", "cv_mode",
    )
    assert lbfgs_params == (
        "loss", "penalty", "X", "y", "max_iter", "tol", "init_coef",
        "history_size", "sample_weight",
    )

    captured = {}

    def fake_lbfgs(loss, penalty, X, y, **kwargs):
        captured["X"] = X
        captured["y"] = y
        return np.zeros(X.shape[1], dtype=np.float64), 1

    monkeypatch.setattr(reloaded, "_lbfgs_solver", fake_lbfgs)
    X, y = _data(seed=16741)
    coef, n_iter = solvers.lbfgs_solver(
        QuantileLoss(quantile=0.3),
        None,
        X.tolist(),
        y.tolist(),
        max_iter=3,
    )
    assert np.all(np.isfinite(coef))
    assert n_iter == 1
    assert isinstance(captured["X"], np.ndarray)
    assert isinstance(captured["y"], np.ndarray)


def test_quantile_irls_validation_installer_is_idempotent_under_reload():
    before = QuantileLoss.irls
    before_wrapped = getattr(before, "__wrapped__", None)

    _irls_contract.install_quantile_irls_validation_contract()
    assert QuantileLoss.irls is before

    importlib.reload(_irls_contract)
    after = QuantileLoss.irls
    assert after is before
    assert getattr(after, _irls_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped


def test_quantile_proximal_contract_is_idempotent_under_reload():
    before = solvers.proximal_irls_quantile_solver
    before_wrapped = getattr(before, "__wrapped__", None)

    assert _prox_contract.install_quantile_proximal_public_contract() is before
    importlib.reload(_prox_contract)
    after = _prox_kernel.proximal_irls_quantile_solver

    assert after is before
    assert getattr(after, _prox_contract._MARKER, False)
    assert getattr(after, "__wrapped__", None) is before_wrapped
