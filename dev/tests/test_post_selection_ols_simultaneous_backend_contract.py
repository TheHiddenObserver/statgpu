import sys
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.inference._results import DebiasedInferenceResult
from statgpu.linear_model import _debiased_simultaneous_backend_contract as native_sim
import statgpu.linear_model._debiased_intercept_parameterization_contract as intercept_contract


def _fixture(torch, *, include_intercept=True):
    X_raw = torch.tensor(
        [
            [1.5, -0.5],
            [0.2, 1.1],
            [-0.7, 0.4],
            [1.0, 0.8],
            [-1.2, -0.3],
            [0.4, -1.0],
        ],
        dtype=torch.float64,
    )
    X_work = X_raw - X_raw.mean(dim=0)
    y_raw = torch.tensor([1.3, 0.2, -0.4, 1.1, -1.0, 0.3], dtype=torch.float64)
    y_work = y_raw - y_raw.mean()
    row_scale = torch.ones(X_raw.shape[0], dtype=torch.float64)
    coef = torch.tensor([0.45, -0.2], dtype=torch.float64)
    M = torch.eye(2, dtype=torch.float64)
    params = np.asarray([0.35, 0.5, -0.25], dtype=np.float64)
    bse = np.asarray([0.12, 0.1, 0.11], dtype=np.float64)
    marginal = np.column_stack([params - 1.96 * bse, params + 1.96 * bse])
    result = DebiasedInferenceResult(
        params=params,
        bse=bse,
        statistic=params / bse,
        pvalues=np.asarray([0.01, 0.02, 0.04]),
        conf_int=marginal,
        metadata={"numerical_backend": "torch", "numerical_device": "cpu"},
    )
    model = SimpleNamespace(
        simultaneous_n_bootstrap=32,
        simultaneous_alpha=0.05,
        simultaneous_random_state=20260908,
        simultaneous_include_intercept=include_intercept,
        simultaneous_method="maxz_bootstrap",
    )
    return model, result, X_raw, y_work, X_work, row_scale, coef, M


def _run_torch_fixture(torch, *, include_intercept=True):
    model, result, X_raw, y_work, X_work, row_scale, coef, M = _fixture(
        torch,
        include_intercept=include_intercept,
    )
    native_sim._native_simultaneous_maxz(
        model,
        result,
        X_arr=X_raw,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef,
        M_native=M,
        backend_name="torch",
    )
    return model, result


@pytest.mark.parametrize("include_intercept", [False, True])
def test_native_simultaneous_maxz_uses_torch_backend_and_target_mask(include_intercept):
    torch = pytest.importorskip("torch")
    model, result, X_raw, y_work, X_work, row_scale, coef, M = _fixture(
        torch,
        include_intercept=include_intercept,
    )
    marginal = np.asarray(result.conf_int).copy()

    output = native_sim._native_simultaneous_maxz(
        model,
        result,
        X_arr=X_raw,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef,
        M_native=M,
        backend_name="torch",
    )

    assert output is result
    assert result.metadata["simultaneous_numerical_backend"] == "torch"
    assert result.metadata["simultaneous_numerical_device"] == "cpu"
    assert result.metadata["simultaneous_reporting_backend"] == "numpy"
    assert result.metadata["simultaneous_reporting_boundary"] == "post_numerical_inference"
    assert np.isfinite(result.simultaneous_critical_value)
    assert result.simultaneous_critical_value >= 0.0

    mask = np.asarray(result.simultaneous_target_mask, dtype=bool)
    expected_mask = np.asarray([include_intercept, True, True], dtype=bool)
    np.testing.assert_array_equal(mask, expected_mask)

    expected = marginal.copy()
    critical = float(result.simultaneous_critical_value)
    if include_intercept:
        expected[:, 0] = result.params - critical * result.bse
        expected[:, 1] = result.params + critical * result.bse
        assert hasattr(model, "_debiased_intercept_influence_cpu")
    else:
        expected[1:, 0] = result.params[1:] - critical * result.bse[1:]
        expected[1:, 1] = result.params[1:] + critical * result.bse[1:]
        assert not hasattr(model, "_debiased_intercept_influence_cpu")
    np.testing.assert_allclose(result.simultaneous_conf_int, expected, rtol=0, atol=1e-12)


def test_native_simultaneous_torch_matches_fixed_multiplier_oracle(monkeypatch):
    torch = pytest.importorskip("torch")
    model, result, X_raw, y_work, X_work, row_scale, coef, M = _fixture(
        torch,
        include_intercept=True,
    )
    rng = np.random.default_rng(20260914)
    xi_np = rng.standard_normal(size=(model.simultaneous_n_bootstrap, X_work.shape[0]))
    xi_t = torch.as_tensor(xi_np, dtype=X_work.dtype)
    calls = {"count": 0}

    def fixed_random(backend_name, *, shape, ref_arr, rng):
        assert backend_name == "torch"
        assert tuple(shape) == tuple(xi_t.shape)
        assert ref_arr is X_work
        calls["count"] += 1
        return xi_t

    monkeypatch.setattr(native_sim, "_random_normal", fixed_random)

    X_raw_np = X_raw.numpy()
    X_work_np = X_work.numpy()
    y_work_np = y_work.numpy()
    coef_np = coef.numpy()
    resid_np = y_work_np - X_work_np @ coef_np
    n = X_work_np.shape[0]
    x_mean_np = X_raw_np.mean(axis=0)
    M_np = M.numpy()
    q_np = np.ones(n, dtype=np.float64) - X_work_np @ (M_np.T @ x_mean_np)
    influence_np = q_np / float(n)
    multiplier_resid = xi_np * resid_np.reshape(1, -1)
    feature_score = (multiplier_resid @ X_work_np) @ M_np.T / float(n)
    z_feature = feature_score / result.bse[1:].reshape(1, -1)
    z_intercept = (multiplier_resid @ influence_np) / float(result.bse[0])
    max_stats = np.maximum(
        np.max(np.abs(z_feature), axis=1),
        np.abs(z_intercept),
    )
    expected_critical = float(np.quantile(max_stats, 1.0 - model.simultaneous_alpha))

    native_sim._native_simultaneous_maxz(
        model,
        result,
        X_arr=X_raw,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef,
        M_native=M,
        backend_name="torch",
    )

    assert calls["count"] == 1
    assert result.simultaneous_critical_value == pytest.approx(
        expected_critical,
        rel=0,
        abs=2e-12,
    )


def test_native_simultaneous_rejects_nonfinite_bootstrap_draws_before_quantile(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    model, result, X_raw, y_work, X_work, row_scale, coef, M = _fixture(
        torch,
        include_intercept=True,
    )
    xi = torch.zeros(
        (model.simultaneous_n_bootstrap, X_work.shape[0]),
        dtype=X_work.dtype,
    )
    xi[0, :] = torch.finfo(X_work.dtype).max
    assert bool(torch.all(torch.isfinite(xi)))

    def extreme_finite_random(backend_name, *, shape, ref_arr, rng):
        assert backend_name == "torch"
        assert tuple(shape) == tuple(xi.shape)
        assert ref_arr is X_work
        return xi

    monkeypatch.setattr(native_sim, "_random_normal", extreme_finite_random)

    with pytest.raises(
        FloatingPointError,
        match=r"non-finite bootstrap max-\|Z\| statistics",
    ):
        native_sim._native_simultaneous_maxz(
            model,
            result,
            X_arr=X_raw,
            y_work=y_work,
            X_work=X_work,
            row_scale=row_scale,
            coef_native=coef,
            M_native=M,
            backend_name="torch",
        )

    assert result.simultaneous_conf_int is None
    assert result.simultaneous_critical_value is None


def test_native_simultaneous_torch_seed_is_repeatable():
    torch = pytest.importorskip("torch")
    _, first = _run_torch_fixture(torch, include_intercept=True)
    _, second = _run_torch_fixture(torch, include_intercept=True)

    assert first.simultaneous_critical_value == pytest.approx(
        second.simultaneous_critical_value,
        rel=0,
        abs=0,
    )
    np.testing.assert_array_equal(
        first.simultaneous_conf_int,
        second.simultaneous_conf_int,
    )


def test_native_simultaneous_cupy_enters_design_device_context(monkeypatch):
    state = {"active": False, "device": None}
    sentinel = object()

    class _FakeDeviceContext:
        def __init__(self, device_id):
            self.device_id = int(device_id)

        def __enter__(self):
            assert not state["active"]
            state["active"] = True
            state["device"] = self.device_id
            return self

        def __exit__(self, exc_type, exc, tb):
            state["active"] = False
            return False

    fake_cupy = SimpleNamespace(
        cuda=SimpleNamespace(Device=lambda device_id: _FakeDeviceContext(device_id))
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    def fake_on_device(*args, **kwargs):
        assert state["active"] is True
        assert state["device"] == 4
        assert kwargs["backend_name"] == "cupy"
        return sentinel

    monkeypatch.setattr(native_sim, "_native_simultaneous_maxz_on_device", fake_on_device)
    X_work = SimpleNamespace(device=SimpleNamespace(id=4))

    output = native_sim._native_simultaneous_maxz(
        object(),
        object(),
        X_arr=object(),
        y_work=object(),
        X_work=X_work,
        row_scale=object(),
        coef_native=object(),
        M_native=object(),
        backend_name="cupy",
    )

    assert output is sentinel
    assert state["active"] is False
    assert state["device"] == 4


def test_gpu_simultaneous_finalizer_suppresses_inner_cpu_simultaneous(monkeypatch):
    torch = pytest.importorskip("torch")
    model, result, X_raw, y_work, X_work, row_scale, coef, M = _fixture(
        torch,
        include_intercept=True,
    )
    setattr(model, intercept_contract._NATIVE_M, M)
    observed = {}

    def fake_marginal_finalizer(model_arg, **kwargs):
        observed["simultaneous_requested"] = kwargs["simultaneous_requested"]
        return result

    monkeypatch.setattr(native_sim, "_ORIGINAL_FINALIZER", fake_marginal_finalizer)

    output = native_sim._finalize_weighted_debiased_result(
        model,
        X_arr=X_raw,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef,
        backend_name="torch",
        original_intercept=True,
        simultaneous_requested=True,
        sample_weighted=True,
    )

    assert output is result
    assert observed["simultaneous_requested"] is False
    assert result.metadata["simultaneous_numerical_backend"] == "torch"
    assert result.simultaneous_conf_int is not None
