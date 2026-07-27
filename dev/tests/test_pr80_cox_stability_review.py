"""Regression gates for the final PR80 Cox stability review."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival._cox import _estimate_breslow_tensor_bytes
from statgpu.survival import _cox_counting as cox_counting
from statgpu.survival._cox_counting import _score_test_statistic, _solve
from statgpu.survival._cox_cv import _compute_partial_likelihood


def _require_device(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA unavailable")
        return cp
    if device == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        return torch
    return np


def _on_device(device, value):
    xp = _require_device(device)
    if device == "cuda":
        return xp.asarray(value)
    if device == "torch":
        return xp.as_tensor(value, device="cuda")
    return np.asarray(value)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
def test_ordinary_cox_centered_large_predictor_stays_finite(ties, device):
    if device != "cpu":
        _require_device(device)
    X = _on_device(device, np.array([[-1000.0], [0.0], [1000.0]]))
    stop = _on_device(device, np.array([1.0, 2.0, 3.0]))
    event = _on_device(device, np.array([1.0, 1.0, 0.0]))
    init = _on_device(device, np.array([1.0]))
    model = CoxPH(
        ties=ties,
        device=device,
        compute_inference=False,
        compute_cindex=False,
        max_iter=40,
    ).fit(X, stop, event, init_coef=init)

    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.log_likelihood)
    assert np.all(np.isfinite(model._objective_history))
    assert model._is_counting_process is False


@pytest.mark.parametrize(
    "field", ["X", "packed", "time", "event", "start", "init_coef"]
)
def test_coxph_fit_rejects_complex_before_high_level_cast(field):
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 1.0, 0.0])
    start = np.zeros(4)
    init = np.zeros(1)
    values = {
        "X": X.astype(np.complex128) + 1j,
        "packed": np.column_stack((stop, event)).astype(np.complex128) + 1j,
        "time": stop.astype(np.complex128) + 1j,
        "event": event.astype(np.complex128) + 1j,
        "start": start.astype(np.complex128) + 1j,
        "init_coef": init.astype(np.complex128) + 1j,
    }
    model = CoxPH(device="cpu", compute_inference=False)
    with pytest.raises(ValueError, match="real-valued"):
        if field == "packed":
            model.fit(X, values[field])
        else:
            model.fit(
                values[field] if field == "X" else X,
                values[field] if field == "time" else stop,
                values[field] if field == "event" else event,
                start=values[field] if field == "start" else start,
                init_coef=values[field] if field == "init_coef" else init,
            )


@pytest.mark.parametrize("field", ["X", "packed", "time", "event", "start"])
def test_coxph_score_rejects_complex_before_high_level_cast(field):
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 1.0, 0.0])
    model = CoxPH(device="cpu", compute_inference=False).fit(X, stop, event)
    values = {
        "X": X.astype(np.complex128) + 1j,
        "packed": np.column_stack((stop, event)).astype(np.complex128) + 1j,
        "time": stop.astype(np.complex128) + 1j,
        "event": event.astype(np.complex128) + 1j,
        "start": np.zeros(4, dtype=np.complex128) + 1j,
    }
    with pytest.raises(ValueError, match="real-valued"):
        if field == "packed":
            model.score(X, values[field])
        else:
            model.score(
                values[field] if field == "X" else X,
                values[field] if field == "time" else stop,
                values[field] if field == "event" else event,
                start=values[field] if field == "start" else None,
            )


@pytest.mark.parametrize("field", ["X", "packed", "time", "event", "start"])
def test_coxphcv_rejects_complex_before_high_level_cast(field):
    X = np.arange(12, dtype=np.float64).reshape(6, 2) / 10.0
    stop = np.arange(1, 7, dtype=np.float64)
    event = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    values = {
        "X": X.astype(np.complex128) + 1j,
        "packed": np.column_stack((stop, event)).astype(np.complex128) + 1j,
        "time": stop.astype(np.complex128) + 1j,
        "event": event.astype(np.complex128) + 1j,
        "start": np.zeros(6, dtype=np.complex128) + 1j,
    }
    model = CoxPHCV(device="cpu", penalties=np.array([0.1]), cv=2)
    with pytest.raises(ValueError, match="real-valued"):
        if field == "packed":
            model.fit(X, values[field])
        else:
            model.fit(
                values[field] if field == "X" else X,
                values[field] if field == "time" else stop,
                values[field] if field == "event" else event,
                start=values[field] if field == "start" else None,
            )


def test_coxphcv_score_rejects_complex_before_high_level_cast():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    stop = np.arange(1, 5, dtype=np.float64)
    event = np.array([1.0, 1.0, 1.0, 0.0])
    fitted = CoxPH(device="cpu", compute_inference=False).fit(X, stop, event)
    model = CoxPHCV(device="cpu", penalties=np.array([0.1]), cv=2)
    model.estimator_ = fitted
    packed = np.column_stack((stop, event)).astype(np.complex128) + 1j
    with pytest.raises(ValueError, match="packed survival target.*real-valued"):
        model.score(X, packed)


@pytest.mark.parametrize("field", ["X", "time", "event", "coef", "entry"])
def test_cv_partial_likelihood_rejects_complex_before_cast(field):
    X = np.arange(8, dtype=np.float64).reshape(4, 2) / 10.0
    stop = np.arange(1, 5, dtype=np.float64)
    event = np.array([1.0, 0.0, 1.0, 0.0])
    coef = np.array([0.1, -0.2])
    entry = np.zeros(4)
    values = {"X": X, "time": stop, "event": event, "coef": coef, "entry": entry}
    values[field] = values[field].astype(np.complex128) + 1j
    with pytest.raises(ValueError, match=rf"{field}.*real-valued"):
        _compute_partial_likelihood(
            values["X"], values["time"], values["event"], values["coef"],
            entry=values["entry"],
        )


def test_no_comparable_pairs_score_contract_is_always_neutral():
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    model = CoxPH(device="cpu", compute_inference=False).fit(
        X, np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 1, 0])
    )
    X_score = np.array([[0.0], [1.0]])
    stop = np.array([1.0, 2.0])
    event = np.array([0, 1])
    assert model.score(X_score, stop, event) == 0.5
    assert model.score(X_score, stop, event, subject_id=np.array([0, 1])) == 0.5


def test_ordinary_fit_uses_stable_suffix_kernel_not_dense_risk_sets(monkeypatch):
    def forbidden_reference(*_args, **_kwargs):
        raise AssertionError("ordinary Cox entered the dense group-by-row objective")

    monkeypatch.setattr(
        cox_counting, "cox_counting_process_objective", forbidden_reference
    )
    rng = np.random.default_rng(2100)
    X = rng.normal(size=(300, 3))
    stop = rng.uniform(0.1, 10.0, size=300)
    event = rng.binomial(1, 0.7, size=300)
    event[0] = 1
    model = CoxPH(
        compute_inference=False, compute_cindex=False, device="cpu"
    ).fit(X, stop, event)
    assert np.all(np.isfinite(model.coef_))


def test_cpu_breslow_tensor_workspace_gate_forces_incremental(monkeypatch):
    rng = np.random.default_rng(2101)
    X = rng.normal(size=(40, 3))
    stop = np.arange(1, 41, dtype=np.float64)
    event = np.zeros(40, dtype=np.int64)
    event[[0, 8, 16, 24, 32]] = 1
    exp_eta = np.ones(40)
    risk_sum = np.cumsum(exp_eta[::-1])[::-1]
    risk_X_sum = np.cumsum((X * exp_eta[:, None])[::-1], axis=0)[::-1]
    first_idx = np.where(event == 1)[0]
    counts = np.ones(first_idx.size)
    model = CoxPH(compute_inference=False)
    expected = model._compute_hessian_breslow_incremental_grouped(
        X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    )
    monkeypatch.setenv("STATGPU_BRESLOW_HESSIAN_MAX_BYTES", "0")

    def forbidden_tensor(*_args, **_kwargs):
        raise AssertionError("tensor Hessian bypassed the workspace gate")

    monkeypatch.setattr(model, "_compute_hessian_breslow_tensor_grouped", forbidden_tensor)
    actual = model._compute_hessian_breslow_fast(
        X, stop, event, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert model._last_breslow_hessian_strategy_ == "incremental"


def test_breslow_workspace_estimate_covers_large_n_tensor_peak():
    assert _estimate_breslow_tensor_bytes(10_000_000, 24, 512) > 90_000_000_000


class SentinelDeviceError(RuntimeError):
    pass


def test_counting_solve_preserves_device_error_and_skips_lstsq():
    calls = []

    class Linalg:
        @staticmethod
        def solve(_information, _score):
            raise SentinelDeviceError("CUDA out of memory sentinel")

        @staticmethod
        def lstsq(*_args, **_kwargs):
            calls.append("lstsq")
            raise AssertionError("lstsq must not run after a device failure")

    with pytest.raises(SentinelDeviceError, match="out of memory sentinel"):
        _solve(np.eye(1), np.ones(1), "cupy", SimpleNamespace(linalg=Linalg()))
    assert calls == []


def test_counting_solve_uses_lstsq_only_for_singularity():
    calls = []

    class Linalg:
        @staticmethod
        def solve(_information, _score):
            raise np.linalg.LinAlgError("Singular matrix")

        @staticmethod
        def lstsq(_information, _score, rcond=None):
            calls.append(rcond)
            return np.array([2.0]), None, None, None

    result = _solve(np.zeros((1, 1)), np.ones(1), "numpy", SimpleNamespace(linalg=Linalg()))
    np.testing.assert_array_equal(result, np.array([2.0]))
    assert calls == [None]


def test_score_test_device_error_propagates_but_singularity_is_diagnostic():
    class DeviceLinalg:
        @staticmethod
        def solve(_information, _score):
            raise SentinelDeviceError("CUDA illegal memory access sentinel")

    with pytest.raises(SentinelDeviceError, match="illegal memory access"):
        _score_test_statistic(
            np.ones(1), np.eye(1), "cupy", SimpleNamespace(linalg=DeviceLinalg())
        )

    class SingularLinalg:
        @staticmethod
        def solve(_information, _score):
            raise np.linalg.LinAlgError("Singular matrix")

    statistic, reason = _score_test_statistic(
        np.ones(1), np.zeros((1, 1)), "numpy", SimpleNamespace(linalg=SingularLinalg())
    )
    assert statistic is None
    assert "numpy null information is singular" in reason


def test_cupy_fused_hessian_does_not_swallow_runtime_error(monkeypatch):
    from statgpu.survival import _cox_efron_cuda

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())

    def fail(*_args, **_kwargs):
        raise SentinelDeviceError("CUDA out of memory sentinel")

    monkeypatch.setattr(_cox_efron_cuda, "compute_breslow_hess_raw", fail)
    with pytest.raises(SentinelDeviceError, match="out of memory sentinel"):
        CoxPH(compute_inference=False)._compute_hessian_breslow_fused_cupy(
            None, None, None, None
        )


def test_legacy_torch_newton_does_not_relabel_device_error(monkeypatch):
    torch = pytest.importorskip("torch")

    def fail(*_args, **_kwargs):
        raise SentinelDeviceError("CUDA out of memory sentinel")

    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(SentinelDeviceError, match="out of memory sentinel"):
        CoxPH(compute_inference=False)._solve_newton_delta_torch(
            -torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
        )


def test_successful_inference_records_score_test_availability():
    rng = np.random.default_rng(2102)
    X = rng.normal(size=(120, 2))
    stop = rng.uniform(0.2, 8.0, size=120)
    event = rng.binomial(1, 0.65, size=120)
    event[0] = 1
    model = CoxPH(
        compute_inference=True, compute_cindex=False, device="cpu"
    ).fit(X, stop, event)
    assert model.score_test_available_ is True
    assert model.score_test_failure_reason_ is None
    assert np.isfinite(model._score_test_stat)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_high_level_complex_guards_preserve_gpu_backend_contract(device):
    _require_device(device)
    X_np = np.arange(12, dtype=np.float64).reshape(6, 2) / 10.0
    stop_np = np.arange(1, 7, dtype=np.float64)
    event_np = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    X = _on_device(device, X_np)
    stop = _on_device(device, stop_np)
    event = _on_device(device, event_np)
    complex_X = _on_device(device, X_np.astype(np.complex128) + 1j)
    complex_target = _on_device(
        device,
        np.column_stack((stop_np, event_np)).astype(np.complex128) + 1j,
    )

    with pytest.raises(ValueError, match="X.*real-valued"):
        CoxPH(device=device, compute_inference=False).fit(
            complex_X, stop, event
        )
    fitted = CoxPH(
        device=device, compute_inference=False, penalty=0.1
    ).fit(X, stop, event)
    with pytest.raises(ValueError, match="packed survival target.*real-valued"):
        fitted.score(X, complex_target)
    with pytest.raises(ValueError, match="X.*real-valued"):
        CoxPHCV(device=device, penalties=np.array([0.1]), cv=2).fit(
            complex_X, stop, event
        )


@pytest.mark.gpu
def test_cupy_breslow_workspace_gate_matches_vectorized(monkeypatch):
    cp = _require_device("cuda")
    rng = np.random.default_rng(2103)
    X = cp.asarray(rng.normal(size=(64, 3)))
    exp_eta = cp.ones(64, dtype=cp.float64)
    risk_sum = cp.cumsum(exp_eta[::-1])[::-1]
    risk_X_sum = cp.cumsum((X * exp_eta[:, None])[::-1], axis=0)[::-1]
    first_idx = cp.asarray([0, 8, 16, 24, 32, 40, 48, 56], dtype=cp.int64)
    counts = cp.ones(first_idx.size, dtype=cp.float64)
    model = CoxPH(compute_inference=False, device="cuda")
    monkeypatch.setenv("STATGPU_BRESLOW_HESSIAN_MAX_BYTES", str(1 << 60))
    expected = model._compute_hessian_breslow_incremental_grouped_cupy(
        X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    )
    monkeypatch.setenv("STATGPU_BRESLOW_HESSIAN_MAX_BYTES", "0")
    actual = model._compute_hessian_breslow_incremental_grouped_cupy(
        X, risk_sum, risk_X_sum, exp_eta, first_idx, counts
    )
    cp.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert model._last_breslow_hessian_strategy_ == "cupy_streaming"
