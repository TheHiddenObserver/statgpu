"""Regression tests for the PR80 follow-up performance review."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from dev.benchmarks.benchmark_exact_ties_scaling import (
    SCENARIOS,
    make_scaling_data,
    summarize_runs,
)
from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.survival import _risk_sets as risk_sets
from statgpu.survival._risk_sets import prepare_counting_process_inputs


def _survival_data(n=56, p=4, seed=8181):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    stop = rng.integers(1, 15, size=n).astype(np.float64)
    event = rng.binomial(1, 0.7, size=n).astype(np.float64)
    event[0] = 1.0
    return X, np.column_stack((stop, event))


def _require_device(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CuPy CUDA unavailable")
    elif device == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
def test_penalized_cox_lla_preprocesses_once_and_releases_cache(
    monkeypatch, penalty, device
):
    _require_device(device)
    calls = []
    original = CoxPartialLikelihoodLoss.preprocess

    def recording_preprocess(self, X, y):
        calls.append((self, X, y))
        return original(self, X, y)

    def unused_fused_objective(*args, **kwargs):
        raise AssertionError("FISTA-LLA must use the gradient-only Cox path")

    monkeypatch.setattr(CoxPartialLikelihoodLoss, "preprocess", recording_preprocess)
    monkeypatch.setattr(
        CoxPartialLikelihoodLoss,
        "fused_value_and_gradient",
        unused_fused_objective,
    )
    X, y = _survival_data()
    model = PenalizedCoxPHModel(
        penalty=penalty,
        alpha=0.04,
        ties="efron",
        max_iter=35,
        max_lla_iters=4,
        tol=1e-4,
        device=device,
        gpu_memory_cleanup=True,
    ).fit(X, y)

    assert len(calls) == 1
    assert np.all(np.isfinite(model.coef_))
    assert model._loss is calls[0][0]
    for name in (
        "_X_sorted",
        "_time_sorted",
        "_event_sorted",
        "_order",
        "_group_first_indices_backend",
        "_group_event_indices_backend",
        "_event_group_codes_backend",
        "_efron_fractions_backend",
        "_preprocessed_target",
    ):
        assert getattr(model._loss, name) is None


def test_cox_preprocessed_contract_and_backend_scalar_value():
    X, y = _survival_data(n=32, p=3)
    loss = CoxPartialLikelihoodLoss(ties="efron")
    X_pre, y_pre = loss.preprocess(X, y)
    assert loss.is_preprocessed(X_pre, y_pre)

    value, gradient = loss.fused_value_and_gradient(
        X_pre, y_pre, np.zeros(X.shape[1])
    )
    assert isinstance(value, np.generic) and value.ndim == 0
    assert gradient.shape == (X.shape[1],)
    metadata_ids = tuple(id(value) for value in loss._backend_group_metadata(np, X_pre))
    loss.gradient_preprocessed(np.zeros(X.shape[1]))
    assert metadata_ids == tuple(
        id(value) for value in loss._backend_group_metadata(np, X_pre)
    )

    _, replacement = loss.preprocess(X, y)
    assert not loss.is_preprocessed(X_pre, y_pre)
    assert replacement is not y_pre


def _backend_extreme_survival_arrays(backend):
    X = np.array([[1000.0], [0.0]], dtype=np.float64)
    y = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
    coef = np.array([1.0], dtype=np.float64)
    if backend == "cupy":
        _require_device("cuda")
        import cupy as cp

        return cp.asarray(X), cp.asarray(y), cp.asarray(coef)
    if backend == "torch":
        _require_device("torch")
        import torch

        return tuple(
            torch.as_tensor(value, dtype=torch.float64, device="cuda")
            for value in (X, y, coef)
        )
    return X, y, coef


def _array_to_numpy(value):
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    if module == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_gradient_preprocessed_extreme_departing_maximum(ties, backend):
    X, y, coef = _backend_extreme_survival_arrays(backend)
    loss = CoxPartialLikelihoodLoss(ties=ties)
    X_pre, y_pre = loss.preprocess(X, y)

    trusted = loss.gradient_preprocessed(coef)
    public = loss.gradient(X_pre, y_pre, coef)
    shared = -loss._shared_objective(
        coef, compute_derivatives=True
    )["score"] / X.shape[0]

    trusted_np = _array_to_numpy(trusted)
    public_np = _array_to_numpy(public)
    shared_np = _array_to_numpy(shared)
    assert np.all(np.isfinite(trusted_np))
    np.testing.assert_allclose(trusted_np, public_np, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(trusted_np, shared_np, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("penalty", ["scad", "mcp"])
@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
def test_penalized_cox_extreme_departing_maximum_stays_finite(
    penalty, ties, device
):
    _require_device(device)
    X, y, _ = _backend_extreme_survival_arrays(
        "numpy" if device == "cpu" else device
    )
    model = PenalizedCoxPHModel(
        penalty=penalty,
        alpha=0.01,
        ties=ties,
        max_iter=5,
        max_lla_iters=1,
        tol=1e-6,
        device=device,
    )
    model._init_coef = np.array([1.0])
    model.fit(X, y)
    assert np.all(np.isfinite(model.coef_))
    np.testing.assert_allclose(model.coef_, [1.0], rtol=0.0, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.memory
@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_penalized_cox_cleanup_does_not_retain_training_gpu_arrays(device):
    _require_device(device)
    X_np, y_np = _survival_data(n=160, p=6)
    if device == "cuda":
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        X = cp.asarray(X_np)
        y = cp.asarray(y_np)
        cp.cuda.Stream.null.synchronize()
        baseline = pool.used_bytes()
    else:
        import torch

        torch.cuda.empty_cache()
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

    model = PenalizedCoxPHModel(
        penalty="scad",
        alpha=0.04,
        ties="efron",
        max_iter=30,
        max_lla_iters=4,
        device=device,
        gpu_memory_cleanup=True,
    ).fit(X, y)
    assert model._loss._X_sorted is None
    gc.collect()
    if device == "cuda":
        cp.cuda.Stream.null.synchronize()
        pool.free_all_blocks()
        active_after = pool.used_bytes()
    else:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        active_after = torch.cuda.memory_allocated()
    assert active_after <= baseline + 1024 * 1024


@pytest.mark.parametrize("bad", [[0.2, 0.8, 1.0], [0.0, np.nan, 1.0]])
def test_fractional_or_nonfinite_strata_are_rejected_before_cast(bad):
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="strata.*integer-valued"):
        prepare_counting_process_inputs(X, stop, event, strata=np.asarray(bad))


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_fractional_strata_validation_is_backend_consistent(backend):
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    strata = np.array([0.0, 0.5, 1.0])
    if backend == "cupy":
        _require_device("cuda")
        import cupy as xp

        X, stop, event, strata = map(xp.asarray, (X, stop, event, strata))
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        X, stop, event, strata = (
            torch.as_tensor(value) for value in (X, stop, event, strata)
        )
    with pytest.raises(ValueError, match="strata.*integer-valued"):
        prepare_counting_process_inputs(X, stop, event, strata=strata)


def test_integral_float_strata_are_accepted():
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    *_, strata = prepare_counting_process_inputs(
        X, stop, event, strata=np.array([0.0, 1.0, 1.0])
    )
    np.testing.assert_array_equal(strata, [0, 1, 1])
    assert strata.dtype == np.int64


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1e30, 2e30], dtype=np.float64),
        np.array([-1e30, 0.0], dtype=np.float64),
        np.array([np.iinfo(np.uint64).max, 0], dtype=np.uint64),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_out_of_int64_range_strata_are_rejected_before_cast(bad, backend):
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    strata = np.resize(bad, 3)
    if backend == "cupy":
        _require_device("cuda")
        import cupy as cp

        X, stop, event, strata = map(cp.asarray, (X, stop, event, strata))
    elif backend == "torch":
        _require_device("torch")
        import torch

        X = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        stop = torch.as_tensor(stop, dtype=torch.float64, device="cuda")
        event = torch.as_tensor(event, dtype=torch.float64, device="cuda")
        # Keep uint64 on the host: Torch 2.0 cannot represent it, and the
        # normalization boundary must still convert that failure to ValueError.
        if strata.dtype.kind != "u":
            strata = torch.as_tensor(
                strata, dtype=torch.float64, device="cuda"
            )
    with pytest.raises(ValueError, match="int64 range"):
        prepare_counting_process_inputs(X, stop, event, strata=strata)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_int64_boundary_strata_are_preserved(backend):
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 0.0, 1.0])
    strata = np.array(
        [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max],
        dtype=np.int64,
    )
    if backend == "cupy":
        _require_device("cuda")
        import cupy as cp

        X, stop, event, strata = map(cp.asarray, (X, stop, event, strata))
    elif backend == "torch":
        _require_device("torch")
        import torch

        X = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        stop = torch.as_tensor(stop, dtype=torch.float64, device="cuda")
        event = torch.as_tensor(event, dtype=torch.float64, device="cuda")
        strata = torch.as_tensor(strata, dtype=torch.int64, device="cuda")
    *_, actual = prepare_counting_process_inputs(
        X, stop, event, strata=strata
    )
    np.testing.assert_array_equal(_array_to_numpy(actual), strata.cpu().numpy() if backend == "torch" else _array_to_numpy(strata))


def test_channelwise_scan_env_parsing_and_auto_gate(monkeypatch):
    monkeypatch.setenv("STATGPU_TORCH_EXACT_SCAN_MIN_ROWS", "not-an-int")
    monkeypatch.setenv("STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS", "999999")
    assert risk_sets._torch_channelwise_scan_limits() == (2048, 4096)

    class FakeCuda:
        @staticmethod
        def get_device_capability(device):
            return (6, 0)

    fake_torch = SimpleNamespace(__version__="2.0.0+cu117", cuda=FakeCuda())
    value = SimpleNamespace(device="cuda:0")
    monkeypatch.setenv("STATGPU_TORCH_EXACT_SCAN_STRATEGY", "auto")
    assert risk_sets._torch_channelwise_scan_strategy(value, fake_torch) == "channelwise"
    fake_torch.__version__ = "2.4.0"
    assert risk_sets._torch_channelwise_scan_strategy(value, fake_torch) == "native"
    monkeypatch.setenv("STATGPU_TORCH_EXACT_SCAN_STRATEGY", "channelwise")
    assert risk_sets._torch_channelwise_scan_strategy(value, fake_torch) == "channelwise"


def test_invalid_exact_workspace_env_values_use_safe_defaults(monkeypatch):
    monkeypatch.setenv("STATGPU_EXACT_NESTED_MAX_BYTES", "invalid")
    monkeypatch.setenv("STATGPU_EXACT_BATCH_MAX_BYTES", "invalid")
    X = np.arange(16, dtype=np.float64).reshape(8, 2) / 10.0
    stop = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64)
    event = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int64)
    result = risk_sets.cox_counting_process_objective(
        np.zeros(2), X, stop, event, ties="exact"
    )
    assert np.isfinite(result["log_likelihood"])


def test_exact_strata_fallback_is_local_to_each_stratum(monkeypatch):
    rng = np.random.default_rng(104)
    n_strata, rows_per_stratum, p = 4, 8, 2
    n = n_strata * rows_per_stratum
    X = rng.normal(size=(n, p))
    strata = np.repeat(np.arange(n_strata), rows_per_stratum)
    stop = np.tile(np.arange(1, rows_per_stratum + 1), n_strata).astype(float)
    event = np.zeros(n, dtype=np.int64)
    event[::rows_per_stratum] = 1
    start = np.zeros(n)
    eta = X @ np.array([0.1, -0.2])
    expected = risk_sets._reference_exact_group_objective(
        eta,
        X,
        stop,
        event,
        start,
        strata,
        score_residuals=False,
        compute_derivatives=True,
    )
    reference = risk_sets._reference_exact_group_objective
    fallback_sizes = []

    def local_reference(eta_s, X_s, *args, **kwargs):
        fallback_sizes.append(int(X_s.shape[0]))
        return reference(eta_s, X_s, *args, **kwargs)

    monkeypatch.setattr(risk_sets, "_nested_exact_group_objective", lambda *a, **k: None)
    monkeypatch.setattr(risk_sets, "_batched_exact_group_objective", lambda *a, **k: None)
    monkeypatch.setattr(risk_sets, "_reference_exact_group_objective", local_reference)
    actual = risk_sets._stratified_exact_group_objective(
        eta,
        X,
        stop,
        event,
        start,
        strata,
        score_residuals=False,
        compute_derivatives=True,
    )
    assert fallback_sizes == [rows_per_stratum] * n_strata
    for key in ("log_likelihood", "score", "information"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_many_strata_delayed_entry_uses_one_gpu_batch(backend, monkeypatch):
    rng = np.random.default_rng(105)
    n_strata, rows, p = 8, 8, 2
    n = n_strata * rows
    X = rng.normal(size=(n, p))
    strata = np.repeat(np.arange(n_strata), rows)
    stop = np.tile(np.arange(2, rows + 2), n_strata).astype(float)
    start = stop * rng.uniform(0.0, 0.5, size=n)
    event = np.zeros(n, dtype=np.int64)
    event[::rows] = 1
    beta = np.array([0.1, -0.2])
    if backend == "cupy":
        _require_device("cuda")
        import cupy as xp

        beta, X, stop, event, start, strata = map(
            xp.asarray, (beta, X, stop, event, start, strata)
        )
    else:
        _require_device("torch")
        xp = pytest.importorskip("torch")
        beta, X, stop, start = (
            xp.as_tensor(value, dtype=xp.float64, device="cuda")
            for value in (beta, X, stop, start)
        )
        event, strata = (
            xp.as_tensor(value, dtype=xp.int64, device="cuda")
            for value in (event, strata)
        )

    selected_sizes = []
    original = risk_sets._batched_exact_group_objective

    def recording_batch(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None:
            selected_sizes.append(int(args[1].shape[0]))
        return result

    monkeypatch.setattr(risk_sets, "_batched_exact_group_objective", recording_batch)
    optimized = risk_sets.cox_counting_process_objective(
        beta, X, stop, event, start=start, strata=strata, ties="exact"
    )
    assert selected_sizes == [n]

    monkeypatch.setenv("STATGPU_EXACT_BATCH_MAX_BYTES", "0")
    reference = risk_sets.cox_counting_process_objective(
        beta, X, stop, event, start=start, strata=strata, ties="exact"
    )
    for key in ("log_likelihood", "score", "information"):
        if backend == "torch":
            assert xp.allclose(optimized[key], reference[key], rtol=1e-11, atol=1e-11)
        else:
            assert xp.allclose(optimized[key], reference[key], rtol=1e-11, atol=1e-11)


def test_scaling_generator_covers_all_four_scenarios():
    for scenario in SCENARIOS:
        X, stop, event, start, strata, _ = make_scaling_data(
            scenario, 48, 3, 91, strata_count=7
        )
        assert X.shape == (48, 3)
        assert stop.shape == event.shape == (48,)
        assert (start is not None) == scenario.startswith("delayed_entry")
        assert (strata is not None) == scenario.endswith("strata")
        if strata is not None:
            assert np.unique(strata).size == 7


def _fake_run(seconds, converged=True, offset=0.0):
    return {
        "status": "complete",
        "seconds": seconds,
        "coef": [1.0 + offset],
        "log_likelihood": -2.0 + offset,
        "covariance": [[0.5 + offset]],
        "iterations": 3,
        "converged": converged,
    }


def test_run_summary_uses_median_rank_and_checks_every_repeat():
    summary = summarize_runs(
        [_fake_run(3.0, offset=3.0), _fake_run(1.0, offset=1.0), _fake_run(2.0, False, 2.0)]
    )
    assert summary["median_seconds"] == 2.0
    assert summary["representative_seconds"] == 2.0
    assert summary["representative_run_index"] == 2
    assert summary["coef"] == [3.0]
    assert summary["all_converged"] is False
    assert summary["converged"] is False
    assert summary["all_finite"] is True


def test_run_summary_records_r_timeout_without_numeric_placeholder():
    summary = summarize_runs(
        [{"status": "timeout", "timeout_seconds": 120, "seconds": None}]
    )
    assert summary["status"] == "timeout"
    assert summary["median_seconds"] is None
    assert summary["timeout_seconds"] == 120
