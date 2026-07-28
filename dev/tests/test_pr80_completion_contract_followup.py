"""Regression gates for the final PR #80 completion-contract follow-up."""

from unittest.mock import Mock
import inspect
import subprocess
import sys

import numpy as np
import pytest

from statgpu.inference import ParameterInferenceResult
from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival import _cox as cox_module
from statgpu.survival import _cox_score as cox_score_module
from statgpu.survival._cox_legacy import (
    _LegacyCoxReference,
    _LegacyCoxReferenceMixin,
)
from statgpu.survival._risk_sets import counting_process_concordance


def _sample(seed=801, n=48, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    stop = np.arange(1, n + 1, dtype=np.float64)
    event = (rng.uniform(size=n) > 0.25).astype(np.float64)
    event[0] = 1.0
    return X, stop, event


def _device_name(backend):
    return {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]


def _fitted_prediction_model(backend):
    X, stop, event = _sample(p=1)
    X, stop, event = _backend_arrays(backend, X, stop, event)
    model = CoxPH(
        device=_device_name(backend),
        compute_inference=False,
        compute_cindex=False,
        gpu_memory_cleanup=True,
    ).fit(X, stop, event)
    model._unique_times = np.array([1.0, 2.0, 3.0])
    model._baseline_cumulative_hazard = np.array([0.1, 0.2, 0.3])
    return model, X, stop, event


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_coxph_public_prediction_and_score_cleanup_on_success(backend):
    model, X, stop, event = _fitted_prediction_model(backend)
    model._cleanup_cuda_memory = Mock()
    model._cleanup_torch_memory = Mock()
    operations = (
        lambda: model.predict_hazard_ratio(X[:3]),
        lambda: model.predict_risk_score(X[:3]),
        lambda: model.predict_survival(X[:3], times=[1.0, 2.0]),
        lambda: model.predict(X[:3]),
        lambda: model.score(X, stop, event),
    )
    for operation in operations:
        model._cleanup_cuda_memory.reset_mock()
        model._cleanup_torch_memory.reset_mock()
        operation()
        model._cleanup_cuda_memory.assert_called_once_with()
        model._cleanup_torch_memory.assert_called_once_with()


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_coxph_public_prediction_and_score_cleanup_on_error(backend):
    model, X, stop, event = _fitted_prediction_model(backend)
    model._cleanup_cuda_memory = Mock()
    model._cleanup_torch_memory = Mock()
    operations = (
        lambda: model.predict_hazard_ratio(np.ones((2, 2))),
        lambda: model.predict_risk_score(np.ones((2, 2))),
        lambda: model.predict_survival(X[:2], times=np.array([1.0 + 1.0j])),
        lambda: model.predict(np.ones((2, 2))),
        lambda: model.score(X, stop, event[:-1]),
    )
    for operation in operations:
        model._cleanup_cuda_memory.reset_mock()
        model._cleanup_torch_memory.reset_mock()
        with pytest.raises((ValueError, RuntimeError)):
            operation()
        model._cleanup_cuda_memory.assert_called_once_with()
        model._cleanup_torch_memory.assert_called_once_with()


def test_summary_reports_real_matrix_call_metadata(capsys):
    X, stop, event = _sample(seed=802, n=40, p=1)
    start = np.zeros_like(stop)
    strata = np.arange(stop.shape[0]) % 2
    subject_id = np.arange(stop.shape[0])
    model = CoxPH(
        ties="efron", compute_inference=False, compute_cindex=False
    ).fit(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        subject_id=subject_id,
    )
    model.summary()
    output = capsys.readouterr().out
    assert "coxph(formula = Surv(time, event) ~ ." not in output
    assert "interface='matrix'" in output
    assert "ties='efron'" in output
    assert "counting_process=True" in output
    assert "stratified=True" in output
    assert "subject_grouped=True" in output


def test_summary_preserves_exact_formula_call(capsys):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    X, stop, event = _sample(seed=803, n=42, p=1)
    frame = pd.DataFrame(
        {
            "start": np.zeros_like(stop),
            "stop": stop,
            "event": event,
            "x": X[:, 0],
            "group": np.where(np.arange(stop.shape[0]) % 2, "b", "a"),
            "stratum": np.arange(stop.shape[0]) % 2,
        }
    )
    formula = "Surv(start, stop, event) ~ x + C(group) + x:C(group)"
    model = CoxPH(
        ties="breslow", compute_inference=False, compute_cindex=False
    ).fit(
        formula=formula,
        data=frame,
        strata=frame["stratum"].to_numpy(),
    )
    model.summary()
    output = capsys.readouterr().out
    assert f"formula={formula!r}" in output
    assert "counting_process=True" in output
    assert "stratified=True" in output


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_formula_side_arrays_use_one_backend_preserving_alignment_helper(
    backend, monkeypatch,
):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1.0, 0.0, 1.0, 0.0],
            "x": [0.2, np.nan, -0.1, 0.4],
        }
    )
    side_arrays_host = {
        "entry": np.array([0.0, 0.2, 0.3, 0.4]),
        "cluster": np.array([10, 11, 12, 13]),
        "strata": np.array([20, 21, 22, 23]),
        "subject_id": np.array([30, 31, 32, 33]),
    }
    side_arrays = {
        name: _backend_arrays(backend, values)[0]
        for name, values in side_arrays_host.items()
    }
    alignment_calls = []
    original_align = cox_module._align_cox_side_array

    def recording_align(values, retained_rows, original_n, name="array"):
        alignment_calls.append(name)
        return original_align(values, retained_rows, original_n, name)

    monkeypatch.setattr(cox_module, "_align_cox_side_array", recording_align)
    captured = {}
    model = CoxPH(compute_inference=False, compute_cindex=False)

    def recording_dispatch(X, time, event, **kwargs):
        captured.update(kwargs)
        model.coef_ = np.zeros(int(X.shape[1]), dtype=np.float64)
        model._log_likelihood = 0.0
        model._is_counting_process = True
        return model

    monkeypatch.setattr(
        model, "_fit_counting_process_dispatch", recording_dispatch
    )
    model.fit(
        formula="Surv(time, event) ~ x",
        data=frame,
        **side_arrays,
    )

    retained = np.array([0, 2, 3])
    for name, values in side_arrays_host.items():
        actual = captured[name]
        if backend == "cupy":
            assert type(actual).__module__.startswith("cupy")
            actual = actual.get()
        elif backend == "torch":
            assert type(actual).__module__.startswith("torch")
            assert actual.device.type == "cuda"
            actual = actual.detach().cpu().numpy()
        np.testing.assert_array_equal(actual, values[retained])
    assert alignment_calls == [
        "entry/start",
        "cluster",
        "strata",
        "subject_id",
    ]


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_canonical_cox_inference_uses_shared_result_contract(backend):
    X, stop, event = _sample(seed=804, n=72, p=2)
    X, stop, event = _backend_arrays(backend, X, stop, event)
    model = CoxPH(
        device=_device_name(backend),
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-8,
    ).fit(X, stop, event)
    result = model._inference_result
    assert isinstance(result, ParameterInferenceResult)
    np.testing.assert_allclose(model._params, model.coef_)
    np.testing.assert_allclose(result.params, model.coef_)
    np.testing.assert_allclose(result.bse, model._bse)
    np.testing.assert_allclose(result.statistic, model._zvalues)
    np.testing.assert_allclose(model._tvalues, model._zvalues)
    np.testing.assert_allclose(result.pvalues, model._pvalues)
    np.testing.assert_allclose(result.conf_int, model._conf_int)
    from statgpu.inference._distributions_backend import norm

    critical = float(norm.ppf(0.975))
    expected = np.column_stack(
        [model.coef_ - critical * model._bse, model.coef_ + critical * model._bse]
    )
    np.testing.assert_allclose(model._conf_int, expected)
    assert result.cov_type == "nonrobust"
    assert result.distribution == "normal"


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_coxphcv_final_refit_exposes_shared_inference_result(backend):
    X, stop, event = _sample(seed=805, n=48, p=2)
    X, stop, event = _backend_arrays(backend, X, stop, event)
    model = CoxPHCV(
        penalties=np.array([0.05]),
        cv=2,
        random_state=0,
        compute_inference=True,
        max_iter=80,
        tol=1e-7,
        device=_device_name(backend),
    ).fit(X, stop, event)
    assert isinstance(model._inference_result, ParameterInferenceResult)
    np.testing.assert_allclose(model._params, model.coef_)
    np.testing.assert_allclose(model._bse, model.estimator_._bse)
    np.testing.assert_allclose(model._pvalues, model.estimator_._pvalues)


def _backend_arrays(backend, *values):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA unavailable: {exc}")
        return tuple(cp.asarray(value) for value in values)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")
        return tuple(
            torch.as_tensor(value, dtype=torch.float64, device="cuda")
            for value in values
        )
    return tuple(np.asarray(value) for value in values)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize(
    "subject_id",
    [
        np.array([0.1, 0.9, 2.0]),
        np.array([0.0, np.nan, 2.0]),
        np.array([np.iinfo(np.uint64).max, 0, 1], dtype=np.uint64),
        np.array(["a", "b", "c"]),
    ],
)
def test_low_level_subject_id_rejects_invalid_integer_codes(
    backend, subject_id
):
    beta = np.array([0.2])
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0])
    beta_b, X_b, stop_b, event_b = _backend_arrays(
        backend, beta, X, stop, event
    )
    with pytest.raises(ValueError, match="subject_id.*integer-valued"):
        counting_process_concordance(
            beta_b,
            X_b,
            stop_b,
            event_b,
            subject_id=subject_id,
        )


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_low_level_subject_id_accepts_safe_host_uint64(backend):
    beta = np.array([0.2])
    X = np.array([[0.0], [1.0], [2.0]])
    stop = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 0.0])
    beta_b, X_b, stop_b, event_b = _backend_arrays(
        backend, beta, X, stop, event
    )
    value = counting_process_concordance(
        beta_b,
        X_b,
        stop_b,
        event_b,
        subject_id=np.array([0, 1, 2], dtype=np.uint64),
    )
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    assert np.isfinite(float(value))


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
def test_ordinary_concordance_tile_accumulators_sync_once(
    backend, monkeypatch
):
    X, stop, event = _sample(seed=806, n=36, p=2)
    X, stop, event = _backend_arrays(backend, X, stop, event)
    model = CoxPH(
        device=_device_name(backend),
        compute_inference=False,
        compute_cindex=False,
    ).fit(X, stop, event)
    expected = model.score(X, stop, event)
    calls = []
    original = cox_score_module._sync_scalars

    def recording_sync(*values, backend):
        calls.append((len(values), backend))
        return original(*values, backend=backend)

    monkeypatch.setattr(
        cox_score_module,
        "_concordance_tile_shape",
        lambda _n_events, _n_samples: (1, 2),
    )
    monkeypatch.setattr(cox_score_module, "_sync_scalars", recording_sync)
    actual = model.score(X, stop, event)
    assert actual == pytest.approx(expected, abs=1e-15)
    assert calls == [(3, backend)]


def test_public_fit_isolated_from_legacy_reference_methods(monkeypatch):
    X, stop, event = _sample(seed=807, n=36, p=2)
    model = CoxPH(compute_inference=True, compute_cindex=False)

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("public fit reached a legacy Cox implementation")

    for name, value in vars(_LegacyCoxReferenceMixin).items():
        if callable(value):
            monkeypatch.setattr(
                _LegacyCoxReferenceMixin, name, reject_legacy
            )
    model.fit(X, stop, event)
    assert np.all(np.isfinite(model._bse))
    assert model._canonical_fit_path == "counting_process"
    source = inspect.getsource(CoxPH._fit_counting_process_dispatch)
    assert "import cupy" not in source
    assert "import torch" not in source


def test_legacy_reference_methods_live_only_in_composition_adapter():
    assert _LegacyCoxReferenceMixin not in CoxPH.__mro__
    legacy_methods = tuple(
        name
        for name, value in vars(_LegacyCoxReferenceMixin).items()
        if callable(value)
    )
    assert all(not hasattr(CoxPH, name) for name in legacy_methods)
    reference = _LegacyCoxReference(CoxPH(compute_inference=False))
    assert isinstance(reference, _LegacyCoxReferenceMixin)

    canonical_source = inspect.getsource(cox_module)
    assert "_cox_legacy" not in canonical_source
    assert "def _fit_cpu(" not in canonical_source
    assert "def _fit_gpu(" not in canonical_source
    assert "def _fit_torch(" not in canonical_source
    assert CoxPH._fit_counting_process_dispatch.__module__ == (
        "statgpu.survival._cox"
    )


def test_public_survival_import_does_not_load_legacy_module():
    code = (
        "import sys; from statgpu.survival import CoxPH; "
        "assert 'statgpu.survival._cox_legacy' not in sys.modules; "
        "assert CoxPH.__module__ == 'statgpu.survival._cox'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
