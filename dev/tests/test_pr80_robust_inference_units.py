"""Strict independent-unit contracts for canonical Cox robust inference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.survival import CoxPH
from statgpu.survival._cox_inference import (
    _standard_errors_from_covariance,
)
from dev.benchmarks import benchmark_cox_cluster


def _sample(seed=9341, n=48, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.3, -0.2, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=2.0, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[: max(6, p + 2)] = 1.0
    return X, stop, event


def _backend_inputs(backend_name, *values):
    if backend_name == "numpy":
        return "cpu", tuple(np.asarray(value) for value in values)
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device is unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA device is unavailable: {exc}")
        return "cuda", tuple(cp.asarray(value) for value in values)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")
    converted = []
    for value in values:
        array = np.asarray(value)
        dtype = torch.float64 if array.dtype.kind == "f" else torch.int64
        converted.append(torch.as_tensor(array, dtype=dtype, device="cuda"))
    return "torch", tuple(converted)


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_single_cluster_is_rejected_before_sandwich(backend_name):
    X, stop, event = _sample(p=2)
    cluster = np.zeros(X.shape[0], dtype=np.int64)
    device, (Xb, stopb, eventb, clusterb) = _backend_inputs(
        backend_name, X, stop, event, cluster
    )
    model = CoxPH(
        device=device,
        cov_type="cluster",
        compute_inference=True,
        compute_cindex=False,
    )
    with pytest.raises(
        RuntimeError, match="cluster covariance requires at least two"
    ):
        model.fit(Xb, stopb, eventb, cluster=clusterb)
    assert model.coef_ is None
    assert model._fitted is False


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_single_cluster_remains_valid_for_estimation_only(backend_name):
    X, stop, event = _sample(seed=9345, p=2)
    cluster = np.zeros(X.shape[0], dtype=np.int64)
    device, (Xb, stopb, eventb, clusterb) = _backend_inputs(
        backend_name, X, stop, event, cluster
    )
    model = CoxPH(
        device=device,
        cov_type="cluster",
        compute_inference=False,
        compute_cindex=False,
    ).fit(Xb, stopb, eventb, cluster=clusterb)
    assert model._fitted is True
    assert model._bse is None
    assert np.all(np.isfinite(model.coef_))


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize("cov_type", ["hc0", "hc1"])
def test_single_subject_is_rejected_before_sandwich(backend_name, cov_type):
    X, stop, event = _sample(seed=9342, p=2)
    subject = np.zeros(X.shape[0], dtype=np.int64)
    device, (Xb, stopb, eventb, subjectb) = _backend_inputs(
        backend_name, X, stop, event, subject
    )
    model = CoxPH(
        device=device,
        cov_type=cov_type,
        compute_inference=True,
        compute_cindex=False,
    )
    with pytest.raises(
        RuntimeError, match=rf"{cov_type} covariance requires at least two"
    ):
        model.fit(Xb, stopb, eventb, subject_id=subjectb)


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_hc1_rejects_n_units_equal_to_n_features(backend_name):
    X, stop, event = _sample(seed=9343, p=3)
    subject = np.arange(X.shape[0], dtype=np.int64) % X.shape[1]
    device, (Xb, stopb, eventb, subjectb) = _backend_inputs(
        backend_name, X, stop, event, subject
    )
    with pytest.raises(
        RuntimeError, match="HC1 covariance requires n_units > n_features"
    ):
        CoxPH(
            device=device,
            cov_type="hc1",
            compute_inference=True,
            compute_cindex=False,
        ).fit(Xb, stopb, eventb, subject_id=subjectb)


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_hc1_accepts_p_plus_one_units_with_positive_standard_errors(
    backend_name,
):
    X, stop, event = _sample(seed=9344, p=3)
    subject = np.arange(X.shape[0], dtype=np.int64) % (X.shape[1] + 1)
    device, (Xb, stopb, eventb, subjectb) = _backend_inputs(
        backend_name, X, stop, event, subject
    )
    common = dict(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    )
    hc0 = CoxPH(cov_type="hc0", **common).fit(
        Xb, stopb, eventb, subject_id=subjectb
    )
    hc1 = CoxPH(cov_type="hc1", **common).fit(
        Xb, stopb, eventb, subject_id=subjectb
    )
    assert np.all(np.isfinite(hc1._bse))
    assert np.all(hc1._bse > 0.0)
    assert np.all(np.isfinite(hc1._pvalues))
    assert np.allclose(hc1._var_matrix, 4.0 * hc0._var_matrix, rtol=2e-8, atol=2e-10)


def test_covariance_diagonal_rejects_material_negative_and_zero_robust_variance():
    with pytest.raises(RuntimeError, match="materially negative diagonal"):
        _standard_errors_from_covariance(
            np.diag([1.0, -1e-4]), cov_type="hc0"
        )
    with pytest.raises(RuntimeError, match="non-positive marginal variance"):
        _standard_errors_from_covariance(
            np.diag([1.0, 0.0]), cov_type="cluster"
        )
    roundoff = _standard_errors_from_covariance(
        np.diag([1.0, -1e-15]), cov_type="nonrobust"
    )
    assert np.array_equal(roundoff, np.array([1.0, 0.0]))


def test_statsmodels_hc1_is_explicitly_unsupported():
    capability = benchmark_cox_cluster.statsmodels_covariance_capability("hc1")
    assert capability["supported"] is False
    assert "n_units/(n_units-p)" in capability["reason"]
    assert benchmark_cox_cluster.json_ready(np.nan) is None


def test_statsmodels_nonfinite_inference_is_not_reported_as_supported():
    finite = SimpleNamespace(
        params=np.array([0.1, -0.2]),
        bse=np.array([0.3, 0.4]),
        pvalues=np.array([0.7, 0.6]),
    )
    nonfinite = SimpleNamespace(
        params=np.array([0.1, -0.2]),
        bse=np.array([np.nan, np.nan]),
        pvalues=np.array([np.nan, np.nan]),
    )
    assert benchmark_cox_cluster.statsmodels_result_has_finite_inference(
        finite, 2
    )
    assert not benchmark_cox_cluster.statsmodels_result_has_finite_inference(
        nonfinite, 2
    )


def test_r_hc1_helper_applies_explicit_finite_unit_correction(monkeypatch, tmp_path):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "FIT_MS=12.5\nN_UNITS=8\nCORRECTION=1.6\n"
                "COEF=1.0,-2.0\nBSE=0.5,0.25\nPVALUES=0.1,0.2\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark_cox_cluster.shutil, "which", lambda _: "Rscript")
    monkeypatch.setattr(benchmark_cox_cluster.subprocess, "run", fake_run)
    result = benchmark_cox_cluster.run_r(tmp_path / "data.csv", "efron", "hc1")

    r_source = recorded["command"][2]
    assert "robust=TRUE" in r_source
    assert "n_units / (n_units - p)" in r_source
    assert result["supported"] is True
    assert result["n_units"] == 8
    assert result["correction"] == pytest.approx(1.6)
