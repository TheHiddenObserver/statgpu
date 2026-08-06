"""Strict independent-unit contracts for canonical Cox robust inference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.inference._covariance import classify_covariance_spectrum
from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival import _cox as cox_module
from statgpu.survival._cox_inference import (
    _joint_wald_from_covariance,
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
    assert model.wald_test_available_ is False
    assert model.wald_test_failure_reason_ == "compute_inference=False"


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
    for model in (hc0, hc1):
        assert model.wald_test_available_ is True
        assert model.wald_test_failure_reason_ is None
        assert np.isfinite(model._wald_test_stat)
        assert np.isfinite(model._wald_test_pvalue)


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize(
    ("cov_type", "n_units", "label_name"),
    [("cluster", 2, "cluster"), ("hc0", 3, "subject_id")],
)
def test_rank_deficient_robust_covariance_keeps_marginal_inference(
    backend_name, cov_type, n_units, label_name
):
    X, stop, event = _sample(seed=9344, p=3)
    labels = np.arange(X.shape[0], dtype=np.int64) % n_units
    device, (Xb, stopb, eventb, labelsb) = _backend_inputs(
        backend_name, X, stop, event, labels
    )
    model = CoxPH(
        device=device,
        cov_type=cov_type,
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    ).fit(Xb, stopb, eventb, **{label_name: labelsb})

    assert model._fitted is True
    assert np.all(np.isfinite(model._bse))
    assert np.all(model._bse > 0.0)
    assert np.all(np.isfinite(model._pvalues))
    assert model._inference_result is not None
    assert model.wald_test_available_ is False
    assert model.wald_test_failure_reason_ == (
        "robust covariance is rank-deficient for the full-parameter Wald test"
    )
    assert np.isnan(model._wald_test_stat)
    assert np.isnan(model._wald_test_pvalue)
    assert model._inference_result.metadata["joint_wald_available"] is False
    assert (
        model._inference_result.metadata["covariance_spectrum"]
        == "rank_deficient_psd"
    )
    assert (
        model._inference_result.metadata["joint_wald_failure_reason"]
        == model.wald_test_failure_reason_
    )


def test_rank_deficient_robust_summary_labels_joint_test_unavailable(capsys):
    X, stop, event = _sample(seed=9344, p=3)
    cluster = np.arange(X.shape[0], dtype=np.int64) % 2
    model = CoxPH(
        cov_type="cluster",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    ).fit(X, stop, event, cluster=cluster)

    model.summary()
    output = capsys.readouterr().out
    assert "Classical likelihood-ratio test:" in output
    assert (
        "Robust Wald test unavailable: robust covariance is rank-deficient"
        in output
    )
    assert "Classical score (logrank) test:" in output
    assert "Wald test: nan" not in output


def test_coxphcv_propagates_joint_wald_unavailability_from_final_refit():
    X, stop, event = _sample(seed=9344, p=3)
    cluster = np.arange(X.shape[0], dtype=np.int64) % 2
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        cov_type="cluster",
        compute_inference=True,
        device="cpu",
        max_iter=60,
        tol=1e-8,
    ).fit(X, stop, event, cluster=cluster)

    assert model._fitted is True
    assert model.estimator_ is not None
    assert np.all(np.isfinite(model._bse))
    assert model.wald_test_available_ is False
    assert model.wald_test_failure_reason_ == (
        "robust covariance is rank-deficient for the full-parameter Wald test"
    )
    assert model.estimator_.wald_test_available_ is False


def test_joint_wald_helper_rejects_near_rank_deficiency_without_losing_marginals():
    near_singular = np.diag([1.0, 0.5, 1e-14])
    spectrum = classify_covariance_spectrum(near_singular)
    assert spectrum.classification == "rank_deficient_psd"
    statistic, failure = _joint_wald_from_covariance(
        np.ones(3),
        near_singular,
        cov_type="hc0",
        spectrum=spectrum,
    )
    assert np.isnan(statistic)
    assert failure == (
        "robust covariance is rank-deficient for the full-parameter Wald test"
    )

    statistic, failure = _joint_wald_from_covariance(
        np.array([1.0, 2.0]),
        np.diag([2.0, 4.0]),
        cov_type="nonrobust",
    )
    assert statistic == pytest.approx(1.5)
    assert failure is None

    with pytest.raises(RuntimeError, match="not positive semidefinite"):
        _joint_wald_from_covariance(
            np.ones(2),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            cov_type="cluster",
        )


def test_covariance_spectrum_distinguishes_psd_roundoff_from_indefinite():
    rank_deficient = np.array([[1.0, 1.0], [1.0, 1.0]])
    rank_spectrum = classify_covariance_spectrum(rank_deficient)
    assert rank_spectrum.classification == "rank_deficient_psd"
    assert np.array_equal(
        _standard_errors_from_covariance(
            rank_deficient,
            cov_type="hc0",
            spectrum=rank_spectrum,
        ),
        np.ones(2),
    )

    roundoff_indefinite = np.array(
        [[1.0, 1.0 + 1e-14], [1.0 + 1e-14, 1.0]]
    )
    roundoff_spectrum = classify_covariance_spectrum(roundoff_indefinite)
    assert roundoff_spectrum.classification == "rank_deficient_psd"
    statistic, reason = _joint_wald_from_covariance(
        np.ones(2),
        roundoff_indefinite,
        cov_type="cluster",
        spectrum=roundoff_spectrum,
    )
    assert np.isnan(statistic)
    assert "rank-deficient" in reason

    materially_indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    indefinite_spectrum = classify_covariance_spectrum(materially_indefinite)
    assert indefinite_spectrum.classification == "materially_indefinite"
    with pytest.raises(RuntimeError, match="not positive semidefinite"):
        _standard_errors_from_covariance(
            materially_indefinite,
            cov_type="cluster",
            spectrum=indefinite_spectrum,
        )


@pytest.mark.parametrize("backend_name", ["numpy", "cupy", "torch"])
def test_materially_indefinite_covariance_fails_and_clears_cox_state(
    monkeypatch, backend_name
):
    X, stop, event = _sample(seed=9350, p=2)
    cluster = np.arange(X.shape[0], dtype=np.int64) % 4
    device, (Xb, stopb, eventb, clusterb) = _backend_inputs(
        backend_name, X, stop, event, cluster
    )
    forced_spectrum = classify_covariance_spectrum(
        np.array([[1.0, 2.0], [2.0, 1.0]])
    )
    monkeypatch.setattr(
        cox_module,
        "_classify_covariance_spectrum",
        lambda _covariance: forced_spectrum,
    )
    model = CoxPH(
        device=device,
        cov_type="cluster",
        compute_inference=True,
        compute_cindex=False,
        max_iter=80,
    )
    with pytest.raises(RuntimeError, match="not positive semidefinite"):
        model.fit(Xb, stopb, eventb, cluster=clusterb)
    assert model._fitted is False
    assert model.coef_ is None
    assert model._bse is None
    assert model._inference_result is None


def test_materially_indefinite_covariance_clears_coxphcv_final_refit(
    monkeypatch,
):
    X, stop, event = _sample(seed=9351, p=2)
    cluster = np.arange(X.shape[0], dtype=np.int64) % 4
    forced_spectrum = classify_covariance_spectrum(
        np.array([[1.0, 2.0], [2.0, 1.0]])
    )
    monkeypatch.setattr(
        cox_module,
        "_classify_covariance_spectrum",
        lambda _covariance: forced_spectrum,
    )
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        cov_type="cluster",
        compute_inference=True,
        device="cpu",
        max_iter=60,
    )
    with pytest.raises(RuntimeError, match="not positive semidefinite"):
        model.fit(X, stop, event, cluster=cluster)
    assert model._fitted is False
    assert model.estimator_ is None
    assert model.coef_ is None
    assert model._inference_result is None


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
    unsupported = benchmark_cox_cluster.external_covariance_contract_fields(
        supported=False,
        requested_contract="cluster score sandwich",
        unsupported_reason="external solver failed",
    )
    assert unsupported == {
        "covariance_contract": "unsupported",
        "requested_covariance_contract": "cluster score sandwich",
        "unsupported_reason": "external solver failed",
    }
    supported = benchmark_cox_cluster.external_covariance_contract_fields(
        supported=True,
        requested_contract="requested",
        actual_contract="actual",
    )
    assert supported == {
        "covariance_contract": "actual",
        "requested_covariance_contract": "requested",
        "unsupported_reason": "",
    }


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
    with pytest.raises(ValueError, match="exactly 2 values"):
        benchmark_cox_cluster.validate_external_vector(
            np.array([0.1]), 2, name="truncated"
        )
    with pytest.raises(ValueError, match="only finite"):
        benchmark_cox_cluster.validate_external_vector(
            np.array([0.1, np.nan]), 2, name="nonfinite"
        )
    with pytest.raises(ValueError, match="identical shapes"):
        benchmark_cox_cluster.safe_diff(
            np.array([0.1, 0.2]), np.array([0.1])
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
    result = benchmark_cox_cluster.run_r(
        tmp_path / "data.csv",
        "efron",
        "hc1",
        n_features=2,
        max_iter=77,
        tol=1e-9,
    )

    r_source = recorded["command"][2]
    assert "robust=TRUE" in r_source
    assert "n_units / (n_units - p)" in r_source
    assert "iter.max=77" in r_source
    assert "eps=1e-09" in r_source
    assert "timefix=FALSE" in r_source
    assert result["supported"] is True
    assert result["n_units"] == 8
    assert result["correction"] == pytest.approx(1.6)


def test_r_helper_rejects_truncated_or_nonfinite_vectors(monkeypatch, tmp_path):
    outputs = iter(
        [
            (
                "FIT_MS=1\nN_UNITS=8\nCORRECTION=1.6\n"
                "COEF=1.0\nBSE=0.5,0.25\nPVALUES=0.1,0.2\n"
            ),
            (
                "FIT_MS=1\nN_UNITS=8\nCORRECTION=1.6\n"
                "COEF=1.0,-2.0\nBSE=nan,0.25\nPVALUES=0.1,0.2\n"
            ),
        ]
    )

    def fake_run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(benchmark_cox_cluster.shutil, "which", lambda _: "Rscript")
    monkeypatch.setattr(benchmark_cox_cluster.subprocess, "run", fake_run)
    for expected in ("exactly 2 values", "only finite"):
        result = benchmark_cox_cluster.run_r(
            tmp_path / "data.csv",
            "breslow",
            "cluster",
            n_features=2,
            max_iter=80,
            tol=1e-8,
        )
        assert result["supported"] is False
        assert expected in result["error"]
