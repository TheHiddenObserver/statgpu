"""Regression coverage for Fama-MacBeth formula, chronology, rank, and inference."""

from __future__ import annotations

import numpy as np
import pytest

import statgpu.panel._linalg as panel_linalg
from dev.benchmarks import validate_fama_macbeth_review_fix_gpu as fmb_gpu_gate
from statgpu.inference._results import ParameterInferenceResult
from statgpu.panel import FamaMacBeth


def _chronology_fixture():
    x_period = np.asarray([-2.0, -0.75, 0.25, 1.25, 2.5])
    # Keep the three period coefficients non-collinear so reordering t2/t10
    # changes the lag-1 covariance and the chronology test has real power.
    period_params = ((0.2, 0.4), (1.1, -0.8), (-0.7, 0.3))
    x = np.tile(x_period, len(period_params))
    y = np.concatenate(
        [intercept + slope * x_period for intercept, slope in period_params]
    )
    labels = np.repeat(np.asarray(["t1", "t2", "t10"], dtype=object), x_period.size)
    numeric = np.repeat(np.arange(len(period_params)), x_period.size)
    return x, y, labels, numeric


def _ordered_time(labels):
    pd = pytest.importorskip("pandas")
    return pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _assert_standard_inference_surface(
    model,
    *,
    statistic_name,
    distribution,
    df,
    feature_names=None,
):
    assert isinstance(model._inference_result, ParameterInferenceResult)
    result = model._inference_result
    assert result.method == "fama_macbeth"
    assert result.statistic_name == statistic_name
    assert result.distribution == distribution
    assert result.cov_type == model._cov_type
    assert result.df == df
    expected_names = None if feature_names is None else list(feature_names)
    actual_names = None if result.feature_names is None else list(result.feature_names)
    assert actual_names == expected_names
    assert result.metadata["covariance_source"] == "period_coefficient_series"
    assert result.metadata["n_periods"] == model.n_periods
    expected_bandwidth = model.bandwidth
    if model._cov_type == "newey-west" and expected_bandwidth is None:
        expected_bandwidth = int(np.floor(4.0 * (model.n_periods / 100.0) ** (2.0 / 9.0)))
        expected_bandwidth = max(0, min(expected_bandwidth, model.n_periods - 1))
    if model._cov_type == "nonrobust":
        expected_bandwidth = None
    assert result.metadata["effective_bandwidth"] == expected_bandwidth

    public_pairs = (
        (model.coef_, model._params),
        (model.bse_, model._bse),
        (model.tvalues_, model._tvalues),
        (model.pvalues_, model._pvalues),
        (model.conf_int_, model._conf_int),
    )
    for public, internal in public_pairs:
        np.testing.assert_allclose(_to_numpy(public), np.asarray(internal), rtol=0, atol=0)
    np.testing.assert_allclose(
        _to_numpy(model.tvalues_), np.asarray(model._zvalues), rtol=0, atol=0
    )
    np.testing.assert_allclose(result.params, model._params, rtol=0, atol=0)
    np.testing.assert_allclose(result.bse, model._bse, rtol=0, atol=0)
    np.testing.assert_allclose(result.statistic, model._tvalues, rtol=0, atol=0)
    np.testing.assert_allclose(result.pvalues, model._pvalues, rtol=0, atol=0)
    np.testing.assert_allclose(result.conf_int, model._conf_int, rtol=0, atol=0)


@pytest.mark.parametrize("formula", ["y ~ 0 + x", "y ~ x - 1"])
def test_fama_macbeth_formula_rejects_explicit_no_intercept(formula):
    pd = pytest.importorskip("pandas")
    x, y, _labels, numeric = _chronology_fixture()
    data = pd.DataFrame({"y": y, "x": x})

    with pytest.raises(
        ValueError,
        match="always includes a period intercept.*no-intercept formulas are not supported",
    ):
        FamaMacBeth(device="cpu").fit(
            formula=formula,
            data=data,
            time_ids=numeric,
        )


def test_fama_macbeth_ordered_categorical_chronology_matches_numeric_array():
    x, y, labels, numeric = _chronology_fixture()
    ordered = _ordered_time(labels)

    expected = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=numeric
    )
    actual = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=ordered
    )
    lexical = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=np.asarray(ordered, dtype=object)
    )

    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.bse_, expected.bse_, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(
        actual.cov_params_, expected.cov_params_, rtol=2e-13, atol=2e-15
    )
    assert not np.allclose(actual.cov_params_, lexical.cov_params_, rtol=1e-10, atol=1e-12)
    _assert_standard_inference_surface(
        actual,
        statistic_name="z",
        distribution="normal",
        df=None,
    )


def test_fama_macbeth_ordered_categorical_chronology_survives_formula_alignment():
    pd = pytest.importorskip("pandas")
    x, y, labels, numeric = _chronology_fixture()
    ordered = _ordered_time(labels)
    x_with_gap = x.copy()
    x_with_gap[1] = np.nan
    data = pd.DataFrame({"y": y, "x": x_with_gap})

    expected = FamaMacBeth(bandwidth=1, device="cpu").fit(
        formula="y ~ x",
        data=data,
        time_ids=numeric,
    )
    actual = FamaMacBeth(bandwidth=1, device="cpu").fit(
        formula="y ~ x",
        data=data,
        time_ids=ordered,
    )
    lexical = FamaMacBeth(bandwidth=1, device="cpu").fit(
        formula="y ~ x",
        data=data,
        time_ids=np.asarray(ordered, dtype=object),
    )

    assert actual._feature_names == ["Intercept", "x"]
    np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual.bse_, expected.bse_, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(
        actual.cov_params_, expected.cov_params_, rtol=2e-13, atol=2e-15
    )
    assert not np.allclose(actual.cov_params_, lexical.cov_params_, rtol=1e-10, atol=1e-12)
    _assert_standard_inference_surface(
        actual,
        statistic_name="z",
        distribution="normal",
        df=None,
        feature_names=["Intercept", "x"],
    )
    summary = actual.summary().to_dict()
    assert summary["feature_names"] == ["Intercept", "x"]
    np.testing.assert_allclose(summary["coef"], actual._params, rtol=0, atol=0)
    np.testing.assert_allclose(summary["bse"], actual._bse, rtol=0, atol=0)
    np.testing.assert_allclose(summary["pvalues"], actual._pvalues, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cov_type,statistic_name,distribution,expected_df",
    [
        ("newey-west", "z", "normal", None),
        ("nonrobust", "t", "t", 2.0),
    ],
)
def test_fama_macbeth_standard_inference_result_numpy(
    cov_type, statistic_name, distribution, expected_df
):
    x, y, _labels, numeric = _chronology_fixture()
    model = FamaMacBeth(cov_type=cov_type, bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=numeric
    )
    _assert_standard_inference_surface(
        model,
        statistic_name=statistic_name,
        distribution=distribution,
        df=expected_df,
    )


def test_fama_macbeth_standard_inference_result_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    x, y, _labels, numeric = _chronology_fixture()
    expected = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=numeric
    )
    actual = FamaMacBeth(bandwidth=1).fit(
        torch.as_tensor(x[:, None], dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(numeric, dtype=torch.int64),
    )
    _assert_standard_inference_surface(
        actual,
        statistic_name="z",
        distribution="normal",
        df=None,
    )
    for name in ("coef_", "bse_", "tvalues_", "pvalues_", "conf_int_"):
        np.testing.assert_allclose(
            _to_numpy(getattr(actual, name)),
            _to_numpy(getattr(expected, name)),
            rtol=2e-12,
            atol=2e-14,
        )


def test_fama_macbeth_reuses_one_rank_revealing_svd_per_retained_period(monkeypatch):
    x, y, _labels, numeric = _chronology_fixture()
    calls = []
    original = panel_linalg._svd_inverse_factors

    def tracked(X, xp):
        calls.append((int(X.shape[0]), int(X.shape[1])))
        return original(X, xp)

    monkeypatch.setattr(panel_linalg, "_svd_inverse_factors", tracked)
    model = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=numeric
    )

    assert model.n_periods == 3
    assert calls == [(5, 2), (5, 2), (5, 2)]


def test_panel_lstsq_keeps_cutoff_backend_native_and_extracts_only_rank(monkeypatch):
    X = np.asarray(
        [
            [1.0, -2.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.5],
            [1.0, 2.5],
        ]
    )
    y = np.asarray([-0.5, 0.1, 0.8, 1.6, 2.4])
    scalar_extractions = []
    original = panel_linalg._to_float_scalar

    def tracked(value):
        scalar_extractions.append(value)
        return original(value)

    monkeypatch.setattr(panel_linalg, "_to_float_scalar", tracked)
    params, rank = panel_linalg.panel_lstsq(X, y, np)

    expected = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(params, expected, rtol=1e-13, atol=1e-14)
    assert rank == 2
    # The singular-value maximum remains an ndarray scalar on the active
    # backend; only the final integer rank crosses the backend boundary.
    assert len(scalar_extractions) == 1


def test_fama_macbeth_focused_gpu_runner_contract_executes_numpy_timing_case():
    assert fmb_gpu_gate.SCHEMA_VERSION == 3
    assert fmb_gpu_gate._validate_acceptance_backends(["cupy", "torch"]) == [
        "cupy",
        "torch",
    ]
    with pytest.raises(ValueError, match="requires exactly both GPU backends"):
        fmb_gpu_gate._validate_acceptance_backends(["cupy"])

    X, y, time_ids = fmb_gpu_gate._timing_fixture()
    assert X.shape == (64 * 128, 4)
    assert y.shape == (64 * 128,)
    assert time_ids.shape == (64 * 128,)

    result = fmb_gpu_gate._timing_case("numpy", warmup=0, repeats=1)
    assert result["status"] == "success"
    assert result["executed_backend"] == "numpy"
    assert result["n_times"] == 64
    assert result["observations_per_period"] == 128
    assert result["n_features"] == 4
    assert result["numpy_baseline"]["median_seconds"] > 0.0
    assert result["backend_timing"]["median_seconds"] > 0.0
    assert len(result["numpy_baseline"]["samples_seconds"]) == 1
    assert len(result["backend_timing"]["samples_seconds"]) == 1
    assert result["backend_over_numpy_median_ratio"] == pytest.approx(1.0)
    assert set(result["max_abs_differences_vs_numpy"]) == set(
        fmb_gpu_gate._SNAPSHOT_KEYS
    )
    assert all(
        value == pytest.approx(0.0)
        for value in result["max_abs_differences_vs_numpy"].values()
    )
    assert "remaining_structure" in result["optimization_notes"]
    assert "interpretation" in result["optimization_notes"]


def _rank_deficient_fixture():
    x = np.concatenate(
        [
            np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.ones(5),
            np.asarray([-1.5, -0.5, 0.5, 1.5, 2.5]),
        ]
    )
    time = np.repeat(np.arange(3), 5)
    y = 0.5 + 0.8 * x + np.repeat(np.asarray([0.0, 0.4, -0.3]), 5)
    return x[:, None], y, time


def test_fama_macbeth_rejects_rank_deficient_retained_period_numpy():
    X, y, time = _rank_deficient_fixture()

    with pytest.raises(
        ValueError,
        match=r"full column rank.*retained time period.*rank deficient.*rank=1, columns=2",
    ):
        FamaMacBeth(device="cpu").fit(X, y, time_ids=time)


def test_fama_macbeth_rejects_rank_deficient_retained_period_torch_cpu():
    torch = pytest.importorskip("torch")
    X, y, time = _rank_deficient_fixture()
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    time_t = torch.as_tensor(time, dtype=torch.int64)

    with pytest.raises(
        ValueError,
        match=r"full column rank.*retained time period.*rank deficient.*rank=1, columns=2",
    ):
        FamaMacBeth().fit(X_t, y_t, time_ids=time_t)
