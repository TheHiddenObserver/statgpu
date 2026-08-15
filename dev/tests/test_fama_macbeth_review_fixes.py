"""Regression coverage for Fama-MacBeth formula, chronology, and rank contracts."""

from __future__ import annotations

import numpy as np
import pytest

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
