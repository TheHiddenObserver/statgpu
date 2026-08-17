"""Maintained Fama-MacBeth covariance/inference matrix coverage."""

from __future__ import annotations

import numpy as np
import pytest

import statgpu.inference._reference_distribution as reference_distribution
from statgpu.inference._results import ParameterInferenceResult
from statgpu.panel import FamaMacBeth


def _fixture():
    x_period = np.asarray([-2.0, -0.75, 0.25, 1.25, 2.5])
    period_params = ((0.2, 0.4), (1.1, -0.8), (-0.7, 0.3))
    x = np.tile(x_period, len(period_params))[:, None]
    y = np.concatenate(
        [intercept + slope * x_period for intercept, slope in period_params]
    )
    time_ids = np.repeat(np.arange(len(period_params)), x_period.size)
    return x, y, time_ids


def _to_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


@pytest.mark.parametrize(
    "cov_type,statistic_name,distribution,expected_df",
    [
        ("newey-west", "z", "normal", None),
        ("nonrobust", "t", "t", 2.0),
    ],
)
def test_fama_macbeth_torch_cpu_covariance_inference_matrix(
    cov_type, statistic_name, distribution, expected_df
):
    torch = pytest.importorskip("torch")
    X, y, time_ids = _fixture()

    expected = FamaMacBeth(cov_type=cov_type, bandwidth=1, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    actual = FamaMacBeth(cov_type=cov_type, bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time_ids, dtype=torch.int64),
    )

    assert actual._backend_name == "torch"
    assert actual._inference_backend_name == "torch"
    assert isinstance(actual._inference_result, ParameterInferenceResult)
    result = actual._inference_result
    assert result.method == "fama_macbeth"
    assert result.cov_type == cov_type
    assert result.statistic_name == statistic_name
    assert result.distribution == distribution
    assert result.df == expected_df
    assert result.metadata["covariance_source"] == "period_coefficient_series"
    assert result.metadata["n_periods"] == 3
    assert result.metadata["effective_bandwidth"] == (
        1 if cov_type == "newey-west" else None
    )
    assert result.metadata["inference_backend"] == "torch"

    for public_name, private_name in (
        ("coef_", "_params"),
        ("bse_", "_bse"),
        ("tvalues_", "_tvalues"),
        ("pvalues_", "_pvalues"),
        ("conf_int_", "_conf_int"),
    ):
        public = _to_numpy(getattr(actual, public_name))
        private = np.asarray(getattr(actual, private_name))
        np.testing.assert_allclose(public, private, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            public,
            _to_numpy(getattr(expected, public_name)),
            rtol=2e-12,
            atol=2e-14,
        )

    np.testing.assert_allclose(
        _to_numpy(actual.tvalues_), np.asarray(actual._zvalues), rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    "cov_type,expected_distribution_name,expected_pvalue_calls",
    [
        ("newey-west", "norm", 1),
        # The fixture has T=3, hence df=2. The centralized inference helper uses
        # the exact backend-native tail identity instead of approximate beta
        # numerics on Torch versions without native betainc.
        ("nonrobust", None, 0),
    ],
)
def test_fama_macbeth_torch_cpu_distribution_inference_is_vectorized_and_backend_native(
    monkeypatch, cov_type, expected_distribution_name, expected_pvalue_calls
):
    """Distribution inference must not scalar-sync through a NumPy/SciPy path."""
    torch = pytest.importorskip("torch")
    X, y, time_ids = _fixture()
    original_get_distribution = reference_distribution.get_distribution
    pvalue_input_shapes = []
    distribution_calls = []

    class CountingDistribution:
        def __init__(self, base):
            self._base = base

        def two_sided_pvalue(self, values, *args):
            pvalue_input_shapes.append(tuple(values.shape))
            return self._base.two_sided_pvalue(values, *args)

        def two_sided_critical_value(self, *args):
            return self._base.two_sided_critical_value(*args)

        def sf(self, values, *args):
            pvalue_input_shapes.append(tuple(values.shape))
            return self._base.sf(values, *args)

        def isf(self, *args):
            return self._base.isf(*args)

    def tracked_get_distribution(name, backend="numpy", device=None):
        distribution_calls.append((name, backend, device))
        return CountingDistribution(
            original_get_distribution(name, backend=backend, device=device)
        )

    monkeypatch.setattr(
        reference_distribution,
        "get_distribution",
        tracked_get_distribution,
    )

    model = FamaMacBeth(cov_type=cov_type, bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time_ids, dtype=torch.int64),
    )

    assert model._backend_name == "torch"
    assert model._inference_backend_name == "torch"
    assert model._inference_result.metadata["inference_backend"] == "torch"
    assert len(pvalue_input_shapes) == expected_pvalue_calls
    if expected_pvalue_calls:
        assert pvalue_input_shapes == [(2,)]
        assert distribution_calls == [(expected_distribution_name, "torch", "cpu")]
    else:
        assert pvalue_input_shapes == []
        assert distribution_calls == []
    assert np.all(np.isfinite(_to_numpy(model.pvalues_)))


def test_exact_t2_tail_remains_nonzero_for_extreme_torch_statistic():
    """Use the rationalized exact t(2) tail rather than a cancelling subtraction."""
    torch = pytest.importorskip("torch")
    statistic = torch.as_tensor([1.0e10], dtype=torch.float64)
    pvalues, critical = reference_distribution.two_sided_reference_inference(
        statistic,
        distribution="t",
        alpha=0.05,
        backend="torch",
        xp=torch,
        df=2,
        device="cpu",
    )

    root = np.sqrt(1.0e20 + 2.0)
    expected = 2.0 / (root * (root + 1.0e10))
    observed = float(pvalues.item())
    assert observed > 0.0
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=0.0)
    assert torch.is_tensor(critical)
    assert critical.device.type == "cpu"



def _large_common_intercept_fixture(n_periods=4):
    x_period = np.asarray([-1.0, 0.0, 1.0])
    X = np.tile(x_period, n_periods)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time_ids = np.repeat(np.arange(n_periods), x_period.size)
    return X, y, time_ids


def test_fama_macbeth_torch_gram_certificate_rejects_nonfinite_rhs_and_falls_back():
    torch = pytest.importorskip("torch")
    X, y, time_ids = _large_common_intercept_fixture()
    expected = FamaMacBeth(bandwidth=0, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    actual = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time_ids, dtype=torch.int64),
    )

    # X' y overflows in the Gram fast path even though the SVD projection and
    # the true period coefficients are finite. Every period must therefore be
    # routed to the rank-revealing fallback rather than certified with Inf beta.
    assert actual._backend_name == "torch"
    assert actual._period_svd_fallbacks == actual.n_periods == 4
    assert np.all(np.isfinite(_to_numpy(actual.betas_)))
    assert np.all(np.isfinite(_to_numpy(actual.coef_)))
    assert np.all(np.isfinite(_to_numpy(actual.cov_params_)))
    assert actual.fit_statistics_.rsquared_overall == 0.0
    assert actual.fit_statistics_.metadata["degenerate_total_ss"]["overall"] is True
    np.testing.assert_allclose(
        _to_numpy(actual.betas_)[:, 0],
        np.asarray(expected.betas_)[:, 0],
        rtol=5e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _to_numpy(actual.coef_)[0],
        np.asarray(expected.coef_)[0],
        rtol=5e-14,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "cov_type,expected_slope_variance",
    [
        ("newey-west", (2.0 / 9.0) * 1.0e308),
        ("nonrobust", (1.0 / 3.0) * 1.0e308),
    ],
)
def test_fama_macbeth_scaled_coefficient_covariance_avoids_representable_overflow(
    cov_type, expected_slope_variance
):
    x_period = np.asarray([-1.0, 0.0, 1.0])
    slopes = np.asarray([-1.0e154, 0.0, 1.0e154])
    X = np.tile(x_period, slopes.size)[:, None]
    y = np.concatenate([slope * x_period for slope in slopes])
    time_ids = np.repeat(np.arange(slopes.size), x_period.size)

    model = FamaMacBeth(cov_type=cov_type, bandwidth=0, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    cov = np.asarray(model.cov_params_)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(
        cov[1, 1], expected_slope_variance, rtol=5e-13, atol=0.0
    )


def test_fama_macbeth_unrepresentable_coefficient_covariance_fails_closed():
    x_period = np.asarray([-1.0, 0.0, 1.0])
    slopes = np.asarray([-1.0e155, 0.0, 1.0e155])
    X = np.tile(x_period, slopes.size)[:, None]
    y = np.concatenate([slope * x_period for slope in slopes])
    time_ids = np.repeat(np.arange(slopes.size), x_period.size)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="covariance contains non-finite values"):
            FamaMacBeth(bandwidth=0, device="cpu").fit(
                X, y, time_ids=time_ids
            )
