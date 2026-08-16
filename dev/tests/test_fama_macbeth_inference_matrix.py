"""Maintained Fama-MacBeth covariance/inference matrix coverage."""

from __future__ import annotations

import numpy as np
import pytest

import statgpu.inference._distributions_backend as distribution_backend
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


@pytest.mark.parametrize("cov_type", ["newey-west", "nonrobust"])
def test_fama_macbeth_torch_cpu_evaluates_pvalue_sf_once_per_fit(monkeypatch, cov_type):
    """The CPU distribution receives one vector, not one synchronized scalar per term."""
    torch = pytest.importorskip("torch")
    X, y, time_ids = _fixture()
    original_get_distribution = distribution_backend.get_distribution
    sf_input_shapes = []

    class CountingDistribution:
        def __init__(self, base):
            self._base = base

        def sf(self, values, *args):
            sf_input_shapes.append(np.asarray(values).shape)
            return self._base.sf(values, *args)

        def isf(self, *args):
            return self._base.isf(*args)

    def tracked_get_distribution(name, backend="numpy"):
        return CountingDistribution(original_get_distribution(name, backend=backend))

    monkeypatch.setattr(
        distribution_backend,
        "get_distribution",
        tracked_get_distribution,
    )

    model = FamaMacBeth(cov_type=cov_type, bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time_ids, dtype=torch.int64),
    )

    assert model._backend_name == "torch"
    assert sf_input_shapes == [(2,)]
    assert np.all(np.isfinite(_to_numpy(model.pvalues_)))
