import numpy as np
import pytest

from statgpu.panel import FamaMacBeth
from statgpu.panel._diagnostic_context import (
    _scaled_column_means,
    pooling_f_from_level_arrays,
)
from statgpu.panel._diagnostics import (
    _build_fit_statistics,
    _scaled_group_means,
    _scaled_mean,
)
from statgpu.panel._reductions import grouped_score_sums
from statgpu.panel._utils import group_means, within_transform


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_subnormal_group_tail(xp, scores):
    tiny = np.nextafter(0.0, 1.0)
    actual = grouped_score_sums(
        scores, np.zeros(3, dtype=np.int64), n_groups=1, xp=xp
    )
    assert float(actual[0, 0]) == tiny


def _collective_subnormal_mean_values():
    tiny = np.nextafter(0.0, 1.0)
    return np.asarray([1.0e308, -1.0e308, tiny, tiny, tiny]), tiny


def test_grouped_score_sum_preserves_subnormal_tail_at_exact_range_boundary_numpy():
    boundary = float(np.finfo(np.float64).max / 3.0)
    tiny = np.nextafter(0.0, 1.0)
    scores = np.asarray([[boundary], [-boundary], [tiny]], dtype=np.float64)
    _assert_subnormal_group_tail(np, scores)


def test_grouped_score_sum_preserves_subnormal_tail_when_large_tier_requires_scaling_numpy():
    tiny = np.nextafter(0.0, 1.0)
    scores = np.asarray([[1.0e308], [-1.0e308], [tiny]], dtype=np.float64)
    _assert_subnormal_group_tail(np, scores)


def test_shared_group_mean_preserves_small_term_after_huge_cancellation_numpy():
    values = np.asarray([1.0e308, 1.0, -1.0e308], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    np.testing.assert_allclose(actual, np.full(3, 1.0 / 3.0), rtol=0.0, atol=0.0)


def test_panel_means_preserve_collectively_representable_subnormal_tail_numpy():
    values, tiny = _collective_subnormal_mean_values()
    groups = np.zeros(values.size, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    assert np.all(actual == tiny)
    assert float(_scaled_mean(values, np)) == tiny


def test_scaled_mean_preserves_small_term_after_huge_cancellation_numpy():
    values = np.asarray([1.0e308, 1.0, -1.0e308], dtype=np.float64)
    assert float(_scaled_mean(values, np)) == 1.0 / 3.0


def test_scaled_group_means_preserve_small_term_after_huge_cancellation_numpy():
    values = np.asarray(
        [1.0e308, 1.0, -1.0e308, 5.0, 5.0, 5.0], dtype=np.float64
    )
    groups = np.asarray([7, 7, 7, 3, 3, 3], dtype=np.int64)
    actual = np.asarray(_scaled_group_means(values, groups, np))
    expected = np.asarray([1.0 / 3.0] * 3 + [5.0] * 3, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_shared_group_mean_scales_at_exact_overflow_boundary_numpy():
    value = float(np.finfo(np.float64).max / 3.0)
    values = np.asarray([value, value, value], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, np.full(3, value))
    np.testing.assert_array_equal(
        np.asarray(within_transform(values, groups, xp=np)), np.zeros(3)
    )


def test_shared_group_mean_preserves_smallest_subnormal_numpy():
    tiny = np.nextafter(0.0, 1.0)
    values = np.asarray([tiny, tiny, tiny], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    assert np.all(actual == tiny)
    assert float(_scaled_mean(values, np)) == tiny


def _multiscale_fmb_fixture():
    x = np.asarray([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    intercepts = np.asarray([1.0e150, 1.0, -1.0e150], dtype=np.float64)
    X = np.tile(x, intercepts.size)[:, None]
    y = np.concatenate([np.full(x.size, value) for value in intercepts])
    time = np.repeat(np.arange(intercepts.size), x.size)
    return X, y, time


def test_fama_macbeth_average_preserves_middle_period_after_large_cancellation_numpy():
    X, y, time = _multiscale_fmb_fixture()
    model = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    np.testing.assert_allclose(float(model.coef_[0]), 1.0 / 3.0, rtol=2e-14, atol=0.0)


def test_torch_cpu_group_mean_and_fmb_cancellation_match_numpy():
    torch = pytest.importorskip("torch")
    values = torch.tensor([1.0e308, 1.0, -1.0e308], dtype=torch.float64)
    groups = torch.zeros(3, dtype=torch.int64)
    grouped = group_means(values, groups, xp=torch)
    np.testing.assert_allclose(
        _to_numpy(grouped), np.full(3, 1.0 / 3.0), rtol=0.0, atol=0.0
    )
    assert float(_scaled_mean(values, torch)) == 1.0 / 3.0

    collective_values, tiny = _collective_subnormal_mean_values()
    collective = torch.as_tensor(collective_values, dtype=torch.float64)
    collective_groups = torch.zeros(collective.numel(), dtype=torch.int64)
    collective_mean = group_means(collective, collective_groups, xp=torch)
    assert np.all(_to_numpy(collective_mean) == tiny)
    assert float(_scaled_mean(collective, torch)) == tiny

    boundary = float(np.finfo(np.float64).max / 3.0)
    for amplitude in (boundary, 1.0e308):
        scores = torch.tensor(
            [[amplitude], [-amplitude], [tiny]], dtype=torch.float64
        )
        _assert_subnormal_group_tail(torch, scores)

    X, y, time = _multiscale_fmb_fixture()
    expected = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    actual = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    np.testing.assert_allclose(
        _to_numpy(actual.coef_), np.asarray(expected.coef_), rtol=3e-13, atol=0.0
    )


def test_torch_cpu_huge_constant_response_remains_degenerate():
    torch = pytest.importorskip("torch")
    x_period = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
    X = np.tile(x_period, 4)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time = np.repeat(np.arange(4), x_period.size)
    model = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert model.fit_statistics_.rsquared_overall == 0.0
    assert model.fit_statistics_.metadata["degenerate_total_ss"]["overall"] is True



def test_pooling_f_column_mean_preserves_cancellation_tail_numpy():
    values = np.asarray(
        [[1.0e308], [1.0], [-1.0e308], [0.0], [0.0], [0.0]],
        dtype=np.float64,
    )
    actual = np.asarray(_scaled_column_means(values, np), dtype=np.float64)
    np.testing.assert_allclose(actual, np.asarray([1.0 / 6.0]), rtol=0.0, atol=0.0)


def test_pooling_f_column_mean_preserves_cancellation_tail_torch_cpu():
    torch = pytest.importorskip("torch")
    values = torch.as_tensor(
        [[1.0e308], [1.0], [-1.0e308], [0.0], [0.0], [0.0]],
        dtype=torch.float64,
    )
    actual = _to_numpy(_scaled_column_means(values, torch))
    np.testing.assert_allclose(actual, np.asarray([1.0 / 6.0]), rtol=0.0, atol=0.0)



def _extreme_centering_fixture(xp):
    y_np = np.asarray([1.0e308] + [-1.0e308] * 10, dtype=np.float64)
    z_np = np.asarray([1.0] + [-1.0] * 10, dtype=np.float64)
    X_np = np.column_stack([np.ones(11), z_np])
    params_np = np.asarray([0.0, 1.0e308], dtype=np.float64)
    if xp is np:
        return y_np, X_np, params_np
    return (
        xp.as_tensor(y_np, dtype=xp.float64),
        xp.as_tensor(X_np, dtype=xp.float64),
        xp.as_tensor(params_np, dtype=xp.float64),
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_extreme_constant_centering_keeps_diagnostics_on_finite_working_scale(backend):
    xp = np if backend == "numpy" else pytest.importorskip("torch")
    y, X, params = _extreme_centering_fixture(xp)
    result = _build_fit_statistics(
        y, X, params, xp=xp, entity_codes=None, has_constant=True,
        rss_fit=0.0, tss_fit=float("inf"), df_resid=9, df_total=10,
    )
    assert result.rsquared_overall == 1.0
    assert result.rsquared_adj == 1.0
    assert np.isposinf(result.f_statistic)
    assert result.f_pvalue == 0.0


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_extreme_pooling_f_centering_avoids_level_overflow(backend):
    xp = np if backend == "numpy" else pytest.importorskip("torch")
    y_np = np.asarray([1.0e308] + [-1.0e308] * 10, dtype=np.float64)
    X_np = np.linspace(-1.0, 1.0, 11, dtype=np.float64)[:, None]
    if xp is np:
        y, X, resid_effects = y_np, X_np, np.zeros(11, dtype=np.float64)
    else:
        y = xp.as_tensor(y_np, dtype=xp.float64)
        X = xp.as_tensor(X_np, dtype=xp.float64)
        resid_effects = xp.zeros(11, dtype=xp.float64)
    result = pooling_f_from_level_arrays(
        y, X, xp=xp, rss_effects=0.0, df_resid_effects=8,
        has_constant=False, resid_effects=resid_effects,
    )
    assert result.applicable
    assert np.isposinf(result.statistic)
    assert result.pvalue == 0.0
