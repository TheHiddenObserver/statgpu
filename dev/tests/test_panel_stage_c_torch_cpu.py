"""Maintained Torch-CPU parity for Panel Tier-1 Stage C covariance."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.backends import get_backend
from statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._covariance import driscoll_kraay_covariance, ols_covariance, two_way_clustered_covariance


torch = pytest.importorskip("torch")


def _panel(seed=12900, *, unbalanced=True):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 8, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.35, size=n_entities), n_times)
    y = 0.75 * X[:, 0] - 0.3 * X[:, 1] + alpha + rng.normal(scale=0.2, size=entity.size)
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[2, 11, 25, 40]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X, y, entity, time


def _torch_arrays(X, y, entity, time):
    return (
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(entity, dtype=torch.int64),
        torch.as_tensor(time, dtype=torch.int64),
    )


def _torch_cpu_estimator(model):
    """Inject Torch CPU internally without weakening public ``device='torch'`` CUDA strictness."""
    torch_cpu_backend = get_backend(backend="torch", device="cpu")
    model._get_backend = lambda backend="auto": torch_cpu_backend
    return model


def _assert_inference(actual, expected, *, rtol=2e-8, atol=2e-10):
    assert actual._backend_name == "torch"
    assert actual._inference_backend_name == "torch"
    assert actual._inference_result.metadata["inference_backend"] == "torch"
    assert_allclose(actual.coef_, expected.coef_, rtol=rtol, atol=atol)
    assert_allclose(actual.bse_, expected.bse_, rtol=rtol, atol=atol)
    assert_allclose(actual.tvalues_, expected.tvalues_, rtol=rtol, atol=atol)
    assert_allclose(actual.pvalues_, expected.pvalues_, rtol=rtol, atol=atol)
    assert_allclose(actual.conf_int_, expected.conf_int_, rtol=rtol, atol=atol)
    assert_allclose(actual._panel_cov_params_raw, expected._panel_cov_params_raw, rtol=rtol, atol=atol)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_hc_primitives_torch_cpu_match_numpy(cov_type):
    rng = np.random.default_rng(12901)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    resid = rng.normal(size=40)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    resid_t = torch.as_tensor(resid, dtype=torch.float64)
    expected = ols_covariance(X, resid, cov_type=cov_type, xp=np)
    actual = ols_covariance(X_t, resid_t, cov_type=cov_type)
    assert torch.is_tensor(actual)
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_ill_conditioned_hc_torch_cpu_matches_numpy(cov_type):
    rng = np.random.default_rng(129011)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack(
        [np.ones(n), x, x + 1.0e-9 * rng.normal(size=n)]
    )
    y = X @ np.array([0.2, 0.7, -0.4]) + rng.normal(scale=0.2, size=n)
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ params
    assert np.linalg.matrix_rank(X) == 3
    assert np.linalg.cond(X) > 1.0e8

    expected = ols_covariance(X, resid, cov_type=cov_type, xp=np)
    actual = ols_covariance(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(resid, dtype=torch.float64),
        cov_type=cov_type,
    )
    assert torch.is_tensor(actual)
    assert_allclose(
        actual.detach().cpu().numpy(),
        expected,
        rtol=2e-5,
        atol=1e-2,
    )
    assert np.all(np.diag(actual.detach().cpu().numpy()) >= 0.0)


def test_stage_c_two_way_cluster_torch_cpu_matches_numpy_with_group_debias():
    rng = np.random.default_rng(12902)
    X = np.column_stack([np.ones(48), rng.normal(size=(48, 2))])
    resid = rng.normal(size=48)
    c1 = np.repeat(np.arange(8), 6)
    c2 = np.tile(np.arange(6), 8)
    expected = two_way_clustered_covariance(
        X, resid, c1, c2, xp=np, group_debias=True
    )
    actual = two_way_clustered_covariance(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(resid, dtype=torch.float64),
        c1,
        c2,
        group_debias=True,
    )
    assert torch.is_tensor(actual)
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_stage_c_dk_torch_cpu_matches_numpy(kernel):
    rng = np.random.default_rng(12903)
    time = np.tile(np.arange(8), 7)
    X = np.column_stack([np.ones(time.size), rng.normal(size=(time.size, 2))])
    resid = rng.normal(size=time.size)
    expected = driscoll_kraay_covariance(
        X, resid, time, bandwidth=2, kernel=kernel, extra_df=3, xp=np
    )
    actual = driscoll_kraay_covariance(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(resid, dtype=torch.float64),
        time,
        bandwidth=2,
        kernel=kernel,
        extra_df=3,
    )
    assert torch.is_tensor(actual)
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_pooled_torch_cpu_hc_matches_numpy(cov_type):
    X, y, entity, time = _panel(12904)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)
    expected = PooledOLS(cov_type=cov_type).fit(X, y, entity_ids=entity)
    actual = _torch_cpu_estimator(PooledOLS(cov_type=cov_type)).fit(
        X_t, y_t, entity_ids=entity_t
    )
    _assert_inference(actual, expected)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_panel_fe_torch_cpu_hc_matches_numpy(cov_type):
    X, y, entity, time = _panel(12905)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)
    expected = PanelOLS(entity_effects=True, cov_type=cov_type).fit(X, y, entity_ids=entity)
    actual = _torch_cpu_estimator(
        PanelOLS(entity_effects=True, cov_type=cov_type)
    ).fit(X_t, y_t, entity_ids=entity_t)
    _assert_inference(actual, expected)


def test_stage_c_panel_dk_torch_cpu_matches_numpy_and_effect_rank_metadata():
    X, y, entity, time = _panel(12906)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    expected = PanelOLS(entity_effects=True, cov_type="dk", bandwidth=2).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    actual = _torch_cpu_estimator(
        PanelOLS(entity_effects=True, cov_type="dk", bandwidth=2)
    ).fit(X_t, y_t, entity_ids=entity_t, time_ids=time_t)
    _assert_inference(actual, expected)
    assert actual._covariance_metadata["extra_df"] == expected._covariance_metadata["extra_df"]
    assert actual._covariance_metadata["design_rank"] == expected._covariance_metadata["design_rank"]


@pytest.mark.parametrize("cov_type", ["robust", "hc0", "hc2", "hc3"])
def test_stage_c_random_effects_torch_cpu_covariance_matches_numpy(cov_type):
    X, y, entity, time = _panel(12907)
    X = np.column_stack([np.ones(len(y)), X])
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)
    expected = RandomEffects(cov_type=cov_type).fit(X, y, entity_ids=entity)
    actual = _torch_cpu_estimator(RandomEffects(cov_type=cov_type)).fit(
        X_t, y_t, entity_ids=entity_t
    )
    _assert_inference(actual, expected, rtol=5e-8, atol=5e-10)
    assert_allclose(
        [actual.variance_components_["sigma2_e"], actual.variance_components_["sigma2_a"]],
        [expected.variance_components_["sigma2_e"], expected.variance_components_["sigma2_a"]],
        rtol=2e-10,
        atol=2e-12,
    )


def test_stage_c_random_effects_dk_torch_cpu_matches_numpy():
    X, y, entity, time = _panel(12908)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    expected = RandomEffects(cov_type="dk", bandwidth=2, kernel="qs").fit(
        X, y, entity_ids=entity, time_ids=time
    )
    actual = _torch_cpu_estimator(
        RandomEffects(cov_type="dk", bandwidth=2, kernel="qs")
    ).fit(X_t, y_t, entity_ids=entity_t, time_ids=time_t)
    _assert_inference(actual, expected, rtol=5e-8, atol=5e-10)
    assert actual._covariance_metadata["all_observed_lags_weighted"] is True


@pytest.mark.parametrize("estimator", [BetweenOLS, FirstDifferenceOLS])
@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_between_and_fd_torch_cpu_hc_match_numpy(estimator, cov_type):
    X, y, entity, time = _panel(12909, unbalanced=False)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    if estimator is BetweenOLS:
        expected = estimator(cov_type=cov_type).fit(X, y, entity_ids=entity)
        actual = _torch_cpu_estimator(estimator(cov_type=cov_type)).fit(
            X_t, y_t, entity_ids=entity_t
        )
    else:
        expected = estimator(cov_type=cov_type).fit(
            X, y, entity_ids=entity, time_ids=time
        )
        actual = _torch_cpu_estimator(estimator(cov_type=cov_type)).fit(
            X_t, y_t, entity_ids=entity_t, time_ids=time_t
        )
    _assert_inference(actual, expected, rtol=5e-8, atol=5e-10)


def _rank_boundary_inputs(seed=12910):
    rng = np.random.default_rng(seed)
    n, k = 100, 3
    q_left, _ = np.linalg.qr(rng.normal(size=(n, k)))
    q_right, _ = np.linalg.qr(rng.normal(size=(k, k)))
    X = q_left @ np.diag([10.0, 1.0, 1.0e-13]) @ q_right.T
    resid = rng.normal(size=n)
    time = np.tile(np.arange(10), 10)
    cluster = np.repeat(np.arange(10), 10)
    cutoff = max(X.shape) * np.finfo(np.float64).eps * np.linalg.svd(X, compute_uv=False)[0]
    assert np.linalg.svd(X, compute_uv=False)[-1] < cutoff
    assert np.linalg.matrix_rank(X) == 2
    return X, resid, time, cluster


def test_stage_c_rank_boundary_covariance_torch_cpu_matches_numpy():
    X, resid, time, cluster = _rank_boundary_inputs()
    X_t = torch.as_tensor(X, dtype=torch.float64)
    resid_t = torch.as_tensor(resid, dtype=torch.float64)

    cases = [
        ("nonrobust", dict(scale=1.0)),
        ("hc0", {}),
        ("hc2", {}),
        ("hc3", {}),
        ("clustered", dict(cluster=cluster)),
        ("driscoll-kraay", dict(time_ids=time, bandwidth=2)),
    ]
    for cov_type, kwargs in cases:
        meta_np = {}
        meta_t = {}
        expected = ols_covariance(
            X, resid, cov_type=cov_type, xp=np, metadata=meta_np, **kwargs
        )
        actual = ols_covariance(
            X_t, resid_t, cov_type=cov_type, metadata=meta_t, **kwargs
        )
        assert torch.is_tensor(actual)
        assert_allclose(
            actual.detach().cpu().numpy(), expected, rtol=5e-10, atol=5e-12
        )
        assert np.all(np.isfinite(expected))
        if cov_type in {"hc0", "hc2", "hc3", "driscoll-kraay"}:
            assert meta_np["design_rank"] == 2
            assert meta_t["design_rank"] == 2


def test_panel_rank_boundary_fit_and_dk_torch_cpu_match_shared_policy():
    X, y, time, _cluster = _rank_boundary_inputs(seed=12911)
    U, singular_values, Vh = np.linalg.svd(X, full_matrices=False)
    cutoff = max(X.shape) * np.finfo(np.float64).eps * singular_values[0]
    inverse_values = np.zeros_like(singular_values)
    retained = singular_values > cutoff
    inverse_values[retained] = 1.0 / singular_values[retained]
    expected_coef = ((Vh.T * inverse_values) @ U.T) @ y

    expected = PanelOLS(cov_type="dk", bandwidth=2).fit(
        X, y, time_ids=time
    )
    actual = _torch_cpu_estimator(PanelOLS(cov_type="dk", bandwidth=2)).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert actual._backend_name == "torch"
    assert_allclose(expected.coef_, expected_coef, rtol=5e-11, atol=5e-13)
    assert_allclose(actual.coef_, expected.coef_, rtol=5e-9, atol=5e-11)
    assert_allclose(
        actual._panel_cov_params_raw,
        expected._panel_cov_params_raw,
        rtol=5e-9,
        atol=5e-11,
    )
    for model in (expected, actual):
        assert model._coefficient_inference_available is False
        assert model.bse_ is None
        assert model.tvalues_ is None
        assert model.pvalues_ is None
        assert model.conf_int_ is None
        assert model._inference_result.metadata["applicable"] is False
    assert expected._covariance_metadata["design_rank"] == 2
    assert actual._covariance_metadata["design_rank"] == 2
    assert expected.fit_statistics_.metadata["diagnostic_df"]["rank_x"] == 2
    assert actual.fit_statistics_.metadata["diagnostic_df"]["rank_x"] == 2


def test_stage_c_pooled_nonrobust_df2_torch_cpu_matches_numpy_high_precision():
    """The shared reference-inference helper preserves the exact t(2) boundary."""
    X = np.asarray([[-1.5], [-0.25], [0.75], [2.0]], dtype=np.float64)
    y = np.asarray([-0.6, 0.2, 1.1, 1.55], dtype=np.float64)

    expected = PooledOLS(cov_type="nonrobust", device="cpu").fit(X, y)
    actual = _torch_cpu_estimator(PooledOLS(cov_type="nonrobust")).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
    )

    assert expected.df_resid == 2
    assert actual.df_resid == 2
    _assert_inference(actual, expected, rtol=2e-12, atol=2e-14)

@pytest.mark.parametrize("kind", ["cluster", "two_way", "hac", "dk"])
def test_stage_c_public_covariance_rejects_nonfinite_residual_torch_cpu(kind):
    from statgpu.panel import clustered_covariance, hac_covariance

    X = torch.column_stack(
        [torch.ones(6, dtype=torch.float64), torch.arange(6, dtype=torch.float64)]
    )
    resid = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64)
    resid[2] = float("nan")
    c1 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    c2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    time = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        if kind == "cluster":
            clustered_covariance(X, resid, c1)
        elif kind == "two_way":
            two_way_clustered_covariance(X, resid, c1, c2)
        elif kind == "hac":
            hac_covariance(X, resid, bandwidth=1)
        else:
            driscoll_kraay_covariance(X, resid, time, bandwidth=1)
