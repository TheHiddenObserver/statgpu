import inspect
import types
import warnings

import numpy as np
import pytest
from scipy import stats
from sklearn.base import clone

from statgpu._config import Device
from statgpu.linear_model import (
    ElasticNet,
    Lasso,
    LassoCV,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
)
import statgpu.linear_model._penalized_inference_api_contract as inference_contract
from statgpu.linear_model.penalized._post_selection_ols import (
    _POST_SELECTION_ACTIVE_TOL,
    compute_post_selection_ols_inference,
)


def _data(seed=123, n=320, p=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [1.4, -0.9, 0.65]
    y = 0.35 + X @ beta + rng.normal(scale=0.5, size=n)
    return X, y


def _active_ols_reference(
    X,
    y,
    penalized_coef,
    *,
    fit_intercept,
    penalized_intercept=0.0,
    sample_weight=None,
):
    penalized_coef = np.asarray(penalized_coef, dtype=float).reshape(-1)
    selected = np.flatnonzero(np.abs(penalized_coef) > _POST_SELECTION_ACTIVE_TOL)
    X_sel = X[:, selected]
    if fit_intercept:
        design = np.column_stack([np.ones(X.shape[0]), X_sel])
    else:
        design = X_sel

    if sample_weight is None:
        design_work = design
        y_work = y
    else:
        sqrt_w = np.sqrt(np.asarray(sample_weight, dtype=float))
        design_work = design * sqrt_w[:, None]
        y_work = y * sqrt_w

    k = design_work.shape[1]
    if k:
        params_sel = np.linalg.pinv(design_work) @ y_work
        resid_work = y_work - design_work @ params_sel
    else:
        params_sel = np.empty(0)
        resid_work = y_work
    df_resid = X.shape[0] - k
    scale = float(resid_work @ resid_work / df_resid)

    params = penalized_coef.copy()
    if fit_intercept:
        params = np.concatenate([[float(penalized_intercept)], params])
    full_dim = params.shape[0]
    bse = np.full(full_dim, np.nan)
    tvalues = np.full(full_dim, np.nan)
    pvalues = np.full(full_dim, np.nan)
    conf_int = np.full((full_dim, 2), np.nan)

    if k:
        XtX_inv = np.linalg.pinv(design_work.T @ design_work)
        bse_sel = np.sqrt(np.maximum(scale * np.diag(XtX_inv), 0.0))
        t_sel = params_sel / (bse_sel + 1e-30)
        p_sel = 2.0 * stats.t.sf(np.abs(t_sel), df=df_resid)
        critical = stats.t.ppf(0.975, df=df_resid)
        ci_sel = np.column_stack(
            [params_sel - critical * bse_sel, params_sel + critical * bse_sel]
        )
        if fit_intercept:
            params[0], bse[0], tvalues[0], pvalues[0], conf_int[0] = (
                params_sel[0], bse_sel[0], t_sel[0], p_sel[0], ci_sel[0]
            )
            target = selected + 1
            params[target] = params_sel[1:]
            bse[target] = bse_sel[1:]
            tvalues[target] = t_sel[1:]
            pvalues[target] = p_sel[1:]
            conf_int[target] = ci_sel[1:]
        else:
            params[selected] = params_sel
            bse[selected] = bse_sel
            tvalues[selected] = t_sel
            pvalues[selected] = p_sel
            conf_int[selected] = ci_sel

    return selected, params, bse, tvalues, pvalues, conf_int, df_resid, scale


@pytest.mark.parametrize("alias", ["cpu_ols", "gpu_ols"])
def test_post_selection_aliases_warn_and_normalize(alias):
    with pytest.warns(FutureWarning, match="post_selection_ols") as caught:
        model = Lasso(inference_method=alias, compute_inference=False)
    assert caught[0].filename == __file__
    assert model.inference_method == alias
    assert model._inference_method == "post_selection_ols"


def test_post_selection_alias_clone_is_warning_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = Lasso(inference_method="cpu_ols", compute_inference=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cloned = clone(model)

    assert not [item for item in caught if issubclass(item.category, FutureWarning)]
    assert cloned.inference_method == "cpu_ols"
    assert cloned.get_params(deep=False)["inference_method"] == "cpu_ols"
    assert cloned._inference_method == "post_selection_ols"


def test_penalized_base_alias_uses_same_deprecation_contract():
    with pytest.warns(FutureWarning, match="post_selection_ols"):
        model = PenalizedLinearRegression(
            penalty="l1",
            inference_method="gpu_ols",
            compute_inference=False,
        )
    assert model.inference_method == "gpu_ols"
    assert model._inference_method == "post_selection_ols"


def test_lassocv_legacy_alias_clone_is_warning_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = LassoCV(inference_method="gpu_ols_inference")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cloned = clone(model)

    assert not [item for item in caught if issubclass(item.category, FutureWarning)]
    assert cloned.inference_method == "gpu_ols_inference"
    assert cloned._inference_method == "post_selection_ols"


def test_set_params_alias_warns_once_and_invalidates_fit_state():
    X, y = _data(seed=1, n=100)
    model = Lasso(compute_inference=False, device="cpu").fit(X, y)
    assert model._fitted

    with pytest.warns(FutureWarning, match="post_selection_ols") as caught:
        model.set_params(inference_method="gpu_ols")

    assert len(caught) == 1
    assert model.inference_method == "gpu_ols"
    assert model._inference_method == "post_selection_ols"
    assert not model._fitted


@pytest.mark.parametrize("alias", ["cpu_ols", "gpu_ols"])
def test_deprecated_aliases_execute_canonical_post_selection_numerics(alias):
    X, y = _data(seed=15, n=180, p=5)
    canonical = Lasso(
        alpha=0.05,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-8,
    ).fit(X, y)
    with pytest.warns(FutureWarning, match="post_selection_ols"):
        migrated = Lasso(
            alpha=0.05,
            inference_method=alias,
            compute_inference=True,
            device="cpu",
            max_iter=4000,
            tol=1e-8,
        )
    migrated.fit(X, y)

    np.testing.assert_allclose(migrated.coef_, canonical.coef_, rtol=0, atol=0)
    np.testing.assert_allclose(migrated._params, canonical._params, rtol=0, atol=0)
    np.testing.assert_allclose(
        migrated._bse, canonical._bse, rtol=0, atol=0, equal_nan=True
    )
    np.testing.assert_allclose(
        migrated._pvalues, canonical._pvalues, rtol=0, atol=0, equal_nan=True
    )
    assert migrated._inference_result.method == "post_selection_ols"
    assert migrated._inference_result.metadata["requested_method"] == alias


def test_post_selection_ols_refits_active_set_and_keeps_penalized_coef():
    X, y = _data()
    model = Lasso(
        alpha=0.05,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=5000,
        tol=1e-8,
    ).fit(X, y)

    penalized_coef = model.coef_.copy()
    ref = _active_ols_reference(
        X,
        y,
        penalized_coef,
        fit_intercept=True,
        penalized_intercept=model.intercept_,
    )
    selected, params, bse, tvalues, pvalues, conf_int, df_resid, scale = ref

    np.testing.assert_allclose(model._params, params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        model._bse, bse, rtol=1e-10, atol=1e-11, equal_nan=True
    )
    np.testing.assert_allclose(
        model._tvalues, tvalues, rtol=1e-10, atol=1e-11, equal_nan=True
    )
    np.testing.assert_allclose(
        model._pvalues, pvalues, rtol=1e-10, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        model._conf_int, conf_int, rtol=1e-10, atol=1e-11, equal_nan=True
    )
    np.testing.assert_allclose(model.coef_, penalized_coef, rtol=0, atol=0)
    assert model._df_resid == df_resid
    assert model._scale == pytest.approx(scale, rel=1e-12, abs=1e-12)
    assert model._inference_result.method == "post_selection_ols"
    assert model._inference_result.metadata["numerical_backend"] == "numpy"
    assert model._inference_result.metadata["numerical_device"] == "cpu"
    assert model._inference_result.metadata["selected_feature_indices"] == selected.tolist()
    assert model._inference_result.metadata["active_set_tolerance"] == _POST_SELECTION_ACTIVE_TOL
    assert not np.allclose(model._params[1:][selected], penalized_coef[selected])

    inactive = np.setdiff1d(np.arange(X.shape[1]), selected)
    if inactive.size:
        np.testing.assert_allclose(model._params[1:][inactive], penalized_coef[inactive])
        assert np.all(np.isnan(model._bse[1:][inactive]))
        assert np.all(np.isnan(model._pvalues[1:][inactive]))
        assert np.all(np.isnan(model._conf_int[1:][inactive]))


def test_post_selection_ols_weighted_refit_matches_wls():
    X, y = _data(seed=7, n=260, p=6)
    rng = np.random.default_rng(8)
    sample_weight = rng.uniform(0.25, 2.0, size=X.shape[0])
    model = Lasso(
        alpha=0.04,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=5000,
        tol=1e-8,
    ).fit(X, y, sample_weight=sample_weight)

    ref = _active_ols_reference(
        X,
        y,
        model.coef_,
        fit_intercept=True,
        penalized_intercept=model.intercept_,
        sample_weight=sample_weight,
    )
    _, params, bse, tvalues, pvalues, conf_int, df_resid, scale = ref
    np.testing.assert_allclose(model._params, params, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        model._bse, bse, rtol=1e-10, atol=1e-11, equal_nan=True
    )
    np.testing.assert_allclose(
        model._tvalues, tvalues, rtol=1e-10, atol=1e-11, equal_nan=True
    )
    np.testing.assert_allclose(
        model._pvalues, pvalues, rtol=1e-10, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        model._conf_int, conf_int, rtol=1e-10, atol=1e-11, equal_nan=True
    )
    assert model._df_resid == df_resid
    assert model._scale == pytest.approx(scale, rel=1e-12, abs=1e-12)
    assert model._inference_result.metadata["sample_weighted"] is True


def test_post_selection_ols_intercept_only_marks_unselected_features_uninferred():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 4))
    y = 1.75 + rng.normal(scale=0.2, size=100)
    model = Lasso(
        alpha=100.0,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
    ).fit(X, y)

    assert np.count_nonzero(model.coef_) == 0
    assert model._params[0] == pytest.approx(float(np.mean(y)), rel=1e-12, abs=1e-12)
    np.testing.assert_array_equal(model._params[1:], model.coef_)
    assert np.all(np.isnan(model._bse[1:]))
    assert np.all(np.isnan(model._tvalues[1:]))
    assert np.all(np.isnan(model._pvalues[1:]))
    assert np.all(np.isnan(model._conf_int[1:]))
    assert model._df_resid == X.shape[0] - 1


def test_post_selection_ols_no_intercept_empty_active_set_is_uninferred():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(80, 3))
    y = 2.0 + rng.normal(scale=0.2, size=80)
    model = Lasso(
        alpha=100.0,
        fit_intercept=False,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
    ).fit(X, y)

    np.testing.assert_array_equal(model._params, model.coef_)
    assert np.all(np.isnan(model._bse))
    assert np.all(np.isnan(model._tvalues))
    assert np.all(np.isnan(model._pvalues))
    assert np.all(np.isnan(model._conf_int))
    assert model._df_resid == X.shape[0]


def test_post_selection_active_tolerance_preserves_numeric_dust_as_uninferred():
    X, y = _data(seed=22, n=90, p=2)
    model = Lasso(compute_inference=False, device="cpu").fit(X, y)
    model.coef_ = np.array([0.5 * _POST_SELECTION_ACTIVE_TOL, 0.25])
    model.intercept_ = 0.0
    model._selected_backend_name = "numpy"
    model._selected_backend_device = "cpu"
    model._effective_intercept = False
    model.inference_method = "post_selection_ols"
    model._inference_method = "post_selection_ols"

    compute_post_selection_ols_inference(model, X[:, :2], y)

    assert model._inference_result.metadata["selected_feature_indices"] == [1]
    assert model._params[0] == pytest.approx(0.5 * _POST_SELECTION_ACTIVE_TOL)
    assert np.isnan(model._bse[0])
    assert np.isnan(model._pvalues[0])
    assert np.all(np.isnan(model._conf_int[0]))


def test_post_selection_ols_reuses_fit_backend_not_raw_input_type():
    pytest.importorskip("torch")
    import torch

    X, y = _data(seed=11, n=180, p=5)
    X_torch = torch.as_tensor(X, dtype=torch.float64)
    y_torch = torch.as_tensor(y, dtype=torch.float64)

    model = Lasso(
        alpha=0.05,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-8,
    ).fit(X_torch, y_torch)

    assert model._selected_backend_name == "numpy"
    assert model._inference_result.metadata["numerical_backend"] == "numpy"
    assert model._inference_result.metadata["numerical_device"] == "cpu"


def test_post_selection_torch_cpu_numerical_kernel_matches_numpy_reporting():
    torch = pytest.importorskip("torch")
    X, y = _data(seed=12, n=180, p=5)

    model = Lasso(
        alpha=0.05,
        compute_inference=False,
        device="cpu",
        max_iter=4000,
        tol=1e-8,
    ).fit(X, y)
    selected = np.flatnonzero(np.abs(model.coef_) > _POST_SELECTION_ACTIVE_TOL)
    ref = _active_ols_reference(
        X,
        y,
        model.coef_,
        fit_intercept=True,
        penalized_intercept=model.intercept_,
    )

    # Hosted CI has Torch CPU only. Exercise the numerical kernel without
    # pretending this is public device='torch' execution (public Torch is CUDA).
    model._selected_backend_name = "torch"
    model._selected_backend_device = "cpu"
    model.inference_method = "post_selection_ols"
    model._inference_method = "post_selection_ols"
    compute_post_selection_ols_inference(
        model,
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
    )

    np.testing.assert_allclose(model._params, ref[1], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        model._bse, ref[2], rtol=1e-9, atol=1e-10, equal_nan=True
    )
    np.testing.assert_allclose(
        model._pvalues, ref[4], rtol=1e-8, atol=1e-10, equal_nan=True
    )
    assert model._inference_result.metadata["numerical_backend"] == "torch"
    assert model._inference_result.metadata["numerical_device"] == "cpu"
    assert model._inference_result.metadata["selected_feature_indices"] == selected.tolist()


def test_auto_auto_can_preserve_native_backend_but_global_explicit_device_wins(monkeypatch):
    model = Lasso(compute_inference=False, device="auto")
    fake_torch = types.SimpleNamespace(kind="torch", is_cuda=True)
    fake_cupy = types.SimpleNamespace(kind="cupy")

    monkeypatch.setattr(
        inference_contract,
        "_is_torch_array",
        lambda x: getattr(x, "kind", None) == "torch",
    )
    monkeypatch.setattr(
        inference_contract,
        "_is_cupy_array",
        lambda x: getattr(x, "kind", None) == "cupy",
    )
    monkeypatch.setattr(
        inference_contract, "_get_configured_device", lambda: Device.AUTO
    )

    assert inference_contract._input_native_device(model, fake_torch) == Device.TORCH
    assert inference_contract._input_native_device(model, fake_cupy) == Device.CUDA

    explicit_cpu = Lasso(compute_inference=False, device="cpu")
    assert inference_contract._input_native_device(explicit_cpu, fake_cupy) is None

    monkeypatch.setattr(
        inference_contract, "_get_configured_device", lambda: Device.CPU
    )
    assert inference_contract._input_native_device(model, fake_cupy) is None


def test_lassocv_default_and_legacy_value_migrate_to_canonical_refit():
    signature = inspect.signature(LassoCV)
    assert signature.parameters["inference_method"].default == "post_selection_ols"

    X, y = _data(seed=13, n=140, p=5)
    model = LassoCV(
        cv=3,
        n_alphas=5,
        compute_inference=True,
        device="cpu",
        max_iter=2500,
        tol=1e-6,
    ).fit(X, y)
    assert model._inference_method == "post_selection_ols"
    assert model.estimator_._inference_result.method == "post_selection_ols"

    with pytest.warns(FutureWarning, match="post_selection_ols"):
        legacy = LassoCV(inference_method="cpu_ols_inference")
    assert legacy.inference_method == "cpu_ols_inference"
    assert legacy._inference_method == "post_selection_ols"


def test_elasticnet_uses_same_canonical_post_selection_contract():
    X, y = _data(seed=14, n=220, p=6)
    model = ElasticNet(
        alpha=0.04,
        l1_ratio=0.8,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-8,
    ).fit(X, y)
    assert model._inference_result.method == "post_selection_ols"
    assert model._inference_result.metadata["numerical_backend"] == "numpy"


def test_post_selection_ols_honors_hc3_covariance_contract():
    X, y = _data(seed=18, n=240, p=6)
    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.045,
        inference_method="post_selection_ols",
        compute_inference=True,
        cov_type="hc3",
        device="cpu",
        max_iter=5000,
        tol=1e-8,
    ).fit(X, y)

    selected = np.flatnonzero(np.abs(model.coef_) > _POST_SELECTION_ACTIVE_TOL)
    design = np.column_stack([np.ones(X.shape[0]), X[:, selected]])
    params_sel = np.linalg.pinv(design) @ y
    resid = y - design @ params_sel
    bread = np.linalg.pinv(design.T @ design)
    leverage = np.sum((design @ bread) * design, axis=1)
    adjusted = resid / np.maximum(1.0 - leverage, 1e-12)
    meat = design.T @ (design * (adjusted * adjusted)[:, None])
    cov = bread @ meat @ bread
    bse_sel = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z_sel = params_sel / (bse_sel + 1e-30)
    p_sel = 2.0 * stats.norm.sf(np.abs(z_sel))
    critical = stats.norm.ppf(0.975)
    ci_sel = np.column_stack(
        [params_sel - critical * bse_sel, params_sel + critical * bse_sel]
    )

    target = np.concatenate([[0], selected + 1])
    np.testing.assert_allclose(model._params[target], params_sel, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(model._bse[target], bse_sel, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(model._pvalues[target], p_sel, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(model._conf_int[target], ci_sel, rtol=1e-9, atol=1e-10)
    assert model._inference_result.cov_type == "hc3"
    assert model._inference_result.distribution == "normal"


def test_post_selection_ols_rejects_non_gaussian_sparse_model_before_fit_dispatch():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(80, 4))
    y = rng.poisson(np.exp(0.1 + 0.15 * X[:, 0])).astype(float)
    model = PenalizedGeneralizedLinearModel(
        loss="poisson",
        penalty="l1",
        alpha=0.02,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
    )
    with pytest.raises(NotImplementedError, match="squared_error"):
        model.fit(X, y)
