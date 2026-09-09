import warnings

import numpy as np
import pytest
from sklearn.base import clone

from statgpu._config import Device
from statgpu.linear_model import (
    LassoCV,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
)
import statgpu.linear_model._penalized_inference_api_contract as inference_contract
from statgpu.penalties import get_penalty


def test_penalty_object_alias_uses_same_warning_and_normalization_contract():
    penalty = get_penalty("l1", alpha=0.05)
    with pytest.warns(FutureWarning, match="post_selection_ols") as caught:
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error",
            penalty=penalty,
            inference_method="gpu_ols",
            compute_inference=False,
            device="cpu",
        )

    assert len(caught) == 1
    assert model.penalty is penalty
    assert model.inference_method == "gpu_ols"
    assert model._inference_method == "post_selection_ols"


def test_penalty_object_alias_clone_is_warning_clean_and_stays_normalized():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error",
            penalty=get_penalty("l1", alpha=0.05),
            inference_method="cpu_ols",
            compute_inference=False,
            device="cpu",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cloned = clone(model)

    assert not [item for item in caught if issubclass(item.category, FutureWarning)]
    assert getattr(cloned.penalty, "name", None) == "l1"
    assert cloned.inference_method == "cpu_ols"
    assert cloned._inference_method == "post_selection_ols"


@pytest.mark.parametrize("penalty_name", ["l1", "elasticnet"])
def test_penalty_object_participates_in_pre_fit_auto_native_scope(
    monkeypatch,
    penalty_name,
):
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=get_penalty(penalty_name, alpha=0.05),
        inference_method="post_selection_ols",
        compute_inference=False,
        device="auto",
    )
    fake_cupy = object()

    monkeypatch.setattr(inference_contract, "_is_cupy_array", lambda value: value is fake_cupy)
    monkeypatch.setattr(inference_contract, "_is_torch_array", lambda value: False)
    monkeypatch.setattr(inference_contract, "_get_configured_device", lambda: Device.AUTO)

    assert inference_contract._supports_sparse_gaussian_migration(model) is True
    assert inference_contract._input_native_device(model, fake_cupy) == Device.CUDA


def test_pre_fit_scope_prefers_current_public_sparse_penalty_over_stale_resolved_state():
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l1",
        inference_method="post_selection_ols",
        compute_inference=False,
        device="auto",
    )
    model._penalty = get_penalty("l2", alpha=0.05)
    assert inference_contract._supports_sparse_gaussian_migration(model) is True


def test_pre_fit_scope_rejects_stale_sparse_state_after_public_penalty_changes_to_l2():
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        inference_method="post_selection_ols",
        compute_inference=False,
        device="auto",
    )
    model._penalty = get_penalty("l1", alpha=0.05)
    assert inference_contract._supports_sparse_gaussian_migration(model) is False


def test_lassocv_set_params_legacy_alias_warns_normalizes_and_invalidates():
    model = LassoCV(inference_method="post_selection_ols", compute_inference=False)
    model._fitted = True
    model.coef_ = np.ones(2, dtype=np.float64)
    model.estimator_ = object()

    with pytest.warns(FutureWarning, match="post_selection_ols") as caught:
        model.set_params(inference_method="gpu_ols_inference")

    assert len(caught) == 1
    assert model.inference_method == "gpu_ols_inference"
    assert model._inference_method == "post_selection_ols"
    assert model._fitted is False
    assert model.coef_ is None
    assert model.estimator_ is None


@pytest.mark.parametrize("cov_type", ["hc3", "hac"])
def test_empty_no_intercept_active_set_preserves_requested_robust_semantics(cov_type):
    rng = np.random.default_rng(5138)
    X = rng.normal(size=(100, 4))
    y = rng.normal(scale=0.25, size=100)

    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=100.0,
        fit_intercept=False,
        inference_method="post_selection_ols",
        compute_inference=True,
        cov_type=cov_type,
        hac_maxlags=2 if cov_type == "hac" else None,
        device="cpu",
        max_iter=2000,
        tol=1e-9,
    ).fit(X, y)

    assert np.count_nonzero(model.coef_) == 0
    result = model._inference_result
    assert result is not None
    assert result.method == "post_selection_ols"
    assert result.cov_type == cov_type
    assert result.distribution == "normal"
    assert result.statistic_name == "z"
    assert result.df is None
    assert result.metadata["n_selected"] == 0
    assert result.metadata["refit_parameter_count"] == 0
    assert result.metadata["refit_rank"] == 0
    assert result.metadata["refit_df_resid"] == X.shape[0]
    np.testing.assert_array_equal(model._bse, 0.0)
    np.testing.assert_array_equal(model._tvalues, 0.0)
    np.testing.assert_array_equal(model._zvalues, 0.0)
    np.testing.assert_array_equal(model._pvalues, 1.0)
    np.testing.assert_array_equal(model._conf_int, 0.0)


def _debiased_cpu_common():
    return dict(
        penalty="l1",
        alpha=0.045,
        fit_intercept=True,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        device="cpu",
        max_iter=4000,
        tol=1e-9,
    )


def _debiased_data(seed=6138):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(140, 5))
    beta = np.asarray([1.2, -0.8, 0.45, 0.0, 0.0])
    y = 0.35 + X @ beta + rng.normal(scale=0.45, size=X.shape[0])
    weights = rng.uniform(0.35, 1.9, size=X.shape[0])
    return X, y, weights


def _assert_debiased_results_close(left, right, *, atol=1e-11):
    np.testing.assert_allclose(left.coef_, right.coef_, rtol=1e-10, atol=atol)
    assert left.intercept_ == pytest.approx(right.intercept_, abs=atol)
    np.testing.assert_allclose(left._params, right._params, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(left._bse, right._bse, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(left._tvalues, right._tvalues, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(left._pvalues, right._pvalues, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(left._conf_int, right._conf_int, rtol=1e-10, atol=atol)


def test_debiased_all_ones_weights_match_unweighted_intercept_inference():
    X, y, _ = _debiased_data(seed=6137)
    common = _debiased_cpu_common()
    unweighted = PenalizedLinearRegression(**common).fit(X, y)
    ones = PenalizedLinearRegression(**common).fit(
        X,
        y,
        sample_weight=np.ones(X.shape[0], dtype=np.float64),
    )

    assert unweighted._inference_result.metadata["sample_weighted"] is False
    assert unweighted._inference_result.metadata["backend_path"] == "cpu_debiased"
    assert ones._inference_result.metadata["sample_weighted"] is True
    assert ones._inference_result.metadata["backend_path"] == "numpy_debiased_weighted"
    _assert_debiased_results_close(ones, unweighted)


def test_weighted_cpu_debiased_is_invariant_to_global_weight_scaling():
    X, y, weights = _debiased_data()
    common = _debiased_cpu_common()
    reference = PenalizedLinearRegression(**common).fit(
        X,
        y,
        sample_weight=weights,
    )
    scaled = PenalizedLinearRegression(**common).fit(
        X,
        y,
        sample_weight=17.0 * weights,
    )

    for model in (reference, scaled):
        result = model._inference_result
        assert result is not None
        assert result.method == "debiased"
        assert result.metadata["backend_path"] == "numpy_debiased_weighted"
        assert result.metadata["sample_weighted"] is True
        assert result.metadata["numerical_backend"] == "numpy"
        assert result.metadata["reporting_boundary"] == "post_numerical_inference"

    _assert_debiased_results_close(scaled, reference)
