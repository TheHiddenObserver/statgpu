import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import PenalizedGeneralizedLinearModel, PenalizedLinearRegression
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
