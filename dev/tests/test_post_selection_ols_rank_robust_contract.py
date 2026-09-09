import numpy as np
import pytest

from statgpu.linear_model import PenalizedLinearRegression
from statgpu.linear_model.penalized._post_selection_ols import (
    compute_post_selection_ols_inference,
)


def test_rank_deficient_hc3_refit_matches_statsmodels_design_svd():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(10138)
    n = 110
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    X = np.column_stack([x, x, z])
    y = 0.65 + 1.7 * x - 0.35 * z + rng.normal(scale=0.3, size=n)

    model = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.04,
        fit_intercept=True,
        inference_method="post_selection_ols",
        compute_inference=False,
        cov_type="hc3",
        device="cpu",
        max_iter=4000,
        tol=1e-9,
    ).fit(X, y)
    model.coef_ = np.asarray([0.2, 0.2, 0.1], dtype=np.float64)
    model._selected_backend_name = "numpy"
    model._selected_backend_device = "cpu"
    model.inference_method = "post_selection_ols"
    model._inference_method = "post_selection_ols"

    compute_post_selection_ols_inference(model, X, y)

    design = np.column_stack([np.ones(n), X])
    reference = sm.OLS(y, design).fit(cov_type="HC3")
    rank = int(np.linalg.matrix_rank(design))
    meta = model._inference_result.metadata

    assert rank == 3 < design.shape[1]
    assert meta["refit_rank"] == rank
    assert meta["refit_parameter_count"] == design.shape[1]
    assert meta["refit_rank_deficient"] is True
    assert model._inference_result.cov_type == "hc3"
    assert model._inference_result.distribution == "normal"
    assert model._inference_result.statistic_name == "z"
    assert model._inference_result.df is None

    np.testing.assert_allclose(model._params, reference.params, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(model._bse, reference.bse, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(model._pvalues, reference.pvalues, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(
        model._conf_int,
        reference.conf_int(),
        rtol=2e-8,
        atol=2e-10,
    )
