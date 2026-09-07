import numpy as np
import pytest

from statgpu.linear_model import Lasso


def test_weighted_debiased_simultaneous_intercept_uses_intercept_in_maxz():
    rng = np.random.default_rng(91381)
    n = 84
    X = rng.normal(size=(n, 3))
    y = 0.8 + X @ np.array([1.0, -0.6, 0.35]) + rng.normal(scale=0.45, size=n)
    weight = rng.uniform(0.3, 1.9, size=n)
    B = 96
    seed = 20260908

    model = Lasso(
        alpha=0.045,
        fit_intercept=True,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        device="cpu",
        max_iter=5000,
        tol=1e-9,
        enable_simultaneous_inference=True,
        simultaneous_include_intercept=True,
        simultaneous_n_bootstrap=B,
        simultaneous_random_state=seed,
    ).fit(X, y, sample_weight=weight)

    influence = np.asarray(
        model._debiased_intercept_influence_cpu,
        dtype=np.float64,
    ).reshape(-1)
    X_design = np.asarray(model._X_design, dtype=np.float64)
    X_feat = X_design[:, 1:]
    resid = np.asarray(model._resid, dtype=np.float64).reshape(-1)
    M = np.asarray(model._debiased_M_cpu, dtype=np.float64)
    bse = np.asarray(model._bse, dtype=np.float64)

    bootstrap_rng = np.random.default_rng(seed)
    xi = bootstrap_rng.standard_normal(size=(B, n))
    multiplier_resid = xi * resid.reshape(1, -1)
    feature_score = (multiplier_resid @ X_feat) @ M.T / float(n)
    z_feature = feature_score / (bse[1:].reshape(1, -1) + 1e-30)
    intercept_score = multiplier_resid @ influence
    z_intercept = intercept_score / (float(bse[0]) + 1e-30)
    expected_max = np.maximum(
        np.abs(z_intercept),
        np.max(np.abs(z_feature), axis=1),
    )
    expected_critical = float(
        np.quantile(expected_max, 1.0 - model.simultaneous_alpha)
    )

    assert model._simultaneous_critical_value == pytest.approx(
        expected_critical,
        rel=0,
        abs=1e-12,
    )
    np.testing.assert_array_equal(
        model._simultaneous_target_mask,
        np.ones(model._params.shape[0], dtype=bool),
    )
    expected_ci = np.column_stack(
        [
            model._params - expected_critical * model._bse,
            model._params + expected_critical * model._bse,
        ]
    )
    np.testing.assert_allclose(
        model._conf_int_simultaneous,
        expected_ci,
        rtol=0,
        atol=1e-12,
    )

    result = model._inference_result
    assert result.simultaneous_critical_value == pytest.approx(
        expected_critical,
        rel=0,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        result.simultaneous_conf_int,
        expected_ci,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        result.simultaneous_target_mask,
        np.ones(model._params.shape[0], dtype=bool),
    )
