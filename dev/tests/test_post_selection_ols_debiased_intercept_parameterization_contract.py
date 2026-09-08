import numpy as np
import pytest

from statgpu.linear_model import Lasso


def _problem(seed=20260908):
    rng = np.random.default_rng(seed)
    n = 128
    X = rng.normal(size=(n, 4)) + np.array([2.0, -1.25, 0.75, 1.5])
    beta = np.array([1.1, -0.8, 0.45, 0.2])
    y = 1.35 + X @ beta + rng.normal(scale=0.5, size=n)
    weight = rng.uniform(0.35, 1.9, size=n)
    return X, y, weight


def _fit(X, y, weight, *, simultaneous=False):
    return Lasso(
        alpha=0.12,
        fit_intercept=True,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        device="cpu",
        max_iter=6000,
        tol=1e-9,
        enable_simultaneous_inference=simultaneous,
        simultaneous_include_intercept=simultaneous,
        simultaneous_n_bootstrap=48,
        simultaneous_random_state=20260908,
    ).fit(X, y, sample_weight=weight)


def test_debiased_reporting_intercept_obeys_feature_translation_identity():
    X, y, weight = _problem()
    shift = np.array([0.7, -1.1, 0.4, 1.6])

    original = _fit(X, y, weight)
    translated = _fit(X + shift, y, weight)

    # Prediction remains owned by the penalized fit.
    np.testing.assert_allclose(
        translated.coef_,
        original.coef_,
        rtol=0,
        atol=2e-8,
    )
    assert translated.intercept_ == pytest.approx(
        original.intercept_ - float(shift @ original.coef_),
        rel=0,
        abs=2e-8,
    )

    # Inference reporting is a different estimator: the intercept must transform
    # with the debiased slope vector that shares its _params layout.
    np.testing.assert_allclose(
        translated._params[1:],
        original._params[1:],
        rtol=0,
        atol=2e-8,
    )
    assert translated._params[0] == pytest.approx(
        original._params[0] - float(shift @ original._params[1:]),
        rel=0,
        abs=3e-8,
    )

    assert original._inference_result.metadata["intercept_estimator"] == "centered_debiased"
    assert original._inference_result.metadata["intercept_influence"] == "centered_nodewise"
    assert not hasattr(original, "_debiased_intercept_influence_cpu")


def test_weighted_debiased_intercept_and_se_match_centered_nodewise_influence():
    X, y, weight = _problem(seed=20260909)
    model = _fit(X, y, weight, simultaneous=True)

    n = X.shape[0]
    n_eff = float(np.sum(weight))
    x_mean = np.sum(weight[:, None] * X, axis=0) / n_eff
    y_mean = float(np.sum(weight * y) / n_eff)
    row_scale = np.sqrt(weight * (float(n) / n_eff))
    X_work = (X - x_mean) * row_scale[:, None]
    y_work = (y - y_mean) * row_scale

    theta_db = np.asarray(model._params[1:], dtype=np.float64)
    expected_intercept = y_mean - float(x_mean @ theta_db)
    assert model._params[0] == pytest.approx(
        expected_intercept,
        rel=0,
        abs=2e-10,
    )

    M = np.asarray(model._debiased_M_cpu, dtype=np.float64)
    q = row_scale - X_work @ (M.T @ x_mean)
    expected_influence = q / float(n)
    np.testing.assert_allclose(
        model._debiased_intercept_influence_cpu,
        expected_influence,
        rtol=0,
        atol=2e-12,
    )

    coef_pen = np.asarray(model.coef_, dtype=np.float64)
    resid_work = y_work - X_work @ coef_pen
    s_hat = int(np.sum(np.abs(coef_pen) > 0))
    scale = float(np.sum(resid_work * resid_work) / max(n - s_hat, 1))
    expected_se = float(np.sqrt(scale * np.sum(q * q)) / n)
    assert model._bse[0] == pytest.approx(
        expected_se,
        rel=0,
        abs=2e-10,
    )

    # The fitted prediction intercept is intentionally not overwritten by the
    # debiased reporting intercept.
    expected_penalized_intercept = y_mean - float(x_mean @ coef_pen)
    assert model.intercept_ == pytest.approx(
        expected_penalized_intercept,
        rel=0,
        abs=2e-10,
    )


def test_weighted_debiased_formula_matches_retained_matrix_problem():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.core.formula import FormulaParser

    rng = np.random.default_rng(20260912)
    n = 150
    x1 = rng.normal(size=n) + 1.3
    x2 = rng.normal(size=n) - 0.7
    group = np.resize(np.asarray(["a", "b", "c"], dtype=object), n)
    y = (
        0.8
        + 1.0 * x1
        - 0.6 * x2
        + 0.3 * (group == "b")
        - 0.15 * (group == "c")
        + rng.normal(scale=0.4, size=n)
    )
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})
    frame.loc[[11, 76, 121], "x2"] = np.nan
    weights = rng.uniform(0.25, 2.1, size=n)
    formula_text = "y ~ x1 + x2 + C(group)"
    common = dict(
        alpha=0.1,
        fit_intercept=True,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        device="cpu",
        max_iter=6000,
        tol=1e-9,
    )

    formula_model = Lasso(**common).fit(
        formula=formula_text,
        data=frame,
        sample_weight=weights,
    )

    parser = FormulaParser(formula_text)
    y_design, X_design, design_info = parser.eval(frame)
    design_names = list(design_info.column_names)
    intercept_pos = design_names.index("Intercept")
    X_direct = np.delete(
        np.asarray(X_design, dtype=np.float64),
        intercept_pos,
        axis=1,
    )
    y_direct = np.asarray(y_design, dtype=np.float64).reshape(-1)
    aligned_weights = np.asarray(weights, dtype=np.float64)[
        np.asarray(parser.row_positions, dtype=np.int64)
    ]
    direct_model = Lasso(**common).fit(
        X_direct,
        y_direct,
        sample_weight=aligned_weights,
    )

    np.testing.assert_allclose(
        formula_model.coef_, direct_model.coef_, rtol=0, atol=2e-11
    )
    assert formula_model.intercept_ == pytest.approx(
        direct_model.intercept_, rel=0, abs=2e-11
    )
    np.testing.assert_allclose(
        formula_model._params, direct_model._params, rtol=0, atol=3e-10
    )
    np.testing.assert_allclose(
        formula_model._bse, direct_model._bse, rtol=0, atol=3e-10
    )
    np.testing.assert_allclose(
        formula_model._pvalues, direct_model._pvalues, rtol=0, atol=3e-10
    )
    np.testing.assert_allclose(
        formula_model._conf_int, direct_model._conf_int, rtol=0, atol=3e-10
    )
    expected_names = [
        "(Intercept)" if name == "Intercept" else name for name in design_names
    ]
    assert formula_model._inference_result.feature_names == expected_names
    assert formula_model._nobs == y_direct.shape[0] == n - 3
    assert formula_model._inference_result.metadata["intercept_estimator"] == "centered_debiased"
