import numpy as np
import pytest

from statgpu.linear_model import Lasso


def _data(seed=24, n=180, p=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [1.4, -0.9, 0.65]
    y = 0.35 + X @ beta + rng.normal(scale=0.5, size=n)
    return X, y


def test_post_selection_summary_reports_both_fit_and_refit_df(capsys):
    X, y = _data()
    model = Lasso(
        alpha=0.2,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=5000,
        tol=1e-9,
    ).fit(X, y)

    assert model._post_selection_df_resid != model._df_resid
    assert model._inference_result.df == pytest.approx(
        float(model._post_selection_df_resid)
    )

    model.summary()
    output = capsys.readouterr().out
    fit_line = next(
        line
        for line in output.splitlines()
        if line.startswith("Penalized-fit Residual DoF:")
    )
    refit_line = next(
        line
        for line in output.splitlines()
        if line.startswith("Post-selection Refit DoF:")
    )
    assert int(fit_line.split(":", 1)[1]) == model._df_resid
    assert int(refit_line.split(":", 1)[1]) == model._post_selection_df_resid
    assert "Penalized-fit diagnostics:" in output


def test_post_selection_formula_weight_alignment_and_feature_names():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.core.formula import FormulaParser

    rng = np.random.default_rng(25)
    n = 140
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    group = np.resize(np.asarray(["a", "b", "c"], dtype=object), n)
    y = (
        0.4
        + 1.1 * x1
        - 0.7 * x2
        + 0.35 * (group == "b")
        - 0.2 * (group == "c")
        + rng.normal(scale=0.35, size=n)
    )
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})
    frame.loc[[7, 83], "x2"] = np.nan
    weights = rng.uniform(0.3, 2.0, size=n)
    formula_text = "y ~ x1 + x2 + C(group)"

    common = dict(
        alpha=0.08,
        inference_method="post_selection_ols",
        compute_inference=True,
        device="cpu",
        max_iter=5000,
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
        np.asarray(X_design, dtype=np.float64), intercept_pos, axis=1
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
        formula_model.coef_, direct_model.coef_, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        formula_model._params, direct_model._params, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        formula_model._bse, direct_model._bse, rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        formula_model._pvalues, direct_model._pvalues, rtol=0, atol=1e-11
    )
    expected_report_names = [
        "(Intercept)" if name == "Intercept" else name for name in design_names
    ]
    assert formula_model._inference_result.feature_names == expected_report_names
    assert formula_model._post_selection_nobs == y_direct.shape[0] == n - 2
    assert (
        formula_model._inference_result.metadata["selected_feature_indices"]
        == direct_model._inference_result.metadata["selected_feature_indices"]
    )
