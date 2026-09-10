import numpy as np
import pytest

from statgpu.linear_model import Lasso, PenalizedLinearRegression


def _frame(seed=44, n=90):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    beta = np.array([1.0, -0.65, 0.35, 0.0])
    y = X @ beta + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3]}), X, y


def _kwargs():
    return dict(
        alpha=0.05,
        nodewise_alpha=0.08,
        fit_intercept=True,
        device="cpu",
        solver="fista",
        max_iter=2000,
        tol=1e-7,
        compute_inference=True,
        inference_method="debiased",
    )


def test_lasso_array_formula_nodewise_precision_parity():
    df, X, y = _frame()
    array_model = Lasso(**_kwargs()).fit(X, y)
    formula_model = Lasso(**_kwargs()).fit(
        formula="y ~ x1 + x2 + x3 + x4",
        data=df,
    )
    assert formula_model.nodewise_alpha_ == pytest.approx(array_model.nodewise_alpha_)
    np.testing.assert_allclose(formula_model.coef_, array_model.coef_, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(formula_model._debiased_M_cpu, array_model._debiased_M_cpu, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(formula_model._params, array_model._params, rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(formula_model._bse, array_model._bse, rtol=3e-6, atol=3e-8)


def test_generic_penalized_formula_preserves_requested_nodewise_alpha():
    df, X, y = _frame(seed=45)
    model = PenalizedLinearRegression(
        penalty="l1",
        **_kwargs(),
    ).fit(formula="y ~ x1 + x2 + x3 + x4", data=df)
    assert model.nodewise_alpha == pytest.approx(0.08)
    assert model.nodewise_alpha_ == pytest.approx(0.08)
    assert model._inference_result.metadata["nodewise_alpha_source"] == "user"
