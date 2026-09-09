import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import Lasso, LassoCV
import statgpu.linear_model._penalized_inference_api_contract as inference_contract
import statgpu.linear_model.wrappers._lasso as lasso_wrapper


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


def test_rank_deficient_active_refit_uses_effective_rank_df():
    sm = pytest.importorskip("statsmodels.api")
    from statgpu.linear_model.penalized._post_selection_ols import (
        compute_post_selection_ols_inference,
    )

    rng = np.random.default_rng(26)
    n = 90
    x = rng.normal(size=n)
    X = np.column_stack([x, x])
    y = 0.75 + 1.8 * x + rng.normal(scale=0.3, size=n)

    model = Lasso(
        alpha=0.05,
        compute_inference=False,
        fit_intercept=True,
        device="cpu",
        max_iter=4000,
        tol=1e-9,
    ).fit(X, y)
    # Force the post-selection diagnostic to include both exactly collinear
    # coordinates. This isolates the refit rank contract from Lasso's choice of
    # one representative coordinate in a particular solver run.
    model.coef_ = np.asarray([0.2, 0.2], dtype=np.float64)
    model._selected_backend_name = "numpy"
    model._selected_backend_device = "cpu"
    model.inference_method = "post_selection_ols"
    model._inference_method = "post_selection_ols"

    compute_post_selection_ols_inference(model, X, y)

    design = np.column_stack([np.ones(n), X])
    reference = sm.OLS(y, design).fit()
    rank = int(np.linalg.matrix_rank(design))
    assert rank == 2 < design.shape[1]
    assert model._post_selection_df_resid == n - rank
    assert model._inference_result.df == pytest.approx(float(n - rank))
    meta = model._inference_result.metadata
    assert meta["refit_rank"] == rank
    assert meta["refit_parameter_count"] == design.shape[1]
    assert meta["refit_rank_deficient"] is True

    np.testing.assert_allclose(model._params, reference.params, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(model._bse, reference.bse, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(model._pvalues, reference.pvalues, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        model._conf_int, reference.conf_int(), rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize(
    "target,device_name",
    [(Device.CUDA, "cuda"), (Device.TORCH, "torch")],
)
def test_lassocv_auto_native_target_pins_cv_and_final_refit(
    monkeypatch,
    target,
    device_name,
):
    X = np.zeros((8, 2), dtype=np.float64)
    y = np.zeros(8, dtype=np.float64)
    model = LassoCV(
        alphas=np.asarray([0.05], dtype=np.float64),
        cv=2,
        compute_inference=False,
        device="auto",
    )
    captured = {}

    monkeypatch.setattr(
        inference_contract,
        "_input_native_device",
        lambda estimator, value: target,
    )
    monkeypatch.setattr(model, "_get_compute_device", lambda: model._device)

    def fake_prepare(X_arg, y_arg, sw_arg, resolved_device):
        captured["prepare_device"] = resolved_device
        return X_arg, y_arg, sw_arg

    monkeypatch.setattr(model, "_prepare_cv_inputs_for_resolved_device", fake_prepare)

    def fake_select(X_arg, y_arg, **kwargs):
        captured["cv_device"] = kwargs["device"]
        return {
            "alpha": 0.05,
            "alphas": np.asarray([0.05], dtype=np.float64),
            "mse_path": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "mean_mse": np.asarray([1.0], dtype=np.float64),
        }

    monkeypatch.setattr(lasso_wrapper, "_select_lasso_alpha_cv", fake_select)

    def fake_lasso_fit(self, X_arg, y_arg, sample_weight=None):
        captured["final_device"] = self._device
        self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self.n_iter_ = 1
        self._fitted = True
        return self

    monkeypatch.setattr(Lasso, "fit", fake_lasso_fit)

    model.fit(X, y)

    assert captured["prepare_device"] == device_name
    assert captured["cv_device"] == device_name
    assert captured["final_device"] == target
    assert model.estimator_._device == target
    assert model._device == Device.AUTO
