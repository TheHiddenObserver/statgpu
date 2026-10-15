"""Regression tests for GLM Formula names in formatted summaries."""

from __future__ import annotations

import numpy as np
import pytest


def test_poisson_formula_summary_uses_design_term_names():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.linear_model import PoissonRegression

    rng = np.random.default_rng(17)
    x1 = rng.normal(size=120)
    x2 = rng.normal(size=120)
    mean = np.exp(0.2 + 0.35 * x1 - 0.15 * x2)
    frame = pd.DataFrame(
        {
            "count": rng.poisson(mean),
            "x1": x1,
            "x2": x2,
        }
    )

    model = PoissonRegression(
        solver="newton",
        device="cpu",
        compute_inference=True,
        cov_type="hc1",
    ).fit(formula="count ~ x1 + x2", data=frame)

    assert model._inference_result.feature_names == ["Intercept", "x1", "x2"]
    summary = model.summary()
    assert "Intercept" in summary
    assert "x1" in summary
    assert "x2" in summary
    assert "Solver: newton" in summary
    assert "Covariance Type: hc1" in summary


def test_glm_summary_reports_resolved_auto_solver():
    from statgpu.linear_model import PoissonRegression

    X = np.arange(40, dtype=float).reshape(-1, 1) / 40.0
    y = np.asarray(([0, 1, 1, 2, 1] * 8), dtype=float)
    model = PoissonRegression(
        solver="auto",
        C=0,
        device="cpu",
    ).fit(X, y)

    assert "Solver: irls" in model.summary()


@pytest.mark.parametrize("solver", ["auto", "newton", "lbfgs"])
@pytest.mark.parametrize(
    "formula, fit_intercept",
    [
        ("count ~ x1 + x2", False),
        ("count ~ 0 + x1 + x2", True),
        ("count ~ C(group) * x1 + I(x2 ** 2)", False),
    ],
)
def test_formula_inference_names_match_design_and_array_results(formula, fit_intercept, solver):
    pd = pytest.importorskip("pandas")
    patsy = pytest.importorskip("patsy")
    from statgpu.linear_model import PoissonRegression

    rng = np.random.default_rng(716)
    frame = pd.DataFrame({
        "x1": rng.normal(scale=0.3, size=120),
        "x2": rng.normal(scale=0.3, size=120),
        "group": pd.Categorical(["a", "b", "c"] * 40),
    })
    frame["count"] = rng.poisson(np.exp(0.1 + 0.2 * frame["x1"]))
    frame.loc[7, "x2"] = np.nan
    weights = np.linspace(0.7, 1.3, len(frame))
    response, design = patsy.dmatrices(formula, frame, return_type="dataframe")
    expected_names = list(design.columns)
    has_intercept = "Intercept" in expected_names
    features = design.drop(columns=["Intercept"]) if has_intercept else design
    common = dict(solver=solver, C=0, device="cpu", compute_inference=True,
                  cov_type="hc1", max_iter=600, tol=1e-9)
    model = PoissonRegression(fit_intercept=fit_intercept, **common).fit(
        formula=formula, data=frame, sample_weight=weights
    )
    array_model = PoissonRegression(fit_intercept=has_intercept, **common).fit(
        features.to_numpy(), np.asarray(response).ravel(),
        sample_weight=weights[design.index.to_numpy()],
    )

    assert model._inference_result.feature_names == expected_names
    assert model._feature_names == list(features.columns)
    assert model._inference_result.to_dataframe()["term"].tolist() == expected_names
    assert model._nobs == len(design)
    for name in expected_names:
        assert name in model.summary()
    for attr in ("_params", "_bse", "_zvalues", "_pvalues", "_conf_int"):
        np.testing.assert_allclose(getattr(model, attr), getattr(array_model, attr),
                                   rtol=1e-12, atol=1e-12)
    for attr in ("loglikelihood", "aic", "bic"):
        assert getattr(model, attr) == pytest.approx(getattr(array_model, attr))


@pytest.mark.parametrize("solver", ["AUTO", "IRLS", "FISTA", "NEWTON", "LBFGS"])
@pytest.mark.parametrize("compute_inference", [False, True])
def test_summary_reports_fit_solver_with_or_without_inference(solver, compute_inference):
    from statgpu.linear_model import PoissonRegression

    X = np.arange(40, dtype=float).reshape(-1, 1) / 40.0
    y = np.asarray([0, 1, 1, 2, 1] * 8, dtype=float)
    model = PoissonRegression(solver=solver, C=0, device="cpu",
                              compute_inference=compute_inference).fit(X, y)
    resolved = "irls" if solver == "AUTO" else solver.lower()
    assert f"Solver: {resolved}" in model.summary()
    assert model.solver == solver
    assert model._fit_metadata["solver_used"] == resolved
    assert model._selected_solver == resolved
    assert model._selected_backend_name == "numpy"
    assert model._selected_backend_device == "cpu"
    if compute_inference:
        assert model._inference_result.feature_names is None
        assert model._inference_result.metadata["solver_used"] == resolved
    else:
        assert model._inference_result is None
        assert "inference not computed" in model.summary()


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson", "gamma",
                                  "inverse_gaussian", "negative_binomial", "tweedie"])
@pytest.mark.parametrize("compute_inference", [False, True])
def test_reporting_shared_ordinary_glm_consumers(kind, compute_inference):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.linear_model import (GeneralizedLinearModel, PoissonRegression,
        GammaRegression, InverseGaussianRegression, NegativeBinomialRegression,
        TweedieRegression)

    rng = np.random.default_rng(612)
    x = rng.normal(scale=0.2, size=120)
    mu = np.exp(0.1 + 0.2 * x)
    if kind == "gaussian":
        factory = lambda **kw: GeneralizedLinearModel(family="gaussian", **kw)
        y = 0.1 + 0.2 * x + rng.normal(scale=0.1, size=len(x))
    elif kind == "binomial":
        factory = lambda **kw: GeneralizedLinearModel(family="binomial", **kw)
        y = rng.binomial(1, 0.5, size=len(x))
    else:
        factory = {
            "poisson": PoissonRegression,
            "gamma": lambda **kw: GammaRegression(link="log", **kw),
            "inverse_gaussian": InverseGaussianRegression,
            "negative_binomial": NegativeBinomialRegression,
            "tweedie": lambda **kw: TweedieRegression(power=1.5, **kw),
        }[kind]
        y = (rng.poisson(mu) if kind in ("poisson", "negative_binomial")
             else mu * rng.lognormal(0.0, 0.05, size=len(x)))
    model = factory(solver="auto", C=0, device="cpu",
                    compute_inference=compute_inference).fit(
        formula="response ~ feature", data=pd.DataFrame({"response": y, "feature": x})
    )
    assert "Solver: irls" in model.summary()
    if compute_inference:
        assert model._inference_result.feature_names == ["Intercept", "feature"]
        assert "feature" in model.summary()


def test_refit_refreshes_solver_metadata_without_inference():
    from statgpu.linear_model import PoissonRegression

    X = np.arange(40, dtype=float).reshape(-1, 1) / 40.0
    y = np.asarray([0, 1, 1, 2, 1] * 8, dtype=float)
    model = PoissonRegression(solver="newton", C=0, device="cpu").fit(X, y)
    model.set_params(solver="auto")
    model.fit(X, y)
    assert model._fit_metadata["solver_used"] == "irls"
    assert "Solver: irls" in model.summary()


def test_formula_to_array_refit_does_not_retain_term_names():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.linear_model import PoissonRegression

    X = np.arange(40, dtype=float).reshape(-1, 1) / 40.0
    y = np.asarray([0, 1, 1, 2, 1] * 8, dtype=float)
    model = PoissonRegression(solver="newton", device="cpu", compute_inference=True)
    model.fit(formula="y ~ original", data=pd.DataFrame({"y": y, "original": X[:, 0]}))
    names = model._inference_result.feature_names
    assert names == ["Intercept", "original"]
    assert model._feature_names == ["original"]
    model.fit(X, y)
    assert names == ["Intercept", "original"]
    assert model._feature_names is None
    assert model._inference_result.feature_names is None
    assert "original" not in model.summary()


def test_irls_reporting_preserves_ridge_inference_curvature():
    from statgpu.linear_model import PoissonRegression

    X = np.arange(40, dtype=float).reshape(-1, 1) / 40.0
    y = np.asarray([0, 1, 1, 2, 1] * 8, dtype=float)
    model = PoissonRegression(solver="auto", C=2, device="cpu",
                              compute_inference=True).fit(X, y)
    assert model._fit_metadata["solver_used"] == "irls"
    assert model._fit_metadata["ridge_alpha_avg"] == 0.25
    np.testing.assert_array_equal(model._fit_metadata["penalty_curvature_diag"], [0, 0.25])
    assert "Solver: irls" in model.summary()
