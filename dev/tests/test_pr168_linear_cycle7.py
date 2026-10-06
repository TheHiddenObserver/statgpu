"""Seventh-pass scalar Gaussian learner examples and public API inventories.

These CPU checks verify documented behavior, not GPU precision or interval
coverage. No production behavior is changed by the documentation repair.
"""
from __future__ import annotations

import inspect
import re
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from statgpu.linear_model import MCPRegression, Ridge, SCADRegression

ROOT = Path(__file__).resolve().parents[2]
CLASSES = [(Ridge, "ridge"), (SCADRegression, "scad"), (MCPRegression, "mcp")]


def _page(language, directory, page):
    return (ROOT / "docs" / language / directory / f"{page}.md").read_text()


def _example(language, page, label):
    text = _page(language, "models", page)
    match = re.search(r"<!-- learner-example: " + re.escape(label)
                      + r" -->\s*```python\n(.*?)```", text, re.DOTALL)
    assert match is not None
    scope = {}
    exec(compile(match[1], f"{language}/{page}/{label}", "exec"), scope)  # noqa: S102
    return scope


def _signature(cls):
    signature = inspect.signature(cls)
    parameters = [parameter.replace(
        annotation=inspect.Parameter.empty,
        default=(parameter.default.value if isinstance(parameter.default, Enum)
                 else parameter.default),
    ) for parameter in signature.parameters.values()]
    return cls.__name__ + str(signature.replace(
        parameters=parameters, return_annotation=inspect.Signature.empty,
    ))


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("cls,page", CLASSES)
def test_scalar_gaussian_examples_execute_independently(language, cls, page):
    label = "ridge-weighted-prediction" if cls is Ridge else page + "-prediction"
    ns = _example(language, page, label)
    model, X, y = ns["model"], ns["X"], ns["y"]
    assert isinstance(model, cls)
    assert ns["prediction"].shape == (40,)
    np.testing.assert_allclose(ns["prediction"], model.intercept_ + X[120:] @ model.coef_)
    expected = [1.821, -.916, .017, -.010, -.020] if cls is Ridge else [1.987, -.982, 0, 0, 0]
    np.testing.assert_allclose(model.coef_, expected, atol=.0005)
    assert model.score(X[120:], y[120:]) == pytest.approx(.957 if cls is Ridge else .967, abs=.0005)
    if cls is Ridge:
        assert model._conf_int.shape == (6, 2)
        assert np.isfinite(model._conf_int).all()
    else:
        assert model.intercept_ == pytest.approx(1.430, abs=.0005)
        assert model.compute_inference is False
        with pytest.raises(RuntimeError, match="compute_inference=False"):
            model.summary()


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("cls,page", CLASSES)
def test_scalar_gaussian_api_references_cover_live_public_interfaces(language, cls, page):
    text = _page(language, "reference", "linear-model-api")
    section = text.split(f"## {cls.__name__}\n", 1)[1].split("\n## ", 1)[0]
    assert _signature(cls) in section
    assert f"../models/{page}.md#" in section
    model_page = _page(language, "models", page)
    for parameter in inspect.signature(cls).parameters:
        assert f"| `{parameter}` |" in model_page, parameter
    for name in dir(cls):
        if name.startswith("_"):
            continue
        member = inspect.getattr_static(cls, name)
        if callable(member) or isinstance(member, property):
            assert name in section, name
    assert f"linear-model-api.md#{cls.__name__.lower()}" in model_page


@pytest.mark.parametrize("cls,page", CLASSES)
def test_scalar_gaussian_installed_help_has_complete_constructor_and_methods(cls, page):
    text = inspect.getdoc(cls)
    for parameter in inspect.signature(cls).parameters:
        assert re.search(r"^" + parameter + r"\s*:", text, re.MULTILINE), parameter
    for name in ["fit(", "predict(", "score(", "summary()", "get_params(",
                 "set_params(", "adjust_pvalues", "combine_pvalues",
                 "bootstrap_statistic", "permutation_test"]:
        assert name in text


@pytest.mark.parametrize("cls,page", CLASSES)
def test_scalar_gaussian_formula_weights_match_retained_array_rows(cls, page):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(71)
    X = rng.normal(size=(50, 2))
    y = X @ np.array([2., -1.]) + rng.normal(scale=.2, size=len(X))
    frame = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})
    frame.loc[[1, 4], "x1"] = np.nan
    weight = np.linspace(.5, 2., len(X))
    keep = np.ones(len(X), dtype=bool)
    keep[[1, 4]] = False
    kwargs = {"alpha": .1, "device": "cpu", "compute_inference": False,
              "max_iter": 5000, "tol": 1e-9}
    formula = cls(**kwargs).fit(formula="y ~ 0 + x1 + x2", data=frame, sample_weight=weight)
    arrays = cls(**kwargs, fit_intercept=False).fit(X[keep], y[keep], sample_weight=weight[keep])
    assert formula.fit_intercept is True  # Formula changes the fit, not the constructor.
    assert formula.intercept_ == 0
    np.testing.assert_allclose(formula.coef_, arrays.coef_, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(formula.predict(frame[keep]), arrays.predict(X[keep]), atol=1e-7)
    with pytest.raises(ValueError, match="missing values"):
        formula.predict(frame)


@pytest.mark.parametrize("cls", [SCADRegression, MCPRegression])
def test_nonconvex_documented_intercept_and_weight_normalization(cls):
    rng = np.random.default_rng(64)
    X = rng.normal(size=(100, 5))
    y = 1.5 + 2*X[:, 0] - X[:, 1] + rng.normal(scale=.4, size=len(X))
    w = np.linspace(.5, 2., len(X))
    args = {"alpha": .1, "device": "cpu", "max_iter": 5000, "tol": 1e-9}
    model = cls(**args).fit(X, y, sample_weight=w)
    scaled = cls(**args).fit(X, y, sample_weight=7*w)
    np.testing.assert_allclose(model.coef_, scaled.coef_, atol=1e-8, rtol=1e-8)
    assert model.intercept_ == pytest.approx(scaled.intercept_, abs=1e-8)
    residual_score = w * (model.predict(X) - y) / w.sum()
    assert abs(residual_score.sum()) < 1e-8  # Unpenalized intercept score.
    active = np.flatnonzero(abs(model.coef_) > 1e-8)
    # The selected slopes exceed the flat-penalty threshold for these examples.
    threshold = model.alpha * (model.a if cls is SCADRegression else model.gamma)
    assert np.all(abs(model.coef_[active]) > threshold)
    np.testing.assert_allclose((X.T @ residual_score)[active], 0, atol=1e-7)
    inactive = np.flatnonzero(abs(model.coef_) <= 1e-8)
    assert np.all(abs((X.T @ residual_score)[inactive]) <= model.alpha + 1e-7)


@pytest.mark.parametrize("cls,parameter,boundary", [(SCADRegression, "a", 2.),
                                                   (MCPRegression, "gamma", 1.)])
def test_nonconvex_parameter_bounds_and_import_locations(cls, parameter, boundary):
    import statgpu
    assert not hasattr(statgpu, cls.__name__)
    with pytest.raises(ValueError, match=parameter + " must be"):
        cls(device="cpu", **{parameter: boundary}).fit(np.arange(6.)[:, None], np.arange(6.))
    with pytest.raises(ValueError, match="alpha must be"):
        cls(alpha=0, device="cpu").fit(np.arange(6.)[:, None], np.arange(6.))


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("page", ["scad", "mcp"])
def test_nonconvex_pages_keep_intercept_formula_and_qualified_bias(language, page):
    text = _page(language, "models", page)
    assert r"\min_{b,\beta}" in text
    assert r"b\mathbf{1}" in text
    assert "fit_intercept=False" in text
    assert "| Nearly unbiased |" not in text
    assert "几乎无偏" not in text
    assert "sample_weight" in _page(language, "reference", "linear-model-api")


@pytest.mark.parametrize("language", ["en", "cn"])
def test_ridge_lasso_cv_example_has_true_loss_schema_and_score_sign(language):
    text = _page(language, "reference", "linear-model-api")
    match = re.search(r"<!-- api-example: ridge-lasso-cv -->\s*```python\n(.*?)```", text, re.DOTALL)
    assert match is not None
    scope = {}
    exec(compile(match[1], f"{language}/ridge-lasso-cv", "exec"), scope)  # noqa: S102
    assert set(scope["models"]) == {"RidgeCV", "LassoCV"}
    for name, model in scope["models"].items():
        assert model.alpha_ == .03
        assert set(model.cv_results_) == {"mse_path"}
        np.testing.assert_allclose(model.mean_mse_, model.cv_results_["mse_path"].mean(axis=1))
        assert model.best_score_ == pytest.approx(-np.min(model.mean_mse_))
        assert model.predict(scope["X"][120:]).shape == (40,)
        expected = .966 if name == "RidgeCV" else .965
        assert model.score(scope["X"][120:], scope["y"][120:]) == pytest.approx(expected, abs=.0005)
        with pytest.raises(RuntimeError, match="compute_inference=False"):
            model.summary()


@pytest.mark.parametrize("language", ["en", "cn"])
def test_cv_live_constructor_inventory_and_installed_help(language):
    from statgpu import LassoCV, RidgeCV
    text = _page(language, "reference", "linear-model-api")
    for cls in [RidgeCV, LassoCV]:
        section = text.split(f"## {cls.__name__}\n", 1)[1].split("\n## ", 1)[0]
        assert _signature(cls) in section
        help_text = inspect.getdoc(cls)
        for parameter in inspect.signature(cls).parameters:
            assert f"| `{parameter}` |" in section
            assert re.search(r"^" + parameter + r"\s*:", help_text, re.MULTILINE), parameter
        for name in dir(cls):
            if not name.startswith("_") and (callable(getattr(cls, name))
                    or isinstance(inspect.getattr_static(cls, name), property)):
                assert name in section, name
        assert "Negative selected mean MSE" in help_text
        assert "no mean_mse dictionary key" in help_text
    assert inspect.signature(LassoCV).parameters["nodewise_alpha"].kind is inspect.Parameter.KEYWORD_ONLY
    lasso_section = text.split("## LassoCV\n", 1)[1].split("\n## ", 1)[0]
    assert "RidgeCV" not in inspect.getdoc(LassoCV)
    assert "RidgeCV" not in lasso_section


class _RidgeCVTrainingSubsetSubstitution(AssertionError):
    """The returned losses match unrequested complements of validation sets."""


def _ridge_custom_fold_problem():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 2))
    y = rng.normal(size=len(X))
    alphas = np.array([.01, .1, 1., 10.])
    folds = [(np.arange(4, 6), np.arange(0, 4)),
             (np.arange(8, 10), np.arange(4, 8)),
             (np.arange(0, 2), np.arange(8, 12))]
    return X, y, alphas, folds


def _analytic_ridge_fold_mse(X, y, alphas, folds):
    return np.array([
        [np.mean((X[valid] @ np.linalg.solve(
            X[train].T @ X[train] + len(train)*alpha*np.eye(X.shape[1]),
            X[train].T @ y[train],
        ) - y[valid])**2) for train, valid in folds]
        for alpha in alphas
    ])


def _assert_requested_ridge_scores(actual, requested, unintended_complements):
    assert actual.shape == requested.shape
    assert np.isfinite(actual).all()
    if np.allclose(actual, requested, atol=1e-12, rtol=1e-12):
        return
    assert np.allclose(actual, unintended_complements, atol=1e-12, rtol=1e-12), (
        "A different numerical failure must not be hidden by the #243 xfail"
    )
    raise _RidgeCVTrainingSubsetSubstitution(
        "RidgeCV ignored explicit training subsets and used validation complements"
    )


@pytest.mark.xfail(strict=True, raises=_RidgeCVTrainingSubsetSubstitution,
                   reason="Issue #243: unweighted RidgeCV substitutes complementary training rows")
def test_ridge_cv_should_honor_valid_noncomplement_training_subsets():
    from statgpu import RidgeCV
    X, y, alphas, folds = _ridge_custom_fold_problem()
    requested = _analytic_ridge_fold_mse(X, y, alphas, folds)
    complements = [(np.setdiff1d(np.arange(len(X)), valid), valid) for _, valid in folds]
    unintended = _analytic_ridge_fold_mse(X, y, alphas, complements)
    model = RidgeCV(alphas=alphas, cv_splits=folds, fit_intercept=False,
                    device="cpu", compute_inference=False).fit(X, y)
    _assert_requested_ridge_scores(model.cv_results_["mse_path"], requested, unintended)
    assert model.alpha_ == alphas[np.argmin(requested.mean(axis=1))]


def test_ridge_custom_fold_guard_distinguishes_repairs_and_other_failures():
    X, y, alphas, folds = _ridge_custom_fold_problem()
    requested = _analytic_ridge_fold_mse(X, y, alphas, folds)
    complements = [(np.setdiff1d(np.arange(len(X)), valid), valid) for _, valid in folds]
    unintended = _analytic_ridge_fold_mse(X, y, alphas, complements)
    assert _assert_requested_ridge_scores(requested, requested, unintended) is None
    with pytest.raises(_RidgeCVTrainingSubsetSubstitution):
        _assert_requested_ridge_scores(unintended, requested, unintended)
    for bad in [np.zeros((1, 1)), np.full_like(requested, np.nan), unintended + .01]:
        with pytest.raises(AssertionError) as error:
            _assert_requested_ridge_scores(bad, requested, unintended)
        assert error.type is AssertionError


@pytest.mark.parametrize("unit_weights", [False, True])
def test_ridge_cv_external_exact_fold_loop_is_an_objective_aligned_alternative(unit_weights):
    from statgpu import RidgeCV
    X, y, alphas, folds = _ridge_custom_fold_problem()
    expected = _analytic_ridge_fold_mse(X, y, alphas, folds)
    actual = np.array([
        [np.mean((Ridge(alpha=alpha, fit_intercept=False, device="cpu", compute_inference=False)
                  .fit(X[train], y[train], sample_weight=np.ones(len(train)) if unit_weights else None)
                  .predict(X[valid]) - y[valid])**2)
         for train, valid in folds]
        for alpha in alphas
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert alphas[np.argmin(actual.mean(axis=1))] == 10.
    if unit_weights:
        model = RidgeCV(alphas=alphas, cv_splits=folds, fit_intercept=False,
                        device="cpu", compute_inference=False).fit(X, y, sample_weight=np.ones(len(X)))
        np.testing.assert_allclose(model.cv_results_["mse_path"], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_ridge_custom_fold_limitation_is_visible_from_learner_page_and_help(language):
    from statgpu import RidgeCV
    model_page = _page(language, "models", "ridge")
    reference = _page(language, "reference", "linear-model-api")
    assert "#custom-ridgecv-training-subsets" in model_page
    assert ("full complement" in reference if language == "en" else "完整补集" in reference)
    assert "validation\ncomplements" in inspect.getdoc(RidgeCV)
