"""User-facing API coverage follows the live public surface, not source links."""
from __future__ import annotations

import ast
import inspect
import re
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from statgpu import ElasticNet, LinearRegression, LogisticRegression
from statgpu._base import BaseEstimator
from statgpu.feature_selection import (
    FixedXKnockoffSelector,
    KnockoffResult,
    KnockoffSelector,
    StepwiseSelector,
    fixed_x_knockoff_filter,
    knockoff_filter,
    model_x_knockoff_filter,
    stepwise_selection,
)
from statgpu.linear_model import ElasticNetCV, LogisticRegressionCV

ROOT = Path(__file__).resolve().parents[2]
LINEAR = (LinearRegression, LogisticRegression, ElasticNet, ElasticNetCV, LogisticRegressionCV)
FEATURE = (StepwiseSelector, stepwise_selection, KnockoffSelector, FixedXKnockoffSelector,
           knockoff_filter, fixed_x_knockoff_filter, model_x_knockoff_filter, KnockoffResult)


def read_reference(lang, name):
    return (ROOT / "docs" / lang / "reference" / f"{name}.md").read_text()


def parameter_names(obj):
    return {p.name for p in inspect.signature(obj).parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}


@pytest.mark.parametrize("lang", ["en", "cn"])
@pytest.mark.parametrize("cls", LINEAR)
def test_live_linear_constructors_and_public_methods_have_readable_reference(lang, cls):
    text = read_reference(lang, "linear-model-api")
    shared = read_reference(lang, "estimator-api")
    assert f"## {cls.__name__}" in text
    for name in parameter_names(cls):
        assert f"`{name}`" in text, (cls.__name__, name)
    for name in dir(cls):
        if name.startswith("_"):
            continue
        member = getattr(cls, name)
        if callable(member):
            assert name in text + shared, (cls.__name__, name)
            for param in parameter_names(member):
                assert param in text + shared, (cls.__name__, name, param)
        elif isinstance(inspect.getattr_static(cls, name), property):
            assert name in text, (cls.__name__, name)


@pytest.mark.parametrize("lang", ["en", "cn"])
@pytest.mark.parametrize("obj", FEATURE)
def test_feature_namespace_parameters_and_methods_are_covered(lang, obj):
    text = read_reference(lang, "feature-selection-api")
    stepwise = (ROOT / "docs" / lang / "models" / "feature-selection.md").read_text()
    combined = text + stepwise
    for name in parameter_names(obj):
        assert name in combined, (obj.__name__, name)
    if inspect.isclass(obj):
        for name in dir(obj):
            if not name.startswith("_") and callable(getattr(obj, name)):
                assert name in combined, (obj.__name__, name)


@pytest.mark.parametrize("lang", ["en", "cn"])
def test_all_shared_helper_parameters_and_return_distinctions_are_documented(lang):
    text = read_reference(lang, "estimator-api")
    for name in ("get_params", "set_params", "adjust_pvalues", "combine_pvalues",
                 "bootstrap_statistic", "permutation_test"):
        assert name in text
        for parameter in parameter_names(getattr(BaseEstimator, name)):
            assert parameter in text
    for field in ("pvalues_adjusted", "reject", "observed", "samples", "confidence_interval",
                  "pvalue", "to_dict", "to_dataframe", "force_vectorized", "statistic_hint"):
        assert field in text


@pytest.mark.parametrize("lang", ["en", "cn"])
@pytest.mark.parametrize("name", ["estimator-api", "linear-model-api", "feature-selection-api"])
def test_reference_python_fences_parse_and_marked_examples_execute(lang, name):
    text = read_reference(lang, name)
    for code in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL):
        ast.parse(code)
    examples = re.findall(r"<!-- api-example: ([^>]+) -->\s*```python\n(.*?)```", text, flags=re.DOTALL)
    assert examples
    for label, code in examples:
        if "formula-models" in label:
            pytest.importorskip("pandas")
            pytest.importorskip("patsy")
        exec(compile(code, f"{lang}/{name}:{label}", "exec"), {"__name__": "__doc_example__"})  # noqa: S102 - repository documentation example


def test_binary_evaluation_namespace_and_return_contract():
    from statgpu import evaluate_binary_classification
    from statgpu.metrics import evaluate_binary_classification as metrics_evaluate

    assert evaluate_binary_classification is metrics_evaluate
    result = evaluate_binary_classification([0, 1, 0, 1], [.1, .9, .2, .8])
    assert set(result) == {"threshold", "confusion_matrix", "classification_table", "roc_auc",
                           "average_precision", "roc_curve", "precision_recall_curve"}
    assert np.array_equal(result["confusion_matrix"], [[2, 0], [0, 2]])
    for curve in (result["roc_curve"], result["precision_recall_curve"]):
        assert len({len(value) for value in curve.values()}) == 1
    for lang in ("en", "cn"):
        text = (ROOT / "docs" / lang / "models" / "logistic-regression.md").read_text()
        assert "statgpu.evaluation" not in text


def test_cv_docstrings_describe_negative_scores_and_actual_schema():
    assert "Negative minimum mean validation MSE" in ElasticNetCV.__doc__
    assert "Negative minimum mean validation log-loss" in LogisticRegressionCV.__doc__
    assert "mean_loss_" in LogisticRegressionCV.__doc__


@pytest.mark.parametrize("lang", ["en", "cn"])
def test_display_formulas_define_intercept_weights_and_debiased_target(lang):
    text = (ROOT / "docs" / lang / "models" / "elastic-net.md").read_text()
    assert r"\min_{b,\beta}" in text
    assert r"\sum_{i=1}^n w_i" in text
    assert r"\hat\theta_{\mathrm{db}}" in text
    assert r"M\widehat\Sigma M^\top" in text
    assert "method" in read_reference(lang, "linear-model-api")


@pytest.mark.parametrize("lang", ["en", "cn"])
@pytest.mark.parametrize("cls", LINEAR)
def test_runtime_constructor_signature_includes_every_default(lang, cls):
    signature = inspect.signature(cls)
    parameters = []
    for parameter in signature.parameters.values():
        default = parameter.default
        if isinstance(default, Enum):
            default = default.value
        parameters.append(parameter.replace(annotation=inspect.Parameter.empty, default=default))
    expected = cls.__name__ + str(signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty))
    assert expected in read_reference(lang, "linear-model-api")


def test_tied_knockoff_warning_tracks_observable_limitation_without_requiring_it():
    from scipy.linalg import hadamard

    design = hadamard(8).astype(float)
    X, Xk = design[:, 1:4], design[:, 4:7]
    y = X[:, 0] + X[:, 1] + Xk[:, 2]
    result = fixed_x_knockoff_filter(X, y, Xk=Xk, q=0.5, method="corr_diff", backend="numpy")
    if np.isfinite(result.threshold):
        full_ratio = (1 + np.count_nonzero(result.W <= -result.threshold)) / max(1, np.count_nonzero(result.W >= result.threshold))
        if full_ratio > result.q:
            for lang in ("en", "cn"):
                text = read_reference(lang, "feature-selection-api")
                assert "W=[8,8,-8]" in text
                assert "estimated_fdr=0.5" in text
    for lang in ("en", "cn"):
        text = read_reference(lang, "feature-selection-api")
        assert all(field in text for field in ("q_trajectory", "fdr_hat", "n_selected", "rank"))


@pytest.mark.parametrize("cls", LINEAR)
def test_documented_parameter_lifecycle_and_clone(cls):
    from sklearn.base import clone

    rng = np.random.default_rng(11)
    X = rng.normal(size=(48, 3))
    y = (X[:, 0] > 0).astype(int) if "Logistic" in cls.__name__ else X[:, 0] + rng.normal(size=48)
    kwargs = {"device": "cpu", "compute_inference": False}
    if cls is ElasticNetCV:
        kwargs.update(alphas=[0.03, 0.1], cv=2, random_state=2)
    elif cls is LogisticRegressionCV:
        kwargs.update(Cs=[0.2, 1.0], cv=2, random_state=2)
    model = cls(**kwargs).fit(X, y)
    expected = model.predict(X).copy()
    assert model.set_params() is model
    np.testing.assert_allclose(model.predict(X), expected)
    with pytest.raises(ValueError):
        model.set_params(not_a_constructor_parameter=True)
    np.testing.assert_allclose(model.predict(X), expected)
    copied = clone(model)
    assert copied.get_params(deep=False) == model.get_params(deep=False)
    assert model.set_params(compute_inference=False) is model
    with pytest.raises((RuntimeError, ValueError)):
        model.predict(X)
    with pytest.raises((RuntimeError, ValueError)):
        copied.predict(X)


def test_shared_helper_result_schema_and_numpy_override():
    model = LinearRegression(device="cpu", compute_inference=False)
    adjusted = model.adjust_pvalues([0.01, 0.04, 0.5], method="hochberg", backend="numpy")
    assert set(adjusted) == {"method", "alpha", "axis", "backend", "pvalues", "pvalues_adjusted", "reject"}
    assert adjusted["backend"] == "numpy"
    assert isinstance(adjusted["pvalues"], np.ndarray)
    assert adjusted["reject"].dtype == bool
    combined = model.combine_pvalues([0.01, 0.04], method="stouffer", weights=[1, 2], backend="numpy")
    assert set(combined) == {"method", "axis", "backend", "pvalues", "weights", "statistic", "pvalue"}
    assert 0 <= float(combined["pvalue"]) <= 1
    with pytest.raises(ValueError):
        model.combine_pvalues([0.01, 0.04], method="fisher", weights=[1, 2])
    for method in [model.adjust_pvalues, model.combine_pvalues]:
        with pytest.raises(RuntimeError):
            method()
    boot = model.bootstrap_statistic(np.mean, np.arange(8.), n_resamples=9, random_state=1, backend="numpy")
    assert boot.observed == 3.5
    assert boot.samples.shape == (9,)
    assert boot.samples.dtype == np.float64
    assert isinstance(boot.to_dict()["samples"], list)
    perm = model.permutation_test(lambda x, y: np.sum(x * y, axis=-1), np.arange(8.), np.arange(8.), n_resamples=9, random_state=1, backend="numpy")
    assert perm.samples.shape == (9,)
    assert 0 < perm.pvalue <= 1
    assert perm.metadata["backend"] == "numpy"


def test_cpu_helper_auto_array_inference_is_described():
    torch = pytest.importorskip("torch")
    model = LinearRegression(device="cpu", compute_inference=False)
    pvalues = torch.tensor([0.01, 0.04], dtype=torch.float64)
    auto = model.adjust_pvalues(pvalues)
    explicit = model.adjust_pvalues(pvalues, backend="numpy")
    assert isinstance(explicit["pvalues_adjusted"], np.ndarray)
    if isinstance(auto["pvalues_adjusted"], torch.Tensor):
        assert auto["backend"] == "auto"
        for lang in ("en", "cn"):
            text = read_reference(lang, "estimator-api")
            assert 'backend="numpy"' in text
            assert "CPU" in text and '"auto"' in text


def test_non_iid_vectorization_matches_documented_module_boundary():
    from statgpu.inference import bootstrap_statistic, permutation_test

    x = np.arange(8.)
    boot = bootstrap_statistic(lambda a: np.mean(a, axis=-1), x, n_resamples=7,
                               strategy="block", block_size=2, force_vectorized=True, random_state=1)
    perm = permutation_test(lambda X, y: np.sum(X * y, axis=-1), x, x, n_resamples=7,
                            strategy="grouped", groups=np.repeat([0, 1], 4),
                            force_vectorized=True, random_state=1)
    assert boot.samples.shape == perm.samples.shape == (7,)
    for lang in ("en", "cn"):
        text = read_reference(lang, "estimator-api")
        assert "force_vectorized=True" in text and "block" in text and "grouped" in text


@pytest.mark.parametrize("cls", [ElasticNetCV, LogisticRegressionCV])
def test_weighted_cpu_cv_loss_and_final_refit_match_direct_estimators(cls):
    rng = np.random.default_rng(37)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] - X[:, 1] + rng.normal(size=50)
    if cls is LogisticRegressionCV:
        y = (y > 0).astype(int)
    weight = np.linspace(0.5, 2., len(y))
    folds = [(np.arange(25, 50), np.arange(25)), (np.arange(25), np.arange(25, 50))]
    common = {"device": "cpu", "compute_inference": False, "max_iter": 1000, "tol": 1e-7}
    if cls is ElasticNetCV:
        model = cls(alphas=[0.1, 0.03], l1_ratio=0.5, cv_splits=folds, **common).fit(X, y, sample_weight=weight)
        grid = model.cv_results_["alphas"][0]
        actual = model.cv_results_["mse_path"][0]
    else:
        model = cls(Cs=[0.2, 1.0], cv_splits=folds, **common).fit(X, y, sample_weight=weight)
        grid = model.Cs_
        actual = model.cv_results_["loss_path"]
    expected = np.empty_like(actual)
    for i, candidate in enumerate(grid):
        for j, (train, valid) in enumerate(folds):
            direct = (ElasticNet(alpha=candidate, l1_ratio=0.5, **common) if cls is ElasticNetCV
                      else LogisticRegression(C=candidate, **common))
            direct.fit(X[train], y[train], sample_weight=weight[train])
            if cls is ElasticNetCV:
                loss = (y[valid] - direct.predict(X[valid])) ** 2
            else:
                p = np.clip(direct.predict_proba(X[valid])[:, 1], 1e-15, 1 - 1e-15)
                loss = -(y[valid] * np.log(p) + (1 - y[valid]) * np.log(1 - p))
            expected[i, j] = np.average(loss, weights=weight[valid])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert model.best_score_ == pytest.approx(-expected.mean(axis=1).min(), rel=2e-5, abs=2e-6)
    np.testing.assert_allclose(model.predict(X), model.estimator_.predict(X))


def test_weighted_ols_likelihood_boundary_is_explicit_when_normalization_missing():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 2))
    y = 1 + X @ np.array([2., -1.]) + rng.normal(size=30)
    weights = np.linspace(0.5, 2., 30)
    model = LinearRegression(device="cpu").fit(X, y, sample_weight=weights)
    scaled = LinearRegression(device="cpu").fit(X, y, sample_weight=10 * weights)
    np.testing.assert_allclose(model.coef_, scaled.coef_, atol=1e-12)
    np.testing.assert_allclose(model._bse, scaled._bse, atol=1e-12)
    residual = y - model.predict(X)
    sigma2 = np.average(weights * residual ** 2)
    analytic_llf = -0.5 * np.sum(np.log(2 * np.pi * sigma2 / weights) + weights * residual ** 2 / sigma2)
    if not np.isclose(model.llf, analytic_llf):
        for lang in ("en", "cn"):
            text = (ROOT / "docs" / lang / "models" / "linear-regression.md").read_text()
            assert "0.5 * sum(log(sample_weight))" in text
            assert all(name in text for name in ("llf", "aic", "bic"))


@pytest.mark.parametrize("lang", ["en", "cn"])
def test_model_specific_formula_and_multivariate_boundaries(lang):
    text = read_reference(lang, "linear-model-api")
    logistic = text.split("## LogisticRegression\n", 1)[1].split("## ElasticNet\n", 1)[0]
    assert "formula syntax overrides" not in logistic
    assert "公式语法决定公式拟合" not in logistic
    assert "TypeError" in text.split("## LogisticRegression\n", 1)[0]
    for cls in (LinearRegression, LogisticRegression, ElasticNet, ElasticNetCV, LogisticRegressionCV):
        for name in parameter_names(cls):
            assert name in inspect.getdoc(cls), (cls.__name__, name)
