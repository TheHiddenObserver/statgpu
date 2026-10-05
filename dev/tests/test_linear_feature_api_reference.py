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
