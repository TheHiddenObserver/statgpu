"""Execute shared CV learning examples and keep their entry-point claims aligned."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.linear_model import LassoCV, PenalizedGLM_CV, RidgeCV
from statgpu.nonparametric import KernelRidgeCV

ROOT = Path(__file__).resolve().parents[2]


def _page(language, name):
    return (ROOT / f"docs/{language}/{name}.md").read_text()


def _run(language, label):
    text = _page(language, "guides/cross-validation")
    match = re.search(
        rf"<!-- api-example: {label} -->\s*```python\n(.*?)```", text, re.DOTALL,
    )
    assert match is not None
    namespace = {}
    exec(compile(match[1], f"{language}/{label}", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("label,cls,alpha,score", [
    ("cv-ridge", RidgeCV, 0.001, 0.975),
    ("cv-lasso", LassoCV, 0.01, 0.976),
])
def test_cv_gaussian_examples_have_separate_test_data(language, label, cls, alpha, score):
    ns = _run(language, label)
    model, X, y = ns["model"], ns["X"], ns["y"]
    assert isinstance(model, cls)
    assert model.device == "cpu" and model.compute_inference is False
    assert X.shape == (100, 4) and y.shape == (100,)
    assert model.alpha_ == pytest.approx(alpha)
    assert ns["prediction"].shape == (20,)
    np.testing.assert_allclose(ns["prediction"], model.predict(X[80:]))
    # Refit the documented 80-row training partition independently. This also
    # detects changing the learner example to train on its held-out test rows.
    params = model.get_params()
    reference = cls(**params).fit(X[:80], y[:80])
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-10)
    assert model.intercept_ == pytest.approx(reference.intercept_)
    assert model.score(X[80:], y[80:]) == pytest.approx(score, abs=0.0005)
    if cls is LassoCV:
        assert model.cv_solver_ == "coordinate_descent"
        assert model.solver == "fista"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_cv_poisson_example_is_a_complete_count_prediction_workflow(language):
    ns = _run(language, "cv-poisson")
    model, y = ns["model"], ns["y"]
    assert isinstance(model, PenalizedGLM_CV)
    assert model.compute_inference is False
    assert (y >= 0).all() and np.array_equal(y, np.floor(y))
    assert model.alpha_ == pytest.approx(0.01)
    assert ns["prediction"].shape == (20,)
    assert np.isfinite(ns["prediction"]).all() and (ns["prediction"] > 0).all()
    np.testing.assert_allclose(ns["prediction"][:3], [2.108, 1.671, 1.750], atol=0.0005)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_ordered_cv_example_respects_documented_gap(language):
    ns = _run(language, "cv-ordered-splits")
    assert isinstance(ns["model"], PenalizedGLM_CV)
    assert ns["model"].alpha_ == pytest.approx(0.05)
    assert len(ns["splits"]) == 3
    for train, validation in ns["splits"]:
        assert train.ndim == validation.ndim == 1
        assert len(train) > 0 and len(validation) > 0
        assert np.intersect1d(train, validation).size == 0
        assert train.max() + 3 == validation.min()
    assert ns["X"].shape == (90, 2)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_shared_cv_inventory_links_kernel_specific_contract(language):
    guide = _page(language, "guides/cross-validation")
    inventory = _page(language, "guides/implemented-methods")
    assert "| `KernelRidgeCV` |" in guide
    assert "| `KernelRidgeCV` |" in inventory
    anchor = "complete-estimator-api" if language == "en" else "完整估计器-api"
    assert "../models/kernel-methods.md#" + anchor in guide
    assert "best_score_" in guide and "sample_weight" in guide
    assert "cv_splits" not in inspect.signature(KernelRidgeCV).parameters
    assert "sample_weight" not in inspect.signature(KernelRidgeCV.fit).parameters


@pytest.mark.parametrize("language", ["en", "cn"])
def test_portal_does_not_promise_unconditional_oracle_property(language):
    text = _page(language, "README")
    for name in ("SCAD", "MCP"):
        line = next(line for line in text.splitlines() if line.startswith(f"- [{name}]"))
        assert "oracle" not in line.lower()
        assert ("shrinkage" if language == "en" else "收缩") in line
