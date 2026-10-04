"""Execute the learner entry examples against the current public API."""

import re
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = ("linear-regression", "logistic-regression", "elastic-net", "feature-selection")


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", _MODELS)
def test_first_learner_example_is_self_contained_cpu(language, page):
    text = (_ROOT / f"docs/{language}/models/{page}.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, f"{page}: missing runnable Python example"
    namespace = {}
    exec(compile(blocks[0], f"{language}/{page}.md", "exec"), namespace)  # noqa: S102
    if page == "logistic-regression":
        model, X, y = (namespace[name] for name in ("model", "X", "y"))
        assert model.C == 0
        assert model.converged_
        probabilities = model.predict_proba(X[600:])
        assert probabilities.shape == (200, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(np.isfinite(model._conf_int))
        assert model.score(X[600:], y[600:]) > 0.55
    elif page == "linear-regression":
        model = namespace["model"]
        assert namespace["prediction"].shape == (60,)
        assert model._conf_int.shape == (4, 2)
        np.testing.assert_allclose(model.coef_, [2.04949606, -0.96581544, -0.08754275], atol=1e-7)
        assert model.score(namespace["X_test"], namespace["y_test"]) > 0.9
    elif page == "feature-selection":
        selector = namespace["selector"]
        assert selector.selected_features_ == [0, 2]
        assert namespace["X_selected"].shape == (60, 2)
        assert selector.score(namespace["X_test"], namespace["y_test"]) > 0.95
    elif page == "elastic-net":
        model, X, y = (namespace[name] for name in ("model", "X", "y"))
        assert model.coef_.shape == (8,)
        np.testing.assert_allclose(
            model.coef_, [0.938, 0.834, -0.552, 0, 0, 0, 0, 0], atol=1e-3
        )
        np.testing.assert_array_equal(np.flatnonzero(np.abs(model.coef_) > 1e-8), [0, 1, 2])
        np.testing.assert_allclose(model.predict(X[300:303]), [-1.575, 1.710, 1.476], atol=1e-3)
        assert model.score(X[300:], y[300:]) == pytest.approx(0.936, abs=1e-3)
        assert np.all(np.isfinite(model.predict(X[300:])))
        assert model.score(X[300:], y[300:]) > 0.8


@pytest.mark.parametrize("language", ("en", "cn"))
def test_logistic_documentation_distinguishes_penalized_score(language):
    from statgpu.linear_model import LogisticRegression

    text = (_ROOT / f"docs/{language}/models/logistic-regression.md").read_text(encoding="utf-8")
    assert r"-\alpha_C\beta=0" in text
    assert r"1/C,&C>0" in text
    assert r"0,&C=0" in text
    for name in signature(LogisticRegression).parameters:
        assert f"| `{name}` |" in text, f"missing constructor parameter {name}"
    assert "sample_weight" in text
    assert "(n_samples, 2)" in text


def test_documented_standalone_logistic_penalty_scale():
    from statgpu.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 3))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + X @ np.array([0.5, -0.3, 0.2])))))
    weights = rng.uniform(0.5, 2, size=len(y))
    for C in (0, 1.0, 3.0):
        model = LogisticRegression(C=C, device="cpu", tol=1e-10, max_iter=500,
                                   compute_inference=False).fit(X, y, sample_weight=weights)
        assert model.converged_
        residual = weights * (y - model.predict_proba(X)[:, 1])
        alpha = 0 if C == 0 else 1 / C
        np.testing.assert_allclose(X.T @ residual - alpha * model.coef_, 0, atol=1e-7)
        np.testing.assert_allclose(residual.sum(), 0, atol=1e-7)
