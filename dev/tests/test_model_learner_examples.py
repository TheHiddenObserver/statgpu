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
def test_linear_column_target_example_scores_a_flattened_response(language):
    from statgpu.linear_model import LinearRegression

    page = _ROOT / f"docs/{language}/models/linear-regression.md"
    text = page.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- learner-example: linear-column-target -->\s*```python\n(.*?)```",
        text, flags=re.DOTALL,
    )
    assert match is not None, f"{page}: missing single-column scoring example"
    namespace = {}
    exec(compile(match.group(1), str(page), "exec"), namespace)  # noqa: S102
    model, X, y_column = (namespace[name] for name in ("model", "X", "y_column"))
    assert y_column.shape == (len(X), 1)
    assert model.predict(X).shape == (len(X),)
    # Exercise the safe public call without requiring the underlying shape bug
    # to persist after a future production fix.
    expected = LinearRegression(device="cpu", compute_inference=False).fit(X, y_column.ravel())
    np.testing.assert_allclose(model.predict(X), expected.predict(X))
    assert namespace["r2"] == pytest.approx(expected.score(X, y_column.ravel()))
    assert namespace["r2"] == pytest.approx(1.0)


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


@pytest.mark.parametrize("language", ("en", "cn"))
def test_elasticnet_initialization_documentation_tracks_runtime(language):
    from statgpu.linear_model import ElasticNet

    rng = np.random.default_rng(82)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.2, size=80)
    model = ElasticNet(alpha=0.03, device="cpu")
    model.fit(X, y, initial_coef=np.array([1.0, 2.0, 3.0]))
    model.fit(X, y)
    retained = getattr(model, "_init_coef", None) is not None

    text = (_ROOT / f"docs/{language}/models/elastic-net.md").read_text(encoding="utf-8")
    retention_warning = {
        "en": "retains this starting vector",
        "cn": "仍会保留该初值",
    }[language]
    # Couple the warning to observed behavior, without requiring future
    # implementations to retain the old initial vector or to raise an error.
    assert (retention_warning in text) == retained
    if retained:
        fresh_advice = {"en": "Create a fresh estimator", "cn": "应新建估计器"}[language]
        assert fresh_advice in text
        assert "one-fit" not in text and "只为一次拟合" not in text

    # The documented fresh-estimator route supports a changed feature count.
    fresh = ElasticNet(alpha=0.03, device="cpu").fit(X[:, :2], y)
    assert fresh.coef_.shape == (2,)
    assert np.all(np.isfinite(fresh.predict(X[:, :2])))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_elasticnet_bootstrap_documentation_matches_public_result(language):
    from statgpu.linear_model import ElasticNet

    rng = np.random.default_rng(82)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.2, size=80)
    model = ElasticNet(
        alpha=0.03, l1_ratio=0.4, device="cpu",
        compute_inference=True, inference_method="bootstrap",
    )
    model.n_bootstrap = 4
    model.bootstrap_random_state = 42
    model.fit(X, y)
    result = model._inference_result
    assert result.method == "residual_bootstrap"
    assert result.metadata["refit_penalty"] == "elasticnet"
    assert result.metadata["numerical_backend"] == "numpy"
    assert result.metadata["numerical_device"] == "cpu"
    assert result.metadata["reporting_backend"] == "numpy"
    assert result.metadata["penalty_conditioning"] == "fixed_penalty"
    assert np.all(np.isfinite(result.conf_int))

    text = (_ROOT / f"docs/{language}/models/elastic-net.md").read_text(encoding="utf-8")
    for contract in ("NumPy/CuPy/Torch", "`sample_weight=None`", '`cov_type="nonrobust"`'):
        assert contract in text
    for stale in ("remains a CPU-native residual-refit path", "当前仍是 CPU 原生的残差重拟合路径"):
        assert stale not in text

    with pytest.raises(NotImplementedError, match="Weighted Gaussian residual-bootstrap"):
        model.fit(X, y, sample_weight=np.ones(len(y)))
    robust = ElasticNet(
        device="cpu", compute_inference=True,
        inference_method="bootstrap", cov_type="hc1",
    )
    with pytest.raises(NotImplementedError, match="nonrobust"):
        robust.fit(X, y)
