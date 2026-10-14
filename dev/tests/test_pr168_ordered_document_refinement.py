"""Guard ordered-model teaching structure, parameter ordering, and probabilities."""
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, ndtr
from scipy.stats import norm

from doc_examples import run_example
from doc_paths import markdown_anchors
from statgpu.linear_model import OrderedLogitRegression, OrderedProbitRegression

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("language", ("en", "cn"))
def test_ordered_reading_order_inventory_and_notation(language):
    text = (ROOT / f"docs/{language}/models/regression/ordered.md").read_text()
    markers = ("<!-- learner-example: ordered-inference -->", "| `_bse` |",
               "### Current Limitations" if language == "en" else "### 当前限制",
               "### Hessian")
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert "[-inf, θ_0, ..., θ_{K-2}, +inf]" in text
    assert "z_r = a_hat_r / bse_r" in text
    assert "z = θ / bse" not in text
    assert "pi**2 / 3" in text
    assert "numpy.linalg.LinAlgError" in text
    assert "NotImplementedError" in text
    assert "fit_intercept=False" in text
    assert "cpu-example" in markdown_anchors(text)
    for cls in (OrderedLogitRegression, OrderedProbitRegression):
        for name in signature(cls).parameters:
            assert f"| `{name}` |" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_ordered_cutpoint_example_and_full_parameter_wald_statistics(language):
    path = ROOT / f"docs/{language}/models/regression/ordered.md"
    text = path.read_text()
    ns = run_example(text, "ordered-basic", str(path))
    model, X = ns["model"], ns["X"]
    cumulative = expit(model._thresh_est - X[300] @ model.coef_)
    np.testing.assert_allclose(ns["cumulative"], cumulative, atol=1e-12)
    np.testing.assert_allclose(ns["from_cutpoints"], model.predict_proba(X[300:301])[0], atol=1e-12)
    np.testing.assert_allclose(cumulative, [0.18412239, 0.51663766], atol=1e-7)
    np.testing.assert_allclose(model._thresh_est, [-0.8253572, 0.72988166], atol=1e-7)
    np.testing.assert_allclose(np.diff(np.r_[0, expit([-0.7, 0.8]), 1]),
                               [0.332, 0.358, 0.310], atol=5e-4)
    np.testing.assert_allclose(model.thresholds_[1:-1], model._thresh_est)
    assert np.isneginf(model.thresholds_[0]) and np.isposinf(model.thresholds_[-1])

    ns = run_example(text, "ordered-inference", str(path))
    fitted = ns["inference_model"]
    params = np.r_[fitted.coef_, fitted._thresh_est]
    np.testing.assert_allclose(fitted._zvalues, params / fitted._bse)
    np.testing.assert_allclose(fitted._pvalues, 2 * norm.sf(abs(fitted._zvalues)))
    np.testing.assert_allclose(fitted._conf_int,
                               params[:, None] + np.array([-1, 1]) * norm.ppf(.975) * fitted._bse[:, None])
    ns = run_example(text, "ordered-probit", str(path))
    probit = ns["probit_model"]
    cumulative_probit = ndtr(probit._thresh_est - X[300] @ probit.coef_)
    np.testing.assert_allclose(np.diff(np.r_[0, cumulative_probit, 1]), ns["probit_probability"][0], atol=1e-12)


@pytest.mark.parametrize("cls", (OrderedLogitRegression, OrderedProbitRegression))
def test_documented_unsupported_inference_and_weights_raise(cls):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 2))
    y = np.digitize(X @ np.array([.8, -.5]) + rng.logistic(size=400), [-.7, .8])
    with pytest.raises(ValueError, match="sample_weight"):
        cls(device="cpu").fit(X, y, sample_weight=np.ones(len(y)))
    with pytest.raises(NotImplementedError, match="nonrobust"):
        cls(device="cpu", compute_inference=True, cov_type="HC0").fit(X, y)


@pytest.mark.parametrize("cls,cdf", ((OrderedLogitRegression, expit), (OrderedProbitRegression, ndtr)))
def test_two_category_threshold_is_negative_binary_intercept(cls, cdf):
    rng = np.random.default_rng(421)
    X = rng.normal(size=(300, 2))
    y = (X @ np.array([.8, -.5]) + rng.logistic(size=300) > .4).astype(int)
    fitted = cls(n_categories=2, device="cpu", max_iter=200, tol=1e-8).fit(X, y)
    probability = fitted.predict_proba(X)
    assert fitted._thresh_est.shape == (1,)
    assert fitted.thresholds_.shape == (3,)
    assert probability.shape == (300, 2)
    eta = X @ fitted.coef_
    np.testing.assert_allclose(probability[:, 1], cdf(eta - fitted._thresh_est[0]), atol=1e-12)
    np.testing.assert_allclose(probability[:, 0], cdf(fitted._thresh_est[0] - eta), atol=1e-12)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
