"""Execute the bilingual companion and check its working-coordinate formulas.

These tests protect the documented examples, equations and known support
boundaries; they are not an empirical confidence-interval coverage study.
"""
from pathlib import Path

import numpy as np
import pytest

from doc_examples import parse_examples, run_example
from statgpu.linear_model import Lasso


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("en", "cn")


def companion(language):
    return ROOT / f"docs/{language}/models/regression/lasso-inference.md"


@pytest.mark.parametrize("language", LANGUAGES)
def test_companion_navigation_and_current_support_boundaries(language):
    path = companion(language)
    text = path.read_text(encoding="utf-8")
    other = "cn" if language == "en" else "en"
    assert f"../../../{other}/models/regression/lasso-inference.md" in text
    for source in (path.with_name("lasso.md"), path.with_name("README.md"),
                   ROOT / f"docs/{language}/guides/nodewise-alpha-migration.md"):
        assert "lasso-inference.md" in source.read_text(encoding="utf-8")
    assert not (ROOT / f"docs/{language}/models/lasso-inference.md").exists()
    for required in ("fit_intercept=False", "NumPy", "nodewise_alpha_",
                     "simultaneous_include_intercept=False",
                     "_inference_result.simultaneous_target_mask",
                     "post_selection_ols", "LassoCV", "max-|Z|"):
        assert required in text
    for obsolete in ("publication gate", "maintained route",
                     "exact-head", "PR #", "../guides/"):
        # A canonical ../../guides/ link legitimately contains ../guides/.
        if obsolete == "../guides/":
            assert "](../guides/" not in text
        else:
            assert obsolete not in text
    # Pin the essential weighted formulas, not arbitrary prose wording.
    for equation in (r"a_i=\sqrt{\frac{n w_i}{W}}", r"\hat r_w=y_w-X_w\hat\beta",
                     r"M=D^{-1}\Theta_ZD^{-1}", r"q=a-X_wM^\top\bar x_w",
                     r"h=q/n", r"\xi^{(b)}\odot\hat r_w",
                     r"\hat r_{w,i}h_i"):
        assert equation in text
    assert "0.2.5" in text and "0.2.6" in text and "../../changelog.md" in text
    assert "Zhang, X., & Cheng, G. (2017)" in text


def test_bilingual_companions_share_formulas_and_executable_steps():
    import re

    en, cn = (companion(lang).read_text(encoding="utf-8") for lang in LANGUAGES)
    assert re.findall(r"\$\$(.*?)\$\$", en, re.DOTALL) == re.findall(
        r"\$\$(.*?)\$\$", cn, re.DOTALL)
    examples = [parse_examples(text) for text in (en, cn)]
    assert set(examples[0]) == set(examples[1]) == {"lasso-inference-companion"}
    assert examples[0]["lasso-inference-companion"].blocks == examples[1][
        "lasso-inference-companion"].blocks


@pytest.mark.parametrize("language", LANGUAGES)
def test_complete_companion_example_executes_without_hidden_setup(language):
    path = companion(language)
    namespace = run_example(path.read_text(encoding="utf-8"),
                            "lasso-inference-companion", str(path))
    model = namespace["model"]
    assert model._conf_int.shape == model._conf_int_simultaneous.shape == (6, 2)
    np.testing.assert_array_equal(model._inference_result.simultaneous_target_mask,
                                  np.ones(6, dtype=bool))
    assert model.nodewise_alpha_ == pytest.approx(np.sqrt(2 * np.log(5) / 400))
    assert np.isfinite(model._conf_int_simultaneous).all()
    assert model._simultaneous_critical_value > 0
    no_inference = Lasso(
        alpha=0.05, device="cpu", solver="coordinate_descent",
        max_iter=5000, tol=1e-10, compute_inference=False,
    ).fit(namespace["X"], namespace["y"])
    np.testing.assert_allclose(model.coef_, no_inference.coef_, atol=1e-12)
    assert model.intercept_ == pytest.approx(no_inference.intercept_, abs=1e-12)


@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("fit_intercept,include_intercept", (
    (True, False), (True, True), (False, False),
))
def test_documented_working_scores_reconstruct_cpu_intervals(
    weighted, fit_intercept, include_intercept,
):
    rng = np.random.default_rng(314)
    n, p = 180, 4
    X = rng.normal(size=(n, p)) + [1.5, -0.8, 0.3, 1.0]
    y = 0.7 + X @ np.array([1.3, 0.0, -0.9, 0.4]) + rng.normal(size=n)
    weights = rng.uniform(0.2, 2.0, size=n) if weighted else np.ones(n)
    model = Lasso(
        alpha=0.08, nodewise_alpha=None, fit_intercept=fit_intercept,
        device="cpu", solver="coordinate_descent", max_iter=5000, tol=1e-10,
        inference_method="debiased", compute_inference=True,
        enable_simultaneous_inference=True, simultaneous_n_bootstrap=48,
        simultaneous_random_state=91, simultaneous_alpha=0.1,
        simultaneous_include_intercept=include_intercept,
    ).fit(X, y, sample_weight=weights if weighted else None)

    # Independently reconstruct the tutorial's normalized working problem.
    a = np.sqrt(n * weights / weights.sum())
    xbar = np.average(X, weights=weights, axis=0) if fit_intercept else np.zeros(p)
    ybar = np.average(y, weights=weights) if fit_intercept else 0.0
    Xw, yw = a[:, None] * (X - xbar), a * (y - ybar)
    rw = yw - Xw @ model.coef_
    M = np.asarray(model._debiased_M_cpu)
    slopes = model.coef_ + M @ Xw.T @ rw / n
    offset = int(fit_intercept)
    np.testing.assert_allclose(model._params[offset:], slopes, atol=2e-10)
    sigma2 = rw @ rw / max(n - np.count_nonzero(model.coef_), 1)
    gram = Xw.T @ Xw / n
    se_slopes = np.sqrt(sigma2 * np.diag(M @ gram @ M.T) / n)
    np.testing.assert_allclose(model._bse[offset:], se_slopes, atol=2e-10)
    n_nw = weights.sum() ** 2 / (weights @ weights)
    assert model.nodewise_alpha_ == pytest.approx(np.sqrt(2 * np.log(p) / n_nw))

    xi = np.random.default_rng(91).standard_normal(size=(48, n))
    scores = ((xi * rw) @ Xw) @ M.T / n
    maxima = np.max(np.abs(scores / se_slopes), axis=1)
    if fit_intercept:
        assert model._params[0] == pytest.approx(ybar - xbar @ slopes, abs=2e-10)
        q = a - Xw @ M.T @ xbar
        se_intercept = np.sqrt(sigma2 * (q @ q)) / n
        assert model._bse[0] == pytest.approx(se_intercept, abs=2e-10)
        if include_intercept:
            intercept_scores = (xi * rw) @ (q / n)
            maxima = np.maximum(maxima, np.abs(intercept_scores / se_intercept))
    critical = np.quantile(maxima, 0.9)
    assert model._simultaneous_critical_value == pytest.approx(critical, abs=2e-10)
    target = np.ones(p + offset, dtype=bool)
    if fit_intercept and not include_intercept:
        target[0] = False
        np.testing.assert_array_equal(model._conf_int_simultaneous[0], model._conf_int[0])
    if include_intercept:
        np.testing.assert_array_equal(model._inference_result.simultaneous_target_mask, target)
    else:
        assert model._inference_result.simultaneous_target_mask is None
    expected = np.column_stack((model._params - critical * model._bse,
                                model._params + critical * model._bse))
    np.testing.assert_allclose(model._conf_int_simultaneous[target], expected[target], atol=2e-10)
