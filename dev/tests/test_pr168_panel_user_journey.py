"""Keep panel lessons runnable before their mathematical/API reference."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2, f

from doc_examples import example_code, run_example
from doc_paths import markdown_anchors
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("panel-ols", "pooled-ols", "random-effects", "between-ols",
          "first-difference-ols", "fama-macbeth")


def read(language, page):
    path = ROOT / "docs" / language / "panel" / (page + ".md")
    return path, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", MODELS)
def test_first_fit_precedes_full_parameter_reference(language, page):
    _, text = read(language, page)
    example = text.index("<!-- example: cpu-basic -->")
    parameters = re.search(r'(?m)^## (?:Parameters|Complete API reference|参数|完整 API 参考)\s*$', text).start()
    assert example < parameters
    assert "predict(" in text
    # Both old tutorial/API entry points must remain available.
    assert 'id="cpu-example"' in text
    assert 'id="external-validation"' in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_diagnostic_tutorial_matches_independent_regressions(language, capsys):
    path, text = read(language, "diagnostics")
    ns = run_example(text, "cpu-diagnostics", str(path))
    X, y, ids = ns["X"], ns["y"], ns["entity_ids"]
    n, groups, periods = len(y), len(np.unique(ids)), 8
    assert (n, groups) == (240, 30)
    assert all(ns[key].device == "cpu" for key in ("fe", "pooled", "re"))
    # Pooling F: independently compare a common intercept with entity dummies.
    pooled_design = np.column_stack([np.ones(n), X])
    dummy_design = np.column_stack([np.eye(groups)[ids], X])
    pooled_beta = np.linalg.lstsq(pooled_design, y, rcond=None)[0]
    full_beta = np.linalg.lstsq(dummy_design, y, rcond=None)[0]
    ep = y - pooled_design @ pooled_beta
    ef = y - dummy_design @ full_beta
    df = n - np.linalg.matrix_rank(dummy_design)
    q = np.linalg.matrix_rank(dummy_design) - np.linalg.matrix_rank(pooled_design)
    expected_f = ((ep @ ep - ef @ ef) / q) / (ef @ ef / df)
    pooling = ns["tests"]["pooling F"]
    assert pooling.applicable
    assert pooling.df == (q, df)
    assert_allclose(pooling.statistic, expected_f, rtol=1e-12)
    # The package evaluates 1-CDF, which can round a tiny tail to zero.
    assert_allclose(pooling.pvalue, f.sf(expected_f, q, df), atol=1e-14)
    group_sums = np.bincount(ids, weights=ep)
    expected_lm = n / (2 * (periods - 1)) * (group_sums @ group_sums / (ep @ ep) - 1) ** 2
    lm = ns["tests"]["Breusch-Pagan LM"]
    assert lm.applicable
    assert_allclose(lm.statistic, expected_lm, rtol=1e-12)
    assert_allclose(lm.pvalue, chi2.sf(expected_lm, 1), rtol=1e-11)
    # Reconstruct FE and Swamy-Arora RE covariance without panel helpers.
    xb = X.reshape(groups, periods, 1).mean(axis=1)
    yb = y.reshape(groups, periods).mean(axis=1)
    xw, yw = X - xb[ids], y - yb[ids]
    bw = np.linalg.lstsq(xw, yw, rcond=None)[0]
    bb = np.linalg.lstsq(xb, yb, rcond=None)[0]
    ew, eb = yw - xw @ bw, yb - xb @ bb
    sigma_e = ew @ ew / (n - 1 - (groups - 1))
    sigma_a = max(0, (eb @ eb / (groups - 1) - sigma_e) / periods)
    theta = 1 - np.sqrt(sigma_e / (sigma_e + periods * sigma_a))
    xr, yr = X - theta * xb[ids], y - theta * yb[ids]
    br = np.linalg.lstsq(xr, yr, rcond=None)[0]
    er = yr - xr @ br
    vf = (ew @ ew / df) * np.linalg.inv(xw.T @ xw)
    vr = (er @ er / (n - 1)) * np.linalg.inv(xr.T @ xr)
    assert np.linalg.eigvalsh(vf - vr).max() < 0
    hausman = ns["tests"]["Hausman"]
    assert not hausman.applicable
    assert hausman.statistic is None and hausman.pvalue is None
    assert "not positive semidefinite" in hausman.reason
    output = capsys.readouterr().out
    assert "not applicable:" in output and "Hausman" in output


def test_bilingual_diagnostic_example_is_identical():
    _, en = read("en", "diagnostics")
    _, cn = read("cn", "diagnostics")
    assert example_code(en, "cpu-diagnostics") == example_code(cn, "cpu-diagnostics")
    for text in (en, cn):
        assert text.index("<!-- example: cpu-diagnostics -->") < text.index("$$")
        assert "result.applicable" in text and "result.reason" in text


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", MODELS)
def test_model_reference_retains_constructor_and_fit_inputs(language, page):
    import inspect
    import statgpu.panel as panel

    classes = dict(zip(MODELS, (panel.PanelOLS, panel.PooledOLS,
                               panel.RandomEffects, panel.BetweenOLS,
                               panel.FirstDifferenceOLS, panel.FamaMacBeth)))
    cls = classes[page]
    _, text = read(language, page)
    for name in inspect.signature(cls).parameters:
        assert "`" + name + "`" in text, (page, name)
    for name in inspect.signature(cls.fit).parameters:
        if name != "self":
            assert re.search(r"\b" + name + r"\s*=", text), (page, name)
    assert text.index("<!-- example: cpu-basic -->") < text.index("$$")


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", MODELS)
def test_tutorial_prediction_semantics(language, page):
    path, text = read(language, page)
    ns = run_example(text, "cpu-basic", str(path))
    model, X, ids = ns["model"], ns["X"], ns["entity_ids"]
    if page in ("pooled-ols", "between-ols", "fama-macbeth"):
        expected = model.coef_[0] + X @ model.coef_[1:]
    else:
        expected = X @ model.coef_
    predicted = model.predict(X)
    assert isinstance(predicted, np.ndarray)  # CPU example, including FMB.
    assert_allclose(predicted, expected, rtol=1e-12, atol=1e-13)
    if page == "panel-ols":
        mean_residual = (ns["y"] - expected).reshape(12, 5).mean(axis=1)
        assert_allclose(model.predict(X, entity_ids=ids),
                        expected + mean_residual[ids], rtol=1e-12, atol=1e-13)
    elif page == "first-difference-ols":
        differences = np.diff(X.reshape(12, 5, 1), axis=1).reshape(48, 1)
        assert_allclose(model.predict(differences), differences @ model.coef_)
    elif page == "between-ols":
        means = X.reshape(12, 5, 1).mean(axis=1)
        assert_allclose(model.predict(means), model.coef_[0] + means @ model.coef_[1:])


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ("fit-statistics", "panel-ols"))
def test_fit_statistics_and_prediction_links_reach_existing_sections(language, page):
    path, text = read(language, page)
    for token in MarkdownIt().parse(text):
        for child in token.children or ():
            if child.type != "link_open":
                continue
            href = child.attrGet("href")
            link = urlsplit(href)
            if link.scheme or link.netloc or not link.fragment:
                continue
            destination = (path.parent / unquote(link.path)) if link.path else path
            assert destination.is_file(), (path, href)
            assert unquote(link.fragment) in markdown_anchors(
                destination.read_text(encoding="utf-8")
            ), (path, href)
