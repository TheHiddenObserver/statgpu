"""Execute bilingual panel tutorials and protect their statistical explanations."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm, t

from doc_examples import example_code, run_example
from statgpu.panel import FamaMacBeth

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "pooled-ols", "panel-ols", "between-ols", "first-difference-ols",
    "random-effects", "fama-macbeth",
)


def _read(language, slug):
    path = ROOT / "docs" / language / "panel" / f"{slug}.md"
    return path, path.read_text(encoding="utf-8")


def _run(language, slug):
    path, text = _read(language, slug)
    return run_example(text, "cpu-basic", str(path))


def _means(values, entities):
    return np.array([values[entities == entity].mean(axis=0)
                     for entity in np.unique(entities)])


def _ols(design, response, df):
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ beta
    covariance = (residual @ residual / df) * np.linalg.inv(design.T @ design)
    return beta, covariance


def _manual_result(slug, ns):
    """Build the documented regression using NumPy, not panel internals."""
    X, y, ids = ns["X"], ns["y"], ns["entity_ids"]
    if slug == "pooled-ols":
        design = np.column_stack([np.ones(len(y)), X])
        df, response = 58, y
    elif slug == "panel-ols":
        design, response = X - _means(X, ids)[ids], y - _means(y, ids)[ids]
        df = 47
    elif slug == "between-ols":
        design = np.column_stack([np.ones(12), _means(X, ids)])
        response, df = _means(y, ids), 10
    elif slug == "first-difference-ols":
        design = np.diff(X.reshape(12, 5, 1), axis=1).reshape(48, 1)
        response, df = np.diff(y.reshape(12, 5), axis=1).ravel(), 47
    elif slug == "random-effects":
        Xbar, ybar = _means(X, ids), _means(y, ids)
        Xw, yw = X[:, 1:] - Xbar[ids, 1:], y - ybar[ids]
        bw = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        bb = np.linalg.lstsq(Xbar, ybar, rcond=None)[0]
        ew, eb = yw - Xw @ bw, ybar - Xbar @ bb
        sigma_e = ew @ ew / 47
        sigma_a = max(0, (eb @ eb / 10 - sigma_e) / 5)
        theta = 1 - np.sqrt(sigma_e / (sigma_e + 5 * sigma_a))
        assert_allclose(ns["model"].theta_, theta, rtol=1e-11, atol=1e-13)
        assert_allclose(
            list(ns["model"].variance_components_.values()),
            [sigma_e, sigma_a], rtol=1e-11, atol=1e-13,
        )
        design, response, df = X - theta * Xbar[ids], y - theta * ybar[ids], 58
    else:
        times = ns["time_ids"]
        betas = []
        for period in range(5):
            keep = times == period
            design = np.column_stack([np.ones(keep.sum()), X[keep]])
            assert np.linalg.matrix_rank(design) == 2
            betas.append(np.linalg.lstsq(design, y[keep], rcond=None)[0])
        betas = np.array(betas)
        beta = betas.mean(axis=0)
        deviations = betas - beta
        gamma1 = deviations[1:].T @ deviations[:-1]
        covariance = (deviations.T @ deviations + 0.5 * (gamma1 + gamma1.T)) / 25
        assert_allclose(ns["model"].betas_, betas, rtol=1e-11, atol=1e-13)
        assert ns["model"].n_periods == 5
        return beta, covariance, 60, 4
    beta, covariance = _ols(design, response, df)
    return beta, covariance, len(response), df


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("slug", MODELS)
def test_panel_cpu_examples_match_documented_transformed_regression(language, slug, capsys):
    ns = _run(language, slug)
    model = ns["model"]
    assert model.device == "cpu"
    beta, covariance, nobs, df = _manual_result(slug, ns)
    se = np.sqrt(np.diag(covariance))
    assert_allclose(model.coef_, beta, rtol=1e-11, atol=1e-13)
    # Full covariance is public only for FamaMacBeth; inspect the stored matrix
    # internally for the other families without claiming a public attribute.
    actual_cov = (model.cov_params_ if slug == "fama-macbeth"
                  else model._panel_cov_params_raw)
    assert_allclose(actual_cov, covariance, rtol=1e-10, atol=1e-13)
    assert_allclose(model.bse_, se, rtol=1e-10, atol=1e-13)
    assert model.nobs == nobs
    assert model.df_resid == df
    statistic = beta / se
    reference = norm if slug == "fama-macbeth" else t(df)
    assert_allclose(model.tvalues_, statistic, rtol=1e-10, atol=1e-12)
    assert_allclose(model.pvalues_, 2 * reference.sf(np.abs(statistic)),
                    rtol=1e-9, atol=1e-14)
    critical = reference.ppf(0.975)
    expected_ci = np.column_stack([beta - critical * se, beta + critical * se])
    assert_allclose(model.conf_int_, expected_ci, rtol=1e-10, atol=1e-12)
    # The prose interprets the slope, rather than promising exact recovery.
    assert abs(model.coef_[-1] - 2.0) < 0.05
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("slug", MODELS)
def test_panel_cpu_examples_have_identical_bilingual_computations(slug):
    _, english = _read("en", slug)
    _, chinese = _read("cn", slug)
    assert example_code(english, "cpu-basic") == example_code(chinese, "cpu-basic")
    assert 'id="external-validation"' in english
    assert 'id="external-validation"' in chinese
    assert "## Comparing with Other Packages" in english
    assert "## 与其他软件包比较" in chinese


def test_random_effects_covariance_change_requires_refit():
    ns = _run("en", "random-effects")
    model, X, y, ids = ns["model"], ns["X"], ns["y"], ns["entity_ids"]
    old_coef = model.coef_.copy()
    old_se = model.bse_.copy()
    old_components = model.variance_components_.copy()
    model.set_params(cov_type="hc1")
    assert not model._fitted
    model.fit(X, y, entity_ids=ids)
    assert_allclose(model.coef_, old_coef, rtol=1e-12, atol=1e-13)
    assert model.variance_components_ == old_components
    assert not np.allclose(model.bse_, old_se)
    # Direct assignment also does not update previously computed inference.
    saved_se = model.bse_.copy()
    model.cov_type = "nonrobust"
    assert_allclose(model.bse_, saved_se, rtol=0, atol=0)


def test_fama_macbeth_rank_requirement_is_not_residual_ols_rank_behavior():
    ns = _run("en", "fama-macbeth")
    duplicate = np.column_stack([ns["X"], ns["X"]])
    with pytest.raises(ValueError, match="rank"):
        FamaMacBeth(device="cpu").fit(duplicate, ns["y"], time_ids=ns["time_ids"])
    _, text = _read("cn", "covariance")
    assert "每个保留时期的设计矩阵必须满列秩" in text
    assert "对于上表五类基于残差 OLS 的估计器" in text


def test_panel_comparison_and_semantic_clarifications_are_preserved():
    _, covariance = _read("cn", "covariance")
    assert "较细划分与交集项相消" in covariance
    assert "较粗划分的协方差" in covariance
    assert 'id="comparing-covariance-definitions"' in covariance
    _, english_cov = _read("en", "covariance")
    assert "HC1 adds a degrees-of-freedom correction" in english_cov
    assert "the five residual-OLS estimators do not expose that attribute" in english_cov
    assert "上表五类基于残差 OLS 的估计器不提供该属性" in covariance
    for language in ("en", "cn"):
        _, re_doc = _read(language, "random-effects")
        assert "set_params()" in re_doc
        _, fm_doc = _read(language, "fama-macbeth")
        for required in ("debiased=True", "debiased=False", "$T-1$", "all_params"):
            assert required in fm_doc
    _, cn_fm = _read("cn", "fama-macbeth")
    for obsolete in ("认证", "发布坐标级", "带入坐标级"):
        assert obsolete not in cn_fm
    _, en_diagnostic = _read("en", "diagnostics")
    _, cn_diagnostic = _read("cn", "diagnostics")
    assert "pooled model with a common intercept" in en_diagnostic
    assert "含公共截距的合并模型" in cn_diagnostic
