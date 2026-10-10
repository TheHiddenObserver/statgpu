"""Executable safeguards for the second PR 168 survival/smoothing review."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from statgpu.nonparametric import (
    KernelDensityEstimator,
    KernelRegression,
    fit_kde,
    kde_confidence_interval,
    kde_pdf,
)
from statgpu.semiparametric import GAM
from statgpu.survival import CoxPH

ROOT = Path(__file__).resolve().parents[2]


def test_cox_penalty_help_matches_independent_analytic_score():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(60, 2))
    time = rng.exponential(np.exp(-x @ np.array([0.8, -0.6])))
    penalty = 10.0
    model = CoxPH(penalty=penalty, device="cpu", compute_inference=False).fit(
        x, time, np.ones(len(x), dtype=int),
    )
    assert model.converged_
    # Independent no-tie score of the summed unpenalized likelihood.
    risk = np.exp(x @ model.coef_)
    score = np.zeros(x.shape[1])
    for i, stop in enumerate(time):
        at_risk = time >= stop
        score += x[i] - (risk[at_risk, None] * x[at_risk]).sum(axis=0) / risk[at_risk].sum()
    np.testing.assert_allclose(score, 2 * penalty * model.coef_, rtol=1e-7, atol=1e-7)
    assert np.max(np.abs(score - penalty * model.coef_)) > 1.0
    assert "no factor of 1/2" in inspect.getdoc(CoxPH)


@pytest.mark.parametrize("obj", (KernelDensityEstimator, KernelRegression, kde_confidence_interval))
def test_smoothing_help_documents_constructor_or_function_arguments(obj):
    doc = inspect.getdoc(obj)
    for name in inspect.signature(obj).parameters:
        assert f"{name} :" in doc, (obj.__name__, name)


def test_non_gaussian_one_shot_density_matches_reusable_model():
    samples = np.linspace(-2, 2, 40)
    query = np.array([-0.5, 0, 0.5, 100])
    model = fit_kde(samples, bandwidth=0.5, kernel="epanechnikov", backend="numpy")
    actual = kde_pdf(samples, query, bandwidth=0.5, kernel="epanechnikov", backend="numpy")
    np.testing.assert_allclose(actual, model.pdf(query))
    assert actual[-1] == 0
    assert "all supported kernels" in inspect.getdoc(kde_pdf)


def test_zero_weight_filter_workaround_has_finite_analytic_tail_log_density():
    samples = np.array([0.0, 100.0, 101.0])
    weights = np.array([0.0, 0.5, 0.5])
    keep = weights > 0
    model = fit_kde(samples[keep], weights=weights[keep], bandwidth=0.5, backend="numpy")
    query = np.array([0.0])
    variance = model.covariance_[0, 0]
    expected = logsumexp(np.log(model.weights_) - samples[keep] ** 2 / (2 * variance))
    expected -= 0.5 * np.log(2 * np.pi * variance)
    assert np.isfinite(expected)
    np.testing.assert_allclose(model.logpdf(query), [expected], rtol=1e-12)
    assert np.isfinite(model.score(query))


def test_automatic_gam_selection_caller_guard_rejects_no_finite_candidate():
    x = np.linspace(-1, 1, 20)
    # Follow the documented guard rather than interpreting an invalid search.
    # A future runtime fail-closed repair may raise before the caller guard.
    with pytest.raises((RuntimeError, ValueError), match="finite|GCV"):
        model = GAM(n_splines=6, gamma=100, device="cpu").fit(x, x)
        if not np.isfinite(model.gcv_score_):
            raise RuntimeError("No finite GCV candidate; revise the model before prediction.")


@pytest.mark.parametrize("language", ("en", "cn"))
def test_learner_gam_example_checks_selection_before_predicting(language):
    source = (ROOT / f"docs/{language}/models/semiparametric.md").read_text()
    guard = source.index("if not np.isfinite(gam.gcv_score_):")
    prediction = source.index("prediction = gam.predict(X_test)")
    assert guard < prediction


@pytest.mark.parametrize("rule", ["nrd", "nrd0"])
def test_normal_reference_bandwidth_rules_ignore_r_selector_toggle(rule):
    from statgpu.nonparametric.kernel_smoothing._bandwidth_selection import (
        select_bandwidth,
    )

    x = np.linspace(-2.0, 3.0, 30).reshape(-1, 1)
    inputs = {
        "n_eff": float(len(x)), "n_features": 1, "samples_2d": x,
        "weights_1d": np.full(len(x), 1.0 / len(x)),
        "data_cov": np.atleast_2d(np.cov(x, rowvar=False)), "xp": np,
    }
    enabled = select_bandwidth(rule, enable_r_selectors=True, **inputs)
    disabled = select_bandwidth(rule, enable_r_selectors=False, **inputs)
    assert np.isfinite(disabled.factor) and disabled.factor > 0
    assert disabled.factor == enabled.factor
    assert not disabled.used_r_selector
    for selector in ("ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
        with pytest.raises(ValueError, match="disabled"):
            select_bandwidth(selector, enable_r_selectors=False, **inputs)
