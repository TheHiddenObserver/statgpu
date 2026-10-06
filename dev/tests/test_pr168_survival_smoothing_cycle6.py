"""Cycle-6 independent smoothing-unit and public guidance checks (NumPy CPU)."""
import inspect
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import gaussian_kde

from statgpu.nonparametric import KernelDensityEstimator, KernelRegression, fit_kde

ROOT = Path(__file__).resolve().parents[2]


class _AbsoluteCovarianceFloorChangesSmoothing(Exception):
    """Finite outputs have the independently identified absolute-floor signature."""


def _fixture():
    return np.linspace(-2., 2., 41), np.array([-.8, 0., .8]), 1e-9


def _assert_covariance_floor_signature(base, small, x, scale):
    """Identify this defect before using an expected-failure marker."""
    raw_variance = np.var(scale * x, ddof=1)
    expected_small_cov = small.bandwidth_factor_ ** 2 * (
        raw_variance + max(raw_variance * 1e-12, 1e-12)
    )
    assert small.covariance_.shape == base.covariance_.shape == (1, 1)
    assert np.isfinite(small.covariance_).all()
    assert_allclose(small.covariance_[0, 0], expected_small_cov, rtol=1e-12)
    assert small.covariance_[0, 0] / scale ** 2 > 1e5 * base.covariance_[0, 0]


@pytest.mark.xfail(strict=True, raises=_AbsoluteCovarianceFloorChangesSmoothing,
                   reason="Issue #239: absolute covariance floor changes smoothing under unit conversion")
@pytest.mark.parametrize("bandwidth", [.4, "scott", "silverman"])
def test_kde_numeric_scott_silverman_should_respect_change_of_units(bandwidth):
    x, query, scale = _fixture()
    base = fit_kde(x, bandwidth=bandwidth, backend="numpy")
    small = fit_kde(scale * x, bandwidth=bandwidth, backend="numpy")
    density = scale * small.pdf(scale * query)
    log_density = small.logpdf(scale * query) + np.log(scale)
    expected = gaussian_kde(x, bw_method=bandwidth)(query)
    assert density.shape == log_density.shape == expected.shape == (3,)
    assert np.isfinite(density).all() and np.isfinite(log_density).all()
    assert_allclose(base.pdf(query), expected, rtol=1e-10)
    assert_allclose(np.exp(log_density), density, rtol=1e-12)
    if not np.allclose(density, expected, rtol=1e-8, atol=1e-12):
        _assert_covariance_floor_signature(base, small, x, scale)
        variance = small.covariance_[0, 0]
        independent = np.exp(-.5 * (scale * query[:, None] - scale * x[None, :]) ** 2 / variance).mean(axis=1)
        independent *= scale / np.sqrt(2 * np.pi * variance)
        assert_allclose(density, independent, rtol=1e-12)
        assert_allclose(log_density, np.log(independent), rtol=1e-12)
        raise _AbsoluteCovarianceFloorChangesSmoothing("KDE density changes after coordinate-unit conversion")
    assert_allclose(log_density, np.log(expected), rtol=1e-8, atol=1e-12)


@pytest.mark.xfail(strict=True, raises=_AbsoluteCovarianceFloorChangesSmoothing,
                   reason="Issue #239: absolute covariance floor changes smoothing under unit conversion")
@pytest.mark.parametrize("bandwidth", [.4, "scott", "silverman"])
@pytest.mark.parametrize("regression", ["nw", "local_linear"])
def test_kernel_regression_should_respect_change_of_units(bandwidth, regression):
    x, query, scale = _fixture()
    y = np.sin(2 * x)
    base = KernelRegression(bandwidth=bandwidth, regression=regression, backend="numpy").fit(x, y)
    small = KernelRegression(bandwidth=bandwidth, regression=regression, backend="numpy").fit(scale * x, y)
    prediction, expected = small.predict(scale * query), base.predict(query)
    assert prediction.shape == expected.shape == (3,)
    assert np.isfinite(prediction).all() and np.isfinite(expected).all()
    if not np.allclose(prediction, expected, rtol=1e-8, atol=1e-12):
        _assert_covariance_floor_signature(base, small, x, scale)
        # Independently solve the actual local equations using the observed H.
        independent = []
        for point in scale * query:
            delta = scale * x - point
            weight = np.exp(-.5 * delta ** 2 / small.covariance_[0, 0]) / len(x)
            if regression == "nw":
                independent.append(weight @ y / weight.sum())
            else:
                design = np.column_stack([np.ones(len(x)), delta])
                independent.append(np.linalg.solve(design.T @ (weight[:, None] * design),
                                                   design.T @ (weight * y))[0])
        assert_allclose(prediction, independent, rtol=1e-8, atol=1e-12)
        raise _AbsoluteCovarianceFloorChangesSmoothing("Local regression changes after coordinate-unit conversion")


def test_standardized_coordinate_workflow_uses_density_jacobian_and_unchanged_response():
    x, query, unit = _fixture()
    raw = unit * x
    location, scale = raw.mean(), raw.std(ddof=1)
    z, z_query = (raw - location) / scale, (unit * query - location) / scale
    model = fit_kde(z, bandwidth=.4, backend="numpy")
    expected = gaussian_kde(raw, bw_method=.4)(unit * query)
    assert_allclose(model.pdf(z_query) / scale, expected, rtol=1e-10)
    assert_allclose(model.logpdf(z_query) - np.log(scale), np.log(expected), rtol=1e-10)
    for mode in ["nw", "local_linear"]:
        standardized = KernelRegression(bandwidth=.4, regression=mode, backend="numpy").fit(z, np.sin(2 * x))
        ordinary = KernelRegression(bandwidth=.4, regression=mode, backend="numpy").fit(x, np.sin(2 * x))
        assert_allclose(standardized.predict(z_query), ordinary.predict(query), atol=1e-12)


@pytest.mark.parametrize("selector", ["nrd", "nrd0"])
def test_one_dimensional_absolute_rule_can_compensate_covariance_increment(selector):
    x, query, scale = _fixture()
    base = fit_kde(x, bandwidth=selector, backend="numpy")
    small = fit_kde(scale * x, bandwidth=selector, backend="numpy")
    # Absolute-width selectors are intentionally distinguished from factor rules.
    assert small.bandwidth_factor_ != pytest.approx(base.bandwidth_factor_)
    assert_allclose(small.covariance_ / scale ** 2, base.covariance_, rtol=1e-12)
    assert_allclose(scale * small.pdf(scale * query), base.pdf(query), rtol=1e-12)


def test_absolute_regression_widths_transform_with_feature_units():
    x, query, scale = _fixture()
    y = np.sin(2 * x)
    options = {"kernel_metric": "diagonal", "regression": "nw", "backend": "numpy"}
    base = KernelRegression(bandwidth_per_feature=[.4], **options).fit(x, y)
    small = KernelRegression(bandwidth_per_feature=[scale * .4], **options).fit(scale * x, y)
    assert_allclose(small.covariance_ / scale ** 2, base.covariance_, rtol=1e-12)
    assert_allclose(small.predict(scale * query), base.predict(query), atol=1e-12)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_smoothing_unit_guidance_gives_density_and_response_mapping(language):
    text = (ROOT / f"docs/{language}/models/nonparametric.md").read_text()
    for phrase in ["1e-9", "0.243898", "0.000997", "density_x = density_z / np.prod(scale)",
                   "log_density_x = log_density_z - np.log(scale).sum()", "bandwidth_per_feature",
                   "Scott", "Silverman", "nrd`/`nrd0"]:
        assert phrase in text
    reference = (ROOT / f"docs/{language}/reference/survival-smoothing-api.md").read_text()
    anchor = "small-coordinate-scales" if language == "en" else "坐标尺度过小"
    assert f"../models/nonparametric.md#{anchor}" in reference
    for cls in [KernelDensityEstimator, KernelRegression]:
        assert "absolute covariance stabilization" in inspect.getdoc(cls).replace("\n", " ")
    assert "density Jacobian" in inspect.getdoc(KernelRegression)


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
@pytest.mark.parametrize("surface", ["density", "regression"])
def test_issue239_marker_does_not_mask_unrelated_failures(monkeypatch, kind, surface):
    """The narrow xfails above must not classify other defects as issue239."""
    def change(model, original, logarithm=False):
        def altered(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if np.ptp(self.samples_) >= 1e-7:
                return result
            if kind == "exception":
                raise RuntimeError("unrelated injected evaluation failure")
            if kind == "shape":
                return result[:-1]
            if kind == "nonfinite":
                result = result.copy()
                result[0] = np.nan
                return result
            # Keep PDF and logPDF internally consistent; the independent mixture
            # oracle, rather than their mutual agreement, must reject this case.
            return result + np.log(.9) if logarithm else result * .9
        return altered

    if surface == "density":
        monkeypatch.setattr(KernelDensityEstimator, "pdf", change(KernelDensityEstimator, KernelDensityEstimator.pdf))
        monkeypatch.setattr(KernelDensityEstimator, "logpdf", change(KernelDensityEstimator, KernelDensityEstimator.logpdf, True))
        probe = lambda: test_kde_numeric_scott_silverman_should_respect_change_of_units(.4)
    else:
        monkeypatch.setattr(KernelRegression, "predict", change(KernelRegression, KernelRegression.predict))
        probe = lambda: test_kernel_regression_should_respect_change_of_units(.4, "nw")
    expected_error = RuntimeError if kind == "exception" else AssertionError
    with pytest.raises(expected_error):
        probe()


def _summed_cox_loglik_score(X, time, event, coef, ties):
    """Small independent tied right-censoring oracle, with no statgpu helpers."""
    eta = X @ coef
    exp_eta = np.exp(eta)
    loglik, score = 0., np.zeros(X.shape[1])
    for t in np.unique(time[event == 1]):
        failures = (time == t) & (event == 1)
        risk = time >= t
        count = int(failures.sum())
        loglik += eta[failures].sum()
        score += X[failures].sum(axis=0)
        risk_sum = exp_eta[risk].sum()
        risk_moment = (exp_eta[risk, None] * X[risk]).sum(axis=0)
        event_sum = exp_eta[failures].sum()
        event_moment = (exp_eta[failures, None] * X[failures]).sum(axis=0)
        for index in range(count):
            fraction = index / count if ties == "efron" else 0.
            denominator = risk_sum - fraction * event_sum
            loglik -= np.log(denominator)
            score -= (risk_moment - fraction * event_moment) / denominator
    return loglik, score


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_penalized_cox_loss_and_gradient_use_rows_not_event_count(ties):
    from statgpu.losses import CoxPartialLikelihoodLoss
    from statgpu.penalties import L2Penalty

    X = np.array([[-1., .5], [-.4, -1.], [.2, .3], [.7, -.2], [1.2, .9], [-.2, 1.4]])
    time = np.array([1., 1., 2., 3., 3., 4.])
    event = np.array([1, 0, 1, 1, 1, 0])
    coef = np.array([.3, -.15])
    y, n, alpha = np.column_stack([time, event]), len(time), .07
    loglik, score = _summed_cox_loglik_score(X, time, event, coef, ties)
    loss = CoxPartialLikelihoodLoss(ties=ties)
    assert n != event.sum()
    assert_allclose(loss.value(X, y, coef), -loglik / n, atol=1e-14)
    assert_allclose(loss.gradient(X, y, coef), -score / n, atol=1e-14)
    step = 1e-6
    finite_difference = []
    for index in range(len(coef)):
        direction = np.eye(len(coef))[index] * step
        upper = _summed_cox_loglik_score(X, time, event, coef + direction, ties)[0]
        lower = _summed_cox_loglik_score(X, time, event, coef - direction, ties)[0]
        finite_difference.append(-(upper - lower) / (2 * step * n))
    assert_allclose(loss.gradient(X, y, coef), finite_difference, atol=1e-9)
    penalty = L2Penalty(alpha=alpha)
    assert_allclose(n * (loss.value(X, y, coef) + penalty.value(coef)),
                    -loglik + n * alpha / 2 * (coef @ coef), atol=1e-14)
    assert_allclose(n * (loss.gradient(X, y, coef) + penalty.gradient(coef)),
                    -score + n * alpha * coef, atol=1e-14)


def _cox_example_data():
    rng = np.random.default_rng(68)
    X = rng.normal(size=(80, 2))
    event_time = rng.exponential(np.exp(-X @ np.array([.4, -.3])))
    censor_time = rng.exponential(2., size=len(X))
    time = np.maximum(.1, np.round(np.minimum(event_time, censor_time), 1))
    event = (event_time <= censor_time).astype(int)
    return X, time, event


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_same_data_l2_conversion_matches_canonical_cox_and_stationarity(ties):
    from statgpu import PenalizedCoxPHModel
    from statgpu.survival import CoxPH

    X, time, event = _cox_example_data()
    n, alpha = len(X), .03
    family = PenalizedCoxPHModel(penalty="l2", alpha=alpha, ties=ties, solver="newton",
                                tol=1e-9, max_iter=100, device="cpu").fit(X, np.c_[time, event])
    canonical = CoxPH(penalty=n * alpha / 2, ties=ties, tol=1e-9, compute_inference=False,
                      device="cpu").fit(X, time, event)
    assert canonical.converged_
    assert family.coef_.shape == canonical.coef_.shape == (2,)
    assert np.isfinite(family.coef_).all()
    assert_allclose(family.coef_, canonical.coef_, atol=2e-7)
    score = _summed_cox_loglik_score(X, time, event, family.coef_, ties)[1]
    assert_allclose(-score / n + alpha * family.coef_, 0., atol=1e-8)


def test_penalized_cox_cv_averages_row_normalized_validation_losses():
    from statgpu import PenalizedCoxPHModel, PenalizedGLM_CV

    X, time, event = _cox_example_data()
    y = np.c_[time, event]
    folds = [(np.arange(20, 80), np.arange(20)), (np.arange(50), np.arange(50, 80))]
    alphas = [.1, .03]
    cv = PenalizedGLM_CV(loss="cox_ph", penalty="l2", alpha_grid=alphas, cv_splits=folds,
                         cv=2, solver="newton", cv_strategy="strict", max_iter=100,
                         tol=1e-9, device="cpu", loss_kwargs={"ties": "efron"}).fit(X, y)
    expected = np.empty((2, 2))
    for fi, (train, validation) in enumerate(folds):
        for ai, alpha in enumerate(alphas):
            fit = PenalizedCoxPHModel(penalty="l2", alpha=alpha, ties="efron", solver="newton",
                                      max_iter=100, tol=1e-9, device="cpu").fit(X[train], y[train])
            loglik = _summed_cox_loglik_score(X[validation], time[validation], event[validation], fit.coef_, "efron")[0]
            expected[fi, ai] = -loglik / len(validation)
    assert np.isfinite(cv.cv_results_["all_scores"]).all()
    assert_allclose(cv.cv_results_["all_scores"], expected, rtol=1e-7, atol=1e-9)
    assert_allclose(cv.cv_results_["mean_score"], expected.mean(axis=0), rtol=1e-7)
    assert cv.best_score_ == pytest.approx(-expected.mean(axis=0).min(), rel=1e-7)
    assert cv.estimator_.alpha == cv.alpha_


def test_installed_cox_help_and_bilingual_docs_distinguish_objective_scales():
    from statgpu import PenalizedCoxPHModel
    from statgpu.losses import CoxPartialLikelihoodLoss

    doc = inspect.getdoc(PenalizedCoxPHModel)
    assert "-ell(coef) / n" in doc and "n * alpha / 2" in doc
    assert "including censored observations" in doc
    assert "row-count normalization, not the event count" in inspect.getdoc(CoxPartialLikelihoodLoss)
    for language in ["en", "cn"]:
        text = (ROOT / f"docs/{language}/models/coxph.md").read_text()
        for phrase in [r"-\ell(\beta)/n", "CoxPH(penalty=n*a/2)",
                       "-ell_validation / n_validation", "best_score_", "alpha=1.0"]:
            assert phrase in text
