"""Focused audience regressions and CPU Cox-fold behavior behind the guidance.

These checks protect known placement/support boundaries. They do not prove that
all public prose is clear, complete, or free of maintainer-oriented language.
"""
from pathlib import Path

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.survival import CoxPHCV

ROOT = Path(__file__).resolve().parents[2]


def read(language, path):
    return (ROOT / f"docs/{language}/{path}.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("language", ("en", "cn"))
def test_model_guidance_uses_public_results_and_links_low_level_contracts(language):
    quantile = read(language, "models/quantile")
    solver = read(language, "guides/solver-algorithms")
    assert "solver-algorithms.md#quantile-low-level-solvers" in quantile
    assert 'id="quantile-low-level-solvers"' in solver
    for private_detail in ("LossBase", "fista_lla_path", "alpha_path"):
        assert private_detail not in quantile
    for public_contract in ("sample_weight", 'solver="fista"', "ConvergenceWarning"):
        assert public_contract in quantile
    assert "two-stage" in quantile.lower() if language == "en" else "两阶段" in quantile
    for low_level_api in ("fista_lla_path", "alpha_path", "return_path", "n_iter"):
        assert low_level_api in solver
    fama = read(language, "panel/fama-macbeth")
    assert "_inference_result" not in fama
    for result in ("coef_", "betas_", "cov_params_", "bse_", "pvalues_", "conf_int_", "summary()"):
        assert result in fama
    panel_covariance = read(language, "panel/covariance")
    assert "ols_covariance" not in panel_covariance
    for api in ("clustered_covariance", "two_way_clustered_covariance", "hac_covariance",
                "driscoll_kraay_covariance", "cov_type", "cov_params_"):
        assert api in panel_covariance


@pytest.mark.parametrize("language", ("en", "cn"))
def test_solver_heuristics_stay_in_developer_reference(language):
    solver = read(language, "guides/solver-algorithms")
    for private_detail in ("1.5 times", "1.5 倍", "warmup-then-sparse"):
        assert private_detail not in solver
    internal = (ROOT / "dev/design/solver-implementation-notes.md").read_text(encoding="utf-8")
    for detail in ("1.5", "Cholesky", "synchron"):
        assert detail in internal
    # Public low-level APIs, numerical reasoning and observable warnings remain.
    for contract in ("sample_weight", "ConvergenceWarning", "float64", "FISTA", "L-BFGS"):
        assert contract in solver


@pytest.mark.parametrize("language", ("en", "cn"))
def test_cox_guide_names_target_and_distinct_fold_failure_outcomes(language):
    guide = read(language, "guides/cross-validation")
    for token in ("n_samples", "time", "event", "fold_valid", "failure_path",
                  "n_effective_folds", "ValueError", "RuntimeError"):
        assert token in guide
    for stale in ("Invalid partitions raise before candidate selection",
                  "不合法的分区会在候选项选择之前报错", "materializes that iterator privately"):
        assert stale not in guide


@pytest.fixture
def cox_fold_data():
    rng = np.random.default_rng(314)
    x = rng.normal(size=(80, 2))
    time = rng.exponential(size=80) + 0.1
    event = np.r_[np.ones(45, dtype=int), np.zeros(35, dtype=int)]
    return x, time, event


def make_cox_model(api, splits):
    shared = dict(cv_splits=splits, device="cpu", max_iter=500,
                  tol=1e-7, compute_inference=False)
    if api == "PenalizedGLM_CV":
        return PenalizedGLM_CV(loss="cox_ph", penalty="l2", alpha_grid=[0.01, 0.1], **shared)
    return CoxPHCV(penalties=[0.01, 0.1], **shared)


def fit_cox(model, data):
    x, time, event = data
    if isinstance(model, PenalizedGLM_CV):
        return model.fit(x, np.column_stack((time, event)))
    return model.fit(x, time, event)


@pytest.mark.parametrize("api", ("PenalizedGLM_CV", "CoxPHCV"))
def test_cox_event_free_fold_can_be_excluded_without_preventing_refit(api, cox_fold_data):
    splits = [(np.arange(30), np.arange(30, 50)),
              (np.arange(60), np.arange(60, 80))]
    model = make_cox_model(api, splits)
    fit_cox(model, cox_fold_data)
    assert model.estimator_ is not None
    assert np.isfinite(model.coef_).all()
    np.testing.assert_array_equal(model.cv_results_["fold_valid"], [True, False])
    if api == "PenalizedGLM_CV":
        assert model.cv_results_["n_effective_folds"] == 1
        np.testing.assert_array_equal(model.cv_results_["validation_event_counts"], [15, 0])
        assert set(model.cv_results_["failure_path"][1]) == {"missing_validation_event"}
    else:
        np.testing.assert_array_equal(model.cv_results_["effective_fold_counts"], [1, 1])
        assert set(model.cv_results_["failure_path"][:, 1]) == {"fold_has_no_train_or_test_events"}


@pytest.mark.parametrize("api", ("PenalizedGLM_CV", "CoxPHCV"))
@pytest.mark.parametrize("case, exception", (
    ("all_event_free", RuntimeError),
    ("overlap", ValueError),
    ("noninteger", ValueError),
))
def test_cox_unevaluable_and_malformed_folds_have_distinct_failures(api, case, exception, cox_fold_data):
    splits = {
        "all_event_free": [(np.arange(60), np.arange(60, 70)),
                           (np.arange(60), np.arange(70, 80))],
        "overlap": [(np.arange(30), np.arange(25, 50))],
        "noninteger": [(np.array([0.1, 1.0, 2.0]), np.arange(30, 50))],
    }[case]
    with pytest.raises(exception):
        fit_cox(make_cox_model(api, splits), cox_fold_data)
