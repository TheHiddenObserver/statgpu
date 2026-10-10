"""Execute named Cox learner examples and verify their public statistical contract."""

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from doc_examples import example_code, parse_examples, run_example

from statgpu.survival import CoxPH, CoxPHCV

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    "coxph-cpu-walkthrough",
    "coxph-cpu-cv",
    "coxph-survival-prediction",
    "coxph-summary",
    "coxph-cupy-fit",
    "coxph-cupy-cv",
    "coxph-torch-fit",
    "coxph-torch-cv",
    "coxph-penalized-family-cv",
)


def _page(language):
    return (ROOT / f"docs/{language}/models/coxph.md").read_text(encoding="utf-8")


def _example(language, name):
    """Select every declared step, never a language-dependent fence index."""
    return example_code(_page(language), name, f"{language}/coxph.md")


def _run(language, name):
    return run_example(_page(language), name, f"{language}/coxph.md")


@pytest.fixture(params=("en", "cn"))
def walkthrough(request):
    return request.param, _run(request.param, "coxph-cpu-walkthrough")


def _concordance(time, event, risk):
    """Small independent right-censored C-index oracle for continuous times."""
    comparable = (event[:, None] == 1) & (time[:, None] < time[None, :])
    assert comparable.sum() > 0
    concordant = (risk[:, None] > risk[None, :]).astype(float)
    concordant += 0.5 * (risk[:, None] == risk[None, :])
    return float(concordant[comparable].mean())


def _partial_log_likelihood(X, time, event, coef):
    """Untied held-out partial likelihood, independent of the CV scorer."""
    eta = X @ coef
    return sum(
        eta[i] - np.logaddexp.reduce(eta[time >= time[i]])
        for i in np.flatnonzero(event)
    )


def test_walkthrough_fit_and_hazard_ratio_semantics(walkthrough):
    _, ns = walkthrough
    model = ns["model"]
    assert model.effective_device_ == "cpu"
    assert model.converged_
    assert model.termination_reason_ == "kkt_converged"
    assert 0 < model.n_iter_ <= model.max_iter
    assert np.isfinite(model.final_kkt_inf_)
    assert model.final_kkt_normalized_ <= model.tol
    assert model.coef_.shape == (3,)
    assert model._conf_int.shape == (3, 2)
    assert np.all(np.isfinite(model._bse))
    assert set(np.unique(ns["event"])) == {0, 1}
    np.testing.assert_array_equal(ns["time"], np.minimum(ns["event_time"], ns["censor_time"]))
    np.testing.assert_allclose(model.coef_, [0.852, -0.457, 0.312], atol=0.001)
    np.testing.assert_allclose(model.hazard_ratios_, np.exp(model.coef_))
    np.testing.assert_allclose(ns["log_risk"], ns["X_test"][:2] @ model.coef_)
    np.testing.assert_allclose(ns["relative_hazard"], np.exp(ns["log_risk"]))
    np.testing.assert_allclose(
        model.predict_hazard_ratio(ns["X_test"][:2]), ns["relative_hazard"]
    )
    reference = ns["X_test"][:1].copy()
    changed = reference.copy()
    changed[0, 0] += 1
    np.testing.assert_allclose(
        model.predict(changed) / model.predict(reference), model.hazard_ratios_[0]
    )


def test_survival_subsection_reuses_walkthrough_and_held_out_score(walkthrough):
    language, _ = walkthrough
    ns = _run(language, "coxph-survival-prediction")
    model, curves, times = (ns[name] for name in ("model", "curves", "curve_times"))
    assert curves.shape == (2, 4)
    assert times.shape == (4,)
    np.testing.assert_array_equal(times, ns["requested_times"])
    assert np.all(np.isfinite(curves))
    assert np.all((curves >= 0) & (curves <= 1))
    np.testing.assert_array_equal(curves[:, 0], 1)
    assert np.all(np.diff(curves, axis=1) <= 0)
    assert np.all(curves[:, 1:] < 1)
    baseline, _ = model.predict_survival(np.zeros((1, 3)), times=times)
    np.testing.assert_allclose(curves, baseline ** ns["relative_hazard"][:, None])
    automatic_curves, automatic_times = model.predict_survival(ns["X_test"][:2])
    assert automatic_curves.shape == (2, len(automatic_times))
    np.testing.assert_array_equal(
        automatic_times, np.unique(ns["time_train"][ns["event_train"] == 1])
    )
    assert np.all(np.diff(automatic_curves, axis=1) <= 0)
    expected = _concordance(
        ns["time_test"], ns["event_test"], model.predict_risk_score(ns["X_test"])
    )
    assert ns["held_out_cindex"] == pytest.approx(expected)
    assert 0.70 < ns["held_out_cindex"] < 0.85
    assert ns["X_train"].shape == (300, 3)
    assert ns["X_test"].shape == (100, 3)


def test_cv_example_selection_folds_scores_and_final_refit(walkthrough):
    language, _ = walkthrough
    ns = _run(language, "coxph-cpu-cv")
    cv_model = ns["cv_model"]
    results = cv_model.cv_results_
    assert results["pl_path"].shape == (4, 3)
    assert results["mean_pl"].shape == (4,)
    assert results["effective_n_folds"] == 3
    np.testing.assert_array_equal(results["effective_fold_counts"], [3] * 4)
    assert results["fold_valid"].all()
    assert results["candidate_complete"].all()
    assert results["converged_path"].all()
    assert np.isfinite(results["pl_path"]).all()
    np.testing.assert_allclose(results["mean_pl"], results["pl_path"].mean(axis=1))
    best_index = int(np.argmax(results["mean_pl"]))
    assert cv_model.penalty_ == cv_model.penalties_[best_index] == 1.0
    assert cv_model.best_score_ == pytest.approx(results["mean_pl"][best_index])
    assert cv_model.best_score_ < 0  # A log-likelihood, not a C-index.
    validation_rows = []
    for fold_index, (train, validation) in enumerate(results["fold_indices"]):
        assert not np.intersect1d(train, validation).size
        np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(300))
        assert ns["event_train"][train].sum() > 0
        assert ns["event_train"][validation].sum() > 0
        validation_rows.extend(validation)
        # Refit the selected candidate independently, without the CV warm start,
        # and check its held-out likelihood rather than only metadata labels.
        candidate = CoxPH(
            penalty=cv_model.penalty_, ties="efron", device="cpu",
            compute_inference=False, compute_cindex=False,
        ).fit(ns["X_train"][train], ns["time_train"][train], ns["event_train"][train])
        assert candidate.converged_
        expected_pl = _partial_log_likelihood(
            ns["X_train"][validation], ns["time_train"][validation],
            ns["event_train"][validation], candidate.coef_,
        )
        assert results["pl_path"][best_index, fold_index] == pytest.approx(expected_pl)
    np.testing.assert_array_equal(np.sort(validation_rows), np.arange(300))
    refit = CoxPH(
        penalty=cv_model.penalty_, ties="efron", device="cpu", compute_inference=False,
    ).fit(ns["X_train"], ns["time_train"], ns["event_train"])
    assert cv_model.converged_
    assert cv_model.estimator_.penalty == cv_model.penalty_
    np.testing.assert_allclose(cv_model.coef_, refit.coef_, atol=1e-9)
    # Preserve the direct CPU/inference-disabled contract formerly exercised
    # by the now-removed duplicate backend-data/backend-cpu documentation.
    assert refit.converged_ and refit.effective_device_ == "cpu"
    risk = refit.predict_risk_score(ns["X_test"][:3])
    assert risk.shape == (3,)
    np.testing.assert_allclose(risk, ns["X_test"][:3] @ refit.coef_)
    assert refit._bse is None
    with pytest.raises(RuntimeError, match="compute_inference=True"):
        refit.predict_survival(ns["X_test"][:3])
    np.testing.assert_allclose(cv_model.coef_, cv_model.estimator_.coef_)
    np.testing.assert_allclose(cv_model.hazard_ratios_, np.exp(cv_model.coef_))
    expected_cindex = _concordance(
        ns["time_test"], ns["event_test"], cv_model.predict_risk_score(ns["X_test"])
    )
    assert ns["cv_test_cindex"] == pytest.approx(expected_cindex)
    assert cv_model._bse is None
    with pytest.raises(RuntimeError, match="compute_inference=True"):
        cv_model.predict_survival(ns["X_test"][:2])


@pytest.mark.parametrize("language", ("en", "cn"))
def test_penalized_family_example_reuses_training_partition(language):
    ns = _run(language, "coxph-penalized-family-cv")
    family = ns["penalized_cv"]
    assert ns["survival_y"].shape == (300, 2)
    np.testing.assert_array_equal(ns["survival_y"][:, 0], ns["time_train"])
    np.testing.assert_array_equal(ns["survival_y"][:, 1], ns["event_train"])
    assert family.coef_.shape == (3,)
    assert np.isfinite(family.coef_).all()
    assert family.alpha_ in (0.1, 0.03, 0.01)
    results = family.cv_results_
    best = int(np.nanargmax(results["mean_test_score"]))
    assert family.alpha_ == results["alpha"][best]
    assert results["valid_score_counts"][best] == results["required_valid_score_count"]
    assert results["n_effective_folds"] == 5
    assert results["final_refit_class"] == "PenalizedCoxPHModel"


@pytest.mark.parametrize("language", ("en", "cn"))
def test_summary_subsection_uses_the_fitted_walkthrough(language, capsys):
    ns = _run(language, "coxph-summary")
    assert ns["model"].compute_inference
    assert ns["model"]._conf_int.shape == (3, 2)
    assert "Cox" in capsys.readouterr().out


@pytest.mark.parametrize("language", ("en", "cn"))
def test_gpu_subsections_declare_backend_and_training_prerequisites(language):
    examples = parse_examples(_page(language), f"{language}/coxph.md")
    assert set(examples) == set(EXAMPLES)
    for backend in ("cupy", "torch"):
        fit = examples[f"coxph-{backend}-fit"]
        cv = examples[f"coxph-{backend}-cv"]
        assert fit.requires == ("coxph-cpu-walkthrough",)
        assert cv.requires == (fit.name,)
        # Static only: do not substitute CPU arrays for CUDA tensors.
        for code in (fit.code, cv.code):
            ast.parse(code)
            assert "compute_inference=False" in code
            assert f'device="{"cuda" if backend == "cupy" else "torch"}"' in code
        for name in ("X_train", "time_train", "event_train", "X_test[:3]"):
            assert name in fit.code
        assert "random" not in fit.code
        assert "CoxPHCV" in cv.code


def test_translations_keep_executable_examples_aligned():
    for name in EXAMPLES:
        # Translated comments may differ; the executable statements must agree.
        assert ast.dump(ast.parse(_example("en", name))) == ast.dump(
            ast.parse(_example("cn", name))
        )


@pytest.mark.parametrize("language", ("en", "cn"))
def test_complete_constructor_references_match_current_api(language):
    text = _page(language)
    for estimator, title in ((CoxPH, "Parameters" if language == "en" else "参数"),
                             (CoxPHCV, "CoxPHCV Parameters" if language == "en" else "CoxPHCV 参数")):
        section = re.split(r"\n#{2,3} ", text.split(f"# {title}\n", 1)[1], maxsplit=1)[0]
        for name in inspect.signature(estimator).parameters:
            assert f"| `{name}` |" in section, f"{language}: missing {estimator.__name__}.{name}"
    for source in ("_cox.py", "_cox_cv.py"):
        assert f"../../../statgpu/survival/{source}" in text
        assert (ROOT / "statgpu/survival" / source).is_file()
