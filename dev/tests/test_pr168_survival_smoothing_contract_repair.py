"""Runtime checks for survival/smoothing documentation repaired in PR 168."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from statgpu.nonparametric import (
    KernelDensityEstimator,
    KernelRegression,
    kde_confidence_interval,
)
from statgpu.nonparametric.splines._penalized import difference_penalty
from statgpu.semiparametric import GAM
from statgpu.survival import CoxPH, CoxPHCV

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cls", (GAM, CoxPH, CoxPHCV))
def test_runtime_constructor_help_covers_every_parameter(cls):
    doc = inspect.getdoc(cls)
    for name in inspect.signature(cls).parameters:
        assert f"{name} :" in doc


def test_gam_actual_basis_shape_and_fresh_instance_recovery(capsys):
    x = np.r_[np.linspace(-2, -.1, 10), np.zeros(10), np.linspace(.1, 2, 10)]
    model = GAM(n_splines=8, lam=1, device="cpu").fit(x, np.sin(x))
    actual_basis_count = len(model.knots_[0]) + model.degree + 1
    assert actual_basis_count < model.n_splines
    assert model.coef_.shape == (1 + actual_basis_count,)
    assert model.gcv_score_ is None
    assert "gcv_score" not in model.summary()
    broken_design = np.column_stack([
        np.linspace(-1, 1, 30), np.r_[np.zeros(27), np.ones(3)],
    ])
    with pytest.raises(ValueError, match="strictly within"):
        model.fit(broken_design, np.arange(30))
    # Follow the documented recovery without treating the failed instance's
    # partially updated fitted fields as a statistical result.
    recovered = GAM(n_splines=8, lam=1, device="cpu").fit(x, np.sin(x))
    assert recovered.summary()["n_features"] == 1
    assert np.all(np.isfinite(recovered.predict([-.5, 0, .5])))
    capsys.readouterr()


def test_float32_inputs_have_documented_float64_outputs():
    x = np.linspace(-2, 2, 40, dtype=np.float32)
    y = np.sin(x).astype(np.float32)
    q = np.array([-.5, 0, .5], dtype=np.float32)
    kde = KernelDensityEstimator(backend="numpy", device="cpu").fit(x)
    reg = KernelRegression(backend="numpy", device="cpu").fit(x, y)
    gam = GAM(n_splines=8, lam=1, device="cpu").fit(x, y)
    for model in (kde, reg, gam):
        assert model.predict(q).dtype == np.float64
    assert kde.samples_.dtype == reg.targets_.dtype == gam.coef_.dtype == np.float64
    cox = CoxPH(compute_inference=False, device="cpu").fit(
        x[:, None], np.random.default_rng(203).exponential(size=40).astype(np.float32),
        np.ones(40, dtype=np.float32),
    )
    assert cox.coef_.dtype == cox.predict(q[:, None]).dtype == np.float64


def test_normal_ci_requires_percentile_bootstrap_control_without_resampling():
    x = np.linspace(-2, 2, 30)
    result = kde_confidence_interval(x, [0], backend="numpy", method="normal")
    assert result.n_resamples == 0
    with pytest.raises(ValueError, match="bootstrap_method"):
        kde_confidence_interval(x, [0], backend="numpy", method="normal", bootstrap_method="bca")


def test_unused_kde_y_is_still_subject_to_public_finite_validation():
    x = np.linspace(-2, 2, 20)
    model = KernelDensityEstimator(backend="numpy", device="cpu").fit(x, np.zeros(20))
    assert model.score(x, np.ones(20)) == model.score(x)
    with pytest.raises(ValueError, match="finite"):
        model.score(x, np.full(20, np.nan))
    with pytest.raises(ValueError, match="finite"):
        KernelDensityEstimator(backend="numpy", device="cpu").fit(x, np.full(20, np.nan))


def test_cox_cv_fold_sum_scale_input_order_and_reusable_split_iterator():
    rng = np.random.default_rng(207)
    x = rng.normal(size=(45, 2))
    time = rng.exponential(np.exp(-x @ np.array([.2, -.1])))
    event = np.ones(45, dtype=int)
    indices = np.arange(45)
    tests = (indices[:10], indices[10:25], indices[25:])
    splits = [(np.setdiff1d(indices, test), test) for test in tests]
    one_shot = iter(splits)
    model = CoxPHCV(penalties=[.1, 1.], cv_splits=one_shot,
                    compute_inference=False, device="cpu")
    params = model.get_params(deep=False)
    assert isinstance(params["cv_splits"], list)
    assert model.cv_splits is one_shot
    reconstructed = clone(model)
    model.fit(x, time, event)
    reconstructed.fit(x, time, event)
    np.testing.assert_array_equal(model.penalties_, [.1, 1.])
    np.testing.assert_array_equal(model.cv_results_["penalty_evaluation_order"], [1., .1])
    np.testing.assert_allclose(model.cv_results_["pl_path"], reconstructed.cv_results_["pl_path"])
    # Independent no-tie held-out partial-log-likelihood sum for unequal folds.
    for penalty_index, penalty in enumerate(model.penalties_):
        for fold_index, (train, test) in enumerate(splits):
            fitted = CoxPH(penalty=penalty, compute_inference=False, device="cpu").fit(
                x[train], time[train], event[train],
            )
            eta = x[test] @ fitted.coef_
            expected = sum(eta[i] - np.log(np.exp(eta[time[test] >= t]).sum())
                           for i, t in enumerate(time[test]))
            assert np.isclose(model.cv_results_["pl_path"][penalty_index, fold_index],
                              expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(model.cv_results_["mean_pl"], model.cv_results_["pl_path"].mean(axis=1))


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_records_dtype_and_cv_iterator_contracts(language):
    page = (ROOT / f"docs/{language}/reference/survival-smoothing-api.md").read_text()
    assert "float64" in page
    assert "float32" in page
    assert "`get_params`" in page
    assert "`best_score_`" in page
    assert "`bootstrap_method" in page


def test_difference_order_help_does_not_claim_spline_degree():
    doc = inspect.getdoc(difference_penalty)
    assert "does not set the polynomial degree" in doc


def test_custom_cox_grid_near_tie_prefers_stronger_and_restores_input_order(monkeypatch):
    from statgpu.survival import _cox_cv_penalty_order_contract as contract

    def selection(*args, **kwargs):
        np.testing.assert_array_equal(kwargs["penalties"], [10., 1.])
        assert kwargs["return_details"] is True
        return 1., {
            "penalty": 1., "best_pl": -10.,
            "mean_pl": np.array([-10. - 5e-10, -10.]),
            "candidate_complete": np.ones(2, dtype=bool),
            "penalties": kwargs["penalties"],
        }

    monkeypatch.setattr(contract, "_ORIGINAL_SELECT_COXPH_PENALTY_CV", selection)
    penalty, details = contract._select_coxph_penalty_cv_order_invariant(
        penalties=np.array([1., 10.]), return_details=True,
    )
    assert penalty == 10.
    assert details["best_pl"] < details["mean_pl"].max()
    np.testing.assert_array_equal(details["penalties"], [1., 10.])
    np.testing.assert_array_equal(details["mean_pl"], [-10., -10. - 5e-10])
