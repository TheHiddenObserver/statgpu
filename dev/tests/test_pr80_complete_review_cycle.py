"""Regression gates from the final complete PR80 review cycle."""

import numpy as np
import pytest

from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival._cox_score import (
    _MAX_CONCORDANCE_PAIR_ENTRIES,
    _concordance_batch_size,
)
from statgpu.survival._risk_sets import counting_process_concordance


def _fit_sample(seed=2401, n=36, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    stop = np.arange(1, n + 1, dtype=np.float64)
    event = np.ones(n, dtype=np.float64)
    event[::5] = 0.0
    event[0] = 1.0
    return X, stop, event


def test_ordinary_concordance_batch_is_bounded():
    batch = _concordance_batch_size(100_000, 1_000)
    assert batch == 2_000
    assert batch * 1_000 <= _MAX_CONCORDANCE_PAIR_ENTRIES
    assert _concordance_batch_size(0, 1_000) == 1


def test_all_censored_concordance_is_neutral_across_public_paths():
    X, stop, event = _fit_sample(p=1)
    fitted = CoxPH(
        compute_inference=False,
        compute_cindex=False,
        max_iter=80,
        tol=1e-7,
    ).fit(X, stop, event)
    X_score = X[:6]
    stop_score = np.arange(1, 7, dtype=np.float64)
    censored = np.zeros(6, dtype=np.float64)

    assert fitted.score(X_score, stop_score, censored) == 0.5
    assert fitted.score(
        X_score,
        stop_score,
        censored,
        start=np.zeros(6),
        strata=np.array([0, 0, 0, 1, 1, 1]),
    ) == 0.5
    assert float(
        counting_process_concordance(
            fitted.coef_,
            X_score,
            stop_score,
            censored,
            start=np.zeros(6),
            strata=np.array([0, 0, 0, 1, 1, 1]),
        )
    ) == 0.5


def test_penalized_cox_all_censored_score_is_neutral():
    X, stop, event = _fit_sample(seed=2402, p=1)
    model = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.2,
        max_iter=80,
        tol=1e-6,
        compute_inference=False,
    ).fit(X, np.column_stack((stop, event)))
    target = np.column_stack((stop[:5], np.zeros(5)))
    assert model.score(X[:5], target) == 0.5


def test_coxphcv_final_refit_skips_hidden_training_concordance():
    X, stop, event = _fit_sample(seed=2403)
    model = CoxPHCV(
        penalties=np.array([1.0]),
        cv=2,
        random_state=0,
        compute_inference=False,
        max_iter=100,
        tol=1e-6,
        device="cpu",
    ).fit(X, stop, event)
    assert model.estimator_.compute_cindex is False
    assert model.estimator_.concordance_ is None
    assert np.isfinite(model.score(X, stop, event))


@pytest.mark.parametrize(
    "name",
    ["fit_intercept", "gpu_memory_cleanup", "compute_inference", "lla"],
)
def test_penalized_cox_rejects_truthy_string_boolean_controls(name):
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        PenalizedCoxPHModel(**{name: "False"})

    model = PenalizedCoxPHModel()
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        model.set_params(**{name: "False"})


def test_penalized_cox_accepts_integer_boolean_controls_and_clones():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedCoxPHModel(
        fit_intercept=0,
        gpu_memory_cleanup=0,
        compute_inference=0,
        lla=1,
    )
    cloned = clone(model)
    assert cloned.fit_intercept == 0
    assert cloned.gpu_memory_cleanup == 0
    assert cloned.compute_inference == 0
    assert cloned.lla == 1
