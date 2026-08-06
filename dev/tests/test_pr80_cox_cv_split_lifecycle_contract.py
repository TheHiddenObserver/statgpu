"""CoxPHCV custom-fold iterator lifecycle regressions."""

from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv_split_lifecycle_contract as lifecycle


def _folds():
    return [
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]


def _generator():
    yield from _folds()


def test_one_shot_cv_splits_are_reused_without_rewriting_public_parameter(
    monkeypatch,
):
    observed = []

    def fake_fit_cv(self, *args, **kwargs):
        observed.append(self.cv_splits)
        return self

    monkeypatch.setattr(lifecycle, "_ORIGINAL_COXPHCV_FIT_CV", fake_fit_cv)
    generator = _generator()
    model = CoxPHCV(
        penalties=[0.1],
        cv=3,
        cv_splits=generator,
        compute_inference=False,
        device="cpu",
    )

    model._fit_cv(None, None, None)
    model._fit_cv(None, None, None)

    assert model.cv_splits is generator
    assert model.get_params(deep=False)["cv_splits"] is observed[0]
    assert len(observed) == 2
    assert observed[0] is observed[1]
    assert isinstance(observed[0], list)
    assert len(observed[0]) == 3


def test_legacy_clone_parameter_round_trip_uses_reusable_snapshot():
    generator = _generator()
    model = CoxPHCV(
        penalties=[0.1],
        cv=3,
        cv_splits=generator,
        compute_inference=False,
        device="cpu",
    )

    params = model.get_params(deep=False)
    reconstructed = type(model)(**params)

    assert model.cv_splits is generator
    assert isinstance(params["cv_splits"], list)
    assert reconstructed.cv_splits is params["cv_splits"]


def test_sklearn_clone_materializes_one_shot_splits_once():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    generator = _generator()
    model = CoxPHCV(
        penalties=[0.1],
        cv=3,
        cv_splits=generator,
        compute_inference=False,
        device="cpu",
    )
    cloned = clone(model)

    assert model.cv_splits is generator
    assert isinstance(cloned.cv_splits, list)
    assert len(cloned.cv_splits) == 3
    assert cloned._fitted is False
    assert sklearn is not None


def test_pickle_serializes_one_shot_splits_as_reusable_sequence():
    generator = _generator()
    model = CoxPHCV(
        penalties=[0.1],
        cv=3,
        cv_splits=generator,
        compute_inference=False,
        device="cpu",
    )

    restored = pickle.loads(pickle.dumps(model))

    assert model.cv_splits is generator
    assert isinstance(restored.cv_splits, list)
    assert len(restored.cv_splits) == 3
    assert restored._cox_cv_split_source is None
    assert restored._cox_cv_split_snapshot is None


def test_set_params_invalidates_private_generator_snapshot(monkeypatch):
    observed = []

    def fake_fit_cv(self, *args, **kwargs):
        observed.append(self.cv_splits)
        return self

    monkeypatch.setattr(lifecycle, "_ORIGINAL_COXPHCV_FIT_CV", fake_fit_cv)
    first = _generator()
    second = (fold for fold in reversed(_folds()))
    model = CoxPHCV(
        penalties=[0.1],
        cv=3,
        cv_splits=first,
        compute_inference=False,
        device="cpu",
    )

    model._fit_cv(None, None, None)
    model.set_params(cv_splits=second)
    model._fit_cv(None, None, None)

    assert model.cv_splits is second
    assert observed[0] is not observed[1]
    np.testing.assert_array_equal(observed[1][0][1], np.array([4, 5]))


def test_public_fit_reuses_one_shot_splits_end_to_end():
    rng = np.random.default_rng(19001)
    X = rng.normal(size=(36, 2))
    beta = np.array([0.35, -0.2])
    time = 0.2 + rng.exponential(
        scale=np.exp(-0.2 * (X @ beta)), size=X.shape[0]
    )
    time += np.arange(X.shape[0], dtype=np.float64) * 1e-7
    event = (np.arange(X.shape[0]) % 3 != 0).astype(np.float64)
    folds = [
        (
            np.concatenate((np.arange(0, start), np.arange(stop, 36))),
            np.arange(start, stop),
        )
        for start, stop in ((0, 12), (12, 24), (24, 36))
    ]
    generator = (fold for fold in folds)
    model = CoxPHCV(
        penalties=[0.2],
        cv=3,
        cv_splits=generator,
        ties="efron",
        max_iter=200,
        tol=1e-8,
        compute_inference=False,
        device="cpu",
    )

    model.fit(X, time, event)
    first_coef = np.asarray(model.coef_, dtype=np.float64).copy()
    first_penalty = float(model.penalty_)
    model.fit(X, time, event)

    np.testing.assert_allclose(model.coef_, first_coef, rtol=0.0, atol=1e-10)
    assert model.penalty_ == pytest.approx(first_penalty)
    assert model.cv_splits is generator


def test_coxphcv_docstring_discloses_one_shot_split_lifecycle():
    documentation = inspect.getdoc(CoxPHCV)
    assert documentation is not None
    assert "Custom split lifecycle" in documentation
    assert "one-shot iterator" in documentation
    assert "repeated fits" in documentation
