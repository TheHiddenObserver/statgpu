"""Fit-time group completion must not mutate constructor penalty objects."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import GroupLassoPenalty


def _data(seed, p):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(72, p))
    beta = np.linspace(0.75, -0.25, p)
    y = 0.3 + X @ beta + rng.normal(scale=0.06, size=X.shape[0])
    return X, y


def test_direct_fit_completes_only_internal_penalty_clone():
    penalty = GroupLassoPenalty(alpha=0.08, groups=[[0, 1]])
    X3, y3 = _data(10701, 3)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty,
        alpha=0.08,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=2000,
        tol=1e-8,
    )

    with pytest.warns(UserWarning, match="Auto-adding 1 single-feature"):
        model.fit(X3, y3)

    assert model.penalty is penalty
    assert penalty.groups == ((0, 1),)
    assert model._penalty is not penalty
    assert model._penalty.groups == ((0, 1), (2,))

    # Reusing the same constructor object on its original two-feature design
    # remains valid because the wider fit did not append a persistent group.
    X2, y2 = _data(10702, 2)
    second = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty,
        alpha=0.08,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=2000,
        tol=1e-8,
    ).fit(X2, y2)
    assert second._penalty.groups == ((0, 1),)
    assert penalty.groups == ((0, 1),)


def test_cv_uses_temporary_penalty_clone_and_restores_parameter_identity():
    penalty = GroupLassoPenalty(alpha=1.0, groups=[[0, 1]])
    X, y = _data(10703, 3)
    cv = PenalizedGLM_CV(
        loss="squared_error",
        penalty=penalty,
        alpha_grid=[0.16, 0.08],
        cv=2,
        random_state=37,
        device="cpu",
        max_iter=1200,
        tol=1e-8,
    )

    with pytest.warns(UserWarning, match="Auto-adding 1 single-feature"):
        cv.fit(X, y)

    assert cv.penalty is penalty
    assert penalty.groups == ((0, 1),)
    assert cv.estimator_ is not None
    assert cv.estimator_._penalty.groups == ((0, 1), (2,))


def test_cv_restores_external_penalty_object_after_failed_prevalidation():
    penalty = GroupLassoPenalty(alpha=1.0, groups=[[0, 4], [1, 2, 3]])
    X, y = _data(10704, 4)
    cv = PenalizedGLM_CV(
        loss="squared_error",
        penalty=penalty,
        alpha_grid=[0.16, 0.08],
        cv=2,
        random_state=41,
        device="cpu",
    )

    with pytest.raises(ValueError, match="outside the design matrix"):
        cv.fit(X, y)

    assert cv.penalty is penalty
    assert penalty.groups == ((0, 4), (1, 2, 3))
    assert cv.estimator_ is None
    assert cv.coef_ is None
