"""Weighted direct-fit and CV contracts for Group MCP/SCAD LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


_INTERLEAVED = [[0, 3], [1, 2]]
_GROUPED = [[0, 1], [2, 3]]
_PERM = np.array([0, 3, 1, 2], dtype=np.int64)
_INVERSE_PERM = np.argsort(_PERM)


def _data():
    rng = np.random.default_rng(9901)
    X = rng.normal(size=(96, 4))
    y = 0.25 + X @ np.array([0.8, -0.5, 0.3, 0.65])
    y += rng.normal(scale=0.08, size=X.shape[0])
    sample_weight = np.linspace(0.4, 1.8, X.shape[0])
    return X, y, sample_weight


def _penalty_kwargs(kind, groups):
    kwargs = {"groups": groups}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    else:
        kwargs["a"] = 3.7
    return kwargs


def _model(kind, groups):
    return PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_penalty_kwargs(kind, groups),
        alpha=0.16,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=500,
        tol=1e-8,
        max_lla_iters=20,
        lla_tol=1e-8,
    )


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_weighted_direct_fit_is_invariant_to_grouped_column_permutation(kind):
    X, y, sample_weight = _data()
    interleaved = _model(kind, _INTERLEAVED).fit(
        X, y, sample_weight=sample_weight
    )
    grouped = _model(kind, _GROUPED).fit(
        X[:, _PERM], y, sample_weight=sample_weight
    )

    np.testing.assert_allclose(
        interleaved.coef_,
        np.asarray(grouped.coef_)[_INVERSE_PERM],
        rtol=3e-6,
        atol=3e-7,
    )
    assert interleaved.intercept_ == pytest.approx(
        grouped.intercept_, rel=3e-6, abs=3e-7
    )
    np.testing.assert_allclose(
        interleaved.predict(X),
        grouped.predict(X[:, _PERM]),
        rtol=3e-6,
        atol=3e-7,
    )


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_weighted_cv_scores_selection_and_refit_are_layout_invariant(kind):
    X, y, sample_weight = _data()
    common = dict(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        alpha_grid=[0.2, 0.1],
        cv=2,
        random_state=23,
        device="cpu",
        max_iter=400,
        tol=1e-7,
    )
    interleaved = PenalizedGLM_CV(
        penalty_kwargs=_penalty_kwargs(kind, _INTERLEAVED),
        **common,
    ).fit(X, y, sample_weight=sample_weight)
    grouped = PenalizedGLM_CV(
        penalty_kwargs=_penalty_kwargs(kind, _GROUPED),
        **common,
    ).fit(X[:, _PERM], y, sample_weight=sample_weight)

    np.testing.assert_allclose(
        interleaved.cv_results_["all_scores"],
        grouped.cv_results_["all_scores"],
        rtol=3e-5,
        atol=3e-7,
    )
    assert interleaved.alpha_ == pytest.approx(grouped.alpha_)
    assert interleaved.estimator_.alpha == pytest.approx(interleaved.alpha_)
    np.testing.assert_allclose(
        interleaved.coef_,
        np.asarray(grouped.coef_)[_INVERSE_PERM],
        rtol=3e-5,
        atol=3e-6,
    )
