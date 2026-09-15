"""Public/internal solver-name boundary for Issue #163."""

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGLM_CV, PenalizedQuantileRegression


@pytest.mark.parametrize("penalty", ["l2", "scad"])
def test_proximal_irls_cd_is_internal_resolved_label_not_public_solver_keyword(penalty):
    rng = np.random.default_rng(16311)
    X = rng.normal(size=(36, 2))
    y = 0.2 + X @ np.array([0.6, -0.25]) + rng.laplace(scale=0.15, size=36)

    model = PenalizedQuantileRegression(
        quantile=0.5,
        penalty=penalty,
        alpha=0.02,
        solver="proximal_irls_cd",
        device="cpu",
    )

    with pytest.raises(ValueError, match="internal resolved Quantile solver label"):
        model.fit(X, y)


@pytest.mark.parametrize("cv_strategy", ["strict", "two_stage"])
def test_cv_rejects_public_use_of_internal_proximal_irls_cd_label_before_grid_work(
    monkeypatch, cv_strategy
):
    rng = np.random.default_rng(16312)
    X = rng.normal(size=(36, 2))
    y = 0.2 + X @ np.array([0.6, -0.25]) + rng.laplace(scale=0.15, size=36)

    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.5},
        penalty="scad",
        cv=2,
        solver="proximal_irls_cd",
        cv_strategy=cv_strategy,
        acknowledge_approx=(cv_strategy == "two_stage"),
        device="cpu",
    )

    def forbidden_grid_work(*args, **kwargs):
        raise AssertionError("alpha-grid numerical work must not run")

    monkeypatch.setattr(model, "_generate_alpha_grid", forbidden_grid_work)
    with pytest.raises(ValueError, match="internal resolved Quantile solver label"):
        model.fit(X, y)
