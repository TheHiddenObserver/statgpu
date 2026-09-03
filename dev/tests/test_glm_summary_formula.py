"""Regression tests for GLM Formula names in formatted summaries."""

from __future__ import annotations

import numpy as np
import pytest


def test_poisson_formula_summary_uses_design_term_names():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    from statgpu.linear_model import PoissonRegression

    rng = np.random.default_rng(17)
    x1 = rng.normal(size=120)
    x2 = rng.normal(size=120)
    mean = np.exp(0.2 + 0.35 * x1 - 0.15 * x2)
    frame = pd.DataFrame(
        {
            "count": rng.poisson(mean),
            "x1": x1,
            "x2": x2,
        }
    )

    model = PoissonRegression(
        solver="newton",
        device="cpu",
        compute_inference=True,
        cov_type="hc1",
    ).fit(formula="count ~ x1 + x2", data=frame)

    assert model._inference_result.feature_names == ["Intercept", "x1", "x2"]
    summary = model.summary()
    assert "Intercept" in summary
    assert "x1" in summary
    assert "x2" in summary
    assert "Solver: newton" in summary
    assert "Covariance Type: hc1" in summary


def test_glm_summary_reports_resolved_auto_solver():
    from statgpu.linear_model import PoissonRegression

    X = np.arange(40, dtype=float).reshape(-1, 1) / 40.0
    y = np.asarray(([0, 1, 1, 2, 1] * 8), dtype=float)
    model = PoissonRegression(
        solver="auto",
        C=0,
        device="cpu",
    ).fit(X, y)

    assert "Solver: irls" in model.summary()
