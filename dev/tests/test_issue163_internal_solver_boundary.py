"""Public/internal solver-name boundary for Issue #163."""

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedQuantileRegression


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
