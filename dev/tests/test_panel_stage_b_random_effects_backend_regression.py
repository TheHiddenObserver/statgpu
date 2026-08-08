"""Regression coverage for backend-stable explicit-constant RandomEffects."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dev.benchmarks.validate_panel_stage_b_gpu import _dataset
from statgpu.panel import RandomEffects


@pytest.mark.parametrize("unbalanced", [False, True])
def test_explicit_constant_re_does_not_depend_on_singular_solve_exception(
    monkeypatch,
    unbalanced,
):
    """The within auxiliary fit must never solve a structural singular system.

    An explicit level constant is annihilated by entity demeaning. NumPy raises
    on the resulting singular normal equations, but some GPU linalg stacks may
    return a value or warning instead. Simulate that non-raising behavior and
    require RandomEffects to avoid the singular solve entirely.
    """
    X, y, entity, _ = _dataset(
        20260808 + int(unbalanced),
        unbalanced=unbalanced,
    )
    X_constant = np.column_stack([np.ones(X.shape[0]), X])

    expected = RandomEffects(device="cpu").fit(
        X_constant,
        y,
        entity_ids=entity,
    )

    original_solve = np.linalg.solve
    singular_calls = []

    def nonraising_solve(A, b):
        A = np.asarray(A)
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            rank = np.linalg.matrix_rank(A)
            if rank < A.shape[0]:
                singular_calls.append((A.shape, int(rank)))
                # Emulate a backend that does not raise on a singular solve.
                return np.zeros_like(np.asarray(b), dtype=np.float64)
        return original_solve(A, b)

    monkeypatch.setattr(np.linalg, "solve", nonraising_solve)
    actual = RandomEffects(device="cpu").fit(
        X_constant,
        y,
        entity_ids=entity,
    )

    assert singular_calls == []
    assert_allclose(actual.coef_, expected.coef_, rtol=1e-11, atol=1e-12)
    assert_allclose(actual.bse_, expected.bse_, rtol=1e-11, atol=1e-12)
    assert_allclose(
        actual.variance_components_["sigma2_e"],
        expected.variance_components_["sigma2_e"],
        rtol=1e-12,
        atol=1e-14,
    )
    assert_allclose(
        actual.variance_components_["sigma2_a"],
        expected.variance_components_["sigma2_a"],
        rtol=1e-12,
        atol=1e-14,
    )
    assert_allclose(actual.theta_, expected.theta_, rtol=1e-12, atol=1e-14)
