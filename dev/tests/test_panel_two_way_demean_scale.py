from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel._utils import demean_variables


def _explicit_two_way_residual(values, entity, time):
    entity_levels = np.unique(entity)
    time_levels = np.unique(time)
    cols = [np.ones(len(entity), dtype=np.float64)]
    cols.extend((entity == level).astype(np.float64) for level in entity_levels[1:])
    cols.extend((time == level).astype(np.float64) for level in time_levels[1:])
    design = np.column_stack(cols)
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coef


def test_two_way_demeaning_convergence_ignores_removed_entity_level_scale():
    entity = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    time = np.array([0, 1, 3, 0, 2, 1, 2, 3, 0, 2, 3], dtype=np.int64)
    rng = np.random.default_rng(20260812)

    within_signal = rng.normal(size=len(entity))
    entity_offsets = np.array([1.0e12, -2.0e12, 3.0e12, -4.0e12])
    y = entity_offsets[entity] + within_signal

    raw_X = rng.normal(size=(len(entity), 2))
    X = np.column_stack(
        [
            _explicit_two_way_residual(raw_X[:, j], entity, time)
            for j in range(raw_X.shape[1])
        ]
    )

    # With a level-data scale reference the first alternating-projection update
    # is only O(1) relative to O(1e12) entity offsets and can be accepted even
    # though entity means have been reintroduced by the time projection.
    with pytest.raises(RuntimeError, match="did not converge"):
        demean_variables(
            y,
            X,
            entity,
            time,
            xp=np,
            max_iter=1,
            tol=1e-10,
        )

    y_d, X_d = demean_variables(
        y,
        X,
        entity,
        time,
        xp=np,
        max_iter=200,
        tol=1e-12,
    )
    y_reference, X_reference = demean_variables(
        within_signal,
        X,
        entity,
        time,
        xp=np,
        max_iter=200,
        tol=1e-12,
    )

    for level in np.unique(entity):
        assert abs(float(np.mean(y_d[entity == level]))) < 1e-9
    for level in np.unique(time):
        assert abs(float(np.mean(y_d[time == level]))) < 1e-9

    # Adding exactly removable entity effects changes the residual only through
    # float64 cancellation in the O(1e12) level values, not through premature
    # alternating-projection termination.
    assert_allclose(y_d, y_reference, rtol=0, atol=2e-4)
    assert_allclose(X_d, X_reference, rtol=0, atol=2e-12)
