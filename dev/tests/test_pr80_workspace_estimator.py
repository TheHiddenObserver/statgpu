"""Regression tests for the PR #80 delayed-entry workspace fallback."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.survival import _risk_sets as risk_sets


def _counting_process_sample(seed=2301):
    rng = np.random.default_rng(seed)
    n, p = 48, 4
    X = rng.normal(size=(n, p))
    beta = np.array([0.25, -0.15, 0.1, -0.05])
    strata = np.repeat(np.array([0, 1], dtype=np.int64), n // 2)
    stop = np.tile(np.repeat(np.arange(2.0, 8.0), 4), 2)
    start = rng.uniform(0.0, np.maximum(stop - 0.5, 0.1))
    event = np.zeros(n, dtype=np.float64)
    for stratum in (0, 1):
        rows = np.flatnonzero(strata == stratum)
        for failure_time in (2.0, 4.0, 6.0):
            candidates = rows[stop[rows] == failure_time]
            event[candidates[:2]] = 1.0
    return beta, X, stop, event, start, strata


def test_dense_workspace_estimate_covers_weighted_design_intermediate():
    n_rows = 1_000_000
    n_features = 100
    itemsize = 8
    estimate = risk_sets._estimate_dense_group_workspace_bytes(
        n_rows,
        n_features,
        itemsize,
        compute_derivatives=True,
        score_residuals=False,
    )
    scalar_rows = n_rows * (2 + 6 * itemsize)
    weighted_design_rows = n_rows * (2 * n_features * itemsize)
    p_squared_outputs = 6 * n_features * n_features * itemsize
    assert estimate >= scalar_rows + weighted_design_rows + p_squared_outputs
    assert estimate > 512 * 1024 * 1024


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_forced_streaming_matches_numpy_for_multiple_groups_and_strata(
    ties, monkeypatch
):
    torch = pytest.importorskip("torch")
    beta, X, stop, event, start, strata = _counting_process_sample()
    reference = risk_sets.cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
        score_residuals=True,
    )

    calls = []
    original = risk_sets._streamed_stratum_group_objective

    def recording_streamed(*args, **kwargs):
        calls.append((int(args[1].shape[0]), int(args[1].shape[1])))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        risk_sets, "_streamed_stratum_group_objective", recording_streamed
    )
    monkeypatch.setenv("STATGPU_COX_GROUP_MAX_BYTES", "256")
    result = risk_sets.cox_counting_process_objective(
        torch.as_tensor(beta, dtype=torch.float64),
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(stop, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.float64),
        start=torch.as_tensor(start, dtype=torch.float64),
        strata=torch.as_tensor(strata, dtype=torch.int64),
        ties=ties,
        score_residuals=True,
    )

    assert calls == [(24, 4), (24, 4)]
    for key in ("log_likelihood", "score", "information", "score_residuals"):
        actual = result[key].detach().cpu().numpy()
        assert_allclose(actual, np.asarray(reference[key]), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_forced_streaming_loglik_only_matches_numpy(ties, monkeypatch):
    torch = pytest.importorskip("torch")
    beta, X, stop, event, start, strata = _counting_process_sample(seed=2302)
    reference = risk_sets.cox_counting_process_objective(
        beta,
        X,
        stop,
        event,
        start=start,
        strata=strata,
        ties=ties,
        compute_derivatives=False,
    )
    monkeypatch.setenv("STATGPU_COX_GROUP_MAX_BYTES", "128")
    result = risk_sets.cox_counting_process_objective(
        torch.as_tensor(beta, dtype=torch.float64),
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(stop, dtype=torch.float64),
        torch.as_tensor(event, dtype=torch.float64),
        start=torch.as_tensor(start, dtype=torch.float64),
        strata=torch.as_tensor(strata, dtype=torch.int64),
        ties=ties,
        compute_derivatives=False,
    )
    assert set(result) == {"log_likelihood"}
    assert_allclose(
        result["log_likelihood"].detach().cpu().numpy(),
        np.asarray(reference["log_likelihood"]),
        rtol=2e-12,
        atol=2e-12,
    )
