"""Regression coverage for optimized Fama-MacBeth backend period solves."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from statgpu.panel import FamaMacBeth
from statgpu.panel._linalg import panel_lstsq, panel_lstsq_batched


def _balanced_fixture(seed=12620, *, n_times=8, per_period=32, p=3):
    rng = np.random.default_rng(seed)
    time = np.repeat(np.arange(n_times), per_period)
    X = rng.normal(size=(time.size, p))
    beta = np.linspace(0.25, 0.85, p)
    shift = np.repeat(rng.normal(scale=0.4, size=n_times), per_period)
    y = 0.3 + X @ beta + shift + rng.normal(scale=0.2, size=time.size)
    return X, y, time


def test_panel_lstsq_batched_matches_serial_numpy_rank_policy():
    rng = np.random.default_rng(12621)
    X = rng.normal(size=(5, 24, 4))
    y = rng.normal(size=(5, 24))

    params, ranks = panel_lstsq_batched(X, y, np)
    assert params.shape == (5, 4)
    assert ranks.shape == (5,)

    for i in range(5):
        expected_params, expected_rank = panel_lstsq(X[i], y[i], np)
        np.testing.assert_allclose(params[i], expected_params, rtol=2e-13, atol=2e-14)
        assert int(ranks[i]) == expected_rank


def test_panel_lstsq_batched_preserves_serial_rank_cutoff_boundary():
    rng = np.random.default_rng(126210)
    n, k = 40, 3
    q_left, _ = np.linalg.qr(rng.normal(size=(n, k)))
    q_right, _ = np.linalg.qr(rng.normal(size=(k, k)))
    cutoff = max(n, k) * np.finfo(np.float64).eps * 10.0
    singular_sets = (
        np.asarray([10.0, 1.0, 0.5 * cutoff]),
        np.asarray([10.0, 1.0, 2.0 * cutoff]),
    )
    X = np.stack(
        [q_left @ np.diag(values) @ q_right.T for values in singular_sets],
        axis=0,
    )
    y = rng.normal(size=(2, n))

    _params, batched_ranks = panel_lstsq_batched(X, y, np)
    serial_ranks = [panel_lstsq(X[i], y[i], np)[1] for i in range(2)]

    assert serial_ranks == [2, 3]
    assert np.asarray(batched_ranks, dtype=np.int64).tolist() == serial_ranks


def test_panel_lstsq_batched_does_not_assume_undocumented_cupy_stacked_svd():
    class FakeCuPy:
        __name__ = "cupy"

    with pytest.raises(NotImplementedError, match="documented stacked-SVD"):
        panel_lstsq_batched(
            np.zeros((2, 8, 3), dtype=np.float64),
            np.zeros((2, 8), dtype=np.float64),
            FakeCuPy(),
        )


def test_fama_macbeth_torch_cpu_batches_balanced_periods_and_matches_numpy():
    torch = pytest.importorskip("torch")
    X, y, time = _balanced_fixture()

    expected = FamaMacBeth(device="cpu", bandwidth=2).fit(X, y, time_ids=time)
    actual = FamaMacBeth(bandwidth=2).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=time,
    )

    assert actual._backend_name == "torch"
    assert actual._inference_backend_name == "torch"
    assert actual._period_solver_mode == "batched"
    assert actual._period_solver_batches == 1
    assert actual._period_rank_syncs == 1
    assert expected._period_solver_mode == "serial"
    assert expected._period_solver_batches == expected.n_periods
    assert expected._period_rank_syncs == expected.n_periods
    for name in ("coef_", "betas_", "bse_", "tvalues_", "pvalues_", "conf_int_"):
        value = getattr(actual, name).detach().cpu().numpy()
        np.testing.assert_allclose(
            value,
            np.asarray(getattr(expected, name)),
            rtol=2e-10,
            atol=2e-12,
        )


def test_fama_macbeth_torch_cpu_buckets_unbalanced_shuffled_periods():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12622)
    counts = np.asarray([24, 32, 24, 40], dtype=np.int64)
    time = np.repeat(np.arange(len(counts)), counts)
    X = rng.normal(size=(int(counts.sum()), 3))
    beta = np.asarray([0.5, -0.3, 0.8])
    shift = np.repeat(np.asarray([0.1, -0.4, 0.7, -0.2]), counts)
    y = 0.2 + X @ beta + shift + rng.normal(scale=0.15, size=time.size)

    permutation = rng.permutation(time.size)
    X = X[permutation]
    y = y[permutation]
    time = time[permutation]

    expected = FamaMacBeth(device="cpu", bandwidth=1).fit(X, y, time_ids=time)
    actual = FamaMacBeth(bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=time,
    )

    assert actual._period_solver_mode == "batched"
    assert actual._period_solver_batches == 3
    assert actual._period_rank_syncs == 3
    np.testing.assert_allclose(
        actual.betas_.detach().cpu().numpy(),
        np.asarray(expected.betas_),
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        actual.coef_.detach().cpu().numpy(),
        np.asarray(expected.coef_),
        rtol=2e-10,
        atol=2e-12,
    )


def test_batched_rank_rejection_reports_earliest_chronological_period():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12623)
    counts = np.asarray([24, 32, 24], dtype=np.int64)
    time = np.repeat(np.arange(3), counts)
    X = rng.normal(size=(int(counts.sum()), 2))
    y = rng.normal(size=time.size)

    starts = np.concatenate([[0], np.cumsum(counts)])
    sl1 = slice(int(starts[1]), int(starts[2]))
    X[sl1, 1] = X[sl1, 0]
    sl2 = slice(int(starts[2]), int(starts[3]))
    X[sl2, :] = 0.0

    with pytest.raises(ValueError, match=r"retained time period 1.*rank deficient"):
        FamaMacBeth().fit(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64),
            time_ids=time,
        )


def test_balanced_torch_reporting_uses_one_rank_snapshot_and_one_reporting_snapshot(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.panel._fama_macbeth as fmb_module

    X, y, time = _balanced_fixture(seed=12624, n_times=6, per_period=28, p=2)
    original = fmb_module._to_numpy
    shapes = []

    def tracked(value):
        shapes.append(tuple(getattr(value, "shape", ())))
        return original(value)

    monkeypatch.setattr(fmb_module, "_to_numpy", tracked)
    model = FamaMacBeth(bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=time,
    )

    assert model._period_solver_batches == 1
    assert model._period_rank_syncs == 1
    assert shapes == [(6,), (6, 3)]


def _assert_runner_help(filename):
    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "dev" / "benchmarks" / filename
    completed = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Fama-MacBeth" in completed.stdout
    assert "--expected-sha" in completed.stdout


def test_fama_macbeth_scaling_runner_is_directly_executable():
    _assert_runner_help("benchmark_fama_macbeth_scaling_gpu.py")


def test_fama_macbeth_optimized_wrapper_is_directly_executable():
    _assert_runner_help("validate_fama_macbeth_optimized_gpu.py")


def test_optimized_wrapper_rewrites_backend_specific_solver_notes(monkeypatch):
    from dev.benchmarks import validate_fama_macbeth_optimized_gpu as optimized

    assert optimized.SCHEMA_VERSION == 4

    def fake_provenance(backend):
        return optimized._expected_provenance(backend)

    monkeypatch.setattr(optimized, "_solver_provenance", fake_provenance)
    payload = {
        "backends": {
            "cupy": {"performance": {"optimization_notes": {"remaining_structure": "serial"}}},
            "torch": {"performance": {"optimization_notes": {"remaining_structure": "serial"}}},
        }
    }
    optimized._rewrite_performance(payload, ["cupy", "torch"])

    cupy_perf = payload["backends"]["cupy"]["performance"]
    assert cupy_perf["solver_provenance"]["cupy"] == optimized._expected_provenance("cupy")
    assert "two-dimensional" in cupy_perf["optimization_notes"]["period_solver"]
    assert "one host" in cupy_perf["optimization_notes"]["period_solver"]

    torch_perf = payload["backends"]["torch"]["performance"]
    assert torch_perf["solver_provenance"]["torch"] == optimized._expected_provenance("torch")
    assert "batched" in torch_perf["optimization_notes"]["period_solver"]
    assert "one solver batch" in torch_perf["optimization_notes"]["period_solver"]

    for backend in ("cupy", "torch"):
        notes = payload["backends"][backend]["performance"]["optimization_notes"]
        assert "scaling runner" in notes["remaining_structure"]
        assert "NumPy/SciPy" in notes["distribution_inference"]
