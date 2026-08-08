from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cv_std_serialization_normalizes_one_ulp_runtime_drift(monkeypatch) -> None:
    """The deterministic frontend bundle must be stable across Python runtimes."""
    from dev.benchmarks.frontend_data.parsers import cv_package

    values = [1.0, 2.0]

    monkeypatch.setattr(
        cv_package.statistics,
        "pstdev",
        lambda _values: 0.03478300871847092,
    )
    upper = cv_package._stable_pstdev_ms(values)

    monkeypatch.setattr(
        cv_package.statistics,
        "pstdev",
        lambda _values: 0.03478300871847091,
    )
    lower = cv_package._stable_pstdev_ms(values)

    assert upper == lower == 0.034783008718


def test_cv_std_serialization_preserves_single_repeat_zero() -> None:
    from dev.benchmarks.frontend_data.parsers.cv_package import _stable_pstdev_ms

    assert _stable_pstdev_ms([12.5]) == 0.0


def test_pr116_sensitive_cv_row_has_stable_canonical_std() -> None:
    """Exercise the real row that previously drifted by one ULP on 3.9/3.11."""
    from dev.benchmarks.frontend_data.parsers.cv_package import parse_cv_benchmark

    source = REPO_ROOT / "results" / "pr116_p100" / "cv_benchmark_pr116_p100.json"
    runs, _models, warnings = parse_cv_benchmark(
        source, "remote-p100-pr116-20260807"
    )

    assert warnings == []
    ridge_numpy = next(
        run
        for run in runs
        if run["model_id"] == "RidgeCV"
        and run["framework"] == "statgpu"
        and run["backend"] == "numpy"
    )
    assert ridge_numpy["metrics"]["timing"]["std_ms"] == 0.083881039848
