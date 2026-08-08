from __future__ import annotations


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
