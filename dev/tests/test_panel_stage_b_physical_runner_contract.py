"""Hosted smoke checks for the PR122 physical GPU acceptance runner."""

from __future__ import annotations

from dev.benchmarks.validate_panel_stage_b_gpu import (
    _dataset,
    _fit_cases,
    _model_snapshot,
)


def test_physical_runner_covers_explicit_constant_re_balanced_and_unbalanced():
    expected_counts = {"balanced": 9, "unbalanced": 8}

    for name, unbalanced in (("balanced", False), ("unbalanced", True)):
        X, y, entity, time = _dataset(20260808 + int(unbalanced), unbalanced=unbalanced)
        models, diagnostics = _fit_cases(
            X,
            y,
            entity,
            time,
            "numpy",
            unbalanced=unbalanced,
        )

        assert len(models) == expected_counts[name]
        assert set(diagnostics) == {f"hausman_{name}"}

        case = f"random_effects_explicit_constant_{name}"
        assert case in models
        contract = _model_snapshot(models[case])["random_effects_diagnostic_contract"]
        assert contract == {
            "has_explicit_constant": True,
            "constant_column_index": 0,
            "restricted_rank": 1,
            "model_f_rank_restricted": 1,
            "model_f_restricted_design_supplied": True,
        }


def test_physical_runner_total_model_case_contract_is_seventeen():
    balanced = _dataset(20260808, unbalanced=False)
    unbalanced = _dataset(20260809, unbalanced=True)
    balanced_models, _ = _fit_cases(*balanced, "numpy", unbalanced=False)
    unbalanced_models, _ = _fit_cases(*unbalanced, "numpy", unbalanced=True)

    case_ids = set(balanced_models) | set(unbalanced_models)
    assert len(case_ids) == 17
    assert "random_effects_explicit_constant_balanced" in case_ids
    assert "random_effects_explicit_constant_unbalanced" in case_ids
