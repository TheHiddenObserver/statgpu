"""Hosted smoke checks for the PR122 physical GPU acceptance runner."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import PanelOLS, RandomEffects

from dev.benchmarks.validate_panel_stage_b_gpu import (
    _dataset,
    _fit_cases,
    _hausman_applicable_dataset,
    _model_snapshot,
    _require_applicable_hausman_coverage,
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
        assert set(diagnostics) == {
            f"hausman_{name}",
            f"hausman_explicit_re_constant_{name}",
        }

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

        absorbed = diagnostics[f"hausman_explicit_re_constant_{name}"]
        assert "identity mismatch" not in (absorbed["reason"] or "")
        assert "no common estimable slope" not in (absorbed["reason"] or "")


def test_physical_runner_total_model_case_contract_is_seventeen():
    balanced = _dataset(20260808, unbalanced=False)
    unbalanced = _dataset(20260809, unbalanced=True)
    balanced_models, balanced_diagnostics = _fit_cases(
        *balanced, "numpy", unbalanced=False
    )
    unbalanced_models, unbalanced_diagnostics = _fit_cases(
        *unbalanced, "numpy", unbalanced=True
    )

    case_ids = set(balanced_models) | set(unbalanced_models)
    diagnostic_ids = set(balanced_diagnostics) | set(unbalanced_diagnostics)
    assert len(case_ids) == 17
    assert len(diagnostic_ids) == 4
    assert "random_effects_explicit_constant_balanced" in case_ids
    assert "random_effects_explicit_constant_unbalanced" in case_ids
    assert "hausman_explicit_re_constant_balanced" in diagnostic_ids
    assert "hausman_explicit_re_constant_unbalanced" in diagnostic_ids



def test_physical_runner_has_stable_nonzero_effect_applicable_hausman_fixture():
    X, y, entity, _time, metadata = _hausman_applicable_dataset()
    fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )
    re = RandomEffects().fit(X, y, entity_ids=entity)
    result = fe.hausman_test(re)
    variance_difference = float(
        np.asarray(fe._panel_cov_params)[0, 0]
        - np.asarray(re._panel_cov_params)[0, 0]
    )

    assert metadata == {
        "seed": 20260810,
        "n_entities": 12,
        "n_times": 4,
        "entity_effect_scale": 0.005,
        "noise_scale": 0.1,
    }
    assert metadata["entity_effect_scale"] > 0.0
    assert X.shape == (48, 1)
    assert result.applicable is True
    assert result.reason is None
    assert result.df == 1.0
    assert np.isfinite(float(result.statistic))
    assert np.isfinite(float(result.pvalue))
    assert variance_difference > 1e-6


def test_physical_runner_requires_recorded_applicable_hausman_per_backend():
    with pytest.raises(AssertionError, match="successful applicable"):
        _require_applicable_hausman_coverage(
            {
                "hausman_balanced": {
                    "status": "success",
                    "applicable": False,
                },
                "hausman_unbalanced": {
                    "status": "success",
                    "applicable": False,
                },
            },
            backend="torch",
        )

    with pytest.raises(AssertionError, match="successful applicable"):
        _require_applicable_hausman_coverage(
            {
                "hausman_missing_values": {
                    "status": "success",
                    "applicable": True,
                }
            },
            backend="cupy",
        )

    applicable = _require_applicable_hausman_coverage(
        {
            "hausman_balanced": {
                "status": "success",
                "applicable": False,
            },
            "hausman_applicable_nonzero_effect": {
                "status": "success",
                "applicable": True,
                "statistic": 1.2,
                "pvalue": 0.27,
                "df": 1.0,
            },
        },
        backend="cupy",
    )
    assert applicable == ["hausman_applicable_nonzero_effect"]
