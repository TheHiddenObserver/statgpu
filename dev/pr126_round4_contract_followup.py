from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# The rank-boundary estimator test is intentionally rank deficient.  Verify the
# identified fit-space covariance and explicit inference-unavailable contract
# instead of calling the full-rank bse/t/p/CI assertion helper.
replace_once(
    "dev/tests/test_panel_stage_c_torch_cpu.py",
    '''    assert_allclose(expected.coef_, expected_coef, rtol=5e-11, atol=5e-13)\n    _assert_inference(actual, expected, rtol=5e-9, atol=5e-11)\n    assert expected._covariance_metadata["design_rank"] == 2\n''',
    '''    assert_allclose(expected.coef_, expected_coef, rtol=5e-11, atol=5e-13)\n    assert_allclose(actual.coef_, expected.coef_, rtol=5e-9, atol=5e-11)\n    assert_allclose(\n        actual._panel_cov_params_raw,\n        expected._panel_cov_params_raw,\n        rtol=5e-9,\n        atol=5e-11,\n    )\n    for model in (expected, actual):\n        assert model._coefficient_inference_available is False\n        assert model.bse_ is None\n        assert model.tvalues_ is None\n        assert model.pvalues_ is None\n        assert model.conf_int_ is None\n        assert model._inference_result.metadata["applicable"] is False\n    assert expected._covariance_metadata["design_rank"] == 2\n''',
)

# The physical-runner hosted contract must accept two kinds of successful
# estimator case: full-rank cases with finite parameter inference and rank-
# deficient cases with finite fit-space covariance but explicitly unavailable
# coordinate-wise inference.
replace_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    '''    for name, model in cases.items():\n        snap = _MOD._snapshot(model)\n        assert np.all(np.isfinite(snap["coef"])), name\n        assert np.all(np.isfinite(snap["bse"])), name\n        assert np.all(np.isfinite(snap["covariance"])), name\n''',
    '''    for name, model in cases.items():\n        snap = _MOD._snapshot(model)\n        assert np.all(np.isfinite(snap["coef"])), name\n        assert np.all(np.isfinite(snap["covariance"])), name\n        if snap["coefficient_inference_applicable"]:\n            assert np.all(np.isfinite(snap["bse"])), name\n            assert np.all(np.isfinite(snap["tvalues"])), name\n            assert np.all(np.isfinite(snap["pvalues"])), name\n            assert np.all(np.isfinite(snap["conf_int"])), name\n            assert snap["coefficient_inference_reason"] is None\n        else:\n            assert snap["bse"] is None, name\n            assert snap["tvalues"] is None, name\n            assert snap["pvalues"] is None, name\n            assert snap["conf_int"] is None, name\n            assert "rank deficient" in snap["coefficient_inference_reason"], name\n''',
)
replace_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    '''        assert fit_rank < len(np.asarray(model.coef_).ravel()), name\n        assert model.df_resid > 0, name\n        assert np.all(np.isfinite(model.bse_)), name\n''',
    '''        assert fit_rank < len(np.asarray(model.coef_).ravel()), name\n        assert model.df_resid > 0, name\n        assert np.all(np.isfinite(model._panel_cov_params_raw)), name\n        assert model._coefficient_inference_available is False, name\n        assert model.bse_ is None, name\n        assert model.tvalues_ is None, name\n        assert model.pvalues_ is None, name\n        assert model.conf_int_ is None, name\n        assert model._inference_result.metadata["applicable"] is False, name\n        assert "rank deficient" in model._inference_result.metadata["reason"], name\n''',
)

# Avoid a Python invalid-escape warning in the newly added regression test.
replace_once(
    "dev/tests/test_panel_stage_c_review_round4.py",
    'with pytest.raises(ValueError, match="unique \\(entity_id, time_id\\)"):',
    'with pytest.raises(ValueError, match=r"unique \\(entity_id, time_id\\)"):',
)

print("PR126 round4 maintained-contract followup applied")
