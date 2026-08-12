from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: replacement count={count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


runner = "dev/benchmarks/validate_panel_stage_c_gpu.py"
replace_once(
    runner,
    """def _snapshot(model):\n    fit = model.fit_statistics_\n""",
    """def _fit_rank(model):\n    \"\"\"Return the estimator fit-space numerical rank for audit payloads.\"\"\"\n    fit = model.fit_statistics_\n    metadata = getattr(fit, \"metadata\", {}) if fit is not None else {}\n    diagnostic_df = metadata.get(\"diagnostic_df\")\n    if isinstance(diagnostic_df, dict) and \"rank_x\" in diagnostic_df:\n        return int(diagnostic_df[\"rank_x\"])\n    if \"diagnostic_rank\" in metadata:\n        return int(metadata[\"diagnostic_rank\"])\n    if hasattr(model, \"rank_\"):\n        return int(model.rank_)\n    return int(len(np.asarray(model.coef_).ravel()))\n\n\ndef _snapshot(model):\n    fit = model.fit_statistics_\n""",
)
replace_once(
    runner,
    """                \"covariance_metadata\": snapshot[\"covariance_metadata\"],\n            }\n""",
    """                \"covariance_metadata\": snapshot[\"covariance_metadata\"],\n                \"fit_rank\": _fit_rank(model),\n                \"parameter_count\": int(snapshot[\"coef\"].size),\n            }\n""",
)

contract = "dev/tests/test_panel_stage_c_physical_runner_contract.py"
replace_once(
    contract,
    """        meta = model._covariance_metadata\n        assert meta.get(\"design_rank\", 0) < meta.get(\"design_columns\", 0), name\n        assert model.df_resid > 0, name\n""",
    """        fit_rank = _MOD._fit_rank(model)\n        assert fit_rank < len(np.asarray(model.coef_).ravel()), name\n        assert model.df_resid > 0, name\n""",
)
