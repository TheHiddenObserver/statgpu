from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '    np.testing.assert_allclose(regular, np.asarray([2.0, -3.0]), rtol=0.0, atol=0.0)\n',
    '    np.testing.assert_allclose(regular, np.asarray([2.0, -3.0]), rtol=2.0e-15, atol=0.0)\n',
    "physical positive-bse rounding tolerance",
)
replace_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    '''    np.testing.assert_array_equal(\n        audit["tiny_positive_bse_statistics"], np.asarray([2.0, -3.0])\n    )\n''',
    '''    np.testing.assert_allclose(\n        audit["tiny_positive_bse_statistics"],\n        np.asarray([2.0, -3.0]),\n        rtol=2.0e-15,\n        atol=0.0,\n    )\n''',
    "hosted positive-bse rounding tolerance",
)
