from pathlib import Path


def replace_all(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"fixture anchor not found in {path}: {old!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


replace_all(
    "dev/tests/test_panel_stage_c_covariance.py",
    "low2 = np.nextafter(low1, np.inf)",
    "low2 = low1 * (1.0 + 1.0e-3)",
)
replace_all(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "low2 = np.nextafter(low1, np.inf)",
    "low2 = low1 * (1.0 + 1.0e-3)",
)
replace_all(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "unsafe_low2 = np.nextafter(unsafe_low1, np.inf)",
    "unsafe_low2 = unsafe_low1 * (1.0 + 1.0e-3)",
)
replace_all(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    "unsafe_low2 = np.nextafter(unsafe_low1, np.inf)",
    "unsafe_low2 = unsafe_low1 * (1.0 + 1.0e-3)",
)
Path("dev/validation/pr126_review_fix_unsafe_fixture_once.py").unlink(missing_ok=True)
