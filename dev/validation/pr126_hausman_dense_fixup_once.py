from pathlib import Path


for path in (
    "statgpu/panel/_diagnostics.py",
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        '"scaled_standardized_eigencoordinates"',
        '"standardized_eigencoordinates"',
    )
    p.write_text(text.rstrip() + "\n", encoding="utf-8")
