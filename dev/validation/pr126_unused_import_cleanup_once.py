"""One-shot cleanup for imports made obsolete by the stable SVD solve."""

from pathlib import Path


for path in (
    "statgpu/panel/_fixed_effects.py",
    "statgpu/panel/_between.py",
    "statgpu/panel/_first_diff.py",
):
    p = Path(path)
    text = p.read_text()
    old = "from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank\n"
    new = "from statgpu.panel._linalg import panel_lstsq\n"
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one obsolete import, found {count}")
    p.write_text(text.replace(old, new, 1))
