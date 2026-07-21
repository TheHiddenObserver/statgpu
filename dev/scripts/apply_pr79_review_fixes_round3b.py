#!/usr/bin/env python3
"""Run round-three PR79 fixes with explicit GPU/Torch block disambiguation."""

from pathlib import Path
import runpy


namespace = runpy.run_path(
    "dev/scripts/apply_pr79_review_fixes_round3.py",
    run_name="pr79_round3_module",
)
original_replace_once = namespace["replace_once"]


def replace_once(path, old, new):
    # Before either backend block is patched, this residual-storage fragment is
    # intentionally present once in the CuPy path and once in the Torch path.
    if old.startswith("        if resid_np.shape[1] == 1:\n"):
        p = Path(path)
        text = p.read_text()
        count = text.count(old)
        if count != 2:
            raise RuntimeError(
                f"{path}: expected GPU+Torch residual blocks, found {count}"
            )
        p.write_text(text.replace(old, new, 1))
        return
    original_replace_once(path, old, new)


namespace["replace_once"] = replace_once
namespace["main"]()
