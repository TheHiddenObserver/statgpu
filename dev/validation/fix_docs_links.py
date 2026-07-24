#!/usr/bin/env python3
"""Fix or check deterministic bilingual documentation link patterns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def apply_replacements(path: Path, replacements: tuple[tuple[str, str], ...]) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    for old, new in replacements:
        updated = updated.replace(old, new)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def collect_changes(write: bool) -> list[Path]:
    changed: list[Path] = []

    for path in sorted((ROOT / "docs" / "en" / "models").glob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated = original.replace("../../models/", "../../cn/models/")
        if path.name == "splines.md":
            updated = updated.replace("../semiparametric.md", "semiparametric.md")
        if updated != original:
            changed.append(path)
            if write:
                path.write_text(updated, encoding="utf-8", newline="\n")

    for path in sorted((ROOT / "docs" / "cn" / "models").glob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated = original.replace("../en/models/", "../../en/models/")
        if path.name == "splines.md":
            updated = updated.replace("../semiparametric.md", "semiparametric.md")
        if updated != original:
            changed.append(path)
            if write:
                path.write_text(updated, encoding="utf-8", newline="\n")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    changed = collect_changes(write=args.write)
    if args.check and changed:
        print("Documentation links require normalization:", file=sys.stderr)
        for path in changed:
            print(f"- {path.relative_to(ROOT)}", file=sys.stderr)
        return 1

    action = "updated" if args.write else "checked"
    print(f"Documentation links {action}; affected files: {len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
