#!/usr/bin/env python3
"""Fix or check deterministic bilingual documentation link patterns."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SWITCH_MARKERS = (
    "Switch:",
    "Language switch:",
    "切换:",
    "切换：",
    "语言切换:",
    "语言切换：",
    "English:",
)

MODEL_LINK_RE = re.compile(
    r"\((?:\.\./)+(?:en/|cn/)?models/[^)#\s]+\.md(?:#[^)]*)?\)"
)


def write_utf8(path: Path, text: str) -> None:
    """Write LF-normalized UTF-8 text on every supported Python version."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def normalize_switch_links(text: str, target: str) -> str:
    """Normalize only bilingual switch lines, never ordinary cross-model links."""
    normalized: list[str] = []
    replacement = f"({target})"
    for line in text.splitlines(keepends=True):
        if any(marker in line for marker in SWITCH_MARKERS):
            line = MODEL_LINK_RE.sub(replacement, line)
        normalized.append(line)
    return "".join(normalized)


def normalize_file(path: Path, target: str) -> str:
    original = path.read_text(encoding="utf-8")
    updated = normalize_switch_links(original, target)
    if path.name == "splines.md":
        updated = updated.replace("../semiparametric.md", "semiparametric.md")
    return updated


def collect_changes(write: bool) -> list[Path]:
    changed: list[Path] = []

    for path in sorted((ROOT / "docs" / "en" / "models").glob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated = normalize_file(path, f"../../cn/models/{path.name}")
        if updated != original:
            changed.append(path)
            if write:
                write_utf8(path, updated)

    for path in sorted((ROOT / "docs" / "cn" / "models").glob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated = normalize_file(path, f"../../en/models/{path.name}")
        if updated != original:
            changed.append(path)
            if write:
                write_utf8(path, updated)

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
