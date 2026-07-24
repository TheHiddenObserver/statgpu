#!/usr/bin/env python3
"""Fix or check deterministic bilingual documentation link patterns."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

SWITCH_MARKERS = (
    "Switch:",
    "Language switch:",
    "切换:",
    "切换：",
    "语言切换:",
    "语言切换：",
    "English:",
)

MARKDOWN_MD_LINK_RE = re.compile(r"(\[[^\]]+\]\()([^)]+\.md(?:#[^)]*)?)(\))")
DEV_DOCS_LINK_RE = re.compile(r"(?:\.\./)+dev/docs/")
RESULTS_LINK_RE = re.compile(r"(?:\.\./)+results/")


def write_utf8(path: Path, text: str) -> None:
    """Write LF-normalized UTF-8 text on every supported Python version."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def relative_target(path: Path, counterpart: Path) -> str:
    return os.path.relpath(counterpart, path.parent).replace(os.sep, "/")


def normalize_switch_links(text: str, target: str) -> str:
    """Normalize bilingual switch lines without changing ordinary cross-links."""
    normalized: list[str] = []
    for line in text.splitlines(keepends=True):
        if any(marker in line for marker in SWITCH_MARKERS):
            line = MARKDOWN_MD_LINK_RE.sub(
                lambda match: f"{match.group(1)}{target}{match.group(3)}",
                line,
                count=1,
            )
        normalized.append(line)
    return "".join(normalized)


def normalize_repository_links(path: Path, text: str) -> str:
    """Repair known links from nested docs pages to repository-root artifacts."""
    if path.name != "pytorch-backend.md" or path.parent.name != "guides":
        return text
    text = DEV_DOCS_LINK_RE.sub("../../../dev/docs/", text)
    return RESULTS_LINK_RE.sub("../../../results/", text)


def normalize_file(path: Path, counterpart: Path) -> str:
    original = path.read_text(encoding="utf-8")
    target = relative_target(path, counterpart)
    updated = normalize_switch_links(original, target)
    updated = normalize_repository_links(path, updated)
    if path.name == "splines.md" and path.parent.name == "models":
        updated = updated.replace("../semiparametric.md", "semiparametric.md")
    return updated


def iter_mirrored_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for language, other_language in (("en", "cn"), ("cn", "en")):
        language_root = DOCS / language
        other_root = DOCS / other_language
        for path in sorted(language_root.rglob("*.md")):
            counterpart = other_root / path.relative_to(language_root)
            if counterpart.is_file():
                pairs.append((path, counterpart))
    return pairs


def collect_changes(write: bool) -> list[Path]:
    changed: list[Path] = []
    for path, counterpart in iter_mirrored_pairs():
        original = path.read_text(encoding="utf-8")
        updated = normalize_file(path, counterpart)
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
