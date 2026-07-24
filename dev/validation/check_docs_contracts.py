#!/usr/bin/env python3
"""Validate maintained Markdown links and release-facing documentation contracts."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]

MAINTAINED_PATHS = (
    ROOT / "README.md",
    ROOT / "docs" / "index.md",
    ROOT / "docs" / "en" / "usage.md",
    ROOT / "docs" / "cn" / "usage.md",
    ROOT / "docs" / "en" / "guides" / "implemented-methods.md",
    ROOT / "docs" / "cn" / "guides" / "implemented-methods.md",
)

MAINTAINED_GLOBS = (
    "docs/en/models/*.md",
    "docs/cn/models/*.md",
)

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")

BANNED_TEXT = {
    "README.md": (
        "## Important Statistical Contracts",
        "## PR #79 Final Validation",
        "## Backend Execution Status",
    ),
    "docs/index.md": (
        "USAGE_CN.md",
        "CoxPHCV` (Skeleton",
    ),
}

BANNED_CURRENT_STATUS = (
    "PARTIAL_REMOTE_PENDING",
    "physical CUDA validation remains pending",
    "physical CuPy/Torch CUDA validation remains pending",
    "真实 CUDA 验证仍待完成",
)


def iter_maintained_files() -> list[Path]:
    files = set(MAINTAINED_PATHS)
    for pattern in MAINTAINED_GLOBS:
        files.update(ROOT.glob(pattern))
    return sorted(path for path in files if path.is_file())


def normalize_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    # Markdown permits an optional quoted title after whitespace.
    target = target.split(maxsplit=1)[0]
    target = unquote(target)
    return target.split("#", 1)[0]


def validate_links(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for match in LINK_RE.finditer(text):
        raw_target = match.group(1)
        target = normalize_link_target(raw_target)
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            errors.append(f"{path.relative_to(ROOT)}: link escapes repository: {raw_target}")
            continue
        if not resolved.exists():
            errors.append(
                f"{path.relative_to(ROOT)}: missing relative link target "
                f"{raw_target!r} -> {resolved.relative_to(ROOT)}"
            )
    return errors


def validate_content(path: Path, text: str) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    errors: list[str] = []
    for banned in BANNED_TEXT.get(rel, ()):
        if banned in text:
            errors.append(f"{rel}: banned release-facing text remains: {banned!r}")
    if rel.startswith("docs/") and not (
        "/changelog" in rel or "/releases/" in rel
    ):
        for banned in BANNED_CURRENT_STATUS:
            if banned in text:
                errors.append(f"{rel}: stale global validation status remains: {banned!r}")
    return errors


def main() -> int:
    errors: list[str] = []
    files = iter_maintained_files()
    for path in files:
        text = path.read_text(encoding="utf-8")
        errors.extend(validate_links(path, text))
        errors.extend(validate_content(path, text))

    if errors:
        print("Documentation contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Documentation contract check passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
