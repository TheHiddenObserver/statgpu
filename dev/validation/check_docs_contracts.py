#!/usr/bin/env python3
"""Validate release-facing Markdown links and documentation contracts."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]

MAINTAINED_PATHS = (
    ROOT / "README.md",
    ROOT / "docs" / "index.md",
)

MAINTAINED_GLOBS = (
    "docs/en/**/*.md",
    "docs/cn/**/*.md",
)

FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HTML_LINK_RE = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)

BANNED_TEXT = {
    "README.md": (
        "## Important Statistical Contracts",
        "## PR #79 Final Validation",
        "## Backend Execution Status",
        "sklearn.base.clone() support",
        "intentional CPU boundaries are limited to",
        "### Real-Data Performance",
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

HISTORICAL_PARTS = (
    "/releases/",
    "changelog",
    "history",
)

SKIP_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "data:",
    "javascript:",
)


def iter_maintained_files() -> list[Path]:
    files = set(MAINTAINED_PATHS)
    for pattern in MAINTAINED_GLOBS:
        files.update(ROOT.glob(pattern))
    return sorted(path for path in files if path.is_file())


def strip_fenced_code(text: str) -> str:
    return FENCED_CODE_RE.sub("", text)


def normalize_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    # Markdown permits an optional quoted title after whitespace.
    target = target.split(maxsplit=1)[0]
    target = unquote(target)
    target = target.split("#", 1)[0]
    target = target.split("?", 1)[0]
    return target


def iter_link_targets(text: str) -> list[str]:
    searchable = strip_fenced_code(text)
    targets = [match.group(1) for match in INLINE_LINK_RE.finditer(searchable)]
    targets.extend(match.group(1) for match in REFERENCE_LINK_RE.finditer(searchable))
    targets.extend(match.group(1) for match in HTML_LINK_RE.finditer(searchable))
    return targets


def validate_links(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for raw_target in iter_link_targets(text):
        target = normalize_link_target(raw_target)
        if not target or target.startswith(("#",) + SKIP_SCHEMES):
            continue

        if target.startswith("/"):
            resolved = (ROOT / target.lstrip("/")).resolve()
        else:
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


def is_historical(rel: str) -> bool:
    normalized = f"/{rel.lower()}"
    return any(part in normalized for part in HISTORICAL_PARTS)


def validate_content(path: Path, text: str) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    errors: list[str] = []
    for banned in BANNED_TEXT.get(rel, ()):
        if banned in text:
            errors.append(f"{rel}: banned release-facing text remains: {banned!r}")

    if rel.startswith("docs/") and not is_historical(rel):
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
