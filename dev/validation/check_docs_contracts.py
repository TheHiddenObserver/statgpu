#!/usr/bin/env python3
"""Validate release-facing Markdown links and documentation contracts."""

from __future__ import annotations

import ast
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
    "docs/en/**/*.markdown",
    "docs/cn/**/*.markdown",
)

FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HTML_LINK_RE = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
API_TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|", re.MULTILINE)

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
        "CoxPH` (Skeleton",
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

# Learner pages may curate the parameter table used for decision guidance, but
# their explicitly marked Complete API reference must stay exhaustive.  Parse
# source with ast instead of importing statgpu so this check remains lightweight
# and runs in the Python-3.9 docs-contract job without package dependencies.
MODEL_API_CONTRACTS = {
    "docs/en/models/ridge.md": (
        ROOT / "statgpu/linear_model/wrappers/_ridge.py",
        "Ridge",
    ),
    "docs/cn/models/ridge.md": (
        ROOT / "statgpu/linear_model/wrappers/_ridge.py",
        "Ridge",
    ),
    "docs/en/models/lasso.md": (
        ROOT / "statgpu/linear_model/wrappers/_lasso.py",
        "Lasso",
    ),
    "docs/cn/models/lasso.md": (
        ROOT / "statgpu/linear_model/wrappers/_lasso.py",
        "Lasso",
    ),
    "docs/en/models/elastic-net.md": (
        ROOT / "statgpu/linear_model/wrappers/_elasticnet.py",
        "ElasticNet",
    ),
    "docs/cn/models/elastic-net.md": (
        ROOT / "statgpu/linear_model/wrappers/_elasticnet.py",
        "ElasticNet",
    ),
}


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
    """Validate source-relative links without reimplementing site routing.

    Root-absolute targets such as ``/en/...``, ``/cn/...`` and ``/dashboard/``
    are deployment routes, not repository paths. They are validated against the
    assembled VitePress/dashboard artifact by ``scripts/verify-site.mjs``. This
    source-level contract checker deliberately owns only links whose targets can
    be resolved unambiguously in the repository tree.
    """
    errors: list[str] = []
    for raw_target in iter_link_targets(text):
        target = normalize_link_target(raw_target)
        if not target or target.startswith(("#",) + SKIP_SCHEMES):
            continue

        if target.startswith("/"):
            # Site-root routes are validated after VitePress clean-URL routing,
            # base-path rewriting, and dashboard assembly have been applied.
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


def validate_archive_counterpart(path: Path) -> list[str]:
    """Require every maintained EN/CN archive to have its bilingual peer."""
    if path.suffix.lower() != ".markdown":
        return []
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if len(parts) < 3 or parts[0] != "docs" or parts[1] not in {"en", "cn"}:
        return []
    other_language = "cn" if parts[1] == "en" else "en"
    counterpart = ROOT / "docs" / other_language / Path(*parts[2:])
    if counterpart.is_file():
        return []
    return [
        f"{rel.as_posix()}: missing bilingual archive counterpart "
        f"{counterpart.relative_to(ROOT).as_posix()}"
    ]


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


def normalize_python_fence(code: str) -> str:
    """Strip doctest prompts while preserving ordinary Python indentation."""
    lines: list[str] = []
    for line in code.splitlines():
        if line.startswith((">>> ", "... ")):
            line = line[4:]
        lines.append(line)
    return "\n".join(lines)


def validate_python_fences(path: Path, text: str) -> list[str]:
    """Require maintained Python examples to be syntactically valid."""
    rel = path.relative_to(ROOT).as_posix()
    if is_historical(rel):
        return []

    errors: list[str] = []
    for index, match in enumerate(PYTHON_FENCE_RE.finditer(text), start=1):
        code = normalize_python_fence(match.group(1))
        try:
            ast.parse(code)
        except SyntaxError as exc:
            errors.append(
                f"{rel}: Python fence {index} is invalid at line "
                f"{exc.lineno}: {exc.msg}"
            )
    return errors


def constructor_parameter_names(source_path: Path, class_name: str) -> list[str]:
    """Return declared public constructor parameter names in source order."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == "__init__":
                args = [
                    *getattr(member.args, "posonlyargs", []),
                    *member.args.args,
                    *member.args.kwonlyargs,
                ]
                return [arg.arg for arg in args if arg.arg != "self"]
        raise RuntimeError(f"{source_path}: class {class_name} has no __init__")
    raise RuntimeError(f"{source_path}: class {class_name} not found")


def documented_constructor_parameters(text: str, class_name: str) -> list[str]:
    """Read the explicitly marked Complete API constructor table."""
    start = f"<!-- API-CONSTRUCTOR-START:{class_name} -->"
    end = f"<!-- API-CONSTRUCTOR-END:{class_name} -->"
    if start not in text or end not in text:
        return []
    section = text.split(start, 1)[1].split(end, 1)[0]
    return API_TABLE_ROW_RE.findall(section)


def validate_model_api_constructor_contract(path: Path, text: str) -> list[str]:
    """Require marked model API tables to match the real wrapper constructor."""
    rel = path.relative_to(ROOT).as_posix()
    contract = MODEL_API_CONTRACTS.get(rel)
    if contract is None:
        return []

    source_path, class_name = contract
    source_params = constructor_parameter_names(source_path, class_name)
    documented_params = documented_constructor_parameters(text, class_name)

    if not documented_params:
        return [
            f"{rel}: missing marked Complete API constructor table for {class_name}"
        ]

    source_set = set(source_params)
    documented_set = set(documented_params)
    errors: list[str] = []

    missing = [name for name in source_params if name not in documented_set]
    extra = [name for name in documented_params if name not in source_set]
    duplicates = sorted(
        name for name in documented_set if documented_params.count(name) > 1
    )

    if missing:
        errors.append(
            f"{rel}: Complete API table is missing {class_name} constructor "
            f"parameters: {', '.join(missing)}"
        )
    if extra:
        errors.append(
            f"{rel}: Complete API table documents non-constructor parameters for "
            f"{class_name}: {', '.join(extra)}"
        )
    if duplicates:
        errors.append(
            f"{rel}: Complete API table repeats {class_name} constructor "
            f"parameters: {', '.join(duplicates)}"
        )

    return errors


def main() -> int:
    errors: list[str] = []
    files = iter_maintained_files()
    for path in files:
        text = path.read_text(encoding="utf-8")
        errors.extend(validate_links(path, text))
        errors.extend(validate_archive_counterpart(path))
        errors.extend(validate_content(path, text))
        errors.extend(validate_python_fences(path, text))
        errors.extend(validate_model_api_constructor_contract(path, text))

    if errors:
        print("Documentation contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Documentation contract check passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())