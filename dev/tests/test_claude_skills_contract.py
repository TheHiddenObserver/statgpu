import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the scalar top-level YAML subset used by repository SKILL.md files.

    This deliberately avoids adding a PyYAML runtime/dev dependency while still
    rejecting the frontmatter failures that would break Claude Code discovery:
    missing delimiters, malformed/indented top-level entries, duplicate keys,
    missing values, and invalid key names.
    """

    lines = text.splitlines()
    assert lines and lines[0].strip() == "---", "SKILL.md must start with YAML frontmatter"
    try:
        end = lines[1:].index("---") + 1
    except ValueError as exc:
        raise AssertionError("SKILL.md frontmatter must have a closing ---") from exc

    metadata: dict[str, str] = {}
    for raw_line in lines[1:end]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        assert raw_line == raw_line.lstrip(), (
            "repository SKILL.md frontmatter must use scalar top-level keys only"
        )
        assert ":" in raw_line, f"malformed frontmatter line: {raw_line!r}"
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        assert re.fullmatch(r"[A-Za-z0-9_-]+", key), f"invalid frontmatter key: {key!r}"
        assert key not in metadata, f"duplicate frontmatter key: {key}"
        assert value, f"frontmatter value must not be empty: {key}"
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        metadata[key] = value
    return metadata


def test_canonical_claude_skill_layout_and_entrypoints():
    expected = {
        "benchmark": "schema.md",
        "code-review": "review-matrix.md",
        "new-module-dev": "workflow.md",
    }

    for skill_name, supporting_name in expected.items():
        skill_dir = SKILLS / skill_name
        entry = skill_dir / "SKILL.md"
        supporting = skill_dir / supporting_name
        assert entry.is_file(), f"missing canonical skill entrypoint: {entry}"
        assert supporting.is_file(), f"missing referenced supporting file: {supporting}"

        text = _read(entry)
        metadata = _parse_frontmatter(text)
        assert metadata["name"] == skill_name
        assert metadata.get("description")
        assert metadata.get("when_to_use")
        assert supporting_name in text, f"{entry} must point readers to {supporting_name}"
        assert len(text.splitlines()) < 500, f"{entry} should stay concise; move detail to supporting files"


def test_code_review_remains_a_blocking_forked_independent_pass():
    text = _read(SKILLS / "code-review" / "SKILL.md")
    metadata = _parse_frontmatter(text)
    assert metadata["context"] == "fork"
    assert metadata["background"] == "false"
    assert "2.1.218" in metadata["compatibility"]
    assert "Do not trigger merely because code is being edited" in metadata["when_to_use"]
    assert "blocking fork behavior requires Claude Code >= 2.1.218" in text


def test_new_capability_defaults_cannot_be_defined_away():
    new_module = _read(SKILLS / "new-module-dev" / "SKILL.md")
    workflow = _read(SKILLS / "new-module-dev" / "workflow.md")
    review_matrix = _read(SKILLS / "code-review" / "review-matrix.md")

    assert "default backend contract is **NumPy + CuPy + Torch**" in new_module
    assert "default completion contract also includes **direct fit + CV" in new_module
    assert "default completion contract is NumPy + CuPy + Torch" in review_matrix
    assert "direct fit + CV path/grid/folds/scoring/selection/final refit" in review_matrix
    assert "Repository-default closure" in workflow
    assert "CPU-only or direct-fit-only first draft" in workflow
    assert "backend/CV deferral" in workflow


def test_api_only_scope_remains_narrow():
    new_module = _read(SKILLS / "new-module-dev" / "SKILL.md")
    review = _read(SKILLS / "code-review" / "SKILL.md")
    assert "Constructor deprecation only" in new_module
    assert "no new numerical backend matrix unless dispatch changes" in new_module
    assert "API-only/refactor-only work" in review
    assert "does not require a new backend implementation" in review


def test_legacy_flat_skill_paths_are_pointers_not_duplicate_authority():
    for name in ("benchmark", "code-review", "new-module-dev"):
        legacy = SKILLS / f"{name}.md"
        assert legacy.is_file(), f"historical link compatibility file missing: {legacy}"
        text = _read(legacy)
        assert text.startswith("# Legacy compatibility pointer")
        assert f"{name}/SKILL.md" in text
        assert not text.startswith("---"), "legacy flat files must not look like skill entrypoints"

    workflow_pointer = ROOT / ".claude" / "workflows" / "new-module-dev.md"
    text = _read(workflow_pointer)
    assert text.startswith("# Legacy workflow pointer")
    assert "../skills/new-module-dev/SKILL.md" in text
    assert "no longer authoritative" in text


def test_current_contributor_entrypoints_reference_canonical_skills():
    current_files = [
        ROOT / "dev" / "AGENTS.md",
        ROOT / "dev" / "plans" / "README.md",
        ROOT / "dev" / "plans" / "TO_DO.md",
    ]
    for path in current_files:
        text = _read(path)
        assert ".claude/skills/" in text, f"{path} should point to canonical skill entrypoints"


def test_vitepress_usage_pages_use_external_repo_links_for_non_docs_files():
    for path in (ROOT / "docs" / "en" / "usage.md", ROOT / "docs" / "cn" / "usage.md"):
        text = _read(path)
        assert "https://github.com/TheHiddenObserver/statgpu/blob/master/.claude/skills/" in text
        assert "https://github.com/TheHiddenObserver/statgpu/blob/master/dev/AGENTS.md" in text
        assert "](../../.claude/skills/" not in text
        assert "](../../dev/AGENTS.md)" not in text


def test_active_gaussian_plan_uses_canonical_review_skill():
    plan = _read(ROOT / "dev" / "plans" / "gaussian_inference_backend_native_plan.md")
    phase7 = plan.split("### Phase 7 — implementation review/fix", 1)[1].split(
        "### Phase 8", 1
    )[0]
    assert ".claude/skills/code-review/SKILL.md" in phase7
    assert ".claude/skills/code-review.md" not in phase7
