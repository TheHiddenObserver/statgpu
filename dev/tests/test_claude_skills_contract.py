from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frontmatter(text: str) -> str:
    lines = text.splitlines()
    assert lines and lines[0].strip() == "---", "SKILL.md must start with YAML frontmatter"
    try:
        end = lines[1:].index("---") + 1
    except ValueError as exc:
        raise AssertionError("SKILL.md frontmatter must have a closing ---") from exc
    return "\n".join(lines[1:end])


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
        frontmatter = _frontmatter(text)
        assert "description:" in frontmatter
        assert supporting_name in text, f"{entry} must point readers to {supporting_name}"
        assert len(text.splitlines()) < 500, f"{entry} should stay concise; move detail to supporting files"


def test_code_review_remains_a_blocking_forked_independent_pass():
    text = _read(SKILLS / "code-review" / "SKILL.md")
    frontmatter = _frontmatter(text)
    assert "context: fork" in frontmatter
    assert "background: false" in frontmatter
    assert "Do not trigger merely because code is being edited" in frontmatter


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
        ROOT / "docs" / "en" / "usage.md",
        ROOT / "docs" / "cn" / "usage.md",
    ]
    for path in current_files:
        text = _read(path)
        assert ".claude/skills/" in text, f"{path} should point to canonical skill entrypoints"
