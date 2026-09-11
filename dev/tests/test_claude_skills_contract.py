import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / ".claude" / "skills"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict:
    lines = text.splitlines()
    assert lines and lines[0].strip() == "---", "SKILL.md must start with YAML frontmatter"
    try:
        end = lines[1:].index("---") + 1
    except ValueError as exc:
        raise AssertionError("SKILL.md frontmatter must have a closing ---") from exc

    raw = "\n".join(lines[1:end])
    metadata = yaml.load(raw, Loader=_UniqueKeyLoader)
    assert isinstance(metadata, dict), "SKILL.md frontmatter must parse to a YAML mapping"
    return metadata


def test_canonical_claude_skill_layout_and_entrypoints():
    expected = {
        "benchmark": ["schema.md"],
        "code-review": ["review-matrix.md", "target-resolution.md"],
        "new-module-dev": ["workflow.md"],
    }

    for skill_name, supporting_names in expected.items():
        skill_dir = SKILLS / skill_name
        entry = skill_dir / "SKILL.md"
        assert entry.is_file(), f"missing canonical skill entrypoint: {entry}"

        text = _read(entry)
        metadata = _frontmatter(text)
        assert metadata["name"] == skill_name
        assert metadata.get("description")
        assert metadata.get("when_to_use")
        for supporting_name in supporting_names:
            supporting = skill_dir / supporting_name
            assert supporting.is_file(), f"missing referenced supporting file: {supporting}"
            assert supporting_name in text, f"{entry} must point readers to {supporting_name}"
        assert len(text.splitlines()) < 500, f"{entry} should stay concise; move detail to supporting files"
        assert text.endswith("\n"), f"{entry} must end with a newline"


def test_code_review_remains_a_blocking_forked_independent_pass():
    text = _read(SKILLS / "code-review" / "SKILL.md")
    metadata = _frontmatter(text)
    assert metadata["context"] == "fork"
    assert metadata["background"] is False
    assert "commit" in metadata["argument-hint"]
    compatibility = metadata["compatibility"]
    assert "2.1.218" in compatibility
    assert "earlier versions block forked skills by default" in compatibility
    assert "read `dev/AGENTS.md`" in text
    assert "target-resolution.md" in text
    assert "Before v2.1.218, forked skills already blocked" in text
    assert "--fix` is accepted as an alias for `auto-fix" in text
    assert "remote" in text and "local unpushed fixes" in text


def test_code_review_target_resolution_is_fail_closed_and_stale_aware():
    target = _read(SKILLS / "code-review" / "target-resolution.md")
    for phrase in (
        "base_sha",
        "head_sha",
        "worktree_fingerprint",
        "Working-tree fingerprint",
        "Explicit PR",
        "Explicit commit",
        "commit_mode",
        "`commit_mode=patch`",
        "`commit_mode=snapshot`",
        "target_kind=commit",
        "merge-commit patch review lacks an explicit parent/base/first-parent instruction",
        "No explicit scope / working-tree review",
        "gh pr view",
        "merge-base",
        "either the PR `base_sha` or `head_sha` changed",
        "dirty working tree is **expected and valid**",
        "PR comment freshness",
        "Fail-closed conditions",
        "target/write-state mismatch",
    ):
        assert phrase in target


def test_benchmark_syncs_concrete_executed_device_and_records_it():
    benchmark = _read(SKILLS / "benchmark" / "SKILL.md")
    schema = _read(SKILLS / "benchmark" / "schema.md")
    review_matrix = _read(SKILLS / "code-review" / "review-matrix.md")
    assert "cp.cuda.Device(device_id).synchronize()" in benchmark
    assert "torch.cuda.synchronize(device=target)" in benchmark
    assert "concrete device that actually executed" in benchmark
    assert "synchronized_device" in schema
    assert "gpu_uuid" in schema
    assert "concrete executed device" in review_matrix


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


def test_new_module_dev_requires_contract_reconnaissance_for_reconciliation():
    skill = _read(SKILLS / "new-module-dev" / "SKILL.md")
    workflow = _read(SKILLS / "new-module-dev" / "workflow.md")

    for phrase in (
        "Contract reconnaissance before implementation",
        "existing capability reconciliation / public contract repair",
        "consumer graph",
        "requested method -> resolved method -> reported result method",
        "resampling unit and data-generating mechanism",
        "current published version",
        "Exact-head evidence closure",
    ):
        assert phrase in skill

    for phrase in (
        "Phase 0 — target and contract reconnaissance",
        "Existing capability reconciliation / contract repair",
        "Resampling contract",
        "Phase 4 — documentation and release boundary",
        "Phase 5 — independent review/fix",
        "Phase 6 — exact-head evidence closure",
    ):
        assert phrase in workflow

    assert workflow.index("Phase 4 — documentation and release boundary") < workflow.index(
        "Phase 5 — independent review/fix"
    )
    assert workflow.index("Phase 5 — independent review/fix") < workflow.index(
        "Phase 6 — exact-head evidence closure"
    )


def test_code_review_covers_inference_identity_resampling_and_consumer_graph():
    review = _read(SKILLS / "code-review" / "SKILL.md")
    matrix = _read(SKILLS / "code-review" / "review-matrix.md")

    for phrase in (
        "existing capability reconciliation / public contract repair",
        "consumer inventory",
        "public requested method -> resolved numerical method",
        "Resampling is blocking",
        "release-boundary",
        "exact-head",
    ):
        assert phrase in review

    for phrase in (
        "Contract reconciliation",
        "requested/resolved/reported method",
        "Resampling checks",
        "generic bootstrap/permutation path that reconstructs an incompatible model/family",
        "Documentation and release-boundary checks",
        "Evidence freshness checks",
    ):
        assert phrase in matrix


def test_skill_eval_definitions_do_not_masquerade_as_runtime_evidence():
    review = _read(SKILLS / "code-review" / "SKILL.md")
    matrix = _read(SKILLS / "code-review" / "review-matrix.md")
    new_module = _read(SKILLS / "new-module-dev" / "SKILL.md")
    workflow = _read(SKILLS / "new-module-dev" / "workflow.md")

    assert "Eval definitions are not runtime execution evidence." in review
    assert "actual skill-runtime/grading run" in review
    assert "not actual Claude Code trigger/output behavior" in matrix
    assert "behavioral-runtime claims" in matrix
    assert "committed eval definitions and static contract tests prove only policy/schema presence" in new_module
    assert "skill-runtime behavioral evidence remains" in new_module
    assert "Project-skill behavioral runtime" in workflow
    assert "static contract tests do not execute the skill" in workflow


def test_skill_evals_cover_reconciliation_resampling_and_release_boundary():
    new_module = json.loads(
        _read(SKILLS / "new-module-dev" / "evals" / "evals.json")
    )["evals"]
    review = json.loads(_read(SKILLS / "code-review" / "evals" / "evals.json"))[
        "evals"
    ]

    assert len(new_module) >= 6
    assert len(review) >= 7

    new_text = "\n".join(
        item["prompt"] + "\n" + item["expected_output"] for item in new_module
    ).lower()
    review_text = "\n".join(
        item["prompt"] + "\n" + item["expected_output"] for item in review
    ).lower()

    for phrase in (
        "capability reconciliation",
        "generic glm bootstrap",
        "0.2.6",
        "exact-head evidence",
    ):
        assert phrase in new_text

    for phrase in (
        "requested/resolved/reported",
        "resampling",
        "consumer",
        "green ci",
    ):
        assert phrase in review_text


def test_penalized_glm_review_eval_is_blind_discovery_fixture():
    data = json.loads(_read(SKILLS / "code-review" / "evals" / "evals.json"))
    snapshot_sha = "e364717bd62ff84d260eab4e0b7020e26a98caf0"
    matches = [item for item in data["evals"] if item["id"] == 7]
    assert len(matches) == 1

    golden = matches[0]
    prompt_lower = golden["prompt"].lower()
    assert snapshot_sha in golden["prompt"]
    for leaked_answer in (
        "inference_method='debiased'",
        "m_estimation",
        "gaussian penalized-linear",
        "penalty='l1'",
        "sandwich results",
    ):
        assert leaked_answer not in prompt_lower, (
            "penalized-GLM discovery prompt must not leak expected findings: "
            f"{leaked_answer!r}"
        )

    combined = "\n".join(
        [golden["expected_output"], *golden["assertions"]]
    ).lower()
    for phrase in (
        "target_kind=commit",
        "commit_mode=snapshot",
        snapshot_sha,
        "debiased",
        "m_estimation",
        "high",
        "gaussian",
        "l1",
        "parameterinferenceresult",
        "critical",
        "blocking",
        "consumer",
    ):
        assert phrase in combined


def test_legacy_paths_do_not_occupy_dynamic_workflow_namespace():
    for name in ("benchmark", "code-review", "new-module-dev"):
        legacy = SKILLS / f"{name}.md"
        assert legacy.is_file(), f"historical link compatibility file missing: {legacy}"
        text = _read(legacy)
        assert text.startswith("# Legacy compatibility pointer")
        assert f"{name}/SKILL.md" in text
        assert not text.startswith("---"), "legacy flat files must not look like skill entrypoints"

    assert not (ROOT / ".claude" / "workflows" / "new-module-dev.md").exists()
    legacy_note = ROOT / ".claude" / "legacy" / "new-module-dev-workflow.md"
    assert legacy_note.is_file()
    assert "Dynamic Workflow" in _read(legacy_note)


def test_claude_bootstrap_routes_forked_agents_to_project_guide():
    bootstrap = _read(ROOT / "CLAUDE.md")
    assert "dev/AGENTS.md" in bootstrap
    assert ".claude/skills/code-review/SKILL.md" in bootstrap
    assert ".claude/workflows/" in bootstrap
    assert "Dynamic Workflow" in bootstrap


def test_skill_eval_definitions_are_present_and_well_formed():
    for skill_name in ("benchmark", "code-review", "new-module-dev"):
        path = SKILLS / skill_name / "evals" / "evals.json"
        data = json.loads(_read(path))
        assert data["skill_name"] == skill_name
        evals = data["evals"]
        assert len(evals) >= 2
        ids = [item["id"] for item in evals]
        assert len(ids) == len(set(ids))
        for item in evals:
            assert item["prompt"].strip()
            assert item["expected_output"].strip()
            assert item.get("assertions")


def test_code_review_has_substantive_known_bug_golden_eval():
    data = json.loads(_read(SKILLS / "code-review" / "evals" / "evals.json"))
    base_sha = "b17c8a6c161065bfbe9efa9f60d0fad69a464773"
    buggy_head_sha = "ac19eed297cff557a1c86595c1ec418e1b505249"

    matches = [
        item
        for item in data["evals"]
        if base_sha in item["prompt"] and buggy_head_sha in item["prompt"]
    ]
    assert len(matches) == 1, "code-review must keep one immutable known-bug golden eval"

    golden = matches[0]
    prompt_lower = golden["prompt"].lower()
    for leaked_answer in ("lassocv", "sklearn 1.2", "debiased", "docstring"):
        assert leaked_answer not in prompt_lower, (
            "known-bug golden prompt must not leak expected findings: "
            f"{leaked_answer!r}"
        )

    combined = "\n".join(
        [golden["prompt"], golden["expected_output"], *golden["assertions"]]
    ).lower()
    for phrase in (
        "three-dot",
        "api/deprecation",
        "green ci",
        "lassocv",
        "cuda/torch",
        "sklearn 1.2",
        "debiased",
        "docstring",
        "blocking",
    ):
        assert phrase in combined, f"known-bug golden eval must cover {phrase!r}"
    assert "review clean" in combined


def test_current_authority_docs_use_current_claude_namespaces_and_version_semantics():
    current_files = [
        ROOT / "dev" / "AGENTS.md",
        ROOT / "dev" / "plans" / "README.md",
        ROOT / "dev" / "plans" / "TO_DO.md",
        ROOT / "dev" / "plans" / "ROADMAP.md",
    ]
    for path in current_files:
        text = _read(path)
        assert ".claude/skills/" in text, f"{path} should point to canonical skill entrypoints"
        assert ".claude/workflows/new-module-dev.md" not in text
        assert "requires Claude Code >= 2.1.218" not in text

    agents = _read(ROOT / "dev" / "AGENTS.md")
    assert ".claude/legacy/new-module-dev-workflow.md" in agents
    assert "earlier versions" in agents or "更早版本" in agents

    roadmap = _read(ROOT / "dev" / "plans" / "ROADMAP.md")
    assert "canonical `.claude/skills/<skill-name>/SKILL.md`" in roadmap
    assert "Dynamic Workflow" in roadmap


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
