"""Keep user contracts visible while engineering history lives with contributors."""
from doc_paths import model_doc
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PANEL_PAGES = (
    "between-ols", "covariance", "diagnostics", "fama-macbeth",
    "first-difference-ols", "fit-statistics", "panel-ols", "pooled-ols",
    "random-effects",
)


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


@pytest.mark.parametrize("language", ["en", "cn"])
def test_ridge_reference_keeps_exact_split_contract_and_preprocessing_guidance(language):
    text = read(f"docs/{language}/reference/linear-model-api.md")
    section = text.split("### Custom RidgeCV training subsets" if language == "en"
                         else "### RidgeCV 自定义训练子集", 1)[1]
    section = section.split("\n## LassoCV", 1)[0]
    section = " ".join(section.split())
    assert "sample_weight" in section
    required = (
        ("exactly the supplied training rows", "with or without", "full complement",
         "every row exactly once", "need not cover all observations", "ValueError",
         "external CV loop", "final refit uses all rows", "designated validation rows",
         "ordinary K-fold CV", "fewer than four observations", "one alpha candidate",
         "`cv>=2`", "positive total sample weight", "individual zero weights") if language == "en" else
        ("无论是否提供", "恰好覆盖每一行一次", "完整补集", "指定训练行", "验证集不必",
         "覆盖全部观测", "ValueError", "外部交叉验证循环", "最终重拟合",
         "全部行", "验证行上评估", "普通 K 折交叉验证", "不足四行", "一个 alpha 候选",
         "`cv>=2`", "权重总量都必须为正", "权重可以为零")
    )
    for term in required:
        assert term in section, term
    for detail in ("/issues/243", "known implementation issue", "已知的实现问题",
                   "currently substitutes", "替换所给训练索引", "folds_are_complete"):
        assert detail not in section, detail


@pytest.mark.parametrize("page", PANEL_PAGES)
def test_panel_user_pages_do_not_present_source_specific_acceptance_as_current(page):
    text = read(f"docs/en/panel/{page}.md")
    for stale in ("current PR", "exact-head", "5068da3f", "8c60db00",
                  "dev/tests/", "dev/validation/", "dev/benchmarks/"):
        assert stale not in text, (page, stale)
    assert "## References" in text or page == "fit-statistics"


@pytest.mark.parametrize("language", ["en", "cn"])
@pytest.mark.parametrize("model", ["anova", "covariance"])
def test_model_evidence_is_routed_out_of_learner_pages(language, model):
    text = read(f"docs/{language}/{model_doc(model)}.md")
    assert "dev/reviews/pr168-model-validation-provenance.md#" in text
    for phrase in ("Maintained tests cover", "维护测试覆盖", "维护中的测试覆盖"):
        assert phrase not in text
    assert "## References" in text or "## 参考文献" in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_general_guides_keep_numerical_contract_without_test_status(language):
    solver = read(f"docs/{language}/guides/solver-algorithms.md")
    assert "lbfgs_solver(QuantileLoss, ...)" in solver
    assert "regression-covered" not in solver
    assert "由回归测试保留" not in solver
    assert 'solver="lbfgs"' in solver and "Quantile" in solver
    precision = read(f"docs/{language}/guides/lbfgs-float32-precision-contract.md")
    for term in ("float32", "float64", "sample_weight"):
        assert term in precision
    assert "Do not loosen the statistical objective" not in precision
    assert "而修改统计目标或改变求解器语义" not in precision
    diagnostics = read(f"docs/{language}/guides/regression-diagnostics.md")
    assert "NumPy" in diagnostics and "CPU" in diagnostics
    assert "Reference tests compare" not in diagnostics
    assert "参考测试与" not in diagnostics


def test_provenance_records_preserve_historical_scope():
    panel = read("dev/reviews/pr168-panel-validation-provenance.md")
    assert "880b7b8474ab4728ed44a4556b1b64faaa6e7b55" in panel
    assert "5068da3f" in panel and "8c60db00" in panel
    models = read("dev/reviews/pr168-model-validation-provenance.md")
    assert "880b7b8474ab4728ed44a4556b1b64faaa6e7b55" in models
    assert "anova_oneway" in models and "GraphicalLassoCV" in models
    general = read("dev/reviews/pr168-general-guide-provenance.md")
    assert "statsmodels.OLSInfluence" in general
    assert "exhaustive_safety_fallback" in general


def test_dashboard_plan_is_developer_history():
    assert not (ROOT / "docs/en/guides/statgpu_benchmark_dashboard_next_phase_plan.md").exists()
    plan = read("dev/plans/statgpu_benchmark_dashboard_next_phase_plan.md")
    assert "PR78" in plan or "PR #78" in plan
    assert "historical" in plan.lower()
    rollout = read("docs/benchmark-dashboard/rollout-plan.md")
    assert "dev/plans/statgpu_benchmark_dashboard_next_phase_plan.md" in rollout


@pytest.mark.parametrize("language", ["en", "cn"])
def test_benchmark_guide_routes_maintenance_and_uses_generated_data(language):
    text = read(f"docs/{language}/guides/benchmarks.md")
    assert "frontend/docs/benchmark-dashboard-maintenance.md" in text
    assert "source_inventory.json" in text
    for stale in ("1,774", "1774", "CV (0)", "CV(0)", "npm run", "pytest "):
        assert stale not in text, stale


def test_dashboard_count_inspection_example_executes(monkeypatch, capsys, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT))
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate

    text = read("frontend/docs/benchmark-dashboard-maintenance.md")
    assert "Generated files are not committed" in text
    assert "first run the bundle-generation command" in text
    start = "python - <<'PYCOUNTS'\n"
    assert text.count(start) == 1
    code = text.split(start, 1)[1].split("\nPYCOUNTS", 1)[0]
    # Exercise the documented generation prerequisite in an empty workspace;
    # a local build or previously committed bundle must not mask missing inputs.
    manifest = load_manifest(ROOT)
    assert manifest is not None
    data, report, inventory = generate(
        ROOT / "results", deterministic=True, manifest=manifest, strict_sources=True,
    )
    bundle = tmp_path / "frontend/public/data"
    bundle.mkdir(parents=True)
    for filename, payload in (
        ("source_inventory.json", inventory),
        ("parse_report.json", report),
        ("benchmark_data.json", data),
    ):
        (bundle / filename).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "dashboard-maintenance-counts", "exec"), {})  # noqa: S102 - repository-owned example
    output = capsys.readouterr().out
    for label in ("registered_sources", "available_registered_sources",
                  "parsed_registered_sources", "normalized runs",
                  "model registry entries", "CV metric rows"):
        assert label in output
