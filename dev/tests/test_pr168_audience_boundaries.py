"""Keep user contracts visible while engineering history lives with contributors."""
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
def test_ridge_known_issue_reference_keeps_conditions_consequence_and_workaround(language):
    text = read(f"docs/{language}/reference/linear-model-api.md")
    section = text.split("### Custom RidgeCV training subsets" if language == "en"
                         else "### RidgeCV 自定义训练子集的已知问题", 1)[1]
    section = section.split("\n## LassoCV", 1)[0]
    assert "https://github.com/TheHiddenObserver/statgpu/issues/243" in section
    assert "sample_weight" in section
    required = (
        ("known implementation issue", "partition every row once", "full complement",
         "supplied training indices", "Validation losses and alpha", "external loop",
         "fits Ridge", "Ordinary complete K-folds") if language == "en" else
        ("已知的实现问题", "恰好覆盖每一行一次", "完整补集", "所给训练索引",
         "验证损失和 alpha", "外部循环", "拟合 Ridge", "普通完整 K 折")
    )
    for term in required:
        assert term in section, term


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
    text = read(f"docs/{language}/models/{model}.md")
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


def test_dashboard_count_inspection_example_executes(monkeypatch, capsys):
    text = read("frontend/docs/benchmark-dashboard-maintenance.md")
    start = "python - <<'PYCOUNTS'\n"
    assert text.count(start) == 1
    code = text.split(start, 1)[1].split("\nPYCOUNTS", 1)[0]
    monkeypatch.chdir(ROOT)
    exec(compile(code, "dashboard-maintenance-counts", "exec"), {})  # noqa: S102 - repository-owned example
    output = capsys.readouterr().out
    for label in ("registered_sources", "available_registered_sources",
                  "parsed_registered_sources", "normalized runs",
                  "model registry entries", "CV metric rows"):
        assert label in output
