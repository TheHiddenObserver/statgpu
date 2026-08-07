from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "docs/en/guides/benchmarks.md",
    "> Last updated: 2026-07-20",
    "> Last updated: 2026-08-07",
)
replace_once(
    "docs/en/guides/benchmarks.md",
    "The canonical dashboard is restricted to benchmark sources dated **2026-06-01 or later**. It currently contains **8 registered sources, 1,774 normalized runs, and 36 models**.",
    "The canonical dashboard is restricted to benchmark sources dated **2026-06-01 or later**. Live source/run counts are read from the dashboard inventory rather than maintained as long-lived prose. **Snapshot (2026-08-07):** the canonical bundle has 9 registered/available/parsed sources and 1,796 normalized runs.",
)
replace_once(
    "docs/en/guides/benchmarks.md",
    "- Validation, accuracy, inference, prediction, convergence, and selection panels.\n- Source provenance, parse-report metadata, and source-inventory coverage.",
    "- Validation, accuracy, inference, cross-validation, prediction, convergence, and selection panels.\n- Keyboard-visible focus, accessible filter naming, disclosure controls, and exact chart-data tables.\n- Source provenance, parse-report metadata, and source-inventory coverage.",
)
replace_once(
    "docs/en/guides/benchmarks.md",
    "npx playwright install --with-deps chromium\nnpm run test:e2e",
    "npx playwright install --with-deps chromium firefox webkit\nnpm run test:e2e\nnpm run test:e2e:production",
)

replace_once(
    "docs/en/guides/statgpu_benchmark_dashboard.md",
    "The canonical manifest registers **eight benchmark sources**, all dated **2026-06-01 or later**. The generated bundle contains **1,774 normalized runs across 36 models**:",
    "The canonical manifest is the source of truth for current benchmark inputs, and the deployed inventory is the source of truth for live counts. **Snapshot (2026-08-07):** 9 registered/available/parsed sources produce 1,796 normalized runs, including the first current six-family CV package:",
)
replace_once(
    "docs/en/guides/statgpu_benchmark_dashboard.md",
    "| `ordered_inference_pr74.json` | Ordered, Quantile, sandwich, oracle, and bootstrap inference configurations |",
    "| `ordered_inference_pr74.json` | Ordered, Quantile, sandwich, oracle, and bootstrap inference configurations |\n| `cv_benchmark_20260807.json` | RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV, PenalizedGLM_CV, and CoxPHCV with explicit backend dispositions |",
)
replace_once(
    "docs/en/guides/statgpu_benchmark_dashboard.md",
    "Panels appear only when filtered rows contain the corresponding metric group:\n\n- **Validation**: pass/warn/fail checks and tolerances.",
    "Panels appear only when filtered rows contain the corresponding metric group:\n\n- **Validation**: pass/warn/fail checks and tolerances.\n- **Cross-validation**: backend disposition, CV/final-refit/total timing, selected parameters, normalized scores, convergence/failure counts, and explicit non-success reasons.",
)
replace_once(
    "docs/en/guides/statgpu_benchmark_dashboard.md",
    "All three files share one `generation_id`. In canonical mode, `eligible_total`, `registered_sources`, `available_sources`, and `parsed_sources` refer only to the eight manifest-registered June-or-later sources.",
    "All three files share one `generation_id`. In canonical mode, inventory fields refer to the manifest-registered June-or-later sources. Read the generated inventory for current counts instead of copying those counts into downstream documentation.",
)
replace_once(
    "docs/en/guides/statgpu_benchmark_dashboard.md",
    "npx playwright install --with-deps chromium\nnpm run test:e2e",
    "npx playwright install --with-deps chromium firefox webkit\nnpm run test:e2e\nnpm run test:e2e:production",
)
replace_once(
    "docs/en/guides/statgpu_benchmark_dashboard.md",
    "The responsive layout keeps paired charts on large screens and stacks them below 1080 px. Summary cards collapse from six to three columns below 1450 px.",
    "The responsive layout keeps paired charts on large screens and stacks them below 1080 px. Summary cards collapse from six to three columns below 1450 px. Charts also expose filter-synchronized exact-value tables with full labels, while primary filters, scale chips, sorting, and metric-panel disclosure support keyboard navigation with visible focus. The production QA suite serves the committed `docs/assets/benchmarks/` path and exercises Chromium, Firefox, and WebKit.",
)

cn = Path("docs/cn/guides/benchmarks.md")
if cn.exists():
    text = cn.read_text(encoding="utf-8")
    text = text.replace("> 最后更新：2026-07-20", "> 最后更新：2026-08-07", 1)
    text = text.replace(
        "当前 canonical dashboard 仅注册 **2026-06-01 或之后** 的 benchmark source，目前包含 **8 个 registered source、1,774 条 normalized run 和 36 个 model**。",
        "当前 canonical dashboard 仅注册 **2026-06-01 或之后** 的 benchmark source。实时 source/run 数量以 dashboard inventory 为准，不在长期文档中维护固定计数。**2026-08-07 快照：**9 个 source 均已 registered/available/parsed，共生成 1,796 条 normalized run。",
        1,
    )
    text = text.replace(
        "npx playwright install --with-deps chromium\nnpm run test:e2e",
        "npx playwright install --with-deps chromium firefox webkit\nnpm run test:e2e\nnpm run test:e2e:production",
        1,
    )
    cn.write_text(text, encoding="utf-8")

for path in ("docs/en/guides/benchmarks.md", "docs/cn/guides/benchmarks.md"):
    target = Path(path)
    if target.exists():
        text = target.read_text(encoding="utf-8")
        text = text.replace("> Last updated: 2026-08-07  \n", "> Last updated: 2026-08-07\n")
        text = text.replace("> 最后更新：2026-08-07  \n", "> 最后更新：2026-08-07\n")
        target.write_text(text, encoding="utf-8")
