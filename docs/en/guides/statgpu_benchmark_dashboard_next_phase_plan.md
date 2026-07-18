# statgpu Benchmark Dashboard — Remaining Work After PR #78

## Status

The benchmark dashboard data-pipeline expansion and frontend modularization are implemented in PR #78.

Completed scope includes:

- Schema v1.1.0 and runtime version checks.
- Manifest-driven canonical source registration.
- Eight canonical benchmark sources and the registered parser implementations.
- Source SHA256 verification and strict issue handling.
- Canonical run identity and speedup-reference resolution.
- Vite + TypeScript + ECharts frontend.
- Modular chart, component, state, identity, and panel code.
- Timing and speedup charts.
- Validation, accuracy, inference, prediction, convergence, and selection panels.
- Progressive filters and data-driven default state.
- Python, TypeScript, build, staleness, and Playwright checks.
- Current frontend, user, schema, parser, and aggregation documentation.

This document now lists only remaining product and maintenance work. It is no longer an implementation plan for features already delivered.

## Integration readiness

PR #78 is suitable for integration when:

1. All required CI checks remain green on the final head commit.
2. The deployed build loads from `docs/assets/benchmarks/index.html`.
3. A final manual browser smoke test finds no blocking issue.
4. Generated data and deployment assets are current.
5. No unresolved review thread or known correctness defect remains.

The dashboard should not be merged solely because the frontend compiles. The manual smoke test should verify both data semantics and interaction behavior.

## Final manual QA

### Page loading

Test the production build rather than only the Vite development server:

```bash
cd frontend
npm ci
npm run build

cd ../docs/assets/benchmarks
python -m http.server 8000
```

Verify:

- No console error.
- All three JSON files load.
- The header metadata is consistent with the generated files.
- Static assets work from the nested `docs/assets/benchmarks/` path.
- Refreshing the page does not break relative asset URLs.

### Default state

Verify that the initial page:

- Selects an environment that actually has runs.
- Prefers `remote-p100` only when populated.
- Prefers `penalized_glm` only when available in the chosen environment.
- Does not open with a valid-but-empty view.

### Filter cascade

Exercise:

```text
Category → Model → Variant → Penalty → Solver → Scale → Backend → External framework
```

Verify:

- Each option list is derived from the relevant upstream context.
- Changing an upstream selection clears incompatible downstream state.
- Scale remains multi-select.
- Backend filtering affects statgpu only.
- External frameworks are hidden by default.
- Only context-relevant external framework checkboxes appear.
- Empty result states remain understandable.

### Timing chart

Verify:

- Framework/backend/implementation series are distinct.
- CoxPH variants and tie regimes are not merged.
- Missing values do not shift bars into another group.
- Tooltips agree with table values.
- Large filtered result sets respect the chart group limit.

### Speedup chart

Verify:

- Computed and runner-reported speedups are distinguishable.
- Labels include enough model, variant, solver, framework/backend, implementation, and scale context.
- A value above one means faster than the reference.
- A value below one is visible as a slowdown.
- Reference framework/backend text is correct.
- The speedup chart respects `speedupChartLimit`.

### Overview table

Verify:

- Every visible column sorts correctly.
- Repeated equal sort values use stable run-ID tie-breaking.
- The default limit is 200.
- “Show all” displays the complete filtered set.
- External framework rows display framework identity instead of a null backend.

### Domain panels

For filters that produce relevant metrics, verify:

- Validation status and checks.
- Accuracy fields and references.
- Inference metrics where present.
- Prediction fields, including C-index where present.
- Convergence summaries.
- Selection metrics and target FDR.
- Panel row limits and expansion behavior.

For filters with no relevant metric group, verify that the corresponding panel is omitted rather than shown as a broken empty card.

## Cross-browser and accessibility pass

The automated browser suite currently uses Chromium. Before treating the dashboard as a polished public documentation surface, perform at least a lightweight pass in:

- Chromium/Chrome.
- Firefox.
- Safari or WebKit when available.

Accessibility checks should cover:

- Keyboard reachability of selects, radios, checkboxes, scale chips, sortable headers, and panel controls.
- Visible focus indication.
- Meaningful labels for interactive controls.
- Sufficient chart/table contrast.
- Screen-reader-readable summaries for chart content, or an equivalent accessible table path.

These checks are product-quality improvements; they do not alter the benchmark schema.

## Documentation integration

The following documentation now exists:

- `frontend/README.md`
- `docs/en/guides/statgpu_benchmark_dashboard.md`
- `docs/benchmark-dashboard/schema-v1.1.md`
- `docs/benchmark-dashboard/parser-contracts.md`
- `docs/benchmark-dashboard/aggregation-contract.md`
- `docs/benchmark-dashboard/rollout-plan.md`
- `docs/benchmark-dashboard/penalized-robust-quantile-plan.md`

Remaining documentation work:

- Add the user guide to the documentation navigation if the repository uses an explicit nav configuration.
- Add screenshots only after the final visual layout stabilizes.
- Keep source/run counts out of long-lived prose unless they are explicitly dated or generated.
- Update contract documentation in the same PR as any schema, identity, parser, or aggregation change.

## Recommended future enhancements

### Penalized robust and quantile integration

A dedicated implementation plan is recorded in:

```text
docs/benchmark-dashboard/penalized-robust-quantile-plan.md
```

The selected design is:

- keep the existing `robust_quantile` category;
- add penalty as a first-class method dimension;
- generate a new P100 benchmark source rather than relabeling unpenalized rows;
- begin with Quantile `q=0.5` and Huber/MAD;
- cover `none`, L1, L2, ElasticNet, SCAD, and MCP;
- use four representative scales, three backends, and Auto/resolved solver identity;
- preserve `none` in Focused mode as the direct baseline;
- defer group/adaptive penalties, additional loss variants, and an exhaustive solver matrix.

This staged core matrix was selected over both a minimal two-case benchmark and an exhaustive cross-product. It provides enough coverage to exercise smooth, nonsmooth, and nonconvex solver paths while remaining interpretable and feasible to validate.

The plan is not complete until a real June-or-later P100 artifact is generated, registered, parsed, and tested. Frontend-only relabeling is explicitly prohibited.

### URL-persisted state

Encode selected environment, categories, model, variant, penalty, solver, scales, backend, and external frameworks in the URL.

Benefits:

- Shareable benchmark views.
- Browser back/forward behavior.
- Reproducible links from issues and documentation.

Any URL-state implementation must validate values against the currently loaded data and drop stale selections safely.

### Responsive layout

Improve behavior on narrow screens:

- Collapsible sidebar.
- Wrapping filter groups.
- Horizontally scrollable tables.
- Chart label truncation with full tooltip text.
- Summary cards that stack cleanly.

### Expanded E2E coverage

Current state tests protect default selection behavior. Future interaction tests should cover:

- Full filter cascade.
- External framework visibility.
- Sorting and table limits.
- Panel conditional rendering.
- Computed vs reported speedup labels.
- Production build loading from the nested base path.

Prefer a small deterministic fixture over coupling every browser test to the complete production dataset.

### Additional benchmark domains

New benchmark families should be added only when all of the following are available:

- Canonical source JSON.
- Manifest registration and SHA256.
- Parser implementation or documented parser reuse.
- Stable case/method identity.
- Metric provenance.
- Parser and contract tests.
- A meaningful frontend presentation path.

Do not add a source by hardcoding it directly into the frontend.

### Performance monitoring

As run count grows, monitor:

- JSON bundle size.
- Parse and initial-render time.
- Filter update time.
- Chart group count.
- Table row count.

Possible future measures include lazy panels, precomputed indexes, virtualized tables, or bundle partitioning. These should be justified by measurements rather than introduced preemptively.

## Maintenance rules

- Keep JSON Schema and TypeScript types synchronized.
- Treat run identity changes as schema-contract changes.
- Never silently accept an unsupported schema version.
- Do not use zero as a placeholder for unavailable dispersion.
- Do not pool across sources merely because they share a comparison ID.
- Keep generated data and deployed assets in the same change as source/parser updates.
- Preserve deterministic generation.
- Keep CI green after each contract-affecting change.

## Verification commands

```bash
python -m pip install -U pytest jsonschema

pytest dev/tests/test_benchmark_frontend_data.py \
       dev/tests/test_frontend_contracts.py \
       dev/tests/test_frontend_domain_coverage.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

cd frontend
npm ci
npm run typecheck
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

For staleness verification, regenerate the deterministic bundle, rebuild, and confirm:

```bash
git status --porcelain -- frontend/public/data docs/assets/benchmarks
```

The command must produce no output.
