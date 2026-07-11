# statgpu Benchmark Dashboard — Next-Phase Development Plan

## 1. Current Status

The benchmark dashboard has completed its Phase 0–2 MVP implementation in PR #76.

The current implementation includes:

- A unified benchmark data schema.
- A Python data generator with parsers for existing benchmark JSON files.
- Committed generated benchmark data and parse report.
- A Vite + TypeScript + ECharts frontend.
- Category, model, penalty, solver, scale, backend, and external framework filters.
- Timing and speedup charts.
- Overview table with sorting and pagination.
- CI for parser tests, frontend build, and generated artifact staleness checks.

The current PR is functionally usable, but before merging into `master`, the dashboard should undergo another round of frontend QA, UI polish, component refactoring, and interaction-level testing.

---

## 2. Branching Strategy

Before merging the dashboard into `master`, use the current dashboard branch as an integration branch.

Current integration branch:

```text
feature/benchmark-frontend-dashboard
```

Future frontend work should be developed in smaller branches based on this integration branch, not based directly on `master`.

Recommended branch structure:

```text
master
  ↑
feature/benchmark-frontend-dashboard
  ↑
frontend-dashboard-qa
frontend-dashboard-ui-polish
frontend-dashboard-component-split
frontend-dashboard-e2e-tests
frontend-dashboard-data-coverage
```

For each follow-up task:

```bash
git fetch origin
git checkout feature/benchmark-frontend-dashboard
git pull origin feature/benchmark-frontend-dashboard

git checkout -b frontend-dashboard-qa
```

Then open a PR with:

```text
base: feature/benchmark-frontend-dashboard
compare: frontend-dashboard-qa
```

Do not set the base to `master` until the final dashboard integration PR is ready.

---

## 3. Merge Policy

The dashboard should be merged into `master` only after the following conditions are satisfied:

```text
1. All CI workflows pass.
2. The dashboard loads correctly from docs/assets/benchmarks/index.html.
3. All main filters work under common and edge-case combinations.
4. Timing and speedup charts show aligned and interpretable data.
5. Generated data is reproducible under CI staleness checks.
6. The UI is acceptable for public-facing documentation.
7. At least one manual QA pass has been completed.
8. No known correctness bug remains in parser, schema, or chart rendering.
```

Until then, keep PR #76 open or keep the integration branch alive as a staging branch.

---

# Phase 2.5 — Manual QA and Bug Fixes

## Goal

Confirm that the current MVP is correct, usable, and free from obvious frontend interaction bugs.

## Branch

```text
frontend-dashboard-qa
```

Base branch:

```text
feature/benchmark-frontend-dashboard
```

## Scope

This phase should focus on manual testing and targeted bug fixes. Avoid large refactoring unless necessary.

## QA Checklist

### 1. Page Loading

Test both development and production modes.

```bash
cd frontend
npm ci
npm run dev
```

Then test production build:

```bash
cd frontend
npm run build

cd ../docs/assets/benchmarks
python -m http.server 8000
```

Verify:

```text
- Dashboard loads without console errors.
- benchmark_data.json loads correctly.
- parse_report.json loads correctly.
- Header shows expected run count and parsed file count.
- Assets work under nested docs/assets/benchmarks/ path.
```

### 2. Category Filter

Test:

```text
- Default selected category.
- Select all categories.
- Clear all categories.
- Select one category at a time.
- Select overlapping categories, especially penalized_glm and glm.
```

Expected behavior:

```text
- Runs are filtered correctly.
- Multi-category runs do not appear duplicated.
- Empty states are handled gracefully.
```

### 3. Progressive Filters

Test:

```text
Model → Penalty → Solver → Scale → Backend → External framework
```

Verify:

```text
- Model selector updates after category changes.
- Penalty selector updates after model changes.
- Solver selector updates after penalty changes.
- Solver selection resets when model or penalty changes.
- Scale chips remain multi-selectable.
- Backend radio affects only statgpu backends.
- External framework checkboxes control sklearn/glmnet/statsmodels visibility.
```

### 4. Timing Chart

Test under:

```text
- All statgpu backends.
- Only numpy.
- Only cupy.
- Only torch.
- With sklearn enabled.
- With glmnet enabled.
- Single model + single penalty.
- Multiple scales selected.
```

Verify:

```text
- x-axis labels match the displayed bars.
- NumPy/CuPy/Torch bars align to the correct model-scale group.
- External frameworks do not appear unless explicitly enabled.
- Missing values do not shift bars into incorrect categories.
- Tooltip values match table values.
```

### 5. Speedup Chart

Verify:

```text
- NumPy baseline rows are excluded.
- Speedup greater than 1 means faster than reference.
- Slowdowns below 1 are visible.
- Auto solver and manual solver rows are distinguishable.
- Reported speedups from solver benchmark are not mixed up with computed timing speedups.
```

### 6. Overview Table

Test:

```text
- Sorting by model.
- Sorting by penalty.
- Sorting by solver.
- Sorting by backend/framework.
- Sorting by time.
- Sorting by speedup.
- Show all.
- Show first 200.
```

Verify:

```text
- Sort state persists after re-render.
- Show all actually displays all rows.
- Table count matches filtered result count.
- External framework rows display framework name instead of null backend.
```

### 7. Accuracy Panel

If accuracy metrics exist:

```text
- Confirm panel opens and closes.
- Confirm PASS/WARN/FAIL thresholds are reasonable.
- Confirm references are displayed correctly.
```

If no accuracy metrics exist:

```text
- Confirm no broken empty panel appears.
```

## Deliverables

```text
- A completed manual QA checklist.
- Bug-fix commits for any discovered issues.
- Updated README if local testing instructions change.
```

## Acceptance Criteria

```text
- All CI passes.
- Manual QA checklist completed.
- No known chart alignment, filtering, or pagination bug remains.
```

---

# Phase 3 — Component Refactoring

## Goal

Reduce frontend technical debt by splitting the current monolithic `main.ts` into smaller maintainable modules.

## Branch

```text
frontend-dashboard-component-split
```

Base branch:

```text
feature/benchmark-frontend-dashboard
```

## Motivation

The current `main.ts` is acceptable for MVP, but it mixes:

```text
- DOM helpers
- global state
- filter rendering
- chart rendering
- table rendering
- accuracy panel rendering
- update loop
```

This will become difficult to maintain once new charts, URL state, mobile layout, or more benchmark categories are added.

## Proposed File Structure

```text
frontend/src/
  main.ts
  schema.ts
  data.ts
  state.ts

  components/
    Header.ts
    Sidebar.ts
    FilterBar.ts
    OverviewTable.ts
    AccuracyPanel.ts
    EmptyState.ts

  charts/
    TimingChart.ts
    SpeedupChart.ts

  utils/
    dom.ts
    format.ts
    grouping.ts
```

## Refactoring Steps

### Step 1 — Extract DOM Utilities

Move helper functions such as:

```text
h()
clear()
formatTime()
formatSpeedup()
```

into:

```text
frontend/src/utils/dom.ts
frontend/src/utils/format.ts
```

### Step 2 — Extract State

Move `AppState`, `createDefaultState`, and update-related state helpers into:

```text
frontend/src/state.ts
```

Keep `filterRuns()` in `data.ts` unless it becomes large.

### Step 3 — Extract Filter Components

Move filter rendering into:

```text
frontend/src/components/FilterBar.ts
frontend/src/components/Sidebar.ts
```

### Step 4 — Extract Charts

Move chart logic into:

```text
frontend/src/charts/TimingChart.ts
frontend/src/charts/SpeedupChart.ts
```

Each chart module should expose:

```ts
export function renderTimingChart(el: HTMLElement, runs: Run[], state: AppState): void
export function renderSpeedupChart(el: HTMLElement, runs: Run[], state: AppState): void
```

### Step 5 — Extract Table

Move table rendering into:

```text
frontend/src/components/OverviewTable.ts
```

## Rules During Refactor

```text
- No schema changes.
- No data generator changes.
- No visual redesign unless required.
- Keep CI green after every commit.
- Preserve current behavior before adding new features.
```

## Acceptance Criteria

```text
- frontend/src/main.ts becomes a thin orchestration layer.
- npm run typecheck passes.
- npm run build passes.
- Manual smoke test confirms no regression in filters, charts, or table.
```

---

# Phase 4 — UI Polish and Public Presentation

## Goal

Improve the dashboard from an MVP-style internal tool into a more polished public-facing documentation page.

## Branch

```text
frontend-dashboard-ui-polish
```

Base branch:

```text
feature/benchmark-frontend-dashboard
```

## Scope

### 1. Layout Polish

Improve:

```text
- Header spacing.
- Sidebar readability.
- Filter bar density.
- Chart card spacing.
- Table scrolling.
- Empty state display.
```

Suggested layout:

```text
Header
Sidebar + Main Content
  Filter bar
  Summary cards
  Chart grid
  Overview table
```

### 2. Summary Cards

Add compact summary cards:

```text
Total runs
Parsed files
Model categories
Fastest GPU speedup
External frameworks available
Latest generated timestamp
```

### 3. Better Empty States

For empty filtered results, show:

```text
No runs match the current filters.
Try clearing scale, solver, or external framework filters.
```

For missing chart data:

```text
No timing data is available for this filtered view.
```

### 4. Better Speedup Labels

Differentiate:

```text
computed speedup
reported solver benchmark speedup
external comparison
```

Possible display:

```text
3.2× computed vs NumPy
2.9× reported by solver benchmark
```

### 5. Documentation Links

Add links from dashboard to:

```text
- Benchmark guide.
- Raw benchmark_data.json.
- parse_report.json.
- GitHub source directory.
```

## Acceptance Criteria

```text
- Dashboard is visually acceptable as a documentation page.
- Empty states are clear.
- Users can understand what each chart means without reading the source code.
- No UI regression in existing filters and charts.
```

---

# Phase 5 — Frontend Testing

## Goal

Add minimal automated frontend tests so future changes do not silently break key dashboard interactions.

## Branch

```text
frontend-dashboard-e2e-tests
```

Base branch:

```text
feature/benchmark-frontend-dashboard
```

## Recommended Tool

Use Playwright.

Install:

```bash
cd frontend
npm install -D @playwright/test
npx playwright install --with-deps
```

Add scripts:

```json
{
  "scripts": {
    "test:e2e": "playwright test"
  }
}
```

## Test Cases

### 1. Page Loads

```text
- Dashboard loads.
- Header appears.
- Category sidebar appears.
- Timing chart container appears.
- Speedup chart container appears.
- Overview table appears.
```

### 2. Category Filtering

```text
- Clear all categories.
- Select penalized_glm.
- Confirm table has rows.
```

### 3. Model / Penalty Filtering

```text
- Select a model.
- Confirm penalty selector appears.
- Select a penalty.
- Confirm result count updates.
```

### 4. Scale Multi-select

```text
- Select one scale chip.
- Confirm other scale chips remain visible.
- Select a second scale chip.
- Confirm both are active.
```

### 5. Show All

```text
- If filtered rows > 200, click Show all.
- Confirm displayed count changes.
- Click Show first 200.
- Confirm table returns to paginated view.
```

### 6. External Framework Toggle

```text
- Confirm external rows hidden by default.
- Enable glmnet.
- Confirm glmnet rows appear.
- Disable glmnet.
- Confirm glmnet rows disappear.
```

## CI Integration

Add to `.github/workflows/benchmark-frontend.yml`:

```yaml
frontend-e2e:
  runs-on: ubuntu-latest
  defaults:
    run:
      working-directory: frontend
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with:
        node-version: '20'
    - run: npm ci
    - run: npx playwright install --with-deps
    - run: npm run build
    - run: npm run test:e2e
```

## Acceptance Criteria

```text
- Playwright tests pass locally.
- Playwright tests pass in CI.
- Core dashboard interactions are protected.
```

---

# Phase 6 — Data Pipeline Expansion

## Goal

Extend benchmark coverage beyond the current Phase 0–2 parsers while preserving schema stability.

## Branch

```text
frontend-dashboard-data-coverage
```

Base branch:

```text
feature/benchmark-frontend-dashboard
```

## Candidate Parser Additions

Add parsers incrementally. Suggested order:

```text
1. inference validation benchmarks
2. unsupervised benchmarks
3. Cox / survival benchmarks
4. robust / quantile benchmarks
5. nonparametric benchmarks
6. panel / ANOVA / covariance / feature selection benchmarks
```

## Parser Development Rules

For every new parser:

```text
- Add one parser function.
- Add parser registry entry.
- Add parser-specific test.
- Add sample expected metric check.
- Update parse_report expectations if necessary.
- Regenerate benchmark_data.json and parse_report.json.
- Confirm frontend still loads.
```

## Metric Rules

Only scalar summary metrics should enter `benchmark_data.json`.

Allowed examples:

```text
fit_time_ms
std_ms
min_ms
max_ms
speedup
coef_l2_diff
coef_max_abs_diff
bse_max_abs_diff
n_iter
converged
p_value
fdr
power
selected_count
```

Avoid large arrays:

```text
coef vectors
bse vectors
raw predictions
full bootstrap samples
selected feature index arrays
large per-iteration traces
```

## Acceptance Criteria

```text
- New parser passes tests.
- Generated JSON remains reasonably sized.
- No fake placeholder metrics are introduced.
- Missing metric groups remain absent rather than null.
- Dashboard handles new categories without UI breakage.
```

---

# Phase 7 — Final Integration into master

## Goal

Merge the benchmark dashboard into `master` only after it is stable, reviewed, and suitable for public documentation.

## Pre-merge Checklist

```text
- All PRs into feature/benchmark-frontend-dashboard are merged.
- All CI workflows pass.
- Manual QA checklist is complete.
- Dashboard loads from docs/assets/benchmarks/index.html.
- Documentation links are correct.
- PR body run count and feature list are up to date.
- Old outdated review threads are resolved.
- Generated data and build artifacts are current.
```

## Final Merge Path

After all follow-up branches are merged into:

```text
feature/benchmark-frontend-dashboard
```

merge PR #76 into:

```text
master
```

Final merge should ideally be squash or merge commit depending on project preference.

Recommended final PR title:

```text
feat: add benchmark frontend dashboard
```

Recommended final PR description should include:

```text
- Data pipeline summary.
- Frontend feature summary.
- CI checks.
- Manual QA summary.
- Known limitations.
- Future parser expansion plan.
```

---

# Suggested Immediate Next Steps

The most practical next three tasks are:

```text
1. Open frontend-dashboard-qa from feature/benchmark-frontend-dashboard.
2. Complete manual QA and fix any interaction bugs.
3. Open frontend-dashboard-component-split to reduce main.ts complexity.
```

After those two branches are merged into the integration branch, decide whether UI polish and Playwright tests are required before merging to `master`.

For a public-facing benchmark page, the recommended minimum before merging to `master` is:

```text
Phase 2.5 Manual QA
Phase 3 Component Split
Phase 4 Basic UI Polish
```

Playwright tests are strongly recommended, but they can be added either before the final merge or immediately after, depending on release urgency.
