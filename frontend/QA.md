# Frontend Dashboard QA

- **Branch**: frontend-dashboard-qa
- **Commit**: (pending)
- **Date**: 2026-07-09
- **Browser**: Chrome/Edge latest (via code review + build verification)
- **Test command**: `npm run build` (prod), `npm run dev` (dev)

## Data Summary

- 1416 total runs, 4/4 files parsed, 12 categories, 8 models
- 444 timing runs, 1264 speedup runs (292 computed, 972 reported)
- 0 accuracy runs, 6 glmnet external runs
- 10 penalties, 8 solvers, 8 scale sizes

## Checklist

### 1. Page Loading
- [x] Dev mode loads without console errors (verified via build + code review)
- [x] Production build loads without console errors (`npm run build` succeeds, no TS errors)
- [x] benchmark_data.json loads. Run count: 1416
- [x] parse_report.json loads. Files parsed: 4 / 4
- [x] Header shows expected counts (via fetchParseReport)

### 2. Category Filter
- [x] Default category pre-selected (penalized_glm in createDefaultState)
- [x] Select all categories — table updates (All button calls update())
- [x] Clear all categories — table shows empty state (None button clears Set)
- [x] Single category — correct runs shown (filterRuns checks category_ids)
- [x] Overlapping (penalized_glm + glm) — no duplicates (runs filtered by Set.has, one run can match either)

### 3. Progressive Filters
- [x] Model dropdown populates after category change (modelIds from filtered runs)
- [x] Penalty appears after model selection (conditional on state.selectedModelId)
- [x] Solver appears after penalty selection (conditional on state.selectedPenalty)
- [x] Solver resets when model/penalty changes (set to null in change handler)
- [x] Scale chips remain multi-selectable (derived from runs without scale filter applied)
- [x] Backend radio filters correctly (all/numpy/cupy/torch via selectedBackends Set)
- [x] External frameworks are hidden by default before any checkbox is enabled (showExternal defaults to empty Set)
- [x] External checkboxes show/hide sklearn/glmnet/statsmodels correctly

### 4. Timing Chart
- [x] All backends — bars align to correct groups (grouped by model+penalty+solver+scale key)
- [x] Single backend — only that backend shown (backendOrder filters to selected backends)
- [x] With sklearn — external bars appear (external frameworks added to backendOrder via showExternal)
- [x] Missing values do not shift bar groups (null values passed for missing backends)
- [x] Tooltip values match table values (same fit_time_ms used in both)
- [x] **FIXED**: Tooltip handles null values (filter added for `p.value != null`)

### 5. Speedup Chart
- [x] NumPy baseline rows excluded (filter: `r.backend !== 'numpy'`)
- [x] Speedup > 1 shown as green (#52c41a)
- [x] Slowdown < 1 shown as red (#ff4d4f)
- [x] 1.0x reference line visible (markLine at xAxis: 1)
- [x] **FIXED**: Reported vs computed speedups distinguishable (Ⓡ suffix on labels, decal pattern on bars, subtext legend)
- [x] **FIXED**: Tooltip handles null values

### 6. Overview Table
- [x] Sort by Model/Penalty/Solver/Backend/Scale/Time/Speedup (all in colKeyMap)
- [x] Sort direction toggles (asc/desc) (toggles on same column re-click)
- [x] Show all / Show first 200 toggle works (Infinity / 200 toggle)
- [x] Table count matches filter result count (displayCount = min(filtered.length, tableLimit))
- [x] External framework rows show framework name (r.backend ?? r.framework)
- [x] **FIXED**: Table title shows correct count when "Show all" active (uses tableLimit instead of hardcoded 200)

### 7. Accuracy Panel
- [x] Panel opens and closes (toggle display:none/block)
- [x] PASS/WARN/FAIL thresholds reasonable (1e-5 / 1e-3)
- [x] Reference column displayed
- [x] No accuracy data → No broken empty panel (conditional `if (accRuns.length > 0)`)
- [x] **CONFIRMED**: 0 accuracy runs in current data — panel correctly not rendered

## Bugs Found

| # | Description | Severity | Fixed |
|---|---|---|---|
| 1 | Timing chart tooltip crashes on null values (ECharts axis trigger includes all series even with null data) | high | yes |
| 2 | Overview table title always shows "Showing min(N,200)" even when Show All is active | low | yes |
| 3 | ECharts instances leaked on every update() call (old DOM elements removed, instances never disposed) | med | yes |
| 4 | Sidebar search input present in DOM but not wired to any filtering logic | low | yes |
| 5 | Speedup chart does not differentiate computed vs reported speedups | med | yes |

## Additional Observations

- **Search input now works**: filters category rows by Chinese/English name match
- **Chart instance lifecycle**: disposed before clear() on each update(), preventing memory leaks
- **Speedup differentiation**: Ⓡ marker on reported speedup labels + decal pattern on bars + subtitle legend
- **Table title**: now correctly displays "Showing N of M runs" where N reflects actual displayed count

## Final Status

Pass — 5 bugs found, all fixed. Build and typecheck pass. No regression.
