# Benchmark Dashboard Next Phase — Implementation Plan

> **Stable reference documents** (`docs/benchmark-dashboard/`, committed before use):
> - `schema-v1.1.md` — JSON Schema + TypeScript types + parity fixture spec
> - `parser-contracts.md` — per-domain ParserContract + field mappings + CANONICAL_PARAMETERS registry
> - `aggregation-contract.md` — FieldAggregationPolicy, pooling rules, merge functions
> - `rollout-plan.md` — this document (committed in A0; updated each PR)

## Context

472 JSON files in `results/`, **4 parsed**. Frontend: `InferenceMetric`, `ConvergenceMetric`, top-level `Quality`, `AccuracyMetric.bse_max_abs_diff` never rendered.

---

## Non-Functional Budget

| Metric | Threshold | CI | Local |
|--------|-----------|-----|-------|
| `benchmark_data.json` | < 10 MB | Assert | Same |
| Initial table rows | ≤ 200 | Assert | Same |
| `dashboard-setup` measure | < 1 s | Not enforced | Median of 3, Chrome, i7, ≤3,000 runs |

---

## Architectural Decisions

1. **Schema v1.1** — single JSON Schema accepting both transitional and canonical IDs via `anyOf`. `type: "string"` on all pattern properties. `additionalProperties: false`. Phase-specific semantic validator (`mode="transitional"|"canonical"`).
2. **Conditional timing deps (transitional)** — JSON Schema allows `std_ms` without `sample_count`/`std_ddof`/`std_scope` in transitional mode. Canonical semantic validator requires all three when `std_ms` present. Legacy `std_ms=0` (fallback for missing source field) → omit `std_ms` entirely.
3. **`generation_id`** — SHA256 of entire 3-file bundle (each stripped of `generation_id` before hashing). 64-char lowercase hex. Frontend validates per-file independently; mismatch → only that metadata hidden; dashboard core always renders.
4. **ParserContract** — `parameter_aliases`, `identity_parameter_names`, `method_parameter_names`, `replicate_parameter_names`, `public_parameter_names`, `ui_parameter_names`, `ignored_parameter_names` (raw keys), `field_aggregation_policy: Mapping[str, Mapping[str, FieldAggregationPolicy]]`. Canonical-name sets (`identity`/`method`/`replicate`/`public`) ⊆ `CANONICAL_PARAMETERS`. `ui ⊆ public`. `parameter_aliases` target ⊆ `CANONICAL_PARAMETERS`.
5. **CANONICAL_PARAMETERS** — global registry of `ParameterSpec(name, normalizer, unit?, validator?)`. Defines type coercion, unit, and validation for every canonical parameter name. `identity ∩ method = ∅`, `identity ∩ replicate = ∅`, `method ∩ replicate = ∅`.
6. **RunIdentity** — `(source_id, case_id, method_config_id, env_id, model_id, variant, implementation, loss, penalty, solver, framework, backend, scale_key)`.
7. **`comparison_id`** — A1b: uses A0-legacy mapping (cross-file ElasticNet sources share `"transitional:elasticnet-cross-framework"`). A2+: manifest. Same comparison → same `env_id` enforced.
8. **FieldAggregationPolicy** — two-level: `metric_group → field → FieldAggregationPolicy(mode, std_field?, count_field?)`. Modes: `mean_std`, `pooled`, `identical`, `strictest`, `recompute`, `min`, `max`, `provenance`. Coverage validated.
9. **Pooled timing** — `TimingSummary(std_ddof, std_scope, sample_count)`. Pooling only when consistent across inputs. `replicate.n_runs` = independent replicates (excl. exact duplicates). `timing.sample_count` = raw timing observations.
10. **ID uniqueness** — `run_id` unique. Registry IDs unique. `case_id`/`method_config_id`/`source_id`: hash injectivity (not global uniqueness).
11. **TimingCellIdentity** — `(comparison_id, benchmark_session_id?, env_id, model_id, case_id, method_config_id, variant, implementation, loss, penalty, solver, scale_key, framework, backend)`.
12. **Metric quality** = provenance merge. **Quality.status** = ordinal merge. Separate functions.
13. **Source hash** = SHA256 of UTF-8 bytes after `\r\n→\n`. Reject BOM/non-UTF-8/bare `\r`. `.gitattributes text eol=lf` in A0.
14. **Generator transactional** — in-memory → validate → temp files → `os.replace` (report→inventory→data). `--deterministic`.
15. **JSON Schema** — `Draft202012Validator`+`FormatChecker`. No fallback. Parity fixture.
16. **Chart render epoch** — `requestAnimationFrame`+`isConnected`. Epoch-specific marks. `escapeHtml`. Registry colors.
17. **State actions** — no direct mutation. `createDefaultState(data)`.
18. **Keyed columns** — `sortValue: string|number|null`; null→last; tie-break `run_id`.
19. **E2E** — deterministic fixture (unconditional). Production smoke. No conditional skips.
20. **CLI mutual exclusion** — A1b+: `--out`/`--report`/`--inventory-out` all-or-none.
21. **Transitional inventory computed**, not hardcoded.
22. **Transitional framework registry** — A1b fixed registry; only entries referenced by runs output; stable sort.
23. **Stable documents** committed before use.
24. **`typecheck`** includes `typecheck:src && typecheck:e2e && typecheck:contract`.

---

## Implementation Order

| PR | Scope | Days |
|----|-------|------|
| **A0** | Audit + .gitattributes + rollout-plan.md + legacy comparison groups | 1 |
| **A1a** | Package split + .gitignore + CI guard (phase-specific) | 1 |
| **A1b** | Schema v1.1 + transitional CLI + transitional inventory + transitional framework/comparison registries + frontend contract | 2 |
| **A2** | Manifest + canonical migration + strict mode + preservation baseline | 4–5 |
| **B** | CoxPH efron + ValidationPanel + context-aware external + production e2e | 3 |
| **C** | Comprehensive validation + Cox package + AccuracyPanel + PredictionPanel | 3 |
| **D** | LassoCV + ConvergencePanel + chart overflow | 3 |
| **E** | Knockoff + SelectionPanel | 3 |
| **F–I** | Remaining domains | 1-2 each |

**Core (A0–E): 21–23 days.**

---

## Phase A0: Audit + .gitattributes + Rollout Plan (1 day)

### Scope
- Add `.gitattributes`: `text eol=lf` for canonical JSON paths
- Extend current CLI with `--update-preflight-baseline`
- Record: run_count=1416, warning_count=0, source SHA256 (`\r\n→\n` normalization), baseline git_sha, convergence provenance, timing provenance, legacy discriminators, `catalog_total: 472`, **legacy comparison groups** (which sources share a comparison)
- Run `--deterministic`; record `legacy_deterministic_output_sha256.json`
- Commit `rollout-plan.md`, `legacy_identity_audit.json`

### Audit fixture
```json
{
  "baseline_git_sha": "3a070d1...", "catalog_total": 472,
  "run_count": 1416, "warning_count": 0,
  "duplicate_run_ids": [], "duplicate_transitional_identities": [],
  "source_sha256": {"penalized_glm_bench_perf_2026-06-22.json": "abc...", ...},
  "convergence_provenance": {"explicit_converged": 0, "parser_inferred_converged": 24},
  "timing_provenance": {
    "penalized_glm_perf": {"sample_count_known": false, "std_ddof": null, "std_scope": "unknown"},
    "elasticnet_statgpu": {"sample_count_known": false, "std_ddof": null, "std_scope": "unknown"},
    "elasticnet_glmnet": {"sample_count_known": false, "std_ddof": null, "std_scope": "unknown"}
  },
  "legacy_discriminators": {
    "penalized_glm": ["scale_name", "model_key"],
    "glm_solver": ["scale_name", "model_key"],
    "elasticnet_statgpu": ["entry_name"],
    "elasticnet_glmnet": ["dataset_name"]
  },
  "legacy_comparison_groups": {
    "penalized_glm_bench_perf_2026-06-22.json": "transitional:penalized-glm-performance",
    "glm_solver_benchmark_2026-06-23.json": "transitional:glm-solver",
    "benchmark_full/benchmark_statgpu_all.json": "transitional:elasticnet-cross-framework",
    "benchmark_full/benchmark_glmnet_all.json": "transitional:elasticnet-cross-framework"
  }
}
```

### Source hash (A0 + A2)
```python
def normalize_utf8_bytes(raw: bytes) -> bytes:
    if raw.startswith(b'\xef\xbb\xbf'): raise ValueError("BOM")
    text = raw.decode("utf-8")
    if '\r' in text.replace('\r\n', ''): raise ValueError("bare CR")
    return text.replace('\r\n', '\n').encode("utf-8")
def source_sha256(path: Path) -> str:
    return hashlib.sha256(normalize_utf8_bytes(path.read_bytes())).hexdigest()
```

### Verification (A0)
```bash
pytest dev/tests/test_benchmark_frontend_data.py -v
python dev/benchmarks/generate_benchmark_data.py --check
```

---

## Phase A1a: Package Split (1 day)

### .gitignore (at EOF)
```gitignore
# === Benchmark dashboard tracked sources ===
!dev/benchmarks/
!dev/benchmarks/frontend_data/
!dev/benchmarks/frontend_data/**
!dev/benchmarks/frontend_sources.json
!dev/benchmarks/benchmark_source_catalog.json

!dev/tests/test_benchmark_frontend_data.py
!dev/tests/test_frontend_parsers.py
!dev/tests/fixtures/benchmark_frontend/
!dev/tests/fixtures/benchmark_frontend/**

!results/
results/*
!results/benchmark_frontend_sources/
!results/benchmark_frontend_sources/**
```

### CI guard — phase-specific

**A1a:**
```bash
paths=(
  dev/benchmarks/generate_benchmark_data.py
  dev/benchmarks/frontend_data/identity.py
  dev/tests/test_benchmark_frontend_data.py
  docs/benchmark-dashboard/rollout-plan.md
)
```
**A1b adds:**
```bash
paths+=(
  dev/tests/test_frontend_parsers.py
  dev/tests/fixtures/benchmark_frontend/schema_ts_parity.json
  frontend/src/identity.ts
  frontend/tsconfig.contract.json
)
```
**A2 adds:**
```bash
paths+=(
  dev/benchmarks/frontend_sources.json
  dev/benchmarks/benchmark_source_catalog.json
  results/benchmark_frontend_sources/<actual-committed-file>.json
)
```
Guard pattern:
```bash
for path in "${paths[@]}"; do
  if git check-ignore -q "$path"; then
    echo "ERROR: tracked file is ignored: $path"
    git check-ignore -v "$path" || true; exit 1
  fi
  git ls-files --error-unmatch "$path" || {
    echo "ERROR: file not tracked: $path"; exit 1;
  }
done
```

### Byte-identical verification
```bash
python dev/benchmarks/generate_benchmark_data.py --deterministic --out /tmp/data.json --report /tmp/report.json
# Compare SHA256 to A0 committed legacy_deterministic_output_sha256.json
```

### Verification (A1a)
```bash
pytest dev/tests/test_benchmark_frontend_data.py -v
python dev/benchmarks/generate_benchmark_data.py --check
```

---

## Phase A1b: Schema v1.1 + Frontend + Transitional Inventory (2 days)

### Stable documents committed before A1b starts
`docs/benchmark-dashboard/{schema-v1.1,parser-contracts,aggregation-contract}.md`

### Schema v1.1 — conditional deps (transitional mode)

JSON Schema allows `std_ms` without `sample_count`/`std_ddof`/`std_scope`:
```json
{
  "if": {"required": ["std_ms"], "properties": {"std_ms": {"not": {"const": 0}}}},
  "then": {"required": ["sample_count", "std_ddof", "std_scope"]}
}
```
Canonical semantic validator (A2) always requires all three when `std_ms` present and non-zero.

**Legacy `std_ms` fix:** `bk_data.get("std_ms") or 0` → only output `std_ms` when source field exists. Missing ≠ zero variance.

### Transitional framework registry (A1b)
```python
TRANSITIONAL_FRAMEWORKS = {
    "statgpu":     {"display_name": "statgpu",      "external": False, "backend_policy": "required"},
    "sklearn":     {"display_name": "scikit-learn",  "external": True,  "backend_policy": "forbidden"},
    "glmnet":      {"display_name": "glmnet",        "external": True,  "backend_policy": "forbidden"},
    "statsmodels": {"display_name": "statsmodels",   "external": True,  "backend_policy": "forbidden"},
}
```
Output only entries referenced by generated runs. Stable sort by `framework_id`. Validation: `run.framework ∈ frameworks[]`; `framework_id` unique; `external`→`backend_policy` from registry.

### Transitional comparison registry (A1b)
Use A0 `legacy_comparison_groups` mapping. Dedup by `comparison_id`. Validate: same `comparison_id` → same `env_id` and `label`. Stable sort.

### Transitional Source + field generation
```python
source = {
    "source_id":     f"transitional:{repo_relative_posix_path}",
    "file":          source_path.name,
    "original_path": repo_relative_posix_path,
    "sha256":        source_sha256(source_path),
    "date":          normalize_iso_date(legacy_source_date),
    "parser":        legacy_parser_name,
    "parser_version": legacy_parser_version,
}

# Per legacy parser:
case_id           = legacy hash of parser-specific discriminator (from A0 preflight)
method_config_id  = "default"
comparison_id     = A0 legacy_comparison_groups[source_file]
```

### CLI all-or-none
```python
output_args = [args.out, args.report, args.inventory_out]
if any(output_args) and not all(output_args):
    parser.error("--out, --report, and --inventory-out must be provided together")
if args.check and any(output_args):
    parser.error("--check cannot be combined with output paths")
if args.update_preflight_baseline and (args.check or any(output_args) or args.strict_sources):
    parser.error("--update-preflight-baseline must run alone")
```

### Transitional inventory — computed
```python
registered = len(transitional_source_registry)
available  = sum(1 for s in transitional_source_registry if s.path.exists())
parsed     = len({r["source"]["source_id"] for r in generated_runs})
inventory = {
    "inventory_version": "1.0", "catalog_version": "1.0",
    "catalog_total": preflight["catalog_total"],
    "eligible_total": registered,
    "registered_sources": registered,
    "available_sources": available,
    "parsed_sources": parsed,
}
```

### Frontend changes
- `createDefaultState(data)`, state actions, render epoch, keyed columns, escapeHtml, registry colors, `generation_id` per-file, FilterContext+FilterOptions, runtime version checks.
- `TimingCellIdentity` = `(comparison_id, benchmark_session_id?, env_id, model_id, case_id, method_config_id, variant, implementation, loss, penalty, solver, scale_key, framework, backend)`.
- `typecheck` = `typecheck:src && typecheck:e2e && typecheck:contract`. `tsconfig.contract.json` typechecks `const typed: BenchmarkData = fixture;`.

### Verification (A1b)
```bash
pytest dev/tests/test_frontend_parsers.py -v
pytest dev/tests/test_benchmark_frontend_data.py -v
python dev/benchmarks/generate_benchmark_data.py --check
python dev/benchmarks/generate_benchmark_data.py --out ... --report ... --inventory-out ...
cd frontend && npm run typecheck && npm run build && npm run test:e2e
```

---

## Phase A2: Manifest + Canonical Migration (4–5 days)

### Scope
- Manifest with `comparisons`, `frameworks`, `sources`. Semantic validator `mode="canonical"`.
- `run_id` 16-char hex. `source_id` slug-date-sha12. `reference_run_id` required for computed speedup. Backend_policy enforced. Speedup reference compatibility (full dimension match). Bare RunIdentity unique.
- Session-less collision audit → migration oracle → bare RunIdentity unique.
- SHA256 run_id. Two-phase speedup. Convergence correction. Hash injectivity.
- Timing conditional: `std_ms` present+non-zero → `sample_count`,`std_ddof`,`std_scope` required (canonical). Legacy timing without these → cannot pool → `identical` or parser-specific.
- Strict-mode + `allowed_issue_codes`. Required/optional source. Preservation baseline.
- Three-file replace: report→inventory→data.

### Preservation baseline scope
Runs: field-level immutable; additions allowed; removals forbidden. Registries: IDs immutable; additions allowed; `model.category_ids` may grow. Excluded volatile: `generated`,`git_sha`,`generation_id`, count deltas from additions only.

### CI guard adds A2 paths.

### Verification (A2)
```bash
pytest dev/tests/test_frontend_parsers.py -v
pytest dev/tests/test_benchmark_frontend_data.py -v
python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
python dev/benchmarks/generate_benchmark_data.py --out ... --report ... --inventory-out ... --strict-sources
cd frontend && npm run typecheck && npm run build && npm run test:e2e
```

---

## Phase B: CoxPH Efron (3 days)

Parser: 3 variants (efron_precision/light_ties/heavy_ties). Backend: `cpu→numpy`, `torch_gpu→torch`, `cupy_gpu→cupy`, `cpu_numba→numpy`+`implementation:"numba"`. Register `statsmodels`. Category: `survival`. Model: `CoxPH`.

ParserContract: `identity={"n_samples","n_features"}`, `method=∅`, `replicate=∅`, `public={"n_samples","n_features"}`, `ui={"n_samples","n_features"}`, `ignored=∅`.

Frontend: PanelTable, ValidationPanel, context-aware external, SummaryCards (replace 3), production e2e.

Acceptance: survival populated; 3 variants distinct; external domain-agnostic; strict-mode 0 errors+0 unallowed; schema v1.1 + referential + TimingCellIdentity unique; staleness; preservation; manual smoke test.

---

## Phase C: Comprehensive Validation + Cox Package (3 days)

**parse_comprehensive_validation**: `external_validation[family]`→accuracy+validation (Poisson,Gamma,InvGaussian,NegBinom,Tweedie). `max_coef_diff→coef_max_abs_diff`; `max_bse_diff→bse_max_abs_diff`; `status→validation.status`; `checks[]`. Contract: `identity={"n_samples","n_features"}`, `replicate={"seed"}`. Category: `glm`.

**parse_coxph_package_comparison**: statgpu+lifelines+scikit_survival+statsmodels. `c_index→prediction.c_index`. Register `lifelines`,`scikit_survival`. Contract: `identity={"n_samples","n_features"}`, `replicate=∅`. Category: `survival`.

Frontend: AccuracyPanel (`bse_max_abs_diff`, reference, validation.status, pagination), PredictionPanel (C-index, MSE, alpha).

Acceptance: glm accuracy populated; lifelines/scikit_survival context-aware; C-index; pagination; pass rate = pass/validation; all B+preservation.

---

## Phase D: LassoCV (3 days)

**parse_lassocv_combined**: Aggregate over seeds. Contract: `aliases={"noise":"noise_std"}`, `identity={"n_samples","n_features","n_signal","noise_std","rho","cv_folds","n_alphas","alpha_min_ratio"}`, `method=∅`, `replicate={"seed"}`. `coef_l2_rel→coef_l2_rel_error`. Framework: `sklearn_lassocv_cpu→sklearn`(backend=null), `statgpu_lassocv_*→statgpu`. `replicate`+`sample_count`. ONE run per RunIdentity.

Frontend: ConvergencePanel, TimingChart top-N+dataZoom, SpeedupChart "Top N of M".

Acceptance: no seed-level runs; replicate+sample_count correct; coef_l2_rel_error; variant cascade e2e; chart overflow; all C+preservation.

---

## Phase E: Knockoff (3 days)

**parse_knockoff_benchmark**: Contract: `aliases={"q":"target_fdr","noise_scale":"noise_std"}`, `identity={"n_samples","n_features","n_signal","noise_std","rho","target_fdr"}`, `method={"selection_budget"}`, `replicate={"seed"}`.

Method-specific `method_config_id`: `marginal_corr_topk`,`statgpu_lasso_topk`→`hash(selection_budget)`; others→`"default"`.

Method mapping:

| Source method | model_id | variant | framework | backend |
|---|---|---|---|---|
| `knockoff_fixedx_numpy` | KnockoffFilter | fixed_x | statgpu | numpy |
| `knockoff_modelx_numpy` | KnockoffFilter | model_x | statgpu | numpy |
| `knockoff_fixedx_cupy` | KnockoffFilter | fixed_x | statgpu | cupy |
| `knockoff_modelx_cupy` | KnockoffFilter | model_x | statgpu | cupy |
| `marginal_corr_topk` | MarginalCorrelationSelector | top_k | statgpu | numpy |
| `statgpu_lasso_topk` | LassoSelector | top_k | statgpu | numpy |
| `sklearn_lasso_cv` | LassoCV | cv | sklearn | null |
| `knockpy_gaussian_lasso` | KnockoffFilter | gaussian_lasso | knockpy | null |

Register `knockpy`. Aggregate: `*_mean/std→selection.*`+`*_std`. `q→selection.target_fdr`.

Frontend: SelectionPanel (FDP≤target=green, >target=red, no target=no colour; mean±std).

Acceptance: feature_selection populated; knockpy context-aware; `*_std` match; invariants; pagination; `test_topk_budget_changes_method_config_id`; all D+preservation.

---

## Phase F–I: Remaining Domains (1-2 days each)

| PR | Domain | Source | Category | Key models |
|----|--------|--------|----------|------------|
| F | Unsupervised | `unsupervised_bench_2026-06-27.json` | `unsupervised` | PCA,KMeans,GMM,NMF,TSVD,IPCA,Agglomerative,UMAP,TSNE,DBSCAN |
| G | Panel+GAM+ANOVA | `compare_external_2026-06-24.json` | `panel`,`anova` | PanelOLS,RandomEffects,PooledOLS,GAM,f_oneway,f_welch |
| H | Robust/Quantile | `loss_functions_bench_2026-06-23.json` | `robust_quantile` | QuantileRegression,HuberRegression |
| I | Nonparametric | `nonparametric_comparison_suite_smoke_*.json` | `nonparametric` | KDE,kernel regression |

ParserContract committed before each PR starts.

---

## Appendix A: Metric Invariants Mode Matrix

| Contract | Transitional (A1b) | Canonical (A2+) |
|----------|-------------------|-----------------|
| legacy run_id format | allowed | forbidden |
| legacy case/method ID format | allowed | forbidden |
| `std_ms` without pooling metadata | allowed | forbidden (must have sample_count+std_ddof+std_scope if std present+non-zero) |
| computed speedup `reference_run_id` | optional | required |
| bare RunIdentity unique | no (+session) | yes |
| `benchmark_session_id` | preserved | omitted after audit |
| `case_id=="default"` iff identity empty | not enforced | enforced |
| `method_config_id=="default"` iff method empty | not enforced | enforced |

**Value bounds (both modes):** `fit_time_ms>0`; `0≤precision,recall,fdp,f1,jaccard,fdr,target_fdr≤1`; `MSE≥0`; `0≤c_index≤1`; `n_iter_mean>0`; `n_selected_mean≥0`; `speedup value>0`; `0≤converged_rate≤1`; `std≥0`; `alpha≥0`; `sample_count≥1` (when present).

**Referential (both modes):** `run.env_id∈environments`; `run.model_id∈models`; `run.framework∈frameworks`; `run.comparison_id∈comparisons`; `category_ids⊆model.category_ids`; `model.primary_category_id∈model.category_ids`; `metric.source_file==run.source.file`; `speedup.reference_framework∈frameworks`.

**Structural (both modes):** `run_id` unique; registry IDs unique; `case_id`/`method_config_id`/`source_id` hash injective; `category_ids` non-empty no-dupes; `model.category_ids` non-empty no-dupes; `run.metrics`≥1; `backend_policy` enforced; `source.sha256` 64-char hex; validation overall=strictest check; provenance merge≠ordinal merge; `identity∪method∪public⊆CANONICAL_PARAMETERS`; `ui⊆public`; `parameter_aliases` target⊆`CANONICAL_PARAMETERS`.

---

## Appendix B: Files Summary

### Stable documents
| Document | Committed |
|----------|-----------|
| `docs/benchmark-dashboard/rollout-plan.md` | A0 |
| `docs/benchmark-dashboard/schema-v1.1.md` | Before A1b |
| `docs/benchmark-dashboard/parser-contracts.md` | Before A1b |
| `docs/benchmark-dashboard/aggregation-contract.md` | Before A1b |

### New files
| File | Phase |
|------|-------|
| `.gitattributes` | A0 |
| `dev/tests/fixtures/benchmark_frontend/legacy_identity_audit.json` | A0 |
| `dev/tests/fixtures/benchmark_frontend/legacy_deterministic_output_sha256.json` | A0 |
| `dev/benchmarks/frontend_data/` (identity.py, canonical.py, registry.py, catalog.py, models.py, cli.py, parsers/) | A1a |
| `dev/benchmarks/frontend_sources.json` | A2 |
| `dev/benchmarks/benchmark_source_catalog.json` | A2 |
| `dev/tests/test_frontend_parsers.py` | A1b |
| `dev/tests/fixtures/benchmark_frontend/schema_ts_parity.json` | A1b |
| `dev/tests/fixtures/benchmark_frontend/v1_0_baseline_normalized.json` | A1b |
| `dev/tests/fixtures/benchmark_frontend/v1_1_preservation_baseline.json` | A2 |
| `dev/tests/fixtures/benchmark_frontend/cross_language_chart_key_vectors.json` | A1b |
| `dev/tests/fixtures/benchmark_frontend/*_minimal.json` | A1b+ |
| `results/benchmark_frontend_sources/*.json` | A2 |
| `frontend/src/identity.ts` | A1b |
| `frontend/e2e/identity.spec.ts` | A1b |
| `frontend/tsconfig.e2e.json` | A1b |
| `frontend/tsconfig.contract.json` | A1b |
| `frontend/src/components/panels/PanelTable.ts` | B |
| `frontend/src/components/panels/ValidationPanel.ts` | B |
| `frontend/src/components/panels/AccuracyPanel.ts` | C |
| `frontend/src/components/panels/PredictionPanel.ts` | C |
| `frontend/src/components/panels/ConvergencePanel.ts` | D |
| `frontend/src/components/panels/SelectionPanel.ts` | E |
| `frontend/e2e-production/dashboard-smoke.spec.ts` | B |
| `frontend/playwright.production.config.ts` | B |

### Modified files
| File | Change | Phase |
|------|--------|-------|
| `dev/benchmarks/generate_benchmark_data.py` | Wrapper (re-exports generate, validate_output, main) | A1a |
| `dev/benchmarks/benchmark_frontend_schema.json` | v1.1.0 (anyOf + conditional deps + $defs.metricQuality) | A1b |
| `frontend/src/schema.ts` | v1.1.0 + FilterContext + FilterOptions + ParseReport v2 + SourceInventory + MetricQuality | A1b |
| `frontend/src/state.ts` | Actions + resetDownstreamFilters(level) + createDefaultState(data) | A1b |
| `frontend/src/data.ts` | Version checks; FilterContext; FilterOptions; filterRuns; null-returning fetches; generation_id cross-validation | A1b |
| `frontend/src/main.ts` | Deferred init; render epoch; perf marks; createDefaultState(data); FilterContext passthrough | A1b |
| `frontend/src/components/Header.ts` | Dynamic counts | A1b |
| `frontend/src/components/Sidebar.ts` | Cascade reset via actions | A1b |
| `frontend/src/components/FilterBar.ts` | Context-aware external; variant selector; option-states with ctx; cascade | A1b |
| `frontend/src/components/OverviewTable.ts` | Keyed columns + null-sort + extension point (A1b); panels (B–E) | A1b–E |
| `frontend/src/components/SummaryCards.ts` | Replace 3 cards; ctx external count; inventory | B |
| `frontend/src/charts/TimingChart.ts` | JSON.stringify group/series (incl method_config_id); escapeHtml; registry colors; top-N+dataZoom (D) | A1b |
| `frontend/src/charts/SpeedupChart.ts` | Dynamic reference; escapeHtml; registry colors; "Top N of M" (D) | A1b |
| `frontend/package.json` | typecheck:src, typecheck:e2e, typecheck:contract, test:e2e:prod; @types/node | A1b |
| `frontend/e2e/dashboard.spec.ts` | Deterministic fixture; unconditional assertions; full cascade; panels; overflow; header | A1b+ |
| `frontend/src/style.css` | overflow-x:auto; scale chip scrollable | A1b |
| `.github/workflows/benchmark-frontend.yml` | Phased jobs; triggers; artifact guard; data volume; prod e2e+setup-python (B) | A1a |
| `.gitignore` | Unignore block at EOF | A1a |
| `dev/tests/test_benchmark_frontend_data.py` | Invariants; canonical normalization; chart cell uniqueness; hash injectivity; backend_policy; collision audit; preservation; mode matrix | A1b |

## Appendix C: Preservation Baseline Update Policy

1. Baseline immutable by default.
2. Modification requires: PR states it, before/after diff per changed run with reason, source evidence, independent reviewer approval.
3. Failure output: lists removed run_ids, changed run_ids with field-level old→new values.
