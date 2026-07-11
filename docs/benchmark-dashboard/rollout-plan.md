# Benchmark Dashboard Rollout Record

## Document status

This file records the rollout completed through PR #78. It replaces the earlier phase proposal, whose planned schema, parser, frontend, and testing work has now largely been implemented.

For current contracts, use:

- `schema-v1.1.md`
- `parser-contracts.md`
- `aggregation-contract.md`
- `../en/guides/statgpu_benchmark_dashboard.md`
- `../../frontend/README.md`

## Current implementation

PR #78 expands the benchmark dashboard from four legacy sources and 1,416 runs to nine manifest-registered canonical sources and 1,491 normalized runs.

Current output characteristics:

- Schema version: `1.1.0`.
- Parse report version: `2.0`.
- Source inventory version: `1.0`.
- Canonical sources: 9.
- Parser implementations: 8.
- Normalized runs: 1,491.
- Model registry entries: 13.
- Domain panels: Validation, Accuracy, Prediction, Convergence, and Selection.

## Delivered phases

| Phase | Delivered scope | Status |
|---|---|---|
| A0 | Legacy audit, line-ending contract, deterministic baseline | Complete |
| A1a | Generator package split and tracked-file/CI protections | Complete |
| A1b | Schema v1.1, transitional compatibility, TypeScript contract | Complete |
| A2 | Manifest-driven canonical sources, source SHA256, canonical run identity, strict mode | Complete |
| B | CoxPH Efron parser, validation panel, context-aware external frameworks | Complete |
| C | Comprehensive validation, Cox package comparison, accuracy and prediction panels | Complete |
| D | LassoCV seed aggregation, convergence panel, chart limits | Complete |
| E | Knockoff parser and selection panel | Complete |
| Review hardening | Staleness enforcement, schema pinning, state regression tests, chart-label disambiguation | Complete |

The earlier F–I placeholders represented future benchmark domains rather than committed implementation scope. New domains should now follow the documented schema and parser contracts instead of extending the old phase numbering.

## Architectural decisions in force

### Versioned three-file bundle

The generator writes:

```text
benchmark_data.json
parse_report.json
source_inventory.json
```

The three files share one SHA256 `generation_id` computed from the full bundle after excluding their `generation_id` fields.

### Manifest-driven canonical mode

`dev/benchmarks/frontend_sources.json` is the source registry for:

- Environments.
- Frameworks and backend policies.
- Comparison groups.
- Canonical source paths.
- Parser selection.
- Source SHA256.
- Required/optional status.
- Allowed issue codes.

When the manifest exists, the generator uses canonical mode.

### Canonical identity

A run ID is the first 16 hexadecimal characters of the SHA256 of canonical `RunIdentity` JSON.

Identity includes source, case, method configuration, environment, model, variant, implementation, loss, penalty, solver, framework, backend, and scale.

Chart identity excludes source ID but includes comparison and method identity, allowing separate source families to be compared without collapsing distinct variants or implementations.

### Strict source behavior

`--strict-sources` fails on:

- Missing required source.
- Missing required SHA256.
- Source hash mismatch.
- Parser exception.
- Unapproved warning or error issue code.
- Structural, schema, or semantic validation error.

### Transactional generation

The generator validates all output before writing. The report, inventory, and data files are first written to temporary files and then replaced atomically.

### No implicit cross-source pooling

Parsers may aggregate replicate-level values within one canonical run. The generator does not statistically pool runs across source IDs. `comparison_id` enables display comparison only.

## Frontend implementation

The frontend is split into:

```text
src/main.ts
src/data.ts
src/state.ts
src/identity.ts
src/charts/
src/components/
src/components/panels/
src/utils/
```

Implemented behavior includes:

- Data-driven default environment/category selection.
- Progressive model → variant → penalty → solver → scale filtering.
- Cascade resets for downstream filters.
- Backend filtering for statgpu only.
- Context-aware external framework filters.
- Timing and speedup chart grouping with framework/backend/implementation series identity.
- Distinct labels for computed and runner-reported speedups.
- Configurable chart row/group limits.
- Keyed, deterministic overview-table sorting.
- Conditional domain panels.

## Generated assets

Canonical generated inputs are committed under:

```text
frontend/public/data/
```

The production build is committed under:

```text
docs/assets/benchmarks/
```

CI regenerates data deterministically, rebuilds the frontend, and fails when either location is stale.

## Verification matrix

| Area | Command or CI job |
|---|---|
| Parser/data tests | `pytest dev/tests/test_benchmark_frontend_data.py dev/tests/test_frontend_contracts.py -v` |
| Generator validation | `python dev/benchmarks/generate_benchmark_data.py --check --strict-sources` |
| TypeScript | `npm run typecheck` |
| Production build | `npm run build` |
| Browser/state tests | `npm run test:e2e` |
| Generated assets | Benchmark Frontend CI staleness job |

CI currently tests Python 3.9 and 3.11, Node.js 20, production build, deterministic staleness, and Playwright Chromium.

## Procedure for a new source

1. Place a canonical source file under `results/benchmark_frontend_sources/`.
2. Normalize its UTF-8 text and compute SHA256 using the generator contract.
3. Add its framework/comparison/source registration to `frontend_sources.json`.
4. Reuse or register a parser.
5. Emit normalized runs and model entries.
6. Add source-specific tests.
7. Run strict validation.
8. Regenerate all three frontend data files.
9. Rebuild deployment assets.
10. Confirm that deterministic generation leaves a clean Git status.

## Remaining rollout work

The data pipeline and frontend architecture are ready for integration. Remaining work is product-level rather than contract-blocking:

- Complete a manual cross-browser QA pass.
- Confirm keyboard navigation and basic accessibility.
- Integrate the user guide into the repository documentation navigation, when applicable.
- Consider URL-persisted filter state.
- Add responsive/mobile layout improvements if the dashboard becomes a primary public entry point.
- Add new benchmark families only when canonical sources and parser tests are available.

These items are tracked in `docs/en/guides/statgpu_benchmark_dashboard_next_phase_plan.md`.
