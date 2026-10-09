# Benchmark dashboard maintenance

This page is for contributors maintaining the benchmark pipeline and frontend. Dashboard users should start with the [reader guide](../../docs/en/guides/statgpu_benchmark_dashboard.md); experiment authors can use the [benchmark runner appendix](../../docs/en/benchmarks.md#advanced-appendix-benchmark-runners).

## Data sources and counts

The [source manifest](../../dev/benchmarks/frontend_sources.json) defines canonical inputs and the minimum source date. The [generated inventory](../public/data/source_inventory.json), [parse report](../public/data/parse_report.json), and [normalized data](../public/data/benchmark_data.json) describe the actual bundle. Do not maintain an independent “current” source/run/model count in prose.

The manifest requires explicit source dates on or after `2026-06-01`; canonical sources are SHA256-verified. Strict generation rejects unapproved issues. The three generated files share a `generation_id`; corresponding files are committed under `frontend/public/data/` and copied to `docs/assets/benchmarks/data/` by the build. Inventory counts distinguish registration, availability, and successful parsing. `runs_generated`, model-registry length, and rows containing `metrics.cross_validation` measure different things.

To inspect those quantities without regenerating files, run from the repository root:

```bash
python - <<'PYCOUNTS'
import json
from pathlib import Path
root = Path("frontend/public/data")
inventory = json.loads((root / "source_inventory.json").read_text())
report = json.loads((root / "parse_report.json").read_text())
data = json.loads((root / "benchmark_data.json").read_text())
for key in ("registered_sources", "available_registered_sources", "parsed_registered_sources"):
    print(key, inventory[key])
print("normalized runs", report["runs_generated"])
print("model registry entries", len(data["models"]))
print("CV metric rows", sum("cross_validation" in row.get("metrics", {}) for row in data["runs"]))
assert inventory["generation_id"] == report["generation_id"] == data["meta"]["generation_id"]
assert report["runs_generated"] == len(data["runs"])
PYCOUNTS
```

## Generate and validate

Use the prerequisites in the [frontend README](../README.md#requirements). Run commands from the repository root unless shown otherwise.

```bash
python -m pip install -U pytest jsonschema
pytest \
  dev/tests/test_benchmark_frontend_data.py \
  dev/tests/test_benchmark_catalog.py \
  dev/tests/test_benchmark_inventory_v2.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_domain_coverage.py \
  dev/tests/test_panel_stage_b_frontend_source.py -v

python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
```

Source/parser changes also need the targeted tests for that source. Check numerical validity, case/method identities, explicit missing values, date policy, and computed speedup references. A matching ratio alone is not proof of comparable objectives or timing boundaries.

## Build, browser QA, and deployed assets

```bash
cd frontend
npm ci
npm run typecheck
npm run build
npx playwright install --with-deps chromium firefox webkit
npm run test:e2e
npm run test:e2e:production
```

The build writes to `docs/assets/benchmarks/`. The production suite serves the repository's nested deployment path and covers Chromium, Firefox, and WebKit. For a manual check, serve the repository root and open `/docs/assets/benchmarks/index.html`; verify JSON loading, relative assets after refresh, filter cascades, scope labels, computed/reported speedups, stable sorting, conditional panels, keyboard focus, and chart-data tables.

Update generated data and deployment assets together when changing a source or parser. A build updates local files; publishing the website is a separate repository deployment action. Before publication, review the generated diff and rerun deterministic generation/build. A staleness check against an already-updated committed bundle should leave this command empty:

```bash
git status --porcelain -- frontend/public/data docs/assets/benchmarks
```

Expected output changes during source maintenance must be reviewed, rather than discarded merely to make this check empty. A prose-only edit does not require regenerating unchanged benchmark measurements.

## Add or change a source

1. Preserve the canonical JSON under `results/benchmark_frontend_sources/`, or the manifest's documented source location.
2. Register SHA256, environment, comparison, parser, allowed issue codes, and `source_date` in [frontend_sources.json](../../dev/benchmarks/frontend_sources.json).
3. Check that the source date meets the manifest's minimum and that the catalog/coverage classification matches the intended registration.
4. Implement or reuse a parser through the [parser registry](../../dev/benchmarks/frontend_data/registry.py), following the [parser contracts](../../docs/benchmark-dashboard/parser-contracts.md).
5. Return schema-compliant runs with stable case and method identities. Validation-only sources must omit unavailable timing and speedup; do not infer measurements from correctness results.
6. Add or update parser, date-policy, domain-coverage, and frontend interaction tests as appropriate.
7. Regenerate the bundle, rebuild the deployed assets, review the diff, and run the relevant checks above.

Keep JSON Schema and TypeScript types synchronized. Do not pool sources solely because they share a comparison ID, replace missing dispersion with zero, or relabel old measurements as a newly implemented method. Exact source commit, validator contract, hardware, concrete device, dtype, timing synchronization, transfer scope, repeats, and numerical accuracy belong in the source evidence.

## Source-specific interpretation and historical context

Some sources deliberately contain no performance measurements. For example, `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260808.json` records the historical clean implementation head `636988751bcbfad3442d24d3073cdfcd2b3ac637` on Tesla P100 (source SHA256 `882892c6e3077fe3b9f6084212647311da795fd05d1ed9f12ec53da1e05d0d4d`). Its 34 CuPy/Torch rows cover Panel Stage-B diagnostics, backend provenance, and Stage-A inference regression, but the validator collected no timings. Preserve absent `metrics.timing`/`metrics.speedup`; Hausman rows with `applicable=false` must not acquire invented test statistics. Later Panel sources have their own scope and provenance.

A rounded distribution Markdown report is not enough to synthesize repeat-level timing or precision records. Track conversion/rerun work in developer plans; do not describe it as measured dashboard coverage.

The [remaining-work plan after PR #78](../../dev/plans/statgpu_benchmark_dashboard_next_phase_plan.md) is historical. Its old counts, readiness conditions, browser coverage, and future-work list are not a current queue. The [historical benchmark artifact index](../../dev/guides/historical-benchmark-artifacts.md) preserves older remote experiment references separately from current user guidance.

## Technical references

- [Schema v1.1](../../docs/benchmark-dashboard/schema-v1.1.md)
- [Parser contracts](../../docs/benchmark-dashboard/parser-contracts.md)
- [Aggregation contract](../../docs/benchmark-dashboard/aggregation-contract.md)
- [Domain coverage audit and plan](../../docs/benchmark-dashboard/domain-coverage-audit-plan.md)
- [Method coverage audit](../../docs/benchmark-dashboard/method-coverage-audit.md)
- [Remaining-module audit](../../docs/benchmark-dashboard/remaining-module-audit.md)
- [Robust-loss comparison plan](../../docs/benchmark-dashboard/robust-loss-comparison-plan.md)
- [Penalized robust/quantile plan](../../docs/benchmark-dashboard/penalized-robust-quantile-plan.md)
