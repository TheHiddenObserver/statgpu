# Production dashboard QA

This suite validates the assembled public website and benchmark dashboard at the
GitHub Pages project path rather than only the Vite development server.

Use the [frontend requirements](../README.md#requirements) and run from the
repository root. Generate the benchmark bundle before building, including on a
fresh checkout:

```bash
npm ci
npm ci --prefix frontend

python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

npm run site:build
npm exec --prefix frontend -- playwright install --with-deps chromium firefox webkit
npm run test:e2e:production --prefix frontend
```

The build consumes the generated bundle; `npm run site:build` does not generate
benchmark data. Keep all three JSON files together under ignored
`frontend/public/data/`.

The production configuration previews `.site-dist/` through VitePress and opens:

```text
/statgpu/dashboard/
```

The suite runs product-level checks in Chromium, Firefox, and WebKit. It covers
project-path asset and JSON loading, canonical family `README` routes, current
API references, documentation navigation and MathJax,
bilingual local search, the CV filter cascade and upstream reset behavior,
keyboard/focus and accessible names, chart-data table alternatives, explicit
empty states, preservation and provenance of failed/repaired CV evidence,
refresh behavior, and text contrast.

`npm run test:e2e` remains the faster Chromium dashboard regression against the
Vite development server. Both suites are required by the `Website and Benchmark
Frontend` workflow; production QA does not replace the existing regression
suite.

To run the same checks for a custom domain, rebuild with
`STATGPU_SITE_BASE=/ npm run site:build`, then run
`STATGPU_SITE_BASE=/ npm run test:e2e:production --prefix frontend`. The preview
base and the artifact must match.
