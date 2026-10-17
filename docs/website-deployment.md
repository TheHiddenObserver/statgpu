# Public website deployment contract

## URL model

The initial canonical deployment is:

```text
https://thehiddenobserver.github.io/statgpu/
https://thehiddenobserver.github.io/statgpu/en/
https://thehiddenobserver.github.io/statgpu/cn/
https://thehiddenobserver.github.io/statgpu/dashboard/
```

`STATGPU_SITE_BASE=/statgpu/` is the release default. A custom-domain build
sets `STATGPU_SITE_BASE=/` and `STATGPU_SITE_URL` to the final HTTPS origin.
The dashboard uses a relative Vite base and therefore remains relocatable under
either site base.

## Repository ownership

- `docs/`: VitePress homepage and bilingual documentation sources.
- `frontend/`: independent Vite, TypeScript, and ECharts dashboard.
- `results/benchmark_frontend_sources/`: canonical benchmark evidence.
- `dev/benchmarks/`: schema, registry, parsers, and data generator.
- `.site-dist/`: ignored deployment staging directory produced by CI.

The dashboard is not rewritten as a VitePress application. A dashboard build
failure blocks publication so the homepage, documentation, and evidence view
always describe one repository revision.

## Build and deployment

The workflow pins Node and npm, installs both lockfiles with `npm ci`,
generates the benchmark bundle deterministically, builds VitePress, builds the
dashboard, copies the dashboard to `.site-dist/dashboard/`, verifies links and
size budgets, runs browser tests, and uploads one GitHub Pages artifact.

VitePress builds with full Git history so `lastUpdated` reflects source history,
and local-search indexing is serialized so repeated builds produce the same
artifact bytes. Pull requests execute all build and validation gates but do not
deploy. Only a push to `master` may upload and deploy through the `github-pages`
environment. Compiled HTML, JavaScript, CSS, search indexes, and generated
dashboard JSON are not committed.

## Verification

The assembled artifact must contain the four public entry routes, the complete
three-file benchmark bundle, and `.nojekyll`. Verification rejects:

- missing internal targets, with deployment-path case sensitivity preserved;
- root-absolute URLs outside the configured site base;
- the legacy `docs/assets/benchmarks` route;
- dashboard JavaScript above 750 KiB;
- `benchmark_data.json` above 6 MiB;
- the two locale search modules and their decoder above 1.6 MiB.

Public documentation remains fully searchable, including complete API references.
Builds losslessly gzip each locale index and verify its decoded byte count and
SHA-256 digest. Browsers use the native `DecompressionStream` gzip API when available and a
bundled, pinned gzip decoder otherwise. The transfer budget counts
both emitted base64 locale modules and the separately emitted decoder, including
all fallback code, without
assuming server-side compression. Cross-browser search tests still exercise
prefetching, Chinese controls, compact results, and reference-page queries.

Before installing npm dependencies, the workflow rejects tracked files under
`node_modules/`, `frontend/node_modules/`, `frontend/test-results/`,
`frontend/playwright-report/`, `frontend/dist/`, `frontend/public/data/`,
`docs/assets/benchmarks/`, `docs/.vitepress/cache/`, `docs/.vitepress/.temp/`,
and `.site-dist/`. This checks the Git index, so an artifact already present in a
clean checkout is rejected even when its path is ignored. Both push and pull
request path filters cover the guard, its tests, ignore rules, and these outputs.

The workflow builds twice and compares complete artifact hashes. It also
requires a clean repository tree, including non-ignored untracked files, before
and after browser QA. Generated artifacts may remain in their ignored directories;
tracked source changes or new non-ignored files block publication.

## Pinned build-tool compatibility overrides

VitePress remains pinned to 1.6.4. The root npm overrides select Vite 6.4.3
instead of VitePress's declared Vite 5 range, and `@xmldom/xmldom` 0.9.12 instead
of `speech-rule-engine` 4.1.4's exact 0.9.10 dependency. These are deliberate
build-tool compatibility exceptions to remove inherited security advisories;
they are not a VitePress framework migration. Keep both overrides explicit and
validate dependency resolution, both site bases, search, MathJax, reproducible
output, and all three browser engines after changing either one.

The dashboard retains ECharts 5.6. The moderate
[Lines-series tooltip advisory](https://github.com/advisories/GHSA-fgmj-fm8m-jvvx)
affects the unused Lines series and its default formatter. The dashboard
registers only BarChart and supplies its own tooltip formatters. An ECharts
major-version migration is separate from this website integration; the package
advisory remains visible in frontend audit results.

## Custom domain migration

Custom-domain activation is a repository-settings and DNS operation. With a
GitHub Actions publishing source, a committed `CNAME` file is not required.
Before changing DNS, rebuild with the root base, run the same artifact and
browser checks, configure the verified domain in GitHub Pages settings, and
retain HTTPS enforcement.
