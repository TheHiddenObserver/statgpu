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
- `benchmark_data.json` above 6 MiB.

The workflow builds twice and compares complete artifact hashes. It also
requires a clean repository tree, including non-ignored untracked files, before
browser QA and publication.

## Custom domain migration

Custom-domain activation is a repository-settings and DNS operation. With a
GitHub Actions publishing source, a committed `CNAME` file is not required.
Before changing DNS, rebuild with the root base, run the same artifact and
browser checks, configure the verified domain in GitHub Pages settings, and
retain HTTPS enforcement.
