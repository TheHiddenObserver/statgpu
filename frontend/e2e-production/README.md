# Production dashboard QA

This suite validates the assembled public website and benchmark dashboard at the
GitHub Pages project path rather than only the Vite development server.

Build the assembled site first, then run from `frontend/`:

```bash
npm ci
npx playwright install --with-deps chromium firefox webkit
npm run test:e2e:production
```

The production configuration previews `.site-dist/` through VitePress and opens:

```text
/statgpu/dashboard/
```

The suite runs product-level checks in Chromium, Firefox, and WebKit. It covers
project-path asset and JSON loading, documentation navigation and MathJax,
bilingual local search, the CV filter cascade and upstream reset behavior,
keyboard/focus and accessible names, chart-data table alternatives, explicit
empty states, preservation and provenance of failed/repaired CV evidence,
refresh behavior, and text contrast.

`npm run test:e2e` remains the faster Chromium dashboard regression against the
Vite development server. Both suites are required by the `Website and Benchmark
Frontend` workflow; production QA does not replace the existing regression
suite.
