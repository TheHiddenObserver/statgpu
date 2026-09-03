# Production dashboard QA

This suite validates the assembled benchmark dashboard at its GitHub Pages project path rather than the Vite development server.

Run from `frontend/`:

```bash
npm ci
npx playwright install --with-deps chromium firefox webkit
npm run test:e2e:production
```

The production configuration previews `.site-dist/` through VitePress and opens:

```text
/statgpu/dashboard/
```

The suite runs the same product-level checks in Chromium, Firefox, and WebKit. It covers project-path asset and JSON loading, the CV filter cascade and upstream reset behavior, keyboard/focus and accessible names, chart-data table alternatives, explicit empty states, preservation of failed CV backend evidence, refresh behavior, and text contrast.

`npm run test:e2e` remains the faster Chromium regression against the Vite development server. Both suites are required by `Benchmark Frontend CI`; production QA does not replace the existing regression suite.
