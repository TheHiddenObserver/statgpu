# statgpu Benchmark Frontend

Interactive benchmark dashboard for statgpu. Built with Vite + TypeScript + ECharts.

## Development

```bash
# Install dependencies
npm install

# Generate benchmark data (from project root)
cd ..
python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json

# Start dev server
cd frontend
npm run dev
# Open http://localhost:5173
```

## Build

```bash
# Type check
npm run typecheck

# Production build → docs/assets/benchmarks/
npm run build

# Serve production build
cd ../docs/assets/benchmarks
python -m http.server 8000
# Open http://localhost:8000
```

## Structure

```
frontend/
├── index.html            # Vite entry point
├── src/
│   ├── main.ts           # Dashboard UI, ECharts, filters, state
│   ├── schema.ts         # TypeScript types (matching benchmark_frontend_schema.json)
│   ├── data.ts           # Data fetching, filtering, state management
│   └── vite-env.d.ts     # Vite type declarations
├── public/
│   └── data/             # Generated benchmark data (committed)
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Data Flow

```
results/*.json
  → dev/benchmarks/generate_benchmark_data.py  (Python parsers)
    → frontend/public/data/benchmark_data.json  (unified schema)
      → Vite build copies to docs/assets/benchmarks/data/
        → GitHub Pages / local http.server
```

## Running Tests

```bash
# Python data pipeline tests (from project root)
pytest dev/tests/test_benchmark_frontend_data.py -v
```
