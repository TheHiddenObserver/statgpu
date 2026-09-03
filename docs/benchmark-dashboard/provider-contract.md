# Benchmark data provider contract

## Status

The dashboard consumes one normalized bundle through a provider boundary. The
current provider reads maintainer-controlled static JSON. A future API provider
may change transport and storage, but it must preserve the normalized bundle
contract consumed by the dashboard.

## Consumer interface

`BenchmarkDataProvider.loadBundle()` returns:

- `data`: required benchmark data using schema `1.1.0`;
- `parseReport`: optional parse report using version `2.0`;
- `sourceInventory`: optional audited inventory using version `2.0`.

The three resources are requested concurrently. Benchmark data is required.
Missing, unavailable, invalid, or generation-mismatched optional metadata is
discarded without hiding valid benchmark data.

All accepted resources must share `data.meta.generation_id`. Unsupported
benchmark schema versions fail closed rather than being interpreted on a
best-effort basis.

## Phase 1: static JSON provider

The static provider loads the bundle relative to the dashboard base URL:

```text
dashboard/
  data/
    benchmark_data.json
    parse_report.json
    source_inventory.json
```

These deployment files are generated in CI from the canonical source registry
and benchmark evidence. They are not committed production artifacts.

Phase 1 data is trusted only because it is maintainer-controlled, schema
validated, semantically validated, and assembled by the protected deployment
workflow. The browser must still render source strings as text rather than
injecting untrusted HTML.

## Future API provider

An API provider must implement the same interface and return the same
normalized TypeScript shapes. Pagination, retries, authentication, transport
DTOs, and API-specific errors remain inside the provider and must not leak into
charts, filters, or panels.

The future service is responsible for:

- authenticating benchmark submissions;
- validating schema, artifact hashes, provenance, and environment metadata;
- separating unreviewed submissions from published benchmark evidence;
- applying rate limits and abuse controls;
- returning a complete, generation-consistent published bundle;
- defining CORS and cache headers for the public read API.

No API credential may be embedded in the static website.

## Compatibility

Backward-compatible optional fields may be added within schema `1.1.x` only
under the versioning rules in [Schema v1.1](schema-v1.1.md). Breaking identity,
field, enum, or metric-semantic changes require a new schema version and an
explicit provider migration.

The default provider caches the complete successful bundle. Request-scoped
loads that carry an `AbortSignal` are not shared through that cache.
