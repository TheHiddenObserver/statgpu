# Benchmark source catalog and coverage contract

This document defines the audit boundary between benchmark artifacts stored in the repository and rows published by the statgpu benchmark dashboard.

## Core distinction

The dashboard has three independent layers:

1. **Discovered artifacts** are JSON files found under the declared catalog scan roots.
2. **Catalog classification** records whether each artifact is current, eligible, registered, historical, incomplete, superseded, unrelated, or still awaiting an explicit disposition.
3. **Canonical registration** is the smaller manifest-protected set parsed into dashboard rows.

Successful parsing of every registered source proves integrity of the registered bundle. It does not prove that all maintained statgpu capabilities have benchmark coverage.

## Machine-readable files

- `dev/benchmarks/benchmark_source_catalog.json` defines scan roots and deterministic classification rules.
- `dev/benchmarks/frontend_sources.json` defines the canonical registered sources, parser contracts, source dates, and SHA256 values.
- `dev/benchmarks/benchmark_coverage_matrix.json` maps maintained capability groups to canonical, partial, non-ready, gap, deferred, or not-applicable states.

The catalog policy and coverage matrix are committed inputs. `source_inventory.json` is a generated summary whose digests bind it to both inputs.

## Catalog classifications

| Classification | Meaning |
|---|---|
| `registered_canonical` | Manifest-registered, SHA-protected source parsed into dashboard rows. |
| `eligible_unregistered` | Audited as canonical-ready but not yet registered; must link an owner issue. |
| `not_canonical_ready` | Current evidence exists, but provenance, parser, timing, or statistical alignment is incomplete or unaudited. |
| `historical_or_excluded` | Audit evidence outside the current canonical date or policy boundary. |
| `superseded_or_duplicate` | Original, duplicate, or intermediate artifact replaced by another classified source. |
| `unrelated_json` | JSON inside a scan root that is not benchmark evidence. |
| `unclassified` | No deterministic disposition exists. This is a validation failure. |

Manifest registration takes precedence over catalog rules. Rules may never make an unregistered source canonical by implication.

## Inventory v2 semantics

The generated inventory uses literal names:

- `discovered_json_artifacts`: all JSON artifacts found in declared scan roots;
- `classified_candidate_sources`: discovered artifacts with a non-`unclassified` disposition;
- `eligible_sources`: registered canonical plus eligible-unregistered sources;
- `registered_sources`: entries in `frontend_sources.json`;
- `available_registered_sources`: registered paths present in the checkout;
- `parsed_registered_sources`: registered source IDs represented in the generated runs;
- `eligible_unregistered_sources`: eligible sources awaiting registration;
- `not_canonical_ready_sources`: current evidence requiring additional work;
- `historical_or_excluded_sources`: historical, superseded, duplicate, or unrelated artifacts excluded from the canonical bundle;
- `unclassified_artifacts`: artifacts without a valid disposition.

`eligible_sources` is independently derived from catalog classification. It is not an alias for `registered_sources`.

## Coverage matrix states

The method-level matrix uses:

- `canonical_current`;
- `partial_canonical`;
- `current_evidence_not_canonical_ready`;
- `benchmark_data_gap`;
- `intentionally_not_benchmarked`;
- `not_applicable`.

Canonical rows must reference source IDs present in the manifest. Partial, non-ready, and gap rows must link an issue or provide an explicit durable disposition.

## Validation invariants

CI fails when:

- a discovered artifact remains unclassified;
- a registered source is missing from the catalog;
- catalog and manifest registration disagree;
- an eligible-unregistered source lacks an owner issue;
- a coverage row references an unknown source ID;
- an unresolved coverage row has neither an issue nor a disposition;
- inventory totals do not reconcile;
- generated bundle or deployment assets are stale.

Existing source SHA256, parser, schema, semantic, speedup-reference, transactional-write, TypeScript, build, and browser checks remain in force.

## No-fabrication rule

Catalog classification never authorizes reconstruction of missing benchmark fields. Rounded prose, partial validation artifacts, or measurements without sufficient case/timing identity remain reported or non-ready evidence until a dedicated benchmark issue produces auditable structured data.

## Current ownership

- Current cross-validation data: #91.
- Distribution: #101.
- Robust Huber/Bisquare/Fair matrix: #102.
- Feature Selection/Knockoff: #103.
- Ordered and ANOVA crossover: #104.
- Linear/GLM inference: #105.
- Covariance and Nonparametric breadth: #106.
- Penalized CoxPH/CoxPHCV: #107.
- Panel breadth: #108.
- Multiple testing, resampling, NNDescent, and operation timings: #109.

Issue #100 remains the umbrella reconciliation tracker rather than duplicating those benchmark executions.
