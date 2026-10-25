# Benchmark dashboard operation and evidence extension v1.2

## Compatibility and scope

The shared JSON Schema, TypeScript types, and loader accept `1.1.0` and `1.2.0`.
Existing committed v1.1 bundles do not change. Unsupported versions are rejected.
Version 1.1 cannot contain the new operation/evidence fields or zero-feature scales.

The standalone `statgpu-benchmark-update-preview-v1` artifact is a different,
lossless diagnostic index, not a normalized benchmark bundle. The dashboard's
local file picker displays it alongside canonical data without uploading or
replacing anything. It renders all timing fields and planned jobs, including
unknown scopes, invalid/missing summaries, blocked jobs and not-run jobs.

The browser checks structure and rejects performance promotion flags, but does
not authenticate hashes, reject duplicate JSON object keys, or rerun validators.
Its validation columns are explicitly recorded, unverified claims. Use the Python
normalizer for export-digest verification; that verification is integrity checking,
not an authenticated source signature or a numerical oracle rerun.

## Explicit operation timing

A v1.2 run can carry `metrics.operation_timing`:

- `duration_ms`: finite, strictly positive number;
- `scope`: nonempty, explicit source timing scope, never inferred as fit;
- `statistic`: nonempty description of the recorded scalar or sample summary;
- `unit`: exactly `ms`;
- `sample_count`: optional positive integer;
- `quality` and `source_file`: the usual metric provenance fields.

`operation_timing` and legacy `timing` are mutually exclusive. The former requires
run-level `evidence`. The frontend groups operation timings by session, scope and
statistic in addition to the existing comparison identity, labels the actual scope,
and offers an Operation metric filter. It never aggregates different timing scopes
or sample statistics inside a chart. Empty raw arrays may carry `sample_count: 0`
only without a timing summary; normalized operation sample counts remain positive.
Seconds-to-milliseconds overflow is rejected. Without a positive summary or explicit scope,
normalization omits operation timing while the local raw viewer preserves the field.

A raw timing sample, scalar summary, minimum, standard deviation or budget are
not interchangeable. Any conversion must preserve original values and pointers in
the diagnostic artifact, and identify its selected statistic explicitly.

## Feature dimensions

For ordinary feature-based workloads, `n_features >= 1` remains unchanged.
Classification metrics can use `n_features: 0` only with
`feature_dimension_kind: "not_applicable"`; their labels must communicate that
meaning. Positive feature counts may use `feature_dimension_kind: "measured"`.
Missing dimensions are not guessed as one or zero. Independent shapes remain
inspectable even if a normalized model/backend mapping is not available.

## Evidence and qualification

An operation run carries `evidence` with:

- qualification: `diagnostic_only`, `operator_attested`, or `machine_verified`;
- recorded correctness and validation status strings;
- effective `performance_eligible` and `canonical_dashboard_ready` booleans;
- separate `machine_verified_exclusivity` boolean and optional operator statement;
- optional unchanged `raw_performance_eligible`, distinct from effective eligibility;
- raw JSON pointer, source commit, optional source fingerprint and CPU-thread details;
- optional evidence-error strings.

An operator statement must not change machine exclusivity or effective performance
eligibility. The current diagnostic exporter and normalizer always keep effective
eligibility/readiness false. Original raw flags are preserved, never overwritten
with a fabricated successful status. No new measured GPU data is supplied by this
schema or its synthetic tests.

## Speedup safety and current support boundary

Operation-speedup generation and canonical performance promotion are **not
supported by this diagnostic-only update**. The Python canonical computed-ratio
validator still requires legacy `metrics.timing.fit_time_ms`; a hypothetical
qualified operation ratio does not become a publishable bundle just because it
passes frontend display guards. All current operation exports remain diagnostic
and emit no speedups. The following schema/display guards are defensive checks,
not an implementation of a new performance-promotion pipeline.


Diagnostic/effectively ineligible records cannot contain or display a speedup.
For new evidence-bearing records, positive eligibility requires machine-verified
qualification, explicit passed correctness/validation, machine exclusivity, unchanged
positive raw eligibility, canonical readiness, an exact 40-hex source commit and
64-hex fingerprint, and no evidence errors.

The frontend additionally withholds any supplied evidence-bearing speedup unless it has an actual qualified reference run in the comparison,
positive times, the same session/environment/chart identity and source provenance,
matching operation scope and statistic, a matching framework/backend reference, and
a ratio consistent with the referenced times. Otherwise it is hidden. The frontend
does not generate ratios. Legacy v1.1 reported and computed speedup semantics remain
unchanged. This extension does not authorize performance promotion or publication.

## Validation

- `dev/tests/test_frontend_operation_schema.py`: v1.1 compatibility, v1.2 operation
  and p=0 rules, invalid durations/scopes, evidence contradictions, absent timing.
- `frontend/e2e/operation-timing.spec.ts`: scope labels, selection, eligibility,
  real comparator matching, missing measurements, legacy compatibility.
- `frontend/e2e/diagnostic-preview.spec.ts`: raw-data parsing and local file UI,
  blocked/not-run jobs, replacement/clear/recovery, safe text rendering, no upload.

Browser-dependent tests require a working Chromium runtime. A successful typecheck,
build or non-browser test does not establish browser interaction coverage.
