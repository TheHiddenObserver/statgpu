# Benchmark data-update tools

Repository home for the six standalone tools and operator-attestation example
from the `benchmark-data-update-v2` handoff. The six regression-test modules live
in `dev/tests/`, as required by the repository benchmark workflow. These
are developer tools, not a public statistical API or benchmark results. The
native campaign runner, validators, and dashboard implementation remain in their
existing repository locations; this directory does not duplicate their archived
source snapshots.

## Generic tools versus archived, pinned tools

- `batch_driver.py` plans bounded registered/independent batches, runs or resumes
  only explicitly named batches, and aggregates a coverage ledger. It is generic:
  the operator supplies a full `--expected-sha`, and the checkout must be clean
  at that identity. Its native planners, resource guards, assessors, and archive
  finalizers remain authoritative. Planning and aggregation run no workloads.
- `preview_export.py` reads stored native campaigns and optional coverage ledgers
  into diagnostic previews. It reassesses stored-artifact contracts, not the
  numerical oracle or GPU execution. `--shard-by-campaign` preserves a complete
  index and reports failed shards instead of silently dropping them.
- `normalize_shards.py` uses the selected checkout's registered dashboard parser
  and strict generator to normalize every complete indexed diagnostic shard.
- `assemble_dashboard_preview.py` validates a generated diagnostic bundle and
  combines it with already built dashboard assets in a fresh external directory.
  It neither builds the frontend nor deploys/publishes the preview.
- `verify_runtime.py` is an **archived preflight**, intentionally pinned to the
  original source commit `91df0d9ebc35a49bde917383ba0680806016f3ae`, tree
  `bdf93d2c86fca661ed3a444b6754325635566843`. It is not a generic verifier for
  this branch, the repair commit, or later commits. By default it checks source
  and environment only; GPU probing requires the separate `--probe-gpu` flag.
- `rerun_thread_bound_jobs.py` is an **archived exact-ten selector**, not a
  general campaign interface. It accepts only the original immutable campaign
  from the `91df0d9...` source and the exact clean repaired source commit
  `f90f5f8b42b62334e6c746ef3488f8a18a1f6217`. The original plan and seal SHA-256
  pins remain unchanged in the script. Its ten selected jobs retain their
  original arguments, warmups, repetitions, shapes, seeds, row/slot identities,
  and 844 expected rows. The per-job timeout may only be lowered from 1800
  seconds. Planning is the default; `--execute` is required to launch the jobs.

Integrating these scripts into a later commit does **not** retarget those two
archived helpers or certify that later commit with old results. Invoke the
archived selector from this directory against a separate, exact clean repaired
checkout. Do not add these files to that pinned checkout or change its source
checks to accommodate the new branch. The generic batch driver can use a new
clean committed checkout when its full SHA is explicitly supplied.

No original evidence is rewritten, no old operator statement is reused, and no
historical acceptance is inherited. Outputs stay diagnostic:
`performance_eligible=false`, `canonical_dashboard_ready=false`, and
`UNVERIFIED_EXCLUSIVITY`. An operator declaration is not machine-verified GPU
exclusivity and does not promote results to canonical performance evidence.

## Paths and prerequisites

Use Python 3.9 or newer. The orchestration/export scripts use the standard
library; actual execution requires the selected native runner's installed
packages and requested physical backends, without silent CPU fallback. Dashboard
normalization/validation requires `jsonschema` and a checkout containing
`export_qualification_preview.py`, `generate_benchmark_data.py`, and the
`qualification_preview` parser. Tests require `pytest`.

Invoke these as standalone scripts, not with `python -m dev...`; their peer imports
are intentionally local to this directory. They can be launched from any working
directory using absolute paths. All relevant inputs are explicit CLI arguments.
The `--repo` argument identifies the source/validator or dashboard checkout for
that operation, not the script's current directory.

Every output path must be fresh and have an existing parent. Keep outputs outside
all source checkouts and evidence inputs. Preserve partial/failed attempts for
diagnosis; do not overwrite, reseal, or edit original evidence. The exact-ten
selector has no automatic retry/resume. The generic driver's `resume` is a
separate explicit command with selected batch IDs and its existing safety checks.

Set these variables to actual absolute paths before using the examples:

```sh
PY=/absolute/path/to/python
TOOLS=/absolute/path/to/statgpu/dev/benchmarks/data_update
REPO=/absolute/path/to/clean/benchmark-checkout
DASHBOARD_REPO=/absolute/path/to/dashboard-checkout
OUT=/absolute/path/to/existing/output-parent
EXPECTED_SHA=FULL_40_CHARACTER_LOWERCASE_COMMIT_SHA
ENVIRONMENT_ID=YOUR_ACTUAL_ENVIRONMENT_LABEL
```

### Plan, aggregate, and export without workloads

```sh
"$PY" "$TOOLS/batch_driver.py" plan \
  --repo "$REPO" --expected-sha "$EXPECTED_SHA" \
  --environment-id "$ENVIRONMENT_ID" --profile representative \
  --output-dir "$OUT/plan-001"

# Once explicitly requested batch executions have produced a results root:
"$PY" "$TOOLS/batch_driver.py" aggregate \
  --repo "$REPO" --plan-dir "$OUT/plan-001" \
  --results-dir "$OUT/results-001" --output-dir "$OUT/ledger-001"

"$PY" "$TOOLS/preview_export.py" \
  --repo "$REPO" --results-root "$OUT/results-001" \
  --coverage-ledger "$OUT/ledger-001/coverage_ledger.json" \
  --shard-by-campaign --output-dir "$OUT/preview-001"
```

Inspect the generated plan and batch IDs before execution. Only when the specific
batches and invocation budget are authorized, use `batch_driver.py execute` (or
`resume`) with `--plan-dir`, `--results-dir`, one `--batch-id` per selected batch,
and an explicit `--max-invocations`. There is no execute-all mode. The default
planning profile is `both`; the example deliberately chooses `representative`.

An optional `--operator-attestation` file must be a new, truthful declaration for
the actual run. Copy the example outside this directory and replace all labels,
including its historical example environment ID. Leaving it absent is supported;
never submit the template as an actual attestation.

### Normalize and assemble a local diagnostic preview

```sh
# Use the actual source date, not the date shown here as a placeholder.
SOURCE_DATE=YYYY-MM-DD
"$PY" "$TOOLS/normalize_shards.py" \
  --repo "$DASHBOARD_REPO" --index "$OUT/preview-001/preview_index.json" \
  --out-dir "$OUT/normalized-001" --source-date "$SOURCE_DATE"

# Select a successfully normalized campaign and existing built frontend assets.
"$PY" "$TOOLS/assemble_dashboard_preview.py" \
  --repo "$DASHBOARD_REPO" \
  --bundle "$OUT/normalized-001/campaign-0001/strict-generator" \
  --built-assets /absolute/path/to/built-dashboard-assets \
  --output-dir "$OUT/dashboard-preview-001"
```

### Plan the archived exact-ten repair

`REPAIRED_REPO` must be a separate clean checkout at the exact `f90f5f8b...`
commit above. `OLD_NATIVE` is the immutable original
`runs/registered-representative-0000/native` directory; retain its adjacent
`native.tar.gz` and `native.tar.gz.sha256`. Those historical inputs and the repair
Git bundle are not stored in this source directory.

```sh
"$PY" "$TOOLS/rerun_thread_bound_jobs.py" \
  --repo "$REPAIRED_REPO" --evidence "$OLD_NATIVE" \
  --output-dir "$OUT/exact-ten-plan-001"
```

A successful plan has `status=plan_only`, `executed=false`, zero attempted
invocations, ten jobs, and 844 expected rows. A plan is not runtime/backend
validation. Only a separately authorized invocation with `--execute` and another
fresh output path runs the selected jobs. Do not use `verify_runtime.py` to
validate this repaired pin: that archived preflight must continue to reject it.

## Local regression tests

From the repository root:

```sh
"$PY" -m pytest -o addopts= -q dev/tests/test_data_update_*.py
```

Or from this directory:

```sh
"$PY" -m pytest -o addopts= -q ../../tests/test_data_update_*.py
```

The tests use synthetic orchestration/evidence, temporary Git fixtures, and pure
native imports; they run no numerical benchmark or physical GPU workload. The
two native-import integration tests default to this repository root and can use
an explicit `STATGPU_THREAD_RERUN_SOURCE=/absolute/path/to/source` override. Their
import checks do not certify a clean source identity; the real selector still
requires the exact repaired commit. Temporary Git-fixture commits are local test
data, not commits in the project checkout.

## Handoff mapping and preservation

The six handoff-root tools map to this directory with unchanged names and bytes:

- `batch_driver.py`
- `preview_export.py`
- `normalize_shards.py`
- `assemble_dashboard_preview.py`
- `rerun_thread_bound_jobs.py`
- `verify_runtime.py`

The `operator-attestation.example.json` file also retains its original name and
bytes. Every handoff-root `test_NAME.py` maps to
`dev/tests/test_data_update_NAME.py`: `batch_driver`, `preview_export`,
`normalize_shards`, `assemble_dashboard_preview`, `rerun_thread_bound_jobs`, and
`verify_runtime`. Test-only integration edits locate the tools in this directory
and default the native-import fixture to the repository root. There are no
duplicate tests here. All original test assertions and source/evidence pins are
preserved.

Bundles, patches, copied source snapshots, count/result snapshots, old checksum
inventories, and large generated artifacts are intentionally excluded. The
repository's native/dashboard files are integrated at their canonical locations
instead of being duplicated here.
