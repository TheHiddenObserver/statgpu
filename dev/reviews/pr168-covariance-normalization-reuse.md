# PR168: call-local covariance normalization reuse

## Exact target and scope

This uncommitted candidate starts from PR168 head
`652b4699c1ed01b5baf327303b098110640e0870`, with comparison base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. Both remote identities were
rechecked while preparing the candidate. Before editing, all 1,517 remote
blobs and modes were verified in a clean isolated source snapshot. Its
full-tree fingerprint was
`92662fce55ad51d11006082f07370f99b4359f30acb35f792286c386c6b626a8`
(SHA-256 of sorted path/NUL/mode/NUL/Git-blob-SHA/newline records).

Classification: narrow numerical refactor preserving public behavior.
Active axes: shared public validation, dtype/device and array ownership,
exception/lifecycle behavior, CV preparation, regression tests and evidence.
There is no new estimator/backend capability, objective or solver change.

The seven-file production/test/CI/changelog payload has fingerprint
`e9f60059192cc4731f3a604f94f84a33c3baf5eb8b101e540866fe3557cd43e9`
(SHA-256 of sorted path/NUL/file-SHA256/newline records). The
[source manifest](pr168-covariance-normalization-source.json) records all seven
files and 95 additional source/consumer/validator context files. This report
and the manifest are excluded from that payload hash to avoid circular hashes.
The snapshot anchor is not a local Git HEAD: local Git metadata contains no
commit and is used only by repository-root provenance tests. Publication, new
exact-head hosted CI and physical acceptance are separate
steps; no commit or push was performed as part of this candidate preparation.

## Refactor

The preceding source normalized NumPy-bound Torch input twice in `fit`, `score`
and `mahalanobis`, and three times in `predict` because its separately guarded
`mahalanobis` call repeated the prevalidation conversion.

- The existing finite-check hook may now return its validated working input.
  The shared wrapper uses bound replacement arguments only if the hook returns
  a non-None object with different identity. Default and validation-only hooks
  retain the exact original call and arguments.
- The covariance override returns the normalization it already computed. All
  seven covariance estimators therefore reuse it for numerical preparation,
  including GraphicalLassoCV's separate probe and nested prediction guards.
- Native Torch FP8 widening is likewise reused on the source device; its Torch
  array type still drives native automatic backend selection.
- No input is cached on an estimator or in shared mutable state. Later calls
  recheck current input values, and no prepared array leaks into clone/refit.
- All finite-check exclusions and all other supplied numerical arguments,
  including covariance's statistically ignored `y`, retain their checks.
  Existing fit-reset handling and CUDA exception tagging remain in place.
- Shared backend conversion helpers, `check_finite`, numerical implementations,
  tolerances, expected-failure settings and unsupported input policies are
  unchanged. Fitted queries continue to follow the fitted backend/device.

These are measured conversion-count reductions. No wall-clock speedup or
physical data-transfer timing is claimed.

## New regression suite and CI

`dev/tests/test_covariance_normalization_reuse.py` adds 664 cases:

- 560 primary CPU cases: all seven classes, ten dtype/view/gradient kinds,
  four public methods, and positional/keyword `X` calls. They count the helper
  plus real CPU, float64 and NumPy conversion operations, verify numerical
  parity and input/backing-storage ownership, and require one normalization.
- 28 native FP8 cases require one source-device widening and no NumPy helper.
- Seven global-CPU-to-native-AUTO refit cases; seven mutable-input/retry/clone
  cases; 14 ignored-`y` rejection cases; three host-input identity cases;
  default/legacy and replacement-hook argument-preservation cases; and a
  concrete noncovariance consumer/installer-idempotence check.
- 42 physical CUDA cases cover all seven estimators, FP32/BF16/FP8 and source
  device indices 0/1. Each case checks all four public methods. There are
  21 single-device and 21 nondefault/two-device cases. Missing hardware skips;
  conversion or kernel errors on present hardware do not skip.

The file is wired into the Python 3.9–3.12 CPU regression matrix and both
existing pinned Torch CPU jobs. Existing physical CUDA and MPS tests remain
unchanged. Existing FP8 ordering emulation remains CPU ordering evidence.

## Candidate verification

- Four existing covariance suites: **1,270 passed / 253 physical-device skips**.
- New reuse suite: **622 passed / 42 physical-device skips**.
- Prior-head negative control: **all 560 primary count regressions fail** on
  the exact 652b469 source; these same tests pass on the candidate.
- Initial full-tree run: **8,450 passed / 1,144 skipped / 166 expected failures**,
  with two checkout-provenance setup failures because the snapshot had no Git
  metadata. Both failures occurred at `git rev-parse --show-toplevel`, before
  the runtime import probe. Initializing empty local Git metadata (no commit or
  source changes) restored the required root; all three provenance tests then
  passed. Final full-tree rerun: **8,452 passed / 1,144 skipped / 166 expected
  failures / zero failures**, 90 warnings, 189.96 seconds. The frozen payload
  hashes were unchanged before and after the run.
- Python 3.9 grammar, workflow YAML and added-line whitespace checks passed.
- Documentation contracts: **169 files passed**; deterministic links clean.
- Ruff was unavailable in the local runtime and was not run.

Local runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
sklearn 1.8.0, Torch 2.14.1+cpu and pytest 9.1.1. CUDA is unavailable and CuPy
is absent. Counts overlap and must not be added to full-tree results.

Independent review of the frozen seven-file payload found no confirmed
correctness/API regression. Its combined covariance set (including the existing
P2 module) passed **1,912 / 295 skipped**; four shared compatibility/formula/
cleanup suites passed **246 / 36 skipped**. Independent probes counted **252**
NumPy-bound calls with one conversion each, compared **413** numerical/error/
lifecycle records with 652b469, and preserved **554** installed wrapper
identities and **64** noncovariance hook records. This bounded local-candidate
review is not a whole-PR verdict or physical CUDA/MPS acceptance.

## Fresh physical acceptance and publication boundary

The user-reported 200-case P100 acceptance and green hosted CI at 652b469 are
historical evidence for that source. They do not establish this changed
candidate. The preceding 46 two-device and seven MPS cases also remain
unvalidated; the added physical conversion-count cases require fresh runs.

After the candidate is published, record its exact commit and clean source
identity before running these commands on the corresponding hardware:

```bash
# All covariance preservation and reuse tests; retains hardware-specific skips.
python -m pytest dev/tests/test_covariance_device_contract.py \
  dev/tests/test_covariance_array_inputs.py \
  dev/tests/test_covariance_torch_inputs.py \
  dev/tests/test_covariance_fp8_prevalidation.py \
  dev/tests/test_covariance_normalization_reuse.py -q -ra

# Added single-CUDA conversion-count acceptance: 21 cases, four methods each.
python -m pytest dev/tests/test_covariance_normalization_reuse.py -q -ra \
  -k 'physical and not nondefault'

# Added two-CUDA/nondefault-source acceptance: 21 cases, four methods each.
python -m pytest dev/tests/test_covariance_normalization_reuse.py -q -ra \
  -k 'physical and nondefault'
```

The original exact-boundary physical commands remain in the
[FP8 repair record](pr168-covariance-fp8-repair.md#physical-rerun-commands-and-evidence-boundary).
Capture the new source SHA/fingerprint, import location, Python/package/device
versions, device ordinals, full pytest output and JUnit XML. Missing hardware
and deselected tests must remain distinct from passes. Publication requires
separate user approval; this report does not authorize a push, PR-body update,
comment or merge.
