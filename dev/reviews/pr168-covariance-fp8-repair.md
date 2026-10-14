# PR168: covariance FP8 prevalidation repair

## Exact target and bounded contract

This repair starts from PR168 head `8e920c0122fa4e9cd92d66ce27855741617db31b`,
tree `f45c45f5c94538691244fd8e8b2af44987b1a2c1`, with comparison base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. Before editing, all **1,513**
remote blobs and executable modes matched. The starting full-tree fingerprint
was `8ea8ee02a04908945c1b5e391c9eede88ec53e1bef507034267d8f8211751429`
(SHA-256 of sorted path/NUL/mode/NUL/Git-blob-SHA/newline records).

Classification: existing capability reconciliation / public contract repair.
Active axes: public validation, dtype/device ownership, failure provenance and
lifecycle, CV, tests, documentation and evidence. Numerical objectives, solvers,
backend converters, ANOVA and model-X production remain unchanged.

The [source manifest](pr168-covariance-fp8-source.json) records changed
production/test/workflow/user-doc bytes and unchanged consumer/validator
context. This record and the source manifest are excluded from their own hashes.
The [consumer inventory](pr168-covariance-fp8-consumers.json) enumerates the
shared-wrapper surface. The published commit/tree and exact-head hosted CI
outcomes are recorded in the PR description after publication. Earlier reviews
and CI remain historical. This bounded review is not another whole-PR audit.

## Failure evidence and cause

For the preceding head, the user reported **133 single-P100 cases: 112 passed,
21 failed, zero skipped**, with Torch **2.5.1+cu118** and CuPy **13.6.0**.
Every reported failure used `float8_e5m2`: 14 NumPy-destination cases and seven
native-AUTO cases. The traceback reached `BaseEstimator`'s public finite guard,
then `check_finite` and `torch.isfinite` on the original CUDA FP8 tensor, raising
`RuntimeError: "abs_cuda" not implemented for 'Float8_e5m2'` before covariance
preparation. These are user-reported results; raw logs were not independently
inspected, and no physical run occurred in this repair environment.

The earlier Torch **2.0.0** result (**112 passed / 21 skipped**) did not validate
FP8. The reported CPU FP8 run (**102 passed**) and CUDA FP8-to-CPU-to-float64
probe narrow the failure to ordering; neither proves the repaired physical path.
The previous 35 two-GPU cases and seven MPS cases remained unrun/skipped.

The shared finite-validator blob is exactly
`011ed63cfbd78332da4093b1e382dad3fddff62b` in the comparison base, preceding
head and this repair. Thus that limitation is inherited, but the preceding
head's new positive CUDA FP8 support contract was not met. The official
[Torch 2.5.1 CUDA abs dispatch](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/AbsKernel.cu)
excludes FP8; this source-level limitation is not a P100-only property.

## Correction and unchanged boundaries

The public wrapper delegates each finite check to a private identity-by-default
check hook, still inside its existing try/reset boundary. The default hook calls
the unchanged shared `check_finite`. Only the covariance family overrides it:

1. For supported dense real Torch `X` on NumPy-bound fit/query paths, detach,
   move to CPU, promote to float64, resolve lazy-negative values, then validate.
   CPU transfer occurs before float64 conversion, preserving MPS-to-CPU input.
2. For dense `float8_e5m2` on Torch-selected paths, validate a detached float64
   temporary on the original source device. Native CUDA validation does not
   copy the array to CPU. The numerical method still receives the original
   object, so its native backend and device resolution are preserved.
3. Fit resolves the current explicit/global policy; queries use fitted backend
   ownership before considering later global changes. CV continues using its
   initial prepared array for folds, candidate scoring, selection and final refit.
4. Conversion and finite-reduction errors are re-raised unchanged. Original
   Torch CUDA backend and concrete device provenance are attached before the
   existing fit-reset handler runs, including a NumPy-bound CPU temporary.

The existing covariance policy resolver was extracted without adding backend
availability checks to prevalidation. Unsupported complex, sparse, quantized,
other-FP8 and Torch-FP8-to-CuPy paths retain their original finite checks. This
is neither an all-input CPU fallback nor an exemption from finite validation.
`y` and other numerical arguments keep their shared validation behavior.
Original inputs, backing storage, metadata and gradients are unchanged.

## Consumer and independent review evidence

The runtime inventory identifies **73 BaseEstimator descendants across eight
families and 557 numerical public-method surfaces**. Only the seven covariance
classes opt into the hook: EmpiricalCovariance, LedoitWolf, OAS,
ShrunkCovariance, GraphicalLasso, GraphicalLassoCV and MinCovDet. The remaining
64 concrete consumers use unchanged object/name validation; two descendants are
abstract. Re-running the public finite-validation installer is idempotent.
Static-only legacy definitions are separately identified in the inventory.

Independent missing-kernel controls cover 14 fits and 42 queries. All 56 work
with this repair and fail when only the covariance hook is replaced by the
original default guard. Independent NaN/positive-infinity/negative-infinity
checks reject 42 fits and 126 queries. Seven no-op-guard controls ensure that
ignored nonfinite `y` remains rejected. These counts overlap with other tests;
they must not be added. The kernel-missing emulator is ordering evidence, not
physical CUDA acceptance.

The independently executed final regression module passed **182 cases**, with
**78 physical-device skips**. Replacing the hook with the original guard makes
all **42** primary ordering tests fail; replacing it with a no-op makes all
**14** ignored-y guard tests fail. These mutations changed no source files.
AST review confirms the shared installer is unchanged except the check-hook
call/import move and the extracted covariance policy is structurally unchanged.

All **101** source-manifest entries were independently verified: nine changed
payload files, one consumer inventory and 91 unchanged context files, covering
all 87 required consumer/installer/direct-validator/legacy source paths. The
nine-file payload fingerprint is
`6065fc7d52199f6a34ecb7e102837b9070b2a8bc29a37a3f20a969a290ee2bda`.
Independent review found no remaining in-scope implementation/test/workflow/docs
issue for those bytes. Physical assertions were reviewed but not executed here.

## Validation

Counts overlap and must not be added. Skips and expected failures are not
passing device paths.

- Complete final test tree with real Torch CPU: **7,830 passed / 1,102 skipped /
  166 expected failures / zero failed**, 91 warnings, 213.01 seconds. All input
  source/test/workflow/user-doc bytes were verified unchanged across the run.
- New module: **182 passed / 78 physical-device skips**, comprising 67 single-GPU
  and 11 two-GPU/nondefault cases; 260 cases collected in total.
- Four covariance input/device/prevalidation modules: **1,270 passed /
  253 skipped / zero failed**, two warnings, 78.88 seconds.
- Ordered ANOVA/covariance/native-isolation command with the new module:
  **1,461 passed / 420 skipped / zero failed**, two warnings, 100.16 seconds.
- Independent lifecycle/formula selection: **31 passed / nine skipped**;
  four complete lifecycle files: **five passed**. CPU CUDA-origin sentinels
  separately verify conversion/finite-error identity, concrete device tags and
  exactly one reset; those sentinels are explicitly not hardware evidence.
- Documentation contracts: **169 files passed**; deterministic link check clean.
  Actual Markdown rendering checked **161 pages / 350 tables / 2,598 rows /
  1,879 local links**, with no defects. This is parser/render/link evidence,
  not new deployed-browser/MathJax visual proof.
- Python 3.9 grammar for changed Python, workflow YAML parsing, whitespace,
  high-signal production Ruff and default Ruff on the new suite pass.
  Differential production Ruff adds no diagnostics; 20 inherited diagnostics
  remain across the two production files. No unrelated lint cleanup was made.

A preliminary full run collected the test module before its final additions;
its result is not used as final evidence. The complete final run above was
repeated after all production, tests, workflow and public documentation edits.

Local runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
sklearn 1.8.0, Torch 2.14.1+cpu and pytest 9.1.1. CUDA is unavailable and CuPy
is absent. No physical CUDA or MPS test is reported as passed here.

The new module is explicit in Python 3.9–3.12 CPU regression commands, the
ordered Torch 2.0.1 CPU command and the pinned Torch 2.5.1 CPU job. The old
Torch pin may skip absent FP8 dtypes; the 2.5.1 preflight requires FP8 creation
and float64 conversion, not a raw-FP8 isfinite kernel. The 2.5.1 job additionally
runs the complete test tree. Hosted outcomes belong to the exact published
head and are recorded separately in the PR description.

## Physical rerun commands and evidence boundary

The original **133 single-CUDA** cases remain byte-for-byte unchanged:

```bash
python -m pytest dev/tests/test_covariance_device_contract.py \
  dev/tests/test_covariance_array_inputs.py \
  dev/tests/test_covariance_torch_inputs.py -q -ra \
  -k 'physical and not nondefault and not physical_two_gpu and not physical_mps'
```

The new prevalidation module adds **67 single-GPU** cases:

```bash
python -m pytest dev/tests/test_covariance_fp8_prevalidation.py -q -ra \
  -k 'physical and not nondefault'
```

It exercises actual FP8 operations without suppressing missing-kernel or
conversion failures. New physical FP8 cases require a Torch runtime exposing
`float8_e5m2`; old Torch's absent dtype is not an acceptance skip. Both source
and destination libraries must work for the original cross-library matrix.
NaN/positive-infinity/negative-infinity failures must be the intended ValueError,
including queries against independently NumPy-trained models.

The original **35** two-GPU/nondefault cases remain unchanged:

```bash
python -m pytest dev/tests/test_covariance_device_contract.py \
  dev/tests/test_covariance_torch_inputs.py -q -ra \
  -k 'nondefault or physical_two_gpu'
```

The new module adds **11** separately selected two-GPU cases:

```bash
python -m pytest dev/tests/test_covariance_fp8_prevalidation.py -q -ra -k nondefault
```

The existing **seven MPS** preservation cases remain unchanged:

```bash
python -m pytest dev/tests/test_covariance_torch_inputs.py -q -ra -k physical_mps
```

Record exact commit and clean worktree/import provenance, Torch/CuPy/NumPy and
Python versions, driver/CUDA and physical device details, full pytest output
and JUnit results. A skip leaves that device/dtype path unverified. The preceding
112-pass/21-failure P100 report remains historical; new CPU and hosted green
checks cannot close those physical failures. The older 222-case P100 report at
`bf9833257d9726b4e8aaf199dc73a30cddd039c3` is also historical and source-specific.
No new R, timing, empirical-FDR, interval-coverage or compiled DBSCAN result is
claimed. No merge/readiness transition, unrelated singular-Torch numerical fix,
#243/#245 implementation or environment upgrade is part of this repair.
