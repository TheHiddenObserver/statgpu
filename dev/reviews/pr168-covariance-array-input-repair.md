# PR168: covariance array-like input compatibility repair

## Target and scope

This is a bounded API/input-contract repair on PR168, starting from exact head
`d3c3a92db52198db22b98e2c52a113bd2c226dbd` and comparison base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. A separate clean worktree was
reconstructed only after all **1,507 original blobs and modes** matched the
fresh GitHub tree. The reconstructed original commit and tree match GitHub;
an older export's Git index was not treated as the target.

The source/validator payload is recorded in
[the companion manifest](pr168-covariance-array-input-source.json). Final
published commit, full-tree fingerprint, and exact-head hosted results are
recorded in the PR description after publication. This record describes the
locally reviewed candidate; it does not itself establish publication or hosted
CI. Earlier reviews and checks
remain evidence for their own source identities, not this repair.

## Finding and correction

The device-policy repair introduced a call to shared transfer helpers before
ordinary covariance input normalization. Those inherited helpers identify
CuPy/Torch-like objects partly by method names. Pandas DataFrame and Series
also have a keyed `get` method, so all seven covariance classes raised
`TypeError: NDFrame.get() missing 1 required positional argument: 'key'`.

The common covariance boundary now recognizes actual CuPy/Torch arrays using
the existing deferred-import type checks. Other array-like inputs are
normalized with `np.asarray(..., dtype=np.float64)` before transfer. Float64
was already the covariance computation dtype; moving this host coercion before
transfer also handles numeric nullable pandas columns and mixed numeric/bool
frames whose ordinary NumPy representation has object dtype. An independent
review caught and closed that intermediate dtype gap before publication.

No pandas import or runtime dependency was added. BaseEstimator and the shared
backend helpers are unchanged. This avoids widening a covariance regression
repair into unrelated inherited converter behavior.

## Consumer and routing closure

- All seven classes use the same preparation boundary: EmpiricalCovariance,
  LedoitWolf, OAS, ShrunkCovariance, GraphicalLasso, GraphicalLassoCV and MinCovDet.
- Their `fit` paths and inherited `score`, `predict`, and `mahalanobis` query
  paths are covered. One-dimensional Series fits retain the existing
  single-feature behavior; query shape rules are unchanged.
- GraphicalLassoCV prepares input once; candidate folds, scoring, and final
  refit retain the selected backend. No fold, alpha, objective or solver logic
  changed.
- Explicit CPU remains NumPy. Explicit/global CUDA and Torch still require
  their named working CUDA backend and fail before this new normalization
  when unavailable. A local explicit CPU request still overrides global GPU.
- Genuine AUTO keeps actual native CuPy/Torch input ownership, including
  Torch CPU. Ordinary array-likes follow existing backend availability rules.
- Fitted evaluation keeps the fitted backend and concrete device even after
  global policy changes. Actual native arrays never pass through NumPy at the
  new boundary; native transfer helpers and indexed-device handling remain.
- Column labels are positional, not an automatic feature-alignment mechanism.
  EN/CN learner guidance and all three changelogs describe the restored input
  behavior without claiming a new statistical method or release.

The helper-use audit traced `_to_array`, `_to_torch`, and `_to_cupy` through all
covariance consumers. Their method-name checks also exist outside covariance;
those inherited callers are outside this introduced call-site repair. Shared
`_detect_backend` consumers outside covariance remain unchanged.

## Regression and CI coverage

`test_covariance_array_inputs.py` covers the original five operations across
all seven classes and six standard/nullable numeric dtypes (**210 cases**),
ordinary array protocols with unrelated `get`/`cpu` methods, mixed numeric/bool
columns, nonfinite/missing/malformed input, positional columns, strict/global/
AUTO policy, real Torch CPU fitted-query ownership, and CV fold/refit behavior.
Numerical controls compare with NumPy and aligned sklearn/SciPy/analytic
covariance and Gaussian-density references. Function-scoped monkeypatch
fixtures restore incoming global policy.

The existing complete CPU job already installs pandas via the formula extra.
The new file is also explicit in the Python 3.9–3.12 regression matrix and the
ordered Torch CPU job. The latter installs `pandas>=1.5,<3`, imports pandas in
its environment check, and retains the real CPU-setting predecessor followed
by native-preservation/restoration tests. Pandas-dependent tests skip in
minimal installations; these maintained CI jobs install it deliberately.

## Final local validation

Final validation results are recorded below before publication. Counts from
focused suites, independent probes, and the complete suite overlap and must
not be added. Skips and expected failures are not passes.

- Complete final test tree with real Torch CPU installed: **7,145 passed,
  933 skipped, 166 expected failures, zero failed**, 88 warnings, 137.06 seconds.
  The eight-file tested payload was hashed before and after the run.
- New array-input suite: **481 passed, 14 physical-CUDA skips**, zero failed.
  Independent covariance/device plus new-input suite: **585 passed, 84
  physical-CUDA skips**, zero failed. Independent expanded input/API probes:
  **504 passed**, zero failed; a blocked-pandas subprocess also exercised all
  seven classes without importing pandas.
- Negative controls: all **210** new pandas matrix cases pass with the
  independently blob-verified exact-base covariance modules; all **210 fail**
  with the original PR-head preparation method restored, with the original
  keyed-`get` TypeError. The repaired matrix passes. Separate original 35-case
  controls were independently repeated at base, original head, and repair.
- Independent review found no remaining in-scope issue in the final eight-file
  payload. Its pre-staging HEAD/diff/untracked-content review fingerprint is
  `740951b9bedaf94f7451adb8e508fde3b0782764fd1031d93bd6bf45907e8bf9`.
  The companion manifest independently verifies all 17 changed/context entries.
- Documentation contracts: **169 files passed**; deterministic link check:
  **zero affected files**. Markdown rendering across **161 active pages,
  350 tables, 2,598 data rows and 1,879 local links** found no table-cell or
  file/heading-anchor problem. This is parser/render/link evidence, not a new
  deployed-browser or MathJax visual inspection.
- Compilation, Python 3.9 grammar for changed Python files, YAML parsing,
  whitespace checks, workflow high-signal Ruff, and default Ruff for the new
  test all pass. Differential production Ruff adds no diagnostics; its three
  pre-existing typing-modernization diagnostics remain.

Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
sklearn 1.8.0, Torch 2.14.1+cpu, pytest 9.1.1. CUDA is unavailable, Torch
reports zero CUDA devices, and CuPy is not installed.

## Physical-GPU evidence boundary

This follow-up **changes production covariance input preparation**. The
previous user's **222-case single-P100 acceptance at
`bf9833257d9726b4e8aaf199dc73a30cddd039c3` remains historical user-reported
evidence**; its raw logs were not independently inspected here and it does
not establish physical GPU acceptance of this candidate.

The minimum affected single-GPU rerun, on a machine with working CuPy and Torch
CUDA, is:

```bash
python -m pytest dev/tests/test_covariance_device_contract.py \
  dev/tests/test_covariance_array_inputs.py -q -ra -k 'physical and not nondefault'
```

This runs the existing **56** single-GPU routing/cross-library cases plus the
new **14** pandas/array-like physical cases, including nullable/mixed numeric
inputs. Record the exact tested commit, library versions, concrete device,
and full output. A skip is not acceptance of that backend.

The existing **14 nondefault-device/two-GPU cases remain separately unrun**.
They require two physical GPUs and can be selected with:

```bash
python -m pytest dev/tests/test_covariance_device_contract.py -q -ra -k nondefault
```

These 14 existing cases are distinct from the 14 new single-GPU pandas cases.
ANOVA/model-X production and their existing validators are unchanged by this
repair, so this narrow delta does not itself require rerunning those portions
of the old 222-case report. No physical GPU, R, timing, empirical-FDR, interval-
coverage or compiled DBSCAN-extension validation was performed here. The
inherited singular-Torch-covariance jitter exception and separately tracked
#243/#245 work are not repaired or relabeled. No merge, auto-merge, or readiness
transition is part of this task.
