# PR168: Torch-to-NumPy covariance input repair

## Exact target and scope

This bounded public-input repair starts from PR168 head
`06852299515dfcf4577f4367c1f5fef9005a3168`, tree
`caf0b5e9f528381dbed42ee953dc688b30afcbda`, and comparison base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The initial clean source was
independently matched to all **1,510 remote blobs and executable modes**.
Its full-tree fingerprint was
`b2c3c9d696ac14f57f4a9a132978b84607a146e08585a29a66025e44678cc5bf`.

Classification: existing capability reconciliation / public contract repair.
Active axes are public input compatibility, dtype/view representation,
backend/device ownership, CV preparation/folds/scoring/refit, tests and evidence.
No covariance objective, solver, statistical definition, inference or benchmark
claim changes. The shared BaseEstimator, finite validator and backend converters
are unchanged.

The [source/validator manifest](pr168-covariance-torch-input-source.json)
identifies eight changed production/test/CI/user-doc payload files and ten
unchanged consumer/converter/validator context files. This record and the
manifest exclude themselves from the payload hash. Published commit, complete
tree identity and fresh hosted CI are recorded in the PR description after
publication. Previous records remain evidence for their recorded source only.
This is a review of this bounded repair, not another complete 330-path PR audit.

## Finding and correction

The preceding whole-PR review established a Torch-to-NumPy bridge regression:
all seven NumPy-fitted covariance estimators rejected `score`, `predict` and
`mahalanobis` inputs with real Torch `bfloat16`, `float8_e5m2` or an unresolved
negative bit. Those **63 query operations** passed on the exact PR base and
failed on the previous head. The same NumPy-target preparation also affected
direct fit. Earlier green suites and CI had not covered these representations.

The common covariance preparation boundary now handles actual real Torch
inputs destined for NumPy as follows:

1. Detach without changing the caller's tensor or autograd metadata.
2. Move the Tensor to the selected CPU destination.
3. Convert to Torch float64 and resolve its lazy-negative values.
4. Convert the representable Tensor to NumPy and continue existing preparation.

Float64 normalization must occur before `Tensor.numpy()`. It occurs after the
CPU move so a source device without float64 support does not acquire a new
restriction. Independent review caught that conversion-order issue in the
initial candidate; a routing-order regression and conditional MPS preservation
tests protect the final order. No physical MPS execution occurred here.

The fix is limited to the NumPy destination and real Torch input. It does not
route NumPy-fitted queries into native Torch to avoid conversion. It does not
expand complex, sparse, quantized or general float8 support. Public finite
checks still precede this helper: a dtype needs its installed Torch runtime's
finite-check and float64-conversion operations. In particular, the other
unsupported float8 formats are not silently accepted by bypassing validation.

## Consumer and ownership checks

- All seven public classes share the preparation boundary: EmpiricalCovariance,
  LedoitWolf, OAS, ShrunkCovariance, GraphicalLasso, GraphicalLassoCV and MinCovDet.
  Direct NumPy-target fits and all three NumPy-fitted query methods are covered.
- NumPy ownership, float64 fitted arrays, shape validation, finite-value
  rejection, fitted-array identity and values, and caller tensor/view/backing
  storage metadata are checked against the tensor's actual quantized values.
- Existing global/explicit CPU selection and fitted ownership survive later
  global CUDA/Torch policy changes. Strict unavailable-GPU behavior is unchanged.
- Genuine native AUTO, native Torch/CuPy routes and indexed-device conversions
  are unchanged. Existing pandas/nullable and ordinary array-like tests remain.
- GraphicalLassoCV's actual fold training/test values, candidate alphas, scores,
  selected alpha and final refit are traced across explicit CPU, global CPU and
  native AUTO. MinCovDet additionally has contaminated-support controls.
- EN/CN learner pages describe the NumPy transfer and autograd boundary; all
  three changelogs describe the bounded repair. No unsupported general
  cross-library low-precision or complex-covariance claim was added.

## Regression and CI coverage

`dev/tests/test_covariance_torch_inputs.py` contains **594 collected cases**:
**503 CPU**, **63 single-CUDA**, **21 two-CUDA/nondefault-source** and **7 MPS**.
The main matrices cover all seven estimators, ten input representations,
direct fit and each query method, including float16/32/64, integer, bool,
requires-grad, noncontiguous and real lazy-negative views. Tests preserve the
existing native-backend tolerances and use tighter 1e-12 NumPy-target parity.
No tolerance or expected-failure guard was weakened.

The new file is explicit in the Python 3.9–3.12 regression gate and the existing
ordered Torch 2.0.1 CPU job. Float8-only cases explicitly skip when that old
release does not provide the dtype; bfloat16 and negative-view coverage still
runs. A separate pinned **Torch 2.5.1 CPU** job requires the float8 dtype,
checks native float8 finiteness and float64 conversion, and runs all three
covariance input/device modules. It prevents optional float8 coverage from
silently skipping on every hosted environment. The pin uses the
[official CPU installation command](https://pytorch.org/get-started/previous-versions/#v251).
Local execution uses a different version and does not stand in for either
hosted pin; fresh hosted outcomes belong in the PR description.

## Final local validation and independent review

Counts below overlap; they must not be added. Skips and expected failures are
not passing tests.

- Entire final `dev/tests` tree with real Torch CPU: **7,648 passed / 1,024
  skipped / 166 expected failures / zero failed**, 90 warnings, 235.90 seconds.
  All 1,511 pre-evidence source blobs were verified unchanged before/after.
- An initial full-tree run read copied pytest bytecode containing the previous
  checkout path and failed 14 warning-filename assertions. The failed log was
  retained. A fresh bytecode prefix passed all **197 affected-module tests**;
  the final full-tree result above uses that fresh prefix with unchanged source.
- New module: **503 passed / 91 physical-device skips / zero failed**.
  Three affected covariance input/device modules: **1,088 passed / 175 skipped**.
- Exact local equivalent of the updated ordered ANOVA/covariance/native Torch
  command: **1,279 passed / 342 skipped / zero failed**. It retains the real
  CPU-setting predecessor and global-policy restoration regression.
- The new primary regression matrix passes all **84** fit/query cases. Restoring
  only the exact previous head's preparation helper makes **84/84 fail**, with
  exactly 28 BFloat16, 28 Float8_e5m2 and 28 negative-bit errors, and zero skips.
  The independent reviewer separately repeated this mutation control.
- Independent final review found no remaining in-scope finding in the eight-file
  payload. A fresh **556-check** independent probe passed: 472 positive,
  ownership/CV/finite/unsupported-preservation checks and 84 expected old-helper
  failures. The reviewer independently verified the complete frozen 1,511-file
  pre-evidence source and repeated the new suite. The changed-payload fingerprint
  is `04fb6ffaaff4cb9809d049d9fc9b09cb4356ad0701414c1d66158388a98ee294`.
- Documentation contracts: **169 files passed**; deterministic links: zero
  affected files. Render/navigation checks covered **161 active pages, 350
  tables, 2,598 data rows and 1,879 local links**, with zero problems. This is
  Markdown parser/render/link evidence, not new browser/MathJax visual proof.
- Python 3.9 grammar, workflow YAML, whitespace, high-signal Ruff and default
  Ruff for the new tests pass. Differential production Ruff adds no diagnostics;
  three existing typing-modernization diagnostics remain.

Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
sklearn 1.8.0, Torch 2.14.1+cpu and pytest 9.1.1. Torch reports no CUDA,
zero CUDA devices and no usable MPS runtime; CuPy is absent.

## Physical-device rerun boundary

This changes production covariance preparation. Historical user-reported
**222 single-P100 cases at `bf9833257d9726b4e8aaf199dc73a30cddd039c3`** do not
validate this candidate. Their raw logs were not independently inspected here.
No physical CUDA or MPS run is claimed.

Run the affected single-CUDA suite with working CuPy CUDA, Torch CUDA and a
Torch version that supports the float8 operations used by the tests:

```bash
python -m pytest dev/tests/test_covariance_device_contract.py \
  dev/tests/test_covariance_array_inputs.py \
  dev/tests/test_covariance_torch_inputs.py -q -ra \
  -k 'physical and not nondefault and not physical_two_gpu and not physical_mps'
```

Fresh collection is **133 cases**: 70 existing routing/cross-library/pandas
cases plus 63 new Torch conversion/native-preservation cases. A skip is not
acceptance of its backend, dtype or device path.

The distinct two-GPU/nondefault-source selection is **35 cases**, comprising
14 existing cases and 21 new Torch-to-NumPy cases:

```bash
python -m pytest dev/tests/test_covariance_device_contract.py \
  dev/tests/test_covariance_torch_inputs.py -q -ra \
  -k 'nondefault or physical_two_gpu'
```

Optional preservation of an existing MPS-to-NumPy input transfer has **7 cases**:

```bash
python -m pytest dev/tests/test_covariance_torch_inputs.py -q -ra -k physical_mps
```

Record the exact tested commit, environment/library versions, concrete device
and complete output. CPU routing spies are not physical-device evidence.
ANOVA/model-X production, numerical objectives and their validators are unchanged.
No new R, timing, empirical-FDR, interval-coverage or compiled DBSCAN validation
is claimed. Unrelated singular-Torch covariance behavior and #243/#245 work are
unchanged. No merge, auto-merge or readiness transition is part of this repair.
