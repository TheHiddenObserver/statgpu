# PR168 ANOVA input, covariance device, and GLM table repair

## Scope and source

This follow-up repairs existing public contracts. It contains runtime changes,
not only documentation qualification. The baseline is
`98b03a560a7b1444055bfa15e9014cf34aa89105`; the PR comparison base is
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. All 1,498 baseline files were
verified against the freshly retrieved Git tree before editing.

The [source manifest](pr168-input-device-source-manifest.json) records the exact
21 runtime, test, workflow, dependency, and user-documentation files in this
repair. It deliberately excludes this report and itself and is not a full-tree
fingerprint. The PR description records the published commit/tree and hosted
check results. Earlier model-X P100 evidence does not validate these changes.

## ANOVA finite observations

`f_oneway` now checks the concatenated native array for finite observations
before computing sums of squares. `cohens_f` inherits the same validation.
`f_twoway` checks the balanced native cell cube before computing cell means.
NaN, positive infinity, and negative infinity raise `ValueError`.

The finite-input contracts of Welch ANOVA, Tukey HSD, Bonferroni comparisons,
and `partial_eta_squared` were preserved and included in the regression matrix.
The change preserves group-size and residual-degree-of-freedom rules, unequal
one-way group sizes, balanced two-way designs, scalar result types, and the
defined NaN/infinite results of finite degenerate samples. It does not treat an
undefined statistic from finite constant observations as an invalid input.

Tests cover NumPy, real Torch CPU, and optional real Torch CUDA/CuPy fixtures;
invalid values in either group or opposite corner cells; explicit and automatic
group dispatch; float32/float64 finite controls; analytic/SciPy agreement; and
the top-level public aliases. Validation reduces on the native array rather
than copying observations to NumPy. The original head failed 78 selected
NumPy/Torch CPU guards and passed 72 controls.

This repair does not change the pre-existing two-way nested-input automatic
backend resolver or scalar F-distribution device resolution. Explicit-backend
tests do not constitute a broader repair of those routing behaviors.

## Covariance device ownership

The affected consumer inventory is `EmpiricalCovariance`, `LedoitWolf`, `OAS`,
`ShrunkCovariance`, `GraphicalLasso`, `GraphicalLassoCV`, and `MinCovDet`.
All fits now use one covariance-local preparation path. The historical shared
`_detect_backend` helper is unchanged for its panel and spline consumers.

| Requested policy | Required computation |
| --- | --- |
| Estimator `cpu` | NumPy CPU, including Torch/CuPy inputs |
| Estimator `cuda` | CuPy CUDA, or a clear unavailable-backend error |
| Estimator `torch` | Torch CUDA, or a clear unavailable-backend error |
| Estimator `auto`, explicit global policy | The global CPU/CuPy/Torch choice |
| Estimator and global policy both `auto` | Preserve native Torch/CuPy input selection; array-like CPU input uses normal automatic selection |

Thus a true-auto Torch CPU input remains a supported Torch CPU path; it is not
used as a fallback for an explicit GPU request. Indexed strings such as
`device="cuda:1"` are not new estimator parameter values. Existing native GPU
input affinity and backend device contexts control concrete CUDA indices.

`score`, `mahalanobis`, and `predict` follow the fitted arrays' backend and
concrete device, even when query inputs or the global policy differ. CPU
queries are allocated directly on the fitted GPU. `score` returns a Python
float; `mahalanobis` and `predict` return NumPy arrays after native computation.
GraphicalLassoCV resolves
once and retains the selected backend through folds, scoring, and final refit.
Each prepared fold still receives shape, minimum-sample, and finite checks;
the one-training-row case continues to raise. Covariance stabilization and
robust support allocations use their reference arrays' devices.

The regression matrix covers all seven classes, explicit/global policy,
default and true-auto behavior, unavailable runtimes, conversion and output
ownership, clone/device-parameter reconstruction, CV fold/final-refit routing,
and evaluation after a global-policy change. Exact-baseline negative controls
failed all 20 selected public-API guards and the repaired source passed them.
On the deterministic NumPy preservation fixture, all seven models' fitted
arrays, scores and distances, and the CV score grid/selected alpha, are exactly
equal before and after repair. Existing aligned sklearn comparisons also pass.

Covariance-specific coefficient inference is not introduced. Generic inherited
resampling helpers and unrelated non-device `set_params` lifecycle behavior
are outside this repair.

## GLM rendered parameter tables

Both shared GLM `C` rows now use `∥beta∥²/(4C)` instead of unescaped pipe
characters. The formula and wording are otherwise unchanged. Actual GFM
rendering of the original row lost the formula, `C=0` rule, and explicit-solver
exceptions; the repaired row preserves all content in three cells.

The mandatory Markdown-to-HTML regression extracts rendered cells and checks
the full description. Deliberate corruptions of pipes, denominator, `C=0`,
solver names, or column count all fail the guard. Pandoc independently confirms
the before/after result, and 22 tables per language were scanned. The parser is
a declared development/validation dependency, tested with markdown-it-py 3.0
and 4.2. It is not an optional skipped test or a runtime package dependency.

## Validation and review

Overlapping test counts below are not additive.

- Full CPU test tree: **6,069 passed, 1,474 skipped, 122 expected failures,
  zero failed**. Skips and expected failures are not correctness passes.
- Combined affected NumPy/real-Torch-CPU suites and rendered tables:
  **381 passed, 237 skipped**, with one existing sklearn convergence warning.
- Independent final new-regression run: **297 passed, 236 physical-GPU skips**.
- Bilingual ANOVA examples: **7 passed**; existing covariance and broader
  bilingual/example contracts are included in the full suite.
- Documentation contracts: **169 files passed**; deterministic link checker:
  **zero affected files**. Strict benchmark-data checking: **2,580 runs,
  47 models, zero errors and zero warnings**.
- Workflow high-signal Ruff, changed-source Python 3.9 grammar, compilation,
  whitespace checks, frontend typecheck/build, universal wheel/source build,
  and distribution metadata checks pass. Default Ruff retains 31 inherited
  diagnostics versus 37 on the same baseline files, with **no new diagnostics**.
- Independent review corrected a temporary CV fold-validation bypass and a
  missing sklearn dependency in the new Torch CPU CI step before publication.
  The Torch 2.0 CPU hosted job now explicitly runs both numerical regression
  files and installs their sklearn reference dependency.

Local numerical environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
pandas 3.0.6, statsmodels 0.15.0, sklearn 1.8.0; optional-backend validation uses
Torch 2.14.1+cpu. CUDA device count is zero and CuPy is not installed.
Small call-order sentinels test routing only and are not GPU execution evidence.

No new physical CUDA, nondefault-GPU, R, speed, empirical-FDR, or interval-
coverage result is claimed. Known issues #216, #221, #224, #243, and #245 are
unchanged by this repair. No merge, readiness transition, or issue change is
part of the work.

## Physical-GPU validation still required

Run on the exact published source after installing the appropriate CuPy extra
for the machine's CUDA major version, Torch CUDA, and validation dependencies.
First verify that both runtimes expose real CUDA hardware, then execute:

```bash
python -m pytest dev/tests/test_anova_finite_inputs.py -q -ra -k 'torch_cuda or cupy'
python -m pytest dev/tests/test_covariance_device_contract.py -q -ra -k physical
```

The covariance matrix includes NumPy/Torch-CPU/native-GPU inputs, cross-library
GPU conversion, numerical parity, CV/refit/evaluation, and nondefault-device
affinity. Nondefault-device cases need at least two physical GPUs. Record the
commit/tree, runtime versions, actual device identifiers, test counts, and
skip reasons. A skipped physical case or a CPU routing sentinel is not a pass
for that GPU contract.
