# Learning benchmark execution-evidence repair

Base: PR #251 commit `9a6145d187da1000703bc4a20d990ef72d44b893`.

## Contract and impact

This is benchmark/evidence-only work. No production numerical method, backend
support, tolerance, or published release changes. The active axes are numerical
execution evidence, diagnostic timing, source identity, and regression tests.

The old adapter blocked host-returning APIs before calling them. A host return
is not proof of host computation, just as a native array is not proof of GPU
computation. The new adapter invokes the explicitly requested backend and
observes selected native numerical primitives inside source-identified stages.
Each stage invocation must independently supply its required arithmetic;
allocation, conversion and copying are not evidence. Wrong-device/host results
cannot satisfy a native requirement. The observer restores hooks on errors and
rejects conflicting profilers. Correctness references run outside observation.

| Consumer | Observed requirement | Honest disposition |
|---|---|---|
| Fixed/Model-X Knockoff, corr_diff | construction solve/eigh and statistic mean/abs per invocation | Native arithmetic within construction/statistic plus host threshold/report; not proof of every operator or FDR calibration |
| KDE normal CI | fit sum and density exp per call | Mixed backend density and host interval assembly |
| KDE bootstrap, CuPy | initial and every replicate fit/density, at least 21 for 20 draws | Mixed native fits/density, host RNG/resampling/quantiles/report |
| KDE bootstrap, Torch | Production rejects explicit replicate weights | Remains explicitly unsupported; not relabeled trace failure |
| Scott/Silverman bandwidth fit | KDE fit sum and selector entry | Native fit arithmetic plus host scalar selection |
| NRD/UCV/BCV/SJ bandwidth fit | KDE fit and specific host selector entry | Host selection with native fit arithmetic |
| Regression CV bandwidth | exp/sum in each CV score, host optimizer entry | Host optimization with observed native objective arithmetic |
| Scott low-level bandwidth helpers | Both selector invocations, no native arithmetic required | Host scalar rule, not GPU work |

## Interpretation and limitations

The observer is deliberately not an exhaustive GPU profiler. It does not trace
all array operators, library internals or fused kernels; native abs after a host
matrix product would not prove that matrix product ran on GPU. Results say
“selected native numerical primitive observations,” never full GPU residency.
The patch removes unjustified preemptive execution blocks only for supported
paths with the required observed stages. All instrumented rows remain
`performance_eligible=false`; measured public calls include observation overhead.
They are diagnostic runs, not canonical performance or speedup data. Other
learning cases retain their earlier ownership-only qualification.

Physical CuPy/Torch CUDA runs remain pending. Real Torch CPU observer tests
exercise tensor operations without asserting CUDA acceptance. New evidence must
record this changed runner/helper fingerprint; the old P100 guide is historical
for its locked source and cannot certify the new observer.

## Local validation

- Targeted CPU smoke for nine affected paths: complete, all correctness gates pass.
- Learning runner and observer harness: 90 passed, 30 physical CuPy/Torch CUDA cases skipped on this CPU-only host.
  The physical matrix covers 15 affected routes on each backend; the Torch
  bootstrap row asserts the existing unsupported boundary, while the other
  29 rows require successful numerical observation and correctness.
- Independent combined runner/observer/manifest/dispatcher regression: 453 passed,
  30 physical GPU skips, 14 existing warnings.
- All seven CPU smoke suites: 316 cases passed, dispatcher exit 0 and
  source_changed=false. These are harness/correctness checks, not GPU evidence.
- Compatibility rerun with Python 3.9 / NumPy 1.24.2 / scikit-learn 1.2.2 in an isolated dependency directory: 84 passed, 36 skipped (30 physical GPU cases plus six Torch-only tests; Torch is not installed in that environment). NumPy generated-wrapper recognition is tested without trusting arbitrary wrappers.
- No SSH, physical GPU execution, commit, push, merge or release performed.
