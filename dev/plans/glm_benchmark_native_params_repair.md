# Restore native parameters for the GLM covariance benchmark check

Base: PR #251 commit `9a6145d187da1000703bc4a20d990ef72d44b893`, tree
`a9bb55f1460fc642eab0e8b5f0a8a8639bbb496c`.

## Evidence and root cause

The submitted P100 evidence archive has SHA256
`102c17d53c336ca8168b1a6cd17f47457b3f4c60be7a4a3c7251b126c00884ef`.
The source was clean and unchanged. All 284 required physical GPU pytest cases
passed, but the eight IG/NB × HC0/nonrobust × CuPy/Torch CUDA benchmark rows
failed with mixed NumPy/native-array TypeErrors. That physical acceptance remains
FAIL; this local change does not rewrite the historical result.

After successful public GLM inference, `ParameterInferenceResult.apply_to` stores
reporting parameters in NumPy, while `_X_design`, `_y_inf`, weights and curvature
retain their numerical backend. The benchmark's additional full-covariance check
passed host `_params` directly into `m_estimation_inference` with the native
design. Real Torch CPU reproduction confirms that fitting succeeds, then the
benchmark recomputation fails on mixed array arithmetic.

## Narrow repair

Before the extra covariance computation, the benchmark creates a temporary
parameter array on the design's backend, concrete device and dtype using
`xp_asarray(..., ref_arr=model._X_design)`. It does not mutate fitted model state.
Only the covariance result is converted to host for the external oracle.
The check stays outside the timed region. Public host reporting, production
numerical code, loss/inference formulas, convergence rules and all numerical
oracle thresholds remain unchanged.

Regression tests cover the four affected family/covariance cases with real Torch
CPU, assert native argument alignment and unchanged host reporting parameters,
and require a corrupted covariance reference to fail. Eight availability-aware
CuPy/Torch CUDA full benchmark cases provide the physical rerun gate.

## Acceptance boundary

Local tests are not P100 evidence. Run the eight targeted GLM benchmark rows
again on physical CuPy/Torch CUDA at the new source identity; preserve original
errors, unchanged thresholds, logs and JSON. Do not count unavailable-backend
skips as success. Existing production pytest evidence is historical for its
recorded source; the changed benchmark acceptance logic requires fresh evidence.
No commit, push, physical GPU execution, merge or release is performed here.
