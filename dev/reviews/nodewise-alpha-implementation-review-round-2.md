# Node-wise alpha implementation review — round 2

## Target identity

- target kind: production-runtime implementation review for PR #139
- comparison base: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- exact production-runtime head: `33ea9b768f832902b0007e0fb1b09822de5eb967`
- default branch: `master`
- branch: `fix/nodewise-alpha-inference-contract`

Subsequent branch commits reviewed in this pass are limited to CI/test/reference-documentation closure (consolidated workflow paths, cache/public-API tests, and EN/CN migration notes). They do not modify the production numerical/runtime files reviewed at `33ea9b7...`. A final PR-wide verdict must still re-resolve the then-current remote head and its CI; this artifact does not transfer the exact runtime SHA to later production changes.

## Fresh review result

**Production runtime: REVIEW CLEAN. PR-wide completion: REMOTE/HOSTED EVIDENCE STILL OPEN.**

No CRITICAL, HIGH, or actionable MEDIUM runtime finding remains in the consolidated production implementation.

## Re-reviewed statistical contract

- `nodewise_alpha` is independent of the main penalized-model `alpha`.
- explicit finite positive real scalars are authoritative;
- `None` uses the standardized design-side rule rather than main-response residual scale;
- analytic weights use response-independent Kish effective sample size for automatic tuning;
- canonical #138 centered/weighted working-data ownership is reused rather than reconstructed;
- standardized node-wise precision is back-transformed as `D^{-1} Theta_Z D^{-1}`;
- the paper-style `tau_j^2` normalizer is checked against the residual cross-product/KKT identity;
- every multi-feature solve must pass an independent full KKT publication gate;
- p=1 uses analytic univariate precision without pretending that a nuisance node-wise Lasso ran;
- degenerate/non-finite scale, precision, KKT, or normalizer state fails closed;
- marginal and requested simultaneous inference remain one publication transaction;
- `nodewise_alpha_` and provenance are cleared on failed/refitted inference.

## Backend / CV / API review

- NumPy, CuPy, and Torch builders implement the same statistical definition.
- explicit GPU inference has no CPU numerical fallback; scalar control/provenance sync and cache identity hashing are not substituted CPU numerical inference.
- GPU standardization, Gram construction, FISTA, KKT, normalizer, back-transform, and #138 simultaneous numerical work remain on the concrete execution device.
- the node-wise cache is now a normal helper rather than a second install-time contract; it keys the actual precision problem and rebuilds request-specific provenance on every call.
- `LassoCV` / `ElasticNetCV` treat `nodewise_alpha` as final-refit inference configuration only; it does not enter candidate scoring/selection.
- public signatures, `get_params`, `set_params`, sklearn clone, warning stacklevel, formula parity, and stale-state clearing are covered by dedicated tests.

## Architecture disposition

One compatibility installer remains: `statgpu.linear_model._nodewise_alpha_contract`.

This is accepted as a bounded integration layer rather than a preferred new numerical architecture. The existing #135/#138 runtime already installs constructor/inference compatibility wrappers and captures centered/weighted debiased originals at import time. Replacing that architecture solely to make this change source-static would materially broaden the task. The new statistical implementation and cache are ordinary helper modules; the single installer only bridges the new public/runtime contract into the already-installed #138 ownership chain.

Final-refit `Lasso` / `ElasticNet` inheritance from the CV owners uses a deliberately narrow internal call-stack match to the exact maintained CV module/function and is regression-tested. This is residual technical debt, but after removal of the earlier `CV.fit` wrapper it avoids changing the established #135 warning callsite contract. It is not classified as an actionable blocking finding in this repair.

## Review/fix changes since round 1

Round-1 findings were addressed as follows:

- multiple new installer layers -> consolidated to one compatibility installer; precision/cache are normal helpers;
- hosted simulation provenance -> records source SHA separately from synthetic checkout SHA;
- cache double-Gram -> removed; cache identity no longer computes an extra full Gram before builder execution;
- durable migration record -> EN/CN `nodewise-alpha-migration.md` documents the intentional default correction and absence of a legacy response-dependent mode.

The implementation also tightened iterative FISTA stopping to `1e-8` and increased the internal budget to 3000 after maintained CPU fixtures showed that the original `1e-5`/500 iterative settings could stop before the independent `1e-5` KKT publication gate. The publication KKT threshold itself was not relaxed.

## Evidence status

Earlier exact runtime-compatible heads established that the numerical/API implementation can pass the repository regression, maintenance, Gaussian-inference, dedicated node-wise, simulation, and static gates. Those earlier green runs are diagnostic only after later non-runtime branch changes.

Remaining before PR-wide clean/completion:

1. re-resolve the latest remote PR head and require its hosted Tests, Maintenance compatibility, Gaussian inference, dedicated Node-wise alpha inference, and relevant docs/frontend checks to pass;
2. execute `dev/benchmarks/validate_nodewise_alpha_gpu.py` on physical CUDA with both CuPy and Torch and record exact-source schema-v1 evidence;
3. perform one final PR-wide freshness review after the physical evidence source SHA is fixed.

Until those are closed, keep PR #139 Draft and unmerged.
