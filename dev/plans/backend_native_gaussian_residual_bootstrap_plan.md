# Backend-native Gaussian residual bootstrap plan

Status: IMPLEMENTATION / REVIEW-FIX
Issue: #145
Parent implementation contract: PR #142 at `6d3c51c54cb2571b547b730d3c98cdc66071a545`

## 1. Scope

Extend the already-defined PR #142 **unweighted Gaussian residual-bootstrap** inference method from NumPy-only execution to the maintained NumPy/CuPy/Torch backends without changing its statistical estimand or DGP.

This work is an execution-capability closure, not a bootstrap-method expansion.

In scope:

- squared-error penalized models whose PR #142 resolver already accepts `inference_method="bootstrap"`;
- unweighted data only;
- `cov_type="nonrobust"` only;
- identical deterministic residual-index schedules across backends;
- backend/device-native bootstrap response construction and child refits;
- existing NumPy reporting boundary;
- `PenalizedGLM_CV` selected-final-refit-only inference;
- exact-source physical CuPy/Torch CUDA evidence.

Explicitly out of scope:

- weighted bootstrap semantics;
- wild/HC bootstrap;
- HAC/block/time-series bootstrap;
- non-Gaussian parametric bootstrap;
- Cox bootstrap;
- batched/multi-bootstrap performance optimization.

## 2. Statistical contract

For a successful penalized Gaussian fit with fitted values `y_hat` and residuals `r = y - y_hat`, draw one deterministic backend-neutral residual-index schedule from `bootstrap_random_state`.

For draw `b`:

`y_star[b] = y_hat + r[index[b]]`.

Refit the same penalized Gaussian estimator on `(X, y_star[b])` with the same fixed penalty/tuning contract. The bootstrap target remains the penalized coefficient distribution.

Published summaries remain:

- sample standard deviation of bootstrap parameter vectors;
- sign-based two-sided p-values;
- percentile confidence intervals;
- approximate statistic `params / bse` retained for the established result schema.

## 3. Backend/device contract

The parent fit's recorded `_selected_backend_name` / `_selected_backend_device` are authoritative.

- NumPy/CPU parent → NumPy/CPU bootstrap refits.
- CuPy `cuda:k` parent → CuPy `cuda:k` bootstrap refits.
- Torch `cuda:k` parent → Torch `cuda:k` bootstrap refits.

The residual-index matrix may be generated on the host as deterministic control-plane integer data so all backends consume exactly the same draws. The bootstrap response and each child optimization remain on the executed backend/device.

Each child refit is inference-disabled and its existing post-fit NumPy parameter snapshot may be collected for final bootstrap reporting. Any child backend/device drift is a hard failure rather than silent fallback.

## 4. Refit ownership

Preserve where applicable:

- penalty object/family;
- alpha;
- l1_ratio;
- penalty kwargs;
- intercept behavior;
- canonical solver request;
- stopping rule;
- LLA controls;
- backend/device;
- no recursive inference.

For GPU refits the deprecated `cpu_solver` alias is intentionally not made authoritative; canonical `solver` owns execution.

## 5. Implementation shape

Install a focused contract after the PR #142 inference and fit-transaction installers.

The focused layer:

1. intercepts only the `resolved == "residual_bootstrap"` post-fit row;
2. keeps all non-bootstrap inference routes delegated unchanged to PR #142;
3. bypasses only PR #142's deliberate CPU-only safety guard;
4. converts X/y/coef to the fit-recorded backend/device through maintained Gaussian backend helpers;
5. constructs one shared residual-index schedule;
6. performs serial backend-native child refits;
7. verifies child execution provenance;
8. publishes the existing residual-bootstrap result/provenance.

## 6. Hosted validation

Required deterministic coverage:

- backend-neutral schedule reproducibility;
- NumPy bootstrap reproducibility;
- penalty preservation (L1, ElasticNet, representative SCAD/MCP where stable);
- weighted bootstrap remains fail closed;
- `n_bootstrap < 2` remains transactional/fail closed;
- Torch backend consumes native bootstrap responses in host-only contract tests;
- child backend/device drift fails closed;
- `PenalizedGLM_CV` inference remains selected-final-refit-only;
- import/installer idempotence and sklearn reconstruction remain unaffected.

Run the full hosted matrix after the final source head is fixed.

## 7. Physical CUDA gate

Add an exact-source Tesla P100 validator with fixed residual-index schedules and a NumPy reference. Cover at least:

- L1 direct fit;
- ElasticNet direct fit;
- CuPy CUDA;
- Torch CUDA;
- Torch→CuPy and CuPy→Torch heterogeneous input crossings where the public fit boundary supports them;
- coefficient/bootstrap-summary parity;
- recorded backend/device provenance;
- no CPU numerical fallback.

Nonconvex SCAD/MCP may be added to the physical gate only if the maintained LLA path is stable enough for a deterministic acceptance threshold; hosted contract coverage is still required.

## 8. Review closure

Fresh review must inspect:

- statistical scope did not broaden beyond PR #142;
- identical resampling schedule semantics;
- child estimator/penalty/solver ownership;
- no backend fallback;
- concrete device affinity;
- final-refit-only CV semantics;
- result provenance/reporting boundary;
- docs/support matrix;
- exact-source hosted and physical evidence.

Do not claim COMPLETE until the exact numerical source has a clean CuPy/Torch physical artifact.