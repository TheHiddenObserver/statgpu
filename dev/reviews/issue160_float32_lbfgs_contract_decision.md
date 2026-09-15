# Issue #160 — float32 L-BFGS precision-contract decision

## Decision

**Option A is selected: preserve native float32 L-BFGS.**

The physical diagnostic matrix does not support a CuPy-specific working-precision promotion. The historical PR151 schema-v6 coefficient mismatch is a float32 path-dependence/stopping effect, not evidence that CuPy is materially less accurate or unstable than the maintained alternatives.

No production L-BFGS numerical code is changed by this decision.

## Evidence source

Canonical physical evidence:

- diagnostic numerical source: `0cf1ec085dc6a1cb0942a5497a3d0b0a8e02f8a3`;
- evidence commit: `e90de58c50e8f21ad2ce33d1cefde15684c6685a`;
- artifact: `dev/reviews/issue160_float32_lbfgs_matrix.json`;
- schema: 1;
- status: `diagnostic_complete`;
- `physical_cuda_complete: true`;
- source clean before and after execution;
- Tesla P100-SXM2-16GB, CuPy 13.6.0, Torch 2.0.0+cu117, NumPy 1.24.2.

The matrix covers float32/float64 × seeds `{151025, 16001, 16002, 16003}` × modes `{unweighted, weighted, weighted_scaled}` × backends `{numpy, torch, torch_cuda, cupy}`. Every trace is checked against the production low-level solver, and every available ordinary-estimator bridge reproduces the traced parameters.

## Why Option B is rejected

### 1. The historical CuPy discrepancy does not indicate worse convergence

For historical seed `151025`, float32 CuPy is much closer to its own float64 solution than float32 NumPy is to its own float64 solution.

Representative weighted row:

- CuPy float32 vs CuPy float64: parameter max-absolute error about `6.33e-8`, with final gradient norm about `7.45e-9`;
- NumPy float32 vs NumPy float64: parameter max-absolute error about `1.77e-4`, with final gradient norm about `3.85e-5`.

The raw CuPy-vs-NumPy float32 coefficient gap therefore mostly reflects the NumPy float32 path stopping at a different finite-precision point. Promoting CuPy alone would move the already-better-converged path, while leaving the NumPy float32 path that generated the historical reference untouched.

### 2. Additional seeds reproduce dtype-level path dependence, not a CuPy-only defect

Across the additional seeds, raw float32 parameter differences versus NumPy reach roughly `9.09e-4`, while objective differences remain around float32 rounding scale (at most about `1.2e-7` in this matrix). For seeds `151025`, `16001`, and `16002`, CuPy's float32 solution remains within roughly `1e-7` of its own float64 parameters even when its raw difference from NumPy float32 is several `1e-4`.

Float64 cross-backend solutions agree essentially to floating-point roundoff, confirming that the mathematical objective and backend implementations reconcile once the working precision is high enough.

### 3. Weight-rescale sensitivity is not CuPy-specific

Seed `16003` is deliberately informative. Global rescaling of the same analytic weights changes the float32 path by about:

- CuPy parameters: `3.184e-4`;
- NumPy parameters: `2.731e-4`;
- Torch CPU parameters: about `1.49e-6`;
- Torch CUDA parameters: about `1.07e-6`.

The corresponding objective differences remain at about `1e-7` or below. This is finite-precision line-search/stopping path sensitivity shared by more than one backend, not a CuPy-specific failure mode.

### 4. Trace evidence points to late finite-precision stopping differences

The traces agree closely in the early L-BFGS iterations. Divergence becomes material only near convergence, where tiny reduction differences alter curvature history, Armijo backtracking, and parameter-resolution stopping. In the historical NumPy path, the solver can terminate with `parameter_step` after the accepted step rounds to zero while the gradient is still around `1e-5`–`1e-4`; CuPy often continues to the gradient criterion.

This is ordinary finite-precision algorithmic path dependence for the existing float32 implementation. It is not evidence of a wrong CuPy objective, wrong gradient, hidden fallback, or broken public backend authority.

## Reviewed float32 acceptance contract

Float32 acceptance must not treat the NumPy float32 parameter vector as an exact oracle. A maintained float32 L-BFGS row is accepted only when all of the following hold:

1. **Trace fidelity**: traced parameters exactly reproduce the production low-level solver (`trace_matches_production_max_abs == 0`).
2. **Public-route fidelity**: where the public ordinary estimator bridge is available, it reproduces the traced parameters and executed solver/backend/device.
3. **Same-backend float64 reference**:
   - objective absolute error `<= 2e-6`;
   - parameter max-absolute error `<= 2e-3`.
4. **Optimization certificate**: final gradient norm `<= 2e-4`. `parameter_step` termination is allowed only as a finite-precision stopping mode within the same objective/parameter bounds; it is not by itself evidence of cross-backend failure.
5. **Cross-backend float32 diagnostics**:
   - objective absolute difference versus NumPy float32 `<= 2e-6`;
   - parameter max-absolute difference versus NumPy float32 `<= 2e-3`.
   The parameter bound is diagnostic rather than an oracle-equivalence claim.
6. **Global analytic-weight rescaling**:
   - objective absolute difference `<= 2e-6`;
   - parameter max-absolute difference `<= 1e-3`;
   - gradient-norm absolute difference `<= 2e-4`.

These bounds are deliberately separated from the existing float64 contract. They are wide enough to cover the physically observed float32 path dependence with margin, while still being orders of magnitude tighter on objective/stationarity than a contract based only on coefficient similarity.

## Float64 contract

Float64 remains the strict reference regime. The physical matrix shows cross-backend and global-weight-rescaling differences at or near float64 roundoff. The existing float64-oriented acceptance requirements are not weakened by this decision.

## Consequences

- Do **not** promote CuPy float32 L-BFGS working arrays/reductions solely for Issue #160.
- Do **not** change explicit backend/device authority.
- Do **not** reinterpret PR151 schema-v6's accidental float32-design fixture as a float64-parity requirement.
- Keep the historical schema-v6 failure immutable as evidence of why a dtype-specific contract was needed.
- Add an executable hosted check that applies the reviewed Option-A contract to the committed physical artifact.
- Because this decision changes no production CUDA numerical path, a new physical P100 run is not required. Any later production L-BFGS numerical change would reopen exact-source physical acceptance.
