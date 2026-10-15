# Issue #160 — float32 CuPy L-BFGS parity investigation and repair plan

## Goal

Resolve the deterministic float32-design CuPy L-BFGS cross-backend discrepancy discovered by PR #151 schema-v6 without weakening the float64-oriented PR151 tolerance or changing generic `LossBase` semantics for unrelated losses.

## Change classification

This starts as a numerical investigation. A production change is authorized only after the diagnostics identify a concrete cause and the public float32 precision contract is explicit.

Active axes if a production fix is needed: solver/convergence, dtype/backend/device, GLM objective, tests/evidence, documentation. CV/inference are validation consumers only where the changed L-BFGS path reaches them.

## Baseline to reproduce

Use the historical schema-v6 logistic fixture plus additional deterministic seeds. Cover:

- NumPy, CuPy, Torch;
- float32 design/response with float32 analytic weights;
- unweighted and weighted paths;
- matching float64 solves on the same numerical data;
- ordinary GLM first, then a representative penalized L2 consumer if the same cause reaches it.

The original P100 discrepancy was on ordinary binomial L-BFGS with float32 X/y/weights and must remain an immutable historical observation rather than being rewritten.

## Diagnostic contract

For every backend/seed/weight mode record at minimum:

- executed backend/device and dtype;
- iteration count;
- final objective;
- final gradient norm;
- accepted step size per iteration or compact step summary;
- Armijo exact-vs-roundoff acceptance mode;
- number of stored curvature pairs and rejected/degenerate pairs;
- two-loop initial inverse-Hessian scale (`gamma`) summary;
- termination reason;
- coefficient/intercept error vs NumPy float32 and vs the float64 reference.

The diagnostic runner must not change production solver behavior merely to observe it.

## Cause isolation

Investigate in this order:

1. fused logistic value/gradient reduction differences;
2. float32 dot products in the two-loop recursion;
3. `y^T s` curvature-pair acceptance and reciprocal scaling;
4. initial inverse-Hessian scaling;
5. Armijo comparisons and the bounded roundoff gate;
6. gradient-norm / parameter-resolution stopping;
7. any backend-specific implicit promotion or scalar synchronization.

Use one-factor diagnostic variants only after the baseline is reproduced so causal attribution remains possible.

## Precision-contract decision

Choose exactly one reviewed public contract based on evidence:

### Option A — preserve native float32 L-BFGS

Use this only if NumPy/CuPy/Torch all converge to statistically/numerically acceptable float32 solutions and the observed cross-backend difference is ordinary floating-point path dependence. Define and document a dtype-appropriate float32 accuracy/parity criterion distinct from the existing float64 acceptance threshold.

### Option B — narrow GLM L-BFGS promotion

Use this if one maintained float32 backend is materially less accurate or unstable. Promote only the demonstrated GLM L-BFGS working state/reductions required for the maintained accuracy contract, while keeping explicit backend/device ownership unchanged. Do not promote generic non-GLM `LossBase` routes solely to satisfy this fixture.

A production fix must preserve float64 results and explicit device authority.

## Hosted tests

Regardless of the final option, add deterministic CPU/Torch-hosted coverage for:

- float32 vs float64 reference diagnostics on multiple seeds;
- weighted/unweighted objective consistency;
- global analytic-weight rescaling;
- final objective/gradient/termination provenance;
- no hidden CPU fallback for explicit Torch paths;
- unchanged float64 behavior.

CuPy-only assertions should be deterministic skips when CUDA/CuPy is unavailable in hosted CI, not masquerade as physical evidence.

## Physical investigation evidence

The diagnostic-only PR cannot choose Option A or B from hosted NumPy/Torch evidence alone. The canonical physical investigation command is:

```bash
python dev/benchmarks/run_issue160_diagnostic_matrix.py \
  --require-cuda \
  --output dev/reviews/issue160_float32_lbfgs_matrix.json
```

`run_issue160_diagnostic_matrix.py` is the complete three-mode matrix (`unweighted`, historical `weighted`, and globally `weighted_scaled`). It invokes the traced runner, verifies each trace against the production low-level solver and the public ordinary GLM route where available, covers all frozen seeds at float32 and float64, and requires both CuPy CUDA and Torch CUDA when `--require-cuda` is supplied. It records the exact clean source before execution and rechecks the same clean HEAD before publishing the artifact.

`diagnose_issue160_float32_lbfgs.py` is the lower-level trace implementation and a useful focused diagnostic, but its narrower weighted-only `run()` output is not by itself sufficient evidence for the Option A/B decision.

The investigation matrix intentionally records observed differences without defining a final float32 parity tolerance in advance. The physical results must be reviewed before selecting a public precision contract or changing production numerics.

## Physical CUDA gate after a production change

If the investigation selects Option B or otherwise changes CuPy/Torch L-BFGS production numerics, a separate frozen exact-source physical acceptance validator is required after the production change. That later validator must freeze its fixture matrix and reviewed tolerances before the acceptance run and record source SHA, clean-tree status, environment, concrete device, and full result summaries.

Without physical investigation evidence, Issue #160 remains unresolved. Without a fresh physical acceptance run after any production CUDA numerical change, the implementation may be review-clean/hosted-clean but remains physically unaccepted.

## Review/fix loop

1. Commit diagnostic runner + hosted tests without changing solver behavior.
2. Review the diagnostic design under `.claude/skills/code-review`.
3. Run the complete physical diagnostic matrix on an exact clean source and retain the artifact.
4. Review that evidence, select Option A or B, and record the precision-contract decision in a review addendum.
5. If needed, implement the narrow production fix and regression tests.
6. Run hosted CI on the exact head.
7. Fresh-review the complete PR diff and fix all actionable findings.
8. If production CUDA numerics changed, run the separately frozen physical validator on the exact clean final source and commit only immutable evidence afterward.
9. Recheck exact-head freshness and evidence provenance before completion.
