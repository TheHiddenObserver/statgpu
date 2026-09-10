# Node-wise alpha plan review — round 4

Reviewed plan commit: `236fcb7fbc4e5a51f3e829068c22ae80268ccd14`
Base master: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
Verdict: **PLAN CHANGES REQUIRED (no new HIGH findings)**

Fresh review result: the statistical transformation, weighted effective-n rule, p=1 analytic path, KKT gate, and tau normalizer are internally consistent. Remaining findings are contract-precision issues that should be closed before implementation.

## Findings

### MEDIUM — intentional default numerical behavior change needs an explicit compatibility decision

Before this repair, omitting any node-wise control gives the response-dependent historical rule on an unstandardized node-wise design. After this repair, omission gives a standardized, design-side auto rule. That changes default debiased inference numerics for existing `Lasso(..., inference_method="debiased")` callers even though the new constructor parameter is additive.

The plan must explicitly classify this as an intentional statistical-default correction authorized by the task. State that there is no legacy public compatibility mode because the old node-wise tuning rule was not user-addressable and has a response-unit defect. Changelog/docs and old-vs-new validation must make the behavior change visible.

### MEDIUM — internal node-wise stopping policy is still missing from the contract

The plan defines FISTA, solver tolerance, iteration cap, and an independent KKT acceptance gate, but does not state the `stopping=` value passed to the underlying FISTA implementation. This affects iteration behavior and reproducibility.

Because existing batched `stopping="kkt"` logic is not the authoritative full active-coordinate KKT calculation used by this repair, prefer:

`NODEWISE_STOPPING = "coef_delta"`

for the underlying iterative stop, followed by the independent full KKT acceptance gate. Record both `nodewise_stopping` and `nodewise_kkt_tol` in metadata. If implementation later replaces/fixes the solver's own KKT mode and wants to use it, that is allowed only with equivalent regression evidence and the independent post-solve gate still remains authoritative.

### MEDIUM — atomic publication must include requested simultaneous calibration

When `enable_simultaneous_inference=True`, a successful marginal precision/debiasing stage is not the end of the public fit transaction. `nodewise_alpha_`, precision provenance, and simultaneous fields must not survive if max-|Z| calibration later fails. Clarify that the atomic commit boundary is after all requested marginal + weighted/centered finalization + simultaneous calibration succeeds, with cleanup on any later failure.

### LOW — metadata should distinguish requested from resolved/not-applicable tuning

For p=1 an explicit constructor `nodewise_alpha` is valid but unused. To make provenance unambiguous, optionally record `nodewise_alpha_requested` in metadata on both p>=2 and p=1 paths. This is useful but not blocking if the constructor/get_params state and `nodewise_alpha_source="not_applicable"` are already clear.

## Required fixes before final plan verdict

1. Record the intentional default-behavior migration and no-legacy-mode decision.
2. Define and provenance-record the underlying node-wise stopping policy.
3. Extend atomic publication through simultaneous inference when requested.
4. Decide whether to add requested-value metadata; either choice should be explicit.

After these edits, perform a fresh review rather than inheriting this no-HIGH state.