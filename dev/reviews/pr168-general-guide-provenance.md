# PR168 general-guide audience cleanup provenance

## Scope and source

This developer record preserves engineering notes removed from user guides during
the 2026-10-09 audience cleanup. Source snapshot:
`880b7b8474ab4728ed44a4556b1b64faaa6e7b55`, comparison base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The passages below were inherited
unchanged from that base. They are not new validation results or a change in
statistical support.

## Regression diagnostics

The former English and Chinese guide endings stated that reference tests compare
influence quantities with `statsmodels.OLSInfluence`. The reporting-side NumPy
transfer, pseudoinverse-based leverage and deleted-residual-variance contract remain
in the guides. The comparison claim belongs with the maintained test implementation,
not as a substitute for explaining these quantities to a user.

## Low-level Quantile compatibility

The solver algorithm reference previously described omitted/uniform-weight
`lbfgs_solver(QuantileLoss, ...)` compatibility as “regression-covered”. The actual
compatibility boundary remains public: estimator/CV Quantile `solver="lbfgs"` and
genuine non-uniform weights are unsupported. Removing the test-status qualifier
neither expands nor narrows that boundary.

## Float32 L-BFGS maintenance invariant

The earlier float32 guidance stated that its documentation did not change the
algorithm, statistical objective, analytic-weight semantics, solver selection or
explicit device behavior. Its closing instruction to maintainers was:

> Do not loosen the statistical objective or change solver semantics merely to
> force bitwise-like float32 coefficient agreement across backends.

That development invariant is preserved here. User guidance instead asks readers
to align objective, weights, penalty and stopping controls and choose float64 when
tighter coefficient agreement is required.

## Historical Cox screening design rationale

The Chinese guide additionally explained why experimental environment-variable
spellings were retained: preserving the experimental interface and diagnostics
would avoid redefining public configuration names for future screening research.
It asked future documentation authors to state which candidates enter the
high-precision stage, fold-budget allocation, whether approximate screening
changes the candidate set, and whether the final fit still uses all data.

These are design considerations, not a statement that screening is implemented
or scheduled. The current guide continues to say that the candidate grid is
fully evaluated, the execution mode is `exhaustive_safety_fallback`, and setting
experimental controls does not promise less computation.

## RidgeCV warning classification

The user warning is deliberately retained in both CV guides and the complete API
reference. [Issue #243](https://github.com/TheHiddenObserver/statgpu/issues/243)
tracks the implementation defect: without `sample_weight`, validation partitions
covering every row once can select complement-statistics reuse even when an
explicit training set is smaller than the validation complement. The consequence
is different training rows, validation losses and possibly selected alpha.
Ordinary complementary K-folds do not suffer that substitution. The supported
workaround described to users is external CV fitting `Ridge` on exactly the
specified training rows. No RidgeCV numerical logic is changed by this cleanup.
