# PR 168 survival and smoothing documentation review

## Target and scope

- Target kind: PR, with an explicitly prepared writable materialized snapshot.
- Repository: `TheHiddenObserver/statgpu`, PR 168.
- Original comparison base: `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.
- Original exact head: `35e131edea5c27f1b5b6cd04564df6e41cd6b6b4`.
- Source files were verified against the exact-head remote Git blob tree before the review.
- Scope: EN/CN CoxPH, GAM, and kernel-smoothing learner pages, the survival/smoothing API reference, their public runtime help, and focused regression/evidence files.
- Change type: documentation/public-contract reconciliation. Active axes: public API, dtype/output ownership, CV scoring and reconstruction, GAM basis lifecycle, and documentation. No numerical behavior was changed.

## Documentation corrections

1. Completed runtime constructor help for GAM, CoxPH, and CoxPHCV. GAM now explains knot placement, GCV gamma, actual basis-dependent coefficient length, `gcv_score_=None` for fixed smoothing, NumPy prediction outputs, and ignored extra fit keywords. Cox help covers missing jobs/cleanup/custom-fold controls and correctly describes printed summaries.
2. Corrected CoxPHCV score terminology: folds contribute summed unpenalized partial log likelihood, averaged across evaluable folds without row/event normalization. Custom-grid numerical near-ties can choose stronger penalties with a slightly lower score. Candidate arrays preserve supplied grid order; one-shot split iterators are materialized and reused, including through `get_params` and clone.
3. Added float64 conversion/output contracts to the EN/CN reference and smoothing learner pages. Fixed `fit_kde` help to include the Torch array backend while not promising CUDA placement from that option alone.
4. Clarified that optional KDE `y` is not used statistically but is still subject to shared finite-input validation, and normal density intervals still require the accepted `bootstrap_method` control.
5. Removed an incorrect mathematical claim from `difference_penalty` help: adjacent-coefficient difference order does not determine spline polynomial degree.
6. Documented safe fresh-instance recovery after GAM basis-construction failure. Kept implementation tracking and evidence here rather than adding issue/PR history to learner pages.

## Implementation defect reported separately

[Issue 189](https://github.com/TheHiddenObserver/statgpu/issues/189) tracks a GAM refit that mutates basis metadata before a later validation failure and leaves old coefficients/EDF with `_fitted=True`. This is not the harmless case of preserving an entirely unchanged old fit after early validation rejection, and it does not require `set_params`.

Minimal reproduction: `dev/validation/repro_pr168_gam_failed_refit.py`.

Original-source locations:

- [Basis-state mutation before construction](https://github.com/TheHiddenObserver/statgpu/blob/35e131edea5c27f1b5b6cd04564df6e41cd6b6b4/statgpu/semiparametric/_gam.py#L279-L289)
- [Coefficient/fitted-state replacement only after successful numerical work](https://github.com/TheHiddenObserver/statgpu/blob/35e131edea5c27f1b5b6cd04564df6e41cd6b6b4/statgpu/semiparametric/_gam.py#L310-L321)

Observed on CPU: a successful one-feature fit has `n_features_=1`, one knot array, and nine coefficients. Refit on a two-feature finite design whose second column has 27 zeros and three ones raises `ValueError` about boundary knots. The object then has `_fitted=True`, `n_features_=2`, two new knot arrays, and the old nine coefficients. `summary()` presents these mixed fields; prediction fails. A safe implementation must either restore the complete previous fit transactionally or clear all fitted state. This documentation pass does not implement that repair.

## Validation

Command:

```bash
PYTHONPATH=. python -m pytest \
  dev/tests/test_pr168_survival_smoothing_contract_repair.py \
  dev/tests/test_survival_smoothing_api_reference.py \
  dev/tests/test_smoothing_learner_examples.py \
  dev/tests/test_coxph_learner_examples.py \
  dev/tests/test_cn_documentation_language.py \
  dev/tests/test_cn_documentation_naturalness_round2.py -q
```

Result on 2026-10-05: **81 passed**. The newly added focused file contributes 12 tests; it checks runtime help inventory, tied-feature basis size, fresh-instance recovery, float32-to-float64 conversion, unused-argument validation, normal-interval controls, an independent held-out Cox partial-log-likelihood formula on unequal folds, input grid order, split-generator cloning, and the custom-grid near-tie rule.

For each of the five edited production Python files, the original comparison copy was verified against the exact-head remote Git blob hash. Removing only module/class/function docstrings and comparing parsed ASTs showed **identical executable ASTs** before and after the edits.

No physical GPU run, R comparison, benchmark, or hosted CI is claimed by this scoped record. Existing physical-backend evidence is not being re-certified. The PR description records final remote-head validation separately.
