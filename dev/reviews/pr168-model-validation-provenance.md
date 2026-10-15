# PR168 model validation provenance

## Scope and source identity

This developer record preserves the validation inventory and evidence-scoping
prose relocated from the bilingual ANOVA, covariance, and LogisticRegression
learner pages. The audience repair starts from PR168 head
`880b7b8474ab4728ed44a4556b1b64faaa6e7b55`, with comparison base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The copied paragraphs below are
historical statements from that head, not new test results or evidence for a
later revision.

The ANOVA validation paragraphs, covariance opening qualifiers, and English
covariance validation paragraph were inherited unchanged from the comparison
base. The Chinese covariance validation paragraph retained the same content
while translating CV refit and benchmark-artifact terminology. The logistic
comparison paragraphs were new at the recorded head and replaced a more detailed
validation catalog. The repair changes documentation placement and comparison
guidance; it does not change the statistical implementation or test definitions.

## ANOVA

Relevant maintained sources:

- [`test_anova_p2.py`](../tests/test_anova_p2.py): one-way/two-way and Welch
  tests, Tukey HSD, Bonferroni, and effect-size cases.
- [`TestWelchBackendAndReference`](../tests/test_second_full_review.py):
  `test_numpy_matches_statsmodels_welch_anova`,
  `test_torch_cpu_matches_numpy_without_full_numpy_fallback`, and
  `test_partial_eta_rejects_invalid_sum_of_squares`.
- [`test_three_backend_native_followup.py`](../tests/test_three_backend_native_followup.py):
  backend-preserving post-hoc comparisons and distribution boundaries.

The Welch reference uses the same groups with
`statsmodels.stats.oneway.anova_oneway(use_var="unequal", welch_correction=True)`.
Compare the statistic, p-value, and numerator/denominator degrees of freedom;
keep the function's fractional denominator degrees of freedom and NaN Welch
eta-squared semantics distinct from ordinary pooled-variance ANOVA. Torch CPU
parity tests are not physical CUDA evidence.

## Covariance

Relevant maintained sources:

- [`test_covariance_p2.py`](../tests/test_covariance_p2.py): shrinkage, robust
  support, Graphical Lasso and CV cases, including sklearn comparisons.
- [`test_module_review_covariance_panel.py`](../tests/test_module_review_covariance_panel.py):
  empirical inversion, feature validation, sklearn Graphical Lasso alignment,
  CV-grid checks, and robust support/centering behavior.
- [`test_three_backend_native_followup.py`](../tests/test_three_backend_native_followup.py):
  Torch CPU array preservation and NumPy comparisons for Graphical Lasso,
  GraphicalLassoCV selection/refit, and MinCovDet support.
- [`test_third_full_review.py`](../tests/test_third_full_review.py):
  `test_covariance_estimators_reject_nonfinite_and_empty_features`.

Backend availability denotes a public execution path, not a blanket numerical
or performance certification. Hardware-specific accuracy/performance claims
must identify the estimator, backend, concrete device, parameters, environment,
source SHA or scoped fingerprint, validator, and result artifact. A test's
presence is an inventory entry; only an execution result demonstrates what ran.
Use its actual tolerances and selected cases rather than converting them into a
universal acceptance threshold. Record skipped and unrun backends explicitly.

## Logistic regression

[`test_external_consistency.py`](../tests/test_external_consistency.py) contains:

- `test_logistic_estimation_and_inference_match_statsmodels`
- `test_logistic_robust_covariance_matches_statsmodels`
- `test_logistic_robust_covariance_gpu_matches_statsmodels`

At the recorded source head these tests use `C=1e10`, an approximately
unpenalized fit, and the CuPy cases are conditional on CUDA availability. The
learner guidance now recommends the exact unpenalized setting `C=0` for an
ordinary unweighted comparison with statsmodels Logit; this does not rewrite
these historical test settings or claim that the tests were rerun. No arbitrary
analytic-weight support in statsmodels Logit is implied. Penalized comparisons
require an equivalent objective and penalty scale, and covariance comparisons
must also align the inferential target and corrections.

## Evidence and acceptance boundary

All validation claims remain scoped to the exact function or estimator, source,
backend, environment, parameters, and executed cases recorded in the associated
result. Fresh runs, skips, and any accepted evidence reuse belong in developer
review or benchmark artifacts. Earlier commit-anchored results do not by
themselves certify this documentation repair or a later revision. See also the
[general model validation reference](../references/model-validation.md).

No new numerical, physical-GPU, R, or performance result is asserted here.
The learner pages retain hypotheses, formulas, covariance assumptions, backend
and transfer boundaries, numerical safeguards, failure behavior, and complete
examples. Existing validation-section anchors continue to resolve to actionable
comparison guidance.

## Relocated source text

These paragraphs are copied verbatim from the recorded head. The immutable
links identify their original location; they are retained for provenance rather
than presented as current execution evidence.

### anova: EN validation paragraph

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/models/anova.md)

> Maintained tests compare Welch ANOVA with `statsmodels.stats.oneway.anova_oneway`
> and exercise NumPy/Torch parity, degrees-of-freedom semantics, balanced-design
> restrictions, effect-size validation, and backend execution boundaries.
> Validation claims remain scoped to the exact function, backend, environment, and
> commit tested.

### covariance: EN validation paragraph

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/models/covariance.md)

> Maintained tests cover finite-input validation, backend-preserving fitted arrays,
> reference comparisons with scientific Python covariance estimators, robust support
> semantics, sparse-precision convergence, and CV refit behavior. Hardware-specific
> accuracy and performance evidence belongs to the corresponding maintained test or
> benchmark artifact.

### covariance: EN opening qualifier

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/models/covariance.md)

> The public estimators expose NumPy, CuPy, and Torch execution paths. Backend
> availability means that the public path exists; numerical and performance claims
> remain scoped to the exact estimator, backend, hardware, and commit tested.

### logistic regression: EN comparison paragraph

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/en/models/logistic-regression.md)

> CPU/GPU estimates and covariance calculations are compared with statistical reference implementations under aligned settings, including near-unregularized comparisons with `statsmodels.Logit`. Such comparisons are scoped to their data and covariance assumptions, not universal accuracy guarantees.

### anova: CN validation paragraph

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/cn/models/anova.md)

> 维护测试将 Welch ANOVA 与 `statsmodels.stats.oneway.anova_oneway` 对齐，并覆盖
> NumPy/Torch 一致性、自由度语义、平衡设计限制、效应量验证和后端执行边界。
> 所有验证结论仅适用于记录中的具体函数、后端、环境和 commit。

### covariance: CN validation paragraph

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/cn/models/covariance.md)

> 维护测试覆盖非有限输入验证、后端保持的拟合数组、与科学 Python 协方差估计器的
> 参考比较、稳健支持语义、稀疏精度收敛和 交叉验证后的重拟合行为。硬件相关的准确性和性能
> 证据应记录在相应维护测试或 基准测试记录 中。

### covariance: CN opening qualifier

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/cn/models/covariance.md)

> 公共估计器提供 NumPy、CuPy 和 Torch 执行路径。这里的后端支持表示公共路径存在；
> 数值与性能结论仅适用于相应测试记录中的具体估计器、后端、硬件和 commit。

### logistic regression: CN comparison paragraph

[Original page](https://github.com/TheHiddenObserver/statgpu/blob/880b7b8474ab4728ed44a4556b1b64faaa6e7b55/docs/cn/models/logistic-regression.md)

> CPU/GPU 的估计与协方差在对齐设定后与统计参考实现进行数值对照，包括近乎无惩罚时与 `statsmodels.Logit` 的比较。这些比较以相应数据与协方差假设为前提，不是对所有问题的统一精度保证。
