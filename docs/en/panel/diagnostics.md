# Panel Diagnostics

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/diagnostics.md)

## Overview

Panel diagnostics help answer model-selection questions such as whether fixed effects are needed or whether a random-effects specification is compatible with the fixed-effects estimate. Each test returns a `PanelTestResult` containing the statistic, p-value, reference distribution, and a readable statement of the null and alternative hypotheses.

## Choose the question, then check applicability

- Do the fitted fixed effects add to a pooled common-intercept model? Use the pooling F test.
- Does a pooled model need an entity error component? Use the Breusch-Pagan LM test.
- Is classical one-way entity RE compatible with FE on the same data? Use the Hausman test.

These are different null hypotheses, not a sequence that automatically selects a valid model. Failure to reject does not establish the assumptions, and a smaller p-value in one test does not make it a substitute for another. The classical Hausman comparison requires the same aligned sample and coefficient design, uniquely identified coefficients, and nonrobust one-way entity FE and RE fits.

## Run the comparisons on one sample

Call `PanelOLS.pooling_f_test()` for pooling F. For the LM test, fit `PooledOLS` with `entity_ids`, then call `breusch_pagan_lm_test()`. For Hausman, use `hausman_test()` on the compatible fitted FE and RE models. The sections below give each test's definition and applicability conditions.

This example uses the same 30 entities and 8 observations per entity for all three fits. FE and RE use the same single slope column without a level constant; pooled OLS adds its own intercept. This simulation illustrates reading the tests, not deciding the assumptions of a real application.

<!-- example: cpu-diagnostics -->
```python
import numpy as np
from statgpu.panel import PanelOLS, PooledOLS, RandomEffects

rng = np.random.default_rng(17)
entity_ids = np.repeat(np.arange(30), 8)
X = rng.normal(size=(240, 1))
y = (2.0 * X[:, 0] + np.repeat(rng.normal(size=30), 8)
     + rng.normal(scale=0.4, size=240))

fe = PanelOLS(entity_effects=True, cov_type="nonrobust", device="cpu").fit(
    X, y, entity_ids=entity_ids
)
pooled = PooledOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
re = RandomEffects(cov_type="nonrobust", device="cpu").fit(
    X, y, entity_ids=entity_ids
)
tests = {
    "pooling F": fe.pooling_f_test(),
    "Breusch-Pagan LM": pooled.breusch_pagan_lm_test(),
    "Hausman": fe.hausman_test(re),
}
for name, result in tests.items():
    if result.applicable:
        print(name, result.statistic, result.pvalue, result.null)
    else:
        print(name, "not applicable:", result.reason)
```
<!-- example-end: cpu-diagnostics -->

Pooling F is about 40.26 and LM about 564.32, both giving strong evidence against their respective pooled nulls in this simulation. Hausman is **not applicable** here because the estimated covariance difference is not positive semidefinite. That is not evidence for or against RE: report the reason rather than manufacturing or interpreting a p-value. The model pages explain [FE](panel-ols.md#cpu-example), [pooled OLS](pooled-ols.md#cpu-example), and [RE](random-effects.md#cpu-example) results in more detail.

## Outputs and Strict Behavior

First check `result.applicable`. If it is false, read `result.reason` rather than interpreting a missing p-value. Otherwise, read the null hypothesis and reference distribution alongside the statistic and p-value.

`PanelTestResult` reports `statistic`, `pvalue`, the reference distribution, degrees of freedom, null and alternative text, and an `applicable` flag. When a test cannot be computed under its documented definition, inspect `reason` to see why; statgpu does not return a different test under the same method name.


## Pooling F

The pooling F test asks whether the entity and/or time effects included in a `PanelOLS` model add explanatory power beyond a pooled model with a common intercept. Under the null, there are no additional effect differences relative to that common level; with the usual identifying constraints, the added fixed-effect coefficients are jointly zero. Failure to reject means the test found no evidence against the pooled specification relative to that fixed-effect alternative, not that the pooled model has been proved correct.

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_U},
$$

where $q$ is the number of independently testable fixed-effect restrictions.

```python
result = fe.pooling_f_test()
print(result.statistic, result.pvalue)
```

## Breusch-Pagan LM

For `PooledOLS` with `entity_ids`, the one-way Breusch-Pagan LM test asks whether an entity-level random component is needed:

$$
H_0:\sigma_a^2=0.
$$

A small p-value is evidence against the pooled error structure in favor of an entity error component. The implementation also supports the Baltagi-Li form used for incomplete or unbalanced panels.

```python
result = pooled.breusch_pagan_lm_test()
print(result.statistic, result.pvalue)
```

## Hausman FE versus RE

The Hausman test compares fixed-effects and random-effects coefficient estimates. Under the classical null, the random-effects estimator is consistent and efficient; systematic disagreement with fixed effects is evidence against that random-effects specification.

$$
H=(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

```python
result = fe.hausman_test(re)
# or
result = re.hausman_test(fe)
```

> **Applicability:** `hausman_test()` implements the classical one-way entity FE-versus-RE comparison. Both models must use `cov_type="nonrobust"`, must be fitted to the same aligned sample and coefficient design, and must have uniquely identified coefficient vectors. If either fitted design is rank deficient, coefficient-level inference is unavailable and Hausman returns `applicable=False` rather than constructing a test from an arbitrary generalized-inverse coefficient representation. If these requirements are not met, the result is returned with `applicable=False` and a reason instead of silently switching to a different Hausman variant.
>
> Numerically, let $D=V_{\mathrm{FE}}-V_{\mathrm{RE}}$. If $D$ has a materially negative eigenvalue, the classical quadratic form is not valid and the test is reported as inapplicable. If $D$ is singular but positive semidefinite, statgpu uses the Moore-Penrose inverse $D^+$ only when the coefficient difference lies in $\operatorname{range}(D)$.

## Numerical limits

For finite extreme-scale inputs, classical model F, pooling F, and Breusch-Pagan LM use range-aware arithmetic without changing their statistical definitions. Public RSS metadata is expressed in the original squared units and may be `inf` when that squared quantity itself exceeds float64 range. This alone does not imply an infinite test statistic: the scale-invariant statistic is protected against avoidable intermediate overflow and underflow.


## References

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271. [https://doi.org/10.2307/1913827](https://doi.org/10.2307/1913827)
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *The Review of Economic Studies*, 47(1), 239-253. [https://doi.org/10.2307/2297111](https://doi.org/10.2307/2297111)
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels. *Econometric Reviews*, 9(1), 103-107. [https://doi.org/10.1080/07474939008800180](https://doi.org/10.1080/07474939008800180)
