# Panel Diagnostics

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/panel/diagnostics.md)

## Overview and Path

Panel diagnostics help answer model-selection questions such as whether fixed effects are needed or whether a random-effects specification is compatible with the fixed-effects estimate. Each test returns a `PanelTestResult` containing the statistic, p-value, reference distribution, and a readable statement of the null and alternative hypotheses.

Implementation: `statgpu/panel/_diagnostics.py` plus the shared diagnostic-context helpers.

## Pooling F

The pooling F test asks whether the fixed effects included in a `PanelOLS` model are jointly unnecessary. Its null hypothesis is that all included fixed effects are zero, so failure to reject supports the pooled specification relative to that fixed-effect alternative.

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

> **Applicability:** `hausman_test()` implements the classical one-way entity FE-versus-RE comparison. Both models must use `cov_type="nonrobust"` and must be fitted to the same aligned sample and coefficient design. If these requirements are not met, the result is returned with `applicable=False` and a reason instead of silently switching to a different Hausman variant.
>
> Numerically, let $D=V_{\mathrm{FE}}-V_{\mathrm{RE}}$. If $D$ has a materially negative eigenvalue, the classical quadratic form is not valid and the test is reported as inapplicable. If $D$ is singular but positive semidefinite, statgpu uses the Moore-Penrose inverse $D^+$ only when the coefficient difference lies in $\operatorname{range}(D)$.

## Outputs and Strict Behavior

`PanelTestResult` reports `statistic`, `pvalue`, the reference distribution, degrees of freedom, null and alternative text, and an `applicable` flag. When a test cannot be computed under its documented definition, inspect `reason` to see why; statgpu does not return a different test under the same method name.

## External Validation

Where definitions overlap, the diagnostic and supporting covariance calculations are compared with pinned Python and R references: `linearmodels==7.0`, `statsmodels==0.14.6`, R `plm==2.6-7`, and `sandwich==3.1-3`. The corresponding checks live in the panel diagnostic tests and `dev/tests/test_panel_stage_c_r_external.py`.

## References

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271. [https://doi.org/10.2307/1913827](https://doi.org/10.2307/1913827)
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *The Review of Economic Studies*, 47(1), 239-253. [https://doi.org/10.2307/2297111](https://doi.org/10.2307/2297111)
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels. *Econometric Reviews*, 9(1), 103-107. [https://doi.org/10.1080/07474939008800180](https://doi.org/10.1080/07474939008800180)
