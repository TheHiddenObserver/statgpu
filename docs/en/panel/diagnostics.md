# Panel Diagnostics

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/diagnostics.md)

## Overview and Path

Panel specification tests return `PanelTestResult` and are implemented in `statgpu/panel/_diagnostics.py` plus the shared diagnostic context helpers.

## Pooling F

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_U},
$$

where $q$ is the effective restriction rank. It tests whether the included fixed effects are jointly zero.

```python
fe.pooling_f_test()
```

## Breusch-Pagan LM

For `PooledOLS` with `entity_ids`, the one-way panel error-components LM tests

$$
H_0:\sigma_a^2=0.
$$

The maintained definition includes the incomplete/unbalanced Baltagi-Li form.

```python
pooled.breusch_pagan_lm_test()
```

## Hausman FE versus RE

$$
H=(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

```python
fe.hausman_test(re)
# or
re.hausman_test(fe)
```

> **Note:** The maintained `hausman_test()` implements the classical FE-versus-RE comparison. It therefore requires one-way entity fixed effects, `cov_type="nonrobust"` for both FE and RE, and matched aligned estimation samples/designs. Let $D=V_{\mathrm{FE}}-V_{\mathrm{RE}}$. If $D$ has a materially negative eigenvalue, the classical Hausman quadratic form is reported as `inapplicable`. If $D$ is singular positive semidefinite, the Moore-Penrose generalized inverse $D^+$ is used only when $\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}\in\operatorname{range}(D)$.

## Outputs and Strict Behavior

`PanelTestResult` reports the statistic, p-value, reference distribution, degrees of freedom, null/alternative text, applicability flag, reason, and metadata. Unsupported Hausman configurations return inapplicable rather than silently changing the test definition.

## External Validation

Specification and covariance-adjacent behavior is checked against maintained Python/R references where definitions overlap. The pinned external workflow uses `linearmodels==7.0`, `statsmodels==0.14.6`, R `plm==2.6-7`, and `sandwich==3.1-3`; see `dev/tests/test_panel_stage_c_r_external.py` and the maintained panel diagnostic tests.

## References

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271. [https://doi.org/10.2307/1913827](https://doi.org/10.2307/1913827)
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *The Review of Economic Studies*, 47(1), 239-253. [https://doi.org/10.2307/2297111](https://doi.org/10.2307/2297111)
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels. *Econometric Reviews*, 9(1), 103-107. [https://doi.org/10.1080/07474939008800180](https://doi.org/10.1080/07474939008800180)
