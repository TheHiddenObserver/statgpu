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

The classical comparison requires one-way entity FE, nonrobust FE and RE covariance, and matched aligned samples/designs. A materially indefinite covariance difference is reported as inapplicable. A singular positive-semidefinite difference uses the documented identified-range generalized-inverse extension only when the coefficient difference lies in that range.

```python
fe.hausman_test(re)
# or
re.hausman_test(fe)
```

## Outputs and Strict Behavior

`PanelTestResult` reports the statistic, p-value, reference distribution, degrees of freedom, null/alternative text, applicability flag, reason, and metadata. Unsupported Hausman configurations return inapplicable rather than silently changing the test definition.

## External Validation

Specification and covariance-adjacent behavior is checked against maintained Python/R references where definitions overlap. The pinned external workflow uses `linearmodels==7.0`, `statsmodels==0.14.6`, R `plm==2.6-7`, and `sandwich==3.1-3`; see `dev/tests/test_panel_stage_c_r_external.py` and the maintained panel diagnostic tests.

## References

Hausman (1978), *Specification Tests in Econometrics*; Breusch and Pagan (1980), *The Lagrange Multiplier Test and Its Applications to Model Specification*; Baltagi and Li (1990), incomplete-panel error-components LM test.
