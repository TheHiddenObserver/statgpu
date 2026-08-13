# Panel Diagnostics

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/diagnostics.md)

All specification tests return `PanelTestResult`.

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
H=
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

The classical comparison requires one-way entity FE, nonrobust FE and RE covariance, and matched aligned samples/designs. A materially indefinite covariance difference is reported as inapplicable.

```python
fe.hausman_test(re)
# or
re.hausman_test(fe)
```
