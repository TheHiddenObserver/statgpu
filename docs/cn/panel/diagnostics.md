# 面板 Diagnostics

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/diagnostics.md)

## Overview and Path

panel specification tests 返回 `PanelTestResult`，实现位于 `statgpu/panel/_diagnostics.py` 及共享 diagnostic-context helpers。

## Pooling F

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_U},
$$

其中 $q$ 为 effective restriction rank；原假设为包含的 fixed effects 联合为 0。

```python
fe.pooling_f_test()
```

## Breusch-Pagan LM

对提供 `entity_ids` 的 `PooledOLS`，one-way panel error-components LM 检验

$$
H_0:\sigma_a^2=0.
$$

维护的定义包含 incomplete/unbalanced panel 的 Baltagi-Li 形式。

```python
pooled.breusch_pagan_lm_test()
```

## Hausman FE versus RE

$$
H=(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

classical comparison 要求 one-way entity FE、FE/RE 都使用 nonrobust covariance，并且 aligned sample/design 匹配。materially indefinite covariance difference 返回 inapplicable。若 difference 为 singular PSD，仅在 coefficient difference 位于 identified range 时使用 documented generalized-inverse extension。

```python
fe.hausman_test(re)
# 或
re.hausman_test(fe)
```

## Outputs and Strict Behavior

`PanelTestResult` 包含 statistic、p-value、reference distribution、degrees of freedom、null/alternative、applicability、reason 与 metadata。不支持的 Hausman configuration 返回 inapplicable，不会静默改换 test definition。

## External Validation

在定义重合处，specification 与 covariance-adjacent behavior 会与维护的 Python/R references 比较。pinned external workflow 使用 `linearmodels==7.0`、`statsmodels==0.14.6`、R `plm==2.6-7` 与 `sandwich==3.1-3`；见 `dev/tests/test_panel_stage_c_r_external.py` 与相关 panel diagnostic tests。

## References

Hausman (1978), *Specification Tests in Econometrics*；Breusch and Pagan (1980), *The Lagrange Multiplier Test and Its Applications to Model Specification*；Baltagi and Li (1990), incomplete-panel error-components LM test。
