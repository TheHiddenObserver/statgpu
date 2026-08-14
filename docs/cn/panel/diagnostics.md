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

```python
fe.hausman_test(re)
# 或
re.hausman_test(fe)
```

> **注意：** 当前 `hausman_test()` 实现 classical FE-versus-RE comparison，因此要求 `PanelOLS` 为 one-way entity fixed effects，且 FE 与 RE 均使用 `cov_type="nonrobust"`，同时两者必须基于匹配并对齐的 estimation sample/design。令 $D=V_{\mathrm{FE}}-V_{\mathrm{RE}}$。若 $D$ 存在 materially negative eigenvalue，则 classical Hausman quadratic form 返回 `inapplicable`。若 $D$ 为 singular positive semidefinite，则仅当 $\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}\in\operatorname{range}(D)$ 时使用 Moore-Penrose generalized inverse $D^+$。

## Outputs and Strict Behavior

`PanelTestResult` 包含 statistic、p-value、reference distribution、degrees of freedom、null/alternative、applicability、reason 与 metadata。不支持的 Hausman configuration 返回 inapplicable，不会静默改换 test definition。

## External Validation

在定义重合处，specification 与 covariance-adjacent behavior 会与维护的 Python/R references 比较。pinned external workflow 使用 `linearmodels==7.0`、`statsmodels==0.14.6`、R `plm==2.6-7` 与 `sandwich==3.1-3`；见 `dev/tests/test_panel_stage_c_r_external.py` 与相关 panel diagnostic tests。

## References

Hausman (1978), *Specification Tests in Econometrics*；Breusch and Pagan (1980), *The Lagrange Multiplier Test and Its Applications to Model Specification*；Baltagi and Li (1990), incomplete-panel error-components LM test。
