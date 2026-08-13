# 面板 Diagnostics

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/diagnostics.md)

所有 specification test 都返回 `PanelTestResult`。

## Pooling F

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_U},
$$

其中 $q$ 为有效 restriction rank；原假设为所有包含的 fixed effects 联合为 0。

```python
fe.pooling_f_test()
```

## Breusch-Pagan LM

对提供 `entity_ids` 的 `PooledOLS`，one-way error-components LM 检验

$$
H_0:\sigma_a^2=0.
$$

维护的定义包含 incomplete/unbalanced panel 的 Baltagi-Li 形式。

```python
pooled.breusch_pagan_lm_test()
```

## Hausman FE vs RE

$$
H=
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

classical comparison 要求 one-way entity FE、FE/RE 都使用 nonrobust covariance，并且 aligned sample/design 匹配。materially indefinite 的 covariance difference 会返回 inapplicable。

```python
fe.hausman_test(re)
# 或
re.hausman_test(fe)
```
