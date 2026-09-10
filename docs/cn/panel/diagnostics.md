# 面板 Diagnostics

> 语言：中文  
> 最后更新：2026-08-18<br>
> 切换：[English](../../en/panel/diagnostics.md)

## 概述与路径

panel diagnostics 用来回答“是否需要 fixed effects”“pooled error structure 是否足够”“random-effects specification 是否与 fixed-effects estimate 相容”等模型选择问题。每个检验都返回 `PanelTestResult`，其中包含 statistic、p-value、reference distribution，以及可直接阅读的 null/alternative hypothesis。

实现：`statgpu/panel/_diagnostics.py` 及共享 diagnostic-context helpers。

## Pooling F 检验

Pooling F test 检验 `PanelOLS` 中加入的 fixed effects 是否可以整体去掉。原假设是所有 included fixed effects 联合为 0；如果不能拒绝原假设，则 pooled specification 相对于该 fixed-effect alternative 更有支持。

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_U},
$$

其中 $q$ 是可以独立检验的 fixed-effect restrictions 数。

```python
result = fe.pooling_f_test()
print(result.statistic, result.pvalue)
```

## Breusch–Pagan LM 检验

对提供 `entity_ids` 的 `PooledOLS`，one-way Breusch-Pagan LM test 检验是否需要 entity-level random component：

$$
H_0:\sigma_a^2=0.
$$

较小的 p-value 表示 pooled error structure 可能不足，更支持存在 entity error component。实现还支持 incomplete/unbalanced panel 使用的 Baltagi-Li 形式。

```python
result = pooled.breusch_pagan_lm_test()
print(result.statistic, result.pvalue)
```

## Hausman FE 与 RE 检验

Hausman test 比较 fixed-effects 与 random-effects coefficient estimates。classical null 下 random-effects estimator 应当 consistent 且 efficient；若 RE 与 FE 的 coefficient 存在系统性差异，则说明该 random-effects specification 可能不合适。

$$
H=(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

```python
result = fe.hausman_test(re)
# 或
result = re.hausman_test(fe)
```

> **适用条件：** `hausman_test()` 实现的是 classical one-way entity FE-versus-RE comparison。FE 与 RE 都必须使用 `cov_type="nonrobust"`，必须基于同一组对齐后的 observation 与 coefficient design，而且 coefficient vector 必须唯一可识别。如果任一 fitted design rank deficient，coefficient-level inference 本身不可用，Hausman 会返回 `applicable=False`，而不会基于任意 generalized-inverse coefficient representation 构造检验。如果不满足这些条件，结果会返回 `applicable=False` 并说明原因，而不会在同一个方法名下悄悄换成另一种 Hausman test。
>
> 数值上令 $D=V_{\mathrm{FE}}-V_{\mathrm{RE}}$。如果 $D$ 存在明显的 negative eigenvalue，classical quadratic form 不再适用，因此 test 会标记为 inapplicable。如果 $D$ 是 singular positive semidefinite，则只有当 coefficient difference 位于 $\operatorname{range}(D)$ 时，statgpu 才使用 Moore-Penrose inverse $D^+$。

## 输出与 strict 行为

`PanelTestResult` 提供 `statistic`、`pvalue`、reference distribution、degrees of freedom、null/alternative text 与 `applicable` flag。若某个检验在文档定义下无法计算，可以查看 `reason` 了解具体原因；statgpu 不会用同一个 method name 返回另一种 test。

对于 finite extreme-scale inputs，classical model F、pooling F 与 Breusch-Pagan LM 会在当前 backend 上用归一化 working values 计算其 scale-invariant quadratic reductions。scalar/column centering 只在 reduction 可能 overflow 时按 reduction length 做缩放；subnormal normalization 也不会直接除以 subnormal denominator。公开的 RSS metadata 会在原始平方尺度可表示时恢复到该尺度；只有真实平方量超出 float64 表示范围时才允许为 `inf`。检验 statistic 不应仅因为可避免的中间 overflow/underflow 而错误变成 `0`、`NaN` 或 `inf`。

## 外部验证

在定义可直接对齐的部分，diagnostics 及其依赖的 covariance calculation 会与 pinned Python/R implementations 比较：`linearmodels==7.0`、`statsmodels==0.14.6`、R `plm==2.6-7` 与 `sandwich==3.1-3`。对应 checks 位于 panel diagnostic tests 与 `dev/tests/test_panel_stage_c_r_external.py`。

## 参考（References）

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271. [https://doi.org/10.2307/1913827](https://doi.org/10.2307/1913827)
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *The Review of Economic Studies*, 47(1), 239-253. [https://doi.org/10.2307/2297111](https://doi.org/10.2307/2297111)
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels. *Econometric Reviews*, 9(1), 103-107. [https://doi.org/10.1080/07474939008800180](https://doi.org/10.1080/07474939008800180)
