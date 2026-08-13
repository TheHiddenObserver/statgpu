# PanelOLS

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/panel-ols.md)

## 模型

对 entity 与 time effects，

$$
y_{it}=x_{it}^{\top}\beta+\alpha_i+\gamma_t+\varepsilon_{it}.
$$

令 $F$ 为 fixed-effect design，并定义

$$
M_F=I-F(F^\top F)^+F^\top.
$$

则

$$
\widehat\beta_{\mathrm{FE}}
=(X^\top M_FX)^+X^\top M_Fy.
$$

单向 entity effect 即 within demeaning。双向 effect 使用 backend-native alternating projection；若在 `demean_max_iter` 内没有达到 `demean_tol`，则 fail closed。

## Covariance

fit space 为

$$
Z=M_FX,
\qquad
e=M_F(y-X\widehat\beta_{\mathrm{FE}}).
$$

统一公式见 [面板 covariance](covariance.md)。Driscoll-Kraay 的 fixed-effect nuisance rank 为

$$
r_F=
\begin{cases}
N, & \text{仅 entity},\\
T, & \text{仅 time},\\
N+T-C, & \text{双向},
\end{cases}
$$

其中 $C$ 为观测 entity-time incidence graph 的 connected-component 数。

## API

```python
from statgpu.panel import PanelOLS

model.fit(X, y, entity_ids=None, time_ids=None, cluster=None)
```

主要选项为 `entity_effects`、`time_effects`、`cov_type`、`bandwidth`、`kernel`、`group_debias`、`demean_max_iter`、`demean_tol`、`alpha` 和 `device`。

Formula 可使用 additive `EntityEffects` / `TimeEffects` token 或 pipe fixed-effect syntax，但不能混用。双向 stored-effect prediction 要求 entity/time label 可识别并属于同一个 fitted incidence component。

Pooling F 与 Hausman 见 [面板 diagnostics](diagnostics.md)。
