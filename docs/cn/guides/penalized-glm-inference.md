# 惩罚 GLM 推断

> 语言：中文  
> 最后更新：2026-10-05  
> 页面定位：系数推断的统计目标与公开支持行为  
> 切换：[English](../../en/guides/penalized-glm-inference.md)

## 这个接口表达什么

`PenalizedGeneralizedLinearModel` 与各类具体的惩罚 GLM 估计器通过 `compute_inference`、`inference_method` 和 `cov_type` 提供系数推断。

对于通用接口，通常可以先从

```python
inference_method="auto"
```

开始。

成功完成推断后，模型会区分调用者请求的方法与最终实际解析得到的方法。相关已拟合属性包括：

- `inference_requested_method_`；
- `inference_resolved_method_`；
- `inference_method_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`。

这些字段在有值时用于说明报告的不确定性究竟对应哪个统计参数，尤其适用于带惩罚拟合或经过 CV 选择之后的结果。

当前 `post_selection_ols` 会把上述公开推断信息字段保留为 `None`；实际方法和所选活跃集的重拟合信息应查看 `model._inference_result.method` 与 `model._inference_result.metadata`，`params` 保存重拟合估计值。公开字段为空并不表示未执行推断。可运行示例见[选择后 OLS](inference-modes.md#post_selection_ols)。

## 支持概览

| 损失函数 / 惩罚项 | 支持的推断 | `auto` 行为 |
|---|---|---|
| 平方误差 + L2/无惩罚 | Gaussian 经典/稳健协方差 | 按 `cov_type` 使用经典或 Gaussian 稳健协方差 |
| 平方误差 + L1/ElasticNet | 去偏推断；显式请求时可用 `post_selection_ols` | `debiased` |
| 光滑非 Gaussian GLM + L2/无惩罚 | 固定惩罚 M-估计 | `m_estimation` |
| 平方误差 + L1/ElasticNet/SCAD/MCP | 显式请求残差自助法；不适用于 L2 | 不自动选择 |
| 平方误差 + SCAD/MCP | CPU 父模型可显式请求活跃集 oracle 重拟合；注意下文子模型设备限制 | 不自动选择 |
| 非 Gaussian SCAD/MCP | `oracle` 接受标量 GLM 分布族，但当前重拟合可能改变模型；详见下文限制 | 不自动选择 |
| 非 Gaussian L1/ElasticNet | 未实现系数推断 | 请求会报错 |
| 分组惩罚 | 仅估计 | 请求推断会报错 |
| `PenalizedCoxPHModel` / `PenalizedGLM_CV` 的 Cox 分支 | 仅估计 | 请求推断会报错 |

稀疏 Gaussian 模型中的 `debiased`、`post_selection_ols` 等方法如何选择与解释，见 [推断模式](inference-modes.md)。

## 固定惩罚 M-估计

对于受支持的光滑非 Gaussian L2 / 无惩罚拟合，statgpu 把已拟合系数看作给定惩罚强度下估计方程的解。

当 L2 惩罚强度为正时，会报告：

```text
inference_target_ = "penalized_estimating_equation"
penalty_conditioning_ = "fixed_penalty"
```

无惩罚拟合（`alpha=0` 或对应的无惩罚配置）的推断目标是普通的无惩罚总体参数。

记单个观测的得分贡献为 $\psi_i$，平均 Hessian 为 $H$，L2 曲率为 $P''$，平均得分外积为 $J$，则 HC0 协方差为

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1}J(H+P'')^{-1}/n.
$$

HC1 在 $n>k$ 时还要乘以 $n/(n-k)$，其中 $n$ 为观测行数，$k$ 为拟合参数个数（有截距时包括截距）。这是有限样本乘数，不校正惩罚选择的不确定性。当前实现会在 $n\le k$ 时省略该乘数，不能将这种结果解释为有效的 HC1 自由度修正。使用分析权重时，该乘数仍按行数计算，而不是按权重总和计算。

`cov_type="nonrobust"` 使用基于模型的惩罚信息矩阵协方差。

当前非 Gaussian 固定惩罚路径支持：

- `nonrobust`；
- `hc0`；
- `hc1`。

HC2、HC3 与 HAC 在该路径上不可用，请求时会报错。

## 分析权重

当所选光滑 GLM 求解器支持分析权重 `sample_weight` 时，整个拟合使用同一个归一化带权目标：

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

对应的 M-估计采用相同的相对权重解释。计算协方差时，可以把权重等价地归一化到均值为 1 的尺度：

$$
\widetilde w_i
=
\frac{n w_i}{\sum_j w_j},
\qquad
\sum_i \widetilde w_i=n.
$$

因此，把所有正的分析权重同时乘上一个常数，不会改变统计目标和推断目标；数值结果只会受到求解器容差范围内有限精度误差的影响。

这里的权重是**分析权重 / 相对重要性权重**，不是频数权重；把全部权重统一放大，并不等价于复制观测从而增大样本量。

如果某个损失函数没有定义所请求的带权拟合，真正的非均匀权重会被拒绝，而不是被静默丢弃。

## 求解器选择与权重

受支持的显式求解器请求会被执行。对于光滑非 Gaussian L2 / 无惩罚路径，Newton 与 L-BFGS 在支持分析权重时使用上面的带权目标。

`solver="auto"` 则继续遵循模型本身的正常求解器分发规则。用户的公开请求仍然是 `auto`；推断描述的是实际成功拟合出的模型，而不会为了推断单独换成无关的求解算法。

求解器兼容性见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

## 后端与设备行为

受支持的非 Gaussian M-估计会在拟合实际使用的后端和设备上完成协方差、统计量、p 值与置信区间的数值计算。

显式 `device="cuda"` 或 `device="torch"` 请求不会被静默替换成 NumPy 推断。数值推断完成后，小型结果数组可以转换为 NumPy；这个用于结果整理的边界不会改变数值程序实际运行的位置。

部分推断方法支持的后端范围比基础估计器更窄。例如，SCAD/MCP 的 oracle 接口会拒绝已经执行 GPU 拟合的父模型。但重拟合子模型目前默认使用 `device="auto"`，因此在有 GPU 的系统上，CPU 父模型并不能保证子模型也在 CPU 上运行。详见下文的 oracle 限制；显式创建单独的重拟合模型，才能同时指定设备与统计模型。

## 残差自助法的范围

`inference_method="bootstrap"` 指的是 Gaussian 惩罚模型的残差自助法，而不是通用 GLM 自助法。

每次重抽样中，statgpu 会：

1. 根据已拟合 Gaussian 模型计算拟合值与残差；
2. 对残差进行有放回抽样；
3. 构造新的自助法响应变量；
4. 使用相同的调参配置重新拟合同一个惩罚模型；
5. 汇总自助法样本形成的系数分布。

需要可复现抽样时设置 `bootstrap_random_state`。`n_bootstrap` 控制重拟合次数，并且至少为 2。

当前残差自助法路径要求：

- `sample_weight=None`；
- `cov_type="nonrobust"`。

带权残差自助法、稳健/HC 自助法、HAC/分块自助法、非 Gaussian 自助法与 Cox 自助法都不由这一接口提供。

这些区间描述的是固定设计、固定调参配置下的残差自助法过程；它们不是一般意义上的选择后推断区间，也不会自动校正调参或变量选择带来的额外不确定性。

报告的区间是重拟合系数的第 2.5 和第 97.5 百分位数，`bse` 是这些系数的样本标准差（`ddof=1`）。`pvalues` 的计算方式是：分别统计系数不大于零、不小于零的比例，取较小者的两倍，并将上限截为一。这个数可以恰好为零，不采用加一修正。显示的 `z` 是原拟合系数除以自助法标准误，但 p 值并非由这个统计量的正态尾概率计算。这些符号比例不来自强制原假设成立的重采样检验，对有偏惩罚估计量也没有一般的有限样本校准保证。增加抽样次数能降低模拟误差，但不能消除收缩偏差或证明假设检验有效。

## SCAD/MCP oracle 推断

`inference_method="oracle"` 必须显式请求，因为它以惩罚拟合已经选出的活跃集为条件。`auto` 不会静默采用这种解释。

这一程序的预期目标是在所选活跃集上去掉原非凸惩罚后重新拟合。即使无惩罚重拟合本身正确，普通区间也不会自动校正在同一数据上选择变量的影响。未选中系数的不确定性不可用，以 `NaN` 表示；如果没有任何特征被选中，所有参数（包括截距）的不确定性均不可用。

<a id="current-non-gaussian-oracle-limitation"></a>

### 当前非 Gaussian oracle 的限制

接口接受 SCAD/MCP 与 `squared_error`、`logistic`、`poisson`、`gamma`、`inverse_gaussian`、`negative_binomial`、`tweedie` 的组合。但当前非 Gaussian 重拟合可能静默使用构造函数默认值，未保留原模型配置：logistic 使用 `C=1`，并非无惩罚拟合；负二项离散参数可能重置为 `alpha=1`，Gamma 链接重置为 `"log"`，Tweedie 幂参数重置为 `1.5`。子模型的求解器与设备也默认是 `"auto"`；例如 Poisson 会采用默认带 Ridge 惩罚的 IRLS 拟合，而非预期的无惩罚 Newton 重拟合。因此，返回标记为 `oracle` 的成功结果，并不能证明原推断目标得到保留。不要使用这些非 Gaussian oracle 系数表进行推断。

如果需要诊断性重拟合，应先以 `compute_inference=False` 拟合选择模型，再明确指定所选列、分布族、链接、离散或幂参数、无惩罚配置及设备，单独拟合一个模型。对 logistic 回归，独立包装类的 `C=0` 表示不使用正则化：

<!-- inference-example: explicit-logistic-diagnostic -->
```python
import numpy as np
from statgpu.linear_model import PenalizedLogisticRegression, LogisticRegression

rng = np.random.default_rng(48)
X = rng.normal(size=(200, 2))
probability = 1 / (1 + np.exp(-(0.6 + X @ np.array([2.0, -1.0]))))
y = rng.binomial(1, probability)
selection = PenalizedLogisticRegression(
    penalty="scad", alpha=0.01, solver="fista", device="cpu",
    compute_inference=False, max_iter=10000, tol=1e-10,
).fit(X, y)
active = np.flatnonzero(np.abs(selection.coef_) > 1e-10)
if active.size == 0:
    raise ValueError("No feature was selected; specify a model with only an intercept")
refit = LogisticRegression(
    C=0, device="cpu", compute_inference=True, max_iter=10000, tol=1e-10,
).fit(X[:, active], y)
print(active.tolist())
print(np.round(np.r_[refit.intercept_, refit.coef_], 6))
```

示例选中 `[0, 1]`，系数输出为 `[0.626286, 2.295259, -1.037398]`。`selection` 仍按原惩罚模型预测，`refit` 是独立的诊断模型。它的普通区间仍未考虑数据驱动的变量选择。若要进行验证性推断，应使用相互独立的数据分别选择变量和检验，或在适用假设下采用明确处理选择不确定性的方法；不能把手工重拟合当作选择后推断的修复。

## 交叉验证

`PenalizedGLM_CV` 把调参和系数推断分成两个阶段：

```text
在各数据折和候选参数上拟合
    -> 选择 alpha
    -> 在全部观测上重拟合所选模型
    -> 只对最终重拟合执行一次推断
```

成功完成推断的 CV 拟合会报告类似：

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

因此，标准误、p 值和置信区间都是**以 CV 已经选择的惩罚强度为条件**的，并不会自动调整调参选择带来的额外不确定性。

对于残差自助法，只有 CV 选定调参值后才开始重抽样；候选项选择过程本身不会进行自助法重抽样。

Cox 分支仍然只提供估计，不提供这一系数推断接口。

一般的选择与最终重拟合约定见 [交叉验证](cross-validation.md)。

## 示例

以下完整 CPU 示例生成有限设计矩阵、非负整数计数和有限正分析权重。

<!-- inference-example: weighted-poisson-m-estimation -->
```python
import numpy as np
from statgpu.linear_model import PenalizedPoissonRegression

rng = np.random.default_rng(72)
X = rng.normal(size=(100, 2))
y = rng.poisson(np.exp(0.2 + X @ np.array([0.3, -0.2])))
w = np.linspace(0.5, 1.5, len(y))
model = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    solver="lbfgs",
    device="cpu",
)
model.fit(X, y, sample_weight=w)

print(model.inference_requested_method_)  # auto
print(model.inference_resolved_method_)   # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
inference = model._inference_result.to_dict()
print(inference["params"])
print(inference["conf_int"])
```

输出方法为 `m_estimation`；由于 L2 惩罚强度为正，推断目标为 `penalized_estimating_equation`。HC0 不会消除拟合的收缩偏差，也不会提供经过选择校正的区间。

通用惩罚 GLM 和具体的非 Gaussian 包装类目前没有 `summary()`。可以读取结果的 `params`、`bse`、`pvalues`、`conf_int`，或如上调用 `to_dict()`；`to_dataframe()` 还需要安装 pandas。

对于稀疏的非 Gaussian L1/ElasticNet，目前没有提供系数推断；应使用 `compute_inference=False`，而不要假定 Gaussian 模型的去偏或自助法程序会自动适用于其他分布族。

## 相关文档

- [推断模式](inference-modes.md) — 推断方法的选择与解释
- [交叉验证](cross-validation.md) — 调参与最终重拟合
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) — 求解器兼容性
- [设备与 GPU 内存](device-and-memory.md) — 设备语义

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
