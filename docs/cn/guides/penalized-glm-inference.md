# 惩罚 GLM 推断

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：系数推断的统计目标与公开支持行为  
> 切换：[English](../../en/guides/penalized-glm-inference.md)

## 这个接口表达什么

`PenalizedGeneralizedLinearModel` 与各类带类型的惩罚 GLM 封装器通过 `compute_inference`、`inference_method` 和 `cov_type` 提供系数推断。

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

这些字段用于说明报告的不确定性究竟对应哪个统计参数，尤其适用于带惩罚拟合或经过 CV 选择之后的结果。

## 支持概览

| 损失函数 / 惩罚项 | 支持的推断 | `auto` 行为 |
|---|---|---|
| 平方误差 + L2/无惩罚 | Gaussian 经典/稳健协方差 | 按 `cov_type` 使用经典或 Gaussian 稳健协方差 |
| 平方误差 + L1/ElasticNet | 去偏推断；显式请求时可用 `post_selection_ols` | `debiased` |
| 光滑非 Gaussian GLM + L2/无惩罚 | 固定惩罚 M-估计 | `m_estimation` |
| 受支持的 Gaussian 惩罚模型 | 显式请求残差自助法 | 不自动选择 |
| 受支持的 SCAD/MCP 标量 GLM | 显式请求活跃集上的 oracle 重拟合 | 不自动选择 |
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

记单个观测的得分贡献为 $\psi_i$，平均 Hessian 为 $H$，L2 曲率为 $P''$，平均得分外积为 $J$，则 HC0/HC1 协方差具有形式

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1}J(H+P'')^{-1}/n.
$$

`cov_type="nonrobust"` 使用基于模型的惩罚信息矩阵协方差。

当前非 Gaussian 固定惩罚路径支持：

- `nonrobust`；
- `hc0`；
- `hc1`。

HC2、HC3 与 HAC 在该路径上不可用，请求时会报错。

## 解析权重

当所选光滑 GLM 求解器支持解析 `sample_weight` 时，整个拟合使用同一个归一化带权目标：

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

因此，把所有正的解析权重同时乘上一个常数，不会改变统计目标和推断目标；数值结果只会受到求解器容差范围内有限精度误差的影响。

这里的权重是**解析权重 / 相对重要性权重**，不是频数权重；把全部权重统一放大，并不等价于复制观测从而增大样本量。

如果某个损失函数没有定义所请求的带权拟合，真正的非均匀权重会被拒绝，而不是被静默丢弃。

## 求解器选择与权重

受支持的显式求解器请求保持权威。对于光滑非 Gaussian L2 / 无惩罚路径，Newton 与 L-BFGS 在支持解析权重时使用上面的带权目标。

`solver="auto"` 则继续遵循模型本身的正常求解器分发规则。用户的公开请求仍然是 `auto`；推断描述的是实际成功拟合出的模型，而不会为了推断单独换成无关的求解算法。

求解器兼容性见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

## 后端与设备行为

受支持的非 Gaussian M-估计会在拟合实际使用的后端和设备上完成协方差、统计量、p 值与置信区间的数值计算。

显式 `device="cuda"` 或 `device="torch"` 请求不会被静默替换成 NumPy 推断。数值推断完成后，小型结果数组可以转换为 NumPy；这个用于结果整理的边界不会改变数值程序实际运行的位置。

部分推断方法支持的后端范围比基础估计器更窄。例如，当前 SCAD/MCP 的 oracle 重拟合只支持 CPU；GPU 拟合后请求这一方法会报错，而不会把 CPU 计算结果描述成原生 GPU 推断。

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

## SCAD/MCP oracle 推断

`inference_method="oracle"` 必须显式请求，因为它以惩罚拟合已经选出的活跃集为条件。`auto` 不会静默采用这种解释。

在支持范围内，该程序会在所选活跃集上进行不含原非凸惩罚的重拟合，并对这一条件重拟合报告不确定性。请求前应检查相应模型和后端的支持范围。

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

```python
from statgpu.linear_model import PenalizedPoissonRegression

model = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    device="cpu",
)
model.fit(X, y, sample_weight=w)

print(model.inference_requested_method_)  # auto
print(model.inference_resolved_method_)   # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
print(model.summary())
```

对于稀疏的非 Gaussian L1/ElasticNet，目前没有提供系数推断；应使用 `compute_inference=False`，而不要假定 Gaussian 模型的去偏或自助法程序会自动适用于其他分布族。

## 相关文档

- [推断模式](inference-modes.md) — 推断方法的选择与解释
- [交叉验证](cross-validation.md) — 调参与最终重拟合
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) — 求解器兼容性
- [设备与 GPU 内存](device-and-memory.md) — 设备语义

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
