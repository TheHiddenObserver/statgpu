# 惩罚 GLM 推断

> 语言：中文  
> 最后更新：2026-10-18<br>
> 页面定位：系数推断的统计目标与公开支持行为  
> 切换：[English](../../en/guides/penalized-glm-inference.md)

## 这个接口表达什么

`PenalizedGeneralizedLinearModel` 与各类具体的惩罚 GLM 估计器通过 `compute_inference`、`inference_method` 和 `cov_type` 提供系数推断。

先根据损失函数和惩罚项确认下表中的支持范围。对于支持自动选择推断方法的组合，可以从

```python
inference_method="auto"
```

开始。

## 支持概览

| 损失函数 / 惩罚项 | 支持的推断 | `auto` 行为 |
|---|---|---|
| 平方误差 + L2/无惩罚 | Gaussian 经典/稳健协方差 | 按 `cov_type` 使用经典或 Gaussian 稳健协方差 |
| 平方误差 + L1/ElasticNet | 去偏推断；显式请求时可用 `post_selection_ols` | `debiased` |
| 光滑非 Gaussian GLM + L2/无惩罚 | 固定惩罚 M-估计 | `m_estimation` |
| 平方误差 + L1/ElasticNet/SCAD/MCP | 显式请求残差自助法；不适用于 L2 | 不自动选择 |
| 平方误差 + SCAD/MCP | 显式请求时，在 CPU 上进行无惩罚活跃集 oracle 重拟合 | 不自动选择 |
| 非 Gaussian SCAD/MCP | 在 CPU 上进行无惩罚活跃集 oracle 重拟合，并保留已拟合的分布族设置 | 不自动选择 |
| 非 Gaussian L1/ElasticNet | 未实现系数推断 | 请求会报错 |
| 分组惩罚 | 仅估计 | 请求推断会报错 |
| `PenalizedCoxPHModel` / `PenalizedGLM_CV` 的 Cox 分支 | 仅估计 | 请求推断会报错 |

稀疏 Gaussian 模型中的 `debiased`、`post_selection_ols` 等方法如何选择与解释，见 [推断模式](inference-modes.md)。

## 固定惩罚 M-估计

这里先把惩罚强度 `alpha` 固定下来，再估计在相同建模条件下重复取样时，带惩罚系数估计会有多大波动。`alpha>0` 时，惩罚也改变了所估计的目标，因此这里的区间不能直接解释为无惩罚总体参数的区间。HC0/HC1 改变协方差的估计方式，不会消除收缩偏差，也不会计入选择 `alpha` 的不确定性。

对于受支持的光滑非 Gaussian L2 / 无惩罚拟合，statgpu 把已拟合系数看作给定惩罚强度下估计方程的解。

当 L2 惩罚强度为正时，会报告：

```text
inference_target_ = "penalized_estimating_equation"
penalty_conditioning_ = "fixed_penalty"
```

无惩罚拟合（`alpha=0` 或对应的无惩罚配置）的推断目标是普通的无惩罚总体参数。

记单个观测的得分贡献为 $\psi_i$，平均观测损失 Hessian 为 $H$，L2 曲率为 $P''$，平均得分外积为 $J$，则 HC0 协方差为

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1}J(H+P'')^{-1}/n.
$$

HC1 在 $n>k$ 时还要乘以 $n/(n-k)$，其中 $n$ 为观测行数，$k$ 为拟合参数个数（有截距时包括截距）。这是有限样本乘数，不校正惩罚选择的不确定性。当前实现会在 $n\le k$ 时省略该乘数，不能将这种结果解释为有效的 HC1 自由度修正。使用分析权重时，该乘数仍按行数计算，而不是按权重总和计算。

未发布源码对逆高斯和负二项 HC0/HC1 的 bread 使用观测损失 Hessian，而不是期望 Fisher scoring 曲率；优化器 Hessian 与 `nonrobust` 协方差保持不变。分析权重在线性 bread 中为一次项，在得分外积 meat 中为平方项，截距不增加惩罚曲率。受支持的 CV 最终重拟合也采用此约定，并以所选调参配置为条件。已发布的 0.3.0 在这些普通/光滑 M-estimation 路径上使用 Fisher bread，因此稳健标准误可能不同；独立的 oracle 路径原本已使用观测 bread。导数解释仅适用于均值未裁剪的范围，本次修复不会使裁剪变成可微操作。

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

部分推断方法支持的后端范围比基础估计器更窄。SCAD/MCP oracle 推断要求父模型在 CPU 上拟合，并显式以 `device="cpu"` 进行活跃集重拟合，即使系统有 GPU 也不会自动切换。推断计算使用 NumPy float64。已经执行 GPU 拟合的模型会被拒绝，不会静默转到 CPU；请求 oracle 推断时请设置 `device="cpu"`。

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

这一程序在所选活跃集上进行无正则化重拟合。报告中的活跃系数和截距来自重拟合，预测仍使用原惩罚系数。未选中系数保留其惩罚拟合点估计，不确定性不可用，以 `NaN` 表示。没有斜率被选中但模型包含截距时，oracle 会重新拟合仅含截距的模型；只有无惩罚目标存在有限解时，才报告截距的不确定性。既没有入选斜率也没有截距时，无系数可供重拟合；逆链接 Gamma 会拒绝这种空模型，因为线性预测值为零时无法定义有限均值。

当 `cov_type="nonrobust"` 时，高斯、Gamma、逆高斯和 Tweedie 的 oracle 重拟合需要估计未知离散尺度。因此，正样本权重的观测数必须大于入选斜率数加上有效截距参数数（若含截距则加一）。无权重时每个观测均计数；增大权重或补充零权重行都不会增加观测数。不满足条件时抛出 `ValueError`，不会报告零标准误或退化为单点的区间。如果只需惩罚模型的点估计，可设置 `compute_inference=False`。这一要求不适用于离散尺度固定的 Poisson/负二项 `nonrobust` 协方差，也不适用于 HC0；单独说明的 HC1 行为保持不变。观测数足够本身并不能保证重拟合可识别或已收敛。这是允许计算推断的检查，不改变原有离散尺度分母：通过检查后，Pearson 尺度仍将加权 Pearson 残差平方和除以 `n - k`，其中 `n` 为重拟合数据的行数（包括零权重行），`k` 为重拟合系数数。因此，补入零权重行仍可能影响该分母。Gamma/Tweedie 的 Pearson 离散尺度使用实际分布族方差，不施加固定方差下限，因此可表示的小响应尺度不会人为缩窄不确定性。所需数值量无法表示时仍会报错，不返回区间。

这些区间使用正态参考分布，并以已选变量集合为条件，不会校正在同一数据上选择变量或调参带来的不确定性。

<a id="current-non-gaussian-oracle-limitation"></a>

### 无惩罚重拟合与分布族设置

接口接受 SCAD/MCP 与 `squared_error`、`logistic`、`poisson`、`gamma`、`inverse_gaussian`、`negative_binomial`、`tweedie` 的组合。非高斯重拟合明确使用 `C=0`，保留已拟合的分布族、Gamma 链接、负二项离散参数 `alpha`、Tweedie 幂参数 `power`，以及父模型的 `max_iter`、截距约定和受支持的分析权重。首次尝试使用父模型的 `tol`；下文的负二项有界精度重试可使用更严格的子模型容差。logistic 和 Poisson 重拟合显式使用 `solver="irls"`，其他非高斯分布族使用 `solver="newton"`，设备均为 `device="cpu"`。若重拟合报告未收敛，则抛出 `RuntimeError`，不返回 oracle 系数表。

非高斯 oracle 重拟合还会检查：在每个正权重观测上，报告系数、拟合均值、得分和协方差必须对应同一个无惩罚模型。当前数值损失会裁剪部分预测值、均值或曲率因子。触发这些变换的重拟合无法提供 oracle 不确定性，会在返回系数表之前抛出 `RuntimeError`。

当前允许范围均为闭区间；精确位于端点的结果仍可接受，但须通过其他收敛和协方差检查：

| 分布族与链接 | 正权重观测须满足的范围 |
| --- | --- |
| Tweedie，对数链接 | 线性预测值位于 `[-50, 50]` |
| Gamma，对数链接 | 线性预测值位于 `[-30, 30]` |
| Gamma，逆幂链接 | 线性预测值位于 `[1e-4, 1e3]` |
| 逆高斯，对数链接 | 均值位于 `[0.05, 1e3]` |
| Poisson，对数链接 | 均值位于 `[1e-10, 1e6]` |
| 负二项，对数链接 | 线性预测值位于 `[-30, 30]`；`nonrobust` 还要求均值不小于 `1e-10`，HC0/HC1 不使用这一额外曲率下限 |
| Logistic，logit 链接 | Bernoulli 方差 `mu * (1 - mu)` 不小于 `1e-10`，使协方差曲率下限不被触发 |

均值范围通过对数链接对应的线性预测值端点检查。这些限制表示当前数值实现能否提供推断，不表示区间外的统计模型没有有限解。零权重观测不会触发这些限制。减小 `tol` 不能消除裁剪，也不能保证修复已被裁剪的重拟合。如果只需要点估计，可设置 `compute_inference=False`；当前实现无法提供上述范围之外的 oracle 不确定性。高斯 oracle 重拟合不使用这些非高斯范围检查。

在上述范围内，Tweedie 和负二项 oracle 重拟合的每个入选坐标（含截距）还必须通过相对一阶得分检查：将有符号加权得分与该分布族真实得分中的正贡献尺度比较，使用 `tol` 加上 float64 求和舍入误差作为容差。这避免响应单位或负二项离散参数掩盖绝对梯度提前停止时的错误结果。其他分布族保留子求解器已有的停止准则；范围检查并不为这些分布族增加相对收敛保证。正负得分的真实抵消仍然有效，零权重观测不参与检查。检查未通过时抛出 `RuntimeError`；若未被裁剪的重拟合仅因绝对梯度容差而提前停止，减小 `tol` 可能有助于达到驻点，若仍失败则需检查响应或设计的尺度。它只检查局部驻点条件，不提供系数或协方差误差界限。

仅对负二项重拟合，相对得分检查失败时最多再进行一次无惩罚 CPU 重拟合。额外尝试的绝对梯度容差根据正的加权得分贡献推导，最终相对检查仍使用调用方原来的 `tol`。每次尝试使用原来的 `max_iter`，因此最多执行两次子模型重拟合。裁剪或曲率范围检查失败不会触发重试。如果推导出的容差无法表示、重试未收敛，或重新计算的得分仍不满足原检查，则抛出 `RuntimeError`。共享求解器不变；这项有界重试不保证一定恢复成功。

Oracle 计算过程中发生错误时，也会清除拟合、预测和推断报告状态，包括此前成功拟合留下的状态。再次预测或读取推断结果前，须重新成功拟合模型。

Oracle 的 HC0/HC1 协方差在计算得分外积前统一缩放权重，避免权重的共同尺度
导致下溢或溢出，同时保留正权重观测对得分的实质贡献。
在入选支持集与重拟合结果固定时，只要所需的加权乘积和得分计算仍可表示且
具有足够精度，统一缩放权重后，协方差在数值容差内保持一致。次正规数本身的
相对精度较低。重拟合权重与非稳健协方差的既有约定不变；离散尺度固定时，
非稳健标准误不一定具有这种尺度不变性。

如果无法计算有限的稳健得分、曲率或重拟合参数的有限协方差，oracle 推断会抛出
`RuntimeError`。若工作精度下既无法验证稳定的共同权重尺度，也无法验证受限的
原始权重计算，则同样抛出该错误，即使此时可能存在有限的精确协方差。保守缩放
界限冲突时，若实际聚合乘积为有限值，所需的非零得分平方均值仍处于正规浮点
范围，且被忽略微小项的界限低于通常的聚合舍入误差，仍可接受未缩放权重。这不保证每个非零乘积都能保留，也不能解决任意
协方差病态问题。应检查响应和设计矩阵的尺度，或设置 `compute_inference=False`
只进行点估计。若活跃参数的 HC0/HC1 协方差整体为零，但有效正权重行上的
非零得分因子证明得分矩阵非零，也会抛出 `RuntimeError`。这项检查针对协方差
整体消失，不保证检测每个局部下溢的条目。真正为零的得分、个别为零的方差，
以及未入选系数的 `NaN` 不确定性仍被允许；零权重或零设计行不会触发这项
稳健协方差检查。

对于 `nonrobust` oracle 推断，若估计离散尺度为零，但任一正权重行上的响应
与计算得到的拟合均值不同，则抛出 `RuntimeError`，零设计行也包含在内。
每个正权重行上计算得到的残差都为零时，仍允许估计离散尺度为零。离散尺度为正时，每个
活跃参数的模型方差都必须为正；协方差的非对角条目仍可为零。Logistic、Poisson
及负二项重拟合的固定离散尺度仍为 1。这些检查不保证在任意尺度或病态程度下
都能准确计算协方差。

无惩罚目标还必须存在有限解。仅考虑正权重观测时，如果 Poisson、负二项或 Tweedie 响应全为零，或 logistic 响应只有一个类别，自由截距可持续改善目标而不存在有限最优值。这些边界拟合会在发布不确定性之前抛出 `ValueError`，即使求解器已停在某个有限近似值。零权重观测不会消除这一边界。全零 Gaussian 响应具有有限的最小二乘解，不会因这条规则被拒绝。

无截距时，还需考虑入选设计：预测变量同时含正负值时，这些边界响应也可能具有有限最优解。检查只有在验证了可持续改善目标的方向后才判定目标不存在有限解；若入选设计的数值条件使该边界检查无法可靠判定，则抛出 `RuntimeError`，不会声称不存在有限解，也不会返回不确定性。这项检查仅针对全零或单类响应，并非一般的分离检测；通常的重拟合收敛检查仍然适用。既无入选系数也无截距时，没有系数可供重拟合，原有不确定性未定义的行为保持不变。

标准误为正时，oracle 的 z 统计量等于重拟合估计值除以该标准误，双侧 p 值使用正态分布尾概率。因此，用同一个正数缩放估计值和标准误，且两者仍可表示时，检验结果保持不变。标准误恰好为零时区间退化为一个点；此时报告的统计量或 p 值不应按常规正态参考检验解释。

推断结果的 `method="oracle"`，使用正态（z）参考分布及所请求的受支持 `cov_type`。元数据包含 `refit_penalty="none"`、`refit_solver`、`refit_family_parameters`、CPU 数值计算信息和 `selection_adjusted=False`。重拟合专属字段仅在实际重拟合了系数时出现。非高斯重拟合还记录 `refit_attempts` 和 `refit_tol`，分别表示实际子模型拟合次数及子求解器容差。具体的负二项和 Tweedie 估计器会保留各自构造参数；若 `loss_kwargs` 也提供同名分布族参数，则其中的显式设置优先。

下例把自动 oracle 系数与在相同入选列上单独进行的无惩罚 logistic 拟合作比较。独立包装类中的 `C=0` 表示不使用正则化：

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
    compute_inference=True, inference_method="oracle", max_iter=10000, tol=1e-10,
).fit(X, y)
active = np.flatnonzero(np.abs(selection.coef_) > 1e-10)
if active.size == 0:
    raise ValueError("No feature was selected; specify a model with only an intercept")
refit = LogisticRegression(
    C=0, solver="irls", device="cpu", compute_inference=True,
    max_iter=10000, tol=1e-10,
).fit(X[:, active], y)
np.testing.assert_allclose(
    selection._inference_result.params[np.r_[0, active + 1]],
    np.r_[refit.intercept_, refit.coef_], rtol=1e-6, atol=1e-7,
)
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

`PenalizedGLM_CV.summary()` 当前会调用最终通用估计器并不存在的
`summary()` 方法，因此即使系数推断成功，也可能抛出 `AttributeError`。
请改读 `model.estimator_._inference_result` 或其 `to_dict()` 输出；
完整用法见 [CV 推断结果示例](../models/regression/generalized-linear-model.md#reading-cv-inference-results)。

Cox 分支仍然只提供估计，不提供这一系数推断接口。

一般的选择与最终重拟合约定见 [交叉验证](cross-validation.md)。

## 查看推断结果

成功完成推断后，模型会区分调用者请求的方法与最终实际解析得到的方法。相关已拟合属性包括：

- `inference_requested_method_`；
- `inference_resolved_method_`；
- `inference_method_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`。

这些字段在有值时用于说明报告的不确定性究竟对应哪个统计参数，尤其适用于带惩罚拟合或经过 CV 选择之后的结果。

当前 `post_selection_ols` 会把上述公开推断信息字段保留为 `None`；实际方法和所选活跃集的重拟合信息应查看 `model._inference_result.method` 与 `model._inference_result.metadata`，`params` 保存重拟合估计值。公开字段为空并不表示未执行推断。可运行示例见[选择后 OLS](inference-modes.md#post_selection_ols)。

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

`params` 是用于这次推断的系数估计，`conf_int` 每行给出相应参数的区间下限和上限。本例共有三个参数：先截距，再按 `X` 的两列排列；`to_dict()` 将结果返回为列表，`conf_int` 有三行、每行两个值，转成数组后形状为 `(3, 2)`。实际采用的方法是 `m_estimation`；目标标签 `penalized_estimating_equation` 提醒我们按上文的固定惩罚目标解释区间，本例固定 `alpha=0.03`。HC0 不会消除拟合的收缩偏差，也不会提供经过选择校正的区间。

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
