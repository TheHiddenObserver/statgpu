# 交叉验证

> 语言：中文  
> 最后更新：2026-09-29  
> 页面定位：CV 选择与最终重拟合的用户指南  
> 切换：[English](../../en/guides/cross-validation.md)

## 概述

statgpu 中的交叉验证是包在基础估计器外层的**模型选择过程**。典型流程是：

```text
候选参数网格
    -> 在训练折上拟合候选模型
    -> 在验证折上计算分数
    -> 选择候选参数
    -> 在全部观测上重新拟合所选配置
```

本页说明用户需要直接配置、或需要正确理解的公开行为：如何设置数据折和调参网格，求解器与设备如何配合 CV，最终重拟合发生在哪一步，以及 CV 之后的推断应当怎样理解。

如果想进一步理解为什么“选择”和“最终重拟合”要分成两个阶段、沿参数路径复用计算和 GPU 批处理怎样加速、选择缓存能够复用什么，以及这些优化为什么不能改变原来的统计问题，请看 [statgpu 的交叉验证如何工作](cross-validation-design.md)。私有快速路径、缓存键的精确字段、辅助函数名称，以及根据基准测量确定的分发阈值，仍属于内部实现。

## 可用的 CV 估计器

| CV 估计器 | 基础模型 | 主要调参目标 |
|---|---|---|
| `RidgeCV` | `Ridge` | `alpha` |
| `LassoCV` | `Lasso` | `alpha` |
| `ElasticNetCV` | `ElasticNet` | `alpha`，以及可选的 `l1_ratio` |
| `LogisticRegressionCV` | `LogisticRegression` | 正则化强度 |
| `PenalizedGLM_CV` | 惩罚 GLM / 惩罚 Cox | 给定损失函数与惩罚项下的 `alpha` |
| `CoxPHCV` | `CoxPH` | Cox 惩罚强度 |

精确的“损失函数 × 惩罚项 × 求解器”支持关系见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。某个具体模型的统计限制应以对应模型页为准。

## 快速开始

### RidgeCV

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=None,
    n_alphas=100,
    cv=5,
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
print(model.score(X_test, y_test))
```

### LassoCV

`LassoCV` 明确区分交叉验证阶段使用的算法与最终全数据重拟合使用的算法：

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    cv=5,
    cv_solver="auto",   # 交叉验证阶段
    solver="fista",     # 最终全数据重拟合
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
print(model.cv_solver_)
```

`cv_solver_` 记录选择阶段实际执行的算法。旧的 `cpu_solver` 参数已经弃用；迁移方式见 [惩罚模型求解器 API 迁移指南](penalized-solver-api-migration.md)。

### PenalizedGLM_CV

```python
from statgpu.linear_model import PenalizedGLM_CV

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="scad",
    cv=5,
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
pred = model.predict(X_test)
```

只要对应的损失函数与惩罚项组合支持某个显式求解器，就以用户的显式请求为准；不支持的显式组合会直接报错，而不是静默切换算法。`solver="auto"` 则按照对应模型族公开的自动分发规则选择算法。

## 数据折

`cv` 为整数时，statgpu 会构造 k 折训练/验证划分；如果估计器提供 `random_state`，可以用它让随机划分可复现。

提供 `cv_splits` 参数的估计器还可以直接接收训练索引与验证索引组成的配对：

```python
from sklearn.model_selection import TimeSeriesSplit
from statgpu.linear_model import PenalizedGLM_CV

tscv = TimeSeriesSplit(n_splits=5)

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l1",
    cv_splits=list(tscv.split(X)),
)
model.fit(X, y)
```

当普通随机数据折不符合时间顺序或分组结构时，应提供合适的自定义划分。每对索引都应是一维整数数组，训练集和验证集均非空、互不重叠，且各自没有重复行。校验范围因估计器而异；共享划分器可能转换或展平索引，并跳过空的数据折，因此调用被接受并不证明划分有效。具体划分是否符合科学问题，仍须由用户判断。

**当前 RidgeCV 限制。** 未提供 `sample_weight` 且各验证集恰好覆盖每个观测一次时，
RidgeCV 可能把指定的训练子集替换为对应验证集的完整补集，使原本排除的行重新进入
训练，进而改变验证误差或选出的 alpha。普通完整 k 折本来就使用这种补集；任意
自定义训练子集则未必如此。需要排除、留空窗或设置隔离期的训练行时，请使用显式
外部交叉验证循环，只拟合指定的行，详见
[RidgeCV 自定义训练参考](../reference/linear-model-api.md#custom-ridgecv-training-subsets)。
拟合成功或评分有限，都不能证明实际使用了预期训练行。


对于 `PenalizedGLM_CV`，`cv_splits` 也可以是 generator 等一次性 iterator。statgpu 会在首次需要时在内部生成可复用的快照，并在重复 `fit()`、scikit-learn clone 和 pickle 序列化中复用；公开的 `cv_splits` 属性本身不会在拟合时被改写。能够重复迭代的 list/tuple 则直接使用。

### Cox 数据

Cox CV 还需要满足事件信息和风险集相关的要求。对于 `PenalizedGLM_CV(loss="cox_ph", ...)`，生存结局在选择过程中保持二维形式，并且每个实际参与评估的训练/验证分区都必须包含 Cox 评分所需的事件信息；不合法的分区会在候选项选择之前报错。

惩罚 Cox CV 不支持 `sample_weight`，也不提供选择后的系数推断。包括 `CoxPHCV` 在内的完整生存分析约定见 [Cox 比例风险模型](../models/coxph.md)。

## 调参网格

`RidgeCV`、`LassoCV`、`ElasticNetCV` 等专用估计器使用 `alphas`；`PenalizedGLM_CV` 使用 `alpha_grid`。

没有显式提供网格时，估计器会根据数据与模型构造相应的候选网格。用户显式提供的网格通过该估计器的公开输入校验后，即成为本次请求的候选集合。

对于 `PenalizedGLM_CV` 的分位数回归路径，自动网格要先确定一个与问题尺度匹配的 `alpha` 起点。它不会用平方残差近似，而是在调用者要求的分位数上、以“仅含截距”的基准模型计算 check 损失（pinball 损失）的次梯度；传入解析权重 `sample_weight` 时，同一个归一化加权次梯度会一致地用于这一步。

分组 SCAD/MCP 的起点会进一步换算到公开的分组惩罚 `alpha` 尺度：先由 `max_g ||score_g||_2 / sqrt(p_g)` 得到特征分数，再与惩罚项的组阈值 `alpha * sqrt(p_g)` 对齐。若 Adaptive L1 的权重是用户给定的固定正值，则先按各坐标的有效自适应权重对次梯度做逐坐标除法，再取最大值；固定正权重的 Adaptive Group Lasso 同理，使用 `max_g ||score_g||_2 / (w_g sqrt(p_g))`。当自适应权重尚未固定、需要由初始化拟合产生时，自动网格仍只是初始化前的启发式起点，并不代表精确的全零 KKT 阈值。

```python
import numpy as np
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=np.logspace(-4, 2, 50),
    cv=5,
)
model.fit(X, y)
```

对于标量响应的惩罚 CV，通常在正值范围内搜索正则化强度。如果研究目标本身就是无惩罚拟合，应直接使用相应估计器的零惩罚或无惩罚配置，不要假定所有 CV 估计器都会把 0 当作普通候选值。

Cox 调参网格有生存分析专属的验证规则，应查看 Cox 模型页，不要从标量响应 CV 的规则直接推断。

## 样本权重

当某个 CV 估计器支持 `sample_weight` 时，权重会进入训练折的目标函数和相应的加权验证准则，并在选择结束后用于全数据最终重拟合。对于 `PenalizedGLM_CV`，每个实际参与评估的训练折和验证折都必须保留有限且严格为正的解析权重总质量；若某一折的权重总和为 0，则所声明的加权目标本身没有定义，系统会在拟合候选之前报错，而不会把该折静默改成不加权评分。

对于分位数路径，严格选择还要求每个 `alpha` 都具备完整且有限的逐折证据：若某一折拟合失败、没有产生有限得分，该 `alpha` 就整体失去候选资格，不能只用其余折的平均分继续参与选择。自动标量/分组非凸分位数路径会把目标 `alpha` 上的明确收敛失败视为“该折不参与评分”；如果普通求解器只是发出 `ConvergenceWarning`、但仍返回有限结果，则不会仅凭这条警告丢弃候选。`cv_strategy="two_stage"` 的第一阶段仍采用宽松筛选，完整证据要求只在严格细化阶段和普通严格交叉验证中执行。

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(cv=5)
model.fit(X, y, sample_weight=w)
```

权重支持并不是“CV”这一名称自动带来的统一能力，而取决于基础损失函数、惩罚项和求解路径。如果显式指定的求解器不支持所请求的带权目标，statgpu 会报错，而不是丢弃权重或静默改成另一个目标函数。具体组合以 [兼容性矩阵](solver-penalty-matrix.md) 和模型页为准。

## 选择与最终重拟合

CV 在各数据折上的拟合只用于评估候选模型。候选网格完成评分后，所选超参数配置会在完整数据上重新拟合一次。

因此：

- `alpha_`（以及适用时的 `l1_ratio_`）属于选择阶段；
- `coef_`、`intercept_`、预测结果与常规已拟合模型诊断属于最终全数据重拟合；
- 当估计器分别提供 CV 求解器和最终重拟合求解器的控制参数时，两个阶段可以合法使用不同算法；
- 支持推断的估计器会在选择完成后，对最终重拟合执行推断，而不是在每个数据折内分别发布推断结果。

为什么这一分离属于 CV 设计本身，而不仅仅是实现细节，见 [statgpu 的交叉验证如何工作](cross-validation-design.md)。

## CV 后推断

对于支持 `compute_inference=True` 的估计器，候选模型的拟合只用于选择。statgpu 会先选择调参值，再在全部观测上拟合所选模型，最后才执行请求的推断。

因此，得到的区间和 p 值是**以已经选定的调参配置为条件**的推断，并不会自动修正“调参值由 CV 选择”所引入的额外不确定性。`PenalizedGLM_CV` 会通过推断元数据说明这一点。

统计解释与支持的方法见 [推断模式](inference-modes.md) 和 [惩罚 GLM 推断](penalized-glm-inference.md)。

## 设备行为

CV 与直接估计器采用相同的显式设备规则：

- `device="cpu"` 请求 NumPy CPU 计算；
- `device="cuda"` 请求 CuPy CUDA，不可用时直接报错；
- `device="torch"` 请求 Torch CUDA，不可用时直接报错；
- `device="auto"` 可以根据估计器、工作量和可用后端自动选择。

自动路由属于实现层面的选择，可能随实际性能测量而变化。应用代码不应依赖某个内部样本量或维度阈值；如果需要固定执行后端，请显式指定 `device`。

公开设计页进一步解释了为什么自动后端选择和 GPU 批处理可以变化，却不能改变 CV 的统计问题：[statgpu 的交叉验证如何工作](cross-validation-design.md)。完整设备语义见 [设备与 GPU 内存](device-and-memory.md)。

## 拟合结果

不同 CV 估计器公开的详细结果结构并不完全相同，但都会通过相应属性暴露所选调参值以及最终拟合结果。常见属性包括：

| 属性 | 含义 |
|---|---|
| `alpha_` | 选定的正则化强度 |
| `l1_ratio_` | 搜索 ElasticNet 混合比例时选定的值 |
| `cv_results_` | 该估计器公开的候选项/数据折评分信息 |
| `estimator_` | 适用时，在全数据上重新拟合的最终估计器 |
| `coef_`, `intercept_` | 最终重拟合的参数 |
| `cv_solver_` | `LassoCV` 实际执行的 CV 阶段求解算法 |

不要假定所有 CV 类的 `cv_results_` 都具有完全相同的数据结构；如果程序依赖某个具体诊断字段，应查看相应估计器或模型的参考文档。

## 如何选择 CV 配置

一个更稳妥的顺序是：

1. 先确定统计模型和惩罚项；
2. 选择符合数据生成结构的数据折；
3. 没有特别理由时，优先使用估计器自动生成的调参网格；
4. 除非需要固定硬件、算法或可复现的执行路径，否则可以保留求解器和设备的自动选择；
5. 除非某种方法明确说明已经校正调参选择带来的不确定性，否则应把 CV 后推断理解为对所选配置条件下的推断。

## 相关文档

- [statgpu 的交叉验证如何工作](cross-validation-design.md) — 公开执行模型、加速思想与统计不变量
- [已实现方法](implemented-methods.md) — 可用的公开估计器
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) — 显式组合的兼容性
- [求解器算法](solver-algorithms.md) — 优化算法
- [设备与 GPU 内存](device-and-memory.md) — 后端与设备语义
- [推断模式](inference-modes.md) — 如何选择推断方法
- [惩罚 GLM 推断](penalized-glm-inference.md) — 惩罚拟合/调参后的推断
- [Cox 比例风险模型](../models/coxph.md) — 生存分析专属的 CV 行为
