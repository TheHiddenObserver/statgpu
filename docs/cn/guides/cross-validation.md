# 交叉验证

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：CV 选择与最终重拟合的用户指南  
> 切换：[English](../../en/guides/cross-validation.md)

## 概述

statgpu 中的交叉验证是包在 estimator 外层的**模型选择过程**。典型流程是：

```text
候选参数网格
    -> 在训练折上拟合候选模型
    -> 在验证折上计算分数
    -> 选择候选参数
    -> 在全部观测上重新拟合所选配置
```

本页只说明用户需要依赖的公开行为：fold 与 tuning grid 如何配置，solver/device 如何与 CV 交互，最终 refit 属于哪一步，以及 CV 后推断应如何解释。

GPU batching、内部 fast path、cache key、benchmark 得出的自动路由阈值等属于实现细节，不是本页的用户契约。

## 可用的 CV estimator

| CV estimator | 基础模型 | 主要 tuning 目标 |
|---|---|---|
| `RidgeCV` | `Ridge` | `alpha` |
| `LassoCV` | `Lasso` | `alpha` |
| `ElasticNetCV` | `ElasticNet` | `alpha`，以及可选的 `l1_ratio` |
| `LogisticRegressionCV` | `LogisticRegression` | 正则化强度 |
| `PenalizedGLM_CV` | penalized GLM / penalized Cox | 给定 loss 与 penalty 下的 `alpha` |
| `CoxPHCV` | `CoxPH` | Cox penalty 强度 |

精确的 loss × penalty × solver 支持关系见 [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md)。某个具体模型的统计限制应以对应模型页为准。

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

`LassoCV` 明确区分 CV 路径使用的算法与最终全数据重拟合使用的算法：

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    cv=5,
    cv_solver="auto",   # CV folds/path
    solver="fista",     # 最终全数据 refit
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
print(model.cv_solver_)
```

`cv_solver_` 记录选择阶段实际执行的算法。旧的 `cpu_solver` 控制已经弃用；迁移方式见 [penalized solver API 迁移指南](penalized-solver-api-migration.md)。

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

只要对应 loss/penalty 组合支持某个显式 solver，请求的 solver 就保持权威；不支持的显式组合会报错，而不是静默换算法。`solver="auto"` 则按对应模型族公开的 dispatch 规则选择。

## Fold

当 `cv` 为整数时，statgpu 构造 k-fold 训练/验证划分；在 estimator 暴露 `random_state` 的情况下，可用它控制可重复的随机划分。

暴露 `cv_splits` 的 estimator 还可以直接接收 train/validation index pair：

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

当随机打乱的普通 k-fold 不符合数据结构时，例如有时间顺序或分组结构，应由用户提供合适的 split。statgpu 会验证 split 的输入契约，但不会替用户判断某个划分是否符合具体应用的科学设计。

### Cox 数据

Cox CV 还需要满足事件信息与 risk-set 相关条件。对 `PenalizedGLM_CV(loss="cox_ph", ...)`，survival target 在选择过程中保持二维，并且每个被评估的训练/验证分区必须包含 Cox scoring 所需的事件信息；非法分区会在候选选择前报错。

penalized Cox CV 不支持 `sample_weight`，也不提供 post-selection coefficient inference。包括 `CoxPHCV` 在内的完整 survival contract 见 [Cox 比例风险模型](../models/coxph.md)。

## Tuning grid

`RidgeCV`、`LassoCV`、`ElasticNetCV` 等专用 estimator 使用 `alphas`；`PenalizedGLM_CV` 使用 `alpha_grid`。

未显式给出网格时，estimator 会依据数据和模型构造相应的候选网格。用户给出的网格经过该 estimator 的公开输入验证后，作为请求的候选集合使用。

```python
import numpy as np
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=np.logspace(-4, 2, 50),
    cv=5,
)
model.fit(X, y)
```

标量响应的 penalized CV 通常搜索正的正则化强度。如果科学问题本身需要无惩罚拟合，应使用相应 direct estimator 的零惩罚/无惩罚配置，而不是假定所有 CV estimator 都把 0 当作普通候选。

Cox grid 有自己的 survival-specific 验证规则，应查看 Cox 模型页，不要从标量响应 CV 推断。

## 样本权重

当某个 CV estimator 支持 `sample_weight` 时，权重进入训练折目标函数与对应的加权验证准则，并在选择完成后用于全数据 final refit。

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(cv=5)
model.fit(X, y, sample_weight=w)
```

权重支持并不是“CV”这一名称自动带来的统一能力，而取决于基础 loss、penalty 与 solver route。如果显式 solver 不支持请求的加权目标，statgpu 会报错，而不是丢弃权重或静默改成另一个目标函数。具体组合以 [兼容性矩阵](solver-penalty-matrix.md)和模型页为准。

## 选择与最终重拟合

CV fold 内的拟合只是候选拟合。候选网格完成评分后，所选 hyperparameter 配置会在完整数据上重新拟合一次。

因此：

- `alpha_`（以及适用时的 `l1_ratio_`）属于选择阶段；
- `coef_`、`intercept_`、预测与普通 fitted-model diagnostics 属于最终全数据 refit；
- 当 estimator 分别提供 CV solver 与 final-refit solver 控制时，两阶段可以合法使用不同算法；
- 支持推断的 estimator 会在选择完成后对最终 refit 做推断，而不是在每个 fold 内分别发布推断结果。

## CV 后推断

对于支持 `compute_inference=True` 的 estimator，candidate fit 保持 selection-only。statgpu 先选择 tuning parameter，再在全部观测上拟合所选模型，最后才执行请求的 inference。

因此得到的区间与 p 值是**以已经选定的 tuning configuration 为条件**的推断，并不会自动修正“参数由 CV 选择”带来的额外不确定性。`PenalizedGLM_CV` 会通过 inference metadata 区分这一点。

统计解释与支持方法见 [推断模式](inference-modes.md)和 [Penalized GLM 推断](penalized-glm-inference.md)。

## Device 行为

CV 与 direct estimator 使用相同的显式设备规则：

- `device="cpu"` 请求 NumPy CPU；
- `device="cuda"` 请求 CuPy CUDA，不可用时直接报错；
- `device="torch"` 请求 Torch CUDA，不可用时直接报错；
- `device="auto"` 可以根据 estimator、工作量与可用 backend 自动选择。

自动路由属于实现选择，可能随实际性能测量而变化。应用代码不应依赖某个内部样本量/维度阈值；如果必须固定执行 backend，请显式指定。

完整设备语义见 [设备与 GPU 内存](device-and-memory.md)。

## 拟合结果

不同 CV estimator 的结果字典并不完全相同，但都会通过公开属性暴露所选 tuning 值以及相应的最终拟合结果。常见属性包括：

| 属性 | 含义 |
|---|---|
| `alpha_` | 选定的正则化强度 |
| `l1_ratio_` | 搜索 ElasticNet mixing 时选定的值 |
| `cv_results_` | 该 estimator 对外公开的 candidate/fold scoring 信息 |
| `estimator_` | 适用时，在全数据上重新拟合的最终 estimator |
| `coef_`, `intercept_` | 最终 refit 的参数 |
| `cv_solver_` | `LassoCV` 实际执行的 CV-path solver |

不要假定所有 CV class 的 `cv_results_` 都具有完全相同的 schema；程序若依赖某个具体诊断字段，应查看相应 estimator/model reference。

## 如何选择 CV 配置

一个更稳妥的顺序是：

1. 先确定统计模型和 penalty；
2. 选择符合数据生成结构的 fold；
3. 没有特别理由时先使用 estimator 的自动 grid；
4. 除非需要固定硬件、算法或可复现执行路径，否则保留 solver/device 的自动选择；
5. 除非某种方法明确说明已经校正 tuning-selection uncertainty，否则把 CV 后 inference 理解为对所选配置条件下的推断。

## 相关文档

- [已实现方法](implemented-methods.md) — 可用 public estimator
- [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md) — 显式组合兼容性
- [求解器算法](solver-algorithms.md) — 优化算法
- [设备与 GPU 内存](device-and-memory.md) — backend/device 语义
- [推断模式](inference-modes.md) — 如何选择 inference method
- [Penalized GLM 推断](penalized-glm-inference.md) — penalized fitting/tuning 后推断
- [Cox 比例风险模型](../models/coxph.md) — survival-specific CV 行为
