# Knockoff 特征选择

> 语言: 中文  
> 最后更新: 2026-10-05  
> 页面定位: 方法文档  
> 切换: [English](../../en/models/knockoff.md)

## 概览

Knockoff 方法以特征选择的 FDR 控制为目标，有效性依赖构造与统计量的假设，并受下文统计量并列问题限制。当前实现包含 `fixed_x` 与 `model_x` 两条路径，统一入口为 `knockoff_filter`。`fixed_x` 通常要求 `n >= 2p`；`model_x` 基于高斯二阶近似（协方差估计 + S 矩阵），支持多次抽样聚合 W 统计量。

全部函数/选择器签名、参数、结果字段与独立可运行的 CPU 示例见[特征选择 API 参考](../reference/feature-selection-api.md)。

错误发现率（FDR）是所选特征中实际无效特征比例的期望值，未选中任何特征时该比例按零计；它不是“所有所选特征都正确”的概率。Knockoff 可以理解为匹配的负对照：每个原特征都要与一个依赖结构相似的人造对应变量竞争。若目标是控制选择错误，且构造假设可信，可以考虑本方法；若主要关注预测，可比较[逐步选择](feature-selection.md)，并用留出数据评价。

## 完整的 CPU 示例

下面使用 240 行、12 个预测变量，满足生成 fixed-X knockoff 的样本量要求。
只有前四列影响响应。应在查看选择结果之前确定 q。

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter

rng = np.random.default_rng(42)
X = rng.normal(size=(240, 12))
y = X[:, :4] @ np.array([3.0, -2.5, 2.0, 1.5])
y += rng.normal(scale=0.5, size=240)
result = fixed_x_knockoff_filter(
    X, y, q=0.25, method="corr_diff", random_state=7, backend="numpy",
)
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
```

该种子下，输出为 `Selected columns: [0, 1, 2, 3]`，阈值 `21.637`，
阈值计数比 `0.25`。入选索引对应原始列；正 W 表示原特征的重要性超过其 knockoff。
阈值计数比是规则给出的估计值，不是这一次样本中实际无效特征的比例。
其他样本可能遗漏信号或选中噪声。过滤函数不拟合系数或预测模型；若随后评价预测，
只能使用训练行进行选择，并将评价数据单独保留。

## 路径

主路径：
- `statgpu.feature_selection.knockoff_filter`
- `statgpu.feature_selection.fixed_x_knockoff_filter`
- `statgpu.feature_selection.model_x_knockoff_filter`
- `statgpu.feature_selection.KnockoffSelector`
- `statgpu.feature_selection.FixedXKnockoffSelector`

顶层别名：
- `statgpu.knockoff_filter`
- `statgpu.fixed_x_knockoff_filter`
- `statgpu.model_x_knockoff_filter`
- `statgpu.KnockoffSelector`
- `statgpu.FixedXKnockoffSelector`

## 目标函数

统计目标是在给定 FDR 水平 `q` 下，通过 knockoff 统计量 `W` 与阈值规则（`knockoff_plus` 或 `knockoff`）选择特征集合。达到该目标仍需满足相关假设，并避开下文说明的并列计数限制。

## 估计方程

构造 knockoff 特征并计算反对称特征统计量 $W_j$ 后，knockoff+ 使用阈值

$$
T=\min\left\{t\in\{|W_j|:|W_j|>0\}:\frac{1+\#\{j:W_j\le-t\}}{\max(1,\#\{j:W_j\ge t\})}\le q\right\},\qquad
\widehat S=\{j:W_j\ge T\}.
$$

q 为目标 FDR，$\#\{\cdot\}$ 表示计数；没有合格阈值时选择为空。`fdr_control="knockoff"` 将分子中的 1 换为 0，其理论错误率目标与 knockoff+ 不同，不能仅把两者理解为数值精度模式。

`model_x` 可通过 `modelx_draws` 进行多次采样并平均统计量。增加抽样有助于减小蒙特卡洛波动，但不自动为平均后的统计量建立 FDR 保证；估计的高斯特征模型也不能保证任意特征分布下的可交换性。

### 统计量并列时的限制

上式是理论 knockoff+ 阈值规则。当前实现遇到绝对统计量并列时可能与该规则不一致：先评估并列组中的部分累计计数，再选择整个阈值组。W=[8,8,-8]、q=0.5 时，可能报告阈值 8、estimated_fdr=0.5，但完整阈值计数比为 (1+1)/2=1，理论规则应不选择任何特征。阈值出现并列时，不能把当前输出解释为满足名义 knockoff+ FDR 控制。偏移为 0 的 `knockoff` 也受同一计数限制影响。重新拟合系数、增加抽样次数或更换设备都不能修复它。

## 协方差与推断

Knockoff 为选择推断框架，不采用回归模型中的 `cov_type` 协方差配置。其统计保证依赖 knockoff 构造假设与阈值规则。

`compat_mode` 支持：
- `statgpu`：默认实现
- `knockpy`：兼容路径（部分采样器分发入口仍为占位）。可选包缺失或 S 矩阵求解报错时，可能改用样本协方差或等相关 S 矩阵。应检查 `metadata["modelx_covariance_estimator"]` 与 `metadata["modelx_smatrix_source"]`，不能仅凭请求名称判断执行方法，详见[兼容模式参考](../reference/feature-selection-api.md)。

## 参数

统一入口 `knockoff_filter` 关键参数如下：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `knockoff_type` | `fixed_x` | `fixed_x` 或 `model_x` |
| `q` | `0.1` | FDR 控制目标，必须在 `(0, 1)` |
| `method` | `corr_diff` | `corr_diff` / `ols_coef_diff` / `lasso_coef_diff` |
| `fdr_control` | `knockoff_plus` | `knockoff_plus` 或 `knockoff` |
| `random_state` | `None` | 随机种子 |
| `backend` | `auto` | 计算后端：`auto` / `numpy` / `cupy` / `torch` |
| `Xk` | `None` | 外部提供 knockoff 设计矩阵，形状需与 `X` 相同 |
| `compat_mode` | `statgpu` | `statgpu` 或 `knockpy` |
| `lasso_cv_impl` | `auto` | `auto` / `statgpu` / `sklearn` |
| `lasso_fast_profile` | `off` | Lasso 快速配置 |
| `modelx_covariance_shrinkage` | `0.20` | model-X 协方差收缩系数 |
| `modelx_s_scale` | `0.999` | model-X `S` 缩放系数 |
| `modelx_draws` | `None` | model-X 抽样次数 |
| `modelx_shrinkage` | `ledoitwolf` | knockpy 兼容路径协方差估计策略 |
| `modelx_smatrix_method` | `mvr` | knockpy 兼容路径 `S`-matrix 方法 |
| `knockpy_sampler` | `None` | 可选分发入口 |
| `knockpy_sampler_method` | `None` | `sampler=gaussian` 时子方法 |

## 重复计算 Lasso 统计量

使用 `method="lasso_coef_diff"` 且 `random_state` 为整数时，原地修改 X、y
或 Xk 可能在没有提示的情况下复用旧统计量。仅新建选择器不能避免这个问题；
旧输入占用的内存被再次使用时也有相同风险。需要对变化后的数据进行可重复分析时，
每次调用应在新的 Python 进程中执行。若提供的 X/y/Xk 均为 float64 NumPy 数组，也可为三个输入都创建新副本，
并在整个分析过程中保留所有旧输入数组且不修改它们。`random_state=None` 可避开
这种按种子复用的行为，但不再具有固定种子的可重复性。详见[限制与安全复制示例](../reference/feature-selection-api.md#repeated-lasso-statistic-calls)。

## 性能与统计边界

运行时间取决于样本量、特征数、统计量、抽样次数及数据传输。不存在对所有问题通用的 GPU 提速倍数。高斯二阶 model-X 的有效性依赖特征分布与构造假设，不能保证任意数据上的分布无关 FDR 控制。

## CPU/GPU 示例

以下可选 GPU 示例使用[独立 CPU 示例](../reference/feature-selection-api.md#runnable-fixed-x-example)生成的 X/y；CuPy/Torch 需要已安装且可用的 GPU 后端。

```python
from statgpu import fixed_x_knockoff_filter, knockoff_filter

# CPU: fixed-X
res_cpu = fixed_x_knockoff_filter(
    X,
    y,
    q=0.1,
    method="ols_coef_diff",
    backend="numpy",
)

# GPU: model-X
res_gpu = knockoff_filter(
    X,
    y,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="cupy",
    modelx_draws=3,
)

# GPU Torch: fixed-X
import torch
X_torch = torch.from_numpy(X).to('cuda')
y_torch = torch.from_numpy(y).to('cuda')

res_torch = fixed_x_knockoff_filter(
    X_torch, y_torch,
    q=0.1,
    method="lasso_coef_diff",
    backend="torch",
)

# GPU Torch: model-X
res_torch_mx = knockoff_filter(
    X_torch, y_torch,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="torch",
    modelx_draws=3,
)
```

## 严格与近似模式的差别

本模块不使用 `strict/approx` 推断口径开关。`fixed_x` 与 `model_x` 采用不同的设计或特征分布假设，应根据统计问题选择，不能把两者当作速度或数值精度档位。`modelx_draws`（必须为正整数）影响计算量与蒙特卡洛波动；后端 `numpy`、`cupy`、`torch` 决定执行位置，均不替代构造假设或阈值规则。

## 输出（Outputs）

过滤函数返回 `KnockoffResult`；选择器 `fit` 返回 `self`，其结果读取 `result_`，所选索引读取 `selected_features_`。结果常用字段：

- `selected_features`
- `W`
- `threshold`
- `estimated_fdr`
- `q_trajectory`
- `metadata`（例如抽样次数、兼容模式、`Xk` 来源）

`selected_features` 为空是有效结果。`estimated_fdr` 是阈值规则给出的估计，不是实际观测到的错误比例，也不是每个特征的 p 值。应结合 `W` 与 `threshold` 检查上文的并列限制，并在未参与选择的数据上评价后续预测模型。

## 常见问题（FAQ）

- **哪些输入会直接报错？**  
  `q` 不在 `(0,1)`、`X` 非二维、`y` 长度与 `X` 不一致、`Xk` 形状不匹配都会报错。
- **`knockpy_sampler` 可以直接用吗？**  
  当前显式传入会触发 `NotImplementedError`（占位保护）；不传时走稳定路径。
- **`fixed_x` 的主要约束是什么？**  
  通常要求样本规模与矩阵秩满足构造条件（常见约束为 `n >= 2p`）。

## 外部验证（External Validation）

推荐验证脚本：

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`

## 参考（References）

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)
