# Knockoff 特征选择

> 语言: 中文  
> 最后更新: 2026-10-06
> 页面定位: 方法文档  
> 切换: [English](../../en/models/knockoff.md)

## 概览

Knockoff 方法以特征选择的 FDR 控制为目标，有效性依赖构造与统计量的假设，并受下文自动构造中心化与统计量并列问题限制。当前实现包含 `fixed_x` 与 `model_x` 两条路径，统一入口为 `knockoff_filter`。`fixed_x` 通常要求 `n >= 2p`；`model_x` 基于高斯二阶近似（协方差估计 + S 矩阵），支持多次抽样聚合 W 统计量。

全部函数/选择器签名、参数、结果字段与独立可运行的 CPU 示例见[特征选择 API 参考](../reference/feature-selection-api.md)。

错误发现率（FDR）是所选特征中实际无效特征比例的期望值，未选中任何特征时该比例按零计；它不是“所有所选特征都正确”的概率。Knockoff 可以理解为匹配的负对照：每个原特征都要与一个依赖结构相似的人造对应变量竞争。若目标是控制选择错误，且构造假设可信，可以考虑本方法；若主要关注预测，可比较[逐步选择](feature-selection.md)，并用留出数据评价。

## 先验证目标错误率

调用过滤函数或拟合选择器前，应检查 `np.isfinite(q) and 0 < q < 1`。
这适用于三个过滤函数、两个选择器类及两种阈值规则。当前验证会漏过 `q=np.nan`，
并可能返回空选择、`threshold=inf`、`estimated_fdr=0.0` 或全为 False 的选择掩码。
此时结果无效，不能解释为在合理目标错误率下未发现特征。请自行拒绝非有限 q，
也不要在查看结果后改换目标错误率。

## 完整的 CPU 示例

下面使用 240 行、12 个预测变量，满足生成 fixed-X knockoff 的样本量要求。
这个示例演示 API；由于下文自动构造的中心化限制，不能据此声称名义 FDR 控制。
后文另有中心化外部配对示例链接。只有前四列影响响应，应在查看结果之前确定 q。

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter

rng = np.random.default_rng(42)
X = rng.normal(size=(240, 12))
y = X[:, :4] @ np.array([3.0, -2.5, 2.0, 1.5])
y += rng.normal(scale=0.5, size=240)
q = 0.25
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
result = fixed_x_knockoff_filter(
    X, y, q=q, method="corr_diff", random_state=7, backend="numpy",
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

统计目标是在给定 FDR 水平 `q` 下，通过 knockoff 统计量 `W` 与阈值规则（`knockoff_plus` 或 `knockoff`）选择特征集合。达到该目标仍需满足相关假设，并考虑下文说明的自动构造中心化与并列计数限制。

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

### 自动 fixed-X 构造的中心化限制

自动 fixed-X 构造会中心化 X，但生成的 Xk 列均值可能不为零。
`corr_diff`、`ols_coef_diff` 和非 knockpy 兼容模式的原生 Lasso 统计量会中心化 y。
因此，原始 Gram 矩阵相等不代表响应中心化投影 P=I−11ᵀ/n 后仍相等：
XᵀPX 与 XkᵀPXk 可能不同。即使列满秩且 n>2p，也会破坏通常的零假设
得分对可交换性论证。这与阈值并列问题独立，单纯增加 n 不能修复当前构造，
避开并列也不足以为自动生成的 fixed-X 输出声称名义 FDR 控制。

例如输入 X=[−1,1]ᵀ 的 Gram 为 2；经过单位范数标准化后，
工作设计为 X_work=[−1,1]ᵀ/√2。构造种子为 7 时，生成的 knockoff 约为
[0.707,0.707]ᵀ。这两个工作矩阵在投影前的 Gram 都是 1，投影后却分别为 1 和 0。
任何非常数中心化响应及其相反数都会得到正的相关差统计量。这展示的是零假设符号对称性丢失，
不是测得 FDR 超标：单特征 knockoff+ 不会选择任何变量。

真正中心化且满足完整配对约束的外部 X/Xk 可以避开这一几何问题。
应检查截距/干扰变量投影后的完整匹配 Gram 条件，而不只是相同形状或边际范数。
[中心化正交 QR 示例](../reference/feature-selection-api.md#repeated-lasso-statistic-calls)
构造了一个要求 n≥2p+1 的特殊设计，不是任意 X 的通用修复；事后单独中心化
自动生成的 Xk 也可能破坏 Gram 约束。外部配对几何有效时，响应假设、
统计量有效性、阈值并列和缓存限制仍然需要检查。

### 响应模型假设

引用的 [fixed-X 有限样本结果](https://arxiv.org/html/1404.5609v3)要求高斯线性响应
模型 y=b1+Xβ+ε，其中 ε∼N(0,σ²I) 为相互独立、同方差的正态误差，
并要求配对设计、统计量和截距处理相容。函数接受有限 y，并不表示这些假设成立。

[Model-X 理论](https://arxiv.org/abs/1610.02351)则依赖特征配对可交换性，
以及给定 X 后 knockoff 与 y 条件独立，响应关系可以是任意形式。
因此它不只是保留相同假设的小样本替代方案。本实现估计高斯特征分布并可平均
多次抽样统计量，仍受下文所述限制。

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
| `q` | `0.1` | 有限 FDR 控制目标，必须在 `(0, 1)`；当前内部检查会漏过 NaN，调用前请自行验证。 |
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
| `modelx_smatrix_method` | `mvr` | knockpy 兼容路径 S 矩阵方法 |
| `knockpy_sampler` | `None` | 可选分发入口 |
| `knockpy_sampler_method` | `None` | `sampler=gaussian` 时子方法 |

<a id="reproducibility-of-generated-torch-model-x"></a>

## Torch 自动 model-X 构造的可重复性

当 `knockoff_type="model_x"`、`compat_mode="statgpu"` 且未提供 `Xk` 时，
当前 Torch 构造使用全局 Torch 随机数状态，而不是请求的 `random_state`。
重复相同种子的调用仍可能改变 W 和入选特征；仅改变 `random_state` 也不能
控制这些构造抽样。相同设置下的 `KnockoffSelector` 也受影响，且这与下文
Lasso 缓存问题不同。Fixed-X 构造不受这一特定种子问题影响。

需要可重复的原生 model-X 构造时，可使用 `backend="numpy"`；也可提供有效的
外部 `Xk`，并另行验证所选统计量的可重复性。提供矩阵只会跳过构造，不能免除
统计假设、阈值及 Lasso 缓存限制。结果中记录了 `random_state`，不代表构造
实际使用了该种子。

## 重复计算 Lasso 统计量

使用 `method="lasso_coef_diff"` 且 `random_state` 为整数时，原地修改 X、y
或 Xk 可能在没有提示的情况下复用旧统计量。仅新建选择器不能避免这个问题；
旧输入占用的内存被再次使用时也有相同风险。需要对变化后的数据进行可重复分析时，
每次调用应在新的 Python 进程中执行。若提供的 X/y/Xk 均为 float64 NumPy 数组，也可为三个输入都创建新副本，
并在整个分析过程中保留所有旧输入数组且不修改它们。`random_state=None` 可避开
这种按种子复用的行为，但不再具有固定种子的可重复性。详见[限制与安全复制示例](../reference/feature-selection-api.md#repeated-lasso-statistic-calls)。

## 如何选择 Lasso 计算配置

`lasso_fast_profile="off"` 保留默认的统计量拟合设置。`auto`、`moderate` 与
`aggressive` 可能改变 CV 折数、候选惩罚、迭代预算或容差，因此 W 与入选特征
也可能变化，不能仅把它当作运行速度开关。应在查看发现结果前选定配置；
比较后端或不同实现时，保持该设置一致。

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

本模块不使用 `strict/approx` 推断口径开关。`fixed_x` 与 `model_x` 采用不同的设计或特征分布假设，应根据统计问题选择，不能把两者当作速度或数值精度档位。`modelx_draws`（必须为正整数）影响计算量与蒙特卡洛波动；后端选择均不替代构造假设或阈值规则。这里 `backend="torch"` 选择 Torch 计算库：NumPy 或 Torch CPU 输入在 CPU 上执行；GPU 计算需传入 CUDA 张量，外部 Xk 也应位于同一设备。这不同于估计器要求 CUDA 的 `device="torch"`。

## 输出（Outputs）

过滤函数返回 `KnockoffResult`；选择器 `fit` 返回 `self`，其结果读取 `result_`，所选索引读取 `selected_features_`。结果常用字段：

- `selected_features`
- `W`
- `threshold`
- `estimated_fdr`
- `q_trajectory`
- `metadata`（例如抽样次数、兼容模式、`Xk` 来源）

`selected_features` 为空是有效结果。`estimated_fdr` 是阈值规则给出的估计，不是实际观测到的错误比例，也不是每个特征的 p 值。应结合 `W` 与 `threshold` 检查上文的自动构造中心化与并列限制，并在未参与选择的数据上评价后续预测模型。

## 常见问题（FAQ）

- **哪些输入会直接报错？**  
  `q <= 0` 或 `q >= 1`、`X` 非二维、`y` 长度与 `X` 不一致、`Xk` 形状不匹配会报错。但当前 `q=np.nan` 会漏过检查并产生无效空选择；调用前必须自行检查 q 有限且在 `(0,1)` 内。
- **`knockpy_sampler` 可以直接用吗？**  
  只有 `compat_mode="knockpy"` 的 model-X 自动构造会使用该参数；目前的采样器尚未实现，会抛出 `NotImplementedError`。默认 `statgpu` 模式、fixed-X 或已经提供 `Xk` 时忽略该参数，并不执行请求的采样器。
- **`fixed_x` 的主要约束是什么？**  
  通常要求样本规模与矩阵秩满足构造条件（常见约束为 `n >= 2p`）。这些条件不能保证中心化配对几何有效，也不能替代高斯线性响应假设。

## 外部验证（External Validation）

推荐验证脚本：

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`

## 参考（References）

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)
