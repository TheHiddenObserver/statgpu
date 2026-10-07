# Knockoff 特征选择

> 语言: 中文  
> 最后更新: 2026-10-07
> 页面定位: 方法文档  
> 切换: [English](../../en/models/knockoff.md)

<a id="概览"></a>

## Knockoff 选择要解决什么问题？

当你希望找出影响响应的预测变量，同时限制错误发现时，可以考虑 Knockoff。
错误发现率（FDR）是所选特征中实际无效特征比例的期望值，未选中任何特征时
该比例按零计；它不是“所有所选特征都正确”的概率。

Knockoff 可以理解为匹配的负对照：每个原特征都与一个依赖结构相似的人造
对应变量竞争。较大的正统计量 W 表示原特征比其 knockoff 更重要；比较正、
负统计量，可以确定选择原始列的阈值。

使用 fixed-X，需要可信的匹配设计，以及具有独立同方差正态误差的高斯线性
响应模型。Model-X 则需要可信的特征分布构造，响应关系可以是任意形式。
当前自动构造和阈值并列存在重要[限制](#统计量并列时的限制)，选择一个 API
分支不代表假设自动成立。若主要关注预测，可比较[逐步选择](feature-selection.md)，
并用留出数据评价。

全部函数/选择器签名、参数和结果字段见[特征选择 API 参考](../reference/feature-selection-api.md)。

<a id="先验证目标错误率"></a>

## 在选择前确定目标错误率

应在查看发现结果前确定 q 和统计量。Knockoff+ 即使在最有利的情况下，阈值
计数比也不会低于 1/p。因此 q=0.20 要求合格阈值至少对应五个正向发现；
20 个预测变量使非空选择成为可能，但不保证一定有发现。应根据可接受的
错误率选择 q，而不是用它调出期望的特征数量。

调用过滤函数或拟合选择器前，应检查 `np.isfinite(q) and 0 < q < 1`。
这适用于三个过滤函数、两个选择器类及两种阈值规则。当前验证会漏过 `q=np.nan`，
并可能返回空选择、`threshold=inf`、`estimated_fdr=0.0` 或全为 False 的选择掩码。
此时结果无效，不能解释为在合理目标错误率下未发现特征。请自行拒绝非有限 q，
也不要在查看结果后改换目标错误率。

<a id="centered-pair-example"></a>

<a id="完整的-cpu-示例"></a>

## 使用有效外部配对的完整 CPU 示例

下面的受控模拟在生成 y 之前一起构造 X 与 Xk。QR 正交化的第一列留给截距，
其余列分成两组中心化且相互正交的变量，因此 XᵀX=XkᵀXk=I、XᵀXk=0。
它们满足 S=I 时 fixed-X 的匹配 Gram 条件，截距投影后仍然成立。
这一构造需要 n≥2p+1。

这是特殊的模拟设计，不是修复任意观测 X 的方法。分析自己的数据时，需要为
原设计取得并验证匹配 X/Xk，包括截距或干扰变量投影后的配对条件。不要用
这段 QR 代码替换已经观测到的预测变量，也不要只对自动生成的 Xk 再做中心化；
前者会改变分析问题，后者可能破坏 Gram 约束。提供 `Xk` 可跳过目前存在
[中心化限制](#自动-fixed-x-构造的中心化限制)的自动构造。

这里预先设定 p=20、q=0.20、六个非零系数和 `corr_diff` 统计量。
响应包含截距与标准差为 0.5 的独立高斯噪声，符合 fixed-X 响应模型假设。

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter

# Prespecify the design, target rate, and statistic before seeing results.
n, p = 240, 20
q = 0.20
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
X, Xk = Q[:, 1:p+1], Q[:, p+1:2*p+1]
np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-14)
np.testing.assert_allclose(Xk.mean(axis=0), 0, atol=1e-14)
np.testing.assert_allclose(X.T @ X, np.eye(p), atol=1e-14)
np.testing.assert_allclose(Xk.T @ Xk, np.eye(p), atol=1e-14)
np.testing.assert_allclose(X.T @ Xk, 0, atol=1e-14)

beta = np.zeros(p)
beta[:6] = [8, -7, 6, -5, 4, -3]
y = 2.0 + X @ beta + rng.normal(scale=0.5, size=n)
result = fixed_x_knockoff_filter(
    X, y, Xk=Xk, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy",
)
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
```

## 如何理解选择结果及其不确定性

该种子下，输出为 `Selected columns: [0, 1, 2, 3, 4, 5, 15]`，阈值 `1.019`，
阈值计数比 `0.143`。索引从零开始，对应原始列。由于模拟真值已知，
我们知道第 0–5 列是真信号，第 15 列是一次错误发现。目标 q 不保证每个
入选特征都正确，也不约束每一次样本的实际错误发现比例。

`corr_diff` 使用 W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|。例如，
`result.W[0]` 约为 6.386，支持原始信号；`result.W[11]` 约为 −0.921，
更支持其 knockoff。本次运行没有绝对统计量并列。在所选阈值处，有七个 W
达到正阈值，没有 W 小于等于 −T，因此 knockoff+ 计数比为 (1+0)/7。
`estimated_fdr` 是规则给出的估计值，不是这一次样本中实际无效特征的比例，
也不是逐特征 p 值。
其他样本可能遗漏信号或选中更多噪声，一次模拟结果也不能实证证明 FDR 控制。

输入有效时，空选择也是有效结果：没有阈值满足规则，`threshold` 为无穷大，
实现报告 `estimated_fdr=0.0`，但这不证明所有预测变量都无效。应保留预设 q，
检查假设和检验效能，而不是看到结果后再放宽 q。API 参考提供了一个
[必然返回空选择的小例子](../reference/feature-selection-api.md#runnable-fixed-x-example)。

过滤函数不拟合系数或预测模型。如果还要评价预测，应在每个训练折内重新选择，
再用选中列拟合独立模型，并将评价行单独保留。

<a id="目标函数"></a>

## 阈值如何工作

统计目标是在给定 FDR 水平 `q` 下，通过 knockoff 统计量 `W` 与阈值规则（`knockoff_plus` 或 `knockoff`）选择特征集合。达到该目标仍需满足相关假设，并考虑下文说明的自动构造中心化与并列计数限制。

<a id="估计方程"></a>

### Knockoff+ 规则

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
[中心化配对示例](#centered-pair-example)
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

当 `knockoff_type="model_x"`、`compat_mode="statgpu"` 且 `Xk=None` 时，
整数 `random_state` 为每次构造抽样的局部 Torch 随机数生成器设定种子，
不推进全局 Torch 随机数状态。在输入、设置以及后端、dtype、设备和软件环境
相同的条件下，重复构造可得到相同结果。`KnockoffSelector` 具有相同行为。
这不保证跨后端或跨 GPU 的数值相同，也不保证实际 FDR 控制，且与下文的
Lasso 缓存限制不同。

原生 Torch/CuPy 构造在 `random_state=None` 时每次抽样使用种子 0，因而
多次抽样使用相同的构造噪声；若要可重复且分别设定种子的多次抽样，应指定
整数种子。NumPy 构造在 `None` 时
仍不固定种子。有效外部 `Xk` 可跳过构造，但仍应单独检查所选统计量的可重复性。
提供矩阵不能免除统计假设、阈值及 Lasso 缓存限制。Fixed-X 仍有独立的构造
要求与统计假设。

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

以下可选 GPU 示例复用[中心化配对 CPU 示例](#centered-pair-example)中的
X、y、Xk 和 q，保留相同的外部 fixed-X 配对、统计量和目标错误率。
CuPy/Torch 需要已安装且可用的 CUDA 后端。这一 QR 配对用于 fixed-X，
不是 model-X 的可交换性构造；model-X 需要符合特征分布的构造方式。

```python
from statgpu import fixed_x_knockoff_filter
import cupy as cp

X_gpu, y_gpu, Xk_gpu = cp.asarray(X), cp.asarray(y), cp.asarray(Xk)
res_gpu = fixed_x_knockoff_filter(
    X_gpu, y_gpu, Xk=Xk_gpu, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="cupy",
)

import torch

X_torch, y_torch, Xk_torch = [torch.from_numpy(a).to("cuda") for a in (X, y, Xk)]
res_torch = fixed_x_knockoff_filter(
    X_torch, y_torch, Xk=Xk_torch, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="torch",
)
```

<a id="严格与近似模式的差别"></a>

## 阈值规则与构造假设

本模块不使用 `strict/approx` 推断口径开关。`fixed_x` 与 `model_x` 采用不同的设计或特征分布假设，应根据统计问题选择，不能把两者当作速度或数值精度档位。`modelx_draws`（必须为正整数）影响计算量与蒙特卡洛波动；后端选择均不替代构造假设或阈值规则。

这里 `backend="torch"` 选择 Torch 计算库，不同于估计器要求 CUDA 的 `device="torch"`。原生 fixed-X 与 model-X 自动构造遵循 X 的设备。Torch CPU 输入在 CUDA 可用时仍留在 CPU；CUDA 输入保留其 GPU 编号。model-X 的局部随机数生成器和随机矩阵均使用该设备，详见 [Torch 设备放置](../reference/feature-selection-api.md#torch-device-placement)。

以上设备保持行为适用于构造阶段。原生 Torch 的 `method="lasso_coef_diff"`
调参与拟合仍请求 CUDA，CPU 输入、提供 Xk 及 fixed-X 也有这一限制，且不能
保证使用输入所在的非默认 GPU。若适合分析问题，Torch CPU 统计量可选择
`corr_diff` 或 `ols_coef_diff`；CPU Lasso 应使用 NumPy 输入与 `backend="numpy"`。
详见 [Torch Lasso 的设备选择](../reference/feature-selection-api.md#torch-lasso-device-routing)。

需要 CPU model-X 构造时，可用 Torch CPU 张量并设 `backend="torch"`，或使用 NumPy X/y 与 `backend="numpy"`。应让 X/y/Xk 位于同一目标设备。经过外部验证的 model-X Xk 可跳过构造，但形状和设备一致不能证明特征对交换性或给定 X 后与 y 的条件独立性。这些原生构造规则适用于 `compat_mode="statgpu"` 且 `Xk=None` 的路径，包括 `KnockoffSelector`；fixed-X 的构造与统计假设另有要求。

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

## 进阶参考

<a id="路径"></a>

### 导入路径

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

<a id="外部验证external-validation"></a>

需要复现实现在不同基线下的比较时，贡献者可阅读可选的
[Knockoff 验证指南](../../../dev/references/model-validation.md#knockoff)。

## 参考（References）

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)
