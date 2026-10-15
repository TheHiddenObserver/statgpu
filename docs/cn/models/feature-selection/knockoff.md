# Knockoff 特征选择

> 语言：中文  
> 最后更新：2026-10-13  
> 本页：原理与方法选择  
> 切换：[English](../../../en/models/feature-selection/knockoff.md)

<a id="概览"></a>

## Knockoff 选择要解决什么问题？

当很多预测变量都可能与响应有关，而把噪声误选为信号的代价较高时，可以让每个变量与一个人工负对照竞争。Knockoff 的目标是在相应假设成立时，筛选原始特征并控制错误发现率（FDR）。

若 R 是选中的特征数，V 是其中零假设成立的特征数，则
$$
\operatorname{FDR}=\mathbb{E}\left[\frac{V}{\max(R,1)}\right].
$$
这是重复抽样下错误发现比例的期望，空选择记作零；它不是逐个特征的正确概率，也不保证每次选择都没有错误。某一次样本的错误发现比例可以超过目标 q。

如果首要目标是预测，可以比较[逐步选择](README.md)，并用留出数据衡量效果。Knockoff 筛选本身不拟合预测模型，也不提供回归系数的 p 值。

<a id="协方差与推断"></a>
<a id="常见问题faq"></a>
## 先选择可信的假设，再选择接口

| 方法 | 哪些条件需要可信？ | 从哪里开始 |
|---|---|---|
| Fixed-X | 截距或其他干扰项投影后的有效配对设计、相容的统计量，以及误差独立同方差且服从正态分布的线性响应模型 | [Fixed-X 教程](knockoff-fixed-x.md) |
| Model-X | 原特征与 knockoff 成对可交换，且给定 X 后 knockoff 与 y 条件独立；响应关系可以非线性 | [Model-X 教程](knockoff-model-x.md) |

选型的关键是哪些假设有充分依据。Fixed-X 依赖配对设计和高斯线性响应模型；Model-X 依赖联合特征分布，允许更广泛的响应关系。两者都会计算 knockoff 统计量并应用阈值，但依赖不同的设计或特征分布假设，不能把两者当作速度或数值精度档位。样本较少本身也不是选择 Model-X 的理由。

尤其要注意：对任意观测特征拟合高斯分布，并不能保证满足精确的 Model-X 条件。两个接口都不会替你验证 FDR 控制所需的全部假设。

### 响应模型假设

Fixed-X 使用 y=b1+Xβ+ε，其中 ε∼N(0,σ²I)，零假设特征在该模型中满足 β_j=0。设计与统计量必须在处理截距或其他干扰项后仍相容。Model-X 则依赖特征分布：交换任意一组原特征与对应 knockoff 列后，联合分布应保持不变，而且给定 X 后 Xk 与 y 条件独立。这里的零假设是：给定其余原始特征后，y 与该特征条件独立。两类构造的理论来源见[参考文献](#参考references)，具体要求见各自教程。

<a id="目标函数"></a>

## 阈值如何工作

1. 构造有效配对矩阵 Xk，不能以破坏假设的方式使用响应信息。
2. 每个特征计算一个原特征对照 knockoff 的统计量 W_j；交换配对时其符号应反转，较大的正值有利于原特征。
3. 利用较大的负统计量估计无效原特征赢得竞争的数量，确定阈值 T，并选择 W_j ≥ T 的原始列。

`corr_diff` 比较原特征和 knockoff 与中心化响应的绝对内积：
$$
W_j=|X_j^\mathsf{T}(y-\bar y)|-|Xk_j^\mathsf{T}(y-\bar y)|.
$$
它衡量边际关联，因此可能遗漏纯非线性效应，例如对称分布特征的平方项。`ols_coef_diff` 与 `lasso_coef_diff` 比较系数绝对值。更复杂的统计量可以改变检出能力，但不能修复无效配对或错误的阈值计数。

<a id="先验证目标错误率"></a>

## 在选择前确定目标错误率

在查看发现结果前，先确定 q、假设集合、统计量和 Lasso 计算配置。每次调用筛选函数或拟合选择器之前，都应检查 `np.isfinite(q) and 0 < q < 1`。当前实现对两种阈值规则都会漏过 `q=np.nan`，可能返回 `threshold=inf`、`estimated_fdr=0.0` 的无效空选择，因此必须自行拒绝非有限 q。

Knockoff+ 的阈值计数比最低为 1/p。q=0.20 时，合格阈值至少需要五个 W 为正且达到阈值的特征；p=20 只是让非空选择成为可能，并不保证会选中变量。不要看完结果再增大 q，也不要为了选中特征而加入不相关的预测变量。

<a id="估计方程"></a>

### Knockoff+ 规则

理论阈值从非零绝对统计量的不同取值中选取：
$$
T = \min \left\{ t\in\{|W_j|:|W_j|>0\} : \frac{1+\#\{j:W_j\le -t\}}{\max(1,\#\{j:W_j\ge t\})}\le q \right\}.
$$
每个候选阈值都必须计入满足条件的**所有**特征；没有合格阈值就不选择任何特征。默认的 `fdr_control="knockoff_plus"` 在分子中使用偏移量 1；`fdr_control="knockoff"` 使用偏移量 0，对应不同的修正 FDR 目标，并不是 knockoff+ 的数值近似。

### 统计量并列时的限制

当前实现可能只累计并列组中的一部分就评估阈值，随后却选中整个阈值组。例如 W=[8,8,-8]、q=0.5 时，可能报告阈值 8、estimated_fdr=0.5，而完整阈值比例为 (1+1)/2=1，理论规则应当不选择。不能把这种并列阈值结果解释为名义 knockoff+ FDR 控制；偏移量为 0 的 `knockoff` 也有相同计数限制。增加抽样次数、重拟合系数或更换设备都不能修复它。除了构造有效性，还要检查并列统计量。

<a id="输出outputs"></a>
## 如何理解选择结果及其不确定性

`selected_features` 是从零开始的原始列索引，`W` 是竞争统计量，`threshold` 是筛选界限。`estimated_fdr` 是阈值规则的估计量，不是实际错误发现比例或逐特征 p 值。`q_trajectory` 记录逐秩诊断信息，不能证明整个并列组已正确计数；查看 `metadata` 可了解实际执行的构造和抽样次数。

输入有效时，空选择可能意味着没有阈值合格。此时 `threshold` 为无穷大，实现报告 `estimated_fdr=0.0`，但这不证明所有预测变量都是零假设特征。[Fixed-X 教程](knockoff-fixed-x.md) 用已知真值的模拟解释一个包含错误发现的结果；[Model-X 教程](knockoff-model-x.md) 则独立演示特征分布构造与非线性响应。

选择器提供 `result_`、`selected_features_`、`get_support()` 和 `transform()`，其 `fit` 返回选择器自身。它们不是预测模型：应在每个训练折内重新筛选，再用所选训练列拟合独立估计器，最后在未参与选择的样本上评价。拟合失败后可能仍能访问上次成功的选择状态；应创建新选择器并成功拟合，才能把输出解释为新数据的结果。

<a id="centered-pair-example"></a>
<a id="完整的-cpu-示例"></a>

<a id="构造并检查外部配对"></a>
<a id="生成响应并选择特征"></a>
<a id="预先设定设计与目标错误率"></a>
<a id="使用有效外部配对的完整-cpu-示例"></a>
## 继续阅读完整示例

可以从 [Fixed-X 教程](knockoff-fixed-x.md) 开始实践：构造中心化的受控 QR 配对，验证全部 Gram 条件，生成高斯响应，再应用筛选函数和选择器。n≥2p+1 是这项特殊模拟构造的条件；不能用 QR 设计替换真实测量的预测变量。另一篇 [Model-X 示例](knockoff-model-x.md) 则从已知的独立高斯特征分布出发，演示非线性响应下的选择。

### 自动 fixed-X 构造的中心化限制

自动 fixed-X 构造会中心化 X，但 Xk 的列均值可能不为零。对中心化响应计算统计量时，原始 Gram 相等未必在 P=I−11ᵀ/n 投影后保持：XᵀPX 与 XkᵀPXk 可能不同，即使没有并列或 n>2p，也可能失去通常的零假设配对得分对称性。仅增大 n 无法修复，原始 Gram 有效也不能据此声称名义 FDR 控制。外部配对必须在投影后满足完整条件；仅在生成后中心化 Xk 可能破坏其 Gram 约束。[Fixed-X 构造说明](knockoff-fixed-x.md) 提供具体反例与检查方法。

<a id="严格与近似模式的差别"></a>

## 阈值规则与构造假设

自动 model-X 的高斯矩估计未必符合真实特征分布。增加 `modelx_draws` 会对多次构造的 W 求平均；减少蒙特卡洛波动不等于该平均统计量已有 FDR 定理。传入 Xk 时只计算一次，不会生成新的独立配对。[Model-X 教程](knockoff-model-x.md) 解释这些限制和兼容模式元数据。显式 `knockpy_sampler` 分派尚未实现，可能抛出 `NotImplementedError`，并没有提供通用特征分布采样器。

<a id="cupy"></a>
<a id="torch-cuda"></a>
<a id="cpugpu-示例"></a>
## GPU 用法与示例入口

[Fixed-X 教程](knockoff-fixed-x.md) 在同一外部配对上给出可选 CuPy 与 Torch CUDA 扩展。`backend="torch"` 选择计算库：原生构造跟随 X 的设备，包括 Torch CPU；但原生 Torch Lasso 调参与拟合会请求 CUDA，即使输入在 CPU 或已提供 Xk，也不保证使用输入的非默认 GPU。适合时在 Torch CPU 上选择 `corr_diff` 或 `ols_coef_diff`；CPU Lasso 使用 NumPy 输入和 `backend="numpy"`。完整说明见[设备位置](../../reference/feature-selection-api.md#torch-设备放置)与 [Lasso 设备路由](../../reference/feature-selection-api.md#torch-lasso-的设备选择)。

<a id="reproducibility-of-generated-torch-model-x"></a>

## Torch 自动 model-X 构造的可重复性

整数 `random_state` 使原生 Torch 自动 model-X 构造在相同后端、数据类型、设备和软件环境内可重复，并且不推进 Torch 全局随机数状态。原生 Torch/CuPy 的 `random_state=None` 会为每次构造使用种子 0，而 NumPy 不固定种子。这不承诺跨后端一致或 FDR 控制；详见[构造随机数说明](../../reference/feature-selection-api.md#torch-model-x-构造的种子与可重复性)。

## 重复计算 Lasso 统计量

设置了种子的 Lasso 调用，可能在原地修改 X、y、Xk 或输入内存被复用后继续使用旧统计量；仅创建新选择器并不能解决。分析更改后的数据时，应使用新的 Python 进程。[缓存限制](../../reference/feature-selection-api.md#repeated-lasso-statistic-calls)说明了两种替代方案及其条件：显式提供 float64 NumPy 输入时，复制全部输入并保留旧数组；或者使用 `random_state=None`，放弃依靠固定种子实现可重复结果。

## 如何选择 Lasso 计算配置

`lasso_fast_profile="off"` 保持默认配置；其他配置可能改变折数、候选惩罚、迭代预算和容差，从而改变 W 与筛选结果。应预先确定配置，并在比较时保持不变，参数解释见[完整 API 参考](../../reference/feature-selection-api.md)。

<a id="参数"></a>
<a id="外部验证external-validation"></a>
<a id="导入路径"></a>
<a id="性能与统计边界"></a>
<a id="路径"></a>
## 进阶参考

完整签名、参数、方法、结果字段、数据类型与设备行为、随机数、缓存和回退规则统一见[特征选择 API 参考](../../reference/feature-selection-api.md)。兼容模式所请求的协方差或 S 矩阵方法可能回退，应查看实际执行元数据，而不是只看请求的方法名称。大问题的耗时可能主要来自构造或 Lasso 拟合；选择 GPU 后端不保证每项兼容模式操作都留在 GPU。

需要复现实现比较时，可参考可选的 [Knockoff 验证指南](../../../../dev/references/model-validation.md#knockoff)。

## 参考（References）

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055–2085. [论文](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *JRSS B*, 80(3), 551–577. [论文](https://doi.org/10.1111/rssb.12265)
