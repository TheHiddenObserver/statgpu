# ANOVA API 参考

> 语言：中文\
> 最后更新：2026-10-12\
> 本页：API 参考\
> 切换：[English](../../en/reference/anova-api.md)

选择方法或学习如何解释结果，请先阅读带完整示例的 [ANOVA 指南](../models/anova.md)。
本页用于查阅全部公开函数的签名、参数、返回字段、输入要求和数值限制。

<a id="public-api"></a>

## 公开 API

以下七个函数均可从 `statgpu.anova` 或 `statgpu` 导入。
五个结果数据类需要从 `statgpu.anova` 导入，不能从顶层 `statgpu` 导入。

| 函数 | 用途 | 返回类型 |
|---|---|---|
| [`f_oneway`](#f-oneway) | 等方差单因素 ANOVA | [`AnovaResult`](#anova-result) |
| [`f_welch`](#f-welch) | 不等方差单因素 ANOVA | [`AnovaResult`](#anova-result) |
| [`f_twoway`](#f-twoway) | 平衡双因素 ANOVA，可选交互项 | [`TwoWayAnovaResult`](#two-way-anova-result) |
| [`tukey_hsd`](#tukey-hsd) | 使用合并方差的全部两两比较 | [`TukeyResult`](#tukey-result) |
| [`bonferroni`](#bonferroni) | 全部两两 Welch 检验，对判定阈值与置信区间作 Bonferroni 校正 | [`PosthocResult`](#posthoc-result) |
| [`cohens_f`](#cohens-f) | 根据分组观测值计算单因素 Cohen's f | `float` |
| [`partial_eta_squared`](#partial-eta-squared) | 根据给定平方和计算偏 eta 平方 | `float` |

`TukeyResult.comparisons` 和 `PosthocResult.comparisons` 的元素为
[`PairwiseComparison`](#pairwise-comparison)，每个对象保存一对组的比较结果。
上表中的七个函数包含检验、两两比较和效应量计算，并不代表七种 ANOVA 设计。

<a id="common-inputs"></a>

## 通用输入与参数规则

五个带 `*groups` 的函数都要求每组分别作为一个参数传入。
如果各组已经放在列表或元组 `groups` 中，应调用 `function(*groups)`，
用 `*` 展开各组，而不是将整个列表作为单组传入。
每组数据先转换为指定 dtype，再通过 `ravel()` 展平；各组长度可以不同。
多维数组也只会被展平成一组观测，不会自动执行按列检验、批处理、多变量分析或重复测量分析。

| 参数 | 默认值 | 含义与可用值 |
|---|---|---|
| `*groups` | 必填 | 至少两组相互独立的样本，观测值为有限实数。最小样本量见下表；组编号按参数传入顺序从零开始。 |
| `backend` | `"auto"` | `"auto"`、`"numpy"`、`"cupy"` 或 `"torch"`。兼容旧别名 `"cpu"` → `"numpy"`、`"cuda"`/`"gpu"` → `"cupy"`；名称不区分大小写，忽略首尾空白。详见[选择与设备](#backend-dtype-device)。 |
| `dtype` | `None` | 计算时使用的数据类型。`None` 使用所选库的 `float64`，即使输入原本为 float32。建议使用对应后端的 `float32` 或 `float64` dtype 对象。 |
| `alpha` | `0.05` | 仅用于 `tukey_hsd` 和 `bonferroni`：必须是严格位于 `(0, 1)` 内的有限实数，表示整个比较族的显著性水平；同时置信水平为 `1 - alpha`。 |

在 `*groups` 函数中，`backend`、`dtype` 和 `alpha` 只能通过关键字传入。
`f_twoway` 的参数可以按位置或关键字传入；`partial_eta_squared` 没有后端或 dtype 参数。

| 函数 | 最小观测数 |
|---|---|
| `f_oneway`、`cohens_f` | 每组至少一个观测值，总观测数 `N` 必须大于组数 `k`。满足 `N > k` 时允许某些组只有一个观测值。 |
| `f_welch`、`tukey_hsd`、`bonferroni` | 每组至少两个观测值。 |
| `f_twoway` | 每个因素至少两个水平；各单元格非空且样本量相等；残差自由度必须为正。详见[单元格布局](#f-twoway)。 |

请传入实数观测，并使用浮点 dtype 计算。不同函数对类型的检查并不完全一致：
复数输入未必会在转换前被拒绝，显式指定的 dtype 也未必会被检查是否为浮点类型。
类型转换可能丢失信息，也可能由数组库直接报错。不要用整数或复数 dtype
替代本文所述的浮点计算。转换失败时，异常类型由底层数组库决定。

<a id="backend-dtype-device"></a>

## 后端、dtype 与设备行为

- 对 `f_oneway`、`f_welch`、`tukey_hsd`、`bonferroni` 和 `cohens_f`，
  `backend="auto"` 根据分组参数中首先遇到的 Torch 张量或 CuPy 数组选择后端。
  只有 Python/NumPy 输入时选择 NumPy，不会因为计算机装有 GPU 就自动使用 GPU。
- 对 `f_twoway`，`backend="auto"` 不检查嵌套单元格，直接选择 NumPy。
  使用 CuPy 或 Torch 单元格数组时，请显式指定相应后端。
- `backend="torch"` 选择 Torch 库，不代表选择 CUDA 设备。可以使用 Torch CPU 张量。
  若希望在 GPU 上计算各组统计量，请提供 Torch CUDA 张量；普通列表或 NumPy 数组遵循
  Torch 创建数组时的默认设备，通常为 CPU。ANOVA 函数没有 `device` 参数，也不采用
  估计器 `device="torch"` 的设备选择约定。
- 请让所有组或单元格使用同一数组库，并位于同一设备上。`f_oneway` 和 `f_twoway`
  转换原生张量时会保留其设备，不会自动将不同设备上的张量搬到一起。
  Welch 和事后比较会参照原生输入的设备准备数据，但仍不应依赖混用数组库或设备的输入方式。
- 显式指定后端不会静默切换到另一个数组库。缺少 CuPy/PyTorch 时抛出 `ImportError`；
  CUDA 运行环境不可用或设备组合不兼容时，可能抛出底层库的运行时异常。
  非法后端名称抛出 `ValueError`。
- `dtype=None` 表示使用 float64 计算，不是保留输入 dtype。
  请分别使用 `np.float32`/`np.float64`、`cp.float32`/`cp.float64` 或
  `torch.float32`/`torch.float64`。不要假设所有 Torch 函数都接受 NumPy dtype 对象。
  使用 float32 可能改变平方和、P 值以及阈值附近的拒绝结论。

各组均值、方差与平方和使用所选数组后端计算。最终的数值结果为 Python 标量，
获取这些标量时可能触发 GPU 同步。分布计算不会单纯为了求一个概率而把完整观测向量
搬到 CPU，但部分标量步骤仍可能在 CPU 上完成：

| 计算 | 分布与标量计算位置 |
|---|---|
| `f_oneway`、`f_twoway` | F 分布计算使用所选分布后端。Torch 的此项标量计算采用默认设备：有 CUDA 时使用 CUDA，否则使用 CPU，不一定与输入张量的设备相同。 |
| `f_welch` | F 分布计算使用所选后端；Torch 使用准备后的输入设备。 |
| `tukey_hsd` | 学生化极差分布的右尾概率与分位数由 CPU 上的 `scipy.stats.studentized_range` 计算；两两比较的标量运算也在 CPU 上进行。 |
| `bonferroni` | 两两 Welch 标量运算和 NumPy/CPU t 分布计算在 CPU 上进行。 |
| `cohens_f` | 调用 `f_oneway`，包括其 P 值计算，再将 eta 平方转换为标量效应量。 |
| `partial_eta_squared` | Python/NumPy 标量计算，不使用分组数组后端。 |

F 分布和 t 分布计算可能使用查找表，也可能回退到其他数值算法；构建查找表时可能调用 CPU 上的 SciPy。
用概率相减计算极端尾部时可能损失精度，极小概率可能舍入为零，临界值也可能不准确或为无穷大。
详见分布指南的[尾部精度限制](../guides/distribution-api.md#尾部精度的限制)与
[后端数值限制](../guides/distribution-api.md#查找表与各后端的数值限制)。ANOVA 不提供 `use_lut` 参数，
也没有严格模式、近似模式或尾部精度控制选项。各后端采用相同的统计定义，
但不保证数值完全一致，也不保证全程在 GPU 上计算。

<a id="finite-and-degenerate-inputs"></a>

## 非有限输入与退化数据

所有基于观测数据的函数（包括 `cohens_f`）都会在数组转换后检查 `NaN`、正无穷和负无穷，
并以 `ValueError` 拒绝这些值。它们不会自动删除缺失观测，也没有 `nan_policy` 参数。
通过这项检查不代表独立性、正态性等统计假设成立。即使输入都是有限值，
中间运算仍可能溢出；若输出出现 `NaN` 或无穷值，应先核查原因，再解释结果。

| 情形 | 结果或异常 |
|---|---|
| `f_oneway`：全部观测相同 | `statistic`、`pvalue`、`eta_squared` 均为 `NaN`；仍返回自由度。 |
| `f_oneway`：各组内部恒定，但组均值不同 | `statistic=inf`、`pvalue=0.0`、`eta_squared=1.0`。 |
| `cohens_f`：上述两种情形 | 全部观测相同时返回 `NaN`；各组恒定但均值不同时返回 `inf`。 |
| `f_welch`：只有部分组的样本方差为零 | 抛出 `ValueError`，因为加权 Welch 计算未定义。 |
| `f_welch`：全部组的样本方差为零 | 组均值被视为相等时 `statistic=pvalue=NaN`，否则为 `statistic=inf`、`pvalue=0.0`。两种情形中 `df_within=N-k` 为整数，`eta_squared=NaN`。此处相等判定为 `max(abs(mean_i - mean_0)) <= 1e-12 * max(1, abs(mean_0))`。 |
| `f_twoway`：残差均方为零 | 每个被检验效应的均方为零时返回 `(F, p)=(NaN, NaN)`，否则返回 `(inf, 0.0)`。总平方和为零时，两项主效应 eta 平方均为 `NaN`；即使 `interaction=True`，`interaction_eta_squared` 也是 `None`。 |
| `tukey_hsd`：合并 MSE 为零 | 每一对的区间宽度为零，端点等于均值差。均值相等时 `pvalue=1.0`、`reject=False`；均值不同时 `pvalue=0.0`、`reject=True`。 |
| `bonferroni`：某一对的两个样本方差均为零 | 该比较采用相同的零宽度区间与 P 值规则。只含一个零方差组的比较可以执行。 |
| `partial_eta_squared(0, 0)` | 返回 `NaN`，因为比值未定义。 |

<a id="f-oneway"></a>

## `f_oneway`

```text
f_oneway(*groups: Any, backend: str = "auto", dtype: Any = None) -> AnovaResult
```

在观测相互独立、误差服从正态分布且总体方差相同的假设下，检验 `k` 个总体均值是否相等。
函数不会自行检查这些假设。`*groups`、`backend` 和 `dtype` 遵循[通用参数规则](#common-inputs)。

设总样本量为 `N`，各组样本量为 $n_i$，组均值为 $\bar y_i$，
加权总均值为 $\bar y=\sum_i n_i\bar y_i/N$，则

$$
SSB=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SSW=\sum_i\sum_j(y_{ij}-\bar y_i)^2,
\qquad F=\frac{SSB/(k-1)}{SSW/(N-k)}.
$$

返回 [`AnovaResult`](#anova-result)，包含 F 分布右尾 P 值和
`eta_squared = SSB / (SSB + SSW)`。少于两组、任一组为空、`N <= k` 或观测非有限时，
抛出 `ValueError`。`SSW=0` 时参见[退化情形](#finite-and-degenerate-inputs)。

<a id="f-welch"></a>

## `f_welch`

```text
f_welch(*groups: Any, backend: str = "auto", dtype: Any = None) -> AnovaResult
```

检验独立组的总体均值是否相等，不要求等方差。使用分母为 $n_i-1$ 的样本方差 $s_i^2$，
并采用 Welch 校正。每组至少需要两个观测值，其他参数遵循[通用规则](#common-inputs)。

令 $w_i=n_i/s_i^2$、$W=\sum_i w_i$、$\bar y_w=\sum_i w_i\bar y_i/W$，
以及 $A=\sum_i(1-w_i/W)^2/(n_i-1)$，则

$$
F_W=\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
{1+2(k-2)A/(k^2-1)},\qquad
df_1=k-1,\qquad df_2=\frac{k^2-1}{3A}.
$$

返回 [`AnovaResult`](#anova-result)。通常 `df_within` 是浮点型
Welch–Satterthwaite 自由度 `df_2`；计算得到的调整量 `A <= 0` 时为 `inf`。
**`eta_squared` 始终为 `NaN`**；经典单因素 ANOVA 的效应量不能直接当作 Welch 效应量。
`cohens_f` 也不提供 Welch 专用效应量。

少于两组、任一组不足两个观测值、存在非有限数据，或零方差组与正方差组混合时，
抛出 `ValueError`。全部组方差为零的行为见[上表](#finite-and-degenerate-inputs)。
没有截尾比例或 Welch 校正开关。与其他实现比较时，可对相同样本调用
[statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)，
并设置 `use_var="unequal"`、`welch_correction=True`、`trim_frac=0`。

<a id="f-twoway"></a>

## `f_twoway`

```text
f_twoway(
    data: Any,
    interaction: bool = True,
    backend: str = "auto",
    dtype: Any = None,
) -> TwoWayAnovaResult
```

用于平衡双因素设计，检验因素 A、因素 B 的主效应，并可选择是否检验交互作用。
推断要求观测独立、各单元格的误差服从正态分布，且总体方差相同。

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `data` | 必填 | 嵌套的单元格数据：A 有 `a` 个水平，B 有 `b` 个水平，每格有 `n` 个观测，逻辑形状为 `(a, b, n)`。`data[i][j]` 对应 A 第 `i` 个水平与 B 第 `j` 个水平的组合。每行必须是 Python 列表或元组；每格转换后展平，长度须为同一个正整数 `n`。 |
| `interaction` | `True` | 布尔值，是否纳入并检验交互项。为 `False` 时使用加性模型，将交互变异及其自由度并入残差。 |
| `backend` | `"auto"` | 可用值见通用参数表；**auto 不检查单元格，直接选择 NumPy**。 |
| `dtype` | `None` | 计算时的数据类型；默认为 float64，取值规则见通用参数表。 |

直接传入形状 `(a, b, n)` 的 NumPy/CuPy/Torch 数组时，不会按嵌套单元格布局解释；
每个数组行会被当作一个单元格，随后因 B 的水平不足而报错。
请按下面的方式用 Python 列表组织各行和单元格：

```python
from statgpu.anova import f_twoway

cells = [
    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]],
]
result = f_twoway(cells, interaction=True, backend="numpy")
```

两个因素都至少需要两个水平。平衡设计的分解为
$SS_{total}=SS_A+SS_B+SS_{AB}+SS_{within\ cells}$；
每个被检验效应使用 $F=(SS_{effect}/df_{effect})/(SS_{error}/df_{error})$。

| 模型 | 残差平方和（`ss_within`） | 残差自由度（`df_within`） |
|---|---|---|
| `interaction=True` | $SS_{within\ cells}$ | $ab(n-1)$；要求 `n >= 2` |
| `interaction=False` | $SS_{within\ cells}+SS_{AB}$ | $abn-a-b+1$；该值为正时允许 `n=1` |

省略交互项会改变两个主效应 F 检验的分母，不只是隐藏交互项字段。
每个单元格只有一个观测时，加性模型无法区分交互变异与单元格内部误差，
应根据研究问题判断加性模型是否合理。

返回 [`TwoWayAnovaResult`](#two-way-anova-result)。效应量字段是普通的
$SS_{effect}/SS_{total}$，**不是偏 eta 平方**。
当 `interaction=False` 时，四个交互项字段全部为 `None`。

数据或单元格为空、行的列数不一致、任一因素不足两个水平、单元格样本量不等、
观测非有限或残差自由度不为正时，抛出 `ValueError`。
不支持不平衡设计，也没有 Type I、II 或 III 平方和选项。

<a id="tukey-hsd"></a>

## `tukey_hsd`

```text
tukey_hsd(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> TukeyResult
```

在误差独立、服从正态分布且各组总体方差相同的假设下，计算全部 $m=k(k-1)/2$ 项两两均值比较。
每组至少两个观测值；`alpha`、`backend` 和 `dtype` 遵循[通用规则](#common-inputs)。
样本量不等时使用 Tukey–Kramer 标准误。函数没有不等方差或 Games–Howell 选项。

设合并均方误差为 $MSE=SSW/(N-k)$，均值差为 $d_{ij}=\bar y_i-\bar y_j$，则

$$
SE_{q,ij}=\sqrt{\frac{MSE}{2}\left(\frac1{n_i}+\frac1{n_j}\right)},
\qquad q_{ij}=\frac{|d_{ij}|}{SE_{q,ij}}.
$$

学生化极差分布使用组数 `k` 与残差自由度 `N-k`。
每个 [`PairwiseComparison`](#pairwise-comparison) 包含：

- `pvalue`：学生化极差分布的右尾概率，已针对全部两两比较进行校正；
- `reject`：`pvalue < alpha`；
- `ci_lower`、`ci_upper`：$d_{ij}\pm q_{1-\alpha;k,N-k}SE_{q,ij}$，即同时置信区间。

返回 [`TukeyResult`](#tukey-result)。少于两组、`alpha` 非法、任一组样本量不足或观测非有限时，
抛出 `ValueError`。要求 `scipy.stats.studentized_range` 可用，否则抛出 `ImportError`。
MSE 为零时见[退化情形](#finite-and-degenerate-inputs)。返回的 P 值限制在 `[0, 1]` 内；
若分布计算得到非有限值，当前实现会在 `q=0` 时替换为 `1.0`，其他情况替换为 `0.0`。
因此，P 值有限本身并不能证明数值计算准确。

等样本量与不等样本量的合并方差比较原理可参阅
[SciPy 的 Tukey HSD 说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html)。
statgpu 的结果类型和选项以本页为准。

<a id="bonferroni"></a>

## `bonferroni`

```text
bonferroni(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> PosthocResult
```

计算全部 $m=k(k-1)/2$ 项独立样本双侧 Welch t 检验，并对判定阈值与置信区间进行 Bonferroni 校正。
不要求等方差；每组至少两个观测值，其他参数遵循[通用规则](#common-inputs)。

对每一对组，样本方差的分母为 `n_i - 1`，并使用

$$
SE_{ij}=\sqrt{s_i^2/n_i+s_j^2/n_j},\qquad
t_{ij}=\frac{\bar y_i-\bar y_j}{SE_{ij}},\qquad
\nu_{ij}=\frac{(s_i^2/n_i+s_j^2/n_j)^2}
{(s_i^2/n_i)^2/(n_i-1)+(s_j^2/n_j)^2/(n_j-1)}.
$$

返回 [`PosthocResult`](#posthoc-result)。每项比较中：

- **`pvalue` 是原始双侧 Welch P 值**，即 $2P(T_{\nu_{ij}}>|t_{ij}|)$，上限为 1；
  它不是 Bonferroni 校正后的 P 值。
- `reject` 使用 `pvalue < alpha / m`，不是 `pvalue < alpha`。
- 区间为 $d_{ij}\pm t_{1-\alpha/(2m),\nu_{ij}}SE_{ij}$。
  它使用校正后的临界值；同时覆盖率的保证仍以各项 Welch 近似适用为前提。

需要报告校正后的 P 值时，可另行计算
`min(1.0, comparison.pvalue * result.n_comparisons)`，返回对象中保存的仍是原始 P 值。
结果不保存各对的 t 统计量、标准误或自由度。

少于两组、`alpha` 非法、任一组样本量不足或数据非有限时，抛出 `ValueError`。
两个方差均为零的比较见[上表](#finite-and-degenerate-inputs)；
只有一个方差为零时，不会像 `f_welch` 的混合零方差情形那样报错。

<a id="cohens-f"></a>

## `cohens_f`

```text
cohens_f(*groups: Any, backend: str = "auto", dtype: Any = None) -> float
```

使用给定分组的 `f_oneway` eta 平方，计算合并方差单因素效应量
$f=\sqrt{\eta^2/(1-\eta^2)}$。参数、最小样本量、异常、计算 dtype 与后端行为均与
[`f_oneway`](#f-oneway) 相同，包括其中的 F 分布计算。
输入是分组数据，不是 ANOVA 结果对象或预先计算的 eta 平方；
该函数也不计算 Welch 专用效应量或偏效应量。

结果为有限值时，是非负的 Python `float`；eta 平方大于或等于 1 时返回 `inf`，
eta 平方未定义时返回 `NaN`。常见的 0.10/0.25/0.40 效应大小划分只是经验约定，
需要结合研究背景解释，不能用作显著性检验阈值。

<a id="partial-eta-squared"></a>

## `partial_eta_squared`

```text
partial_eta_squared(ss_effect: float, ss_error: float) -> float
```

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `ss_effect` | 必填 | 有限非负实数标量：目标效应的平方和。 |
| `ss_error` | 必填 | 有限非负实数标量：同一拟合模型的残差或误差平方和。 |

两项输入均通过 `float()` 转换。不能转换的输入抛出 `TypeError`；
转换后非有限或为负数时抛出 `ValueError`。
返回 Python 浮点数 $\eta_p^2=SS_{effect}/(SS_{effect}+SS_{error})$，通常位于 `[0, 1]`；
两项平方和均为零时返回 `NaN`。
`ss_effect=0` 且误差为正时返回 0；效应平方和为正且 `ss_error=0` 时返回 1。

此函数只计算比值，没有后端、dtype、模型拟合或自由度参数。
调用者应确保两项平方和对应同一个模型及所要衡量的效应。
单因素设计中，该比值等于普通 eta 平方；多因素设计中，一般不等于
`TwoWayAnovaResult` 返回的普通 eta 平方字段。

<a id="result-objects"></a>

## 结果对象与可调用接口

以下五种结果类型都是 Python 数据类。构造参数的顺序见各自签名，所有字段均为必填，
没有默认值；直接构造对象也不会检查其中的统计内容是否合理。
无论使用哪种数组后端，返回结果中的数值字段均为 Python `float`/`int`，
拒绝判定为 `bool`，两两比较结果保存在 Python 列表中。

**五个数据类均没有 `summary()` 或其他公开统计方法。**
可以直接读取字段，使用 `print(result)` 查看对象内容，
或使用 Python 的 `dataclasses.asdict(result)` 转换为字典。
它们不是 SciPy 的命名元组结果，不支持元组解包，也没有 `confidence_interval()` 方法；
事后比较区间已经保存在字段中。

<a id="anova-result"></a>

### `AnovaResult`

```text
AnovaResult(
    statistic: float,
    pvalue: float,
    df_between: int,
    df_within: float,
    eta_squared: float,
) -> None
```

签名末尾的 `-> None` 是 `__init__` 方法的返回注解；调用 `AnovaResult(...)` 仍会得到一个结果对象。
`f_oneway` 和 `f_welch` 返回此类型。

| 字段 | 运行时类型 | 含义 |
|---|---|---|
| `statistic` | `float` | 经典或 Welch F 统计量；退化情形下可能为 `NaN` 或 `inf`。 |
| `pvalue` | `float` | F 分布右尾概率，可能为 `NaN`。结果对象不保存 alpha 或拒绝判定。 |
| `df_between` | `int` | 分子自由度 `k - 1`。 |
| `df_within` | `int` 或 `float` | `f_oneway` 返回整数 `N-k`；Welch 通常返回小数形式的分母自由度。所有组方差都为零时的例外见上文。该字段的数据类注解为 `float`。 |
| `eta_squared` | `float` | `f_oneway` 中为 `SSB / (SSB + SSW)`；`f_welch` 中始终为 `NaN`。 |

<a id="two-way-anova-result"></a>

### `TwoWayAnovaResult`

```text
TwoWayAnovaResult(
    factor_a_statistic: float,
    factor_a_pvalue: float,
    factor_a_df: int,
    factor_a_eta_squared: float,
    factor_b_statistic: float,
    factor_b_pvalue: float,
    factor_b_df: int,
    factor_b_eta_squared: float,
    interaction_statistic: Optional[float],
    interaction_pvalue: Optional[float],
    interaction_df: Optional[int],
    interaction_eta_squared: Optional[float],
    df_within: int,
    ss_within: float,
) -> None
```

构造 `TwoWayAnovaResult` 对象；`f_twoway` 返回此类型。

| 字段 | 类型 | 含义 |
|---|---|---|
| `factor_a_statistic` | `float` | 因素 A 的 F 统计量，分母使用所选模型的残差均方。 |
| `factor_a_pvalue` | `float` | 因素 A 的 F 分布右尾 P 值。 |
| `factor_a_df` | `int` | `a - 1`。 |
| `factor_a_eta_squared` | `float` | `SS_A / SS_total`；总平方和为零时为 `NaN`。 |
| `factor_b_statistic` | `float` | 因素 B 的 F 统计量，使用同一个残差均方。 |
| `factor_b_pvalue` | `float` | 因素 B 的 F 分布右尾 P 值。 |
| `factor_b_df` | `int` | `b - 1`。 |
| `factor_b_eta_squared` | `float` | `SS_B / SS_total`；总平方和为零时为 `NaN`。 |
| `interaction_statistic` | `float` 或 `None` | 交互项 F 统计量；省略交互项时为 `None`。 |
| `interaction_pvalue` | `float` 或 `None` | 交互项 F 分布右尾 P 值；省略时为 `None`。 |
| `interaction_df` | `int` 或 `None` | `(a - 1) * (b - 1)`；省略时为 `None`。 |
| `interaction_eta_squared` | `float` 或 `None` | `SS_AB / SS_total`；省略交互项**或总平方和为零**时为 `None`。 |
| `df_within` | `int` | 残差自由度：含交互项为 `ab(n-1)`，不含交互项为 `abn-a-b+1`。 |
| `ss_within` | `float` | 所选模型的残差平方和；`interaction=False` 时包含交互平方和。 |

各效应平方和、残差均方和总平方和没有单独的结果字段。
这些 P 值分别检验各项效应，不会自动对 A、B 和交互项三个检验进行多重比较校正。

<a id="tukey-result"></a>

### `TukeyResult`

```text
TukeyResult(
    comparisons: List[PairwiseComparison],
    alpha: float,
    n_groups: int,
    df_within: int,
    mse: float,
) -> None
```

构造 `TukeyResult` 对象；`tukey_hsd` 返回此类型。

| 字段 | 类型 | 含义 |
|---|---|---|
| `comparisons` | `list[PairwiseComparison]` | 全部 `k(k-1)/2` 个无序组对，满足 `i < j`，顺序为 `(0,1), (0,2), …, (1,2), …`。 |
| `alpha` | `float` | 传给函数的整个比较族的显著性水平。 |
| `n_groups` | `int` | 组数 `k`。 |
| `df_within` | `int` | 合并残差自由度 `N-k`。 |
| `mse` | `float` | 合并组内均方 `SSW / (N-k)`。 |

<a id="posthoc-result"></a>

### `PosthocResult`

```text
PosthocResult(
    comparisons: List[PairwiseComparison],
    alpha: float,
    n_comparisons: int,
) -> None
```

构造 `PosthocResult` 对象；`bonferroni` 返回此类型。

| 字段 | 类型 | 含义 |
|---|---|---|
| `comparisons` | `list[PairwiseComparison]` | 全部无序组对，顺序与 `TukeyResult` 相同。 |
| `alpha` | `float` | 整个比较族的显著性水平，**不是**单项比较的阈值。 |
| `n_comparisons` | `int` | `m = k(k-1)/2`；单项比较阈值为 `alpha / m`。 |

<a id="pairwise-comparison"></a>

### `PairwiseComparison`

```text
PairwiseComparison(
    group_i: int,
    group_j: int,
    mean_diff: float,
    pvalue: float,
    ci_lower: float,
    ci_upper: float,
    reject: bool,
) -> None
```

构造 `PairwiseComparison` 对象；两种事后比较结果列表均以此类型为元素。

| 字段 | 类型 | 含义 |
|---|---|---|
| `group_i` | `int` | 第一组从零开始的输入索引。 |
| `group_j` | `int` | 第二组从零开始的输入索引，大于 `group_i`。 |
| `mean_diff` | `float` | 第 `i` 组均值减第 `j` 组均值。 |
| `pvalue` | `float` | Tukey：已校正的学生化极差分布 P 值；Bonferroni：**原始**双侧 Welch P 值。 |
| `ci_lower` | `float` | `i - j` 均值差的区间下限，按相应方法进行校正。 |
| `ci_upper` | `float` | 同一均值差的区间上限。 |
| `reject` | `bool` | Tukey：`pvalue < alpha`；Bonferroni：`pvalue < alpha / m`。恰好等于阈值时不拒绝。 |

<a id="limitations"></a>

## 范围与不支持的请求

- 本模块没有公式或数据框标签接口、权重、`axis`、`keepdims`、`nan_policy`，
  也没有估计器式的 `fit`/`predict` 接口。
- 双因素 ANOVA 要求平衡单元格，没有 Type I/II/III 选项。
  这些函数不实现重复测量 ANOVA、ANCOVA、MANOVA、随机或混合效应 ANOVA，以及更高阶析因设计。
- Tukey 使用合并方差；Bonferroni 使用两两 Welch 检验，保留原始 P 值，对判定阈值与区间作校正。
  两者都不接收已拟合双因素模型的结果，也不计算基于该模型的边际均值或对比。
- 模块不会替用户检验独立性、正态性或等方差假设。省略交互项、P 值很小或结果有限，
  都不能证明模型假设成立；总体检验本身也不能指出哪些组对存在差异。
- 没有单独的严格模式、近似模式或数值尾部策略；仍应注意
  [后端与分布计算限制](#backend-dtype-device)。

<a id="references"></a>

## 参考文献

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- [statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)：Welch 比较时的对齐设置。
- [SciPy `tukey_hsd`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html)：Tukey/Tukey–Kramer 的假设与解释；其 API 与 statgpu 不同。
- [statgpu 分布指南](../guides/distribution-api.md)：分布计算位置与数值限制。
