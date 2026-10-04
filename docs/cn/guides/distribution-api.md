# Distribution API 使用指南

> 语言：中文
> 最后更新：2026-10-04
> 页面定位：使用指南
> 切换：[English](../../en/guides/distribution-api.md)

## 概率分布能回答什么问题？

概率分布把模型假设转化为概率、分位数、密度或模拟观测值。已知参考分布及其参数时，可以用本 API 计算正态测量误差的概率、Student t 检验统计量的尾概率，或 Poisson 事件计数的分位数。它不会根据数据拟合分布，也不会检查所选分布是否合适。分布形态未知时，可以考虑[核密度估计](../models/nonparametric.md)；处理回归推断时，应先查看拟合模型提供的推断结果，不能忽略拟合过程而自行指定参考分布。

新代码建议从 `statgpu.inference` 导入对象式 API。以下示例各自独立，安装 statgpu、NumPy 和 SciPy 后即可在 CPU 上运行，无需 GPU。

## 从概率和分位数开始

对标准正态变量，`cdf(1)` 表示观测值不超过 1 的概率；`ppf(0.975)` 则寻找一个界值，使其左侧包含 97.5% 的概率。这两个操作从相反方向回答同一个问题。

```python
# Example: normal_probabilities
import numpy as np
from statgpu.inference import norm

x = np.array([-1.0, 0.0, 1.0])
cdf = norm.cdf(x, backend="numpy")
sf = norm.sf(x, backend="numpy")
density = norm.pdf(x, backend="numpy")
quantiles = norm.ppf(np.array([0.025, 0.5, 0.975]), backend="numpy")
upper_cutoff = norm.isf(0.025, backend="numpy")
central_probability = norm.cdf(1.96, backend="numpy") - norm.cdf(-1.96, backend="numpy")
print("CDF:", np.round(cdf, 6))
print("SF:", np.round(sf, 6))
print("Density:", np.round(density, 6))
print("Quantiles:", np.round(quantiles, 6))
print("Upper cutoff:", round(float(upper_cutoff), 6))
print("Central probability:", round(float(central_probability), 6))
```

四舍五入后的预期输出：

```text
CDF: [0.158655 0.5      0.841345]
SF: [0.841345 0.5      0.158655]
Density: [0.241971 0.398942 0.241971]
Quantiles: [-1.959964  0.        1.959964]
Upper cutoff: 1.959964
Central probability: 0.950004
```

该分布约有 95% 的概率落在 -1.96 到 1.96 之间。零点的密度约为 0.399，但这**不是**观测值恰好等于零的概率：连续分布在单个点上的概率为零。密度可以大于 1，区间上的密度面积才是概率。

### 按问题选择方法

| 方法 | 含义 | 输入与解释 |
|---|---|---|
| `cdf(x, ...)` | $P(X\le x)$ | 输入观测界值，返回累积概率。 |
| `sf(x, ...)` | $P(X>x)$ | 严格右尾概率。整数计数满足 $P(X\ge k)=\operatorname{sf}(k-1)$。 |
| `ppf(q, ...)` | 下侧分位数 | `q` 为 `[0, 1]` 内的概率；对于离散分布及 `0 < q < 1`，返回支持集中使 CDF 不小于 `q` 的最小整数。 |
| `isf(q, ...)` | 上侧分位数 | 输入右尾概率，对应 `ppf(1-q)`；离散分布遵循离散分位数约定。 |
| `pdf(x, ...)` | 连续概率密度 | 连续分布提供此方法；结果不是单点概率。 |
| `pmf(k, ...)` | $P(X=k)$ | Poisson 和二项计数提供此方法；非整数计数的概率质量为零。 |
| `rvs(..., size=...)` | 随机观测值 | `size` 可为整数、维度元组或 `None`；`None` 返回标量形状的结果。样本是观测值，不是概率。 |

分位数端点可能是无穷大。Poisson 和二项分布的 `ppf(0)` 采用“支持集下界减一”的约定，`loc=0` 时为 `-1`；它不是可能抽到的观测值。验证分位数函数与 CDF 的对应关系时，优先选用严格介于 0 和 1 之间的概率。无效参数常会产生 `nan`，不支持的参数则可能直接报错，因此应主动检查输入，不能假设所有错误都有相同的处理方式。

## 计数：概率质量、累积概率与上界

```python
# Example: count_probabilities
import numpy as np
from statgpu.inference import poisson, binom

k = np.array([0, 1, 2, 3])
mass = poisson.pmf(k, mu=3.0, backend="numpy")
cumulative = poisson.cdf(k, mu=3.0, backend="numpy")
probability_at_least_three = poisson.sf(2, mu=3.0, backend="numpy")
cutoff_95 = poisson.ppf(0.95, mu=3.0, backend="numpy")
binomial_quantiles = binom.ppf(np.array([0.1, 0.5, 0.9]), n=20, p=0.2, backend="numpy")
print("Mass:", np.round(mass, 6))
print("CDF:", np.round(cumulative, 6))
print("P(X >= 3):", round(float(probability_at_least_three), 6))
print("95% count cutoff:", int(cutoff_95))
print("Binomial quantiles:", binomial_quantiles.astype(int))
```

预期概率质量为 `[0.049787, 0.149361, 0.224042, 0.224042]`，CDF 为 `[0.049787, 0.199148, 0.423190, 0.647232]`，至少出现三次的概率为 `0.576810`，95% 界值为 `6`，二项分位数为 `[2, 4, 6]`。对均值为 3 的 Poisson 分布，至少 95% 的概率落在 6 及以下，而不超过 5 的概率还不到 95%。

在相应假设成立时，无上限的非负事件计数可用 Poisson 分布；固定次数的独立试验、每次成功概率相同时，成功次数可用二项分布。

调用 `poisson.cdf(k, mu=...)` 时，应把界值放在第一个位置，也可以写成 `x=k`。模块级代理的 CDF 形参名为 `x`，所以 `cdf(k=..., mu=...)` 不是有效的代理调用。使用代理时，`mu`、`df`、`n`、`p` 等形状参数必须使用关键字传入。

## 双侧检验与置信区间的完整流程

假设这些观测值相互独立，来自方差未知的正态总体。单样本 t 统计量为 $t=(\bar x-\mu_0)/(s/\sqrt n)$，自由度为 `df=n-1`。参考分布由这些假设支持；调用 CDF 本身不会验证假设。

```python
# Example: student_t_inference
import numpy as np
from statgpu.inference import t

values = np.array([2.1, 2.4, 2.2, 2.6, 2.8, 2.0, 2.7, 2.3, 2.9, 2.5, 2.4])
null_mean = 2.0
alpha = 0.05
mean = values.mean()
standard_error = values.std(ddof=1) / np.sqrt(values.size)
df = values.size - 1
statistic = (mean - null_mean) / standard_error
pvalue = t.two_sided_pvalue(abs(statistic), df=df, backend="numpy", use_lut=False)
critical = t.two_sided_critical_value(alpha, df=df, backend="numpy", use_lut=False)
interval = mean + np.array([-1.0, 1.0]) * critical * standard_error
print("Mean and SE:", round(mean, 6), round(standard_error, 6))
print("t and two-sided p:", round(statistic, 6), round(float(pvalue), 6))
print("Critical value:", round(float(critical), 6))
print("95% mean interval:", np.round(interval, 6))
print("Reject at 5%:", bool(pvalue < alpha))
```

预期均值为 `2.445455`，标准误为 `0.086722`，统计量为 `5.136596`，p 值为 `0.000440`，右侧临界值为 `2.228139`，区间为 `[2.252226, 2.638683]`。`pvalue < alpha` 的结果为 `True`，因此在 5% 显著性水平下拒绝原假设；原假设中的均值 2 也位于区间之外。

p 值是原假设成立时的尾概率，不是原假设为真的概率。这里得到的是总体均值的置信区间，不是下一次观测值的预测区间。

`norm` 和 `t` 提供 `two_sided_pvalue(stat_abs, ...)` 与 `two_sided_critical_value(alpha, ...)`。传入标准化统计量的绝对值，并保证 `0 < alpha < 1`；这两个辅助方法不接受 `loc` 或 `scale`。右侧单侧检验应使用 `sf(statistic, ...)` 与 `alpha` 比较，用 `isf(alpha, ...)` 求临界值。不能忽略统计量符号和事先确定的备择假设，直接把双侧 p 值除以二。多重检验需要适当校正，参见[推断指南](inference-api.md)。

### 尾部精度的限制

优先使用 `sf`，不要自行计算 `1-cdf`，正态分布的右尾尤其如此。不过，当前部分分布的 `sf` 内部仍通过减法实现，原生 `isf` 也会先计算 `1-q`。极小的 `q` 可能因舍入丢失，导致临界值为无穷大或不准确；`use_lut=False` 不能消除这种消减误差。

极端尾部、密度奇异端点和特殊参数范围都需要针对具体方法核对参考结果。如果需要专门的尾部算法，可以直接在 CPU 上使用 SciPy 对应的 `sf`、`logsf` 或 `isf`；这些原生对象不提供 `logcdf` 或 `logsf`。此外，部分取值范围为非负数的分布及 beta 分布的密度实现，在支持集端点处返回零，而非解析端点极限。核对它们的 PDF 时，应在支持集内部取值。

## 原生分布与参数选择

下表列出全部原生分布名。每一行均提供 `cdf`、`sf`、`ppf` 和 `isf`；连续分布还提供 `pdf`，离散分布提供 `pmf`。除下文说明的 F 分布采样故障外，其余分布均可使用 `rvs`。分布参数应为标量，可批量传入界值或概率，但不要把一组不同形状参数当作可广播输入。支持位置和尺度参数时，`loc` 默认是 `0`，`scale` 默认是 `1`，且 `scale` 必须为正。

| 分布名 | 必需形状参数 | 默认位置和尺度下的支持集 | 位置与尺度控制 |
|---|---|---|---|
| `norm`、`t` | 正态分布无；t 分布需 `df > 0` | 实数轴 | `loc`、`scale` |
| `uniform` | 无 | `[0, 1]` | `loc`、`scale`（区间宽度） |
| `expon` | 无 | `[0, infinity)` | `loc`、`scale`（超过 `loc` 部分的均值，即速率的倒数） |
| `cauchy`、`laplace`、`logistic` | 无 | 实数轴 | `loc`、`scale` |
| `chi2` | `df > 0` | `[0, infinity)` | 均不支持 |
| `gamma` | `a > 0` | `[0, infinity)` | `loc`、`scale`；均值为 `loc + a*scale` |
| `beta` | `a > 0`、`b > 0` | `[0, 1]` | `loc`、`scale` |
| `f` | `dfn > 0`、`dfd > 0` | `[0, infinity)` | 均不支持 |
| `weibull_min` | `c > 0` | `[0, infinity)` | `loc`、`scale` |
| `lognorm` | `s > 0` | `(0, infinity)` | `loc`、`scale`；`loc=0` 时，`log(X)` 的均值为 `log(scale)`，标准差为 `s` |
| `poisson` | `mu >= 0` | 整数 `0, 1, ...` | 仅支持整数 `loc` |
| `binom` | 整数 `n >= 0`、`0 <= p <= 1` | 整数 `0, ..., n` | 仅支持整数 `loc` |

正态分布的 `scale` 是标准差，不是方差。Gamma 分布的 `scale` 是速率的倒数，`a` 是形状参数。二项分布的 `n` 应由调用者保证为整数，不能把实现中的整数转换当作输入验证。离散分布的 CDF 会将非整数计数界值向下取整。

这些对象采用类似 SciPy 的命名，但不是 `scipy.stats` 的完整替代：不支持通过调用分布对象创建冻结分布，也不提供 `fit`、`stats`、`mean`、`var` 或对数密度方法。

## 后端选择、输出类型与转换

自动选择后端有两套不同规则：

- **模块级代理**，例如 `norm.cdf(x)`，会检查调用参数中的 Torch 张量或 CuPy 数组，以第一个识别出的此类数组决定后端。只有 Python 标量、列表或 NumPy 输入时使用 NumPy。未显式覆盖时，Torch 输入张量的设备也会被沿用。不要在一次调用中混用不同数组库或设备。
- **`get_distribution(name, backend="auto")`** 在构造时没有输入数组可供检查，而是依次尝试 CuPy、Torch、NumPy。没有 CUDA 时，Torch 也可能选择 CPU。构造成功不代表所有后续 GPU 运算都能执行；需要可预测行为时，应显式指定后端。

代理允许每次调用指定 `backend="numpy"`、`"cupy"`、`"torch"` 或 `"auto"`，以及 `device`（仅 Torch）和 `use_lut`。固定后端对象在构造时接收这些设置，不应再把它们传给对象的方法。

分布函数允许 `backend="torch", device="cpu"`；这与估计器中严格请求 CUDA 的 `device="torch"` 不是同一个约定。

```python
# Example: fixed_numpy_backend
import numpy as np
from statgpu.inference import norm, get_distribution, list_available_distributions

x = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
automatic = norm.cdf(x)
fixed = get_distribution("norm", backend="numpy", use_lut=False)
result = fixed.cdf(x, loc=0.0, scale=1.0)
scalar = float(fixed.cdf(0.0))
np.testing.assert_allclose(automatic, result)
print("Shape and dtype:", result.shape, result.dtype)
print("Scalar CDF:", scalar)
print("Native count:", len(list_available_distributions()))
```

预期形状是 `(2, 2)`，类型是 `float64`，标量概率是 `0.5`，原生分布数为 15。原生计算会把数值输入转换为 float64。NumPy 调用返回 NumPy 数组、标量或零维结果；CuPy 调用返回 CuPy 数组；Torch 调用返回所选设备上的张量。数组输出保留界值或概率输入的形状。

仅对单元素结果使用 `float(result)`，且 GPU 标量转换可能触发同步。数组应显式转换：NumPy 用 `np.asarray(result)`，CuPy 用 `cupy.asnumpy(result)`，Torch 用 `result.detach().cpu().numpy()`。把 GPU 结果移到 CPU 有传输开销，张量的 `detach()` 还会断开梯度连接。

使用 GPU 时，可以先通过 `cupy.asarray(..., dtype=cupy.float64)` 创建输入，再调用代理；也可以用 `get_distribution("norm", backend="torch", device="cuda:0")` 创建固定对象，并传入兼容张量。显式请求不可用的后端或设备时，构造或计算阶段都可能报错。安装和设备背景见[设备与内存指南](device-and-memory.md)。CPU 示例不能作为 GPU 精度或性能的验证证据。

## 可重复的随机采样

当前原生 `rvs` 方法都先用 NumPy/SciPy **在 CPU 上生成**样本，再转换到指定后端。它们使用 NumPy 的全局随机状态，不接受 `random_state`，而且目前不会应用签名中保留的 `dtype` 参数。若需在同一环境中复现，先设置 NumPy 种子，再按需显式转换结果类型。单独创建 `np.random.default_rng(...)` 不会影响这些调用。不同库版本不保证产生完全相同的序列。

```python
# Example: reproducible_sampling
import numpy as np
from statgpu.inference import norm, poisson

np.random.seed(7)
sample = norm.rvs(size=(2, 3), backend="numpy")
counts = poisson.rvs(mu=3.0, size=4, backend="numpy")
counts_integer = counts.astype(np.int64)
print("Normal sample:", np.round(sample, 6))
print("Count sample:", counts_integer)
print("Original dtypes:", sample.dtype, counts.dtype)
```

正态样本四舍五入后为 `[[1.690526, -0.465937, 0.032820], [0.407516, -0.788923, 0.002066]]`，计数样本为 `[4, 1, 4, 5]`。两个原始数组都是 `float64`，即使 Poisson 样本取值为整数。六个正态观测值远不足以验证一个分布。

### 当前 F 分布采样的限制

`f.rvs(dfn=..., dfd=...)` 和 `rf_gpu(...)` 目前会抛出 `TypeError`，因为原生采样辅助函数向 NumPy 传入了不支持的参数名。F 分布的密度、CDF 和分位数方法不受此问题影响。在 CPU 上采样时，可以直接使用 SciPy；它的采样方法支持 `random_state`，与原生 `rvs` 不同：

```python
# Example: f_sampling_workaround
import numpy as np
from scipy import stats

sample = stats.f.rvs(dfn=5, dfd=10, size=4, random_state=np.random.default_rng(17))
print("F sample:", np.round(sample, 6))
```

预期样本四舍五入后为 `[2.350997, 0.207108, 0.491596, 3.343483]`。这些是 F 分布下的正值观测，不是尾概率。

## 查找表与各后端的数值限制

`use_lut=True` 是默认设置。满足条件的不完全 beta/gamma 反函数使用查找表和迭代修正，这些函数用于 `t`、`f`、`beta`、`chi2`、`gamma` 的分位数计算。是否使用查找表取决于参数及后端。该开关不保证固定加速倍数或全局误差上界；正态分位数不走这些查找表。

| 后端 | `use_lut` 的作用及注意事项 |
|---|---|
| NumPy | 合适的 beta/gamma 反函数使用查找表插值和 Newton 修正；设为 `False` 时使用 SciPy 特殊函数反函数。 |
| CuPy | 合适的 beta/gamma 反函数同样使用查找表；设为 `False` 时使用 `cupyx.scipy.special` 反函数，因此该开关确实影响 CuPy。建表使用主机端 SciPy，再把表传到设备。 |
| Torch | 控制符合条件的查找表路径与数值替代算法；不完全 beta 正向计算也有查找表路径。是否存在原生特殊函数要在运行时检查，不能仅凭 Torch 版本号保证。 |

Torch 建表也依赖主机端 SciPy。缺少原生不完全 beta 函数时，非标量参数的不完全 beta 计算还可能回退到 CPU 上的 SciPy，再把结果转回。因此，返回 GPU 数组或张量不意味着全部计算都留在 GPU。关闭查找表既不保证完全不用 CPU，也不保证所有 Torch 参数范围都更准确。应验证实际需要的方法、参数、尾部范围、后端和库版本。

```python
# Example: compare_lut_paths
import numpy as np
from scipy import stats
from statgpu.inference import get_distribution

q = np.array([0.025, 0.1, 0.9, 0.975])
fast = get_distribution("t", backend="numpy", use_lut=True)
reference_path = get_distribution("t", backend="numpy", use_lut=False)
fast_values = fast.ppf(q, df=10)
reference_values = reference_path.ppf(q, df=10)
scipy_values = stats.t.ppf(q, df=10)
np.testing.assert_allclose(reference_values, scipy_values, atol=1e-9, rtol=1e-9)
np.testing.assert_allclose(fast_values, reference_values, atol=1e-8, rtol=1e-8)
print("Quantiles:", np.round(reference_values, 6))
```

预期分位数为 `[-2.228139, -1.372184, 1.372184, 2.228139]`。断言只验证这几个适中概率以及 `df=10`，不能据此推断全局误差上界、GPU 一致性或加速倍数。重复调用同一个固定对象可保留实例级缓存，代理则会在每次调用时构造分布对象。

## 为额外分布显式启用 SciPy 回退

`get_distribution` 只接受上表中的原生分布名称，即使指定 `backend="numpy"` 也不例外；`get_distribution("gumbel_r", backend="numpy")` 会抛出 `ValueError`。兼容工厂 `get_distribution_gpu` 可以通过 `allow_fallback=True` 显式包装额外的 SciPy 分布：

```python
# Example: explicit_scipy_fallback
import numpy as np
from scipy import stats
from statgpu.inference import get_distribution_gpu

x = np.array([0.0, 1.0, 2.0])
dist = get_distribution_gpu("gumbel_r", allow_fallback=True)
raw = dist.cdf(x)
# The wrapper may return a GPU array on a GPU-equipped machine.
if hasattr(raw, "get"):
    out = raw.get()
elif hasattr(raw, "detach"):
    out = raw.detach().cpu().numpy()
else:
    out = np.asarray(raw)
np.testing.assert_allclose(out, stats.gumbel_r.cdf(x))
print("Gumbel CDF:", np.round(out, 6))
```

预期 CDF 为 `[0.367879, 0.692201, 0.873423]`。实际计算始终在 CPU 上的 SciPy 中进行：GPU 输入会复制到 CPU，输出则可能被转换到全局自动选择的 GPU 后端，即使输入原本是 NumPy 数组也是如此。这个包装器不支持逐次指定 `backend`。如果必须返回 NumPy 结果且不希望发生 GPU 转换，直接调用 `scipy.stats.gumbel_r.cdf(x)`。

`allow_fallback=False` 会拒绝非原生的 SciPy 分布，未知名称也会报错。对于原生分布名称，`get_distribution_gpu` 仍使用原生工厂的自动后端选择。

## 兼容接口与迁移

新代码优先使用对象式接口。以下 R 风格命名仍作为兼容包装器保留（F 采样限制见上文），但这不代表支持 R 的全部参数，例如不会额外提供 `lower.tail`、`log.p` 或 gamma 的 `rate`。

| 分布族 | 密度 / 概率质量 | CDF | 分位数 | 采样 |
|---|---|---|---|---|
| 正态 | `dnorm_gpu` → `norm.pdf` | `pnorm_gpu` → `norm.cdf` | `qnorm_gpu` → `norm.ppf` | `rnorm_gpu` → `norm.rvs` |
| Student t | `dt_gpu` → `t.pdf` | `pt_gpu` → `t.cdf` | `qt_gpu` → `t.ppf` | `rt_gpu` → `t.rvs` |
| 卡方 | `dchisq_gpu` → `chi2.pdf` | `pchisq_gpu` → `chi2.cdf` | `qchisq_gpu` → `chi2.ppf` | `rchisq_gpu` → `chi2.rvs` |
| Gamma | `dgamma_gpu` → `gamma.pdf` | `pgamma_gpu` → `gamma.cdf` | `qgamma_gpu` → `gamma.ppf` | `rgamma_gpu` → `gamma.rvs` |
| Beta | `dbeta_gpu` → `beta.pdf` | `pbeta_gpu` → `beta.cdf` | `qbeta_gpu` → `beta.ppf` | `rbeta_gpu` → `beta.rvs` |
| F | `df_gpu` → `f.pdf` | `pf_gpu` → `f.cdf` | `qf_gpu` → `f.ppf` | `rf_gpu` → `f.rvs` |
| Poisson | `dpois_gpu` → `poisson.pmf` | `ppois_gpu` → `poisson.cdf` | `qpois_gpu` → `poisson.ppf` | `rpois_gpu` → `poisson.rvs` |
| 二项 | `dbinom_gpu` → `binom.pmf` | `pbinom_gpu` → `binom.cdf` | `qbinom_gpu` → `binom.ppf` | `rbinom_gpu` → `binom.rvs` |

这些包装器也应使用关键字形状参数。像 `dpois_gpu(3, 4.0)` 这样转发位置形状参数的调用会被代理拒绝，应改为 `dpois_gpu(3, mu=4.0)`，或迁移为 `poisson.pmf(3, mu=4.0)`。

```python
# Example: compatibility_migration
import numpy as np
from statgpu.inference import dpois_gpu, pnorm_gpu, qt_gpu, poisson, norm, t

mass_old = dpois_gpu(3, mu=4.0)
mass_new = poisson.pmf(3, mu=4.0, backend="numpy")
np.testing.assert_allclose(mass_old, mass_new)
np.testing.assert_allclose(pnorm_gpu(1.96), norm.cdf(1.96, backend="numpy"))
np.testing.assert_allclose(qt_gpu(0.975, df=10), t.ppf(0.975, df=10, backend="numpy"))
print("Poisson mass:", round(float(mass_new), 6))
```

预期概率质量为 `0.195367`。与上表不同，以下**不采用 R 风格命名的旧接口会发出 `DeprecationWarning`**，应迁移到对象方法：

- `norm_cdf_gpu`、`norm_sf_gpu`、`norm_ppf_gpu`、`norm_isf_gpu` → 对应的 `norm.cdf`、`norm.sf`、`norm.ppf`、`norm.isf`。
- `norm_two_sided_pvalue_gpu`、`norm_two_sided_critical_value_gpu` → `norm.two_sided_pvalue`、`norm.two_sided_critical_value`。
- `t_cdf_gpu`、`t_sf_gpu`、`t_ppf_gpu` → `t.cdf`、`t.sf`、`t.ppf`。
- `t_two_sided_pvalue_gpu`、`t_two_sided_critical_value_gpu` → `t.two_sided_pvalue`、`t.two_sided_critical_value`。

## 完整 API 与参考资料

- 工厂签名：`get_distribution(name, backend="auto", device=None, *, use_lut=True)`。分布名称不区分大小写，`list_available_distributions()` 返回原生名称列表。
- 除 `rvs` 外，代理方法通过第一个位置参数接收界值或概率，并接受上表中的分布参数，以及每次调用可设置的控制项 `backend`、`device`、`use_lut`；`rvs` 仅接收关键字参数。固定对象只接受其原生方法参数。`t.ppf`、`t.isf`、`t.two_sided_critical_value` 还接受 `max_bisect_steps=60`，用于二分法备用路径，它不是精度容差。所有原生 `rvs` 签名均包含 `size=None` 和 `dtype=None`，但应注意前述 dtype 限制。
- 兼容工厂：`get_distribution_gpu(name, *, allow_fallback=False)` 和 `list_available_distributions_gpu(include_scipy=True)`。
- 完整签名及方法实现：[分布对象、代理、后端与工厂](../../../statgpu/inference/_distributions_backend.py)。公开导入入口：[`statgpu.inference`](../../../statgpu/inference/__init__.py)。兼容签名及警告：[函数式包装器](../../../statgpu/linear_model/legacy/_distributions_legacy_gpu.py)。Student t 在 `df=1`、`df=2` 下的双侧运行时处理：[低自由度参考实现](../../../statgpu/inference/_low_df_reference_contract.py)。
- 概率术语及外部比较参考：[SciPy 统计分布](https://docs.scipy.org/doc/scipy/reference/stats.html)、[正态分布](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)、[Student t 分布](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)、[Poisson 分布](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)。
