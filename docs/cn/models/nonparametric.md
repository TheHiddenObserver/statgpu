# 非参数方法

> 语言: 中文  
> 最后更新: 2026-10-06  
> 页面定位: 非参数方法总览  
> 切换: [English](../../en/models/nonparametric.md)

## 先明确问题，再选择平滑器

非参数不等于没有假设。核方法利用邻近观测的信息，带宽决定邻域的宽度。

- **观测集中在哪里？** 用核密度估计 KDE，只需要样本 `X`，不需要响应 `y`。
- **连续响应如何随特征变化？** 用核回归，需要成对的 `X, y`。Nadaraya–Watson（`"nw"`）计算局部加权平均；`"local_linear"` 在每个邻域拟合直线，可以减轻边界偏差。
- **需要每个特征各有一条可加曲线？** 参见 [GAM](semiparametric.md)。其他方法见[核岭回归](kernel-methods.md)和[样条基函数](splines.md)。

本页提供 KDE 与核回归的入门流程，不是全部非参数算法的百科。它们更适合低维连续数据；高维、稀疏邻域及外推需要格外谨慎。

## 估计的到底是什么？

对于等权一维数据和绝对带宽 $h$，KDE 估计概率密度：

$$
\hat f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\!\left(\frac{x-X_i}{h}\right).
$$

密度不是“恰好观测到 x 的概率”，数值可以大于 1；对某一区间积分才能得到概率。`pdf` / `predict` 返回密度，`logpdf` / `score_samples` 返回对数密度。KDE 的 `score(X)` 是这些观测的**平均对数密度**，不是分类准确率或 $R^2$。

Nadaraya–Watson 估计条件均值：

$$
\hat m(x)=\frac{\sum_i w_i K_H(x-X_i)y_i}{\sum_i w_i K_H(x-X_i)}.
$$

其中 $w_i\ge0$ 是归一化观测权重，$\sum_i w_i=1$；$H$ 是正定带宽矩阵。
缩放后的核及加权密度定义为

$$
K_H(u)=|H|^{-1/2}K(H^{-1/2}u),\qquad
\widehat f(x)=\sum_i w_iK_H(x-X_i).
$$

$p$ 维高斯核为 $K(v)=(2\pi)^{-p/2}\exp(-v^\top v/2)$。
等权且 $H=h^2$ 时，加权公式退化为上面的一维形式。
局部线性回归在每个查询点选择局部截距 $a$ 和斜率向量 $b$：

$$
(\widehat a(x),\widehat b(x))=
\arg\min_{a,b}\sum_i w_iK_H(x-X_i)
\{y_i-a-b^\top(X_i-x)\}^2,\qquad \widehat m(x)=\widehat a(x).
$$

每个查询点都重新拟合；两种回归都不提供一组全局斜率或系数 p 值。
局部系统奇异时可采用下文说明的稳定化或 NW 回退。

## CPU 示例一：拟合与评价密度

以下 CPU 示例均可单独运行，固定随机种子并显式选择 NumPy。反复查询时应复用已拟合对象；一次性函数方便，但每次调用都会重新拟合。

注意：下面的 `bandwidth=0.35` 是无量纲的**带宽因子**，不是上式的绝对带宽 $h$。对于一维等权数据，$h$ 约为 `0.35` 乘以训练样本标准差；详细规则见后文“带宽、核与调参边界”。

<!-- example: kde-cpu -->
```python
import numpy as np
from statgpu.nonparametric import fit_kde, kde_pdf

rng = np.random.default_rng(42)
x_train = rng.normal(size=300)
x_test = rng.normal(size=80)
grid = np.linspace(-4, 4, 201)

kde = fit_kde(x_train, bandwidth=0.35, kernel="gaussian", backend="numpy")
density = kde.pdf(grid)
log_density = kde.score_samples(grid)
one_shot = kde_pdf(x_train, grid, bandwidth=0.35, backend="numpy")
mass_on_grid = np.sum((density[1:] + density[:-1]) * np.diff(grid) / 2)
print(density.shape)
print(f"Mass on grid: {mass_on_grid:.4f}")
print(f"Held-out mean log density: {kde.score(x_test):.4f}")
```

四舍五入后的输出为 `(201,)`、网格积分 `1.0000`、留出平均对数密度 `-1.4740`。`density` 与 `one_shot` 一致；这里 `log_density` 与 `np.log(density)` 一致。积分是在较宽的有限网格上计算，不能保证任意区间上的积分都为 1。高斯核在观测范围外仍有尾部。

选择带宽时，在**相同留出点和相同计量尺度**上比较平均对数密度，越高越好。不要反复用最终测试集调参。较大带宽会合并细节；较小带宽能显示细节，也可能只是在显示抽样噪声。

## CPU 示例二：预测响应

模拟数据有已知的非线性均值。测试响应加入了新的噪声，因此不应期待零测试误差。这里的绝对带宽使用特征原始单位。

<!-- example: kernel-regression-cpu -->
```python
import numpy as np
from statgpu.nonparametric import KernelRegressionRegressor, kernel_regression_predict

rng = np.random.default_rng(42)
x_train = rng.uniform(-2, 2, 160)
y_train = np.sin(2 * x_train) + rng.normal(0, 0.1, 160)
x_test = np.linspace(-1.8, 1.8, 61)
mean_test = np.sin(2 * x_test)
y_test = mean_test + rng.normal(0, 0.1, x_test.size)

regressor = KernelRegressionRegressor(
    regression="local_linear", kernel="gaussian",
    kernel_metric="diagonal", bandwidth_per_feature=[0.18],
    backend="numpy", device="cpu",
).fit(x_train, y_train)
prediction = regressor.predict(x_test)
one_shot = kernel_regression_predict(
    x_train, y_train, x_test, regression="local_linear",
    kernel_metric="diagonal", bandwidth_per_feature=[0.18], backend="numpy",
)
mse = np.mean((prediction - y_test) ** 2)
baseline_mse = np.mean((y_train.mean() - y_test) ** 2)
print(prediction.shape)
print(f"Test MSE: {mse:.4f}; mean baseline: {baseline_mse:.4f}")
print(f"Test R2: {regressor.score(x_test, y_test):.4f}")
```

四舍五入后输出为 `(61,)`、测试 MSE `0.0126`（均值基线 `0.4690`）、测试 $R^2$ `0.9730`。拟合对象与一次性函数的预测一致。这里 $R^2$ 评价留出响应预测，与 KDE 的平均对数密度得分含义不同。多目标回归的 `score` 会将全部目标展平后计算一个 $R^2$；目标尺度不同时应分别评价。

`0.18` 是示范预设，不是通用最优带宽。真实数据应先划分，再选择带宽、变换及核设置。在验证折上比较候选，用选定设置重拟合训练数据，最后只评价一次测试集。

## CPU 示例三：逐点密度区间

入门 bootstrap 使用独立、等权观测。数值带宽**因子**固定为 `0.4`；每次重采样仍重新计算样本协方差，因此绝对带宽会变化。100 次重采样是为了让示例轻便；正式分析应增加次数并检查区间端点是否稳定。

<!-- example: kde-bootstrap-cpu -->
```python
import numpy as np
from statgpu.nonparametric import kde_bootstrap_confidence_interval

rng = np.random.default_rng(42)
samples = rng.normal(size=120)
points = np.array([-1.0, 0.0, 1.0])
ci = kde_bootstrap_confidence_interval(
    samples, points, bandwidth=0.4, kernel="gaussian", backend="numpy",
    n_resamples=100, confidence_level=0.95, random_state=17,
    method="percentile", return_bootstrap_samples=True,
)
print(ci.estimate.shape, ci.bootstrap_samples.shape)
print(np.column_stack([ci.estimate, ci.lower, ci.upper]).round(3))
```

```text
(3,) (100, 3)
[[0.249 0.165 0.315]
 [0.431 0.369 0.505]
 [0.239 0.190 0.305]]
```

三列依次是估计、下限、上限。这些是查询点处平滑密度的**逐点百分位 bootstrap 区间**，不是同时置信带、未来观测值区间或概率区间。它们不校正平滑偏差、样本依赖性或先前带宽搜索的不确定性。百分位区间不一定包含原始估计，但下限必须不大于上限。

需要注意：

- NumPy 一维高斯 bootstrap 快速路径会固定最初选出的因子，即使 `bandwidth` 是字符串；其他路径会在每次重采样时重新运行选择器。如果希望每次重采样都使用相同的带宽因子，可像示例一样直接指定数值因子。样本协方差仍会重新计算，因此绝对带宽仍可能变化；不要假定各后端都以相同方式计入选择器的不确定性。
- 非均匀 `weights` 会同时用于抽样概率及抽样后的再次加权。这并不等价于所有频数权重、调查权重或重要性权重 bootstrap。本示例及解释限定于等权观测；加权区间必须先确认重采样方式适合研究设计。
- 该包装函数仅接受 `method="percentile"`。相关的 `kde_confidence_interval` 提供默认的 `method="normal"`（仅一维高斯核的渐近近似）或 `"bootstrap"`；都不是偏差校正区间或同时置信带。

对于置信水平 $1-\alpha$，每个固定查询点的百分位区间取 $B$ 次密度估计的经验分位数：

$$
[L(x),U(x)]=[Q_{\alpha/2}\{\widehat f_b^*(x)\}_{b=1}^B,
Q_{1-\alpha/2}\{\widehat f_b^*(x)\}_{b=1}^B].
$$

独立的正态方法使用拟合绝对宽度 $h$，以及由归一化权重定义的
$n_{\mathrm{eff}}=1/\sum_i w_i^2$。高斯核满足
$R(K)=\int K(u)^2du=1/(2\sqrt\pi)$，因此

$$
\widehat{\mathrm{SE}}(x)=\sqrt{\frac{\widehat f(x)R(K)}{n_{\mathrm{eff}}h}},
\qquad
[L(x),U(x)]=[\max\{0,\widehat f(x)-z_{1-\alpha/2}\widehat{\mathrm{SE}}(x)\},
\widehat f(x)+z_{1-\alpha/2}\widehat{\mathrm{SE}}(x)].
$$

$z_{1-\alpha/2}$ 是标准正态分位数。两种区间都逐点构造；正态公式使用渐近方差近似，
不校正偏差。见[正态区间示例与完整参数](../reference/survival-smoothing-api.md#密度置信区间)。

## 形状与接口选择

以下接口均从 `statgpu.nonparametric` 导入：

| 需求 | API | 返回 / 用法 |
|---|---|---|
| 可复用密度拟合 | `fit_kde(samples, ...)` | 已拟合 `KDE`，为 `KernelDensityEstimator` 的子类。 |
| 估计器式密度拟合 | `KernelDensityEstimator(...).fit(X)` | 复用 `pdf`、`logpdf`、`predict`、`score_samples`、`score` 或 `__call__`（密度）。 |
| 一次性密度 | `kde_pdf(samples, points, ...)` | 密度向量；`return_log=True` 返回对数密度。 |
| 密度区间 | `kde_bootstrap_confidence_interval(...)`、`kde_confidence_interval(...)` | `KDEBootstrapResult`。 |
| 可复用回归拟合 | `fit_kernel_regression(samples, targets, ...)` | 已拟合 `KernelRegression`。 |
| 估计器式回归 | `KernelRegressionRegressor(...).fit(X, y)` | `KernelRegression` 的别名子类；使用 `predict`、`score`、`__call__`。 |
| 一次性回归 | `kernel_regression_predict(samples, targets, points, ...)` | 响应预测数组。 |

- 训练样本：有限数值，形状 `(n_samples,)` 或 `(n_samples, n_features)`，至少两条观测。回归目标：有限数值，形状 `(n_samples,)` 或 `(n_samples, n_targets)`，行数须匹配。
- 拟合与评价将数值输入转换为 float64；输入为 float32 时，拟合数组和预测不会保留该精度。
- 查询：`(n_query, n_features)`。单特征模型的一维向量代表多个点；多特征模型的一维向量若长度等于特征数，则代表一个点。特征数和顺序须与训练一致。
- KDE 输出为 `(n_query,)`；一维目标的回归输出为 `(n_query,)`，二维目标则为 `(n_query, n_targets)`，包括 `(n_query, 1)`。NumPy 路径输出 NumPy 数组，普通 GPU 预测保留后端数组类型；区间结果数组会转为 NumPy。
- 拟合用 `weights` 须有限、非负、长度为 `n_samples` 且总和为正，内部会归一化。权重全部集中于单个观测时协方差估计失败。错误形状、非有限输入、非正带宽、未知核名会报错。必须先 `fit` 再预测。

原始坐标偏移量很大时，应先用同一训练偏移量中心化样本与查询。当前距离计算
可能在未中心化时损失精度，尤其是对数密度及多元密度/回归；见 [API 数值限制](../reference/survival-smoothing-api.md#核密度估计)。

使用加权高斯 KDE 的 `logpdf`、`score_samples` 或 `score` 时，请先删除零权重观测及其对应权重。虽然这些观测不贡献概率质量，但当前实现可能因它们的存在而将尾部对数密度错误地算成 `-inf`。删除后会自动重新归一化权重。如果分析希望根据正权重观测选择新带宽，应先筛选再选择。重新运行字符串选择器可能改变所选因子：`nrd`/`nrd0` 使用原始样本的尺度统计量，`ucv`、`bcv`、`sj-ste` 与 `sj-dpi` 则可能在删除后改变加权/重采样路径。若需保留已经选定的平滑因子，请保存 `original.bandwidth_factor_`，再用数值参数 `bandwidth=original.bandwidth_factor_`、正权重观测及其保留权重重新拟合。这会保留指定的核协方差与密度，但不代表受零权重行影响而选出的因子适合原本的科学问题。

### 当前 Torch 限制

多特征 Torch 模型的查询请显式传入二维数组，即使只有一个点也应使用 `X[:1]`，
而不是 `X[0]`。当前一维向量的形状检查会抛出 `TypeError`；增加行维度可避开这一问题。

Torch 拟合目前还会在传入 `weights` 或回归的 `bandwidth_per_feature` 时抛出
`TypeError`，后者无论传标量还是向量都会失败。Torch KDE 的 bootstrap 区间即使使用
`weights=None` 也会失败，因为每次重采样都会显式传入权重。需要加权拟合、逐特征
绝对带宽或 bootstrap 区间时，请使用 CPU 数组和 `backend="numpy"`。不要为了让程序
运行而省略权重或替换带宽，这会改变分析含义。不加权 Torch KDE、标量因子回归以及
一维高斯核的正态区间使用不同的计算路径。

## 带宽、核与调参边界

| 控制项 | 实际含义 |
|---|---|
| `bandwidth="scott"` / `"silverman"` | 根据有效样本量和维度计算的起始规则，默认 Scott。 |
| 数值 `bandwidth` | 正的**无量纲因子** $b$，核协方差为 $H=b^2\widehat\Sigma$。一维绝对宽度约为 $b\,s_x$，不是 $b$ 本身。 |
| `kernel_metric="full"` | 回归默认值：以完整协方差定义邻域。KDE 也使用完整协方差。 |
| `kernel_metric="diagonal"` | 回归删除协方差非对角项，改变距离度量，不是精确/近似开关。 |
| `bandwidth_per_feature` | 仅回归、仅对角度量；每个特征以自身单位给出正的**绝对宽度**，标量可广播。设定 $H=\operatorname{diag}(h_j^2)$ 并跳过 `bandwidth` 选择。 |
| `kernel` | `gaussian`、`rectangular`、`triangular`、`epanechnikov`、`biweight`、`triweight`、`cosine`、`optcosine`；后两种仅一维。 |
| `batch_size` | 正的查询批大小，默认 `1024`；NumPy 一维高斯密度快速路径可能一起评价查询点。 |

对于 $p$ 个特征和归一化权重，默认因子规则为

$$
n_{\mathrm{eff}}=\frac1{\sum_i w_i^2},\qquad
b_{\mathrm{Scott}}=n_{\mathrm{eff}}^{-1/(p+4)},\qquad
b_{\mathrm{Silverman}}=\left(\frac{n_{\mathrm{eff}}(p+2)}4\right)^{-1/(p+4)}.
$$

它们设定 $H=b^2\widehat\Sigma$，其中 $\widehat\Sigma$ 是加权样本协方差，
实现另加数值稳定项。等权时 $n_{\mathrm{eff}}=n$；一维时 $h=\sqrt{H_{11}}$。

其他带宽名包括 `nrd0`、`nrd`、`ucv`、`bcv`、`sj`、`sj-ste`、`sj-dpi`，并非都以预测损失为目标。R 风格选择器使用高斯参考规则；非均匀权重可能用分位数重采样，多元扩展用一维主轴投影。通过 `bandwidth_info_` / `to_numpy_metadata()` 查看实际因子和策略，不应把这些扩展称为与多元 R 方法完全相同。常数或过于稀疏的数据等情形可能使选择器失败。

核回归还接受 `"cv"`、`"cv_ls"`、`"cv-nw"`、`"cv-ll"`，以留一 MSE 搜索标量因子。选择器使用完整协方差，局部线性 CV 修正仅在一维实现；多维时即使使用 `"cv-ll"`，目标也采用 NW 预测。多元局部线性或对角度量模型应针对**实际打算使用的模型**显式验证候选宽度，不能假定该选择器优化的就是相同配置。这些 CV 名称不是 KDE 的带宽选项。

没有统一的严格/近似开关。紧支撑核的查询点可能附近没有观测：KDE 返回零密度（`logpdf=-inf`）；回归在局部有效权重太低时（默认 `min_effective_weight=1e-12`）返回训练目标的加权均值。局部线性求解失败或不稳定时可采用稳定化或退回 NW。远处能返回数值不代表外推可靠；高斯回归在足够远处也可能进入低权重回退。

### 坐标尺度过小

训练特征方差很小时，协方差的绝对稳定化增量可能主导带宽。使用数值带宽因子、
Scott 或 Silverman 规则时，仅改变计量单位就可能显著改变 KDE 和核回归结果。
例如，41 个从 -2 到 2 的等距样本搭配 `bandwidth=0.4`，在零处的 KDE 约为
0.243898；样本和查询同时乘以 `1e-9` 后，将密度换回原单位却约为 0.000997，
而不是 0.243898。结果仍为有限值，因此只检查是否含 NaN/Inf 无法发现问题。

拟合前，根据训练数据确定每个特征的正缩放尺度，对查询使用相同变换
`z = (x - location) / scale`。让训练数据的典型变动幅度接近 1，可减轻这里
演示的问题；仅中心化不够。KDE 返回原始单位下的密度时，应使用
`density_x = density_z / np.prod(scale)`，对数密度则使用
`log_density_x = log_density_z - np.log(scale).sum()`。密度有量纲，不能直接把
标准化坐标下的密度当作原始密度。核回归的响应预测无需乘除密度的雅可比因子。
变换坐标时，数值带宽因子保持不变；若指定绝对宽度 `bandwidth_per_feature`，
则各宽度须除以对应特征的缩放尺度。调参时应在训练折内学习预处理参数。

不同带宽路径受影响的程度不同：一维 `nrd`/`nrd0` 根据绝对宽度换算因子，
可抵消本例中的协方差增量；显式 `bandwidth_per_feature` 也不直接采用带有该
增量的样本协方差。更换选择器会改变平滑规则，不能视为同一统计模型的数值修复。
缩放也不能保证奇异、病态或其他不稳定拟合的准确性；仍应独立验证目标协方差和预测。

## 完整 API 与诊断参考

[面向使用者的完整 API 参考](../reference/survival-smoothing-api.md#核密度估计)
集中列出构造、函数和方法签名、默认值、限制、返回形状及选择器/区间结果字段。
特别注意：KDE 的 `batch_size` 是评价参数，不是构造参数；回归则可以在两处设置。
下面保留实现链接供深入查阅：

- [KDE 与区间](../../../statgpu/nonparametric/kernel_smoothing/_kde.py)：`KernelDensityEstimator`、`KDE`、`fit_kde`、`kde_pdf`、`kde_confidence_interval`、`kde_bootstrap_confidence_interval`、`KDEBootstrapResult`。估计器还接受 `weights=None`、`backend="auto"`、`device="auto"`、`n_jobs=None`、`gpu_memory_cleanup=False`。区间控制包括 `n_resamples=200`、`confidence_level=0.95`、`random_state=None`、`return_bootstrap_samples=False`、`batch_size=1024`；通用区间函数的 `bootstrap_method="percentile"`。
- [核回归](../../../statgpu/nonparametric/kernel_smoothing/_kernel_regression.py)：`KernelRegression`、`KernelRegressionRegressor`、两个函数式接口、全部拟合/预测控制及 `to_numpy_metadata()`。构造函数还包含相同的设备选择（`device`）、并行任务数（`n_jobs`）和 GPU 内存清理（`gpu_memory_cleanup`）参数，以及 `batch_size` / `min_effective_weight`；`predict` 可覆盖后两项。一次性函数选择 `backend`，不接受 `device` 参数。
- [带宽选择](../../../statgpu/nonparametric/kernel_smoothing/_bandwidth_selection.py)：`select_bandwidth` 返回带诊断的 `BandwidthSelectionResult`；`select_bandwidth_factor` 返回标量。这些底层函数需要样本、协方差、权重和后端等输入，通常直接通过估计器选择即可。
- [共享校验与核定义](../../../statgpu/nonparametric/kernel_smoothing/_kernel_common.py)、[共享估计器参数](../../../statgpu/_base.py)、[非参数模块完整导出清单](../../../statgpu/nonparametric/__init__.py)。

常用拟合属性包括 `samples_`、归一化 `weights_`、`bandwidth_factor_`、`bandwidth_info_`（显式数值因子时为 `None`）、`covariance_`、`inv_covariance_`、`kernel_`、`backend_`、`n_samples_`、`n_features_`。回归还包括 `targets_`、`n_targets_`、`target_mean_`、`regression_`、`kernel_metric_`、`bandwidth_per_feature_`。`to_numpy_metadata()` 提供主机端诊断。区间结果有 `points`、`estimate`、`lower`、`upper`、`confidence_level`、`n_resamples`、`random_state`、`kernel`、`backend`、`metadata`、可选的 `(n_resamples, n_query)` `bootstrap_samples` 以及 `to_dict()`。

## 可选 GPU 路径与外部对照

`backend` 接受 `"numpy"`、`"cupy"`、`"torch"` 或 `"auto"`。显式设置选择数组库；`"auto"` 会参考估计器/全局设备配置。`device` 是估计器构造参数，但 KDE 与核回归目前未始终执行显式加速器请求：NumPy 或 Torch CPU 输入搭配 `device="torch"` 和 `backend="auto"` 或 `"torch"`，仍可能在 Torch CPU 上计算。显式 `backend="torch"` 搭配 `device="cuda"` 也可能在 CPU 上运行；`backend="numpy"` 则会覆盖这两种加速器请求并返回 CPU 数组。

因此，仅让设备与后端字符串一致仍不够。应同时检查 `samples_` 与密度/预测数组：Torch 的 `.device`、`.is_cuda` 显示张量位置，CuPy 的 `.device` 显示 GPU，NumPy 数组位于 CPU。不能根据 `model.device` 或 `backend_` 认定 CUDA 执行。明确选择 CPU 时，请用 NumPy 输入并设置 `device="cpu", backend="numpy"`。这些当前例外并未改变[设备与内存](../guides/device-and-memory.md)说明的严格设备约定。

下面的片段与 CPU 流程分开，需要可工作的 CuPy/CUDA，不能在仅 CPU 安装上运行；显式指定但缺失的后端不会静默替换为 NumPy。

<!-- example: kde-gpu -->
```python
import cupy as cp
from statgpu.nonparametric import fit_kde

samples_gpu = cp.linspace(-2, 2, 100)
points_gpu = cp.linspace(-3, 3, 41)
kde_gpu = fit_kde(samples_gpu, bandwidth=0.35, backend="cupy")
density_gpu = kde_gpu.pdf(points_gpu)  # CuPy 输出
```

部分带宽选择及区间步骤使用主机数组，不能假定整个流程常驻 GPU 或小数据也会加速。耗时取决于样本数、查询数、维度、分批、选择器及传输成本。

与 SciPy `gaussian_kde` 对照时，对齐数据方向、权重和协方差带宽因子；与 statsmodels 核回归对照时，对齐核、回归模式、对角度量及**绝对逐特征宽度**，这里的标量因子并不是相同参数。贡献者可查阅[验证参考](../../../dev/references/model-validation.md#nonparametric-models)。

## 参考文献

- Rosenblatt, M. (1956). Remarks on some nonparametric estimates of a density function. *Annals of Mathematical Statistics*, 27(3), 832–837. [DOI](https://doi.org/10.1214/aoms/1177728190).
- Parzen, E. (1962). On estimation of a probability density function and mode. *Annals of Mathematical Statistics*, 33(3), 1065–1076. [DOI](https://doi.org/10.1214/aoms/1177704472).
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141–142. [DOI](https://doi.org/10.1137/1109020).
- Watson, G. S. (1964). Smooth regression analysis. *Sankhya: The Indian Journal of Statistics, Series A*, 26(4), 359–372.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
