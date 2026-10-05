# GAM（广义可加模型）

> 语言: 中文  
> 最后更新: 2026-10-04  
> 页面定位: 模型文档  
> 切换: [English](../../en/models/semiparametric.md)

## 当一条直线不够用时

如果连续响应随某个特征先上升、再趋平，单一线性斜率可能过于简单。`GAM` 为每个特征学习一条平滑曲线，再将它们相加：

$$
y = \alpha + \sum_j f_j(x_j) + \epsilon.
$$

当特征作用近似**平滑且可加**，又希望比线性模型更灵活时，可以考虑它。本实现针对**连续响应的惩罚最小二乘**。虽然类名是广义可加模型，它并没有用于二分类或计数响应的 `family` / `link` API，也不会自动生成交互项。响应分布模型见 [GLM](generalized-linear-model.md)，局部平滑见[非参数方法](nonparametric.md)，自行构造平滑项见[样条基函数](splines.md)。

## 直觉与目标函数

每条曲线都是若干 B 样条基函数的加权组合。更多基函数能表达更细的变化；平滑惩罚则抑制不必要的波动。拟合时按照训练数据将各基列中心化，因此截距表示总体响应水平。

$$
\min_\beta \|y-B\beta\|_2^2 + \lambda\beta^\top S\beta,
\qquad (B^\top B+\lambda S)\hat\beta=B^\top y.
$$

$B$ 拼接截距列和每个特征的中心化样条基；$S$ 是各平滑项 $D^\top D$ 的块对角矩阵，截距不受惩罚。$D$ 是差分矩阵，其阶数由 `penalty_order` 指定。损失是残差平方**和**，不是均值。实现优先使用带数值稳定项的 Cholesky 求解，必要时改用一般线性求解或最小二乘；稳定项也影响报告的有效自由度（EDF）。

`degree=3` 决定分段**三次**基函数；`penalty_order=2` 惩罚相邻**基系数**的二阶差分。两者作用不同：将惩罚改为一阶不会把三次样条变为分段线性样条。若需要分段线性基，应选择 `degree=1`。

## 完整 CPU 流程

只用训练样本拟合并选择平滑强度。下面的留出样本位于训练范围内，便于在有数据支持的区间评价曲线。

<!-- example: gam-cpu -->
```python
import numpy as np
from statgpu.semiparametric import GAM

rng = np.random.default_rng(42)
X_train = rng.uniform(-2, 2, size=(240, 2))
y_train = (np.sin(2 * X_train[:, 0]) + 0.4 * X_train[:, 1] ** 2
           + rng.normal(0, 0.15, 240))
X_test = rng.uniform(-1.9, 1.9, size=(80, 2))
y_test = (np.sin(2 * X_test[:, 0]) + 0.4 * X_test[:, 1] ** 2
          + rng.normal(0, 0.15, 80))

gam = GAM(n_splines=12, lam=None, device="cpu").fit(X_train, y_train)
prediction = gam.predict(X_test)
mse = np.mean((prediction - y_test) ** 2)
baseline_mse = np.mean((y_train.mean() - y_test) ** 2)
print(prediction.shape)
print(f"Test MSE: {mse:.4f}; mean baseline: {baseline_mse:.4f}")
print(f"lambda: {gam.lam_:.4f}; EDF: {gam.edf_:.2f}; GCV: {gam.gcv_score_:.4f}")

fixed = GAM(n_splines=12, lam=gam.lam_, device="cpu").fit(X_train, y_train)
print(fixed.gcv_score_)  # None: a fixed-lambda fit does not run GCV
```

典型输出（已四舍五入）：

```text
(80,)
Test MSE: 0.0214; mean baseline: 0.6278
lambda: 0.1963; EDF: 17.78; GCV: 0.0269
None
```

### 如何解释结果

- 测试 MSE 的单位是响应单位的平方。在未参与拟合的样本上优于训练均值基线，比训练误差很小更有说服力。还应检查残差及不同特征区间的表现。
- `edf_` 表示平滑后的有效模型复杂度，可以不是整数，也不等于原始系数数量。
- `gcv_score_` 是广义交叉验证（GCV）得分，用于在训练数据内选择平滑强度，不是测试 MSE 或 p 值。比较同一数据、相同 `gamma` 的候选模型时，越低越好。
- `coef_` 包含截距和样条基系数，**不能当作原始特征的斜率**。可以改变某个特征并观察预测来理解曲线。由于基列已中心化，`intercept_` 近似等于训练响应均值。
- 固定 lambda 重拟合沿用选中的数值，预测应与 `gam` 一致；但 `fixed.gcv_score_` 为 `None`，因为没有重新搜索，不代表拟合失败。

## 如何选择平滑强度

`lam=None` 时，模型在 $10^{-10}$ 到 $10^{10}$ 的 100 个对数间隔候选值中最小化

$$
\operatorname{GCV}(\lambda)=\frac{n\operatorname{RSS}}{(n-\gamma\operatorname{edf})^2},
\qquad
\operatorname{edf}=\operatorname{clip}\!\left(\operatorname{tr}\!\left((A+\delta I)^{-1}B^\top B\right),0,m\right).
$$

其中 $A=B^\top B+\lambda S$，$m$ 是包含截距的基系数数量，$\delta=10^{-10}\operatorname{tr}(A)/m$。这是通常的稳定化 Cholesky 路径所报告的 EDF。完整样条基中心化后，$A$ 可能仍然奇异，因此不能假定未加稳定项的矩阵存在普通逆。如果 EDF 的数值求解失败，实现会报告 $m$；应检查拟合情况，不要把这个回退值当作经过独立验证的模型复杂度。标准 GCV 使用 `gamma=1`，更大的 `gamma` 对有效复杂度施加更强惩罚。当平方前的修正项 $1-\gamma\operatorname{edf}/n$ 非正或过于接近零时，该候选值的 GCV 得分会被设为无穷大；应确认最终选择的分数有限。

可以从三次基和二阶差分惩罚开始。仅在曲线明显受限时增加 `n_splines`，并重新检查留出误差。增大 `lam` 通常使曲线更平滑。按分位数设置节点时，数据较密集的区间会布置更多节点；均匀节点则在观测范围内等距排列。用验证集或交叉验证选择这些设置，最后保留一个未参与调参的测试集。

GCV 是离散网格搜索，可能错过两个候选值之间的最优值；固定 `lam` 只是跳过选择，仍使用同一数值求解器。`GAM` 构造函数不接受自定义 lambda 网格。需要细搜时，可在训练/验证数据上比较若干固定值，再按选定设置重拟合。

## 输入、边界与限制

- `fit(X, y)` 接受有限数值的 `X`，形状为 `(n_samples, n_features)`，以及每行一个连续响应，通常为 `(n_samples,)`。一维 `X` 被视作单特征；`y` 会展平，因此单列目标也可用。空输入、长度不匹配、非有限值、常数特征列会触发 `ValueError`。不要自行添加全 1 截距列。
- 面向数值连续特征。大量重复值/离散值可能产生重复分位数节点；实现会去重，但落在训练边界上的节点可能触发 `ValueError`。增加基函数数量并不能使类别型特征适合直接进行平滑拟合。
- `predict(X)` 的特征数量和顺序必须与训练一致，推荐显式传入 `(n_query, n_features)`。单特征模型的一维向量表示多条查询；多特征模型的长度为 `n_features_` 的向量表示一条查询。输出始终为 `(n_query,)` 的 **NumPy 数组**，包括 GPU 拟合后的预测。
- 预测复用训练节点和边界。超出某个特征的训练范围时，其 B 样条基在中心化前为零；这不是可靠的线性或平滑外推规则。应在有训练支持的范围解释结果。
- 可加结构可能遗漏交互；强相关特征的单独平滑效应也可能难以解释，即使整体预测尚可。
- 这里不提供系数标准误、p 值、置信带、`cov_type`、样本加权拟合或 family/link 似然。EDF 和 GCV 不能替代不确定性区间。

输入与基函数计算使用 float64，不保留 float32 输入精度。重拟合若在构造基函数时失败，实例可能保留旧系数，同时已部分替换节点及特征信息。此时应丢弃该实例，在新模型上成功拟合后再预测或查看摘要。

## 完整构造参数与输出参考

导入：`from statgpu.semiparametric import GAM`。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `n_splines` | `20` | 为每个特征设置的 B 样条基函数数量；须为大于 `degree + 1` 的整数。节点去重后，实际使用的基函数数量可能减少。 |
| `degree` | `3` | 非负整数样条次数。 |
| `lam` | `None` | 自动 GCV 选择，或有限非负的固定惩罚强度。 |
| `penalty_order` | `2` | 正整数差分阶数，须小于每个特征实际使用的基函数数量。 |
| `knot_method` | `"quantile"` | `"quantile"` 或 `"uniform"`；使用这些小写形式。 |
| `gamma` | `1.0` | GCV 中正且有限的 EDF 乘数；不改变固定 lambda 的目标函数。 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`；见下文设备说明。 |
| `n_jobs` | `None` | 用于兼容通用估计器接口；当前 GAM 拟合过程不使用该参数进行并行计算。 |

| 输出 | 形状/类型 | 含义 |
|---|---|---|
| `coef_` | 后端数组，`(1 + sum(n_basis_j),)` | 截距后依次为各特征中心化基的系数。 |
| `intercept_` | `float` | 截距系数。 |
| `edf_` | `float` | 总有效自由度。 |
| `gcv_score_` | `float` 或 `None` | 搜索到的最小 GCV；固定 `lam` 时为 `None`。 |
| `lam_` | 标量 | 拟合使用的平滑参数。 |
| `knots_` | 后端数组列表 | 各特征的内部节点。 |
| `n_features_` | `int` | 训练特征数。 |

方法：`fit(X, y)` 返回自身；`predict(X)` 返回预测；`summary()` 打印诊断并返回字典（固定 `lam` 时省略 `gcv_score` 键）；`get_params(deep=True)` / `set_params(**params)` 用于参数访问。改参后请重新拟合。本类没有专用的 `score()` 方法，可按示例自行计算留出指标。

[完整方法与输出参考](../reference/survival-smoothing-api.md#gam)列出准确调用签名、
摘要字典键、继承工具的适用边界以及改参行为。`fit` 只使用 `X` 和 `y`；
额外拟合关键字目前不能启用其他能力，请勿传入。特别是 `sample_weight` 不会产生加权拟合。

算法源码：[GAM](../../../statgpu/semiparametric/_gam.py)、[惩罚最小二乘与 GCV](../../../statgpu/nonparametric/splines/_penalized.py)、[基函数构造](../../../statgpu/nonparametric/splines/_bspline_basis.py)、[共享估计器方法](../../../statgpu/_base.py)。

## 可选 GPU 路径

先运行 CPU 示例，再运行下面的 GPU 片段；它需要可工作的 CuPy/CUDA 环境。`device="torch"` 选择 Torch CUDA 路径，显式请求 GPU 不会静默退回 CPU；`device="auto"` 允许自动选择。详见[设备与内存](../guides/device-and-memory.md)。GPU 收益取决于基维度、数据传输和硬件，应针对实际工作负载测量。

<!-- example: gam-gpu -->
```python
# 可选：复用 CPU 示例中的 X_train、y_train、X_test。
gam_gpu = GAM(n_splines=12, device="cuda").fit(X_train, y_train)
prediction_gpu = gam_gpu.predict(X_test)  # NumPy 输出
```

## 外部对照与参考文献

与 pyGAM 等平滑器比较时，应对齐节点、基次数、差分惩罚、损失归一化、lambda 和 GCV 的 `gamma`。相似类名不代表拟合或推断完全相同。一个 CPU 示例的测试不构成 GPU/性能基准证据。

- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman & Hall/CRC.
