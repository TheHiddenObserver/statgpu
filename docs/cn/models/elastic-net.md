# Elastic Net 弹性网络

> 语言：中文  
> 最后更新：2026-10-04<br>
> 页面定位：模型文档  
> 切换：[English](../../en/models/elastic-net.md)

## 概述

`ElasticNet` 结合 L1 与 L2 正则化，在稀疏特征选择（Lasso）和系数收缩（Ridge）之间取得平衡。支持 CPU、CuPy GPU 与 PyTorch GPU。直接拟合统一使用**与后端无关的 `solver` 接口**；`device` 只负责决定在哪里执行。

## 什么时候使用 Elastic Net？

如果希望部分系数精确为零，但相关预测变量又使纯 Lasso 的选择不够稳定，可以考虑 Elastic Net。只需要稳定收缩、不需要稀疏性时，Ridge 更简单；对于事先确定、维数不高且不需要收缩惩罚的模型，可使用[线性回归](linear-regression.md)。入选变量不自动代表因果效应。

## 完整的 CPU 示例

下面生成相关预测变量，只用训练行估计标准化参数，再在留出行上评估预测。示例中的调参值用于演示接口，实际应用应通过验证数据选择。

```python
import numpy as np
from statgpu.linear_model import ElasticNet

rng = np.random.default_rng(7)
X_raw = rng.normal(size=(400, 8))
X_raw[:, 1] = 0.8 * X_raw[:, 0] + 0.2 * rng.normal(size=400)
y = 0.4 + X_raw @ np.array([1.2, 0.8, -0.7, 0, 0, 0, 0, 0])
y += rng.normal(scale=0.5, size=400)

mean = X_raw[:300].mean(axis=0)
scale = X_raw[:300].std(axis=0)
X = (X_raw - mean) / scale
model = ElasticNet(
    alpha=0.08, l1_ratio=0.5, solver="fista", device="cpu",
    max_iter=5000, tol=1e-8, compute_inference=False,
).fit(X[:300], y[:300])

print("coef:", np.round(model.coef_, 3))
print("selected columns:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("predictions:", np.round(model.predict(X[300:303]), 3))
print("held-out R2:", round(float(model.score(X[300:], y[300:])), 3))
```

使用这个随机种子，CPU 输出约为：系数 `[0.938, 0.834, -0.552, 0, 0, 0, 0, 0]`，入选列 `[0, 1, 2]`，预测值 `[-1.575, 1.710, 1.476]`，留出 R² 为 `0.936`。本次拟合保留了相关的第 0、1 列，五个噪声列的系数为零。较高的测试 R² 说明在这组模拟留出数据上的预测效果，不代表入选变量已经成为经过检验的科学发现。不同数值环境可能产生小幅浮点差异。

### 结果解释与参数选择

- `coef_` 按输入列顺序保存系数。由于示例标准化了 `X`，特征增加一个单位表示增加一个训练集标准差。`intercept_` 与 `coef_` 都属于预测拟合结果。
- 正 `l1_ratio` 允许系数精确为零。示例打印的入选列序号只是惩罚拟合的数值摘要，不代表具有错误率控制保证的发现。
- `predict(X_new)` 返回一维响应预测，`score(X_new, y_new)` 返回 R²。新数据上的 R² 可能为负，训练 R² 也不能证明泛化能力。
- `alpha` 越大，总体正则化越强；`l1_ratio` 越接近 1 越偏向稀疏，越接近 0 越偏向 L2 收缩。预测任务应通过验证数据或 `ElasticNetCV` 同时选择两者，并单独保留最终测试集。
- 每次划分只在训练部分估计预处理参数。除以标准差前先移除或处理常量列；不要在交叉验证前用全部数据估计中心化与缩放参数。

## 输入与预测要求

`X` 应为有限数值组成的 `(n_samples, n_features)` 矩阵，`y` 为一维响应。预测时必须保持训练时的列顺序与预处理方式。拟合接口为 `fit(X=None, y=None, sample_weight=None, initial_coef=None, **kwargs)`；可选的非负分析权重（analytic weights）进入归一化加权损失。`initial_coef` 只为一次拟合提供初始系数，不是持久的 `warm_start` 构造参数。共享的可选公式接口通过拟合关键字接受 `formula=` 与 `data=`。

## 路径

`statgpu.linear_model.ElasticNet`

## 目标函数

Elastic Net 优化问题为：

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \lambda \|\beta\|_1 + \frac{\alpha}{2}(1 - \lambda)\|\beta\|_2^2.
$$

其中：
- `alpha` (α) 控制整体正则化强度；
- `l1_ratio` (λ) 混合 L1 与 L2：λ=1 对应 Lasso 目标，λ=0 对应纯 L2 目标；
- `1/(2n)` 表示公开 `alpha` 使用平均损失尺度。

**正则化缩放说明**：`ElasticNet` 与 `Ridge` 使用同一平均损失约定，因此 `l1_ratio=0` 时，相同公开 `alpha` 下目标函数退化为对应的 L2 目标。不过 `ElasticNet` 估计器仍保留自己的求解器和推断默认设置；若明确需要 Ridge 的估计器契约，应直接使用 `Ridge`。

## 估计方程

消去未惩罚截距（等价地，在中心化数据上）后，系数满足 KKT 条件：

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0.
$$

对**直接单次拟合**，`solver` 在所有后端上都决定实际使用的算法。`device` 单独控制 CPU/CuPy/Torch 执行位置。历史 `cpu_solver` 参数已进入弃用流程，在统一引擎中不再代表第二套 CPU 直接拟合求解器。参见 [penalized solver API 迁移指南](../guides/penalized-solver-api-migration.md)。

## 估计算法

默认使用 **FISTA**（快速迭代收缩阈值算法），也就是带 Nesterov 加速的近端梯度法。其他求解器能否使用，以公开的求解器兼容性约束为准。

### 近端更新

Elastic Net 的 L1/L2 部分由近端算子共同处理：

```text
# 平均平方误差项的梯度
grad = (X.T @ X @ w - X.T @ y) / n

# Elastic Net 近端更新
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (
    1 + alpha * (1 - l1_ratio) * step
)
```

### 收敛判据

`stopping` 提供两种停止模式：

| 模式 | 说明 |
|------|------|
| `coef_delta` | 系数变化低于 `tol` 时停止 |
| `kkt` | KKT 条件的违反程度低于设定阈值时停止 |

数值收敛只表示声明的优化问题被求解到相应精度，并不构成另一种统计近似模型。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | `1.0` | 总体正则化强度 |
| `l1_ratio` | `0.5` | L1 惩罚占比：0=纯 L2 目标，1=Lasso 目标 |
| `fit_intercept` | `True` | 拟合不受惩罚的截距 |
| `max_iter` | `1000` | 最大求解迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | `"coef_delta"` 或 `"kkt"` 停止准则 |
| `device` | `"auto"` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"` |
| `n_jobs` | `None` | 适用 CPU 路径的并行度 |
| `solver` | `"fista"` | 与后端无关的直接拟合优化方法 |
| `cpu_solver` | `"fista"` | **弃用兼容参数**；新代码请使用 `solver` |
| `lipschitz_L` | `None` | 可选的用户指定 Lipschitz 常数 |
| `gpu_memory_cleanup` | `False` | 在支持的后端上于拟合后释放内存池 |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"post_selection_ols"` 或 `"bootstrap"`；`cpu_ols` / `gpu_ols` 暂时作为弃用别名接受 |
| `nodewise_alpha` | `None` | 纠偏推断中逐节点 Lasso 的惩罚强度；可显式指定正数；省略时，根据标准化后的设计矩阵自动确定 |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |

公开接口不单独提供 `backend`、`warm_start` 或 `random_state` 构造参数。计算后端由 `device` 控制；如需为一次拟合提供热启动，可使用 `fit(initial_coef=...)`。

## 补充 CPU/GPU 示例

先运行前面的数据准备；这里复用 `X` 与 `y`。推断示例与预测调参分开使用。

```python
from statgpu.linear_model import ElasticNet

# CPU：solver 选择算法，device 选择执行后端。
model_cpu = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cpu",
    solver="fista",
)
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# 显式逐节点调参只改变纠偏推断，不改变主 Elastic Net 拟合。
model_db = ElasticNet(
    alpha=0.1,
    l1_ratio=0.7,
    nodewise_alpha=0.08,
    device="cpu",
    compute_inference=True,
    inference_method="debiased",
)
model_db.fit(X, y)
print(model_db.nodewise_alpha_)

# GPU 仍使用同一个 solver 接口
model_gpu_cupy = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    gpu_memory_cleanup=True,
)
model_gpu_cupy.fit(X, y)

model_gpu_torch = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="torch",
    solver="fista",
)
model_gpu_torch.fit(X, y)
```

后端性能取决于样本量、特征维数、数据类型（dtype）、硬件、数据驻留位置与传输成本。应针对实际工作负载进行基准测试。

## 协方差/推断

`ElasticNet` 默认仅进行估计。设置 `compute_inference=True` 后，通过共享的惩罚线性模型推断框架执行拟合后推断。默认 `inference_method="debiased"` 与稀疏高斯 Lasso 路径使用同一套标准化逐节点 Lasso 一步纠偏构造，用于构造近似精度矩阵，并计算纠偏系数、标准误、z 统计量、p 值和置信区间。其统计有效性仍依赖设计、稀疏性、正则化尺度和模型假设；纠偏 Lasso 文献提供主要理论背景，但不等于对任意 `l1_ratio` 都自动给出无条件保证。推断成功后可调用 `summary()`。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `compute_inference` | `False` | 启用拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"post_selection_ols"` 或 `"bootstrap"` |
| `nodewise_alpha` | `None` | `debiased` 的逐节点精度矩阵调参；可显式指定正数；省略时，根据标准化后的设计矩阵自动确定 |
| `cov_type` | `"nonrobust"` | 在相应推断方法中使用的协方差约定 |
| `hac_maxlags` | `None` | 所选方法支持 HAC 时使用的滞后阶数 |

`nodewise_alpha` 与主模型 `alpha` 完全不同：它只影响纠偏推断中的近似精度矩阵，不改变惩罚预测拟合。省略时，statgpu 对已经完成中心化/加权处理的工作设计矩阵进行标准化，并采用

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重下使用 Kish 型有效样本量。该规则不依赖响应变量尺度，因此只改变 `y` 的计量单位不会改变基于设计矩阵的精度矩阵估计。成功的多特征纠偏推断通过 `nodewise_alpha_` 暴露解析出的实际值，并在 `_inference_result.metadata` 中记录调参来源、有效样本量和 KKT 残差。单特征问题直接使用解析精度矩阵，不需要逐节点惩罚参数。

`post_selection_ols` 是与硬件无关的规范活跃集诊断。统一封装类中历史 `cpu_ols` 与 `gpu_ols` 同时进入弃用期，一个兼容周期内仍接受并发出 `FutureWarning`，随后映射到 `post_selection_ols`；它们不负责选择设备。

`post_selection_ols` 先根据惩罚模型拟合得到的系数确定活跃集，再在本次拟合实际使用的后端上，只对该活跃集执行无惩罚 OLS；传入 `sample_weight` 时则执行 WLS。原始 `coef_` 不变，仍用于预测；活跃集重拟合结果通过 `_params`、`_inference_result` 等字段用于推断和报告。

选择后 OLS 仍属于启发式诊断，不提供一般意义上的选择性推断覆盖保证。其推断以已经选定的正则化参数为条件，也不会改变原始惩罚拟合系数。

设备选择与统计方法正交：显式 `cpu` / `cuda` / `torch` 始终以用户请求为准；只有真正的 `device="auto"` 才允许后端原生的 CuPy 或 Torch-CUDA 输入参与自动路由。`post_selection_ols` 复用拟合解析出的后端；CuPy/Torch 的 `debiased` 推断也会把数值推断留在实际执行的 GPU 后端，包括正态参考分布的标量临界值。残差 `bootstrap` 当前仍是 CPU 原生的残差重拟合路径；显式 GPU `device` 会控制惩罚拟合，但不会让 bootstrap 变成 GPU 原生。

对于带截距的 `debiased` 推断，公开 `coef_`/`intercept_` 继续属于 **惩罚预测拟合**。推断/报告使用纠偏（debiased）斜率 `_params[1:]`，以及与它们配套的原始坐标系截距 `_params[0] = ybar_w - xbar_w @ _params[1:]`；因此第一行标准误/z 值/p 值/置信区间（SE/z/p-value/CI）描述的是该纠偏报告截距，而不是预测 `intercept_`。结果元数据会记录 `intercept_estimator="centered_debiased"` 与 `intercept_influence="centered_nodewise"`。分析权重在 NumPy/CuPy/Torch 上使用同一个加权中心化平均损失问题，因此整体乘以正常数不会改变这套推断。

对于 `ElasticNetCV`，`compute_inference=True` 仅作用于 alpha 与 `l1_ratio` 选定后的最终全数据重拟合；各折模型仍仅用于估计和评分。`nodewise_alpha` 也只属于最终重拟合的推断配置，不进入候选网格或折内评分；推断成功时，外层 `nodewise_alpha_` 与最终 `estimator_` 一致。当前 `ElasticNetCV` 仍固定最终推断方法为 `debiased`，这是当前推断方法选择的限制。

## 求解器与推断语义

对于直接 `ElasticNet.fit`，**CPU 与 GPU 都使用 `solver`**。`device` 决定执行后端，`solver` 决定优化算法。`cpu_solver` 是早期按硬件区分求解器 API 的弃用兼容参数，新代码不应继续使用。

同样，需要活跃集 OLS/WLS 诊断时应使用 `inference_method="post_selection_ols"`，而不是根据硬件去选 `cpu_ols` 或 `gpu_ols`；后两者只是同一个统计方法的弃用别名。

`compute_inference=False` 时只返回惩罚估计结果；开启推断后，主模型拟合系数保持不变，再执行所选的拟合后推断方法。

## 输出属性

拟合后可用以下属性：

| 属性 | 说明 |
|------|------|
| `coef_` | 用于预测的惩罚拟合系数 |
| `intercept_` | 用于预测的惩罚拟合截距 |
| `n_iter_` | 收敛所需迭代次数 |
| `nodewise_alpha_` | 多特征 `debiased` 推断成功后解析出的逐节点调参值；其他情况为 `None` |
| `_params` | 推断成功时用于报告的参数向量；`debiased` 下包含相互一致的纠偏截距与纠偏斜率；`post_selection_ols` 下保存按完整参数布局嵌入的活跃集 OLS/WLS 重拟合结果 |
| `_inference_result` | 结构化推断结果，以及数值后端和逐节点调参的元数据 |
| `aic` | 可用时的兼容性代入式拟合诊断；不是考虑惩罚项有效自由度后的信息准则 |
| `bic` | 可用时的兼容性代入式拟合诊断；不是考虑惩罚项有效自由度后的信息准则 |

方法：`fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## 常见误区与完整 API

- `summary()` 要求推断成功；默认 `compute_inference=False` 时，应查看系数、预测与评分结果。
- 选择后 OLS 是在入选变量上的诊断，并不普遍修正先搜索变量带来的不确定性。纠偏报告系数可能不同于预测系数，详见前面的推断说明。
- 达到 `max_iter` 可能表示数值精度尚不足。先检查警告、迭代次数和特征尺度，再比较更严格容差或更大迭代次数下的结果。
- 显式 `device="cuda"` 或 `device="torch"` 要求对应的 GPU 后端可用，不会静默转到 CPU。详见[设备与内存](../guides/device-and-memory.md)。
- 改用其他求解器前先核对[求解器与惩罚兼容矩阵](../guides/solver-penalty-matrix.md)，`device` 不能替代 `solver`。

上面的构造参数表覆盖 `ElasticNet` 的完整构造接口。完整公开类、方法签名与说明见 [ElasticNet API 源码](../../../statgpu/linear_model/wrappers/_elasticnet.py)，也可通过 `help(ElasticNet)` 查看已安装版本。继承的 `get_params` / `set_params` 用于估计器配置。交叉验证请查阅独立的 [ElasticNetCV API 源码](../../../statgpu/linear_model/cv/_elasticnet_cv.py)，不要假定其参数与直接拟合相同。

## 数值验证

支持的后端会在相同模型设定下检查数值一致性，并与参考实现进行对照。求解器参数迁移、逐节点调参以及选择后 OLS/WLS 的行为也都有回归测试覆盖。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.

