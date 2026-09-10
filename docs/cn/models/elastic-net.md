# Elastic Net 弹性网络

> Language: Chinese (中文)  
> Last updated: 2026-09-10<br>
> This page: 模型文档  
> Language switch: [English](../../en/models/elastic-net.md)

## 概述

`ElasticNet` 结合 L1 与 L2 正则化，在稀疏特征选择（Lasso）和系数收缩（Ridge）之间取得平衡。支持 CPU、CuPy GPU 与 PyTorch GPU。直接拟合统一使用**与后端无关的 `solver` 接口**；`device` 只负责决定在哪里执行。

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

**正则化缩放说明**：`ElasticNet` 与 `Ridge` 使用同一平均损失约定，因此 `l1_ratio=0` 时，相同公开 `alpha` 下目标函数退化为对应的 L2 目标。不过 `ElasticNet` wrapper 仍保留自己的求解器和推断默认设置；若明确需要 Ridge 的估计器契约，应直接使用 `Ridge`。

## 估计方程

消去未惩罚截距（等价地，在中心化数据上）后，系数满足 KKT 条件：

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0.
$$

对**直接单次拟合**，`solver` 是所有后端上的权威算法选择器。`device` 单独控制 CPU/CuPy/Torch 执行位置。历史 `cpu_solver` 参数已进入弃用流程，在统一引擎中不再代表第二套 CPU direct-fit solver。参见 [penalized solver API 迁移指南](../guides/penalized-solver-api-migration.md)。

## 估计算法

默认路径使用 **FISTA**（快速迭代收缩阈值算法），即带 Nesterov 加速的近端梯度法。其他求解器是否可用取决于公开 solver compatibility contract。

### 关键优化洞察

Elastic Net 的 L1/L2 部分由近端算子共同处理：

```python
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
| `kkt` | KKT 次梯度违反低于配置阈值时停止 |

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
| `solver` | `"fista"` | 与后端无关的 direct-fit 优化方法 |
| `cpu_solver` | `"fista"` | **弃用兼容参数**；新代码请使用 `solver` |
| `lipschitz_L` | `None` | 可选的用户指定 Lipschitz 常数 |
| `gpu_memory_cleanup` | `False` | 在支持的后端上于拟合后释放内存池 |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"post_selection_ols"` 或 `"bootstrap"`；`cpu_ols` / `gpu_ols` 暂时作为弃用别名接受 |
| `nodewise_alpha` | `None` | 纠偏推断中逐节点 Lasso 的惩罚强度；显式正标量优先于标准化设计侧自动规则 |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |

公开 wrapper 不接受单独的 `backend`、`warm_start` 或 `random_state` 构造参数。后端由 `device` 控制；单次热启动可通过 `fit(initial_coef=...)` 提供。

## CPU/GPU 示例

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

后端性能取决于样本量、特征维数、dtype、硬件、数据驻留位置与传输成本。应针对实际工作负载进行基准测试。

## 协方差/推断

`ElasticNet` 默认仅进行估计。设置 `compute_inference=True` 后，通过共享 penalized-linear inference engine 执行拟合后推断。默认 `inference_method="debiased"` 与稀疏 Gaussian Lasso 路径复用同一套标准化逐节点 Lasso 一步纠偏构造，用于建立近似精度矩阵、纠偏系数、标准误、z 统计量、p 值与置信区间。其统计有效性仍依赖设计、稀疏性、正则化尺度和模型假设；纠偏 Lasso 文献提供主要理论背景，但不等于对任意 `l1_ratio` 都自动给出无条件保证。推断成功后可调用 `summary()`。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `compute_inference` | `False` | 启用拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"post_selection_ols"` 或 `"bootstrap"` |
| `nodewise_alpha` | `None` | `debiased` 的逐节点精度矩阵调参；可显式给正标量，或使用标准化设计侧自动规则 |
| `cov_type` | `"nonrobust"` | 在相应推断方法中使用的协方差约定 |
| `hac_maxlags` | `None` | 所选方法支持 HAC 时使用的滞后阶数 |

`nodewise_alpha` 与主模型 `alpha` 完全不同：它只影响纠偏推断中的近似精度矩阵，不改变惩罚预测拟合。省略时，statgpu 对已经完成中心化/加权处理的工作设计进行标准化，并采用

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重下使用 Kish 型有效样本量。该规则不依赖响应变量尺度，因此只改变 `y` 的计量单位不会改变设计侧精度矩阵问题。成功的多特征纠偏推断通过 `nodewise_alpha_` 暴露解析出的实际值，并在 `_inference_result.metadata` 中记录调参来源、有效样本量和 KKT 证据。单特征问题使用解析精度矩阵，不消费逐节点惩罚。

`post_selection_ols` 是与硬件无关的规范活跃集诊断。统一 wrapper 中历史 `cpu_ols` 与 `gpu_ols` 同时进入弃用期，一个兼容周期内仍接受并发出 `FutureWarning`，随后映射到 `post_selection_ols`；它们不负责选择设备。

`post_selection_ols` 会先使用 penalized model 的 fitted coefficients 确定活跃集，再在成功拟合所记录的后端上，只对该活跃集做无惩罚 OLS；传入 `sample_weight` 时做 WLS。原始 penalized `coef_` 保持不变并继续用于预测，活跃集重拟合通过 `_params` / `_inference_result` 等字段参与推断与报告。

选择后 OLS 仍是启发式诊断，不提供一般选择性推断覆盖保证。推断条件于已选择的正则化参数，并不会改变 penalized coefficients。

设备选择与统计方法正交：显式 `cpu` / `cuda` / `torch` 始终具有权威性；只有真正的 `device="auto"` 才允许 backend-native CuPy 或 Torch-CUDA 输入参与自动路由。`post_selection_ols` 复用 fit-resolved backend；维护中的 CuPy/Torch `debiased` 路径也会把数值推断留在实际执行的 GPU backend，包括 normal-reference 的 scalar critical value。残差 `bootstrap` 当前仍是 CPU-native residual-refit 路径；显式 GPU `device` 会控制 penalized fit，但不会让 bootstrap 变成 GPU-native。

对于带截距的 `debiased` inference，公开 `coef_`/`intercept_` 继续属于 **penalized prediction fit**。推断/reporting 使用 debiased slopes `_params[1:]`，以及与它们配套的原始坐标系截距 `_params[0] = ybar_w - xbar_w @ _params[1:]`；因此第一行 SE/z/p-value/CI 描述的是该 debiased reporting intercept，而不是 prediction `intercept_`。result metadata 会记录 `intercept_estimator="centered_debiased"` 与 `intercept_influence="centered_nodewise"`。分析权重在 NumPy/CuPy/Torch 上使用同一个加权中心化平均损失问题，因此整体乘以正常数不会改变这套推断。

对于 `ElasticNetCV`，`compute_inference=True` 仅作用于 alpha 与 `l1_ratio` 选定后的最终全数据重拟合；各折模型仍仅用于估计和评分。`nodewise_alpha` 也只属于最终重拟合的推断配置，不进入候选网格或折内评分；推断成功时，外层 `nodewise_alpha_` 与最终 `estimator_` 一致。当前 `ElasticNetCV` 仍固定最终推断方法为 `debiased`，这是既有的 inference-selector 限制，与本次逐节点调参修复分开处理。

## 求解器与推断语义

对于直接 `ElasticNet.fit`，**CPU 与 GPU 都使用 `solver`**。`device` 决定执行后端，`solver` 决定优化算法。`cpu_solver` 是早期 hardware-split API 的弃用兼容参数，新代码不应继续使用。

同样，需要活跃集 OLS/WLS 诊断时应使用 `inference_method="post_selection_ols"`，而不是根据硬件去选 `cpu_ols` 或 `gpu_ols`；后两者只是同一个统计方法的弃用别名。

`compute_inference=False` 只返回 penalized estimate；开启推断后保留同一拟合系数，再运行所选 post-fit inference method。

## 输出属性

拟合后可用以下属性：

| 属性 | 说明 |
|------|------|
| `coef_` | 用于预测的 penalized coefficients |
| `intercept_` | 用于预测的 penalized fitted intercept |
| `n_iter_` | 收敛所需迭代次数 |
| `nodewise_alpha_` | 多特征 `debiased` 推断成功后解析出的逐节点调参值；其他情况为 `None` |
| `_params` | 推断成功时的 reporting 参数向量；`debiased` 下包含 coherent debiased intercept 与 debiased slopes；`post_selection_ols` 下是嵌入完整参数布局的 active-set OLS/WLS 重拟合 |
| `_inference_result` | structured inference result，以及数值后端与逐节点调参 metadata |
| `aic` | 可用时的兼容性 plug-in 拟合诊断；不是 penalty-aware 有效自由度准则 |
| `bic` | 可用时的兼容性 plug-in 拟合诊断；不是 penalty-aware 有效自由度准则 |

方法：`fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## 数值验证

维护中的回归测试会按 dtype 与 solver path 检查支持后端之间及与参考实现的数值一致性。solver API 迁移行为由 `dev/tests/test_penalized_solver_api_cleanup.py` 覆盖；逐节点调参契约由 `dev/tests/test_nodewise_alpha_inference_contract.py` 覆盖；post-selection OLS API 迁移与 active-set OLS/WLS 行为由 `dev/tests/test_post_selection_ols_inference_api.py` 覆盖。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
