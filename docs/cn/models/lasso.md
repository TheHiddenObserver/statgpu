# Lasso 回归

> 语言：中文  
> 最后更新：2026-09-10  
> 切换：[English](../../en/models/lasso.md)

## 它解决什么问题？

`Lasso` 是带 L1 惩罚的线性回归。它不仅会收缩系数，还能把较弱的系数直接压到 0，因此同一次拟合可以同时用于预测和稀疏特征选择。

当你相信大量候选变量中只有一小部分真正携带主要信号时，Lasso 很合适。如果多数变量都有小但真实的作用，[Ridge](ridge.md) 往往更稳定；如果重要变量成组高度相关，[Elastic Net](elastic-net.md) 通常更合适。

## 直觉

Lasso 在“拟合数据”和“为系数绝对值付出代价”之间折中。L1 惩罚在 0 处有尖角，因此较弱的更新会通过 soft-thresholding 被直接压到 0，而不仅是变小。

```text
OLS   ：不做正则化
Ridge ：平滑收缩所有系数
Lasso ：收缩，并删除较弱系数
```

稀疏性很有用，但选择结果依赖数据和调参。某个系数在本次拟合中为 0，并不等价于证明其总体真实效应严格为 0。

## 模型与目标函数

带未惩罚截距 $b$ 时，statgpu 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top\beta)^2
+\alpha\lVert\beta\rVert_1.
$$

`alpha` 越大，收缩越强，通常会出现更多 0。由于 L1 直接作用于系数尺度，连续预测变量在正则化前通常应处在可比尺度上。

## 最小可运行示例

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 12))
true_coef = np.zeros(12)
true_coef[[1, 5, 9]] = [2.0, -1.5, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print(model.coef_)
print(np.flatnonzero(np.abs(model.coef_) > 1e-8))
print(model.score(X, y))
```

`coef_` 与 `intercept_` 始终属于 penalized prediction fit。非零系数仍然经过收缩，不能直接解释为普通 OLS 系数。

## 关键参数

| 参数 | 默认值 | 如何理解 |
|---|---:|---|
| `alpha` | `1.0` | 主预测/选择调参。预测用途应优先用验证或 `LassoCV`，而不是训练拟合优度来选择。 |
| `fit_intercept` | `True` | 除非理论上确定无截距或设计矩阵已显式编码截距，通常保留。 |
| `device` | `"auto"` | 选择 CPU、CuPy CUDA、Torch CUDA 或自动路由；显式 GPU 请求不可用时会报错，不静默回退 CPU。 |
| `solver` | `"fista"` | 所有后端上 direct fit 的权威数值求解器。 |
| `stopping` | `"coef_delta"` | 需要基于最优性条件判断收敛时可用 `kkt`。 |
| `compute_inference` | `True` | 只做预测/选择时可关闭。 |
| `inference_method` | `"debiased"` | 选择拟合后的统计推断程序，与 `device` 独立。 |
| `nodewise_alpha` | `None` | 只用于 `debiased` 推断内部逐节点精度矩阵问题的独立调参。 |

## CPU、GPU、Formula 与权重

```python
model = Lasso(
    alpha=0.08,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=False,
).fit(X, y)
```

显式 `device="cuda"` 使用 CuPy CUDA，显式 `device="torch"` 使用 Torch CUDA；不可用时明确失败。`fit()` 也支持 `sample_weight=` 以及共享 `formula=` / `data=` 接口。

分析权重遵循维护中的平均损失约定。把所有权重同时乘以同一个正数，不改变预期的 weighted sparse-Gaussian 统计问题。

## 与相邻方法比较

| 方法 | 精确 0？ | 高相关预测变量 | 典型用途 |
|---|:---:|---|---|
| OLS / `LinearRegression` | 否 | 可能不稳定 | 无惩罚估计 |
| [Ridge](ridge.md) | 否 | 稳定性强 | 不需要删变量的预测 |
| **Lasso** | 是 | 可能只选一组中的一个 | 稀疏预测 / 特征选择 |
| [Elastic Net](elastic-net.md) | L1 比例 > 0 时是 | 更适合成组变量 | 高相关特征下的稀疏模型 |

## 进阶：求解器

对直接 `Lasso.fit`，`solver` 选择算法，`device` 选择执行后端。历史 `cpu_solver` 仅为兼容保留，不再替代直接拟合的 `solver`。

| `solver` | CPU | CuPy / Torch | 说明 |
|---|:---:|:---:|---|
| `fista` | 是 | 是 | 默认近端梯度路径 |
| `auto` | 是 | 是 | 当前 Gaussian + L1 自动路由 |
| `fista_bb` | 是 | 是 | spectral-step 变体 |
| `admm` | 是 | 是 | split solver；权重存在额外限制 |
| `coordinate_descent` | 是 | 否 | CPU-only direct-fit 路径 |

## 进阶：Lasso 拟合后的推断

数据驱动稀疏选择后的推断不是普通 fixed-model OLS 推断。statgpu 提供几种统计含义不同的路径：

| `inference_method` | 做什么 | 主要限制 |
|---|---|---|
| `debiased` | 一步纠偏 / de-sparsified 系数推断 | 有效性依赖高维稀疏性、设计、噪声与调参假设 |
| `post_selection_ols` | 在 fit-resolved backend 上对活跃集做 OLS/WLS 重拟合 | 属于选择后诊断，不是一般 selective-inference 保证 |
| `bootstrap` | 残差自助法重拟合 | 计算更重，也不是普适的选择不确定性修正 |

`post_selection_ols` 是与硬件无关的规范名称。历史 `cpu_ols` 与 `gpu_ols` 是弃用别名，会映射到同一个统计方法；它们不负责选择设备。

`LassoCV(compute_inference=True)` 会先完成主 `alpha` 选择，再只在最终全数据重拟合上进行推断。目前的推断条件于已选调参值，并不会额外修正 CV 调参不确定性。

### `alpha` 与 `nodewise_alpha`

主 `alpha` 控制 penalized prediction/selection fit；`nodewise_alpha` 只控制 `inference_method="debiased"` 内部用于近似设计精度矩阵的逐节点 Lasso（node-wise Lasso）。只改变 `nodewise_alpha` 不应改变 penalized `coef_`，也不应改变 `LassoCV` 的 alpha grid、fold score 或最终 `alpha_`。

显式有限正 `nodewise_alpha` 具有最高优先级。`nodewise_alpha=None` 且 $p\ge2$ 时，statgpu 先标准化规范的中心化/加权工作设计，再使用

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重下使用 Kish 型有效样本量。$\sqrt{\log(p)/n}$ 的量级具有理论动机，但具体常数和加权有效样本量约定属于 statgpu 默认选择。

这个新规则有意做到**与响应变量尺度无关**。历史内部实现曾把逐节点惩罚乘以响应残差尺度估计；该规则已经被取代，并且不作为 legacy public mode 暴露。

逐节点求解在标准化设计上进行，发布结果前必须通过独立完整 KKT gate，然后再把精度矩阵变换回原工作特征尺度。`p=1` 时不存在 nuisance node-wise regression，statgpu 使用一维解析精度矩阵，并让 `nodewise_alpha_` 保持为 `None`。

对 `LassoCV`，`nodewise_alpha` 只属于最终重拟合推断配置。更完整的统计构造见 [Lasso 推断](lasso-inference.md)，迁移细节见 [逐节点调参迁移说明](../guides/nodewise-alpha-migration.md)。

### 后端与 reporting boundary

NumPy、CuPy 与 Torch 使用同一个维护中的逐节点统计定义。显式 CUDA/Torch 的 debiased 数值推断不会静默替换成 CPU 实现；只有在后端原生数值推断完成后，小型 reporting 数组才转成 NumPy，metadata 会记录实际 numerical backend/device。

`fit_intercept=True` 时，debiased reporting 使用 coherent centered parameterization。`coef_` / `intercept_` 继续属于 penalized prediction fit，而 `_params` 属于纠偏后的 reporting 参数。可选的截距同时推断使用 max-|Z| multiplier bootstrap，并把 `_conf_int_simultaneous` 与边际 `_conf_int` 分开报告。

## 常见误区

- 不要把“被 Lasso 选中”理解为因果证据或总体效应必然非零。
- 不要忽略 L1 正则化前的特征尺度。
- 不要用训练 $R^2$ 调 `alpha`。
- 不要把 `post_selection_ols` 当作一般选择性推断。
- 不要认为 `LassoCV` 自动修正了调参不确定性。
- 不要混淆 `alpha` 与 `nodewise_alpha`：后者仅用于推断。
- 数值 KKT residual 很小，只说明所声明的优化问题被精确求解，不等于高维推断假设已经成立。

## 完整 API 参考

运行时公开 constructor 是静态 wrapper constructor 加上由维护中的 node-wise inference compatibility contract 注入的 `nodewise_alpha=None`：

```python
Lasso(
    alpha=1.0,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    inference_method="debiased",
    n_bootstrap=200,
    bootstrap_random_state=None,
    enable_simultaneous_inference=False,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=None,
    simultaneous_include_intercept=False,
    device="auto",
    n_jobs=None,
    compute_inference=True,
    solver="fista",
    cpu_solver="coordinate_descent",
    lipschitz_L=None,
    admm_rho=1.0,
    gpu_memory_cleanup=False,
    nodewise_alpha=None,
)
```

下面带 marker 的表继续作为本 Draft 的 source-only checker 所校验的**静态 wrapper 参数清单**；runtime 注入的公开扩展紧随其后单独列出。

<!-- API-CONSTRUCTOR-START:Lasso -->
| 参数 | 默认值 | 参考含义 |
|---|---:|---|
| `alpha` | `1.0` | L1 惩罚强度。 |
| `fit_intercept` | `True` | 是否拟合未惩罚截距。 |
| `max_iter` | `1000` | 最大求解迭代次数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | `coef_delta` 或 `kkt`。 |
| `inference_method` | `"debiased"` | `debiased`、规范 `post_selection_ols` 或 `bootstrap`；`cpu_ols` / `gpu_ols` 已弃用。 |
| `n_bootstrap` | `200` | 残差 bootstrap 次数。 |
| `bootstrap_random_state` | `None` | 残差 bootstrap 随机种子。 |
| `enable_simultaneous_inference` | `False` | 是否在 debiased 推断后启用 simultaneous max-|Z| 区间。 |
| `simultaneous_method` | `"maxz_bootstrap"` | simultaneous calibration 方法。 |
| `simultaneous_alpha` | `0.05` | family-wise error level。 |
| `simultaneous_n_bootstrap` | `1000` | multiplier-bootstrap 次数。 |
| `simultaneous_random_state` | `None` | simultaneous bootstrap 随机种子。 |
| `simultaneous_include_intercept` | `False` | 是否把 coherent debiased intercept 纳入 simultaneous target family。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | `None` | 适用路径中的并行提示。 |
| `compute_inference` | `True` | 是否执行所选 post-fit inference。 |
| `solver` | `"fista"` | backend-neutral direct-fit solver。 |
| `cpu_solver` | `"coordinate_descent"` | legacy/shared 兼容控制；不是 direct-fit 的权威选择器。 |
| `lipschitz_L` | `None` | 可选预计算 Lipschitz 常数。 |
| `admm_rho` | `1.0` | ADMM penalty 参数。 |
| `gpu_memory_cleanup` | `False` | 拟合后的 best-effort GPU cache 清理。 |
<!-- API-CONSTRUCTOR-END:Lasso -->

**Runtime 注入的公开扩展：** `nodewise_alpha=None` —— `None` 使用标准化设计侧自动规则；有限正实数显式指定逐节点惩罚。成功的多特征 debiased inference 会在 `nodewise_alpha_` 中发布解析值。

### `fit` 与核心方法

`fit(X=None, y=None, sample_weight=None, formula=None, data=None)` 返回 `self`。`predict(X, return_cpu=True)` 给出连续预测；`score(X, y, sample_weight=None)` 返回 $R^2$；`summary()` 报告可用推断。`get_params` / `set_params` 与 sklearn clone 会保留用户请求的 `nodewise_alpha`；修改它会使旧 fitted inference state 失效。

`adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` 等继承的推断工具见 [推断 API](../guides/inference-api.md)。

### 重要 fitted 字段

| 属性 | 含义 |
|---|---|
| `coef_`, `intercept_` | penalized prediction fit |
| `n_iter_` | 数值迭代次数 |
| `nodewise_alpha_` | 成功多特征 debiased inference 后的逐节点调参解析值；其他情况为 `None` |
| `_params`, `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | 推断成功后的 reporting 数组 |
| `_conf_int_simultaneous` | 显式启用并成功校准后的同时置信区间 |
| `_inference_result` | 包含逐节点与后端 provenance 的结构化推断结果 |

可用的 penalized-fit `rsquared_adj`、`fvalue`、`f_pvalue`、`aic`、`bic` 是采用普通参数计数/残差自由度约定的兼容/plugin 诊断，不是 selection-aware、tuning-aware 或 effective-DoF-aware 标准。

## 验证

维护中的覆盖包括 direct/API/clone/set-params contract、响应尺度不变性、特征尺度等变性、一维解析路径、KKT fail-closed、权重不变性、CV final-refit 隔离、formula parity、独立双特征解析参考、cache provenance、NumPy/CuPy/Torch parity，以及 accepted node-wise implementation 的物理 CUDA 验证。

## 参考文献

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
