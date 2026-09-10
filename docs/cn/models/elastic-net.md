# Elastic Net 弹性网络

> 语言：中文  
> 最后更新：2026-09-10  
> 切换：[English](../../en/models/elastic-net.md)

## 它解决什么问题？

`ElasticNet` 把 L1 的稀疏性和 L2 的稳定化结合起来。当你希望一部分系数可以精确变成 0，同时重要预测变量又具有较强相关性、纯 Lasso 的选择可能不稳定时，它尤其有用。

可以把 `l1_ratio` 理解成一个目标函数层面的连续谱：

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

这里只是说目标函数。`ElasticNet(l1_ratio=0)` 的惩罚退化成纯 L2，但 `ElasticNet` wrapper 仍保留自己的 solver/default/inference contract；如果需要专门的 Ridge estimator surface，应直接使用 `Ridge`。

## 模型与目标函数

带未惩罚截距 $b$，并令 $\lambda=$ `l1_ratio`，statgpu 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top\beta)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2.
$$

`alpha` 控制总正则强度，`l1_ratio` 控制其中 L1 所占比例。除非原始尺度就是建模约定的一部分，连续变量通常应在正则化前标准化。

## 最小示例

```python
import numpy as np
from statgpu.linear_model import ElasticNet

rng = np.random.default_rng(2)
X = rng.normal(size=(500, 12))
beta = np.zeros(12)
beta[[1, 2, 7, 8]] = [1.2, 1.0, -0.9, -0.8]
y = 0.5 + X @ beta + rng.normal(scale=0.8, size=500)

model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print(model.coef_)
print(model.score(X, y))
```

当 `l1_ratio>0` 时可以出现精确 0。非零系数仍是 penalized estimates，而不是普通 OLS 效应估计。

## 关键参数

| 参数 | 默认值 | 如何理解 |
|---|---:|---|
| `alpha` | `1.0` | 总正则强度。 |
| `l1_ratio` | `0.5` | L1 比例；接近 1 更像 Lasso，接近 0 更像 Ridge。 |
| `device` | `"auto"` | CPU/CuPy/Torch 执行位置；显式不可用 GPU 会明确失败。 |
| `solver` | `"fista"` | direct fit 的权威数值求解器。 |
| `compute_inference` | `False` | 只有确实需要受支持的 post-fit inference 时再开启。 |
| `nodewise_alpha` | `None` | 仅用于 `debiased` inference 的逐节点精度矩阵构造调参。 |

预测用途通常应使用 `ElasticNetCV` 联合调 `alpha` 与 `l1_ratio`，而不是根据训练拟合指标选择。

## 与 Ridge 和 Lasso 比较

| 性质 | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| 平滑收缩 | 是 | 是 | 是 |
| 精确 0 | 通常否 | 是 | `l1_ratio>0` 时是 |
| 高相关预测变量 | 稳定 | 可能任意选择其中一个 | 更适合成组变量 |
| 主要调参 | `alpha` | `alpha` | `alpha` + `l1_ratio` |

## CPU、GPU、Formula、权重与热启动

```python
model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    compute_inference=False,
).fit(X, y)
```

公开 estimator 在相应环境中支持 NumPy CPU、CuPy CUDA 与 Torch CUDA。`fit()` 支持 `sample_weight=`，并转发共享 `formula=` / `data=` 接口。单次拟合可通过 `initial_coef=` 提供热启动。

## 进阶：优化

在消去未惩罚截距后的中心化系数问题上，KKT 条件为

$$
\frac{1}{n}X_c^\top(X_c\hat\beta-y_c)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1=0.
$$

对直接 `ElasticNet.fit`，`solver` 在所有后端上具有权威性。历史 `cpu_solver` 只是兼容状态，不是第二个 direct-fit solver selector。

## 进阶：推断

`ElasticNet` 默认只估计。`compute_inference=True` 时，post-fit inference 不会改变用于预测的 penalized `coef_` / `intercept_`。

| `inference_method` | 用途 | 主要限制 |
|---|---|---|
| `debiased` | 使用共享逐节点精度矩阵引擎做一步纠偏系数推断 | 软件复用不等于任意 `l1_ratio` 自动继承所有 Lasso 定理；有效性仍取决于初始 estimator 与具体假设 |
| `post_selection_ols` | 在 fit-resolved NumPy/CuPy/Torch backend 上做活跃集 OLS/WLS 诊断 | 不是一般 selective-inference 保证 |
| `bootstrap` | 残差重采样替代路径 | 计算更重且依赖相应假设 |

历史 `cpu_ols` / `gpu_ols` 是硬件无关 `post_selection_ols` 的 deprecated alias；设备由 `device` 单独选择。

### `debiased` 推断中的逐节点调参

`nodewise_alpha` 与主 Elastic Net `alpha` 相互独立。它只改变用于近似设计精度矩阵的逐节点 Lasso；不能改变 penalized prediction fit。

显式有限正值具有最高优先级。`nodewise_alpha=None` 且 $p\ge2$ 时，statgpu 标准化规范的中心化/加权工作设计并解析

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重下使用 Kish 型有效样本量。具体常数和 effective-n 约定是 statgpu 默认选择，并非某个定理唯一规定的形式。

这个规则与响应变量尺度无关，有意替代历史内部的 response-residual-scaled 规则。标准化逐节点解必须通过独立 KKT publication gate，随后才把精度矩阵变换回工作特征尺度。`p=1` 时使用解析一维精度矩阵，不消费 `nodewise_alpha`。

NumPy、CuPy 与 Torch 采用同一套维护中的统计定义；显式 CUDA/Torch 数值推断不会静默回退 CPU。成功的多特征 debiased inference 会发布 `nodewise_alpha_`，并在 `_inference_result.metadata` 中记录详细 provenance。

对 `ElasticNetCV`，`nodewise_alpha` 只属于最终全数据重拟合的 inference config，不进入 `alpha` / `l1_ratio` 网格、fold score 或调参选择。目前 CV 后推断仍条件于已选调参值。参见 [逐节点调参迁移说明](../guides/nodewise-alpha-migration.md)。

## 常见误区

- 预测用途应同时考虑 `alpha` 与 `l1_ratio`。
- 不要认为 `l1_ratio=0` 会让整个 wrapper 等同于 `Ridge`。
- 稳定的 active set 也不等价于因果识别。
- 不要忽略特征尺度。
- 不要把训练 $R^2$ 或兼容 AIC/BIC/F 字段当成 penalty-aware 调参准则。
- 不要混淆主 `alpha` 与 inference-only `nodewise_alpha`。
- `post_selection_ols` 是诊断，不是自动的选择性推断。

## 完整 API 参考

运行时公开 constructor 是静态 wrapper constructor 加上维护中的 node-wise inference contract 注入的 `nodewise_alpha=None`：

```python
ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    device="auto",
    n_jobs=None,
    solver="fista",
    cpu_solver="fista",
    lipschitz_L=None,
    gpu_memory_cleanup=False,
    compute_inference=False,
    inference_method="debiased",
    cov_type="nonrobust",
    hac_maxlags=None,
    nodewise_alpha=None,
)
```

下面带 marker 的表继续作为本 Draft 的 source-only checker 所校验的**静态 wrapper 参数清单**；runtime 注入的公开扩展紧随其后单独列出。

<!-- API-CONSTRUCTOR-START:ElasticNet -->
| 参数 | 默认值 | 参考含义 |
|---|---:|---|
| `alpha` | `1.0` | 总正则强度。 |
| `l1_ratio` | `0.5` | L1 惩罚比例。 |
| `fit_intercept` | `True` | 是否拟合未惩罚截距。 |
| `max_iter` | `1000` | 最大求解迭代次数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | `coef_delta` 或 `kkt`。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | `None` | 适用路径的并行提示。 |
| `solver` | `"fista"` | backend-neutral direct-fit solver。 |
| `cpu_solver` | `"fista"` | legacy/shared 兼容控制；不是 direct-fit 的权威选择器。 |
| `lipschitz_L` | `None` | 可选预计算 Lipschitz 常数。 |
| `gpu_memory_cleanup` | `False` | 拟合后的 best-effort GPU cache 清理。 |
| `compute_inference` | `False` | 是否执行所选 post-fit inference。 |
| `inference_method` | `"debiased"` | `debiased`、规范 `post_selection_ols` 或 `bootstrap`；`cpu_ols` / `gpu_ols` 为 deprecated alias。 |
| `cov_type` | `"nonrobust"` | 适用推断路径的协方差约定。 |
| `hac_maxlags` | `None` | 支持 HAC 时的 lag 数。 |
<!-- API-CONSTRUCTOR-END:ElasticNet -->

**Runtime 注入的公开扩展：** `nodewise_alpha=None` —— `None` 使用标准化设计侧自动规则；有限正实数显式指定逐节点惩罚。成功多特征 debiased inference 会在 `nodewise_alpha_` 中发布解析值。

### `fit` 与重要输出

`fit(X=None, y=None, sample_weight=None, initial_coef=None, **kwargs)` 返回 `self`；共享转发关键字包括 `formula` 与 `data`。

| 属性 | 含义 |
|---|---|
| `coef_`, `intercept_` | penalized prediction fit |
| `n_iter_` | 数值迭代次数 |
| `nodewise_alpha_` | 成功多特征 debiased inference 后的逐节点调参值；其他情况为 `None` |
| `_params`, `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | 推断成功后的 reporting 数组 |
| `_inference_result` | 带逐节点和 backend provenance 的结构化结果 |

`predict`、`score`、`summary`、`get_params`、`set_params` 遵循共享 estimator contract。通过 `set_params` 修改 `nodewise_alpha` 会使旧 inference state 失效，同时 clone/introspection 保留用户请求值。

## 验证

维护中的覆盖检查 Elastic Net 目标函数、solver/KKT、direct fit 对 node-wise 调参的不变性、`ElasticNetCV` final-refit isolation、weighted inference、formula/public API、NumPy/CuPy/Torch precision parity，以及共享 node-wise implementation 的物理 CUDA acceptance。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *JRSS B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
