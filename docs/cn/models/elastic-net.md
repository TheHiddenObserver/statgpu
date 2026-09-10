# Elastic Net 弹性网络

> 语言：中文  
> 最后更新：2026-09-10  
> 切换：[English](../../en/models/elastic-net.md)

> **版本提示：** 当前正式发布版为 **0.2.5**。本页涉及的公开参数 `nodewise_alpha` 以及新的自动逐节点调参规则已经进入当前 `master`，计划随 **0.2.6** 发布；正式版 0.2.5 尚不包含这一公开接口和新默认规则。

## 它解决什么问题？

`ElasticNet` 把 L1 惩罚带来的稀疏性与 L2 惩罚带来的稳定性结合起来。当你希望一部分系数可以精确变成 0，同时重要预测变量又高度相关、使纯 Lasso 的变量选择不够稳定时，Elastic Net 尤其有用。

可以把 `l1_ratio` 理解为一个目标函数层面的连续谱：

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

这里说的是**目标函数**。当 `l1_ratio=0` 时，L1 项消失，惩罚形式退化为纯 L2；但 `ElasticNet` 仍然保留自己的求解器默认值、参数检查和推断接口。如果需要完整的 Ridge 接口和行为，应直接使用 `Ridge`，而不是把 `ElasticNet(l1_ratio=0)` 当作 Ridge 的完全替代。

## 模型与目标函数

带不受惩罚的截距 $b$，并令 $\lambda=$ `l1_ratio`，statgpu 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top\beta)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2.
$$

`alpha` 控制总正则化强度，`l1_ratio` 控制其中 L1 惩罚所占比例。除非原始变量尺度本身就是建模约定的一部分，连续预测变量通常应在正则化前标准化。

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

当 `l1_ratio>0` 时，部分系数可以精确变成 0。非零系数仍是经过惩罚收缩的估计值，不能直接解释为普通 OLS 的无偏效应估计。

## 关键参数

| 参数 | 默认值 | 如何理解 |
|---|---:|---|
| `alpha` | `1.0` | 总正则化强度。值越大，整体收缩通常越强。 |
| `l1_ratio` | `0.5` | L1 惩罚所占比例；接近 1 更像 Lasso，接近 0 更像 Ridge。 |
| `device` | `"auto"` | 选择 CPU、CuPy CUDA、Torch CUDA 或自动路由；显式指定 GPU 而设备不可用时会直接报错。 |
| `solver` | `"fista"` | 直接拟合时实际选择数值算法的参数，对各后端采用统一含义。 |
| `compute_inference` | `False` | 只做预测或变量选择时保持关闭；需要受支持的拟合后推断时再开启。 |
| `nodewise_alpha` | `None` | 仅用于 `debiased` 纠偏推断中估计近似精度矩阵的逐节点 Lasso 调参。计划自 0.2.6 起公开。 |

用于预测时，通常应使用 `ElasticNetCV` 联合选择 `alpha` 和 `l1_ratio`，而不是根据训练集拟合指标手工调参。

## 与 Ridge 和 Lasso 比较

| 性质 | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| 平滑收缩 | 是 | 是 | 是 |
| 能产生精确的 0 | 通常否 | 是 | `l1_ratio>0` 时可以 |
| 面对高度相关的预测变量 | 通常较稳定 | 可能任意保留其中一个 | 更倾向于保留成组相关变量 |
| 主要调参 | `alpha` | `alpha` | `alpha` + `l1_ratio` |

## CPU、GPU、公式接口、权重与热启动

```python
model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    compute_inference=False,
).fit(X, y)
```

在相应环境中，公开接口支持 NumPy CPU、CuPy CUDA 和 Torch CUDA。`fit()` 支持 `sample_weight=`，并转发共享的 `formula=` / `data=` 公式接口。单次拟合还可以通过 `initial_coef=` 提供初始系数，用于热启动。

## 进阶：优化与 KKT 条件

在消去不受惩罚的截距后，也就是对中心化后的系数问题，KKT 条件为

$$
\frac{1}{n}X_c^\top(X_c\hat\beta-y_c)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1=0.
$$

对一次直接的 `ElasticNet.fit`，`solver` 决定实际数值算法，并在各后端上具有统一含义。历史参数 `cpu_solver` 仅用于兼容旧接口，不再是另一个直接拟合求解器选择项。

## 进阶：拟合后的统计推断

`ElasticNet` 默认只做估计。设置 `compute_inference=True` 后，拟合后推断不会改变用于预测的惩罚系数 `coef_` 和截距 `intercept_`。

| `inference_method` | 用途 | 主要限制 |
|---|---|---|
| `debiased` | 使用共享的逐节点近似精度矩阵做一步纠偏系数推断 | 软件实现复用并不意味着任意 `l1_ratio` 都自动继承所有 Lasso 理论；统计有效性仍取决于初始估计量和相应假设 |
| `post_selection_ols` | 在本次拟合实际采用的 NumPy/CuPy/Torch 后端上，对活跃集做 OLS/WLS 重拟合 | 属于选择后诊断，不是一般的选择性推断保证 |
| `bootstrap` | 使用残差重采样进行替代性不确定性评估 | 计算开销较大，并依赖相应重采样假设 |

历史别名 `cpu_ols` 和 `gpu_ols` 都会映射到与硬件无关的 `post_selection_ols`；真正的设备仍由 `device` 单独选择。

### `debiased` 推断中的逐节点调参

`nodewise_alpha` 与主 Elastic Net 的 `alpha` 相互独立。它只改变纠偏推断中用于估计设计矩阵近似精度矩阵的逐节点 Lasso，不会改变用于预测和变量选择的惩罚拟合结果。

显式给出的有限正实数直接生效。若 `nodewise_alpha=None` 且 $p\ge2$，statgpu 会先标准化已经完成中心化和加权处理的工作设计，然后取

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无分析权重时 $n_{\mathrm{nw}}=n$；存在非均匀分析权重时使用 Kish 型有效样本量。$\sqrt{\log(p)/n}$ 的量级有高维理论背景，但公式中的具体常数和有效样本量定义属于 statgpu 的默认选择，并非某个定理唯一规定的形式。

**版本变化：** 这套与响应变量尺度无关的自动规则计划自 **0.2.6** 起成为公开行为。0.2.5 及更早的内部实现曾把响应残差尺度带入逐节点惩罚；旧规则不会作为公开兼容模式继续保留。

逐节点问题求解结束后，statgpu 会**独立重新计算一次完整的 KKT 残差**。只有这项数值检查通过，才采用得到的近似精度矩阵继续生成推断结果；之后再把精度矩阵从标准化尺度变换回原工作特征尺度。`p=1` 时没有辅助逐节点回归，直接使用一维解析精度矩阵，也不会实际使用 `nodewise_alpha`。

NumPy、CuPy 和 Torch 采用同一套统计定义。显式选择 CUDA 或 Torch 时，纠偏推断的数值计算不会静默改用 CPU。多特征纠偏推断成功后，实际采用的逐节点惩罚值记录在 `nodewise_alpha_` 中，`_inference_result.metadata` 则保存 KKT 检查、缓存和实际计算后端/设备等溯源信息。

对于 `ElasticNetCV`，`nodewise_alpha` 只用于最终全数据重拟合后的纠偏推断，不进入 `alpha` / `l1_ratio` 候选网格，也不影响各折评分和最终调参选择。当前交叉验证后的推断仍是在已选调参值条件下进行。详见 [逐节点调参迁移说明](../guides/nodewise-alpha-migration.md)。

## 常见误区

- 用于预测时应同时考虑 `alpha` 与 `l1_ratio`，不要只调其中一个。
- 不要认为 `l1_ratio=0` 会让整个 `ElasticNet` 接口和行为完全等同于 `Ridge`。
- 更稳定的活跃集不等价于因果识别。
- 不要忽略正则化前的特征尺度。
- 不要用训练集 $R^2$，或兼容性的 AIC/BIC/F 字段，代替针对惩罚模型的验证或交叉验证。
- 不要混淆主模型 `alpha` 与 `nodewise_alpha`：后者只影响纠偏推断中的近似精度矩阵估计。
- `post_selection_ols` 是选择后诊断，不是自动获得有效选择性推断的捷径。

## 完整 API 参考

当前 `master` 上的公开构造函数在原有 Elastic Net 参数之外，还包含计划随 0.2.6 发布的 `nodewise_alpha=None`：

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

下面的参数表对应 Elastic Net 源码中静态定义的构造参数；运行时新增的 `nodewise_alpha` 紧随表后单独说明。

<!-- API-CONSTRUCTOR-START:ElasticNet -->
| 参数 | 默认值 | 含义 |
|---|---:|---|
| `alpha` | `1.0` | 总正则化强度。 |
| `l1_ratio` | `0.5` | L1 惩罚所占比例。 |
| `fit_intercept` | `True` | 是否拟合不受惩罚的截距。 |
| `max_iter` | `1000` | 最大求解迭代次数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | 使用 `coef_delta` 或 `kkt` 判断停止。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | `None` | 适用路径中的并行计算提示。 |
| `solver` | `"fista"` | 直接拟合时使用的后端无关求解器选择。 |
| `cpu_solver` | `"fista"` | 为旧接口保留的兼容参数；不再决定直接拟合算法。 |
| `lipschitz_L` | `None` | 可选的预计算 Lipschitz 常数。 |
| `gpu_memory_cleanup` | `False` | 拟合结束后尽力释放缓存的 GPU 内存。 |
| `compute_inference` | `False` | 是否执行所选的拟合后推断。 |
| `inference_method` | `"debiased"` | `debiased`、规范名称 `post_selection_ols` 或 `bootstrap`；`cpu_ols` / `gpu_ols` 为弃用别名。 |
| `cov_type` | `"nonrobust"` | 适用推断方法使用的协方差约定。 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的最大滞后阶数。 |
<!-- API-CONSTRUCTOR-END:ElasticNet -->

**运行时新增的公开参数：** `nodewise_alpha=None`。`None` 使用标准化设计上的自动规则；显式有限正实数用于指定逐节点惩罚。多特征纠偏推断成功后，实际采用的值会记录在 `nodewise_alpha_` 中。

### `fit` 与重要输出

`fit(X=None, y=None, sample_weight=None, initial_coef=None, **kwargs)` 返回 `self`；共享转发关键字包括 `formula` 与 `data`。

| 属性 | 含义 |
|---|---|
| `coef_`, `intercept_` | 用于预测的惩罚拟合结果 |
| `n_iter_` | 数值求解迭代次数 |
| `nodewise_alpha_` | 多特征纠偏推断成功后实际采用的逐节点调参值；其他情况为 `None` |
| `_params`, `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | 推断成功后的参数、标准误、统计量、p 值和置信区间 |
| `_inference_result` | 包含逐节点调参、KKT 检查以及计算后端/设备溯源信息的结构化结果 |

`predict`、`score`、`summary`、`get_params` 和 `set_params` 遵循共享的估计器接口约定。通过 `set_params` 修改 `nodewise_alpha` 会使旧的推断状态失效；clone 和参数查询则保留用户指定的值。

## 验证

维护中的测试覆盖 Elastic Net 目标函数、求解器与 KKT 条件、直接拟合对 `nodewise_alpha` 的不变性、`ElasticNetCV` 最终重拟合隔离、加权推断、公式接口与公开 API、NumPy/CuPy/Torch 近似精度矩阵一致性，以及共享逐节点实现的物理 CUDA 验证。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *JRSS B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
