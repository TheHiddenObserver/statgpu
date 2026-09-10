# Lasso 回归

> 语言：中文  
> 最后更新：2026-09-10  
> 切换：[English](../../en/models/lasso.md)

> **版本提示：** 当前正式发布版为 **0.2.5**。本页涉及的公开参数 `nodewise_alpha` 以及新的自动逐节点调参规则已合入当前 `master`，计划随 **0.2.6** 发布；正式版 0.2.5 尚不包含这一公开接口和新默认规则。

## 它解决什么问题？

`Lasso` 是带 L1 惩罚的线性回归。它不仅会收缩系数，还能把较弱的系数直接压到 0，因此一次拟合就可以同时用于预测和稀疏变量选择。

当你认为大量候选变量中只有少数变量携带主要信号时，Lasso 往往很合适。如果多数变量都有小但真实的作用，[Ridge](ridge.md) 通常更稳定；如果重要变量成组高度相关，[Elastic Net](elastic-net.md) 往往更合适。

## 直觉

Lasso 在“拟合数据”和“限制系数绝对值总量”之间折中。L1 惩罚在 0 处有尖点，因此较弱的系数更新会经过**软阈值（soft-thresholding）**后直接变成 0，而不只是被缩小。

```text
OLS   ：不做正则化
Ridge ：平滑收缩所有系数
Lasso ：收缩，并把较弱系数压到 0
```

稀疏性便于解释和部署，但变量选择结果依赖样本与调参。某个系数在这次拟合中等于 0，并不等价于证明它在总体中的真实效应严格为 0。

## 模型与目标函数

带不受惩罚的截距 $b$ 时，statgpu 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top\beta)^2
+\alpha\lVert\beta\rVert_1.
$$

`alpha` 越大，收缩越强，通常会有更多系数变为 0。由于 L1 惩罚直接作用在系数尺度上，连续预测变量在正则化前通常应先放到可比尺度。

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

`coef_` 与 `intercept_` 始终表示**用于预测的惩罚拟合结果**。非零系数仍然经过收缩，不能直接当成普通 OLS 的无偏效应估计。

## 关键参数

| 参数 | 默认值 | 如何理解 |
|---|---:|---|
| `alpha` | `1.0` | 主模型的正则化强度。用于预测时应优先通过验证集或 `LassoCV` 选择，而不是看训练集拟合优度。 |
| `fit_intercept` | `True` | 除非理论上确定截距为 0，或设计矩阵已经显式包含截距，一般保留。 |
| `device` | `"auto"` | 选择 CPU、CuPy CUDA、Torch CUDA 或自动路由；显式指定 GPU 而设备不可用时会报错，不会悄悄改用 CPU。 |
| `solver` | `"fista"` | 直接拟合时实际选择数值算法的参数，对各后端采用统一含义。 |
| `stopping` | `"coef_delta"` | 需要按照最优性条件判断收敛时可选 `kkt`。 |
| `compute_inference` | `True` | 只做预测或变量选择时可以关闭推断。 |
| `inference_method` | `"debiased"` | 选择拟合完成后的统计推断方法，与计算设备的选择彼此独立。 |
| `nodewise_alpha` | `None` | 仅用于 `debiased` 纠偏推断中估计近似精度矩阵的逐节点 Lasso 调参。计划自 0.2.6 起公开。 |

## CPU、GPU、公式接口与权重

```python
model = Lasso(
    alpha=0.08,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=False,
).fit(X, y)
```

显式 `device="cuda"` 使用 CuPy CUDA，显式 `device="torch"` 使用 Torch CUDA；对应设备不可用时会直接报错。`fit()` 也支持 `sample_weight=`，以及共享的 `formula=` / `data=` 公式接口。

分析权重沿用平均损失的定义。把所有权重同时乘以同一个正数，不会改变预期的加权稀疏高斯线性模型问题。

## 与相邻方法比较

| 方法 | 能产生精确的 0？ | 高相关预测变量 | 典型用途 |
|---|:---:|---|---|
| OLS / `LinearRegression` | 否 | 可能不稳定 | 无惩罚估计 |
| [Ridge](ridge.md) | 否 | 稳定性较好 | 不要求删变量的预测 |
| **Lasso** | 是 | 可能只保留一组相关变量中的一个 | 稀疏预测 / 变量选择 |
| [Elastic Net](elastic-net.md) | L1 比例 > 0 时可以 | 更适合成组相关变量 | 高相关特征下的稀疏模型 |

## 进阶：求解器

对于一次直接的 `Lasso.fit`，`solver` 选择算法，`device` 选择在哪个后端执行。历史参数 `cpu_solver` 仅为兼容旧接口保留，不再决定直接拟合使用的算法。

| `solver` | CPU | CuPy / Torch | 说明 |
|---|:---:|:---:|---|
| `fista` | 是 | 是 | 默认的近端梯度算法 |
| `auto` | 是 | 是 | 当前 Gaussian + L1 问题的自动选择 |
| `fista_bb` | 是 | 是 | 使用 Barzilai–Borwein 谱步长的 FISTA 变体 |
| `admm` | 是 | 是 | ADMM 分裂算法；分析权重存在额外限制 |
| `coordinate_descent` | 是 | 否 | 仅 CPU 支持的坐标下降直接拟合路径 |

## 进阶：Lasso 拟合后的推断

数据驱动的稀疏变量选择之后，不能直接套用“模型事先给定”时的普通 OLS 推断。statgpu 提供几种统计含义不同的方法：

| `inference_method` | 做什么 | 主要限制 |
|---|---|---|
| `debiased` | 对 Lasso 系数进行一步纠偏，并给出系数级推断 | 有效性依赖高维稀疏性、设计、噪声和调参条件 |
| `post_selection_ols` | 在本次拟合实际采用的后端上，对 Lasso 选出的活跃集做 OLS/WLS 重拟合 | 属于选择后诊断，不是一般的选择性推断保证 |
| `bootstrap` | 对惩罚模型做残差自助法重拟合 | 计算开销更大，也不是对模型选择不确定性的普适修正 |

`post_selection_ols` 是与硬件无关的规范名称。历史别名 `cpu_ols` 和 `gpu_ols` 都会映射到同一个统计方法；设备仍由 `device` 单独决定。

`LassoCV(compute_inference=True)` 会先完成主模型 `alpha` 的选择，然后只在最终的全数据重拟合上计算推断。当前输出是**在已经选定的调参值条件下**得到的，并没有额外修正交叉验证带来的调参不确定性。

### `alpha` 与 `nodewise_alpha`

`alpha` 控制用于预测和变量选择的主 Lasso 拟合；`nodewise_alpha` 只控制 `inference_method="debiased"` 中用于估计设计矩阵近似精度矩阵的逐节点 Lasso（node-wise Lasso）。因此，只改变 `nodewise_alpha` 不应改变惩罚拟合得到的 `coef_`，也不应改变 `LassoCV` 的候选 `alpha` 网格、各折评分或最终 `alpha_`。

如果显式给出有限正实数 `nodewise_alpha`，statgpu 直接使用该值。若 `nodewise_alpha=None` 且 $p\ge2$，则先标准化已经完成中心化和加权处理的工作设计，再使用

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重下使用 Kish 型有效样本量。$\sqrt{\log(p)/n}$ 的量级具有高维理论背景，但公式中的具体常数和加权有效样本量定义属于 statgpu 的默认选择。

**版本变化：** 上述自动规则计划自 **0.2.6** 起成为公开行为，并且有意做到与响应变量的计量尺度无关。0.2.5 及更早的内部实现曾把响应残差尺度估计乘入逐节点惩罚；旧规则不会作为公开兼容模式继续保留。

逐节点问题在标准化设计上求解。求解结束后，statgpu 会**另行重新计算一次完整的 KKT 残差**；只有这项数值检查通过，才采用得到的近似精度矩阵继续生成推断结果。随后再把精度矩阵变换回原工作特征尺度。

当 `p=1` 时没有需要拟合的辅助逐节点回归，statgpu 直接使用一维解析精度矩阵，此时 `nodewise_alpha_` 保持为 `None`。

对于 `LassoCV`，`nodewise_alpha` 只属于最终全数据重拟合后的推断配置。更完整的统计构造见 [Lasso 推断](lasso-inference.md)，版本迁移细节见 [逐节点调参迁移说明](../guides/nodewise-alpha-migration.md)。

### 计算后端与结果返回

NumPy、CuPy 与 Torch 使用同一套逐节点统计定义。显式选择 CUDA 或 Torch 后端时，纠偏推断的数值计算不会静默改用 CPU；只有后端上的数值推断完成后，少量用于展示和汇总的数组才转换为 NumPy。`_inference_result.metadata` 会记录实际执行数值计算的后端和设备。

`fit_intercept=True` 时，纠偏推断采用与中心化模型一致的参数化。`coef_` / `intercept_` 继续表示用于预测的惩罚拟合结果，而 `_params` 表示纠偏后的推断参数。若启用截距的同时推断，程序使用 max-|Z| 高斯乘子自助法，并把同时置信区间 `_conf_int_simultaneous` 与边际区间 `_conf_int` 分开保存。

## 常见误区

- 不要把“被 Lasso 选中”理解为因果证据，或理解为总体效应必然非零。
- 不要忽略 L1 正则化前的特征尺度。
- 不要用训练集 $R^2$ 选择 `alpha`。
- 不要把 `post_selection_ols` 当作一般的选择性推断方法。
- 不要认为 `LassoCV` 会自动修正调参不确定性。
- 不要混淆 `alpha` 与 `nodewise_alpha`：后者只影响纠偏推断中的近似精度矩阵估计。
- KKT 残差很小只说明相应逐节点优化问题在数值上求解充分，并不意味着高维推断所需的统计假设自动成立。

## 完整 API 参考

当前 `master` 上的公开构造函数在原有 Lasso 参数之外，还包含计划随 0.2.6 发布的 `nodewise_alpha=None`：

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

下面的参数表对应 Lasso 源码中静态定义的构造参数；运行时新增的 `nodewise_alpha` 紧随表后单独说明。

<!-- API-CONSTRUCTOR-START:Lasso -->
| 参数 | 默认值 | 含义 |
|---|---:|---|
| `alpha` | `1.0` | L1 惩罚强度。 |
| `fit_intercept` | `True` | 是否拟合不受惩罚的截距。 |
| `max_iter` | `1000` | 最大求解迭代次数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | 使用 `coef_delta` 或 `kkt` 判断停止。 |
| `inference_method` | `"debiased"` | `debiased`、规范名称 `post_selection_ols` 或 `bootstrap`；`cpu_ols` / `gpu_ols` 为弃用别名。 |
| `n_bootstrap` | `200` | 残差自助法的重采样次数。 |
| `bootstrap_random_state` | `None` | 残差自助法随机种子。 |
| `enable_simultaneous_inference` | `False` | 是否在纠偏推断后计算 max-|Z| 同时置信区间。 |
| `simultaneous_method` | `"maxz_bootstrap"` | 同时推断的校准方法。 |
| `simultaneous_alpha` | `0.05` | 同时推断的族错误率水平。 |
| `simultaneous_n_bootstrap` | `1000` | 高斯乘子自助法的重复次数。 |
| `simultaneous_random_state` | `None` | 同时推断的随机种子。 |
| `simultaneous_include_intercept` | `False` | 是否把纠偏后的截距纳入同时推断的目标参数集合。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | `None` | 适用路径中的并行计算提示。 |
| `compute_inference` | `True` | 是否执行所选的拟合后推断。 |
| `solver` | `"fista"` | 直接拟合时使用的后端无关求解器选择。 |
| `cpu_solver` | `"coordinate_descent"` | 为旧接口保留的兼容参数；不再决定直接拟合算法。 |
| `lipschitz_L` | `None` | 可选的预计算 Lipschitz 常数。 |
| `admm_rho` | `1.0` | 使用 ADMM 时的增广拉格朗日惩罚参数。 |
| `gpu_memory_cleanup` | `False` | 拟合结束后尽力释放缓存的 GPU 内存。 |
<!-- API-CONSTRUCTOR-END:Lasso -->

**运行时新增的公开参数：** `nodewise_alpha=None`。`None` 使用标准化设计上的自动规则；显式有限正实数用于指定逐节点惩罚。多特征纠偏推断成功后，实际采用的值会记录在 `nodewise_alpha_` 中。

### `fit` 与核心方法

`fit(X=None, y=None, sample_weight=None, formula=None, data=None)` 返回 `self`。`predict(X, return_cpu=True)` 给出连续预测；`score(X, y, sample_weight=None)` 返回 $R^2$；`summary()` 汇总可用的推断结果。`get_params` / `set_params` 与 sklearn clone 会保留用户指定的 `nodewise_alpha`；修改它会使旧的拟合后推断状态失效。

`adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` 等继承的推断工具见 [推断 API](../guides/inference-api.md)。

### 重要拟合后字段

| 属性 | 含义 |
|---|---|
| `coef_`, `intercept_` | 用于预测的惩罚拟合结果 |
| `n_iter_` | 数值求解迭代次数 |
| `nodewise_alpha_` | 多特征纠偏推断成功后实际采用的逐节点调参值；其他情况为 `None` |
| `_params`, `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | 推断成功后的参数、标准误、统计量、p 值与边际置信区间 |
| `_conf_int_simultaneous` | 显式启用并成功校准后的同时置信区间 |
| `_inference_result` | 包含逐节点调参、KKT 检查和后端溯源信息的结构化推断结果 |

如果提供 `rsquared_adj`、`fvalue`、`f_pvalue`、`aic`、`bic` 等惩罚拟合诊断量，它们沿用普通参数计数和残差自由度约定，主要用于兼容与报告；不能把它们当成已经修正变量选择、调参过程或有效自由度的模型选择准则。

## 验证

维护中的测试覆盖直接拟合、公开 API、clone / `set_params`、响应尺度不变性、特征尺度等变性、一维解析路径、KKT 检查失败时中止推断、权重不变性、CV 最终重拟合隔离、公式接口一致性、独立双特征解析参考、缓存溯源、NumPy/CuPy/Torch 一致性，以及已经验收的物理 CUDA 验证。

## 参考文献

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
