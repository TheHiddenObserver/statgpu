# Lasso

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：模型文档  
> 切换：[English](../../en/models/lasso.md)

## 概览

`Lasso` 提供 L1 正则化线性回归，支持 CPU/GPU 拟合以及多种拟合后推断方法。直接拟合统一使用与后端无关的 `solver` 参数：设备选择、求解算法和统计推断方法彼此独立。

公开路径：`statgpu.linear_model.Lasso`

## 目标函数

优化目标为

$$
\min_{\beta,b}
\frac{1}{2n}\|y-X\beta-b\|_2^2
+\alpha\|\beta\|_1,
$$

其中 `alpha` 控制稀疏化强度。

Lasso 没有普通最小二乘那样的闭式解，需要通过迭代算法求解。停止条件可以由系数变化（`coef_delta`）或 KKT 一致性（`kkt`）控制。

对于**单次直接拟合**，真正控制算法的是 `solver`：

- CPU 坐标下降：`solver="coordinate_descent"`；
- CPU 或 GPU 的近端路径：`solver="fista"`，以及当前明确支持的其他求解器；
- 执行位置单独通过 `device="cpu"`、`"cuda"` 或 `"torch"` 选择。

历史参数 `cpu_solver` 已弃用。为了兼容旧代码仍可暂时传入，但它**不再决定直接拟合算法**。新代码应使用 `solver=...`；详见 [惩罚模型求解器 API 迁移](../guides/penalized-solver-api-migration.md)。

## 推断方法

`inference_method` 控制拟合后采用哪一种统计推断程序：

- `post_selection_ols`：在惩罚拟合选出的活跃集上进行 OLS/WLS 诊断性重拟合；
- `debiased`：去偏（de-biased / de-sparsified）Lasso 推断；
- `bootstrap`：固定设计、固定调参配置下的 Gaussian 残差自助法。

旧名称 `cpu_ols` 与 `gpu_ols` 已弃用，并统一映射到 `post_selection_ols`。它们不是两种不同的统计方法；计算设备由 `device` 决定。`LassoCV` 在兼容范围内也会接受更早的 `cpu_ols_inference` / `gpu_ols_inference` 拼写，并映射到同一规范方法。

这些方法的统计目标不同，不能互换解释。一般性说明见 [推断模式](../guides/inference-modes.md)。

### `post_selection_ols`

惩罚拟合先选出活跃特征集，随后 statgpu 只在这些特征上进行无惩罚 OLS；如果传入 `sample_weight`，则进行 WLS。协方差和参考分布推断在成功拟合所使用的数值后端上完成，数值计算结束后再把小型结果数组统一整理为 NumPy 结果。

原始惩罚 `coef_` 和 `intercept_` 保持不变，并继续用于预测。活跃集 OLS/WLS 重拟合用于推断与报告，结果保存在 `_params`、`_inference_result`、`_bse`、`_tvalues` / `_zvalues`、`_pvalues`、`_conf_int` 等字段中。

需要注意：

- `post_selection_ols` 是选择后的诊断性重拟合。使用同一份数据先选变量再做普通 OLS/WLS，并不会自动获得一般意义上的选择后推断覆盖保证；
- 活跃设计秩亏时，残差自由度使用有效秩，系数重拟合与协方差计算基于设计矩阵层面的 Moore–Penrose/SVD，而不是通过正规方程进一步放大条件数问题。

### `debiased`

去偏推断通过逐节点 Lasso 估计设计矩阵精度矩阵的近似，再对惩罚估计量进行一步修正。它适用于需要高维系数级推断、且相应理论假设对应用场景合理的情形。

普通 `_conf_int` 是各个系数的**边际区间**。如果需要对一组参数同时控制覆盖率，应显式启用同时推断，而不能把普通边际区间直接当成联合区间。

#### 逐节点调参

主模型的 `alpha` 控制用于预测和变量选择的 Lasso 拟合；`nodewise_alpha` 是另一个独立参数，只在 `inference_method="debiased"` 时用于逐节点 Lasso。

如果显式给出 `nodewise_alpha`，statgpu 会在标准化后的逐节点设计上使用该正标量。若省略（`None`），自动规则为

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无解析权重时 $n_{\mathrm{nw}}=n$；非均匀解析权重下使用 Kish 型有效样本量。成功的多特征去偏推断会通过 `nodewise_alpha_` 暴露实际采用的值。单特征问题不存在需要控制的其他特征，因此直接使用一维解析精度值，并令 `nodewise_alpha_` 保持为 `None`。

对 `LassoCV` 而言，`nodewise_alpha` 只属于最终全数据重拟合的推断配置，不参与主模型 `alpha` 的候选网格、折内评分或参数选择。更完整的说明见 [逐节点 Lasso 推断调参迁移](../guides/nodewise-alpha-migration.md)。

#### 截距的参数归属

当 `inference_method="debiased"` 时，预测参数与推断参数有意分开：

- 公开的 `coef_` 与 `intercept_` 始终属于**惩罚预测拟合**；
- `_params[1:]` 保存去偏后的斜率；
- `_params[0]` 保存与这些去偏斜率处于同一原始坐标系参数化下的截距。

因此，`_bse[0]`、第一个 z 统计量/p 值以及 `_conf_int[0]` 描述的是去偏推断中的截距，而不是预测用的 `intercept_`。

对 `LassoCV(compute_inference=True, inference_method="debiased")`，外层 CV 估计器公开的推断结果来自选定 `alpha` 后的最终全数据重拟合；公开 `coef_` / `intercept_` 仍属于这个选定参数下的惩罚预测模型。

#### 同时推断

设置 `enable_simultaneous_inference=True` 后，Lasso 使用乘子自助法的 max-|Z| 临界值。普通 `_conf_int` 仍是边际区间，同时置信区间单独保存在 `_conf_int_simultaneous`。

- `simultaneous_alpha` 必须严格位于 `(0,1)`；
- `simultaneous_n_bootstrap` 必须为正整数；
- `simultaneous_include_intercept=True` 时，去偏截距会真正进入 max-|Z| 校准和最终联合区间的目标集合，而不是只额外显示一行结果。

在受支持的 CuPy/Torch 路径上，同时推断的主要数值计算保持在成功拟合的 GPU 后端上；最终小型结果数组再统一整理为 NumPy 结果。

### `bootstrap`

残差自助法采用固定设计、固定调参配置的重复重拟合：

1. 根据已拟合模型计算拟合值和残差；
2. 对残差进行有放回抽样；
3. 在拟合值周围构造新的 Gaussian 响应变量；
4. 使用同一套 Lasso 配置重新拟合；
5. 汇总重复拟合形成的系数分布。

`n_bootstrap` 控制重拟合次数，`bootstrap_random_state` 控制可重复性。

当前残差自助法要求：

- `sample_weight=None`；
- `cov_type="nonrobust"`。

带权残差自助法、稳健/HC 自助法、HAC/分块自助法、非 Gaussian 自助法与 Cox 自助法不受支持，请求时会直接报错。得到的区间描述固定设计、固定调参配置下的抽样波动，不会自动校正变量选择或调参选择带来的额外不确定性。

## 设备与后端

`inference_method` 只选择统计程序，不选择硬件：

- `device="cpu"` 请求 NumPy CPU；
- `device="cuda"` 请求 CuPy CUDA，不可用时直接报错；
- `device="torch"` 请求 Torch CUDA，不可用时直接报错；
- `device="auto"` 允许在受支持且可用的后端之间自动选择。

显式 CUDA/Torch 请求不会被静默改成 CPU 推断。数值推断完成后，小型结果数组可以转换到 NumPy 统一展示；这不表示推断本身重新在 CPU 上计算。

## 解析权重

直接 Lasso 拟合与受支持的去偏推断，在 NumPy/CuPy/Torch 上使用同一个加权中心化平均损失约定。因此，把所有正解析权重同时乘上同一个常数不会改变统计问题。

`LassoCV` 的自动 `alpha` 网格、各个带权训练折、验证 MSE 以及最终全数据重拟合也遵循同一权重约定。常数正权重会退化为与无权重等价的选择问题。

关于 CV 的选择与最终重拟合语义，见 [交叉验证](../guides/cross-validation.md)。

## 参数

下表列出 `statgpu.linear_model.Lasso` 的公开构造参数。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L1 正则化强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 优化最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | 停止准则：`coef_delta` / `kkt` |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`；`cpu_ols` 和 `gpu_ols` 暂时作为弃用别名接受 |
| `nodewise_alpha` | `None` | 去偏推断中逐节点 Lasso 的惩罚强度 |
| `n_bootstrap` | `200` | 残差自助法推断的抽样次数 |
| `bootstrap_random_state` | `None` | 残差自助法随机种子 |
| `enable_simultaneous_inference` | `False` | 是否启用同时推断（仅 `debiased`） |
| `simultaneous_method` | `"maxz_bootstrap"` | 同时推断方法；当前为 `maxz_bootstrap` |
| `simultaneous_alpha` | `0.05` | 同时推断的族错误率水平，必须严格位于 `(0,1)` |
| `simultaneous_n_bootstrap` | `1000` | max-|Z| 乘子自助法的抽样次数，必须为正整数 |
| `simultaneous_random_state` | `None` | 同时推断随机种子 |
| `simultaneous_include_intercept` | `False` | 是否把去偏截距纳入同时推断目标集合 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda`（CuPy）/ `torch`（Torch CUDA） |
| `n_jobs` | `None` | 适用 CPU 路径的并行度 |
| `compute_inference` | `True` | 是否计算拟合后推断 |
| `solver` | `"fista"` | 与后端无关的直接拟合求解器；CPU 坐标下降使用 `coordinate_descent` |
| `cpu_solver` | `"coordinate_descent"` | **弃用兼容参数**；不再决定直接拟合算法，请改用 `solver` |
| `lipschitz_L` | `None` | 兼容迭代求解器可使用的显式 Lipschitz 常数 |
| `admm_rho` | `1.0` | 选择 ADMM 路径时的惩罚参数 |
| `gpu_memory_cleanup` | `False` | 在支持的路径上，拟合后是否请求释放可回收的 GPU 缓存内存 |

## CPU 与 GPU 示例

```python
from statgpu.linear_model import Lasso

# CPU 坐标下降：solver 选择算法，device 选择执行位置。
m_cpu = Lasso(
    alpha=0.1,
    device="cpu",
    solver="coordinate_descent",
    stopping="kkt",
)
m_cpu.fit(X, y)

# 显式逐节点调参只影响去偏推断，不改变主 Lasso 拟合。
m_db = Lasso(
    alpha=0.1,
    nodewise_alpha=0.08,
    device="cpu",
    inference_method="debiased",
)
m_db.fit(X, y)
print(m_db.nodewise_alpha_)

# GPU FISTA + 与硬件无关的选择后 OLS/WLS 推断方法。
m_gpu = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="post_selection_ols",
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)

# 预测使用惩罚拟合；推断使用活跃集 OLS/WLS 重拟合。
penalized_coef = m_gpu.coef_
post_selection_params = m_gpu._params
```

同时推断示例：

```python
m_sim = Lasso(
    alpha=0.1,
    device="cpu",
    inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=7,
    simultaneous_include_intercept=True,
)
m_sim.fit(X, y)

ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
```

## 不同推断方法的定位

- `debiased`：高维系数推断的主要路径；
- `post_selection_ols`：较轻量的活跃集 OLS/WLS 诊断性重拟合；
- `bootstrap`：计算成本更高的重抽样方法。

三者的统计主张不同，不能互换解释。

## 输出

- 惩罚预测拟合：`intercept_`、`coef_`、`n_iter_`；
- 推断（启用时）：`_params`、`_bse`、`_tvalues` / `_zvalues`、`_pvalues`、`_conf_int`、`_inference_result`；
- 成功的多特征 `debiased` 推断：`nodewise_alpha_` 记录实际采用的逐节点调参值；
- `post_selection_ols`：`coef_` 仍是惩罚系数，`_params` 保存嵌入完整参数布局的活跃集 OLS/WLS 重拟合结果；
- `debiased`：`_params[1:]` 保存去偏斜率，`_params[0]` 保存与之匹配的原始坐标系截距；
- 开启同时推断后：`_conf_int_simultaneous` 保存配置目标集合上的联合区间；
- 方法：`fit`、`predict`、`score`、`summary`；
- 可用时还包括 `aic`、`bic` 等诊断。

## 常见问题

- **`alpha` 与 `nodewise_alpha` 有什么区别？**  
  `alpha` 定义用于预测/变量选择的主 Lasso 拟合；`nodewise_alpha` 只用于去偏推断中的近似精度矩阵构造。
- **为什么同样 `tol` 下 CPU/GPU 迭代数不同？**  
  不同数值后端和算法可能产生不同收敛轨迹；比较时应固定 `solver` 与 `stopping`。
- **CPU 用户应该设置 `cpu_solver` 吗？**  
  不应该。直接拟合统一使用 `solver`；`cpu_solver` 只是旧 CPU/GPU 分离接口的弃用兼容参数。
- **应该根据硬件选择 `cpu_ols` 或 `gpu_ols` 吗？**  
  不应该。两者都是 `post_selection_ols` 的弃用别名。统计方法由 `inference_method` 选择，执行位置由 `device` 选择。
- **`post_selection_ols` 会改变 `coef_` 吗？**  
  不会。预测继续使用惩罚系数；活跃集重拟合结果保存在 `_params`、`_inference_result` 等推断字段中。
- **为什么 `debiased` 下 `intercept_` 可能和 `_params[0]` 不同？**  
  `intercept_` 属于惩罚预测拟合；`_params[0]` 是与去偏斜率配套的推断截距。
- **`post_selection_ols` 能当作严格的选择后推断置信程序吗？**  
  不能，应把它理解为选择后的诊断性重拟合。
- **普通 `debiased` 区间是联合区间吗？**  
  不是。普通 `_conf_int` 是边际区间；需要联合控制时应使用同时推断路径。
- **如何把截距纳入联合覆盖？**  
  设置 `simultaneous_include_intercept=True`；去偏截距会同时进入 max-|Z| 校准和最终联合区间目标集合。

## 相关文档

- [推断模式](../guides/inference-modes.md) — 去偏、选择后重拟合与残差自助法的统计解释
- [逐节点 Lasso 推断调参迁移](../guides/nodewise-alpha-migration.md) — `nodewise_alpha`
- [交叉验证](../guides/cross-validation.md) — `LassoCV` 的选择与最终重拟合
- [惩罚模型求解器 API 迁移](../guides/penalized-solver-api-migration.md) — `solver` / `cpu_solver` 迁移
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

## 参考文献

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing in high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
