# Lasso

> 语言：中文  
> 最后更新：2026-10-05  
> 页面定位：模型文档  
> 切换：[English](../../en/models/lasso.md)

## 为什么使用 Lasso？

Lasso 在拟合线性预测模型的同时，把系数向零收缩。L1 惩罚可以使部分系数恰好为零，
适合候选变量很多、其中一些可能贡献很小的稀疏建模问题。预测变量高度相关时，
入选变量可能不稳定；若希望保留相关变量组，可比较 [Elastic Net](elastic-net.md)；
低维且不需要惩罚时，可使用[普通最小二乘](linear-regression.md)。
Lasso 系数非零本身不是显著性检验，也不代表因果效应。

可从 `statgpu` 或 `statgpu.linear_model` 导入 `Lasso`。仅关注预测时，建议先用
`compute_inference=False`；构造默认值为 `True`，会额外请求去偏推断。
选择预测惩罚强度与解释系数不确定性是两个不同任务。完整方法签名、
参数默认值和输出见 [Lasso API 参考](../reference/linear-model-api.md#lasso)。

## 模型与目标函数

对于 n 个观测、p 个特征、响应 y 和形状为 `(n,p)` 的设计矩阵 X，Lasso 求解

$$
\min_{b,\beta}\frac{1}{2n}\sum_{i=1}^n(y_i-b-x_i^\top\beta)^2
+\alpha\|\beta\|_1.
$$

截距 b 不受惩罚；`fit_intercept=False` 固定 b=0。在这一**平均损失**约定下，
`alpha` 越大，收缩越强。使用分析权重时，平方损失项替换为

$$
\frac{1}{2\sum_{i=1}^n w_i}\sum_{i=1}^n w_i(y_i-b-x_i^\top\beta)^2,
$$

权重必须有限、非负且总和为正。全部权重乘以同一正常数不改变目标函数。
特征单位会影响惩罚：缩放参数只能从训练行学习，再原样应用到留出数据。
不要假设 Lasso 会自动标准化用于预测的特征。

## 完整的 CPU 示例

模拟特征已有相近尺度。拟合前先留出最后 60 行；只有前两列生成响应中的信号。

<!-- learner-example: lasso-prediction -->
```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(42)
X = rng.normal(size=(240, 6))
y = 1 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=240)
X_train, X_test = X[:180], X[180:]
y_train, y_test = y[:180], y[180:]
model = Lasso(
    alpha=0.1, device="cpu", solver="coordinate_descent",
    max_iter=5000, tol=1e-10, compute_inference=False,
).fit(X_train, y_train)
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Test R2:", round(model.score(X_test, y_test), 3))
```

该种子下，斜率约为 `[1.887, -0.880, 0, 0, 0, 0]`，留出 R² 约为 `0.980`。
前两个斜率相对生成值 2 和 −1 有所收缩。`prediction` 形状为 `(60,)`，
`coef_` 每项对应一个输入特征，`intercept_` 是独立标量。
其他数据可能选中噪声或漏掉真实信号；测试评分衡量预测，不衡量置信区间覆盖率。

## 选择 alpha 与检查拟合

- 使用训练数据内的验证或 `LassoCV` 选择 `alpha`，不要让测试集参与调参。
  上面的 0.1 只是示例，不是所有问题通用的选择。
- 比较合理惩罚范围内的留出误差和系数稳定性；需要学习预处理参数时，
  应在每个验证训练折内重新学习。
- `tol` 与 `max_iter` 控制数值求解，不控制稀疏度或统计不确定性。
  可收紧容差、增加迭代预算检查稳定性；迭代次数本身不能证明最优性。
- 需要不确定性结果时，应主动选择推断方法。同一数据上的调参和变量选择
  会引入额外不确定性，下文方法不会自动消除它，详见[交叉验证](../guides/cross-validation.md)。

## 求解器与当前控制参数限制

`solver` 选择算法，`device` 选择执行位置。CPU 坐标下降使用
`solver="coordinate_descent"`；受支持的 CPU/GPU 近端拟合可使用 `solver="fista"`。
其他组合见[求解器与惩罚兼容参考](../guides/solver-penalty-matrix.md)。

当前直接高斯拟合会保存 `stopping="coef_delta"` 或 `"kkt"`，但**不据此切换停止检查**。
CPU FISTA 和坐标下降检查系数变化绝对值之和，GPU FISTA 也检查系数变化；
ADMM 检查原始与对偶残差。因此，`stopping="kkt"` 不能证明直接拟合满足 KKT 容差。
尺度悬殊时，即使系数变化很小，最优性误差仍可能很大；应合理缩放特征，
对精度要求高时独立检查目标函数或 KKT 残差。单独的 Lasso CV/路径辅助算法
有自己的停止逻辑，路径检查成功不代表最终直接模型重拟合也通过了 KKT 检查。

`admm_rho` 当前仅被保存，统一 ADMM 实际以 rho=1.0 启动。是否自适应调整
取决于求解方式：平方误差的直接 Cholesky 求解保持 rho 不变，其他路径可以自适应调整。
因此该构造参数不是有效调参手段；转交给 `LassoCV` 最终重拟合时也有此限制。
`cpu_solver` 是弃用兼容参数，直接算法应使用 `solver` 选择，详见
[惩罚模型求解器 API 迁移指南](../guides/penalized-solver-api-migration.md)。

## 推断方法

`inference_method` 控制拟合后采用哪一种统计推断程序。普通 `auto` 解析为
`debiased`；启用同时推断时应显式填写 `debiased`，因为构造函数当前拒绝
这一组合中的 `auto`。默认值仍是 `debiased`：

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

无分析权重时 $n_{\mathrm{nw}}=n$；非均匀分析权重下使用 Kish 型有效样本量。成功的多特征去偏推断会通过 `nodewise_alpha_` 暴露实际采用的值。单特征问题不存在需要控制的其他特征，因此直接使用一维解析精度值，并令 `nodewise_alpha_` 保持为 `None`。

对 `LassoCV` 而言，`nodewise_alpha` 只属于最终全数据重拟合的推断配置，不参与主模型 `alpha` 的候选网格、折内评分或参数选择。更完整的说明见 [逐节点 Lasso 推断调参迁移](../guides/nodewise-alpha-migration.md)。

#### 截距的参数归属

当 `inference_method="debiased"` 且拟合截距时，预测参数与推断参数有意分开：

- 公开的 `coef_` 与 `intercept_` 始终属于**惩罚预测拟合**；
- `_params[1:]` 保存去偏后的斜率；
- `_params[0]` 保存与这些去偏斜率处于同一原始坐标系参数化下的截距。

不拟合截距时，`_params` 的所有行都是输入特征对应的斜率，`intercept_` 为零。

拟合截距时，`_bse[0]`、第一个 z 统计量/p 值以及 `_conf_int[0]` 描述的是去偏推断中的截距，而不是预测用的 `intercept_`。

对 `LassoCV(compute_inference=True, inference_method="debiased")`，外层 CV 估计器公开的推断结果来自选定 `alpha` 后的最终全数据重拟合；公开 `coef_` / `intercept_` 仍属于这个选定参数下的惩罚预测模型。

#### 同时推断

设置 `enable_simultaneous_inference=True` 后，Lasso 使用乘子自助法的 max-|Z| 临界值。普通 `_conf_int` 仍是边际区间，同时置信区间单独保存在 `_conf_int_simultaneous`。

- `simultaneous_alpha` 必须严格位于 `(0,1)`；
- `simultaneous_n_bootstrap` 必须为正整数；
- `simultaneous_include_intercept=True` 时，去偏截距会真正进入 max-|Z| 校准和最终联合区间的目标集合，而不是只额外显示一行结果。

在受支持的 CuPy/Torch 路径上，只有 `fit_intercept=True` 的中心化同时推断保持在成功拟合的 GPU 后端上，最终结果再整理为 NumPy。`fit_intercept=False` 的同时推断使用 NumPy 主机辅助程序，即使拟合本身在 GPU 上执行，也不属于 GPU 原生同时推断。

### `bootstrap`

残差自助法采用固定设计、固定调参配置的重复重拟合：

1. 根据已拟合模型计算拟合值和残差；
2. 对残差进行有放回抽样；
3. 构造新的自助响应（拟合值加重采样残差）；
4. 使用同一套 Lasso 配置重新拟合；
5. 汇总重复拟合形成的系数分布。

这里的 Gaussian 指高斯线性模型族；抽样来自经验残差，并非每次重新生成正态噪声。`n_bootstrap` 控制重拟合次数，`bootstrap_random_state` 控制可重复性。

当前残差自助法要求：

- `sample_weight=None`；
- 非稳健协方差配置（`cov_type="nonrobust"` 是 Lasso 的默认配置，不能作为其构造参数传入）。

带权残差自助法、稳健/HC 自助法、HAC/分块自助法、非 Gaussian 自助法与 Cox 自助法不受支持，请求时会直接报错。得到的区间描述固定设计、固定调参配置下的抽样波动，不会自动校正变量选择或调参选择带来的额外不确定性。

## 设备与后端

`inference_method` 只选择统计程序，不选择硬件：

- `device="cpu"` 请求 NumPy CPU；
- `device="cuda"` 请求 CuPy CUDA，不可用时直接报错；
- `device="torch"` 请求 Torch CUDA，不可用时直接报错；
- `device="auto"` 允许在受支持且可用的后端之间自动选择。

显式 CUDA/Torch 拟合请求不可用时会报错。选择后重拟合与边际去偏推断复用成功拟合的后端，再把结果整理为 NumPy；无截距同时推断则明确使用主机计算，不能把所有 NumPy 转换都理解为仅为展示。

## 分析权重

直接 Lasso 拟合与受支持的去偏推断，在 NumPy/CuPy/Torch 上使用同一个加权中心化平均损失约定。因此，把所有正分析权重同时乘上同一个常数不会改变统计问题。

训练属性 `rsquared`/`rsquared_adj` 另有一项限制：加权去偏推断后会再次中心化
变换后的工作响应，因此不一定反映原始观测上的加权 R²；基于残差的
`fvalue`/`f_pvalue` 也使用同一错误总离差。应使用原始数据调用
`model.score(X, y, sample_weight=weights)`，并自行确认评价权重有限、非负、
长度正确且总和为正；共享评分方法当前不能可靠地拒绝负权重。
可参见[加权评分示例](elastic-net.md#加权训练诊断)。

`LassoCV` 的自动 `alpha` 网格、各个带权训练折、验证 MSE 以及最终全数据重拟合也遵循同一权重约定。常数正权重会退化为与无权重等价的选择问题。

关于 CV 的选择与最终重拟合语义，见 [交叉验证](../guides/cross-validation.md)。

## 参数

下表列出 `statgpu.linear_model.Lasso` 的全部公开构造参数。方法签名、输入输出形状、已拟合属性、公式输入与继承的辅助方法见[完整 Lasso API 参考](../reference/linear-model-api.md#lasso)。Lasso 不提供 ElasticNet 的 `l1_ratio`、`cov_type` 或 `hac_maxlags` 构造参数。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 平均平方损失下的非负 L1 正则化强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 优化最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | 保存 `coef_delta` / `kkt` 请求，但当前直接拟合忽略此选项，见前述求解限制 |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`；普通 `auto` 解析为 `debiased`。设置 `enable_simultaneous_inference=True` 时应显式使用 `debiased`，构造函数当前拒绝 `auto`。`cpu_ols` 和 `gpu_ols` 暂时作为弃用别名接受 |
| `nodewise_alpha` | `None` | 去偏推断中逐节点 Lasso 的惩罚强度 |
| `n_bootstrap` | `200` | 残差自助法重拟合次数，至少为 2 |
| `bootstrap_random_state` | `None` | 残差自助法随机种子 |
| `enable_simultaneous_inference` | `False` | 是否启用同时推断（仅 `debiased`） |
| `simultaneous_method` | `"maxz_bootstrap"` | 同时推断方法；当前为 `maxz_bootstrap` |
| `simultaneous_alpha` | `0.05` | 同时推断的族错误率水平，必须严格位于 `(0,1)` |
| `simultaneous_n_bootstrap` | `1000` | max-\|Z\| 乘子自助法的抽样次数，必须为正整数 |
| `simultaneous_random_state` | `None` | 同时推断随机种子 |
| `simultaneous_include_intercept` | `False` | 是否把去偏截距纳入同时推断目标集合 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda`（CuPy）/ `torch`（Torch CUDA） |
| `n_jobs` | `None` | 共享 CPU 工作线程设置，不选择求解器，也不保证并行拟合 |
| `compute_inference` | `True` | 是否计算拟合后推断 |
| `solver` | `"fista"` | 与后端无关的直接拟合求解器；CPU 坐标下降使用 `coordinate_descent` |
| `cpu_solver` | `"coordinate_descent"` | **弃用兼容参数**；不再决定直接拟合算法，请改用 `solver` |
| `lipschitz_L` | `None` | 兼容迭代求解器可使用的显式 Lipschitz 常数 |
| `admm_rho` | `1.0` | 当前仅被保存，统一 ADMM 忽略此值，以 rho=1.0 启动。是否自适应调整取决于求解方式；直接 Cholesky 求解保持 rho 不变 |
| `gpu_memory_cleanup` | `False` | 在支持的路径上，拟合后是否请求释放可回收的 GPU 缓存内存 |

## 完整的同时推断示例

这个独立 CPU 示例自行生成数据。区间是在固定 alpha 后计算的，
没有校正同一响应上选择 alpha 所带来的不确定性。

<!-- learner-example: lasso-simultaneous -->
```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(42)
X = rng.normal(size=(180, 6))
y = 1 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=180)
m_sim = Lasso(
    alpha=0.1, device="cpu", solver="coordinate_descent",
    max_iter=5000, tol=1e-10, inference_method="debiased",
    enable_simultaneous_inference=True, simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=200, simultaneous_random_state=7,
    simultaneous_include_intercept=True,
).fit(X, y)
ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
print(ci_marginal.shape, ci_simul.shape)
```

这里两者形状都是 `(7,2)`，截距位于六个斜率之前。200 次抽样让示例保持轻量，
更多抽样可提高蒙特卡洛精度。可选 GPU 执行需要可用的 CuPy/Torch CUDA 后端，
参见[设备指南](../guides/device-and-memory.md)，并注意前述无截距同时推断边界。

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
- `debiased` 且拟合截距时：`_params[1:]` 保存去偏斜率，`_params[0]` 保存匹配的原始坐标系截距；不拟合截距时，所有行均对应输入特征；
- 开启同时推断后：`_conf_int_simultaneous` 保存配置目标集合上的联合区间；
- 方法：`fit`、`predict`、`score`、`summary`、`get_params`、`set_params`，以及继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test`；参数与返回值见 [Lasso API 参考](../reference/linear-model-api.md#lasso)；
- 可用时还包括 `aic`、`bic` 等诊断。

## 常见问题

- **`alpha` 与 `nodewise_alpha` 有什么区别？**  
  `alpha` 定义用于预测/变量选择的主 Lasso 拟合；`nodewise_alpha` 只用于去偏推断中的近似精度矩阵构造。
- **为什么同样 `tol` 下 CPU/GPU 迭代数不同？**  
  不同数值后端和算法可能产生不同收敛轨迹；比较时应固定 `solver`、容差与数据尺度；当前直接拟合的 `stopping` 选项不生效。
- **CPU 用户应该设置 `cpu_solver` 吗？**  
  不应该。直接拟合统一使用 `solver`；`cpu_solver` 只是旧 CPU/GPU 分离接口的弃用兼容参数。
- **应该根据硬件选择 `cpu_ols` 或 `gpu_ols` 吗？**  
  不应该。两者都是 `post_selection_ols` 的弃用别名。统计方法由 `inference_method` 选择，执行位置由 `device` 选择。
- **`post_selection_ols` 会改变 `coef_` 吗？**  
  不会。预测继续使用惩罚系数；活跃集重拟合结果保存在 `_params`、`_inference_result` 等推断字段中。
- **为什么拟合截距且使用 `debiased` 时，`intercept_` 可能和 `_params[0]` 不同？**  
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
