# GeneralizedLinearModel 与 Penalized GLM

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：模型文档  
> 切换：[English](../../en/models/generalized-linear-model.md)

## 概览

`GeneralizedLinearModel` 是普通广义线性模型（GLM）的统一入口，适用于 Gaussian、binomial、Poisson 等常见分布族。`GammaRegression`、`InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression` 等具体模型类在同一套 GLM 基础设施上提供各自的分布族和链接函数行为。

需要正则化时，可以使用 `PenalizedGeneralizedLinearModel`，或 `PenalizedLinearRegression`、`PenalizedLogisticRegression`、`PenalizedPoissonRegression` 等具体封装类。`Ridge`、`Lasso`、`ElasticNet` 是 sklearn 风格的便捷封装，底层仍使用带惩罚的 Gaussian 回归。

如果主要关心**带权重的 GLM**，先记住四点即可：

- `sample_weight` 在受支持的 GLM 路径中表示**目标函数中的解析权重（analytic weights）**；
- 加权损失按 `sum(sample_weight)` 归一化，因此把全部权重同时乘以同一个正数不会改变最优解；
- `sample_weight` 不会改变显式指定的 `solver`；如果组合不受支持，则直接报错；
- 显式 `device="cuda"` 与 `device="torch"` 在受支持的路径中分别使用 CuPy CUDA 与 Torch CUDA。

带惩罚 GLM 的系数推断见 [Penalized GLM inference](../guides/penalized-glm-inference.md)；完整求解器表见 [Solver × Penalty 兼容性矩阵](../guides/solver-penalty-matrix.md)。

## 公开入口

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.GammaRegression`
- `statgpu.linear_model.InverseGaussianRegression`
- `statgpu.linear_model.NegativeBinomialRegression`
- `statgpu.linear_model.TweedieRegression`
- `statgpu.linear_model.PenalizedGeneralizedLinearModel`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`

内部 GLM 目标函数层为 `statgpu.glm_core`。

## 目标函数与 `sample_weight`

普通无权重 GLM 最小化对应分布族的平均负对数似然：

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta).
$$

在受支持的加权路径中，`sample_weight=w` 会把数据拟合项改为归一化加权平均：

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}.
$$

这个定义有三个直接结果：

1. 所有权重同时乘以同一个正数，不会改变最优解；
2. 权重为 0 的观测不参与数据拟合项；
3. 权重必须是有限的非负数，并且总和必须严格大于 0。

带惩罚 GLM 在数据拟合项上再加入惩罚：

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta),
$$

使用解析权重时则为

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}
+ \alpha P(\beta).
$$

截距项不参与惩罚。`statgpu.glm_core` 有意只处理 GLM 目标；Cox、稳健回归、分位数回归等非 GLM 目标保留各自独立的统计定义和支持范围。

## 求解器选择

对大多数用户，推荐先使用 `solver="auto"`。当前直接拟合中的主要分发规则如下：

| 设置 | `solver="auto"` 行为 |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` + NumPy/CPU | `exact` 闭式 L2 路径 |
| squared error + L1/ElasticNet | FISTA/FISTA-BB 稀疏路径 |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | Newton |
| 非凸 SCAD/MCP | FISTA + LLA 延续路径 |
| quantile | FISTA 或分位数专用路径 |

`solver="auto"` 表示使用当前模型的自动分发规则。若显式指定 `solver`，该请求不会因 `sample_weight` 而改变；如果对应组合不受支持，则拟合会报错。

设备选择也遵循同样原则：显式 `device="cuda"` 使用 CuPy CUDA，显式 `device="torch"` 使用 Torch CUDA。公式解析可以在 CPU 上进行，但拟合与预测的数值计算仍使用选定的计算后端。

### 显式 Newton / L-BFGS 的加权支持

在受支持的普通 GLM 路径中，只要对应分布族和链接函数支持该求解器，显式 `solver="newton"` 和 `solver="lbfgs"` 都可以接受**非均匀解析权重**。

两个求解器都优化前面给出的归一化加权目标，并且同一组权重贯穿整个优化过程：

- Newton 在目标函数、梯度、Hessian 和每一次 Armijo 线搜索评估中使用同一组权重；
- L-BFGS 在初始梯度、当前目标函数、每个线搜索候选点以及接受新点后的梯度中使用同一组权重。

因此，加权搜索方向与线搜索使用的是同一个目标函数。

`sample_weight` 不会更换显式指定的 Newton 或 L-BFGS。若所请求的加权组合不受支持，则直接报错。均匀权重，以及历史上与均匀权重等效的情况，继续使用既有的无权重数值路径。

#### inverse-power Gamma 的当前限制

`GammaRegression(link="inverse_power")` 目前有一个较窄的初始化限制。对真正的非均匀权重，显式 Newton/L-BFGS 要求 `fit_intercept=True`，因为当前初始化需要构造一个严格为正、满足该分布族和链接函数定义域的初始线性预测子。

因此：

- 非均匀权重 + 显式 Newton/L-BFGS + `fit_intercept=True`：支持；
- 非均匀权重 + 显式 Newton/L-BFGS + `fit_intercept=False`：当前会在拟合前报错；
- 未传权重、均匀权重或等效均匀权重 + `fit_intercept=False`：保持历史无权重行为。

这属于**当前实现的可行初值限制**，不是 Gamma inverse-power 模型本身的理论限制。后续支持由 [GitHub issue #152](https://github.com/TheHiddenObserver/statgpu/issues/152) 跟踪。

#### 带权 L-BFGS 的适用范围

非均匀权重下的 L-BFGS 支持目前是 **GLM 损失函数的明确能力**，并不自动扩展到所有底层 `LossBase`。稳健回归、分位数回归、Cox 等非 GLM 损失函数有各自的权重定义，直接调用 L-BFGS 时可能不接受非均匀 `sample_weight`。Ordered GLM 也保留独立的权重规则。

带惩罚的光滑 GLM 使用同一套解析权重定义，但求解器仍由直接拟合或交叉验证的既有分发规则决定。是否提供权重不会单独决定一个 L2 模型使用 Newton 还是 L-BFGS。

## 协方差与推断

对于通用的 `PenalizedGeneralizedLinearModel` 以及各类专用的带惩罚 GLM 封装，推荐使用 `inference_method="auto"`。在支持推断的拟合成功后，模型会记录用户请求的方法、实际采用的方法、报告的方法、推断目标，以及调参与模型选择所依赖的条件信息。

对于受支持的光滑非高斯 L2/无惩罚模型，`auto` 解析为固定惩罚的 `m_estimation`。正 L2 惩罚对应带惩罚的估计方程；无惩罚别名会规范为惩罚强度为 0 的 L2，对应无惩罚总体参数。当前协方差支持 `nonrobust`、`hc0`、`hc1`；HC2/HC3/HAC 在这一路径中不可用，并会明确报错。

受支持的推断路径可以使用解析权重，数值计算跟随实际执行拟合的计算后端和具体设备。普通 GLM 的推断与拟合使用同一套权重定义。非高斯 L1/ElasticNet 的系数推断目前尚未产品化；SCAD/MCP 的 oracle 型推断需要显式请求；组惩罚与带惩罚 Cox 目前只提供估计。

对于受支持的高斯稀疏惩罚，`inference_method="bootstrap"` 表示 `cov_type="nonrobust"` 的**无权重残差自助法（residual bootstrap）**。设计矩阵和拟合后的调参配置保持固定：每次从残差中有放回抽样，在拟合值周围构造新的高斯响应，再用同一个带惩罚模型重新拟合。`n_bootstrap` 控制重拟合次数，`bootstrap_random_state` 控制随机数可复现性。

CPU 拟合使用 NumPy；CuPy/Torch CUDA 拟合会把自助法重拟合留在同一个 GPU 设备。最终推断数组统一返回 NumPy。带权残差自助法、稳健/HC 协方差对应的自助法、HAC/分块自助法、非高斯自助法与 Cox 自助法当前都不支持。

完整支持矩阵、重采样边界和统计解释见 [Penalized GLM inference](../guides/penalized-glm-inference.md) 与 [Inference Modes](../guides/inference-modes.md)。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `family` | 模型相关 | GLM 分布族，例如 `"gaussian"`、`"binomial"`、`"poisson"` |
| `penalty` | `"l2"` 或模型默认 | `none`、`l1`、`l2`、`elasticnet` 以及结构化惩罚 |
| `alpha` | `1.0` 或模型默认 | statgpu 目标函数尺度下的惩罚强度 |
| `l1_ratio` | `None` | ElasticNet 的 L1/L2 混合比例 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `solver` | `"auto"` | 求解器；见上文“求解器选择” |
| `device` | `"auto"` | 根据模型支持情况使用 `cpu`、`cuda`、`torch` 或 `auto` |
| `max_iter` | 模型相关 | 最大迭代次数 |
| `tol` | 模型相关 | 收敛阈值 |
| `compute_inference` | 通常为 `False` | 是否计算系数推断 |
| `inference_method` | 通常为 `"auto"` | 推断方法请求 |
| `cov_type` | `"nonrobust"` | 协方差类型；非高斯带惩罚 M-估计当前支持 nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | HAC 相关控制；不代表上述路径已支持 HAC |
| `formula` | `None` | 可选 patsy 风格公式，需要与 `data` 一起使用 |
| `data` | `None` | 公式模式下的数据表 |

不同框架中的同名正则化参数未必采用同一尺度。常见换算包括：

- Ridge：`sklearn_alpha = n_samples * statgpu_alpha`
- Logistic L2：`sklearn_C = 1 / (n_samples * statgpu_alpha)`
- Poisson L2：与 sklearn `PoissonRegressor(alpha=...)` 对齐
- Poisson L1/ElasticNet：与 statsmodels `fit_regularized` 对齐

## CPU + GPU 示例

```python
from statgpu.linear_model import GeneralizedLinearModel, PenalizedLogisticRegression

# 带权 Poisson GLM，显式使用 L-BFGS；也可以改为 Newton。
weighted_pois = GeneralizedLinearModel(
    family="poisson",
    solver="lbfgs",
    device="cuda",
)
weighted_pois.fit(X, y_count, sample_weight=weights)

# CPU L2 logistic：让 auto 按直接拟合规则选择求解器。
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary, sample_weight=weights)

# GPU L2 logistic 使用相同的加权目标函数。
logit_gpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cuda",
)
logit_gpu.fit(X, y_binary, sample_weight=weights)
```

公式功能是可选依赖：

```bash
pip install statgpu[formula]
```

```python
from statgpu.linear_model import LinearRegression, PenalizedPoissonRegression

lm = LinearRegression()
lm.fit(formula="y ~ x1 + x2 + C(group)", data=df)
pred = lm.predict(df_new)

pois = PenalizedPoissonRegression(penalty="l2", alpha=0.01)
pois.fit(formula="count ~ exposure + x1", data=df)
```

公式解析在 CPU 上完成。若同时使用 `formula` 和 `sample_weight`，权重会先与公式和缺失值处理后实际保留的观测行对齐，再进入数值拟合。大规模 GPU 任务通常更适合直接传入 `X, y` 数组。

## 严格与近似 CV

`PenalizedGLM_CV` 默认使用 `cv_strategy="strict"`。严格模式下，每个交叉验证折和每个 `alpha` 都使用用户给定的 `max_iter` 与 `tol`；GPU 优化只改变实现效率，不改变候选模型的求解标准。

可选的 `cv_strategy="two_stage"` 会先用较宽松的求解条件筛选 `alpha` 网格，再严格复核候选 `alpha`，并进行严格的最终重拟合。由于第一阶段可能在非常接近的 CV 曲线上改变 `alpha` 排名，该模式默认发出 `ApproximateCVWarning`；如果用户明确接受这种近似，可以传入 `acknowledge_approx=True`。

```python
from statgpu.linear_model import PenalizedGLM_CV

strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

fast_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="two_stage",
    acknowledge_approx=True,
    refine_top_k=3,
    device="cuda",
)
```

### 生存分析的带惩罚 Cox 交叉验证

`PenalizedGLM_CV(loss="cox_ph")` 使用独立的生存分析路径，不进入标量响应 GLM 的评分器。`y` 必须是 `(n_samples, 2)` 数组，两列依次为 `[time, event]`。L1、L2、ElasticNet、SCAD、MCP 均支持 NumPy、CuPy CUDA 和 Torch CUDA。

该路径会：

- 保留二维生存目标，且不拟合截距；
- 用未惩罚的逐行 Cox 负部分似然评价验证折；
- 只有在每个可评估折都得到有限数值时才选择 `alpha`；
- 如果所有候选都无效，则直接失败且不发布拟合状态；
- 最终以 `compute_inference=False` 重拟合 `PenalizedCoxPHModel`。

```python
survival_y = np.column_stack([time, event])
cox_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="scad",
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",
).fit(X, survival_y)
```

该 Cox 分支不支持 `cv_strategy="two_stage"`、`sample_weight`、字典形式的目标变量或选择后的系数推断。`cv_results_` 会记录逐折损失、有效证据数、事件数、失败原因、并列事件处理方法（ties）和最终重拟合模型类型。

## 输出

常见拟合属性和方法包括：

- `coef_`
- `intercept_`
- `n_iter_`（取决于求解器是否提供）
- `fit`
- `predict`
- `predict_proba`（逻辑回归模型）
- `score`（对应模型实现时）
- `cv_results_`（`PenalizedGLM_CV`），包括 `cv_strategy_`、`cv_selected_device_`、`refined_mask`，以及两阶段筛选启用时的第一阶段分数

统一 `FitResult` 是未来预留，不属于本页当前公开接口。

## 相关文档

- [Solver × Penalty 兼容性矩阵](../guides/solver-penalty-matrix.md) — 损失函数 × 惩罚项 × 求解器的完整分发表、交叉验证路径和推断支持状态。
- [求解器算法](../guides/solver-algorithms.md) — Newton、L-BFGS、FISTA 等求解器的算法说明。
- [损失函数](losses.md) — 底层损失函数接口及其能力边界。

## 常见问题

- **这里的 `sample_weight` 表示什么？** 在受支持的普通或带惩罚 GLM 路径中，它表示目标函数中的解析权重，并按 `sum(weights)` 归一化。它不是调查抽样权重，也不是用于自助法抽样的权重；同时不会自动启用带权残差自助法。
- **加入 `sample_weight` 会改变求解器吗？** 不会。显式指定的 `solver` 保持不变；`solver="auto"` 继续按照对应模型的自动分发规则选择求解器。
- **`device="cuda"` 是否保证 GLM 求解器在 GPU 上运行？** 对受支持的路径，是的：数值计算使用 CuPy；若组合不受支持或设备不可用，则报错。
- **大规模 GPU 任务是否建议使用 `formula`？** 通常不建议。公式接口主要用于便利建模，大规模任务更适合显式数组。
- **`Ridge`、`Lasso`、`ElasticNet` 是别名吗？** 不是。它们是保留 sklearn 风格构造器语义的薄封装类。

## 外部验证

本地与托管测试覆盖导入、求解器/目标函数不变量、CPU 参考结果与回归矩阵。GPU 数值一致性和具体设备行为在发布能力声明前由物理 CUDA 验证程序单独检查。

验证范围包括：

- CPU/CuPy/Torch 的系数与截距一致性；
- 解析权重的全局缩放、均匀权重与零权重观测恒等性；
- 带惩罚路径的目标函数差异与 KKT 残差；
- 在目标函数定义可对齐时与 sklearn/statsmodels 进行比较；
- 性能测试需要时使用预热与 GPU 同步。

这些开发侧验证结果与上文用户 API 分开理解。远程凭据必须通过环境变量提供，不得写入代码或文档。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.