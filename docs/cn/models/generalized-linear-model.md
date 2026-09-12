# GeneralizedLinearModel 与 Penalized GLM

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：模型文档  
> 切换：[English](../../en/models/generalized-linear-model.md)

## 概览

`GeneralizedLinearModel` 是普通 GLM 的统一入口，适用于 Gaussian、binomial、Poisson 等核心 family。`GammaRegression`、`InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression` 等 typed estimator 则在同一套 GLM 基础设施上提供各自的 family/link 行为。

需要正则化时，使用 `PenalizedGeneralizedLinearModel` 或 typed wrapper，例如 `PenalizedLinearRegression`、`PenalizedLogisticRegression`、`PenalizedPoissonRegression`。`Ridge`、`Lasso`、`ElasticNet` 是 sklearn 风格的薄包装，底层仍是 penalized Gaussian regression。

如果你主要关心**带权重的 GLM**，可以先记住下面几条：

- `sample_weight` 在受支持的 GLM 路径上表示**目标函数中的解析权重（analytic objective weights）**；
- weighted loss 用 `sum(sample_weight)` 归一化，因此把全部权重同时乘以一个正数不会改变最优解；
- 显式 `solver="newton"` 或 `solver="lbfgs"` 不会因为加入权重而被悄悄换成其他 solver；
- 显式 `device="cuda"` / `device="torch"` 在受支持路径上会留在对应 GPU backend；
- 不支持的组合会明确报错，而不是静默换 solver 或回退到 CPU。

惩罚 GLM 的系数推断见 [Penalized GLM inference](../guides/penalized-glm-inference.md)；完整 solver 表见 [Solver × Penalty 兼容性矩阵](../guides/solver-penalty-matrix.md)。

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

内部 GLM objective 层为 `statgpu.glm_core`。

## 目标函数与 `sample_weight`

普通无权重 GLM 最小化对应 family 的平均负对数似然：

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta).
$$

在受支持的 weighted 路径上，`sample_weight=w` 会把数据拟合项改为归一化加权平均：

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}.
$$

这个定义有三个直接结果：

1. 所有权重同时乘以同一个正数，不会改变最优解；
2. 权重为 0 的观测不参与数据拟合项；
3. 权重必须是有限的非负数，并且权重总和必须严格大于 0。

Penalized GLM 在数据拟合项上再加入惩罚：

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta),
$$

有 analytic weights 时则为

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}
+ \alpha P(\beta).
$$

截距项不惩罚。`statgpu.glm_core` 有意只处理 GLM objective；Cox、robust、quantile 等非 GLM 目标保留各自独立的统计语义和支持边界。

## 求解器选择

光滑 GLM 在可用时使用维护中的一阶或二阶 solver；非光滑的 penalized objective 使用 FISTA/FISTA-BB 等 proximal/KKT 路径。

常见 direct-fit `solver="auto"` 行为如下：

| 设置 | `solver="auto"` 行为 |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` + NumPy/CPU | exact 闭式 L2 路径 |
| squared error + L1/ElasticNet | FISTA/FISTA-BB 稀疏路径 |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | 维护中的 direct-fit dispatch 选择 Newton |
| 非凸 SCAD/MCP | FISTA + LLA continuation |
| quantile | FISTA / quantile-specific path |

`solver="auto"` 的含义是“使用维护中的自动分发规则”。如果你显式指定 solver，statgpu 要么运行这个受支持的 solver，要么报错；不会静默换成另一个 solver。

设备也遵循同样原则：显式 `device="cuda"` 使用 CuPy CUDA，显式 `device="torch"` 使用 Torch CUDA。Formula parsing 可以在 CPU 上完成，但 fit/predict 的数值计算仍跟随选定 backend。

### 显式 Newton / L-BFGS 的加权支持

在维护中的普通 GLM 路径上，只要对应 family/link 支持该 solver，显式 `solver="newton"` 和 `solver="lbfgs"` 都可以接受**非均匀 analytic weights**。

两个 solver 优化的都是前面给出的归一化 weighted objective，并且同一组权重会贯穿整个优化过程：

- Newton 在 objective、gradient、Hessian 和每一次 Armijo line-search 评估中都使用同一组权重；
- L-BFGS 在初始 gradient、当前 objective、每个 line-search candidate 和接受新点后的 gradient 中都使用同一组权重。

这保证了“带权搜索方向”不会与“无权 line search”混在一起。

加入 `sample_weight` **不会**把显式 Newton/L-BFGS 请求换成 IRLS 或 FISTA；显式 CUDA/Torch 也不会退回 CPU。uniform 或历史上等效于 uniform 的权重继续使用既有 unweighted 数值路径。

#### inverse-power Gamma 的一个例外

`GammaRegression(link="inverse_power")` 有一个有意保留的边界。对真正非均匀的权重，显式 Newton/L-BFGS 要求 `fit_intercept=True`，因为维护中的初始化需要构造一个严格为正、满足该 family/link 定义域的初始 predictor。

因此：

- 非均匀权重 + 显式 Newton/L-BFGS + `fit_intercept=True`：支持；
- 非均匀权重 + 显式 Newton/L-BFGS + `fit_intercept=False`：拟合前明确报错；
- 未传权重、uniform 权重或等效 uniform 权重 + `fit_intercept=False`：继续保持历史 unweighted 行为。

这是一个很窄的初始化边界，不代表 Gamma regression 整体不支持 no-intercept。

#### weighted L-BFGS 的适用范围

weighted L-BFGS 是 **GLM loss 的能力**，不是所有底层 `LossBase` 的统一承诺。robust、quantile、Cox 等直接调用的非 GLM loss 有各自的权重语义，direct L-BFGS 可能拒绝非均匀 `sample_weight`。Ordered GLM 也保留独立的 weight policy。

weighted penalized smooth GLM 使用同一套 analytic-weight convention，但 solver 仍由既有 direct-fit/CV policy 决定：某个 L2 row 使用 Newton 还是 L-BFGS，不是由“是否有权重”决定的。

## 协方差与推断

generic/typed penalized GLM 推荐用 `inference_method="auto"`。成功的 inference-enabled fit 会记录 requested/resolved/reported method、inferential target，以及 tuning/selection conditioning provenance。

对受支持的 smooth non-Gaussian L2/no-penalty 模型，`auto` 解析为 fixed-penalty `m_estimation`。正 L2 penalty 的 target 是 penalized estimating equation；no-penalty alias 会规范成零强度 L2，对应 unpenalized population parameter。当前 covariance 支持 `nonrobust`、`hc0`、`hc1`；HC2/HC3/HAC 在该 penalized non-Gaussian 路径上不可用，并会明确报错。

维护中的推断路径支持 analytic weights，数值 inference 跟随真正执行 fit 的 backend 和具体 device。普通 GLM inference 与 estimation 使用同一个 analytic-weight convention。non-Gaussian L1/ElasticNet coefficient inference 当前尚未产品化。SCAD/MCP oracle inference 需要显式请求；group penalty 与 penalized Cox 仍然只提供 estimation。

对受支持的 Gaussian sparse penalty，`inference_method="bootstrap"` 表示 `cov_type="nonrobust"` 的**无权重 residual bootstrap**。设计矩阵和拟合后的 tuning 配置保持固定：每个 bootstrap draw 对 residual 有放回抽样，在 fitted values 周围构造新的 Gaussian response，然后用同一个 penalized model 重拟合。`n_bootstrap` 控制重拟合次数，`bootstrap_random_state` 控制可复现性。

bootstrap 的执行位置跟随成功拟合使用的 backend 和具体 device。CPU fit 使用 NumPy；CuPy/Torch CUDA fit 会把 bootstrap refit 留在同一个 GPU device。最终 inference arrays 使用统一的 NumPy reporting boundary。weighted residual bootstrap、robust/HC 或 HAC/block bootstrap、non-Gaussian bootstrap 与 Cox bootstrap 当前都不支持。

完整 support matrix、resampling 边界和统计解释见 [Penalized GLM inference](../guides/penalized-glm-inference.md) 与 [Inference Modes](../guides/inference-modes.md)。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `family` | model-specific | GLM family，例如 `"gaussian"`、`"binomial"`、`"poisson"` |
| `penalty` | `"l2"` 或模型默认 | `none`、`l1`、`l2`、`elasticnet` 以及预留 structured penalties |
| `alpha` | `1.0` 或模型默认 | statgpu 目标函数尺度下的惩罚强度 |
| `l1_ratio` | `None` | ElasticNet 的 L1/L2 混合比例 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `solver` | `"auto"` | solver 调度；见上文“求解器选择” |
| `device` | `"auto"` | 根据 estimator 支持情况使用 `cpu`、`cuda`、`torch` 或 `auto` |
| `max_iter` | model-specific | 最大迭代次数 |
| `tol` | model-specific | 收敛阈值 |
| `compute_inference` | generic/typed penalized GLM 默认为 `False` | 是否计算 coefficient inference |
| `inference_method` | generic/typed penalized GLM 默认为 `"auto"` | public inference request；specialized sparse-Gaussian wrapper 保留既有默认 |
| `cov_type` | `"nonrobust"` | covariance convention；non-Gaussian penalized M-estimation 当前支持 nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | 保留控制；不代表 non-Gaussian penalized M-estimation 已支持 HAC |
| `formula` | `None` | 可选 patsy 风格公式，需要与 `data` 一起使用 |
| `data` | `None` | formula 模式下的数据表 |

alpha scaling 必须显式对齐，不能直接比较不同框架中的同名参数：

- Ridge：`sklearn_alpha = n_samples * statgpu_alpha`
- Logistic L2：`sklearn_C = 1 / (n_samples * statgpu_alpha)`
- Poisson L2：与 sklearn `PoissonRegressor(alpha=...)` 对齐
- Poisson L1/ElasticNet：与 statsmodels `fit_regularized` 对齐

## CPU + GPU 示例

```python
from statgpu.linear_model import GeneralizedLinearModel, PenalizedLogisticRegression

# 普通 weighted Poisson GLM，显式选择 smooth solver。
weighted_pois = GeneralizedLinearModel(
    family="poisson",
    solver="lbfgs",       # 也可以显式使用 "newton"
    device="cuda",        # CuPy CUDA；"torch" 表示 Torch CUDA
)
weighted_pois.fit(X, y_count, sample_weight=weights)

# CPU L2 logistic：auto 使用维护中的 direct-fit solver。
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary, sample_weight=weights)

# GPU L2 logistic 使用相同的 weighted objective。
logit_gpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cuda",
)
logit_gpu.fit(X, y_binary, sample_weight=weights)
```

Formula 是可选依赖：

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

Formula 解析在 CPU 上完成，适合作为便利建模层。formula 与 `sample_weight` 同时使用时，weights 会先对齐到 formula/missing-data 处理后真正保留下来的行，再进入 numerical fit。大规模 GPU 任务建议直接传入显式 `X, y` 数组。

## 严格与近似 CV

Penalized GLM inference 采用 fail-closed 规则：受支持的 non-Gaussian L2/no-penalty row 提供 fixed-penalty M-estimation（nonrobust/HC0/HC1）；不支持的 loss × penalty × method 组合直接报错，不会自动替换成另一个统计 procedure。residual bootstrap 与 SCAD/MCP oracle 都保持窄而显式的支持范围。

`solver="auto"` 遵循维护中的 direct-fit dispatch。analytic weights 不会改写 public solver request：显式 smooth solver 仍保持显式，`auto` 则继续按对应拟合路径的维护规则分发。

`PenalizedGLM_CV` 默认使用 `cv_strategy="strict"`。strict 模式下，每个 fold/alpha 都使用用户给定的 `max_iter` 与 `tol`；GPU 优化只做缓存、fused kernel 和 validation score 批量传输。

可选的 `cv_strategy="two_stage"` 会先用放松的 CV 求解筛选 alpha grid，再对候选 alpha 做 strict 复核，并进行 strict final refit。由于筛选阶段在 CV 曲线很接近时可能改变 alpha 排名，two-stage 默认发出 `ApproximateCVWarning`；如果用户明确接受这种近似，可传入 `acknowledge_approx=True`。

```python
from statgpu.linear_model import PenalizedGLM_CV

# 默认：strict CV。
strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

# 显式启用 approximate screening；候选复核与最终 refit 仍保持 strict。
fast_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="two_stage",
    acknowledge_approx=True,
    refine_top_k=3,
    device="cuda",
)
```

### 生存分析的 penalized Cox CV

`PenalizedGLM_CV(loss="cox_ph")` 使用独立的 survival 路径，不进入标量响应 GLM scorer。`y` 必须是 `(n_samples, 2)` 数组，两列依次为 `[time, event]`。L1、L2、ElasticNet、SCAD、MCP 支持 NumPy、CuPy CUDA 和 Torch CUDA。

该路径会：

- 保留二维 survival target，且不拟合截距；
- 用未惩罚的逐行 Cox 负 partial likelihood 评价 held-out fold；
- 只有当每个可评估 fold 都提供有限证据时才选择 alpha；
- 如果所有候选都无效，则直接失败且不发布拟合状态；
- 最终以 `compute_inference=False` 重拟合 `PenalizedCoxPHModel`。

```python
survival_y = np.column_stack([time, event])
cox_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="scad",              # l1、l2、elasticnet、scad 或 mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # 也可以用 "cuda" / "torch"
).fit(X, survival_y)
```

该 Cox 分支不支持 `cv_strategy="two_stage"`、`sample_weight`、字典 target 或 post-selection coefficient inference。`cv_results_` 会记录逐 fold loss、有效证据数、event 数、失败原因、ties 方法和最终重拟合模型类型。

## 输出

常见拟合属性和方法包括：

- `coef_`
- `intercept_`
- `n_iter_`（取决于 solver 是否暴露）
- `fit`
- `predict`
- `predict_proba`（logistic 模型）
- `score`（对应模型实现时）
- `cv_results_`（`PenalizedGLM_CV`），包括 `cv_strategy_`、`cv_selected_device_`、`refined_mask`，以及 two-stage screening 启用时的 stage-1 scores

统一 `FitResult` 是未来预留，不属于本页当前 public contract。

## 相关文档

- [Solver × Penalty 兼容性矩阵](../guides/solver-penalty-matrix.md) — loss × penalty × solver 的完整分发表、CV 路径和推断支持状态。
- [求解器算法](../guides/solver-algorithms.md) — Newton、L-BFGS、FISTA 等 solver 的算法说明。
- [损失函数](losses.md) — 底层 loss 接口及其能力边界。

## 常见问题

- **这里的 `sample_weight` 表示什么？** 在受支持的 ordinary/penalized GLM 路径上，它表示目标函数中的 analytic weight，并按 `sum(weights)` 归一化。它不是 survey-bootstrap weight，也不会自动请求 weighted residual bootstrap。
- **加入 `sample_weight` 会改变 solver 吗？** 不会。显式 solver 保持显式；`solver="auto"` 继续使用维护中的 dispatch policy。
- **`device="cuda"` 是否保证 GLM solver 在 GPU 上运行？** 对受支持的 route，是的：数值计算使用 CuPy。如果 route 不支持或设备不可用，会明确报错，不会静默回 CPU。
- **大规模 GPU 任务是否建议使用 formula？** 通常不建议。formula 是 CPU 侧便利层，大规模任务更适合显式数组。
- **`Ridge`、`Lasso`、`ElasticNet` 是 alias 吗？** 不是。它们是保留 sklearn 风格 constructor 的薄包装类。

## 外部验证

本地和 hosted checks 覆盖 import、solver/objective invariants、CPU reference 与 regression matrix。GPU 数值一致性和 concrete-device 行为在发布能力声明前由维护中的 physical-CUDA validator 单独验证。

验证范围包括：

- CPU/CuPy/Torch coefficient 与 intercept 一致性；
- analytic-weight 的全局缩放、uniform-weight 与 zero-weight-row 恒等性；
- penalized path 的 objective gap 与 KKT residual；
- 在目标函数定义可对齐时与 sklearn/statsmodels 进行对比；
- 性能测试需要时使用 warm-up 和 GPU synchronization。

这些 developer validation assets 与上文的用户 API 分开理解。远程凭据必须通过环境变量提供，不得写入代码或文档。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
