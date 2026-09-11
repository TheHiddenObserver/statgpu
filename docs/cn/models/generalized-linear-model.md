# GeneralizedLinearModel 与 Penalized GLM

> 语言: 中文  
> 最后更新: 2026-09-11
> 页面定位: 模型文档  
> 切换: [English](../../en/models/generalized-linear-model.md)

语言切换: [English](../../en/models/generalized-linear-model.md)

## Overview

`GeneralizedLinearModel` 是普通 GLM 的统一入口，当前覆盖 Gaussian、binomial、Poisson 等普通 GLM family。`PenalizedGeneralizedLinearModel` 及 typed wrapper 用于带惩罚项的 GLM，并为 L1、L2、ElasticNet、group、adaptive 等 penalty 预留统一接口。

推荐用户使用 typed penalized estimator：

```python
from statgpu.linear_model import (
    GeneralizedLinearModel,
    PoissonRegression,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
)
```

`Ridge`、`Lasso`、`ElasticNet` 是 sklearn 风格的薄包装，内部走 penalized Gaussian regression。

penalized coefficient inference 的完整统计口径见 [Penalized GLM inference](../guides/penalized-glm-inference.md)。

## Path

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.PenalizedGeneralizedLinearModel`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`
- 内部 GLM core：`statgpu.glm_core`

## Objective Function

普通 GLM 最小化对应 family 的平均负对数似然：

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta)
$$

Penalized GLM 在此基础上加入惩罚项：

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta)
$$

截距项不惩罚。`statgpu.glm_core` 只表示 GLM 专用核心层；Cox partial likelihood、panel objective、time-series likelihood、zero-inflated composite likelihood 不应强行塞入 `glm_core`，后续应通过更通用的 objective 层共享底层能力。

## Estimating Equation

smooth GLM 在可用时使用维护中的二阶/一阶优化路径；非平滑 penalized objective 使用 FISTA/FISTA-BB 等 proximal/KKT 路径。

当前 direct-fit `solver="auto"` 的主要行为：

| 设置 | `solver="auto"` 行为 |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` on NumPy/CPU | exact 闭式 L2 路径 |
| squared error + L1/ElasticNet | FISTA/FISTA-BB sparse path |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | 维护中的 direct-fit dispatch 选择 Newton |
| non-convex SCAD/MCP | FISTA + LLA continuation |
| quantile | FISTA / quantile-specific path |

显式 `device="cuda"` 保持 CuPy，显式 `device="torch"` 保持 Torch CUDA；不受支持的显式 solver/backend 组合会直接报错，不静默回 CPU。Formula parsing 可以在 CPU 上进行，但 fit/predict 数值计算跟随 selected backend。

一个 inference-specific execution choice 会明确暴露：inference-enabled、weighted、non-Gaussian L2/no-penalty 且 public `solver="auto"` 时，使用已有 weight-capable FISTA，因为 Newton 当前拒绝 non-uniform analytic weights。public request 仍保持 `auto`，estimation-only benchmark dispatch 不变。

## Covariance/Inference

generic 与 typed penalized GLM estimator 推荐使用 `inference_method="auto"` 作为 public request。成功的 inference-enabled fit 会公开 requested/resolved/reported method、inferential target 以及 tuning/selection conditioning provenance。

对受支持的 smooth non-Gaussian L2/no-penalty 模型，`auto` 解析为 fixed-penalty `m_estimation`。正 L2 penalty 的 target 是 penalized estimating equation；no-penalty alias 会 canonicalize 成零强度 L2，对应 unpenalized population parameter。当前 covariance 支持 `nonrobust`、`hc0`、`hc1`；HC2/HC3/HAC 在该 penalized non-Gaussian path 上 fail closed。

analytic weights 受支持，数值 inference 跟随真正执行 fit 的 backend/concrete device。non-Gaussian L1/ElasticNet coefficient inference 当前不 productize，会 fail closed。SCAD/MCP oracle 必须显式请求；group penalty 与 penalized Cox 仍为 estimation-only。

本 contract 的 `inference_method="bootstrap"` 只表示 `cov_type="nonrobust"` 的 unweighted CPU Gaussian residual bootstrap：保留真实 penalty/tuning，至少需要 2 次 resample，且 CuPy/Torch 已执行拟合不会静默切到 CPU 重拟合。

完整 support matrix、resampling 边界与统计解释见 [Penalized GLM inference](../guides/penalized-glm-inference.md) 与 [Inference Modes](../guides/inference-modes.md)。

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `family` | model-specific | GLM family，例如 `"gaussian"`、`"binomial"`、`"poisson"` |
| `penalty` | `"l2"` 或模型默认 | `none`、`l1`、`l2`、`elasticnet` 以及预留 structured penalties |
| `alpha` | `1.0` 或模型默认 | statgpu 目标函数尺度下的惩罚强度 |
| `l1_ratio` | `None` | ElasticNet 的 L1/L2 混合比例 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `solver` | `"auto"` | solver 调度规则见 Estimating Equation |
| `device` | `"auto"` | 按 estimator 支持情况使用 `cpu`、`cuda`、`torch` 或 `auto` |
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

## CPU+GPU Examples

```python
from statgpu.linear_model import GeneralizedLinearModel, PenalizedLogisticRegression

# 普通 Poisson GLM；所选路径支持 GPU 时可在 GPU 上运行。
glm = GeneralizedLinearModel(family="poisson", device="cuda")
glm.fit(X, y_count)

# CPU L2 logistic 路径：auto 选择 Newton。
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary)

# GPU L2 logistic 路径：auto 选择 backend-native Newton。
logit_gpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cuda",
)
logit_gpu.fit(X, y_binary)
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

Formula 解析在 CPU 上完成，适合作为便利建模层。大规模 GPU 任务建议直接传入显式 `X, y` 数组。

## strict/approx difference

Penalized GLM inference 采用 fail-closed contract：受支持的 non-Gaussian L2/no-penalty 行公开 fixed-penalty M-estimation（nonrobust/HC0/HC1）；不受支持的 loss × penalty × method 组合直接报错，不替换成另一个统计 procedure。residual bootstrap 与 SCAD/MCP oracle 都保持窄而显式的边界。

`solver="auto"` 遵循维护中的 direct-fit dispatch（smooth non-Gaussian L2 包括 Newton）。inference-enabled weighted non-Gaussian L2/no-penalty fit 则使用上文说明的 fit-local weight-capable FISTA，同时 public `auto` request 不变。

`PenalizedGLM_CV` 默认使用 `cv_strategy="strict"`。strict 模式下，每个 fold/alpha 都使用用户传入的 `max_iter` 与 `tol`；GPU 优化只做缓存、fused kernel 和 validation score 批量传输，不做 alpha 粗筛。可选的 `cv_strategy="two_stage"` 会先用放松的 CV 求解筛选 alpha grid，再对候选 alpha 做 strict 复核，并且最终 refit 仍然是 strict/full-iteration。由于粗筛阶段在 CV 曲线很接近时可能改变 alpha 排名，two-stage 模式默认发出 `ApproximateCVWarning`；如果用户已确认接受该近似，可传入 `acknowledge_approx=True` 静默该 warning。

```python
from statgpu.linear_model import PenalizedGLM_CV

# 默认: strict CV。
strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

# 显式启用 approximate screening；候选复核和最终 refit 仍使用 strict。
fast_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="two_stage",
    acknowledge_approx=True,
    refine_top_k=3,
    device="cuda",
)
```

### 生存感知的惩罚 Cox 交叉验证

`PenalizedGLM_CV(loss="cox_ph")` 使用独立的生存分析路径，不会进入标量响应
GLM scorer。`y` 必须是 `(n_samples, 2)` 数组，两列依次为 `[time, event]`。
L1、L2、ElasticNet、SCAD 与 MCP 均支持 NumPy、CuPy CUDA 和 Torch CUDA。
该路径会：

- 保留二维生存目标，且绝不拟合截距；
- 用未惩罚的逐行 Cox 负 partial likelihood 评价 held-out fold；
- 仅在每个可评估 fold 都提供有限证据时选择 alpha；
- 所有候选均无效时 hard-fail，且不发布任何拟合状态；
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
    device="cpu",                # 也可用 "cuda" / "torch"
).fit(X, survival_y)
```

该 Cox 分支不支持 `cv_strategy="two_stage"`、`sample_weight`、字典 target
或 post-selection 系数推断。`cv_results_` 会记录逐 fold loss、有效证据数、
event 数、失败原因、ties 方法和最终重拟合模型类型。

## Outputs

常见拟合属性和方法包括：

- `coef_`
- `intercept_`
- `n_iter_`，取决于 solver 是否暴露
- `fit`
- `predict`
- `predict_proba`，用于 logistic 模型
- `score`，在对应模型中提供
- `cv_results_`，用于 `PenalizedGLM_CV`，包含 `cv_strategy_`、`cv_selected_device_`、`refined_mask`，以及两阶段筛选启用时的 stage-1 scores

统一 `FitResult` 是未来预留，不属于本页当前 public contract。

## 参见

- [Solver × Penalty 兼容性矩阵](../guides/solver-penalty-matrix.md) — loss × penalty × solver 完整分发表、CV 快速路径、推断支持状态。

## FAQ

- 为什么不保留 `statgpu.losses` 作为兼容入口？因为未提交的 `losses` 层实际只服务 GLM，改名为 `glm_core` 可以避免误导为全项目通用 objective 系统。
- `device="cuda"` 是否强制所有 GLM solver 使用 GPU？对已支持的 GLM solver 路径，是的：核心计算使用 CuPy；如果依赖或设备不可用，会清晰报错，不会静默回落到 CPU。
- 大规模 GPU 数据是否建议使用 formula？通常不建议。formula 是 CPU 侧便利层，大规模任务应使用显式数组。
- `Ridge`、`Lasso`、`ElasticNet` 是 alias 吗？不是。它们是薄包装类，用于保留清晰的 sklearn 风格构造器语义。

## External Validation

本地只做 import 和 smoke 检查。accuracy、runtime、GPU 行为和外部框架对比统一放在远程 `myconda` 环境。

**v23c 全矩阵基准测试 (2026-05-20):** 1043/1043 ALL PASS，覆盖 7 families x 10 penalties x 3 规模 x 3 backends，vs sklearn 和 vs statsmodels 全部通过。详见 `dev/tests/_bench_v23c_report.md` 和 `dev/tests/_bench_full_matrix.py`。
- Gaussian penalized 与 sklearn Ridge/Lasso/ElasticNet 对比。
- Logistic 与 sklearn 对比。
- Poisson L2 与 sklearn 对比。
- Poisson L1/ElasticNet 与 statsmodels `fit_regularized` 对比。
- 含 warm-up 与 GPU synchronization 的 runtime benchmark。

历史 v23c matrix 属于 estimation evidence，并不能单独证明新的 coefficient-inference contract。PR #142 另有 targeted hosted inference tests 与 exact-source physical CuPy/Torch CUDA validator；在该 validator 真正执行前，不宣称已有 physical GPU pass。

远程凭据必须从环境变量读取，不得写入代码或文档。

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
