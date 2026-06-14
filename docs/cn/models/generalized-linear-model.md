# GeneralizedLinearModel 与 Penalized GLM

> 语言: 中文  
> 最后更新: 2026-05-20  
> 页面定位: 模型文档  
> 切换: [English](../en/models/generalized-linear-model.md)

语言切换: [English](../en/models/generalized-linear-model.md)

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

光滑 GLM 在可用时通过 IRLS/Newton 风格更新求解 score equation。带非光滑惩罚项的目标函数通过 FISTA 等 proximal/KKT 风格优化路径求解。

当前 `solver="auto"` 的行为如下：

| 设置 | `solver="auto"` 行为 |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` | `solver="exact"` 闭式 L2 路径 |
| `PenalizedLinearRegression(penalty="l1"|"elasticnet")` | FISTA |
| `PenalizedLogisticRegression(penalty="l2")` on NumPy/CPU | IRLS |
| `PenalizedPoissonRegression(penalty="l2")` on NumPy/CPU | IRLS |
| `PenalizedLogisticRegression(penalty="l2")` on CuPy/Torch GPU | FISTA |
| `PenalizedPoissonRegression(penalty="l2")` on CuPy/Torch GPU | FISTA |
| 显式 `solver="irls"` | NumPy/CuPy/Torch 对应后端原生 IRLS |
| 显式 `solver="newton"` | smooth objective 上使用对应后端 Newton |
| 显式 `solver="lbfgs"` | smooth objective 上使用对应后端 L-BFGS（所有 GLM family + L2/ElasticNet） |
| 非光滑 penalty 搭配 `solver="newton"` 或 `solver="lbfgs"` | 直接抛出 `ValueError` |

重要设备规则：显式 `device="cuda"` 会保持 CuPy 核心计算，显式 `device="torch"` 会保持 Torch CUDA 核心计算；显式 solver 不允许静默回落到 CPU。Formula/DataFrame 解析可以作为 CPU 预处理，但 fit/predict 核心计算必须转换到所选后端。

## Covariance/Inference

本页覆盖的是估计主线的 GLM 与 penalized GLM API。新的 penalized GLM 层尚未暴露完整 strict inference 输出。当前 inference 较完整的模型仍分别见独立页面：

- `LinearRegression`：classical、HC0-HC3、HAC。
- `Ridge`：classical、HC0-HC3、HAC。
- `LogisticRegression`：classical、HC0-HC3、HAC。
- `Lasso`：OLS-style 与 bootstrap inference 路径。

后续 GLM inference 扩展需要先满足项目统一的 strict inference gate，再进入稳定文档。

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

# CPU L2 logistic 路径：auto 选择 IRLS。
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary)

# GPU L2 logistic 路径：auto 选择 GPU-capable FISTA。
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

当前 GLM 重构的 strict 数值验证通过远程 CPU/GPU accuracy 和外部框架对比脚本完成。新的 penalized GLM 层还没有公开 robust standard error、confidence interval 等 strict inference 输出。

`solver="auto"` 对 penalized GLM 是 device-aware 的。Gaussian L2 使用 exact Ridge；CPU logistic/poisson L2 使用 IRLS；CuPy/Torch GPU logistic/poisson L2 使用 FISTA。显式 `irls`、`newton`、`lbfgs` 在数学上适用时会运行在用户选择的后端。

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

远程凭据必须从环境变量读取，不得写入代码或文档。

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
