# GeneralizedLinearModel 与 Penalized GLM

> 语言: 中文  
> 最后更新: 2026-09-11  
> 页面定位: 模型文档  
> 切换: [English](../../en/models/generalized-linear-model.md)

语言切换: [English](../../en/models/generalized-linear-model.md)

## Overview

`GeneralizedLinearModel` 是普通 GLM 的统一入口。`PenalizedGeneralizedLinearModel` 与 typed wrappers 在同一 average-loss 目标上加入 L1、L2、ElasticNet、group、adaptive 等 penalty，并保持 sklearn 风格的 `fit` / `predict` API。

推荐 regularized GLM 使用 typed estimator：

```python
from statgpu.linear_model import (
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
)
```

`Ridge`、`Lasso`、`ElasticNet` 是 penalized Gaussian regression 上的 sklearn 风格薄包装。

惩罚模型系数推断的完整统计口径见 [Penalized GLM inference](../guides/penalized-glm-inference.md)。

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
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta).
$$

截距项不惩罚。Cox partial likelihood、panel objective 等不强行塞入 `glm_core`，而使用各自维护的 objective 路径。

## Estimating Equation 与 solver dispatch

smooth GLM 可使用二阶/一阶优化；带 non-smooth penalty 的目标使用 FISTA/FISTA-BB 等 proximal 路径。

当前 direct-fit `solver="auto"` 是 table-driven dispatch。尤其需要注意：smooth non-Gaussian L2 当前解析到 Newton，而不是旧文档曾写的 IRLS。

| 设置 | 当前 `solver="auto"` 行为 |
|---|---|
| squared error + L2，NumPy/CPU | exact closed-form L2 path |
| squared error + L1/ElasticNet | FISTA/FISTA-BB sparse path |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | Newton |
| SCAD/MCP | FISTA + LLA continuation |
| quantile | FISTA / quantile-specific path |

显式 `device="cuda"` 或 `device="torch"` 不允许静默落回 NumPy。显式 solver 请求也保持权威性；不支持的组合直接报错。

本次 inference contract 有一个窄例外：对于 **开启 inference 的 weighted non-Gaussian L2/无惩罚拟合**，若用户保留 `solver="auto"`，该次 fit 会使用已有 backend-native FISTA 路径，因为 Newton 当前不接受非均匀 analytic weights。公开参数仍保持 `solver="auto"`，estimation-only 的 benchmark dispatch 不受影响。

## Covariance / Inference

Penalized GLM coefficient inference 只对明确支持的 loss × penalty × method × covariance 组合开放。generic / typed penalized GLM 推荐：

```python
inference_method="auto"
```

### non-Gaussian L2 / 无惩罚

对 smooth non-Gaussian L2/无惩罚模型，`auto` 解析为 fixed-penalty M-estimation。正 L2 penalty 时，推断目标明确报告为 penalized estimating equation，而不是伪装成 unpenalized/debiased coefficient。

当前支持：

- `cov_type="nonrobust"`：model-based penalized-information covariance；
- `cov_type="hc0"`；
- `cov_type="hc1"`。

HC2、HC3、HAC 尚未为这条 penalized non-Gaussian path 实现，会 fail closed。

analytic weights 受支持。covariance/statistic/p-value/CI 的 numerical work 会跟随拟合实际使用的 backend 与 concrete device；只有 numerical inference 完成后才允许把小型 reporting arrays snapshot 到 NumPy。

### sparse / non-convex rows

- Gaussian L1/ElasticNet 保持已有 debiased 与 `post_selection_ols` contract。
- non-Gaussian L1/ElasticNet inference 未实现，直接 fail closed。
- SCAD/MCP oracle 必须显式请求 `inference_method="oracle"`；`auto` 不会静默选择 active-set-conditional procedure。
- group penalties 与 penalized Cox 仍是 estimation-only。
- 本次 `inference_method="bootstrap"` 仅表示 **无权重、CPU 执行的 Gaussian residual bootstrap**，并非通用 GLM fallback；它保持原 penalty family、alpha、intercept 语义及 ElasticNet mixing，并要求 `cov_type="nonrobust"`。

### requested / resolved / reported

成功完成 coefficient inference 的拟合会发布：

- `inference_requested_method_`；
- `inference_resolved_method_`；
- `inference_method_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`。

L2/无惩罚模型显式使用历史 `inference_method="debiased"` 时，暂时作为 deprecated compatibility spelling 接受，并发出 `FutureWarning`；拟合后的字段仍报告真实方法。L2 inference 从来不是 debiased-Lasso inference。

完整支持矩阵与统计解释见 [Penalized GLM inference](../guides/penalized-glm-inference.md)。

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `loss` / `family` | model-specific | GLM family/loss，例如 `"logistic"`、`"poisson"` |
| `penalty` | model-specific | `none`、`l1`、`l2`、`elasticnet` 以及受支持 structured penalties |
| `alpha` | model-specific | statgpu average-loss 目标函数下的 penalty strength |
| `l1_ratio` | model-specific | ElasticNet mixing |
| `fit_intercept` | `True` | 是否拟合截距 |
| `solver` | `"auto"` | backend-neutral solver request |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `max_iter` | model-specific | 最大迭代次数 |
| `tol` | model-specific | 收敛阈值 |
| `compute_inference` | generic/typed penalized GLM 默认 `False` | 是否运行 coefficient inference |
| `inference_method` | generic/typed penalized GLM 默认 `"auto"` | 根据 loss/penalty/covariance 解析受支持方法；sparse Gaussian wrapper 保留既有显式默认 |
| `cov_type` | `"nonrobust"` | non-Gaussian penalized M-estimation 当前支持 nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | 保留接口字段；并不意味着 penalized non-Gaussian 已支持 HAC |
| `formula` | `None` | 可选 patsy-style formula |
| `data` | `None` | formula 所用 DataFrame |

不同框架的 penalty scaling 不能仅按同名参数直接比较，必须先对齐 objective normalization。

## 示例

```python
from statgpu.linear_model import PenalizedPoissonRegression

model = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    solver="auto",
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    device="cpu",
)
model.fit(X, y_count, sample_weight=w)

print(model.inference_requested_method_)  # auto
print(model.inference_resolved_method_)   # m_estimation
print(model.inference_method_)            # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
print(model._bse)
print(model._pvalues)
```

formula 是可选依赖：

```bash
pip install statgpu[formula]
```

```python
pois = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.01,
    compute_inference=True,
    cov_type="hc0",
)
pois.fit(formula="count ~ exposure + x1", data=df)
```

formula parsing 是 CPU 侧 model-matrix convenience；后续 numerical fit/inference 仍跟随所选 backend。

## Strict / approximate 边界

unsupported inference row 会直接失败，而不会静默改成另一种统计方法。

Residual bootstrap 与 SCAD/MCP oracle 都是有明确条件的窄路径，`auto` 不把它们当作 universal fallback。

`PenalizedGLM_CV` 默认 `cv_strategy="strict"`。fold/path/grid fit 不运行 coefficient inference；若 `compute_inference=True`，只在选定 alpha 的 full-data final refit 上运行一次 inference，并报告：

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

因此这些 SE/p-value/CI 条件于 CV 选出的 penalty，并未校正 tuning-selection uncertainty。

`cv_strategy="two_stage"` 仍属于 opt-in approximate screening，随后对候选做 strict refinement/final refit。

### penalized Cox CV

`PenalizedGLM_CV(loss="cox_ph")` 使用独立生存路径，保留二维 `[time, event]` target 并进行 survival-aware held-out scoring。该分支仍是 estimation-only；`compute_inference=True` 会被拒绝。

## Outputs

常见拟合结果包括：

- `coef_`；
- `intercept_`；
- `n_iter_`；
- `fit` / `predict` 及 family-specific prediction helper；
- `PenalizedGLM_CV.cv_results_`；
- coefficient inference 成功时的 `_bse`、`_pvalues`、`_conf_int`、`_inference_result`，以及 requested/resolved/target/conditioning provenance 字段。

## 参见

- [Penalized GLM inference](../guides/penalized-glm-inference.md)
- [Solver × Penalty 兼容性矩阵](../guides/solver-penalty-matrix.md)
- [Cross-Validation](../guides/cross-validation.md)

## FAQ

- **`inference_method="auto"` 是 silent fallback 吗？** 不是。它只解析已支持的 row；不支持的组合 fail closed。
- **non-Gaussian L2 M-estimation 消除了 shrinkage bias 吗？** 没有。其目标就是 fixed-penalty estimating equation，并明确公开该 target。
- **CV inference 校正 alpha 选择了吗？** 没有；`penalty_selection_adjusted_=False`。
- **non-Gaussian L1/ElasticNet 可以用 bootstrap 兜底吗？** 不可以。本次维护 bootstrap 仅为 Gaussian residual bootstrap。
- **显式 CUDA/Torch inference 会静默用 CPU 吗？** 不会。支持路径跟随实际 backend/device，否则明确失败。

## External Validation

维护测试覆盖 CPU regression、formula parity、独立 covariance algebra、clone/API compatibility 与 backend dispatch contract。physical CUDA acceptance 属于独立 evidence tier，只有在真实 CuPy/Torch CUDA 设备上执行后才可声称通过。

历史 v23c 的 1043/1043 全矩阵结果验证的是 estimation matrix；它本身不能替代本次 coefficient-inference contract 的专项验收。

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.
