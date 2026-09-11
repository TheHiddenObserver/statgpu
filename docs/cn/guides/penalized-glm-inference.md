# 惩罚 GLM 推断

> 语言：中文  
> 状态：目标版本为 0.2.6；当前已发布版本仍是 0.2.5。  
> 切换：[English](../../en/guides/penalized-glm-inference.md)

## 这个接口表达什么

`PenalizedGeneralizedLinearModel` 与各 typed penalized GLM wrapper 通过 `compute_inference`、`inference_method` 和 `cov_type` 暴露系数推断。接口明确区分 **用户请求的方法**、**statgpu 解析得到的方法** 与 **结果实际报告的方法**。

对于 generic / typed penalized-GLM 接口，推荐默认值为：

```python
inference_method="auto"
```

成功完成推断的拟合会发布：

- `inference_requested_method_`；
- `inference_resolved_method_`；
- `inference_method_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`。

相同 provenance 也会写入 `_inference_result.metadata`，并在适用时记录实际 numerical backend/device。

## 支持矩阵

| loss / penalty | 支持的推断 | `auto` 解析 |
|---|---|---|
| squared error + L2/无惩罚 | 既有 Gaussian classical/robust covariance | `cov_type="nonrobust"` 时为 `classical`，否则走既有 Gaussian sandwich 路径 |
| squared error + L1/ElasticNet | debiased inference；既有 `post_selection_ols` contract | `debiased` |
| smooth non-Gaussian GLM + L2/无惩罚 | 固定惩罚 M-estimation | `m_estimation` |
| Gaussian L1/ElasticNet/SCAD/MCP | 显式请求时可用无权重 CPU residual bootstrap | 不自动选择 |
| SCAD/MCP scalar GLM family | 显式请求的 active-set oracle refit | 不自动选择 |
| non-Gaussian L1/ElasticNet | 未实现 | fail closed |
| group penalties | 仅估计 | fail closed |
| `PenalizedCoxPHModel` / `PenalizedGLM_CV` 的 Cox 分支 | 仅估计 | fail closed |

历史的 `cpu_ols` / `gpu_ols` 拼写继续保留既有的一周期迁移，仅在 sparse Gaussian 模型中映射到 `post_selection_ols`。它们不再承担执行硬件选择含义。

对于 L2/无惩罚模型，显式 `inference_method="debiased"` 暂时作为 deprecated compatibility spelling 接受。它会发出 warning，并解析到该模型真正使用的推断方法；L2 推断并不是 debiased-Lasso 推断。

## 固定惩罚 M-estimation

对于受支持的 non-Gaussian smooth L2/无惩罚拟合，statgpu 将拟合系数视作 penalized estimating equation 的解。正 L2 penalty 时会报告：

```text
inference_target_ = "penalized_estimating_equation"
penalty_conditioning_ = "fixed_penalty"
```

若 `alpha=0`，则目标是普通的未惩罚 population coefficient。

数值引擎采用 average-loss 标度。记单样本 score contribution 为 `psi_i`，average Hessian 为 `H`，L2 curvature 为 `P''`，则 HC0/HC1 covariance 采用

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1} J (H+P'')^{-1}/n_{\mathrm{eff}},
$$

其中 `J` 是 statgpu analytic-weight 约定下的 average score outer product。`cov_type="nonrobust"` 使用既有 model-based penalized-information covariance。

该 non-Gaussian 路径当前支持：

- `nonrobust`；
- `hc0`；
- `hc1`。

HC2、HC3 与 HAC 尚未为 penalized non-Gaussian M-estimation 实现，会明确报错。

## Analytic weights

non-Gaussian L2 M-estimation covariance 支持 analytic weights，并且 numerical inference 跟随实际执行拟合的 backend/device。

这里存在一个需要公开说明的 solver 边界：通用 `solver="auto"` benchmark dispatch 对 smooth GLM 可能选择 Newton，而 Newton 当前不接受非均匀 analytic weights。因此，对于 **开启 inference 的 weighted non-Gaussian L2/无惩罚拟合**，本 contract 仅在该次 fit 内选择已有的 backend-native FISTA 路径。公开请求仍保持 `solver="auto"`；这不会改变 estimation-only 的全局 benchmark policy。

显式指定 solver 时仍以用户请求为准，不会被静默替换。

## Backend / device provenance

受支持的 non-Gaussian M-estimation 会在拟合实际选择的 backend 上完成 covariance/statistic/p-value/CI 数值计算：

- CPU 上的 NumPy；
- 拟合记录的具体 CUDA device 上的 CuPy；
- 拟合记录的具体 Torch device 上的 Torch。

显式 `device="cuda"` 或 `device="torch"` 不会静默改为 NumPy inference。只有在 numerical inference 完成后，较小的报告数组才允许 snapshot 到 NumPy。结果 metadata 会记录 `numerical_backend`、`numerical_device`、`reporting_backend` 与 reporting boundary。

## Residual bootstrap 的范围

本次修复中的 `inference_method="bootstrap"` **不是通用 GLM bootstrap**，而是无权重 Gaussian residual bootstrap：

- fixed design；
- 固定相同 `alpha`；
- 保持相同 penalty family；
- 对 ElasticNet 保持相同 `l1_ratio` / penalty kwargs；
- 保持相同 intercept 语义；
- 每个 bootstrap sample 内重新拟合 penalized estimator；
- bootstrap 内不重新进行 CV/tuning。

本次维护实现仅支持 CPU execution，并要求 `cov_type="nonrobust"`。weighted residual bootstrap、robust/HAC bootstrap 与 family-aware non-Gaussian bootstrap 都需要单独的统计设计，因此当前会 fail closed，而不是猜测语义。

这些区间是 heuristic penalized-estimator bootstrap interval，不应解释成 selective-inference coverage guarantee。

## SCAD/MCP oracle 边界

`inference_method="oracle"` 必须显式请求，因为它条件于已选择的 active set；`auto` 不会静默选择 oracle。当前 oracle implementation 使用 CPU active-set refit，因此实际在 CuPy/Torch 上完成拟合后请求 oracle inference 会明确失败，而不会伪装成 backend-native oracle inference。

## Cross-validation

`PenalizedGLM_CV` 增加以下控制项：

```python
PenalizedGLM_CV(
    ...,
    compute_inference=False,
    inference_method="auto",
    cov_type="nonrobust",
    hac_maxlags=None,
)
```

fold/path/grid 拟合始终保持 estimation-only。如果请求 inference，statgpu 先完成 `alpha` 选择，然后只在全数据 selected-penalty final refit 上运行一次推断。成功的 CV inference 会报告：

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

因此标准误、p-value 与 confidence interval **条件于 CV 选出的 penalty**，并没有校正 tuning-selection uncertainty。

Cox 分支仍保持 estimation-only。

## 示例

```python
from statgpu.linear_model import PenalizedPoissonRegression

model = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    device="cpu",
)
model.fit(X, y, sample_weight=w)

print(model.inference_requested_method_)  # auto
print(model.inference_resolved_method_)   # m_estimation
print(model.inference_method_)            # m_estimation
print(model.inference_target_)            # penalized_estimating_equation
print(model._bse)
print(model._pvalues)
```

对于 sparse non-Gaussian L1/ElasticNet，请使用 `compute_inference=False`；本次修复不会为了补齐表格而虚构新的 debiasing 方法。

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
