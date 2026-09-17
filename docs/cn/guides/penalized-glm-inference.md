# 惩罚 GLM 推断

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：coefficient inference 的统计 target 与公开支持行为  
> 切换：[English](../../en/guides/penalized-glm-inference.md)

## 这个接口表达什么

`PenalizedGeneralizedLinearModel` 与各 typed penalized GLM wrapper 通过 `compute_inference`、`inference_method` 和 `cov_type` 暴露 coefficient inference。

对 generic interface，通常从

```python
inference_method="auto"
```

开始即可。

成功完成 inference 的拟合会区分调用者请求的方法与该模型最终解析得到的方法。相关 fitted attribute 包括：

- `inference_requested_method_`；
- `inference_resolved_method_`；
- `inference_method_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`。

这些字段用于说明报告的不确定性究竟对应哪个参数，尤其是在 penalization 或 CV selection 之后。

## 支持概览

| loss / penalty | 支持的 inference | `auto` 行为 |
|---|---|---|
| squared error + L2/无惩罚 | Gaussian classical/robust covariance | 按 `cov_type` 使用 classical 或 Gaussian robust covariance |
| squared error + L1/ElasticNet | debiased inference；显式请求时可用 `post_selection_ols` | `debiased` |
| smooth non-Gaussian GLM + L2/无惩罚 | fixed-penalty M-estimation | `m_estimation` |
| 受支持的 Gaussian penalized model | 显式请求 residual bootstrap | 不自动选择 |
| 受支持的 SCAD/MCP scalar GLM | 显式请求 active-set oracle refit | 不自动选择 |
| non-Gaussian L1/ElasticNet | 未实现 coefficient inference | 请求会报错 |
| group penalties | 仅估计 | inference 请求会报错 |
| `PenalizedCoxPHModel` / `PenalizedGLM_CV` 的 Cox 分支 | 仅估计 | inference 请求会报错 |

Sparse Gaussian 的 `debiased`、`post_selection_ols` 等方法如何选择与解释，见 [推断模式](inference-modes.md)。

## Fixed-penalty M-estimation

对于受支持的 smooth non-Gaussian L2/无惩罚拟合，statgpu 把 fitted coefficient 看作给定 penalty strength 下 estimating equation 的解。

正 L2 penalty 时会报告：

```text
inference_target_ = "penalized_estimating_equation"
penalty_conditioning_ = "fixed_penalty"
```

无惩罚拟合（`alpha=0` 或对应 no-penalty 配置）的 target 是普通 unpenalized population coefficient。

记单样本 score contribution 为 $\psi_i$，average Hessian 为 $H$，L2 curvature 为 $P''$，average score outer product 为 $J$，则 HC0/HC1 covariance 具有形式

$$
\widehat{\mathrm{Var}}(\hat\beta)
=
(H+P'')^{-1}J(H+P'')^{-1}/n.
$$

`cov_type="nonrobust"` 使用 model-based penalized-information covariance。

当前 non-Gaussian fixed-penalty 路径支持：

- `nonrobust`；
- `hc0`；
- `hc1`。

HC2、HC3 与 HAC 在该路径上不可用，请求时会报错。

## Analytic weights

当所选 smooth GLM solver 支持 analytic `sample_weight` 时，整个拟合使用同一个归一化加权目标：

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

对应 M-estimation 使用同一种 relative-weight interpretation。计算 covariance 时，可把权重等价表示成 mean-one scale：

$$
\widetilde w_i
=
\frac{n w_i}{\sum_j w_j},
\qquad
\sum_i \widetilde w_i=n.
$$

因此所有正 analytic weight 同乘一个常数不会改变统计 objective 与 inferential target，数值结果只会受到 solver tolerance 范围内的有限精度影响。

这里的权重是**analytic / relative-importance weights**，不是 frequency weights；把全部权重统一放大并不表示样本通过复制而增大。

如果某个 loss 没有定义请求的 weighted fit，真正的 non-uniform weights 会被拒绝，而不是静默丢弃。

## Solver 选择与权重

受支持的显式 solver 请求保持权威。对于 smooth non-Gaussian L2/无惩罚行，Newton 与 L-BFGS 在支持 analytic weights 时使用上面的 weighted objective。

`solver="auto"` 则继续服从模型本身的正常 solver dispatch。public request 仍然是 `auto`；inference 描述的是实际成功拟合的模型，而不会为了推断单独切换成无关 solver。

solver compatibility 见 [Solver × Penalty 矩阵](solver-penalty-matrix.md)。

## Backend 与 device 行为

受支持的 non-Gaussian M-estimation 在成功拟合实际使用的 backend/device 上完成 covariance/statistic/p-value/CI 的数值计算。

显式 `device="cuda"` 或 `device="torch"` 不会被静默替换成 NumPy inference。numerical inference 完成后，小型 reporting array 可以转换为 NumPy；这种 reporting boundary 不改变数值 procedure 实际运行的位置。

部分 inference method 的 backend support 比 parent estimator 更窄。例如当前 SCAD/MCP oracle refit 是 CPU-only；GPU fit 后请求该方法会报错，而不会把结果描述成 backend-native oracle inference。

## Residual bootstrap 的范围

`inference_method="bootstrap"` 是 Gaussian penalized-model residual bootstrap，不是通用 GLM bootstrap。

每个 draw 中，statgpu：

1. 根据 fitted Gaussian model 计算 fitted value 与 residual；
2. 对 residual 做有放回抽样；
3. 构造 bootstrap response；
4. 使用相同 tuning configuration 重新拟合同一 penalized model；
5. 汇总 bootstrap coefficient distribution。

需要可复现抽样时设置 `bootstrap_random_state`。`n_bootstrap` 控制 refit 次数，并且至少为 2。

当前 residual-bootstrap 路径要求：

- `sample_weight=None`；
- `cov_type="nonrobust"`。

Weighted residual bootstrap、robust/HC 或 HAC/block bootstrap、non-Gaussian bootstrap 与 Cox bootstrap 不由这个接口提供。

这些区间描述 fixed-design、fixed-tuning residual-bootstrap procedure；它们不是一般 selective-inference interval，也不会自动校正 tuning 或 variable-selection uncertainty。

## SCAD/MCP oracle inference

`inference_method="oracle"` 必须显式请求，因为它条件于 penalized fit 已经选择的 active set。`auto` 不会静默采用这种解释。

在支持范围内，该 procedure 会在所选 active set 上进行不含原 non-convex penalty 的 refit，并对这个 conditional refit 报告 uncertainty。请求前应检查 model/backend support。

## Cross-validation

`PenalizedGLM_CV` 把 tuning 与 coefficient inference 分成两个阶段：

```text
fold/path/grid fits
    -> select alpha
    -> refit selected model on all observations
    -> run inference once on the final refit
```

成功的 inference-enabled CV fit 会报告类似：

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

因此 standard error、p-value 与 confidence interval 都是以 CV-selected penalty 为条件的，并不会自动调整 tuning-selection uncertainty。

对于 residual bootstrap，只有 CV 选定 tuning parameter 后才开始 resampling；candidate-selection process 本身不会做 bootstrap。

Cox 分支仍为 estimation-only。

一般 selection/refit contract 见 [交叉验证](cross-validation.md)。

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
print(model.inference_target_)            # penalized_estimating_equation
print(model.summary())
```

对于 sparse non-Gaussian L1/ElasticNet，目前没有提供 coefficient inference；应使用 `compute_inference=False`，而不是假定 Gaussian 的 debiasing/bootstrap procedure 自动适用于其他 family。

## 相关文档

- [推断模式](inference-modes.md) — method 选择与解释
- [交叉验证](cross-validation.md) — tuning 与 final refit
- [Solver × Penalty 矩阵](solver-penalty-matrix.md) — solver compatibility
- [设备与 GPU 内存](device-and-memory.md) — device 语义

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
