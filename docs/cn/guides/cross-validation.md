# 交叉验证

> 语言：中文  
> 最后更新：2026-09-11
> 页面定位：CV 用户指南 + 架构实现 + 缓存机制（统一页面）  
> 切换：[English](../../en/guides/cross-validation.md)

## 概述

statgpu 为惩罚模型提供交叉验证 estimator。candidate scoring 与 selected full-data refit 是两个阶段；当某个 estimator 支持 coefficient inference 时，推断属于 selected final refit，而不是 CV folds。

| CV 估计器 | 基础模型 | 主要调参对象 |
|-----------|---------|-------------|
| `RidgeCV` | `Ridge` | L2 alpha |
| `LassoCV` | `Lasso` | L1 alpha |
| `ElasticNetCV` | `ElasticNet` | alpha / 可选 l1_ratio |
| `LogisticRegressionCV` | `LogisticRegression` | logistic regularization |
| `PenalizedGLM_CV` | `PenalizedGeneralizedLinearModel` 或 `PenalizedCoxPHModel` | family-specific penalty alpha |

`PenalizedGLM_CV` coefficient inference 的 target 与 method resolver 见 [Penalized GLM inference](penalized-glm-inference.md)。

## 快速开始

```python
from statgpu.linear_model import PenalizedGLM_CV

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l2",
    alpha_grid=[0.2, 0.05, 0.01],
    cv=5,
    device="auto",
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
)
model.fit(X, y)

print(model.alpha_)
print(model.inference_method_)          # m_estimation
print(model.penalty_conditioning_)     # cv_selected_penalty
```

fold/path/grid 阶段不做 coefficient inference；推断只在 alpha 选择完成后的全数据 final refit 上运行一次。

## 参数参考

### RidgeCV

| 参数 | 常见默认值 | 说明 |
|------|------------|------|
| `alphas` | `None` | 用户 alpha grid 或自动网格 |
| `cv` | estimator default | fold 数 |
| `compute_inference` | estimator default | 对 selected full-data refit 做 inference |
| `cov_type` | `"nonrobust"` | final-refit covariance |

### LassoCV

`LassoCV` 将 CV path solver 与最终 full-data refit solver 分开。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alphas` | `None` | Alpha 网格 |
| `n_alphas` | `12` | 自动网格大小 |
| `solver` | `"fista"` | final full-data `Lasso` refit solver |
| `cv_solver` | `"auto"` | fold/path solver |
| `cpu_solver` | `None` | 已弃用的历史 CPU-CV 控制 |
| `method` | `"standard"` | CV-path profile |
| `gpu_cv_mixed_precision` | `True` | 受支持 GPU CV path 的 mixed precision 控制 |
| `compute_inference` | estimator default | 只对 selected final refit 做 inference |

拟合后 `cv_solver_` 记录 CV 阶段实际执行的算法。

### ElasticNetCV

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `l1_ratio` | `0.5` | mixing 参数或待搜索列表 |
| `alphas` | `None` | Alpha 网格 |
| `compute_inference` | estimator default | 对 selected final refit 运行维护中的 sparse-Gaussian inference |

### PenalizedGLM_CV

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss` | `"squared_error"` | scalar-response loss 或 `"cox_ph"` survival branch |
| `penalty` | `"l2"` | penalty 名称/object |
| `alpha_grid` | `None` | candidate alpha grid |
| `n_alphas` | `100` | 自动网格大小 |
| `cv` | `5` | fold 数 |
| `cv_splits` | `None` | 可选自定义 folds |
| `device` | `"auto"` | CV execution device policy |
| `compute_inference` | `False` | 是否只在 selected full-data final refit 上运行 coefficient inference |
| `inference_method` | `"auto"` | final-refit inference request |
| `cov_type` | `"nonrobust"` | final-refit covariance；non-Gaussian L2 M-estimation 当前支持 nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | 保留底层控制；不代表 non-Gaussian penalized HAC 已支持 |

成功的 `PenalizedGLM_CV` inference-enabled fit 会委托 final estimator 的 result/provenance，并额外记录：

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

因此 SE、p-value 与 confidence interval 条件于 selected alpha；它们**没有**校正 tuning-selection uncertainty。

## 自定义 folds

`cv_splits` 可提供显式 `(train_idx, validation_idx)` pairs。一次性 generator 会先 materialize，保证所有 candidate work 使用相同 folds。

```python
from sklearn.model_selection import TimeSeriesSplit

folds = list(TimeSeriesSplit(n_splits=5).split(X))
model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l1",
    cv_splits=folds,
)
model.fit(X, y)
```

survival branch 会额外做事件支持检查；每个可评估 Cox train/validation partition 都必须包含 observed events，无效索引会在 candidate fit 之前 fail closed。

## 样本权重

sample-weight 支持取决于具体 loss/penalty/solver path；不能把某个 solver 的限制外推成整个 estimator 永远不支持权重。

对于本次新增的 non-Gaussian L2 coefficient-inference contract，analytic weights 由 selected final refit 与 M-estimation covariance 支持。inference-enabled weighted non-Gaussian L2/no-penalty final fit 如果保持 `solver="auto"`，会在该次 fit 内使用维护中的 backend-native FISTA，因为 Newton 当前拒绝非均匀 analytic weights。公开 solver request 仍保持 `auto`，显式 solver 请求始终权威。

penalized Cox CV branch 当前拒绝 `sample_weight`。

## Alpha 网格

自动网格在 statgpu average-loss scale 上生成；具体 zero-model/KKT 规则取决于 loss/penalty，ElasticNet 会按其 L1 mixing 调整阈值。

用户网格在 candidate fitting 前校验。scalar-response 与 survival CV 在统计 contract 不同处有意采用不同的零值/正值规则。

## 拟合属性

常见 CV 输出包括：

- `alpha_`；
- `best_score_`；
- `cv_results_`；
- `estimator_` — selected hyperparameter 的 full-data final refit；
- `coef_` 与 `intercept_`（适用时）。

当 `PenalizedGLM_CV(compute_inference=True)` 解析到受支持 final-refit row 时，还会发布 `_inference_result`、`_bse`、`_pvalues`、`_conf_int` 以及 requested/resolved/target/conditioning provenance。

## 设备选择

`device="auto"` 将问题规模、loss×penalty heuristic 与实际 backend 可用性结合。显式 `device="cuda"` / `"torch"` 使用严格语义：请求 backend 不可用时不会静默切回 CPU。

当前 scalar CV 含 benchmark-backed threshold，例如 small problem CPU、部分高维 sparse GLM Torch path，以及 large-effective-work fallback。这些阈值属于 execution policy，而不是统计定义。

## CV 后推断

### Ridge / sparse Gaussian CV

Ridge/Lasso/ElasticNet 保持各自维护的 inference contract。folds 用于 selection；inference 属于 selected full-data fit。

### PenalizedGLM_CV

coefficient inference 是 **final-refit-only**：

1. folds/grid/path candidate fits 全部 inference disabled；
2. 根据 held-out evidence 选出 alpha；
3. 若请求 inference，用公开 inference controls 对 selected model 做 full-data refit；
4. final estimator 的 `_inference_result` 与 provenance 委托给 CV estimator；
5. 明确报告 selection adjustment 为 false。

当前主要支持 row 包括：

- smooth non-Gaussian L2/no penalty → `m_estimation`；
- 维护中的 Gaussian L2 与 sparse-Gaussian contract；
- 仅在 underlying estimator 已支持时保留显式且窄的 SCAD/MCP oracle/bootstrap row。

non-Gaussian L1/ElasticNet coefficient inference 尚未实现。Cox branch 仍为 estimation-only。

完整矩阵见 [Penalized GLM inference](penalized-glm-inference.md)。

## Strict 与 two-stage CV

`PenalizedGLM_CV` 默认 `cv_strategy="strict"`，按维护的 iteration/tolerance 设置评价 candidate grid。

`cv_strategy="two_stage"` 是 opt-in approximate screening：

1. 用放松设置在完整 grid 上做 stage-1 scoring；
2. 保留 top candidates；
3. 对候选做 strict refinement；
4. strict/full final refit。

stage-1 approximation 在 CV 曲线非常接近时可能改变 ranking，因此该模式会明确标记 approximate，并要求 acknowledgement 才能静默 warning。

## 架构

scalar-response 与 survival CV preparation 顺序不同，但共享一个原则：**selection 与 inference 分离**。

### Scalar response

```text
validate/generate alpha grid
        ↓
materialize folds
        ↓
resolve CV device + solver
        ↓
score candidates (inference disabled)
        ↓
select alpha
        ↓
full-data final refit
        ↓
optional coefficient inference exactly once
```

### Penalized Cox

```text
normalize (time,event)
        ↓
validate/materialize event-supported folds
        ↓
resolve CV device
        ↓
survival-aware held-out scoring
        ↓
require complete finite evidence
        ↓
PenalizedCoxPHModel final refit (inference disabled)
```

## 主要 scoring path

### Ridge eigendecomposition

squared-error L2 CPU scoring 可在每个 fold 上复用一次 eigendecomposition 评价多个 alpha；维护中的 full-data refit 保持精确 float64 语义。

### Fold-batch sparse GLM

部分 GPU sparse-GLM path 将多个 fold 的线性代数批量化，以减少 kernel launch 与 host synchronization。

### Sparse Gram + warm start

squared-error L1/ElasticNet 可复用 fold Gram matrix，并按递减 alpha warm-start。

### SCAD/MCP LLA

non-convex penalty 需要 continuation/LLA outer work，因此 CV path 比 convex sparse path 更贵。

### Generic fallback

没有 optimized path 时，CV 对每个 fold/candidate 使用维护中的 estimator fit，并评价声明的 loss。infrastructure recovery 不允许为了跑完流程而把 non-Gaussian objective 静默替换成 MSE。

## 缓存

`LassoCV` 使用 selection-only LRU cache：缓存 alpha-selection evidence，不缓存 final estimator。`X`、`y` 与 `sample_weight` 的内容身份由 `_array_identity_token(...)` 表示；token 包含 backend、shape、dtype 与内容 digest，大数组的 CuPy/Torch 路径只把用于 hash 的抽样行送回 host。

当前 selection key 由 `_make_lasso_cv_auto_cache_key(...)` 构造。key 包含数据/权重 identity token、完整实际评价 alpha-grid digest、每个 fold 的完整 train/validation index 数组、intercept/GPU execution mode、iteration/tolerance、解析后的 CV solver/method、`cd_kkt_check_every` 以及 `gpu_cv_mixed_precision`。final-refit `solver` 被有意排除，因为它不会改变 alpha scoring。

LRU 默认容量为 **64**，可在 import 时通过 `STATGPU_LASSO_CV_CACHE_SIZE` 配置。cache hit 返回 NumPy payload array 的副本，避免调用方原地修改缓存 selection evidence。

## Alpha 约定

statgpu penalized estimator 使用 average-loss objective；跨框架比较前必须先对齐 penalty scaling。例如 Ridge 常需要：

```text
sklearn_alpha = statgpu_alpha * n_samples
```

不同框架目标函数 normalization 不同时，同名 alpha 不能直接比较。

## Penalized Cox branch

`PenalizedGLM_CV(loss="cox_ph")` 在整个 selection 流程中保持 `(n_samples, 2)` 的 `[time, event]` target，使用 survival-aware held-out partial-likelihood scoring，验证 event support，并以无 intercept 的 `PenalizedCoxPHModel` final refit。

当前不支持：

- coefficient inference；
- `sample_weight`；
- `cv_strategy="two_stage"`；
- dictionary target。

## 已知限制

- non-Gaussian L1/ElasticNet coefficient inference 未实现。
- non-Gaussian penalized M-estimation 当前只支持 nonrobust/HC0/HC1 covariance。
- weighted/robust/family-aware bootstrap 不在维护中的 Gaussian residual-bootstrap contract 内。
- 显式 solver 的 sample-weight 能力仍由 solver 自身决定；inference contract 不覆盖显式 solver 请求。
- penalized Cox CV 仍是 estimation-only。

## FAQ

**CV inference 会校正 hyperparameter selection 吗？**  
不会。`penalty_selection_adjusted_=False`；结果条件于 selected alpha。

**为什么 cache miss？**  
selection-relevant data/fold/grid/solver/tolerance/execution 字段发生变化会 miss。

**为什么 SCAD/MCP CV 更慢？**  
因为需要 LLA/continuation outer work。

**non-Gaussian L1/ElasticNet 能用 bootstrap 兜底吗？**  
不能。本次维护中的 bootstrap 只表示 Gaussian residual bootstrap。

## 参见

- [Penalized GLM inference](penalized-glm-inference.md)
- [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md)
- [Ridge 模型](../models/ridge.md)
- [ElasticNet 模型](../models/elastic-net.md)
- [GLM 模型](../models/generalized-linear-model.md)

## External Validation

维护中的 CV suite 覆盖参数校验、fold safety、solver migration、selection cache、Ridge/Lasso/ElasticNet path、backend routing 与 penalized-Cox evidence rules。本次 penalized-GLM inference repair 额外覆盖 final-refit-only inference、selected-alpha reuse、clone/API compatibility、formula parity 与 selection-conditioning provenance。
