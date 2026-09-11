# 交叉验证

> 语言：中文  
> 最后更新：2026-09-11
> 页面定位：CV 用户指南 + 架构实现 + 缓存机制（统一页面）  
> 切换：[English](../../en/guides/cross-validation.md)

## 概述

statgpu 为所有惩罚模型提供交叉验证估计器。每个 CV 估计器自动在正则化参数网格上搜索，通过 k 折交叉验证选择最优参数。`PenalizedGLM_CV` 支持多种损失函数和 penalty，并针对不同 loss×penalty×device 组合使用专门求解路径。

| CV 估计器 | 基础模型 | 惩罚 | 路径 |
|-----------|---------|------|------|
| `RidgeCV` | `Ridge` | l2 | `statgpu.linear_model.RidgeCV` |
| `LassoCV` | `Lasso` | l1 | `statgpu.linear_model.LassoCV` |
| `ElasticNetCV` | `ElasticNet` | elasticnet | `statgpu.linear_model.ElasticNetCV` |
| `LogisticRegressionCV` | `LogisticRegression` | l2 | `statgpu.linear_model.LogisticRegressionCV` |
| `PenalizedGLM_CV` | `PenalizedGeneralizedLinearModel` 或 `PenalizedCoxPHModel` | family 支持的 penalty | `statgpu.linear_model.PenalizedGLM_CV` |

## 快速开始

### RidgeCV

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=None,
    n_alphas=100,
    cv=5,
    fit_intercept=True,
    device="auto",
)
model.fit(X, y)

print(f"最优 alpha: {model.alpha_}")
print(f"CV MSE 路径形状: {model.cv_results_['mse_path'].shape}")
print(f"R²: {model.score(X_test, y_test):.4f}")
```

### ElasticNetCV

```python
from statgpu.linear_model import ElasticNetCV

model = ElasticNetCV(
    l1_ratio=0.5,
    alphas=None,
    cv=5,
    device="auto",
)
model.fit(X, y)

print(f"最优 alpha: {model.alpha_}")
print(f"最优 l1_ratio: {model.l1_ratio_}")
```

### PenalizedGLM_CV（通用）

```python
from statgpu.linear_model import PenalizedGLM_CV

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l2",
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

fold/path/grid 阶段不做 coefficient inference；推断只在 alpha 选择完成后的全数据 final refit 上运行一次。详细统计口径见 [Penalized GLM inference](penalized-glm-inference.md)。

### 惩罚 Cox 交叉验证

Cox target 在整个选择流程中必须保持二维：

```python
survival_y = np.column_stack([time, event])
model = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="elasticnet",
    l1_ratio=0.4,
    alpha_grid=[0.2, 0.05, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cuda",
).fit(X_cuda, survival_y_cuda)
```

该路径要求每个可评估 fold 都提供有限 held-out Cox partial-likelihood 证据，并以无截距 `PenalizedCoxPHModel` 完成最终重拟合。若所有候选均无完整证据，会直接抛错而不是默认选择第一个 alpha。该分支仅提供估计，不发布 coefficient inference；`compute_inference=True` 会被拒绝。不支持 `two_stage`、sample weights 或字典 target。

### LogisticRegressionCV

```python
from statgpu.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(cv=5, device="auto")
model.fit(X, y)
print(f"最优 C: {model.C_}")
print(f"准确率: {model.score(X_test, y_test):.4f}")
```

## 参数参考

### 通用参数（所有 CV 估计器）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cv` | int | `5` | CV 折数。必须 ≥ 2。 |
| `random_state` | int | estimator-specific | 折洗牌随机种子。 |
| `device` | str/Device | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`，视 estimator 支持情况而定。 |

### RidgeCV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alphas` | array | `None` | Alpha 网格。`None` = 自动生成。 |
| `n_alphas` | int | `100` | 自动生成时的 alpha 数量。 |
| `alpha_min_ratio` | float | `1e-3` | 最小 alpha 与最大 alpha 的比值。 |
| `compute_inference` | bool | estimator default | CV 后对最终全数据 refit 计算 inference。 |
| `cov_type` | str | `"nonrobust"` | 推断协方差类型。 |

### LassoCV 专用

`LassoCV` 将 CV solver 与最终全数据重拟合 solver 分开。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alphas` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `12` | 自动生成时的 alpha 数量。 |
| `solver` | str | `"fista"` | 最终全数据 `Lasso` refit solver。 |
| `cv_solver` | str | `"auto"` | CV folds/path solver；CPU/GPU 按维护策略解析。 |
| `cpu_solver` | str/None | `None` | **已弃用**旧 CPU-CV 控制。 |
| `method` | str | `"standard"` | CV path profile。 |
| `cd_kkt_check_every` | int/None | `None` | coordinate descent KKT 扫描频率。 |
| `gpu_cv_mixed_precision` | bool | `True` | GPU CV path 是否用混合精度。 |
| `compute_inference` | bool | estimator default | 只在 selected final refit 上执行 inference。 |

拟合后 `cv_solver_` 记录实际 CV 算法。`cpu_solver` 迁移语义见 [penalized solver API 迁移指南](penalized-solver-api-migration.md)。

### ElasticNetCV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `l1_ratio` | float/list | `0.5` | L1 混合比。 |
| `alphas` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `100` | Alpha 数量。 |
| `compute_inference` | bool | estimator default | 对 selected final refit 运行维护中的 sparse-Gaussian inference。 |

### PenalizedGLM_CV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss` | str | `"squared_error"` | 损失族。 |
| `penalty` | str / penalty object | `"l2"` | 惩罚类型。 |
| `penalty_kwargs` | dict | `None` | penalty 参数。 |
| `alpha_grid` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `100` | Alpha 数量。 |
| `cv_splits` | iterable | `None` | 自定义 folds。 |
| `loss_kwargs` | dict | `None` | loss 选项；Cox 接受 ties 设置。 |
| `compute_inference` | bool | `False` | 是否在 selected full-data final refit 上运行 coefficient inference。fold/grid/path 始终 estimation-only。 |
| `inference_method` | str | `"auto"` | final refit inference 方法请求。 |
| `cov_type` | str | `"nonrobust"` | final refit covariance。non-Gaussian L2 M-estimation 当前支持 nonrobust/HC0/HC1。 |
| `hac_maxlags` | int/None | `None` | 保留底层接口字段；不代表 non-Gaussian penalized M-estimation 已支持 HAC。 |

成功的 `PenalizedGLM_CV` coefficient inference 会从 `estimator_` 委托 `_inference_result` 及 requested/resolved/target provenance，并额外记录：

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

因此 SE/p-value/CI 条件于 CV 选出的 alpha，并未调整 tuning-selection uncertainty。

## 自定义 CV 分割

`cv_splits` 可传入自定义 fold：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l1",
    cv_splits=list(tscv.split(X)),
)
model.fit(X, y)
```

`cv_splits=None` 时使用维护中的 k-fold generator。Cox 自定义 fold 额外要求 train/validation 都有事件支持，并在 candidate fit 前严格验证索引类型、范围、重复和交叠。

## 样本权重

标量响应 CV 的 sample-weight 能力取决于具体 loss/penalty/solver path。不要从某个 solver 的限制外推出整个 estimator 都不支持权重。

对本次新增的 non-Gaussian L2 coefficient inference，analytic weights 由 final-refit M-estimation contract 支持。若 inference-enabled final refit 保留 `solver="auto"`，weighted non-Gaussian L2 会在该次 fit 内使用已有 backend-native FISTA，以避免 Newton 对非均匀权重的限制；公开 solver request 仍保持 `auto`。

Cox branch 仍拒绝 `sample_weight`。

## Alpha 网格

### 自动生成网格

自动网格在 statgpu average-loss scale 上构造。不同 loss/penalty 的零模型 KKT 边界和启发式细节由对应 CV 实现负责；ElasticNet 需要按 L1 component 调整 alpha_max。无 penalty 的 Cox 别名不可调，应直接拟合无惩罚 estimator。

### 自定义网格

标量响应 CV 会在 candidate work 前验证用户网格，保留有效数值网格的顺序；不同 penalty 对零值/正值要求按各自 contract 执行。Cox 网格使用更严格 fail-closed 规则。

## 拟合属性

常见 CV 拟合属性：

| 属性 | 说明 |
|------|------|
| `alpha_` | 选定 alpha |
| `best_score_` | 最优 CV score |
| `cv_results_` | CV 路径、fold score 与策略 provenance |
| `estimator_` | selected hyperparameter 的全数据 final refit |
| `coef_` | final refit 系数 |
| `intercept_` | final refit 截距 |

当 `PenalizedGLM_CV(compute_inference=True)` 且 selected final-refit row 受支持时，还会发布 `_inference_result`、`_bse`、`_pvalues`、`_conf_int` 及 inference provenance。

## 评分

```python
pred = model.predict(X_test)
score = model.score(X_test, y_test)
```

`score()` 委托给 selected final estimator 或对应 CV family 的维护评分语义。

## 设备选择

`device="auto"` 根据问题规模、loss×penalty 与实际 GPU 可用性选择后端。显式 `device="cuda"` / `"torch"` 使用严格设备语义：不可用时抛错，不静默回 CPU。

当前标量 CV 仍包含 benchmark-backed threshold，例如 small problem CPU、部分高维 sparse GLM 选择 Torch，以及 large effective-work fallback。具体阈值属于实现策略，不是统计定义。

## CV 后推断

### RidgeCV / sparse Gaussian CV

Ridge/Lasso/ElasticNet 的维护推断继续遵循各自模型页和专用 contract。

### PenalizedGLM_CV

`PenalizedGLM_CV` 的 coefficient inference 是 **final-refit-only**：

1. folds/grid/path scoring 全部 `compute_inference=False`；
2. 先完成 alpha 选择；
3. 若请求 inference，只对全数据 selected-alpha estimator 运行一次；
4. 直接委托 final estimator 的 inference result/provenance；
5. 明确报告 selection adjustment 为 `False`。

支持矩阵与统计解释见 [Penalized GLM inference](penalized-glm-inference.md)。重点包括：

- smooth non-Gaussian L2/no penalty：`m_estimation`；
- non-Gaussian L1/ElasticNet：当前不支持 coefficient inference；
- SCAD/MCP oracle 不由 `auto` 静默选择；
- Cox branch：estimation-only。

## 性能建议

1. 大规模问题优先使用 `device="auto"` 并让 benchmark policy 决定后端。
2. 不需要高分辨率 path 时减少 alpha 数量。
3. `cv_strategy="two_stage"` 是显式 approximate screening；strict refinement/final refit 仍使用维护的严格设置。
4. 自定义 folds 可用于 time-series/repeated-holdout 等设计。

## 架构

`PenalizedGLM_CV.fit()` 分成 scalar-response 与 survival-aware 两条 preparation/routing 路径。共同原则是：**selection 与 inference 分离**。

### scalar-response

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

### penalized Cox

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

## 主要 CV scoring 路径

### Ridge eigendecomposition

squared-error L2 的 CPU scoring 可通过每 fold 一次 eigendecomposition 复用多个 alpha。最终 refit 保持既有精确 float64 语义。

### Fold-batch GLM sparse path

部分 GPU sparse GLM 路径将多个 fold 的线性代数批量化，以减少 kernel launch 和 host synchronization。

### Sparse Gram / warm start

squared-error L1/ElasticNet 可复用 fold Gram matrix，并按递减 alpha warm-start。

### SCAD/MCP LLA

non-convex penalty 使用 continuation + LLA/FISTA；因此比 convex sparse path 更昂贵。

### generic fallback

没有专用 path 时使用逐 fold / candidate 的维护 estimator fit，并保持声明的 loss objective；不能为了“跑完 CV”静默替换成 MSE。

## 两阶段 CV

`cv_strategy="two_stage"`：

1. 在完整 alpha grid 上使用放松设置筛选；
2. 选择 top candidate；
3. 对候选运行 strict refinement；
4. final refit 仍使用 strict/full settings。

该策略可能在极接近的 CV 曲线下改变 ranking，因此需要显式 opt-in/acknowledgement。

## GPU 加速原则

维护实现通过 backend-native FISTA/FISTA-BB、预计算、warm start、批量 validation scoring、延迟同步和 Torch compile 等手段减少 CV 总开销。性能策略不能改变显式 device 语义、loss objective 或 inference target。

## 结果缓存

`LassoCV` 的 selection cache 仅缓存 alpha-selection evidence，不缓存 final estimator。key 包含数据/权重身份、实际 alpha grid、完整 fold indices、CV solver/method、容差与执行模式等会影响 selection 的字段。最终 refit solver 不影响 alpha scoring，因此不属于 selection key。

## Alpha 约定

statgpu penalized estimators 使用 average-loss objective。和 sklearn/R 比较时必须先对齐 penalty scale；例如 Ridge 常需要 `sklearn_alpha = statgpu_alpha * n`。同名 alpha 不保证不同框架目标函数相同。

## 已知限制

- Penalized Cox CV 不支持 coefficient inference 或 sample weights。
- non-Gaussian L1/ElasticNet coefficient inference 尚未实现。
- non-Gaussian penalized M-estimation 当前只支持 nonrobust/HC0/HC1 covariance。
- weighted/robust/family-aware bootstrap 不在当前 residual-bootstrap contract 中。
- 显式 solver 的 sample-weight 能力仍由该 solver 自身决定；inference contract 不会覆盖用户的显式 solver 选择。

## FAQ

**Q: 为什么 CV cache 没命中？**  
selection evidence 相关的 data/fold/grid/solver/tolerance/execution 字段变化会导致 miss。

**Q: `n_jobs` 有什么作用？**  
部分接口保留该参数用于兼容/未来并行策略；具体 fold execution 以当前实现为准。

**Q: SCAD/MCP 为什么比 L1/ElasticNet 慢？**  
它们需要 LLA/continuation 外循环。

**Q: CV inference 是否校正 hyperparameter selection？**  
否。`penalty_selection_adjusted_=False`，结果条件于 selected alpha。

## 参见

- [Penalized GLM inference](penalized-glm-inference.md) — final-refit inference target、支持矩阵、backend/resampling 边界
- [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md)
- [Ridge 模型](../models/ridge.md)
- [ElasticNet 模型](../models/elastic-net.md)
- [GLM 模型](../models/generalized-linear-model.md)

## External Validation

CV 维护测试覆盖参数验证、fold 完整性、selection cache、solver migration、Ridge/ElasticNet/Lasso 路径及 penalized Cox safety contract。本次 PenalizedGLM_CV inference repair 另外覆盖 final-refit-only inference、selected-alpha 复用、clone/API compatibility 与 selection-conditioning provenance。
