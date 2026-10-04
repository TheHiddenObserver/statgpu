# CoxPH

> 语言：中文<br>
> 最后更新：2026-10-04<br>
> 页面定位：模型文档<br>
> 切换：[English](../../en/models/coxph.md)

## 概览

当问题是**哪些因素与事件发生前的等待时间有关**，且观察可能在事件发生之前结束时，
可以使用 Cox 回归，例如分析机器发生故障的时间。只对已发生故障的样本做时间回归，
会丢失尚未发生故障的随访信息；只把结果分成“发生/未发生”，又会忽略观察时长。

对于普通右删失数据，每行包含特征 `X`、正的观察时长 `time` 和事件指示 `event`：

- `event=1`：事件在 `time` 时刻发生；
- `event=0`：观察在 `time` 时刻结束，尚未观察到事件。真实事件时间晚于该随访时间，
  不能把它当作零，也不能认为事件永远不会发生。

应保留删失行。通常的统计解释要求：给定模型中的协变量与研究设计后，删失过程与事件过程独立。

模型为 $h(t\mid x)=h_0(t)\exp(x^\top\beta)$：基线风险描述瞬时事件风险如何随时间变化，
协变量则以乘法方式改变风险。同一分层内，对于固定的协变量，**比例风险（PH）假设**要求
两种特征组合之间的风险比不随时间变化。其他特征保持不变时，第 `j` 个特征增加一个单位，
瞬时风险乘以 `exp(coef_[j])`。风险比不是事件概率、生存时间，也不自动具有因果含义。
应结合研究设计检查 PH 假设是否合理；优化器收敛并不意味着这些假设成立。
时变协变量允许 `x(t)` 变化，但拟合的回归系数仍不随时间变化。

`CoxPH` 在 NumPy、CuPy CUDA 与 Torch CUDA 后端实现比例风险回归，支持
Breslow、Efron 与 Exact 三种并列事件处理方式，同时覆盖普通右删失、延迟进入、
计数过程 `(start, stop]` 行、独立分层（`strata`）、时变协变量、稳健/聚类协方差，以及
通过 `CoxPHCV` 选择 L2 惩罚。

重要行为：

- 显式 `device="cuda"` 与 `device="torch"` 不会静默回退 CPU；
- `entry=` 与 `start=` 是互斥的别名；
- 某行在时刻 `t` 进入风险集，当且仅当 `start < t <= stop`，且其分层标签
  与事件所属分层相同；
- `subject_id=` 标识同一受试者的重复行，用于 concordance、sandwich 聚合，并确保同一受试者的记录不会被拆到不同的交叉验证折中；
- `compute_inference=False` 仅执行估计，推断字段和基线风险字段保持未设置。

## 第一个 CPU 完整示例

这个完整示例只需要 NumPy 与 StatGPU，无需 GPU 或外部数据。它模拟独立受试者的右删失数据，
三个特征已处于可比尺度，并在拟合前留出最后 100 位受试者。真实数据应按受试者、群组或时间
选择合适的划分方式；缩放等预处理只能使用训练部分来学习。

<!-- example: coxph-cpu-walkthrough -->
```python
import numpy as np
from statgpu.survival import CoxPH, CoxPHCV

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 3))
true_coef = np.array([0.8, -0.5, 0.3])
event_time = rng.exponential(scale=np.exp(-(X @ true_coef)))
censor_time = rng.exponential(scale=2.0, size=len(X))
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(np.int64)

X_train, X_test = X[:300], X[300:]
time_train, time_test = time[:300], time[300:]
event_train, event_test = event[:300], event[300:]

model = CoxPH(
    ties="efron", device="cpu", compute_inference=True,
).fit(X_train, time_train, event_train)
if not model.converged_:
    raise RuntimeError(
        f"{model.optimization_stop_reason_}: "
        f"normalized KKT={model.final_kkt_normalized_}"
    )
print("Coefficients:", model.coef_)
print("Per-feature hazard ratios:", model.hazard_ratios_)
print("Convergence:", model.termination_reason_, model.n_iter_)
print(model.summary())

log_risk = model.predict_risk_score(X_test[:2])
relative_hazard = model.predict(X_test[:2])
requested_times = np.array([0.0, 0.5, 1.0, 2.0])
curves, curve_times = model.predict_survival(
    X_test[:2], times=requested_times,
)
held_out_cindex = model.score(X_test, time_test, event_test)
print("Log-risk:", log_risk)
print("Relative hazard:", relative_hazard)
print("Survival shape and times:", curves.shape, curve_times)
print("Held-out C-index:", held_out_cindex)
```
<!-- /example: coxph-cpu-walkthrough -->

这个随机种子下，系数约为 `[0.852, -0.457, 0.312]`，对应的风险比约为
`[2.345, 0.633, 1.367]`。例如，其他特征保持不变时，第一个特征增加一个单位，估计的
瞬时风险约变为原来的 2.35 倍；第二个特征与更低的风险相关。这些是模拟数据中的关联，
不应作为实际干预建议。

`predict_risk_score(X)` 返回 `X @ coef_`；`predict(X)` 和
`predict_hazard_ratio(X)` 返回 `exp(X @ coef_)`，其参照是同一分层内协变量全为零的样本。
比较两种特征组合时，应对两者的对数风险之差取指数。`hazard_ratios_` 则是每个**特征**的风险比。

`predict_survival` 返回 **`(curves, times)` 元组**，并非单独一个矩阵：
`curves` 的形状为 `(n_new, n_times)`，`times` 的形状为 `(n_times,)`；
本例分别是 `(2, 4)` 与 `(4,)`。每行估计该样本在各时刻之后仍未发生事件的概率，
取值在 `[0, 1]` 内，并在递增时间网格上单调不增。`times=None` 使用拟合基线的事件时间网格
（有分层时取各层的并集）；显式传入时间时保留请求顺序。基线为阶梯函数，在最后一个事件时间
之后保持不变，因此延长预测网格不意味着获得了可靠的长期外推。
对于计数过程数据，每个预测行表示固定的协变量组合，不会自动沿未来协变量轨迹积分。

留出集 C-index 约为 `0.766`：在可比较的样本对中，模型倾向于给更早发生事件的样本更高风险。
它衡量排序区分能力，不是概率校准指标或 R-squared。接近 `0.5` 表示中性排序，
`1.0` 表示可比较样本对上的完美排序，低于 `0.5` 提示排序可能相反。
没有可比较样本对时也返回 `0.5`；此时是评估证据不足，不能据此认定模型表现等同于随机。
不要把训练集 concordance 当作留出评估。

## 输入形状与 fit API

矩阵接口为 `CoxPH(...).fit(X, time, event, ...)` 或
`CoxPHCV(...).fit(X, time, event, ...)`；两者均返回拟合后的估计器。

| 输入 | 形状与约定 |
|---|---|
| `X` | 有限实数矩阵 `(n_samples, n_features)`，不要添加截距/常量列；预测时保持特征顺序。`CoxPH` 也接受一维单特征输入；`CoxPHCV` 应使用矩阵。 |
| `time` | 有限正数向量 `(n_samples,)`，统一时间单位；表示事件或删失时间，也可表示区间终止时间。 |
| `event` | 仅含 `0` 或 `1` 的向量 `(n_samples,)`；拟合至少需要一个已观察事件。 |
| `entry` / `start` | 可选向量 `(n_samples,)`，满足 `0 <= start < time`；二者为互斥别名，不能同时提供。省略时从零时刻进入。 |
| `strata` | 可选标签 `(n_samples,)`，各层具有独立风险集和基线风险，但共享系数。 |
| `subject_id` | 可选标签 `(n_samples,)`，标识同一受试者的重复记录，用于 concordance、稳健协方差聚合及 CV 划分。 |
| `cluster` | 可选标签 `(n_samples,)`；`cov_type="cluster"` 时必须提供。仅提供 cluster 不会使 CV 自动按聚类分组；应使用合适的 `subject_id` 或显式 `cv_splits`。 |
| `init_coef` | `CoxPH` 可选的有限初始系数向量 `(n_features,)`；不是 `CoxPHCV.fit` 参数。 |
| `formula`, `data` | `CoxPH.fit` 的公式接口，见后文；`CoxPHCV.fit` 不接受这两个参数。 |

两者也支持省略 `event` 的 `fit(X, y)`：`y` 可以是列为 `[time, event]` 的
`(n_samples, 2)` 数组，或列为 `[start, stop, event]` 的 `(n_samples, 3)` 数组。
三列形式不能再单独提供 `entry`/`start`。`score` 也接受这些组合响应；
其区间参数叫 `start`，不是 `entry`。矩阵拟合前应先处理数值缺失，非有限数组会被拒绝，
不会静默删除对应行。

完整构造参数见[参数](#参数)与 [CoxPHCV 参数](#coxphcv-参数)。规范的方法签名与实现契约见
[`CoxPH`](../../../statgpu/survival/_cox.py) 和
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py)。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`、`"efron"` 或 `"exact"` |
| `tol` | `1e-9` | Newton/KKT 收敛阈值 |
| `max_iter` | `100` | 最大迭代次数 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"` |
| `n_jobs` | `None` | 接受共享的 CPU 作业数设置；当前 Cox 拟合/CV 循环不通过它并行处理各折。 |
| `compute_inference` | `True` | 计算协方差、检验与基线风险 |
| `compute_cindex` | `True` | 计算训练集 concordance |
| `cov_type` | `"nonrobust"` | `"nonrobust"`、`"hc0"`、`"hc1"` 或 `"cluster"` |
| `penalty` | `0.0` | 非负 L2 惩罚 |
| `inference_mode` | `"strict"` | `"strict"` 或兼容别名 `"approx"`；两者均执行精确推断 |
| `gpu_memory_cleanup` | `False` | 尝试释放 CuPy/Torch 缓存 |

## 交叉验证

`CoxPHCV` 使用相同的 `ties`、`start`/`entry`、`strata` 与后端语义评估 L2 惩罚
网格，再以最佳 `penalty` 重拟合 `CoxPH`。传入 `subject_id` 后，同一受试者的
全部行会被保留在同一自动生成的交叉验证折中；若用户提供的 `cv_splits` 使同一
受试者同时出现在训练集和验证集中，则会被拒绝。`inference_mode` 与
`compute_inference` 会转发到最终重拟合。

在[第一个 CPU 完整示例](#第一个-cpu-完整示例)之后运行以下代码。只用其中 300 位训练受试者
选择惩罚，再在同一份未参与选择的 100 人测试集上评估最终重拟合模型。
L2 会收缩系数，因此特征尺度会影响惩罚；真实数据的预处理应在每个训练折内拟合，不能使用验证集或测试集信息。

<!-- example: coxph-cpu-cv -->
```python
cv_model = CoxPHCV(
    penalties=[0.0, 0.1, 1.0, 10.0],
    cv=3,
    random_state=42,
    ties="efron",
    device="cpu",
    compute_inference=False,
).fit(X_train, time_train, event_train)
if not cv_model.converged_:
    raise RuntimeError(cv_model.optimization_stop_reason_)

cv_mean_pl = cv_model.cv_results_["mean_pl"]
cv_fold_counts = cv_model.cv_results_["effective_fold_counts"]
cv_test_cindex = cv_model.score(X_test, time_test, event_test)
print("Penalty grid:", cv_model.penalties_)
print("Mean held-out partial log-likelihood:", cv_mean_pl)
print("Effective fold counts:", cv_fold_counts)
print("Selected penalty:", cv_model.penalty_)
print("Test C-index:", cv_test_cindex)
```
<!-- /example: coxph-cpu-cv -->

本例选择 `penalty_=1.0`，测试 C-index 约为 `0.766`；CV 并不保证提高这个指标。
`best_score_` 是最大的**平均留出部分对数似然**，不是 `score()` 返回的 C-index，
只适合在相同数据、划分和评分约定下比较。`cv_results_["pl_path"]` 形状为
`(n_penalties, n_folds)`；`mean_pl` 与 `effective_fold_counts` 的形状均为 `(n_penalties,)`。
无法评估候选时，可检查 `converged_path`、`failure_path` 和 `fold_valid`。
可被选中的候选必须在同一组全部有效折上收敛且分数有限；每个有效折的训练和验证部分都必须有事件。
没有合格候选时，拟合会报错，不会发布选择结果。

`estimator_` 是使用所选 `penalty_`，在全部传入训练行上重拟合的 `CoxPH`。
`compute_inference=False` 时仍能预测对数风险和风险比，但没有生存曲线及推断结果。
启用最终重拟合推断也不会校正调参不确定性或收缩偏差，详见[惩罚强度缩放与推断](#惩罚强度缩放与推断)。

### CoxPHCV 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `penalties` | `None` | 非空、有限、非负的一维 L2 惩罚网格；`None` 自动生成。 |
| `n_penalties` | `100` | 自动网格的候选数量。 |
| `penalty_min_ratio` | `1e-3` | 自动网格最小值/最大值的比值，范围 `(0, 1]`。 |
| `cv` | `5` | 自动生成的折数，至少为二。 |
| `cv_splits` | `None` | 显式 `(train_indices, validation_indices)` 对，覆盖自动划分；每组索引应非空、一维、互不重叠，且为范围内的整数。 |
| `ties` | `"breslow"` | 候选及重拟合使用 `"breslow"`、`"efron"` 或 `"exact"`。 |
| `tol` | `1e-9` | 候选及重拟合的收敛容差。 |
| `max_iter` | `100` | 候选及重拟合的最大迭代次数。 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`。 |
| `n_jobs` | `None` | 共享 CPU 作业数选项，传给最终重拟合；当前 CV 循环不通过它并行化。 |
| `compute_inference` | `True` | 仅在最终重拟合计算推断和基线风险。 |
| `cov_type` | `"nonrobust"` | 最终重拟合的协方差约定。 |
| `inference_mode` | `"strict"` | `"strict"` 或兼容别名 `"approx"`，两者均执行已支持的精确推断。 |
| `gpu_memory_cleanup` | `False` | 在公共计算边界尽力清理 GPU 缓存。 |
| `random_state` | `None` | 自动生成 CV 划分的随机种子。 |

与 `CoxPH` 不同，`CoxPHCV` 构造器没有 `penalty` 或 `compute_cindex`；
应使用 `penalties` 并显式调用 `score`。完整源代码参考见
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py)。


同样的惩罚搜索也可使用 CuPy 或 Torch CUDA 数组。先运行 [CPU 与 GPU 示例](#cpu-与-gpu-示例)中的相应准备代码：

```python
cupy_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1], cv=5, device="cuda",
    compute_inference=False,
).fit(X_cp, time_cp, event_cp)

torch_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1], cv=5, device="torch",
    compute_inference=False,
).fit(X_t, time_t, event_t)
```

### L1/L2/ElasticNet/SCAD/MCP 模型族交叉验证

上面的 `CoxPHCV` 是标准的 L2 Cox 选择器，并可按配置执行最终重拟合推断。
公开的带惩罚模型族则使用 `PenalizedGLM_CV` 的独立生存感知分支：

以下补充示例使用本页任一数据准备示例中的 `X`、`time` 与 `event`。

<!-- example: coxph-penalized-family-cv -->
```python
from statgpu.linear_model import PenalizedGLM_CV

survival_y = np.column_stack([time, event])
penalized_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="mcp",               # l1、l2、elasticnet、scad 或 mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # 也可用 "cuda" / "torch"
).fit(X, survival_y)
```
<!-- /example: coxph-penalized-family-cv -->

该分支始终保留二维 `(time, event)` 响应，禁止截距，使用留出数据上的未惩罚偏似然评分，并要求每个可评估的交叉验证折都提供有限证据。若不存在满足
契约的 `alpha`，拟合会抛错，并且不会发布已选 `alpha` 或已拟合估计器。最终重拟合使用
`PenalizedCoxPHModel(compute_inference=False)`；不支持选择后系数推断、`two_stage`、样本权重或字典形式响应。无惩罚别名不可调，因此该 CV
路径会拒绝，需改为直接拟合模型。

自定义交叉验证折可以采用一般的非空、训练集与验证集互不重叠的划分，包括前向
`TimeSeriesSplit` 或重复留出；无需互为补集，也无需让每行恰好进入一次验证集。索引会在任何候选模型拟合前校验，必须是一维、精确且位于范围内的
整数。自动网格中，ElasticNet 在 `l1_ratio > 0` 时采用零模型 KKT 边界
`alpha_max = ||gradient L(0)||_inf / l1_ratio`，惩罚对象使用自身的混合比例参数。
纯 L2（`l1_ratio=0`）不存在有限的全零 KKT 阈值，因此把零模型得分的原始
无穷范数作为已文档化的网格启发式规则。

大规模 `device="auto"` 搜索只在 Torch 或 CuPy 的 CUDA 后端确认设备实际可用
后选择 GPU。回退路径的规模估计只统计训练集和验证集都含事件的可评估折；其他规范化后的折仍记录在 `failure_path` 中，但不会夸大 GPU 工作量。CuPy
可导入但无法运行时会回退 CPU；显式 `device="cuda"` 仍严格抛错，不会静默回退。

## 预测与评分

对数组输入，`predict`、`predict_risk_score`、`predict_hazard_ratio`、
`predict_survival` 与 `score` 都在拟合后端执行。使用 `device="auto"` 拟合后，模型会固定实际的
`effective_device_`；后续修改全局设备设置不会迁移既有模型的预测或评分后端。分层生存预测要求每个预测行
提供一个训练时已知的分层标签；即使拟合时只有一个显式分层，也不能省略
标签，缺失或未知标签会抛出 `ValueError`。生存曲线在对数域中累计基线风险，
以提高数值稳定性。使用公式接口拟合的模型会在预测前应用已保存的设计矩阵转换。

`score()` 复用同一行标签编码器：传入的 `strata` 必须具有 `(n_samples,)` 形状；
显式分层模型只接受训练时已知标签，多分层拟合在评分时必须提供标签。
标量、二维、长度错误或未知标签都会在后端计算 concordance 前统一抛出
`ValueError`。

若某个已拟合分层没有观察到任何事件，其空的基线风险状态是合法状态。
该分层在任意时间的累计基线风险均为零，因此 `predict_survival()` 精确返回
1。显式 `times`、自动 `times`、混合分层预测行和 `CoxPHCV` 委托路径都遵守此契约；
存储的时间/风险数组形状不匹配仍属于非法状态。

`predict_risk_score()` 返回未取指数的对数风险。标准 CoxPH、CV 与带惩罚 Cox 的风险比预测 API 共享严格的 float64 指数边界；标准 CoxPH/CV 拟合后
`hazard_ratios_` 采用相同边界。会溢出为无穷或下溢为零的值，在标准 CoxPH/CV 拟合时抛出 `CoxFitNumericalError`，在预测时抛出 `FloatingPointError`，不会按估计器专属阈值静默截断。`PenalizedCoxPHModel` 也提供 `predict_risk_score()`，因此
极端但有限的对数风险仍可直接读取。

## 输出

- 参数：`coef_`、`hazard_ratios_`；
- 推断：启用时的 `_bse`、`_zvalues`、`_pvalues`、`_conf_int`；
- 诊断：定义时的 `log_likelihood`、`aic`、`bic`、`concordance_index`；
- 收敛：`converged_`、`termination_reason_`、`optimization_stop_reason_`、`n_iter_`、
  `final_kkt_inf_`、`final_kkt_normalized_`；
- 推断来源与执行信息：`inference_method_`、`inference_backend_`、
  `inference_approximate_`、`inference_fallback_reason_`、
  `inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_`、
  `full_host_transfer_performed_`。

`CoxPHCV` 还会公开 `cv_full_host_transfer_performed_`、
`final_refit_full_host_transfer_performed_` 与 `orchestration_device_`，避免数据移动审计
将主机端的 CV 选择与最终重拟合混淆。
`cv_results_` 会区分选择来源字段（`scoring_device`、
`selection_origin_device`、`candidate_preparation_origin_device` 与总准备次数）和本次调用字段（`selection_cache_hit`、
`requested_fit_device`、`effective_device` 与 `*_this_call` 次数）。缓存命中时，本次调用不会重复准备交叉验证折，也不会再次传输响应向量；同时不会改写原先记录的选择来源设备。
若有限输入的候选返回非有限系数或似然，`CoxPH` 会抛出
`CoxFitNumericalError`（`FloatingPointError` 子类）；`CoxPHCV` 只排除这类
候选，输入错误、内存分配器错误、CUDA 错误以及其他非预期运行时错误仍会原样传播。

## CPU 与 GPU 示例

三个后端使用相同的统计输入，并在拟合后端返回预测数组。以下可选后端示例与前面的留出评估示例独立。先运行一次以下确定性数据准备：

<!-- example: coxph-backend-data -->
```python
import numpy as np

from statgpu.survival import CoxPH, CoxPHCV

rng = np.random.default_rng(20260730)
n = 256
X = rng.normal(size=(n, 3))
log_risk = X @ np.array([0.45, -0.30, 0.20])
event_time = rng.exponential(scale=np.exp(-log_risk))
censor_time = rng.exponential(scale=1.8, size=n)
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(np.float64)
```
<!-- /example: coxph-backend-data -->

NumPy / CPU：

<!-- example: coxph-backend-cpu -->
```python
cpu_model = CoxPH(
    ties="efron",
    device="cpu",
    compute_inference=False,
).fit(X, time, event)
cpu_log_risk = cpu_model.predict_risk_score(X[:3])
```
<!-- /example: coxph-backend-cpu -->

CuPy / CUDA：

```python
import cupy as cp

X_cp = cp.asarray(X)
time_cp = cp.asarray(time)
event_cp = cp.asarray(event)
cupy_model = CoxPH(
    ties="efron",
    device="cuda",
    compute_inference=False,
).fit(X_cp, time_cp, event_cp)
cupy_log_risk = cupy_model.predict_risk_score(X_cp[:3])
```

Torch / CUDA：

```python
import torch

X_t = torch.as_tensor(X, dtype=torch.float64, device="cuda")
time_t = torch.as_tensor(time, dtype=torch.float64, device="cuda")
event_t = torch.as_tensor(event, dtype=torch.float64, device="cuda")
torch_model = CoxPH(
    ties="efron",
    device="torch",
    compute_inference=False,
).fit(X_t, time_t, event_t)
torch_log_risk = torch_model.predict_risk_score(X_t[:3])
```

当相应软件包、CUDA 运行环境或设备不可用时，显式 CUDA 请求会报错，不会静默转到 CPU。需要协方差、检验或生存曲线时，设置 `compute_inference=True`。

## 目标函数与估计方程

对第 `i` 行的起始时间 `a_i`、终止时间 `b_i`、事件指示 `delta_i` 与分层
`s_i`，时刻 `t` 的风险集为

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

无并列失败时，分层 Cox 部分对数似然为

$$
\ell(\beta)=\sum_s\sum_{i:\delta_i=1,\ s_i=s}
\left[x_i^\top\beta-
\log\left\{\sum_{j\in R_s(b_i)}\exp(x_j^\top\beta)\right\}\right].
$$

Breslow、Efron 与 Exact 按各自定义替换并列事件分母，但沿用相同的
`(start, stop]` 风险集。令 `penalty=lambda`，StatGPU 最大化总和尺度的目标：

$$
Q_\lambda(\beta)=\ell(\beta)-\lambda\lVert\beta\rVert_2^2.
$$

记 `U(beta)` 为未加惩罚的偏似然得分，拟合系数满足

$$
U_\lambda(\beta)=U(\beta)-2\lambda\beta=0.
$$

若 $J(\beta)=-\partial U(\beta)/\partial\beta$ 是未惩罚观测信息，则带惩罚的 Newton 方法
使用的导数为 $A(\beta)=J(\beta)+2\lambda I_p$。

## 风险集与并列事件处理

`ties="breslow"` 和 `ties="efron"` 使用对应的并列事件偏似然；
`ties="exact"` 通过基本对称多项式（elementary-symmetric）动态规划计算 Exact 分母。
延迟进入、分层、Exact 并列事件、L2 惩罚拟合与 GPU 稳健推断共用同一套
计数过程风险集计算，因此三个后端遵循一致的 `(start, stop]` 约定。
公开的 `CoxPH` 与 `CoxPHCV` 估计器会对一维标签进行编码：主机端字符串/对象以及
有限的 CuPy/Torch 数值标签都会在内部转换为连续的 int64 编码。底层计数过程函数
不再重复执行这一编码，因此传入的数值编码必须有限、为整数，并且能够由有符号 int64 表示。

对于普通右删失的 Exact 拟合，风险集在各分层内部具有嵌套结构。
StatGPU 先按分层、再按终止时间 `stop` 降序排列样本，并在 NumPy、CuPy、Torch 上让
所有事件组复用同一个分段基本对称多项式前缀动态规划，不再通过 Python
逐分层循环，也避免随事件组数量重复扫描风险集。
事件组分子改用后端原生的分组归约，不再构造 `事件组 × 样本` 的稠密掩码。
前缀工作区默认上限为 512 MiB，由 `STATGPU_EXACT_NESTED_MAX_BYTES` 控制，且在
分配前完成检查。

在 Torch CUDA 上，PyTorch 2.0 对长轴执行多维 `cumsum(dim=0)` 时，可能成为这条
线性前缀 DP 的主要耗时。当样本数至少为 2,048、尾部矩通道数不超过 64 时，
StatGPU 会将每个通道连续布局，分别执行高效的一维 CUDA 扫描，再在设备上拼回原
形状。`STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` 与
`STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` 可配置这两个保守门禁。CPU、小样本和宽
张量保留 Torch 原生多维扫描。`STATGPU_TORCH_EXACT_SCAN_STRATEGY` 可设为 `auto`、
`native` 或 `channelwise`；`auto` 仅在已有实测证据的 Torch 2.0 + Pascal/P100
组合启用分通道扫描，未经验证的 Torch/GPU 组合使用原生扫描。额外的转置与输出
工作区也计入嵌套路径的内存检查：
若基础 DP 所需内存可容纳、但分通道扫描的额外空间不足，则继续使用嵌套算法的原生 Torch 扫描，不会回退到开销更高的通用 Exact 路径。

延迟进入不满足嵌套前缀条件。当分层数量至少为 8 时，GPU 后端会先通过一次后端原生批处理处理所有合格的事件组；分层更少时，GPU 与 NumPy 都按分层逐批处理，以避免计算跨分层的空掩码。独立的 512 MiB 上限由
`STATGPU_EXACT_BATCH_MAX_BYTES` 控制。全局批处理工作区超限时先按分层重试，
再使用逐组内存受限路径；构造得分残差或触发保守的数值范围检查时，也保留归一化实现。这些都是显式算法回退，不会隐式回退到 CPU。

对于 Breslow/Efron 的延迟进入目标函数，
`STATGPU_COX_GROUP_MAX_BYTES` 控制稠密事件组工作区，默认
512 MiB。如果单个事件组已经超过上限，所选 GPU 后端会改用
数值稳定的多遍逐行流式矩计算，从而避免即使最小批量大小为 1 时仍分配不受限制的 `O(n)` 掩码。

完整拟合的推断阶段还需要构造 Breslow 基线风险。对于普通右删失行，
StatGPU 现在在每个分层内按终止时间 `stop` 降序排列，并通过一次对数风险前缀
得到所有风险分母：NumPy 使用 `logaddexp.accumulate`，Torch 使用
`logcumsumexp`，CuPy 在线性预测量处于保守数值范围内时使用平移后的指数累积和。
极端 CuPy 线性预测量与延迟进入数据继续使用数值稳定的后端原生逐事件组实现。
这移除了普通右删失常用路径中原先的 `事件组 × 样本` 风险掩码扫描。

## 公式接口

以下为接口示意，并非独立示例：需要包含对应列的 pandas DataFrame `df` 和可选 pandas/Patsy 公式依赖。支持两种生存响应：

```python
CoxPH().fit(formula="Surv(time, event) ~ age + C(group)", data=df)
CoxPH().fit(
    formula="Surv(start, stop, event) ~ age + treatment",
    data=df,
    strata=df["clinic"],
    subject_id=df["patient_id"],
)
```

公式接口删除缺失行时，会同步对齐 `entry`/`start`、`cluster`、`strata` 与
`subject_id`。三列 `Surv(start, stop, event)` 已定义起始时间，不能再同时传入
`entry=` 或 `start=`。

## 优化与收敛

Newton 迭代使用线搜索，并在最终参数处执行 KKT 检查。线搜索失败时不会更新系数，也不会报告收敛。公开拟合状态包括：

- `converged_`；
- `termination_reason_`；
- `optimization_stop_reason_`；
- `n_iter_`；
- `final_kkt_inf_`；
- `final_kkt_normalized_`。

似然、梯度、Hessian 矩阵、协方差、基线风险与公开收敛状态均根据最终系数向量重新计算。
`termination_reason_` 是解释后的用户级分类，只会是 `kkt_converged`、
`line_search_failed` 或 `stalled_with_large_kkt`。`optimization_stop_reason_`
保留底层求解器的原始退出原因（包括 `max_iter`），警告信息也会报告该原始值；
因此预算耗尽可以审计，但不会被误当作独立的收敛证书。

## 惩罚强度缩放与推断

`penalty` 就是上述总和尺度的偏似然目标中的 `lambda`，不会除以样本数或
事件数；CoxPH 也没有需要惩罚的截距。因此，复制全部观测会使似然与得分贡献加倍，却不会自动加倍用户提供的 `penalty`，从而改变有效正则强度。跨数据集或
样本规模比较时，应在目标抽样尺度下用 `CoxPHCV` 调参；复现采用平均损失的外部软件时，需要显式换算其 `penalty` 的尺度定义，不能假设数值直接相同。

正 L2 惩罚下，记 `J` 为拟合系数处未加惩罚的 Cox 观测信息，
`A = J + 2 * penalty * I_p`，则固定惩罚强度的频率学派代入式协方差为：

```text
A^-1 J A^-1
```

而不是 `A^-1`；后者更接近惩罚曲率或 Laplace 近似下的量，不能直接作为频率学派
抽样协方差发布。带惩罚的稳健推断同样使用带惩罚的 bread 矩阵，而 meat 矩阵仍由未加惩罚的聚合得分外积构成。

因此 SE/z/p/CI 与带惩罚 Wald 检验都以给定 `penalty` 为条件，目标是带惩罚的估计方程；它们不是针对无惩罚系数的纠偏推断，也不校正系数收缩偏差，或交叉验证选择 `penalty` 带来的额外不确定性。`CoxPHCV` 从最终重拟合复制相同契约，
并明确报告 `penalty_selection_adjusted_=False`。沿用 `PenalizedGLM` 的结果命名，
正 `penalty` 拟合的 `inference_method_` 使用简洁的 `"m_estimation"`；bread、meat、协方差定义、推断目标和条件化方式仍分别保留在推断结果的元数据中。

带惩罚拟合会关闭经典似然比检验、得分检验以及 AIC/BIC，不会把惩罚估计
当作无约束最大似然结果报告。该契约与 `PenalizedCoxPHModel` 分开；后者的
L1/Elastic Net/SCAD/MCP 接口仍仅支持估计。

## 协方差与推断

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 模型协方差；无惩罚时为信息矩阵的逆，有惩罚时为固定惩罚强度下的 sandwich 协方差 |
| `"hc0"` | 基于得分的 sandwich 协方差 |
| `"hc1"` | 带有限独立单元修正的基于得分的 sandwich 协方差 |
| `"cluster"` | 聚类稳健协方差；在 `fit` 时传入 `cluster=` |

无惩罚拟合的 `nonrobust` 协方差仍是通常的观测信息逆；正 `penalty` 协方差遵循
上一节的专门契约。

Breslow 与 Efron 的严格稳健推断使用 statgpu 内部的精确计数过程得分残差，不依赖 `statsmodels`。同一受试者的重复行会先按 `subject_id` 汇总再
形成 HC0/HC1 的 meat 矩阵；聚类稳健协方差按 `cluster` 汇总。

稳健推断必须具有可识别的独立单元变异。按受试者或聚类单元汇总后，HC0 与
聚类稳健协方差至少需要两个独立单元；HC1 还要求
`n_units > n_features`，因为其有限单元修正严格为
`n_units / (n_units - n_features)`。违反这些条件会抛出 `RuntimeError`，
不会把非正自由度分母替换为任意有限值。实质性负协方差对角线或非正稳健边际
方差同样会令严格推断失败，而不会发布零标准误与误导性的显著性结果。

边际方差为正并不保证稳健协方差在完整参数空间有效。StatGPU 会先用尺度感知容忍度
分类对称化后的协方差矩阵谱：正定矩阵同时支持边际推断和联合 Wald；半正定
但秩亏的矩阵仍保留逐系数的稳健标准误/z 值/p 值/置信区间，同时设置
`wald_test_available_=False` 并记录 `wald_test_failure_reason_`，`summary()` 会显示 `Robust Wald test unavailable`，不会使用不稳定逆矩阵或打印裸 `nan`。若存在实质性
负特征值，该矩阵已不是合法的协方差估计量；严格推断会抛出
`RuntimeError` 并清空本次拟合状态，而不会仅凭正对角线发布边际推断。即使逐系数与
Wald 推断使用稳健协方差，似然比检验与得分检验仍是经典的基于模型检验；`summary()` 会明确标注这一差异。

`inference_mode="strict"` 是默认值。为保持向后兼容，公开 API 仍接受
`inference_mode="approx"`，但统一拟合路径会把它作为仅用于兼容的别名，
继续计算精确的计数过程得分 sandwich 协方差。因此成功拟合会报告
`inference_approximate_=False`，且不会记录近似回退原因。

Exact 并列事件当前只支持模型协方差（`cov_type="nonrobust"`）。若在
`ties="exact"` 下请求 HC0、HC1 或 cluster 推断，会抛出
`NotImplementedError`。当 `compute_inference=False` 时，可以保留稳健
`cov_type` 标签，但不会计算协方差。

推断来源通过以下字段公开：

- `inference_method_`；
- `inference_backend_`；
- `inference_approximate_`；
- `inference_fallback_reason_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`；
- `wald_test_available_` 与 `wald_test_failure_reason_`；
- `full_host_transfer_performed_`。

对于 `CoxPHCV`，`full_host_transfer_performed_` 描述整个拟合过程，包括在主机端组织的交叉验证折构造与 `penalty` 选择。`cv_full_host_transfer_performed_` 与
`final_refit_full_host_transfer_performed_` 分别标记 CV 与最终重拟合阶段是否
将至少一个完整的设备端训练组件移到主机端；这包括排序后的响应向量，以及需要
保留的 `entry`、`strata` 或 `subject_id` 向量，即使设计矩阵仍留在 GPU 也会如实标记。
`orchestration_device_` 记录 CV 编排设备。
普通 GPU Breslow/Efron 预处理在选定后端完成排序，再把完整的已排序 `time` 与 `event` 向量复制到主机端以构建事件组元数据，因此会报告
`full_host_transfer_performed_=True`。

当请求 `STATGPU_COXPHCV_TWO_STAGE` 或
`STATGPU_COXPHCV_SUCCESSIVE_HALVING` 时，为保证正确性，NumPy、CuPy 与 Torch
当前都会禁用实验性筛选。CoxPHCV 会发出 `RuntimeWarning`，并对全部候选惩罚强度只执行一次完整精度的穷举评估。公开诊断记录
`staged_safety_strategy="single_pass_exhaustive"`、请求值与实际值两组状态、全为 `True` 的 `full_precision_candidate_mask`，以及全为 `False` 的
`screened_out_candidate_mask`。每个实际使用的交叉验证折在这次评估中只准备一次；不会启用分阶段保留缓存，也不会跨阶段重复准备。准备次数与响应向量传输次数仍保存在 `cv_results_` 中。
`selection_cache_hit`、
`requested_fit_device`、`fold_backend_preparation_count_this_call` 与
`candidate_target_host_transfer_count_this_call` 描述本次调用；
`selection_origin_device`、`candidate_preparation_origin_device` 和
`scoring_device` 记录选择结果的来源；`effective_device` 记录本次请求以及最终重拟合使用的设备。一次响应准备表示一整套 `time`/`event` 元数据准备；向量传输计数记录实际发生的两条向量复制。

## 支持矩阵

| 能力 | Breslow | Efron | Exact | NumPy | CuPy | Torch |
|---|---|---|---|---|---|---|
| 普通右删失 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 延迟进入 / `(start, stop]` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 独立 `strata` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 非负 L2 `penalty` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `nonrobust` 推断 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| HC0 / HC1 / `cluster` 推断 | 支持 | 支持 | 未实现 | 支持 | 支持 | 支持 |
| 后端原生预测数组 | 支持 | 支持 | 支持 | NumPy | CuPy | Torch |

`predict_survival` 需要已拟合的基线风险，因此需要生存曲线时应保留
`compute_inference=True`。风险得分与风险比预测不依赖基线风险。

## 外部验证与可复现性

维护的 R 基线使用 R 4.4.1 与 `survival` 3.8.9，并对齐并列事件处理方式、Newton
`max_iter=80` 和 `tol=1e-8`。在 `n=3000`、`p=10` 的 Breslow/Efron 比较中，
HC1 使用 3,000 个独立单元，聚类稳健协方差使用 120 个聚类单元。StatGPU 相对 R 的最大
系数/SE/p 值差异：HC1 为 `5.55e-16`/`1.39e-16`/`8.00e-19`，聚类稳健协方差为
`5.55e-16`/`1.32e-16`/`2.22e-16`。statsmodels 不支持的协方差模式会明确记录为不支持，不会换名后充当外部证据。

这些结果对应特定的数据规模和验证设置，不应解释为对所有数据与硬件都成立的统一精度或性能保证。Exact 并列事件和性能表现还会随问题规模与硬件而变化。具体的机器可读验证记录保留在开发验证材料中。

### 已发布的物理 GPU 验证

CuPy 与 Torch CUDA 验证检查数值一致性、设备归属和错误边界。稳定的
[已发布 GPU 验证记录](https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139)
（[原始 JSON](https://gist.githubusercontent.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139/raw/pr80_final_gpu_suite_schema3.json)）
绑定到源码提交 `a726937a39eb0ed5a370dd03362884b63a9e9818`，产物 SHA-256 为
`e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb`。
它只证明记录中的源码和运行环境；后续提交不会自动继承这一验证结论。
源码哈希、导入路径、硬件和验证来源保留在产物及
[开发验证材料](../../../dev/reviews/pr80_review_fix.md)中。

这份记录不是新的性能交叉点 benchmark，也不是新的 R 对齐运行，更不是对所有数据规模、
后端版本或 GPU 都成立的统一精度和性能保证；这些比较仍须以各自的专用产物为准。

## FAQ 与常见失败模式

| 现象 | 含义与处理 |
|---|---|
| 显式 `device="cuda"` 或 `device="torch"` 失败 | 对应软件包、CUDA 运行环境或设备不可用。安装兼容后端或改用 `device="cpu"`；StatGPU 不会静默回退。 |
| `predict_survival()` 提示基线风险不可用 | 使用 `compute_inference=True` 重新拟合；风险得分与风险比预测不需要基线风险。 |
| 分层生存预测拒绝标签 | 只要拟合时显式分层，每个预测行都必须提供一个训练时已知的分层标签，形状为 `(n_samples,)`；训练时只有一个分层也不能省略。 |
| 分层评分拒绝标签 | 多分层拟合必须逐行提供已知标签；单分层拟合可省略标签，但一旦提供，仍必须具有 `(n_samples,)` 形状且属于训练标签。 |
| 已知分层的生存率恒为 1 | 该分层在拟合数据中没有观察到事件，因此累计基线风险恒为零；这是合法的拟合状态，并不表示基线风险数据缺失。 |
| `HC1 covariance requires n_units > n_features` | 增加独立受试者/聚类单元，或减少特征，或采用研究设计能够支持的协方差契约。 |
| 稳健协方差要求至少两个独立单元 | 只有一个受试者/聚类单元时无法估计单元间变异；可用 `compute_inference=False` 仅执行估计。 |
| 观测信息矩阵奇异 | 检查共线性、常量列、分离（separation）或饱和（saturation）以及事件支持；减少设计或使用有明确依据的 L2 惩罚。 |
| 风险比预测抛出 `FloatingPointError` | `exp(X @ coef_)` 超出有限 float64 范围。检查 `predict_risk_score()`、缩放特征并检查外推。 |
| `converged_` 为 `False` | 检查 `optimization_stop_reason_`、`final_kkt_inf_` 与 `final_kkt_normalized_`；单纯增加 `max_iter` 不能修复线搜索失败或病态设计。 |
| Exact 并列事件很慢或触发工作区内存限制 | Exact 似然对最大并列事件组具有组合复杂度；科学上允许时使用 Breslow/Efron，或减小最大 Exact 并列事件组。 |
| `score()` 返回 `0.5` | 可能是风险排序没有区分度，也可能没有可用于 concordance 计算的样本对；后一种情况同样返回中性值 `0.5`。应检查数据是否包含可比较的样本对。 |

## 限制

- Exact 并列事件尚不支持稳健/聚类协方差；
- Exact 并列事件使用组合动态规划，适合规模适中的并列事件组，不适合无限制的大型并列事件组；
- 尚未实现 frailty（共享脆弱性）/随机效应项；
- 可选 `torch.compile` 加速要求兼容 Triton 的硬件，不属于跨平台正确性保证。

## 参考文献

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
- R survival 文档：[`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)。
