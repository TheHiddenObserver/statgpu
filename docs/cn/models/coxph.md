# CoxPH

> 语言：中文<br>
> 最后更新：2026-10-09<br>
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

模型为

$$
h_s(t\mid x)=h_{0s}(t)\exp(x^\top\beta).
$$

其中 $h_{0s}$ 为第 $s$ 层的基线风险率，$x$ 为协变量向量，$\beta$ 为各层共享的系数。
基线描述瞬时事件风险率如何随时间变化，协变量以乘法方式改变它。同一分层内，对于固定的协变量，**比例风险（PH）假设**要求
两种特征组合之间的风险比不随时间变化。其他特征保持不变时，第 `j` 个特征增加一个单位，
瞬时风险乘以 `exp(coef_[j])`。风险比不是事件概率、生存时间，也不自动具有因果含义。
应结合研究设计检查 PH 假设是否合理；优化器收敛并不意味着这些假设成立。
时变协变量允许 `x(t)` 变化，但拟合的回归系数仍不随时间变化。

`CoxPH` 在 NumPy、CuPy CUDA 与 Torch CUDA 后端实现比例风险回归，支持
Breslow、Efron 与 Exact 三种并列事件处理方式，同时覆盖普通右删失、延迟进入、
计数过程 `(start, stop]` 行、独立分层（`strata`）、时变协变量、稳健/聚类协方差，以及
通过 `CoxPHCV` 选择 L2 惩罚。

<!-- example: coxph-cpu-walkthrough -->
## 导入

```python
import numpy as np
from statgpu.survival import CoxPH
```

## 第一个 CPU 完整示例

下面各小节按顺序运行，只需 NumPy 与 StatGPU。先完成右删失数据的拟合和留出评估；
后面的生存曲线、交叉验证与 GPU 示例会复用这里的数据，不必重新生成。

### 准备输入

`X` 的每行对应一位受试者，每列对应一个特征，形状为 `(n_samples, n_features)`。
`time` 与 `event` 都是一维向量，长度等于 `X` 的行数。`time` 是事件或删失发生的时间；
`event=1` 表示观察到事件，`event=0` 表示右删失。不要给 `X` 添加常量/截距列。

下面模拟 400 位独立受试者和三个尺度相近的特征。实际使用时，可将此块替换为自己的数据读取步骤。

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(400, 3))
true_coef = np.array([0.8, -0.5, 0.3])
event_time = rng.exponential(scale=np.exp(-(X @ true_coef)))
censor_time = rng.exponential(scale=2.0, size=len(X))
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(np.int64)
```

### 留出评估数据

拟合前留出最后 100 位受试者。真实数据应按受试者、群组或时间选择合适的划分方式；
缩放等预处理只能使用训练部分来学习。

```python
X_train, X_test = X[:300], X[300:]
time_train, time_test = time[:300], time[300:]
event_train, event_test = event[:300], event[300:]
```

### 拟合并检查收敛

先使用显式 CPU 后端和 Efron 并列事件处理。保留 `compute_inference=True`，
这样后续可以读取系数不确定性与生存曲线所需的基线。它不会自动检验比例风险假设。

```python
model = CoxPH(
    ties="efron", device="cpu", compute_inference=True,
).fit(X_train, time_train, event_train)
if not model.converged_:
    raise RuntimeError(
        f"{model.optimization_stop_reason_}: "
        f"normalized KKT={model.final_kkt_normalized_}"
    )
```

收敛后，先看各特征的方向和风险比：

```python
print("Coefficients:", model.coef_)
print("Per-feature hazard ratios:", model.hazard_ratios_)
print("Convergence:", model.termination_reason_, model.n_iter_)
```

这个随机种子下，系数约为 `[0.852, -0.457, 0.312]`，对应的风险比约为
`[2.345, 0.633, 1.367]`。例如，其他特征保持不变时，第一个特征增加一个单位，估计的
瞬时风险约变为原来的 2.35 倍；第二个特征与更低的风险相关。这些是模拟数据中的关联，
不应作为实际干预建议。

### 预测相对风险

先比较两个留出样本的风险，而不把风险比误读为概率。

```python
log_risk = model.predict_risk_score(X_test[:2])
relative_hazard = model.predict(X_test[:2])
print("Log-risk:", log_risk)
print("Relative hazard:", relative_hazard)
```

`predict_risk_score(X)` 返回 `X @ coef_`；`predict(X)` 和
`predict_hazard_ratio(X)` 返回 `exp(X @ coef_)`，其参照是同一分层内协变量全为零的样本。
比较两种特征组合时，应对两者的对数风险之差取指数。`hazard_ratios_` 则是每个**特征**的风险比。

### 评估留出集排序

用全部 100 个留出样本及其时间、事件指示计算 C-index。

```python
held_out_cindex = model.score(X_test, time_test, event_test)
print("Held-out C-index:", held_out_cindex)
```
<!-- example-end: coxph-cpu-walkthrough -->

留出集 C-index 约为 `0.766`：在可比较的样本对中，模型倾向于给更早发生事件的样本更高风险。
它衡量排序区分能力，不是概率校准指标或 $R^2$。接近 `0.5` 表示中性排序，
`1.0` 表示可比较样本对上的完美排序，低于 `0.5` 提示排序可能相反。
没有可比较样本对时也返回 `0.5`；此时是评估证据不足，不能据此认定模型表现等同于随机。
不要把训练集一致性指数当作留出评估。

## 预测生存曲线

先运行[第一个 CPU 完整示例](#第一个-cpu-完整示例)，复用其 `model` 与 `X_test`。
若问题是某个时刻仍未发生事件的概率，就需要在相对风险之外加入拟合基线。

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-survival-prediction -->
```python
requested_times = np.array([0.0, 0.5, 1.0, 2.0])
curves, curve_times = model.predict_survival(
    X_test[:2], times=requested_times,
)
print("Survival shape and times:", curves.shape, curve_times)
```
<!-- example-end: coxph-survival-prediction -->

`predict_survival` 返回 **`(curves, times)` 元组**，并非单独一个矩阵：
`curves` 的形状为 `(n_new, n_times)`，`times` 的形状为 `(n_times,)`；
本例分别是 `(2, 4)` 与 `(4,)`。每行估计该样本在各时刻之后仍未发生事件的概率，
取值在 `[0, 1]` 内，并在递增时间网格上单调不增。`times=None` 使用拟合基线的事件时间网格
（有分层时取各层的并集）；显式传入时间时保留请求顺序。基线为阶梯函数，在最后一个事件时间
之后保持不变，因此延长预测网格不意味着获得了可靠的长期外推。
对于计数过程数据，每个预测行表示固定的协变量组合，不会自动沿未来协变量轨迹积分。

### 从相对风险到生存概率

对于第 $s$ 层内固定的协变量组合，

$$
H_s(t\mid x)=H_{0s}(t)\exp(x^\top\beta),\qquad
S_s(t\mid x)=\Pr(T>t\mid x,s)=\exp\{-H_s(t\mid x)\}.
$$

$H_{0s}(t)=\int_0^t h_{0s}(u)\,du$ 是累积基线风险。当前实现按不同事件时刻的
Breslow 增量估计它：

$$
\widehat H_{0s}(t)=\sum_{t_k\le t}
\frac{d_{sk}}{\sum_{j\in R_s(t_k)}\exp(x_j^\top\widehat\beta)}.
$$

$d_{sk}$ 为第 $s$ 层在 $t_k$ 的事件数；$R_s(t_k)$ 包含同层中满足
`start < t_k <= stop` 的记录。即使系数拟合使用 `ties="efron"` 或 `"exact"`，
基线仍使用这些增量。因此，仅有相对风险不能给出生存概率，还需要拟合基线。
[API 参考示例](../reference/survival-smoothing-api.md#系数推断与-cv-结果)重建返回的曲线，
并说明系数区间与风险比区间的区别。

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
| `subject_id` | 可选标签 `(n_samples,)`，标识同一受试者的重复记录，用于一致性指数、稳健协方差聚合及 CV 划分。 |
| `cluster` | 可选标签 `(n_samples,)`；`cov_type="cluster"` 时必须提供。仅提供 cluster 不会使 CV 自动按聚类分组；应使用合适的 `subject_id` 或显式 `cv_splits`。 |
| `init_coef` | `CoxPH` 可选的有限初始系数向量 `(n_features,)`；不是 `CoxPHCV.fit` 参数。 |
| `formula`, `data` | `CoxPH.fit` 的公式接口，见后文；`CoxPHCV.fit` 不接受这两个参数。 |

两者也支持省略 `event` 的 `fit(X, y)`：`y` 可以是列为 `[time, event]` 的
`(n_samples, 2)` 数组，或列为 `[start, stop, event]` 的 `(n_samples, 3)` 数组。
三列形式不能再单独提供 `entry`/`start`。`score` 也接受这些组合响应；
其区间参数叫 `start`，不是 `entry`。矩阵拟合前应先处理数值缺失，非有限数组会被拒绝，
不会静默删除对应行。

完整调用签名、返回形状、方法限制和推断输出见[面向使用者的 API 参考](../reference/survival-smoothing-api.md#coxph-与-coxphcv)。
构造参数也见[参数](#参数)与 [CoxPHCV 参数](#coxphcv-参数)。规范的方法签名与实现契约见
[`CoxPH`](../../../statgpu/survival/_cox.py) 和
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py)。

重要行为：

- 显式 `device="cuda"` 与 `device="torch"` 不会静默回退 CPU；
- `entry=` 与 `start=` 是互斥的别名；
- 某行在时刻 `t` 进入风险集，当且仅当 `start < t <= stop`，且其分层标签
  与事件所属分层相同；
- `subject_id=` 标识同一受试者的重复行，用于一致性指数、三明治协方差聚合，并确保同一受试者的记录不会被拆到不同的交叉验证折中；
- `compute_inference=False` 仅执行估计，推断字段和基线风险字段保持未设置。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`、`"efron"` 或 `"exact"` |
| `tol` | `1e-9` | Newton/KKT 收敛阈值 |
| `max_iter` | `100` | 最大迭代次数 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"` |
| `n_jobs` | `None` | 接受共享的 CPU 作业数设置；当前 Cox 拟合/CV 循环不通过它并行处理各折。 |
| `compute_inference` | `True` | 计算协方差、检验与基线风险 |
| `compute_cindex` | `True` | 计算训练集一致性指数 |
| `cov_type` | `"nonrobust"` | `"nonrobust"`、`"hc0"`、`"hc1"` 或 `"cluster"` |
| `penalty` | `0.0` | 非负 L2 惩罚 |
| `inference_mode` | `"strict"` | `"approx"` 仅为兼容别名；两种设置使用相同的稳健协方差计算方法，见[协方差与推断](#协方差与推断)。 |
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

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-cpu-cv -->
```python
from statgpu.survival import CoxPHCV

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

```

先看候选分数和有效折数，再在留出测试集评估最终模型：

```python
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
`best_score_` 是所选的**平均留出偏对数似然**，不是 `score()` 返回的 C-index，
只适合在相同数据、划分和评分约定下比较。`cv_results_["pl_path"]` 形状为
`(n_penalties, n_folds)`；`mean_pl` 与 `effective_fold_counts` 的形状均为 `(n_penalties,)`。
各折贡献偏对数似然总和，不除以行数或事件数。自定义网格中，数值上近似并列时优先较强惩罚，
因此所选分数可能略小于平均分数最大值。自定义网格按惩罚从强到弱评价，但结果保留输入顺序。
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
| `cv_splits` | `None` | 显式 `(train_indices, validation_indices)` 对，覆盖自动划分；每组索引应非空、一维、互不重叠，且为范围内的整数。一次性迭代器只读取一次并复用，参数检查或克隆也可能触发首次读取。 |
| `ties` | `"breslow"` | 候选及重拟合使用 `"breslow"`、`"efron"` 或 `"exact"`。 |
| `tol` | `1e-9` | 候选及重拟合的收敛容差。 |
| `max_iter` | `100` | 候选及重拟合的最大迭代次数。 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`。 |
| `n_jobs` | `None` | 共享 CPU 作业数选项，传给最终重拟合；当前 CV 循环不通过它并行化。 |
| `compute_inference` | `True` | 仅在最终重拟合计算推断和基线风险。 |
| `cov_type` | `"nonrobust"` | 最终重拟合的协方差约定。 |
| `inference_mode` | `"strict"` | `"approx"` 仅为兼容别名；两种设置使用相同的稳健协方差计算方法，见[协方差与推断](#协方差与推断)。 |
| `gpu_memory_cleanup` | `False` | 在公共计算边界尽力清理 GPU 缓存。 |
| `random_state` | `None` | 自动生成 CV 划分的随机种子。 |

与 `CoxPH` 不同，`CoxPHCV` 构造器没有 `penalty` 或 `compute_cindex`；
应使用 `penalties` 并显式调用 `score`。完整源代码参考见
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py)。


### L1/L2/ElasticNet/SCAD/MCP 模型族交叉验证

上面的 `CoxPHCV` 是标准的 L2 Cox 选择器，并可按配置执行最终重拟合推断。
公开的带惩罚模型族则使用 `PenalizedGLM_CV` 的独立生存分析专用分支：

以下补充示例复用[CPU 示例](#第一个-cpu-完整示例)的 `X_train`、`time_train` 与
`event_train`，测试样本不参与调参。

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-penalized-family-cv -->
```python
from statgpu.linear_model import PenalizedGLM_CV

survival_y = np.column_stack([time_train, event_train])
penalized_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="mcp",               # l1、l2、elasticnet、scad 或 mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # 也可用 "cuda" / "torch"
).fit(X_train, survival_y)
```
<!-- /example: coxph-penalized-family-cv -->

该分支始终保留二维 `(time, event)` 响应，禁止截距，使用留出数据上的未惩罚偏似然评分，并要求每个可评估的交叉验证折都得到有限的偏似然评分。若不存在满足
这一要求的 `alpha`，拟合会抛错，并且不会发布已选 `alpha` 或已拟合估计器。最终重拟合使用
`PenalizedCoxPHModel(compute_inference=False)`；不支持选择后系数推断、`two_stage`、样本权重或字典形式响应。无惩罚别名不可调，因此该 CV
路径会拒绝，需改为直接拟合模型。

两种 Cox API 的损失尺度不同。`PenalizedCoxPHModel` 最小化

$$
-\ell(\beta)/n + P_\alpha(\beta),
\qquad P_\alpha(\beta)=\frac{\alpha}{2}\lVert\beta\rVert_2^2
\quad\text{对于内置 L2 惩罚}.
$$

这里 `n` 是实际参与拟合的行数，包含删失观测；公式删除缺失行后按保留行计数，
并非事件数。在训练行、特征和并列事件
处理方法相同的情况下，内置 `penalty="l2", alpha=a` 的目标等价于
`CoxPH(penalty=n*a/2)`，不能直接交换 `alpha` 与 `penalty` 的数值。
带惩罚模型族默认 `alpha=1.0`，属于正则化拟合；`CoxPH` 默认 `penalty=0.0`。
若传入惩罚对象，则使用对象自身的惩罚强度。

带惩罚模型族的 CV 分支先计算每折的 `-ell_validation / n_validation`，
再最小化跨折均值；其 `best_score_` 是所选均值的相反数，即按验证行数
归一化后再跨折平均的留出对数偏似然。数值上近似并列时优先选择更大的 alpha。
这与 `CoxPHCV` 对各折总和尺度对数
偏似然取均值不同。此外，每个训练折和最终重拟合各有自己的行数，因此用
单个 `n` 换算直接拟合的惩罚值，不能让两种 CV 搜索等价。分析中应始终沿用
所选 API 的参数选择和最终重拟合规则。

自定义交叉验证折可以采用一般的非空、训练集与验证集互不重叠的划分，包括前向
`TimeSeriesSplit` 或重复留出；无需互为补集，也无需让每行恰好进入一次验证集。索引会在任何候选模型拟合前校验，必须是一维、精确且位于范围内的
整数。自动网格中，ElasticNet 在 `l1_ratio > 0` 时采用零模型 KKT 边界
`alpha_max = ||gradient L(0)||_inf / l1_ratio`，惩罚对象使用自身的混合比例参数。
纯 L2（`l1_ratio=0`）不存在有限的全零 KKT 阈值，因此把零模型得分的原始
无穷范数用于启发式地确定网格上限。

大规模 `device="auto"` 搜索只在 Torch 或 CuPy 的 CUDA 后端确认设备实际可用
后选择 GPU。无法评估的折仍记录在 `failure_path` 中。CuPy
可导入但无法运行时会回退 CPU；显式 `device="cuda"` 仍严格抛错，不会静默回退。

## 预测与评分

对数组输入，`predict`、`predict_risk_score`、`predict_hazard_ratio`、
`predict_survival` 与 `score` 都在拟合后端执行。使用 `device="auto"` 拟合后，模型会固定实际的
`effective_device_`；后续修改全局设备设置不会迁移既有模型的预测或评分后端。分层生存预测要求每个预测行
提供一个训练时已知的分层标签；即使拟合时只有一个显式分层，也不能省略
标签，缺失或未知标签会抛出 `ValueError`。生存曲线在对数域中累计基线风险，
以提高数值稳定性。使用公式接口拟合的模型会在预测前应用已保存的设计矩阵转换。

调用 `score()` 时，传入的 `strata` 必须具有 `(n_samples,)` 形状；
显式分层模型只接受训练时已知标签，多分层拟合在评分时必须提供标签。
标量、二维、长度错误或未知标签都会在后端计算一致性指数前统一抛出
`ValueError`。

若某个已拟合分层没有观察到任何事件，其空的基线风险状态是合法状态。
该分层在任意时间的累计基线风险均为零，因此 `predict_survival()` 精确返回
1。显式 `times`、自动 `times`、混合分层预测行和 `CoxPHCV` 预测均有这一行为。

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

系数推断的 `_bse`、`_zvalues`、`_pvalues` 是 `(p,)` NumPy 数组；`_conf_int`
是 `(p,2)` 的固定 **95% 逐系数、系数尺度**区间。`np.exp(model._conf_int)`
才是 `summary()` 展示的风险比区间。关闭推断时这些字段为 `None`。
`summary()` 直接打印并返回 `None`。

`log_likelihood`、`aic`、`bic`、`concordance_index` 是 CoxPH 属性。
正惩罚拟合后读取 AIC/BIC 会报错；无惩罚 BIC 使用事件数而不是行数。
CV 模型应通过 `estimator_` 读取这些属性，见[输出表与可运行验证](../reference/survival-smoothing-api.md#系数推断与-cv-结果)。

<a id="cpu-与-gpu-示例"></a>
## 可选：在 GPU 上拟合

先按顺序运行[第一个 CPU 完整示例](#第一个-cpu-完整示例)，再继续本节。
下面只将同一训练集和三个留出样本移到所选 GPU，不重新生成数据或重复 CPU 拟合。
分别选择 CuPy 或 Torch 示例运行；无需同时安装两个后端。

### CuPy / CUDA

将训练输入和预测输入转换为 CuPy 数组：

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-cupy-fit -->
```python
import cupy as cp

X_cp = cp.asarray(X_train)
time_cp = cp.asarray(time_train)
event_cp = cp.asarray(event_train)
X_test_cp = cp.asarray(X_test[:3])
```

拟合后预测同一留出集中的三个样本；结果仍为 CuPy 数组。

```python
cupy_model = CoxPH(
    ties="efron", device="cuda", compute_inference=False,
).fit(X_cp, time_cp, event_cp)
cupy_log_risk = cupy_model.predict_risk_score(X_test_cp)
```
<!-- example-end: coxph-cupy-fit -->

若需要选择 L2 惩罚，复用以上 CuPy 训练数组：

<!-- example-requires: coxph-cupy-fit -->
<!-- example: coxph-cupy-cv -->
```python
from statgpu.survival import CoxPHCV

cupy_cv = CoxPHCV(
    penalties=[0.0, 0.1, 1.0, 10.0], cv=3, random_state=42,
    ties="efron", device="cuda", compute_inference=False,
).fit(X_cp, time_cp, event_cp)
```
<!-- example-end: coxph-cupy-cv -->

### Torch / CUDA

也可以从 CPU 示例的 NumPy 数组开始，独立运行下面的 Torch 版本：

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-torch-fit -->
```python
import torch

X_t = torch.as_tensor(X_train, dtype=torch.float64, device="cuda")
time_t = torch.as_tensor(time_train, dtype=torch.float64, device="cuda")
event_t = torch.as_tensor(event_train, dtype=torch.float64, device="cuda")
X_test_t = torch.as_tensor(X_test[:3], dtype=torch.float64, device="cuda")
```

使用 `device="torch"` 拟合后，预测结果仍为 CUDA 张量。

```python
torch_model = CoxPH(
    ties="efron", device="torch", compute_inference=False,
).fit(X_t, time_t, event_t)
torch_log_risk = torch_model.predict_risk_score(X_test_t)
```
<!-- example-end: coxph-torch-fit -->

Torch 交叉验证同样复用上述训练数组：

<!-- example-requires: coxph-torch-fit -->
<!-- example: coxph-torch-cv -->
```python
from statgpu.survival import CoxPHCV

torch_cv = CoxPHCV(
    penalties=[0.0, 0.1, 1.0, 10.0], cv=3, random_state=42,
    ties="efron", device="torch", compute_inference=False,
).fit(X_t, time_t, event_t)
```
<!-- example-end: coxph-torch-cv -->

当相应软件包、CUDA 运行环境或设备不可用时，显式 CUDA 请求会报错，不会静默转到 CPU。
以上示例关闭了推断；需要协方差、检验或生存曲线时，设置 `compute_inference=True`。

## 目标函数与估计方程

对第 `i` 行的起始时间 `a_i`、终止时间 `b_i`、事件指示 `delta_i` 与分层
`s_i`，时刻 `t` 的风险集为

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

无并列失败时，分层 Cox 偏对数似然为

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
`CoxPH` 与 `CoxPHCV` 接受一维分层、受试者和聚类标签。
Exact 使用动态规划避免逐一枚举事件组合，但计算量和内存需求仍随数据规模、
并列事件组大小及特征维数增加。

临时工作区超出配置上限时，会在所选后端采用内存需求更低的算法，不会静默转到 CPU。
`STATGPU_EXACT_NESTED_MAX_BYTES`、`STATGPU_EXACT_BATCH_MAX_BYTES` 和
`STATGPU_COX_GROUP_MAX_BYTES` 分别控制对应工作区，默认均为 512 MiB；
它们不是整个拟合过程的总内存上限。详细配置、算法实现及适用条件见
[开发参考](../../../dev/references/coxph-implementation-and-evidence.md#implementation-snapshot)。

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
因此可以判断是否已达到最大迭代次数；达到迭代上限并不表示模型已经收敛。

## 惩罚强度缩放与推断

`penalty` 就是上述总和尺度的偏似然目标中的 `lambda`，不会除以样本数或
事件数；CoxPH 也没有需要惩罚的截距。相同的 `penalty` 数值不保证在不同数据规模下
具有相同的有效正则强度。跨数据集或样本规模比较时，应在目标抽样尺度下用 `CoxPHCV` 调参；复现采用平均损失的外部软件时，需要显式换算其 `penalty` 的尺度定义，不能假设数值直接相同。

正 L2 惩罚下，记 `J` 为拟合系数处未加惩罚的 Cox 观测信息，
`A = J + 2 * penalty * I_p`，则固定惩罚强度的频率学派代入式协方差为：

$$
\widehat{\operatorname{Var}}(\widehat\beta)=A^{-1}JA^{-1},
\qquad A=J+2\lambda I_p.
$$

而不是 `A^-1`；后者更接近惩罚曲率或 Laplace 近似下的量，不能直接作为频率学派
抽样协方差发布。带惩罚的稳健推断同样使用带惩罚的逆曲率矩阵作为两侧矩阵（bread），中间矩阵（meat）仍由未加惩罚的聚合得分外积构成。

因此，标准误、z 统计量、p 值、置信区间与带惩罚 Wald 检验都以给定 `penalty` 为条件，目标是带惩罚的估计方程；它们不是针对无惩罚系数的纠偏推断，也不校正系数收缩偏差，或交叉验证选择 `penalty` 带来的额外不确定性。`CoxPHCV` 从最终重拟合复制相同契约，
并明确报告 `penalty_selection_adjusted_=False`。沿用 `PenalizedGLM` 的结果命名，
正 `penalty` 拟合的 `inference_method_` 使用简洁的 `"m_estimation"`；两侧矩阵、中间矩阵、协方差定义、推断目标和条件化方式仍分别保留在推断结果的元数据中。

带惩罚拟合会关闭经典似然比检验、得分检验以及 AIC/BIC，不会把惩罚估计
当作无约束最大似然结果报告。该契约与 `PenalizedCoxPHModel` 分开；后者的
L1/Elastic Net/SCAD/MCP 接口仍仅支持估计。

## 协方差与推断

[CPU 示例](#第一个-cpu-完整示例)已启用推断；直接查看该模型的系数表，无需重新拟合。
表中的标准误、z 检验和区间使用默认的模型协方差，具体含义与其他协方差选择见下文。

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-summary -->
```python
model.summary()
```
<!-- example-end: coxph-summary -->

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 模型协方差；无惩罚时为信息矩阵的逆，有惩罚时为固定惩罚强度下的三明治协方差 |
| `"hc0"` | 基于得分的三明治协方差 |
| `"hc1"` | 基于得分的三明治协方差，并按独立单元数进行有限样本修正 |
| `"cluster"` | 聚类稳健协方差；在 `fit` 时传入 `cluster=` |

无惩罚拟合的 `nonrobust` 协方差仍是通常的观测信息逆；正 `penalty` 协方差遵循
上一节的专门契约。

Breslow 与 Efron 的严格稳健推断使用 statgpu 内部的精确计数过程得分残差，不依赖 `statsmodels`。同一受试者的重复行会先按 `subject_id` 汇总再
形成 HC0/HC1 的中间矩阵；聚类稳健协方差按 `cluster` 汇总。

稳健推断必须具有可识别的独立单元变异。按受试者或聚类单元汇总后，HC0 与
聚类稳健协方差至少需要两个独立单元；HC1 还要求
`n_units > n_features`，因为其有限单元修正严格为
`n_units / (n_units - n_features)`。违反这些条件会抛出 `RuntimeError`，
不会把非正自由度分母替换为任意有限值。实质性负协方差对角线或非正稳健边际
方差同样会令严格推断失败，而不会发布零标准误与误导性的显著性结果。

边际方差为正并不保证稳健协方差在完整参数空间有效。StatGPU 会先对协方差矩阵进行对称化，再根据特征值判断其正定性；
数值容差会随矩阵尺度调整：正定矩阵同时支持边际推断和联合 Wald；半正定
但秩亏的矩阵仍保留逐系数的稳健标准误/z 值/p 值/置信区间，同时设置
`wald_test_available_=False` 并记录 `wald_test_failure_reason_`，`summary()` 会显示 `Robust Wald test unavailable`，不会使用不稳定的逆矩阵或仅显示 `nan`。若存在实质性
负特征值，该矩阵已不是合法的协方差估计量；严格推断会抛出
`RuntimeError` 并清空本次拟合状态，而不会仅凭正对角线发布边际推断。即使逐系数与
Wald 推断使用稳健协方差，似然比检验与得分检验仍是经典的基于模型检验；`summary()` 会明确标注这一差异。

`inference_mode="strict"` 是默认值。为保持向后兼容，公开 API 仍接受
`inference_mode="approx"`，但统一拟合路径会把它作为仅用于兼容的别名，
继续计算精确的计数过程得分三明治协方差。因此成功拟合会报告
`inference_approximate_=False`，且不会记录近似回退原因。
这里的“精确”指计数过程得分残差的计算，不表示有限样本精确检验。
系数的 z 检验和置信区间仍采用大样本正态近似；`ties="exact"` 也不会改变这一点。

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

## 设备传输与交叉验证筛选

GPU 拟合的部分预处理仍需要主机内存。普通 GPU Breslow/Efron 拟合会把完整的
已排序 `time` 和 `event` 向量复制到主机端以构建事件组元数据；即使设计矩阵
留在 GPU，也会报告 `full_host_transfer_performed_=True`。对于 `CoxPHCV`，
该标记涵盖 CV 与最终重拟合两个阶段。分阶段传输标记，以及如何区分本次调用
的设备和复用选择结果的来源，见[设备与 CV 诊断参考](../reference/coxph-diagnostics.md)。

请求 `STATGPU_COXPHCV_TWO_STAGE` 或
`STATGPU_COXPHCV_SUCCESSIVE_HALVING` 时，当前会发出 `RuntimeWarning`，
并在 NumPy、CuPy 与 Torch 上对全部候选项执行一次完整精度的穷举评估。
这些控制目前不会减少候选集合，也不会启用近似筛选。`cv_results_` 报告
`staged_safety_strategy="single_pass_exhaustive"`；请求状态、实际状态和候选项
掩码的完整说明见[筛选控制指南](../guides/cox-cv-staged-safety.md)。

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

## 外部对照与实现参考

与其他软件比较时，应对齐设计矩阵、并列事件处理方式、惩罚尺度、协方差类型和收敛设置。
[外部对照与 GPU 验证记录](../../../dev/references/coxph-implementation-and-evidence.md#historical-external-validation)
列出了对应的源码和运行环境；这些历史结果不是对当前版本或所有数据、硬件的统一保证。

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
| 观测信息矩阵奇异 | 检查共线性、常量列、分离（separation）或饱和（saturation）以及事件支持；减少冗余特征，或使用有明确依据的 L2 惩罚。 |
| 风险比预测抛出 `FloatingPointError` | `exp(X @ coef_)` 超出有限 float64 范围。检查 `predict_risk_score()`、缩放特征并检查外推。 |
| `converged_` 为 `False` | 检查 `optimization_stop_reason_`、`final_kkt_inf_` 与 `final_kkt_normalized_`；单纯增加 `max_iter` 不能修复线搜索失败或病态设计。 |
| Exact 并列事件很慢或触发工作区内存限制 | Exact 使用动态规划避免逐一枚举组合，但计算量和内存需求仍随风险集、并列事件组及特征维数增加。应根据数据规模评估成本；研究设计允许时，可考虑 Breslow 或 Efron。 |
| `score()` 返回 `0.5` | 可能是风险排序没有区分度，也可能没有可用于一致性指数计算的样本对；后一种情况同样返回中性值 `0.5`。应检查数据是否包含可比较的样本对。 |

## 限制

- Exact 并列事件尚不支持稳健/聚类协方差；
- Exact 并列事件使用动态规划；大型风险集或并列事件组仍可能需要较多计算和内存，应结合特征维数评估资源需求；
- 尚未实现 frailty（共享脆弱性）/随机效应项；
- 可选 `torch.compile` 加速要求兼容 Triton 的硬件；拟合本身不要求开启此加速。

## 参考文献

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
- R survival 文档：[`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)。
