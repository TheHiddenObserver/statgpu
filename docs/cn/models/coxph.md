# CoxPH

> 语言：中文<br>
> 最后更新：2026-10-12<br>
> 页面定位：模型文档<br>
> 切换：[English](../../en/models/coxph.md)

## 概览

如果想研究**哪些因素与事件发生的早晚有关**，但部分样本在观察结束时仍未发生事件，
可以考虑 Cox 比例风险回归。例如，研究机器何时发生故障时，只分析已经故障的机器，
会丢掉其余机器仍在运行的信息；只记录“是否故障”，又无法区分运行一天和运行一年。
Cox 回归同时利用观察时长和事件是否发生这两类信息。

对于普通右删失数据，每行记录特征 `X`、正的观察时长 `time` 和事件指示 `event`：

- `event=1`：事件在 `time` 时刻发生；
- `event=0`：观察在 `time` 时刻结束，事件尚未发生。此时只知道真实事件时间晚于
  观察结束时间，不能将其记为零，也不能据此认定事件永远不会发生。

删失记录仍提供有效信息，应予保留。通常的统计解释要求：在给定模型中的协变量和研究设计后，
删失过程与事件过程独立。

模型将风险率写成基线风险率与协变量效应的乘积：

$$
h_s(t\mid x)=h_{0s}(t)\exp(x^\top\beta).
$$

其中，$h_{0s}$ 是第 $s$ 层的基线风险率，描述事件的瞬时发生率如何随时间变化；
$x$ 是协变量向量，$\beta$ 是各层共享的回归系数。风险率描述的是到当前时刻尚未发生事件的个体，
在接下来很短时间内发生事件的快慢。

在同一分层内，**比例风险（PH）假设**要求两个固定协变量组合之间的风险比不随时间变化。
其他特征不变时，第 `j` 个特征每增加一个单位，瞬时风险率就乘以 `exp(coef_[j])`。
这个风险比不是事件概率，也不是生存时间的倍数，不能直接解释为因果效应。
应结合研究设计判断 PH 假设是否合理；优化器收敛只说明数值拟合达到要求，并不验证统计假设。
使用时变协变量时，`x(t)` 可以变化，但拟合的回归系数仍不随时间变化。

`CoxPH` 支持 NumPy、CuPy CUDA 和 Torch CUDA 后端，以及 Breslow、Efron、Exact
三种并列事件处理方法。它可用于普通右删失、延迟进入、计数过程 `(start, stop]` 数据和
时变协变量；也支持分层（`strata`）、稳健/聚类协方差，以及用 `CoxPHCV` 选择 L2 惩罚强度。

初次使用可先完成下面的 CPU 示例，再阅读预测结果和参数说明。需要分层、并列事件处理或
统计推断时，可继续查看后面的原理与限制；完整方法清单见
[API 参考](../reference/survival-smoothing-api.md#coxph-与-coxphcv)。

<!-- example: coxph-cpu-walkthrough -->
## 导入

先导入 NumPy 和模型类，后续示例均从这里开始：

```python
import numpy as np
from statgpu.survival import CoxPH
```

## 第一个 CPU 完整示例

本节只需 NumPy 和 StatGPU。请按顺序运行各小节，完成数据准备、模型拟合和留出集评估。
后面的生存曲线、交叉验证与 GPU 示例都复用这里的数据。

### 准备输入

`X` 的每行对应一位受试者，每列对应一个特征，形状为 `(n_samples, n_features)`。
`time` 与 `event` 都是一维向量，长度与 `X` 的行数相同：前者记录事件发生或观察结束的时间，
后者用 `1` 表示已观察到事件、用 `0` 表示右删失。Cox 偏似然不能单独识别截距，
因此不要给 `X` 添加截距或常量列。

下面生成 400 位独立受试者的模拟数据，每人有三个尺度相近的特征。
分析实际数据时，将这一段替换为自己的数据读取步骤即可。

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

先留出最后 100 位受试者作为测试集，其余 300 位用于训练。实际分析应根据研究设计，
按受试者、群组或时间划分数据；特征缩放等预处理所需的参数也只能从训练集估计。

```python
X_train, X_test = X[:300], X[300:]
time_train, time_test = time[:300], time[300:]
event_train, event_test = event[:300], event[300:]
```

### 拟合并检查收敛

本例明确指定 `device="cpu"`，并使用 Efron 方法处理同一时刻发生的多个事件。
保留 `compute_inference=True`，以便后续读取标准误、置信区间和预测生存曲线所需的基线风险。
拟合完成后先检查 `converged_`；启用推断并不会自动检验比例风险假设。

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

确认收敛后，查看系数的正负方向及对应风险比：

```python
print("Coefficients:", model.coef_)
print("Per-feature hazard ratios:", model.hazard_ratios_)
print("Convergence:", model.termination_reason_, model.n_iter_)
```

使用上述随机种子时，系数约为 `[0.852, -0.457, 0.312]`，对应的风险比约为
`[2.345, 0.633, 1.367]`。例如，其他特征保持不变时，第一个特征增加一个单位，估计的
瞬时风险率约变为原来的 2.35 倍；第二个特征的系数为负、风险比小于 1，表示它与较低的
瞬时风险率相关。这些结果反映模拟数据中的关联，不应据此提出实际干预建议。

### 预测相对风险

先计算两个测试样本的对数风险和相对风险，看看模型如何比较不同特征组合。

```python
log_risk = model.predict_risk_score(X_test[:2])
relative_hazard = model.predict(X_test[:2])
print("Log-risk:", log_risk)
print("Relative hazard:", relative_hazard)
```

`predict_risk_score(X)` 返回 `X @ coef_`；`predict(X)` 和
`predict_hazard_ratio(X)` 返回 `exp(X @ coef_)`，其参照是同一分层内协变量全为零的样本。
比较两个样本时，对两者的对数风险之差取指数即可得到风险比。
注意，预测结果是一行样本对应一个值，而 `hazard_ratios_` 是一个**特征**对应一个风险比；
两者都不是事件发生概率。

### 评估留出集排序

用全部 100 个测试样本及其时间、事件指示计算一致性指数（C-index）：

```python
held_out_cindex = model.score(X_test, time_test, event_test)
print("Held-out C-index:", held_out_cindex)
```
<!-- example-end: coxph-cpu-walkthrough -->

本例的测试集 C-index 约为 `0.766`。在可以比较事件先后顺序的样本对中，模型通常会给
更早发生事件的样本更高的预测风险。C-index 衡量的是排序区分能力，不衡量概率校准，
也不是 $R^2$：接近 `0.5` 表示排序没有明显方向，`1.0` 表示所有可比较样本对的排序均正确，
低于 `0.5` 则提示排序可能相反。

没有可比较样本对时，`score()` 也返回 `0.5`。这表示评估信息不足，不能据此断言模型表现
等同于随机猜测。训练集上的 C-index 则只描述训练数据，不能代替留出集评估。

## 预测生存曲线

本节复用[第一个 CPU 完整示例](#第一个-cpu-完整示例)中的 `model` 和 `X_test`。
相对风险只能比较事件发生的快慢；要估计某个时刻之后仍未发生事件的概率，还需要拟合得到的基线风险。

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

`predict_survival` 返回 **`(curves, times)` 元组**，需要分别接收曲线和时间：

- `curves` 的形状为 `(n_new, n_times)`，本例为 `(2, 4)`。每行对应一个样本，
  每列是该样本在相应时刻之后仍未发生事件的估计概率。
- `times` 的形状为 `(n_times,)`，本例为 `(4,)`。显式传入时间时，返回顺序与请求顺序一致；
  `times=None` 则使用拟合基线中的事件时刻，有分层时取各层事件时刻的并集。

生存概率位于 `[0, 1]`，随时间递增而单调不增。估计基线是阶梯函数，在最后一个事件时刻
之后保持不变，因此单纯延长预测时间范围并不能得到可靠的长期外推。
对于计数过程数据，每个预测行仍表示一个固定的协变量组合；模型不会自动沿未来的协变量轨迹积分。

### 从相对风险到生存概率

对于第 $s$ 层内固定的协变量组合，先由相对风险得到累积风险，再转换为生存概率：

$$
H_s(t\mid x)=H_{0s}(t)\exp(x^\top\beta),\qquad
S_s(t\mid x)=\Pr(T>t\mid x,s)=\exp\{-H_s(t\mid x)\}.
$$

$H_{0s}(t)=\int_0^t h_{0s}(u)\,du$ 是累积基线风险，即从零时刻到 $t$ 的基线风险率积分。
模型在每个不同的事件时刻计算一个 Breslow 增量，再将这些增量相加：

$$
\widehat H_{0s}(t)=\sum_{t_k\le t}
\frac{d_{sk}}{\sum_{j\in R_s(t_k)}\exp(x_j^\top\widehat\beta)}.
$$

$d_{sk}$ 为第 $s$ 层在 $t_k$ 的事件数；$R_s(t_k)$ 包含同层中满足
`start < t_k <= stop` 的记录。即使系数拟合使用 `ties="efron"` 或 `"exact"`，
基线仍按上述 Breslow 增量计算。因此，系数决定相对风险，拟合基线则补充了得到生存概率所需的时间信息。
[API 参考示例](../reference/survival-smoothing-api.md#系数推断与-cv-结果)重建返回的曲线，
并说明系数区间与风险比区间的区别。

<a id="输入形状与-fit-api"></a>
## 输入格式与拟合接口

使用矩阵数据时，调用 `CoxPH(...).fit(X, time, event, ...)` 或
`CoxPHCV(...).fit(X, time, event, ...)`。两者都返回模型自身，拟合结果保存在相应属性中。

| 输入 | 形状与约定 |
|---|---|
| `X` | 有限实数矩阵 `(n_samples, n_features)`，不要添加截距/常量列；预测时保持特征顺序。`CoxPH` 也接受一维单特征输入；`CoxPHCV` 应使用矩阵。 |
| `time` | 有限正数向量 `(n_samples,)`，统一时间单位；表示事件或删失时间，也可表示区间终止时间。 |
| `event` | 仅含 `0` 或 `1` 的向量 `(n_samples,)`；拟合至少需要一个已观察事件。 |
| `entry` / `start` | 可选向量 `(n_samples,)`，满足 `0 <= start < time`；二者为互斥别名，不能同时提供。省略时从零时刻进入。 |
| `strata` | 可选标签向量 `(n_samples,)`；各层分别构建风险集、估计基线风险，但共享回归系数。 |
| `subject_id` | 可选标签向量 `(n_samples,)`，标识同一受试者的重复记录，用于计算一致性指数、汇总稳健协方差得分及划分交叉验证折。 |
| `cluster` | 可选标签向量 `(n_samples,)`；`cov_type="cluster"` 时必须提供。仅传入 `cluster` 不会让交叉验证自动按聚类分组；应使用合适的 `subject_id` 或显式指定 `cv_splits`。 |
| `init_coef` | `CoxPH` 可选的有限初始系数向量 `(n_features,)`；不是 `CoxPHCV.fit` 参数。 |
| `formula`, `data` | `CoxPH.fit` 的公式接口，见后文；`CoxPHCV.fit` 不接受这两个参数。 |

也可以将响应合并为一个数组，省略单独的 `event` 参数，调用 `fit(X, y)`：

- 两列形式：形状为 `(n_samples, 2)`，列顺序为 `[time, event]`。
- 三列形式：形状为 `(n_samples, 3)`，列顺序为 `[start, stop, event]`；
  此时已经指定起始时间，不能再传入 `entry` 或 `start`。

`score` 同样接受这两种响应数组，但单独传入区间起点时，参数名是 `start`，不是 `entry`。
矩阵接口不会自动删除缺失行：拟合前需先处理缺失值，包含非有限值的数值数组会被拒绝。

完整调用签名、返回形状、方法限制和推断输出见 [API 参考](../reference/survival-smoothing-api.md#coxph-与-coxphcv)。
构造参数汇总在[参数](#参数)与 [CoxPHCV 参数](#coxphcv-参数)中。需要核对实现时，可查看
[`CoxPH`](../../../statgpu/survival/_cox.py) 和
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py)。

使用这些接口时，还应注意：

- 显式 `device="cuda"` 与 `device="torch"` 不会静默回退 CPU；
- `entry=` 与 `start=` 是互斥的别名；
- 某行在时刻 `t` 进入风险集，当且仅当 `start < t <= stop`，且其分层标签
  与事件所属分层相同；
- `subject_id=` 标识同一受试者的重复行，用于一致性指数、三明治协方差聚合，并确保同一受试者的记录不会被拆到不同的交叉验证折中；
- `compute_inference=False` 仅估计系数，不计算推断结果和基线风险，因此无法预测生存曲线。

## 参数

下面列出 `CoxPH` 的全部构造参数。通常先根据数据选择 `ties` 和 `cov_type`，
再决定是否使用 L2 惩罚；`tol`、`max_iter` 用于控制数值求解，不能替代模型假设检查。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | 并列事件处理方法，可选 `"breslow"`、`"efron"` 或 `"exact"`，见[风险集与并列事件处理](#风险集与并列事件处理)。 |
| `tol` | `1e-9` | Newton 求解和 KKT 收敛判定的容差。 |
| `max_iter` | `100` | 最大迭代次数；达到上限不代表已经收敛。 |
| `device` | `"auto"` | 计算设备，可选 `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`；只有 `"auto"` 允许自动选择可用后端。 |
| `n_jobs` | `None` | 接受通用的 CPU 作业数设置，但当前 Cox 拟合和交叉验证不会用它并行处理各折。 |
| `compute_inference` | `True` | 是否计算协方差、统计检验和基线风险；预测生存曲线时需要启用。 |
| `compute_cindex` | `True` | 是否计算训练集一致性指数。 |
| `cov_type` | `"nonrobust"` | 协方差估计方法，可选 `"nonrobust"`、`"hc0"`、`"hc1"` 或 `"cluster"`。 |
| `penalty` | `0.0` | 非负 L2 惩罚强度；`0.0` 表示不加惩罚。尺度定义见[惩罚强度缩放与推断](#惩罚强度缩放与推断)。 |
| `inference_mode` | `"strict"` | `"approx"` 仅为兼容别名；两种设置使用相同的稳健协方差计算方法，见[协方差与推断](#协方差与推断)。 |
| `gpu_memory_cleanup` | `False` | 是否尝试清理 CuPy/Torch 中可释放的缓存。 |

## 交叉验证

`CoxPHCV` 在交叉验证中比较一组 L2 惩罚强度，选定 `penalty` 后，再用全部训练数据
重新拟合一个 `CoxPH`。候选模型和最终模型使用相同的并列事件处理方法、进入时间、
分层方式与后端设置；`inference_mode` 和 `compute_inference` 决定最终重拟合时如何计算推断。

传入 `subject_id` 后，自动划分会将同一受试者的全部记录放在同一折，避免训练集与验证集
共享同一人的信息。若显式提供的 `cv_splits` 使同一受试者同时出现在某折的训练集和验证集中，
该划分会被拒绝。

下面复用[第一个 CPU 完整示例](#第一个-cpu-完整示例)：只用 300 位训练受试者选择惩罚强度，
再用未参与调参的 100 人测试集评估最终模型。L2 惩罚会收缩系数，特征尺度因而会影响惩罚效果；
实际分析中的缩放等预处理应在每个训练折内单独拟合，不能使用验证集或测试集的信息。

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

拟合后，先查看各候选惩罚的得分和有效折数，再计算最终模型的测试集 C-index：

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

本例选中 `penalty_=1.0`，测试集 C-index 约为 `0.766`。交叉验证优化的指标与 C-index 不同，
因此不保证提高 C-index。解读结果时，应区分以下几项：

- `best_score_` 是所选候选的**平均留出偏对数似然**，而 `score()` 返回 C-index。
  前者只适合在相同数据、划分和评分方式下比较。
- `cv_results_["pl_path"]` 的形状为 `(n_penalties, n_folds)`，保存各候选在各折上的得分；
  `mean_pl` 和 `effective_fold_counts` 的形状均为 `(n_penalties,)`，分别保存平均得分和有效折数。
- 每折得分是偏对数似然的总和，不除以行数或事件数。自定义网格按惩罚从强到弱评估，
  返回结果仍保留输入顺序；得分在数值上近似并列时优先选择较强惩罚，因此所选得分可能略低于最大平均得分。

**拟合成功不一定表示所有折都参与了评估。** 每个有效折的训练集和验证集都必须至少有一个
已观察事件。缺少事件的折会被排除，且不会为此发出警告；只要剩余折满足条件，
选择和重拟合仍可能成功。因此，还需检查 `cv_results_` 中的以下字段：

- `fold_valid`：哪些折可以参与评估。
- `failure_path`：各折或候选失败的原因；缺少事件时记录 `fold_has_no_train_or_test_events`。
- `effective_fold_counts`：各候选实际获得有效结果的折数。
- `converged_path`：各候选在各折上是否收敛。

候选必须在同一组全部有效折上收敛、且得分有限，才有资格被选中。若没有有效折或合格候选，
拟合会抛出 `RuntimeError`，不返回选择结果；索引格式不正确或训练/验证索引重叠则抛出 `ValueError`。

`estimator_` 保存最终重拟合的 `CoxPH`，它使用所选 `penalty_` 和全部传入的训练行。
本例设置了 `compute_inference=False`，因此可以预测对数风险和风险比，但不能得到生存曲线或推断结果。
即使启用最终重拟合推断，也不会校正调参不确定性或收缩偏差，详见
[惩罚强度缩放与推断](#惩罚强度缩放与推断)。

### CoxPHCV 参数

下面列出 `CoxPHCV` 的全部构造参数。用 `penalties` 指定候选值，或保留 `None` 自动生成网格；
需要自行控制分组或时间顺序时，使用 `cv_splits`。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `penalties` | `None` | 非空、有限、非负的一维 L2 惩罚网格；`None` 自动生成。 |
| `n_penalties` | `100` | 自动生成网格时的候选数量。 |
| `penalty_min_ratio` | `1e-3` | 自动网格中最小惩罚与最大惩罚的比值，取值范围为 `(0, 1]`。 |
| `cv` | `5` | 自动生成的折数，至少为二。 |
| `cv_splits` | `None` | 显式提供 `(train_indices, validation_indices)` 对，代替自动划分。每组索引都应非空、一维且为范围内的整数，训练与验证索引不能重叠。接受生成器；重复拟合和克隆可复用其划分，见[迭代器使用说明](../reference/survival-smoothing-api.md#拟合与选择)。 |
| `ties` | `"breslow"` | 候选模型和最终模型的并列事件处理方法，可选 `"breslow"`、`"efron"` 或 `"exact"`。 |
| `tol` | `1e-9` | 候选拟合和最终重拟合的收敛容差。 |
| `max_iter` | `100` | 候选拟合和最终重拟合的最大迭代次数。 |
| `device` | `"auto"` | 计算设备，可选 `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`。 |
| `n_jobs` | `None` | 通用 CPU 作业数设置，会传给最终模型；当前交叉验证不会用它并行处理各折。 |
| `compute_inference` | `True` | 仅在最终重拟合计算推断和基线风险。 |
| `cov_type` | `"nonrobust"` | 最终重拟合使用的协方差估计方法。 |
| `inference_mode` | `"strict"` | `"approx"` 仅为兼容别名；两种设置使用相同的稳健协方差计算方法，见[协方差与推断](#协方差与推断)。 |
| `gpu_memory_cleanup` | `False` | 在相关计算步骤结束时尝试清理可释放的 GPU 缓存。 |
| `random_state` | `None` | 自动划分交叉验证折时使用的随机种子。 |

与 `CoxPH` 不同，`CoxPHCV` 构造器没有 `penalty` 或 `compute_cindex`；
指定候选强度应使用 `penalties`，计算一致性指数则显式调用 `score`。实现细节见
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py)。


<a id="l1l2elasticnetscadmcp-模型族交叉验证"></a>
### 使用其他惩罚：L1、L2、ElasticNet、SCAD 与 MCP

如果只需选择 L2 惩罚强度，并希望保留最终重拟合的推断功能，使用 `CoxPHCV`。
如果需要 L1、L2、ElasticNet、SCAD 或 MCP 惩罚下的交叉验证，且不需要系数推断，
可以使用 `PenalizedGLM_CV(loss="cox_ph", ...)`。这两个接口的惩罚参数和评分尺度不同，
不能直接互换数值。

以下示例选择 MCP 惩罚，复用 [CPU 示例](#第一个-cpu-完整示例)的 `X_train`、
`time_train` 和 `event_train`；测试样本仍不参与调参。

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

**输入与有效折。** `y` 必须是 `(n_samples, 2)` 数组，两列依次为 `[time, event]`，
Cox 模型不拟合截距。每折的训练集和验证集都必须至少包含一个事件；缺少事件的折会被排除，
不会为此发出警告，其余折仍可能完成选择和重拟合。拟合后应查看 `cv_results_`：

- `fold_valid` 和 `n_effective_folds` 分别表示有效折及其数量。
- `train_event_counts` 和 `validation_event_counts` 记录各折训练集、验证集中的事件数。
- `failure_path` 记录失败原因；缺少事件时为 `missing_training_event` 或 `missing_validation_event`。

候选 alpha 按留出数据上的未惩罚偏似然评分，每个有效折的验证分数都必须有限。
没有有效折或合格 alpha 时，拟合会抛出 `RuntimeError`，不提供已选 alpha 或已拟合估计器；
索引格式不正确或训练/验证索引重叠则抛出 `ValueError`。

**支持范围。** 最终重拟合使用 `PenalizedCoxPHModel(compute_inference=False)`。
通过 `loss_kwargs` 可选择 `"breslow"` 或 `"efron"`，不支持 Exact 并列事件。
该交叉验证接口不支持选择后系数推断、`two_stage`、样本权重或字典形式响应。
无惩罚别名没有可调的惩罚强度，也会被拒绝；此时应直接拟合模型。

**惩罚尺度。** `PenalizedCoxPHModel` 最小化的目标为

$$
-\ell(\beta)/n + P_\alpha(\beta),
\qquad P_\alpha(\beta)=\frac{\alpha}{2}\lVert\beta\rVert_2^2
\quad\text{对于内置 L2 惩罚}.
$$

这里的 `n` 是实际参与拟合的行数，包含删失记录；使用公式接口删除缺失行时，按保留行计数，
不是按事件数计数。在训练行、特征和并列事件处理方法相同的情况下，内置
`penalty="l2", alpha=a` 的目标等价于 `CoxPH(penalty=n*a/2)`。
因此，`alpha` 与 `penalty` 不能使用同一个数值来表示同等强度的惩罚。
带惩罚模型族默认 `alpha=1.0`，而 `CoxPH` 默认 `penalty=0.0`，两者的默认拟合也不等价。
如果传入的是惩罚对象，则使用对象自身的惩罚强度。

**交叉验证评分。** `PenalizedGLM_CV(loss="cox_ph", ...)` 先计算各折的
`-ell_validation / n_validation`，再选择跨折平均损失最小的 alpha；近似并列时优先选择较大的 alpha。
其 `best_score_` 是所选平均损失的相反数，也就是先按验证行数归一化、再跨折平均的留出偏对数似然。
相比之下，`CoxPHCV` 直接对各折的偏对数似然总和取均值。

此外，各训练折和最终重拟合使用的行数不同。因此，用单个 `n` 换算一次惩罚值，
并不能使这两个交叉验证搜索过程等价。分析时应始终沿用所选接口的调参和重拟合规则。

**自定义划分与自动网格。** 自定义折的训练集和验证集只需非空且互不重叠，
可以使用前向 `TimeSeriesSplit` 或重复留出；两者不必互为补集，也不要求每行恰好进入一次验证集。
所有索引都会在候选模型拟合前校验，必须是一维、精确的整数，且不能超出数据范围。

自动生成网格时，ElasticNet 在 `l1_ratio > 0` 下使用零系数模型的 KKT 边界
`alpha_max = ||gradient L(0)||_inf / l1_ratio`；若传入惩罚对象，则使用该对象的混合比例。
纯 L2（`l1_ratio=0`）没有能让所有系数恰好为零的有限惩罚阈值，
因此以零系数处得分的无穷范数作为网格上限的启发式取值。

**设备选择。** 大规模搜索设置 `device="auto"` 时，只有确认 Torch 或 CuPy 的 CUDA
设备实际可用后才会选择 GPU。CuPy 虽可导入但无法运行时，自动模式可回退到 CPU；
显式设置 `device="cuda"` 则会报错，不会静默回退。无法评估的折仍会在 `failure_path` 中留下记录。

## 预测与评分

`CoxPH` 和 `CoxPHCV` 的 `predict`、`predict_risk_score`、`predict_hazard_ratio`、
`predict_survival` 与 `score` 对数组输入都使用拟合时的后端。即使设置了 `device="auto"`，
拟合后也会固定实际设备 `effective_device_`；之后修改全局设备设置，不会自动迁移已有模型。
通过公式接口拟合时，预测前会按拟合时保存的规则构建设计矩阵。

分层模型需要区分生存曲线预测和评分的标签要求：

- **生存曲线预测**：只要拟合时显式传入了 `strata`，每个预测行就必须提供一个训练时已知的标签，
  即使训练数据只有一个分层也不能省略。缺少标签或出现未知标签会抛出 `ValueError`。
- **评分**：多分层模型必须提供标签，单分层模型可以省略。一旦传入 `strata`，
  其形状必须为 `(n_samples,)`，显式分层模型还要求标签属于训练时的已知集合。
  标量、二维数组、长度不匹配或未知标签都会在计算一致性指数前抛出 `ValueError`。

生存曲线的基线风险在对数域中累计，以提高数值稳定性。若某个训练分层没有任何已观察事件，
该层的累计基线风险为零，`predict_survival()` 会返回全为 1 的生存概率。
这是可以出现的拟合结果，不表示该人群在现实中永远不会发生事件。
无论显式指定时间、使用自动时间网格、混合多个分层的预测行，还是通过 `CoxPHCV` 预测，
这一行为都相同。

风险比计算还有数值范围限制：`CoxPH`、`CoxPHCV` 和带惩罚 Cox 模型的风险比预测都按
float64 的指数范围检查；`CoxPH` 与 `CoxPHCV` 拟合后的 `hazard_ratios_` 也使用相同检查。
如果取指数会溢出为无穷或下溢为零，`CoxPH`/`CoxPHCV` 拟合时会抛出 `CoxFitNumericalError`，
上述三类模型预测时会抛出 `FloatingPointError`，不会悄悄截断到某个阈值。`predict_risk_score()` 不取指数，
可用于查看这些极端但仍有限的对数风险；`PenalizedCoxPHModel` 也提供此方法。

<a id="输出"></a>
## 拟合结果与诊断信息

拟合后，可以按分析目的读取以下结果：

- **系数与风险比**：`coef_` 保存系数，`hazard_ratios_` 保存每个特征对应的风险比。
- **系数推断**：启用推断时，`_bse`、`_zvalues`、`_pvalues`、`_conf_int`
  分别保存标准误、z 统计量、p 值和置信区间。
- **拟合与排序指标**：在相应条件下可读取 `log_likelihood`、`aic`、`bic`、`concordance_index`。
- **收敛诊断**：`converged_`、`termination_reason_`、`optimization_stop_reason_`、`n_iter_`、
  `final_kkt_inf_`、`final_kkt_normalized_`，详见[优化与收敛](#优化与收敛)。
- **推断方法与执行信息**：`inference_method_`、`inference_backend_`、
  `inference_approximate_`、`inference_fallback_reason_`、
  `inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_`、
  `full_host_transfer_performed_`，用于核对推断目标、近似状态及数据传输情况。

其中，`p` 表示特征数。`_bse`、`_zvalues` 和 `_pvalues` 是形状为 `(p,)` 的 NumPy 数组；
`_conf_int` 的形状为 `(p, 2)`，给出固定的 **95% 逐系数置信区间，区间位于系数尺度**。
对它取指数，即 `np.exp(model._conf_int)`，才得到 `summary()` 中展示的风险比区间。
这些是分别针对每个系数的区间，不是同时覆盖所有系数的联合区间。关闭推断时，上述字段均为 `None`。
`summary()` 会直接打印结果，返回值为 `None`。

`log_likelihood`、`aic`、`bic`、`concordance_index` 是 CoxPH 属性。
正惩罚拟合后不能读取经典 AIC/BIC，尝试读取会报错；无惩罚拟合的 BIC 使用事件数，而不是数据行数。
交叉验证模型可通过 `estimator_` 读取这些属性，见
[结果说明与示例](../reference/survival-smoothing-api.md#系数推断与-cv-结果)。

需要检查设备和数据传输时，`CoxPHCV` 还提供 `cv_full_host_transfer_performed_`、
`final_refit_full_host_transfer_performed_` 和 `orchestration_device_`，以区分交叉验证选择与最终重拟合。
`cv_results_` 中的设备信息也分为两类：

- **生成选择结果时的信息**：`scoring_device`、`selection_origin_device`、
  `candidate_preparation_origin_device`，以及总准备次数。
- **本次调用的信息**：`selection_cache_hit`、`requested_fit_device`、`effective_device`，
  以及以 `*_this_call` 结尾的计数。

复用缓存中的选择结果时，本次调用不会重新准备各折，也不会再次传输响应向量；
原来生成这些结果的设备信息仍会保留。字段详情见[设备与 CV 诊断参考](../reference/coxph-diagnostics.md)。

如果输入有限，但拟合得到的系数或似然不是有限值，`CoxPH` 会抛出
`CoxFitNumericalError`，它是 `FloatingPointError` 的子类。`CoxPHCV` 会排除这类候选；
输入错误、内存分配错误、CUDA 错误及其他非预期运行时错误则会原样抛出，不会被当作普通候选失败忽略。

<a id="cpu-与-gpu-示例"></a>
## 可选：在 GPU 上拟合

完成[第一个 CPU 完整示例](#第一个-cpu-完整示例)后，可将同一训练集和三个测试样本移到 GPU，
比较自己所需的计算后端。下面的 CuPy 和 Torch 示例二选一即可，无需同时安装两者，
也无需重新生成数据或重复 CPU 拟合。

### CuPy / CUDA

安装兼容的 CuPy/CUDA 环境后，先将训练数据和预测数据转换为 CuPy 数组：

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-cupy-fit -->
```python
import cupy as cp

X_cp = cp.asarray(X_train)
time_cp = cp.asarray(time_train)
event_cp = cp.asarray(event_train)
X_test_cp = cp.asarray(X_test[:3])
```

指定 `device="cuda"` 拟合，再预测这三个测试样本的对数风险；返回结果仍是 CuPy 数组。

```python
cupy_model = CoxPH(
    ties="efron", device="cuda", compute_inference=False,
).fit(X_cp, time_cp, event_cp)
cupy_log_risk = cupy_model.predict_risk_score(X_test_cp)
```
<!-- example-end: coxph-cupy-fit -->

如需通过交叉验证选择 L2 惩罚强度，可直接复用上述 CuPy 训练数组：

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

使用 Torch CUDA 时，同样从 CPU 示例中的 NumPy 数组开始，并显式创建 float64 CUDA 张量：

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-torch-fit -->
```python
import torch

X_t = torch.as_tensor(X_train, dtype=torch.float64, device="cuda")
time_t = torch.as_tensor(time_train, dtype=torch.float64, device="cuda")
event_t = torch.as_tensor(event_train, dtype=torch.float64, device="cuda")
X_test_t = torch.as_tensor(X_test[:3], dtype=torch.float64, device="cuda")
```

指定 `device="torch"` 拟合并预测，返回结果仍是 CUDA 张量。

```python
torch_model = CoxPH(
    ties="efron", device="torch", compute_inference=False,
).fit(X_t, time_t, event_t)
torch_log_risk = torch_model.predict_risk_score(X_test_t)
```
<!-- example-end: coxph-torch-fit -->

Torch 版本的交叉验证也可直接复用这些训练张量：

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

如果所需软件包、CUDA 运行环境或设备不可用，显式指定 CUDA 后端会报错，不会静默转到 CPU。
以上示例只估计系数；需要协方差、统计检验或生存曲线时，将 `compute_inference` 设为 `True`。

## 目标函数与估计方程

Cox 模型用偏似然估计协变量效应，不必预先指定基线风险率的函数形式。
其基本思路是：每当事件发生，将发生事件的记录与当时仍处于风险集中的记录比较。

记第 `i` 行的起始时间为 `a_i`、终止时间为 `b_i`、事件指示为 `delta_i`、分层标签为 `s_i`。
时刻 `t` 在第 `s` 层中的风险集为

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

风险集包括已经进入观察、但尚未在该时刻之前退出的同层记录。没有并列事件时，
分层 Cox 偏对数似然为

$$
\ell(\beta)=\sum_s\sum_{i:\delta_i=1,\ s_i=s}
\left[x_i^\top\beta-
\log\left\{\sum_{j\in R_s(b_i)}\exp(x_j^\top\beta)\right\}\right].
$$

每一项用事件记录的线性预测值，减去当时风险集中所有记录的相对风险总和的对数。
删失记录虽然没有单独的事件项，但在退出观察之前仍可进入风险集，因此会影响估计。
Breslow、Efron 与 Exact 对并列事件采用不同的分母计算方式，风险集均遵循 `(start, stop]` 约定。

令 `penalty=lambda`，`CoxPH` 最大化以下总和尺度的带惩罚目标，不对偏对数似然取平均：

$$
Q_\lambda(\beta)=\ell(\beta)-\lambda\lVert\beta\rVert_2^2.
$$

记 `U(beta)` 为未加惩罚的偏似然得分，即偏对数似然对系数的梯度。
带 L2 惩罚的估计方程为

$$
U_\lambda(\beta)=U(\beta)-2\lambda\beta=0.
$$

若 $J(\beta)=-\partial U(\beta)/\partial\beta$ 是未加惩罚的观测信息矩阵，
则 Newton 更新所用的信息矩阵为 $A(\beta)=J(\beta)+2\lambda I_p$。
这里 $I_p$ 是 $p$ 维单位矩阵，$p$ 为特征数；L2 惩罚在各系数方向上增加相同的曲率。

## 风险集与并列事件处理

同一时刻观察到多个事件时，需要指定并列事件处理方法。`ties="breslow"` 和
`ties="efron"` 分别使用 Breslow 与 Efron 偏似然；`ties="exact"` 则通过基本对称多项式
动态规划计算 Exact 分母，避免逐一枚举可能的事件组合。Exact 的计算和内存开销仍会随
数据规模、同一时刻的事件数量及特征数增加，选择时应同时考虑统计设定和计算成本。

无论是否使用延迟进入、分层、L2 惩罚，三个后端都按相同的 `(start, stop]` 约定构建风险集：
记录在起点 `start` 尚未进入风险集，在终点 `stop` 仍属于风险集；分层之间分别计算。
GPU 稳健推断也使用这一约定。`CoxPH` 与 `CoxPHCV` 的分层、受试者和聚类标签均应是一维向量。

临时工作区超过配置上限时，会在所选后端改用内存需求更低的算法，不会静默转到 CPU。
`STATGPU_EXACT_NESTED_MAX_BYTES`、`STATGPU_EXACT_BATCH_MAX_BYTES` 和
`STATGPU_COX_GROUP_MAX_BYTES` 分别控制对应工作区，默认均为 512 MiB；
这些设置限制的是各自的临时工作区，不是整个拟合过程的总内存。具体配置和算法说明见
[开发参考](../../../dev/references/coxph-implementation-and-evidence.md#implementation-snapshot)。

## 公式接口

如果数据保存在 pandas DataFrame 中，可以用公式描述响应和特征。
需先安装可选的 pandas/Patsy 公式依赖，并准备包含相应列的 `df`。
下面展示两种写法：`Surv(time, event)` 表示普通右删失，`Surv(start, stop, event)`
表示区间记录；代码用于说明接口，不能在没有 `df` 的情况下独立运行。

```python
CoxPH().fit(formula="Surv(time, event) ~ age + C(group)", data=df)
CoxPH().fit(
    formula="Surv(start, stop, event) ~ age + treatment",
    data=df,
    strata=df["clinic"],
    subject_id=df["patient_id"],
)
```

`C(group)` 将 `group` 作为分类变量处理。公式接口删除缺失行时，会同时对齐
`entry`/`start`、`cluster`、`strata` 和 `subject_id`，确保这些信息仍对应保留的记录。
三列形式 `Surv(start, stop, event)` 已包含起始时间，不能再额外传入 `entry=` 或 `start=`。

## 优化与收敛

模型使用带线搜索的 Newton 迭代，并在最终系数处检查 KKT 条件，判断估计方程是否已充分满足。
线搜索失败时不会继续更新系数，也不会报告收敛。建议先检查 `converged_`，
再结合以下字段判断是否需要调整数据或求解设置：

| 字段 | 如何解读 |
|---|---|
| `converged_` | 最终参数是否通过收敛检查。 |
| `termination_reason_` | 汇总后的停止状态：`kkt_converged`、`line_search_failed` 或 `stalled_with_large_kkt`。 |
| `optimization_stop_reason_` | 求解器原始停止原因，包括 `max_iter`；警告信息也会报告此值。 |
| `n_iter_` | 实际迭代次数。 |
| `final_kkt_inf_` | 最终 KKT 残差的无穷范数，反映估计方程尚未满足的程度。 |
| `final_kkt_normalized_` | 归一化后的最终 KKT 残差，用于核对收敛判定。 |

似然、梯度、Hessian 矩阵、协方差、基线风险和收敛状态都根据最终系数重新计算。
达到 `max_iter` 只表示迭代次数已用尽，不表示模型已经收敛。
若残差仍较大，应先检查特征尺度、共线性和事件信息是否充足，再决定是否增加迭代次数。

## 惩罚强度缩放与推断

`penalty` 对应目标函数中的 `lambda`。`CoxPH` 使用偏对数似然总和，不按样本数或事件数
归一化，也没有需要单独处理的截距。因此，同一个 `penalty` 数值用于不同规模的数据时，
不一定产生同等程度的收缩。跨数据集或样本规模分析时，应按实际使用的数据规模通过 `CoxPHCV`
选择惩罚强度；复现采用平均损失的软件结果时，则需先换算目标函数和惩罚的尺度。

使用正 L2 惩罚时，记 `J` 为最终系数处未加惩罚的 Cox 观测信息矩阵，
`A = J + 2 * penalty * I_p`。将惩罚强度视为固定值时，模型采用以下频率学派协方差估计：

$$
\widehat{\operatorname{Var}}(\widehat\beta)=A^{-1}JA^{-1},
\qquad A=J+2\lambda I_p.
$$

这里不能直接使用 `A^-1` 代替整个表达式：`A^-1` 描述惩罚后的逆曲率，常见于 Laplace
近似等计算，但不能直接当作这里的频率学派抽样协方差。带惩罚的稳健推断也以带惩罚的
逆曲率矩阵为两侧矩阵（bread），中间矩阵（meat）则由未加惩罚的汇总得分外积构成。

解读这类结果时，最重要的是明确其推断范围：标准误、z 统计量、p 值、置信区间和带惩罚
Wald 检验都**以给定的 `penalty` 为条件，针对带惩罚估计方程进行推断**。
它们没有将系数纠偏为无惩罚估计，也没有校正收缩偏差，或通过交叉验证选择 `penalty`
所带来的额外不确定性。

`CoxPHCV` 最终重拟合的推断也按这一方式解释，`penalty_selection_adjusted_=False`
表示尚未校正选择不确定性。正惩罚拟合报告 `inference_method_="m_estimation"`，
表示估计方程推断；两侧矩阵、中间矩阵、协方差定义、推断目标和条件化方式另有元数据记录。

带惩罚拟合不提供经典似然比检验、得分检验和 AIC/BIC，因为此时不能将拟合结果视为
无惩罚的最大似然估计。也不要把这里的推断支持套用到 `PenalizedCoxPHModel`：
后者的 L1/ElasticNet/SCAD/MCP 接口仍只支持估计。

## 协方差与推断

[CPU 示例](#第一个-cpu-完整示例)已经启用推断，可直接打印系数表，无需重新拟合。
本例的标准误、z 检验和置信区间使用默认的模型协方差；若使用其他 `cov_type`，
系数不确定性的估计方式也会相应改变。

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-summary -->
```python
model.summary()
```
<!-- example-end: coxph-summary -->

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 基于模型的协方差。无惩罚时为观测信息矩阵的逆；有惩罚时使用固定惩罚强度下的三明治协方差。 |
| `"hc0"` | 基于得分残差的三明治协方差。 |
| `"hc1"` | 在 HC0 基础上，按独立单元数进行有限样本修正。 |
| `"cluster"` | 聚类稳健协方差；需要在 `fit` 时传入 `cluster=`，先按聚类汇总得分。 |

无惩罚拟合的 `nonrobust` 使用通常的观测信息矩阵逆；正惩罚拟合则使用上一节介绍的
固定惩罚强度协方差，不能把两者混为一谈。

**重复记录与独立单元。** Breslow 和 Efron 的稳健推断由 StatGPU 直接计算精确的计数过程
得分残差，不依赖 `statsmodels`。计算 HC0/HC1 时，同一受试者的重复行先按 `subject_id`
汇总，再形成三明治协方差的中间矩阵；聚类稳健协方差则按 `cluster` 汇总。
因此，独立单元数取决于受试者或聚类的数量，不能简单用数据行数代替。

稳健推断需要足够的独立单元来估计单元间变异。汇总后，HC0 和聚类稳健协方差至少需要
两个独立单元；HC1 还要求 `n_units > n_features`，其修正系数为
`n_units / (n_units - n_features)`。不满足这些条件时会抛出 `RuntimeError`，
不会人为替换非正的自由度分母。若协方差对角线存在明显负值，或稳健边际方差不为正，
严格推断同样会失败，不会返回零标准误或误导性的显著性结果。

**逐系数推断与联合检验。** 各系数的方差为正，并不保证整个协方差矩阵有效。
StatGPU 会先将矩阵对称化，再用随矩阵尺度调整的数值容差检查特征值：

- 正定矩阵支持逐系数推断和联合 Wald 检验。
- 半正定但秩亏的矩阵，在边际方差为正时仍可提供各系数的稳健标准误、z 值、p 值和置信区间，
  但不提供联合 Wald 检验。此时 `wald_test_available_=False`，原因记录在
  `wald_test_failure_reason_` 中；`summary()` 显示 `Robust Wald test unavailable`，
  不会求不稳定的逆矩阵或只给出一个没有解释的 `nan`。
- 存在明显负特征值时，矩阵不是有效的协方差估计。严格推断会抛出 `RuntimeError`，
  并清空本次拟合结果；即使对角线为正，也不会继续提供边际推断。

即使系数和 Wald 检验使用稳健协方差，似然比检验与得分检验仍是经典的基于模型的检验。
`summary()` 会标明这一差异；正惩罚拟合则不提供后两种检验。

**“精确”与“近似”的含义。** 默认值为 `inference_mode="strict"`。
`inference_mode="approx"` 仍可使用，但只是为兼容旧调用保留的别名，
不会切换为另一种近似稳健算法，仍计算精确的计数过程得分三明治协方差。
成功完成推断时会报告 `inference_approximate_=False`，也不会记录近似回退原因。

这里的“精确”仅指得分残差的计算，不表示有限样本精确检验。系数的 z 检验和置信区间仍依赖
大样本正态近似；`ties="exact"` 描述的是并列事件处理方法，也不会改变这一推断前提。

**Exact 的推断限制。** `ties="exact"` 目前仅支持 `cov_type="nonrobust"`。
在启用推断时请求 HC0、HC1 或聚类协方差，会抛出 `NotImplementedError`。
如果 `compute_inference=False`，可以保留稳健 `cov_type` 设置，但不会实际计算协方差。

如需核对一次拟合实际采用了什么推断，可查看：

- 方法与后端：`inference_method_`、`inference_backend_`。
- 是否近似及其原因：`inference_approximate_`、`inference_fallback_reason_`。
- 推断目标与惩罚条件：`inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_`。
- Wald 检验是否可用：`wald_test_available_`、`wald_test_failure_reason_`。
- 是否将完整数据向量传至主机：`full_host_transfer_performed_`。

## 设备传输与交叉验证筛选

使用 GPU 不代表所有预处理都在 GPU 上完成。普通 Breslow/Efron GPU 拟合会将排序后的
完整 `time`、`event` 向量复制到主机端，用于整理事件分组信息。因此，即使设计矩阵仍在 GPU 上，
`full_host_transfer_performed_` 也会为 `True`。
对 `CoxPHCV` 而言，这个总标记涵盖交叉验证与最终重拟合两个阶段。若需分开检查，
或区分本次调用设备与缓存结果的原始设备，请参阅[设备与 CV 诊断参考](../reference/coxph-diagnostics.md)。

目前，启用 `STATGPU_COXPHCV_TWO_STAGE` 或 `STATGPU_COXPHCV_SUCCESSIVE_HALVING`
会触发 `RuntimeWarning`，随后仍以完整精度对所有候选项进行一次穷举评估。
NumPy、CuPy 和 Torch 都采用这一行为：这两个设置不会减少候选数，也不会启用近似筛选。
对应的 `cv_results_` 记录 `staged_safety_strategy="single_pass_exhaustive"`。
请求模式、实际模式及候选掩码的完整说明见[筛选控制指南](../guides/cox-cv-staged-safety.md)。

<a id="支持矩阵"></a>
## 功能支持范围

下表汇总 `CoxPH` 和 `CoxPHCV` 的主要能力。CuPy 和 Torch 列均指 CUDA 后端；
并列事件方法与后端需要结合阅读，例如三个后端都支持稳健推断，但 Exact 并列事件不支持稳健协方差。

| 能力 | Breslow | Efron | Exact | NumPy | CuPy | Torch |
|---|---|---|---|---|---|---|
| 普通右删失 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 延迟进入 / `(start, stop]` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 分层 `strata`，各层单独估计基线 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 非负 L2 `penalty` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| `nonrobust` 推断 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| HC0 / HC1 / `cluster` 推断 | 支持 | 支持 | 未实现 | 支持 | 支持 | 支持 |
| 预测结果保留后端数组类型 | 支持 | 支持 | 支持 | NumPy | CuPy | Torch |

预测生存曲线需要拟合基线风险，因此使用 `predict_survival` 前，应在拟合时保留
`compute_inference=True`。只预测对数风险或风险比时不需要基线，可以关闭推断。

## 外部对照与实现参考

与其他软件对照结果前，应先确保设计矩阵、并列事件处理方法、惩罚尺度、协方差类型和收敛设置一致。
仅使用相同的参数数值并不一定代表相同的模型，尤其要核对损失是否按样本数归一化。
[外部对照与 GPU 验证记录](../../../dev/references/coxph-implementation-and-evidence.md#historical-external-validation)
注明了验证时所用的源码和运行环境，可供复现时参考；它们不能替代对当前数据与运行环境的检查。

<a id="faq-与常见失败模式"></a>
## 常见问题与排查方法

遇到报错或异常结果时，先确认输入、模型设置和以下条件，再决定是否重新拟合：

| 现象 | 含义与处理 |
|---|---|
| 显式 `device="cuda"` 或 `device="torch"` 提示后端不可用 | 检查对应软件包、CUDA 运行环境和设备，安装兼容后端，或明确改用 `device="cpu"`；StatGPU 不会静默回退。 |
| `predict_survival()` 提示基线风险不可用 | 使用 `compute_inference=True` 重新拟合；风险得分与风险比预测不需要基线风险。 |
| 分层生存预测拒绝标签 | 只要拟合时显式分层，每个预测行都必须提供一个训练时已知的分层标签，形状为 `(n_samples,)`；训练时只有一个分层也不能省略。 |
| 分层评分拒绝标签 | 多分层拟合必须逐行提供已知标签；单分层拟合可省略标签，但一旦提供，仍必须具有 `(n_samples,)` 形状且属于训练标签。 |
| 已知分层的生存率恒为 1 | 检查该分层是否没有已观察事件。此时累计基线风险为零，属于可出现的拟合结果，并非基线数据缺失，也不能解释为真实风险为零。 |
| `HC1 covariance requires n_units > n_features` | 独立单元数不足以计算 HC1 修正。可增加独立受试者/聚类数量、减少特征，或选择研究设计能够支持的协方差估计方法。 |
| 稳健协方差要求至少两个独立单元 | 只有一个受试者或聚类时，无法估计单元间变异。若暂不需要推断，可设置 `compute_inference=False` 仅估计系数。 |
| 观测信息矩阵奇异 | 检查共线性、常量列、分离（separation）、模型饱和（saturation）以及事件信息是否充足。可减少冗余特征，或在有明确依据时使用 L2 惩罚。 |
| 风险比预测抛出 `FloatingPointError` | `exp(X @ coef_)` 溢出或下溢。先用 `predict_risk_score()` 查看对数风险，再检查特征尺度及是否存在过度外推。 |
| `converged_` 为 `False` | 查看 `optimization_stop_reason_`、`final_kkt_inf_` 和 `final_kkt_normalized_`；仅增加 `max_iter` 不能解决线搜索失败或严重共线性等问题。 |
| Exact 并列事件计算很慢或触发工作区内存限制 | 动态规划避免了逐一枚举，但开销仍随风险集、并列事件组及特征数增长。先评估资源需求；统计设定允许时，可考虑 Breslow 或 Efron。 |
| `score()` 返回 `0.5` | 可能是排序缺少区分度，也可能根本没有可比较的样本对。先检查事件和观察时间能否形成可比较样本对，再解释该指标。 |

## 限制

选择模型前，还需确认以下限制是否影响分析：

- Exact 并列事件处理尚不支持稳健或聚类协方差。
- Exact 虽然使用动态规划，大型风险集或并列事件组仍可能消耗较多时间与内存，需结合特征数评估资源需求。
- 尚未实现 frailty（共享脆弱性）或随机效应项，不能将分层功能当作这类模型使用。
- 可选的 `torch.compile` 加速需要兼容 Triton 的硬件；普通拟合不要求开启这一加速。

## 参考文献

Cox 模型、并列事件处理和稳健推断的统计背景可参阅以下文献；R 的 `coxph` 文档也可作为对照参考。

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
- R survival 文档：[`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)。
