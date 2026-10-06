# 生存分析与平滑方法 API 参考

> 语言：中文  
> 最后更新：2026-10-05  
> 切换：[English](../../en/reference/survival-smoothing-api.md)

本页用于在阅读 [CoxPH](../models/coxph.md)、[GAM](../models/semiparametric.md) 或[非参数方法](../models/nonparametric.md)示例后查找具体调用。签名列出公开参数与默认值；`*` 后的参数须按名称传入。形状记号：`n` 为训练行数，`p` 为特征数，`q` 为查询行数，`r` 为目标列数。预测等方法需要先成功拟合。通用[参数管理](estimator-api.md#parameter-management)和[推断辅助方法](estimator-api.md#inference-helpers)另有说明；通用重采样工具不会自动提供适合生存模型的推断或 GAM 置信带。

本页模型的浮点计算使用 float64，不保留 float32 输入精度；后端原生数值预测仍使用拟合时的数组库类型。

## CoxPH 与 CoxPHCV

从 `statgpu.survival` 导入。构造参数的含义、允许值和推断限制见 [CoxPH 参数表](../models/coxph.md#参数)及 [CoxPHCV 参数表](../models/coxph.md#coxphcv-参数)。

<!-- signature: CoxPH -->
```text
CoxPH(ties='breslow', tol=1e-09, max_iter=100, device='auto', n_jobs=None, compute_inference=True, compute_cindex=True, cov_type='nonrobust', gpu_memory_cleanup=False, penalty=0.0, inference_mode='strict')
```

<!-- signature: CoxPHCV -->
```text
CoxPHCV(penalties=None, n_penalties=100, penalty_min_ratio=0.001, cv=5, cv_splits=None, ties='breslow', tol=1e-09, max_iter=100, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', inference_mode='strict', gpu_memory_cleanup=False, random_state=None)
```

<!-- signature: CoxPH.fit -->
```text
CoxPH.fit(X=None, time=None, event=None, entry=None, cluster=None, init_coef=None, formula=None, data=None, *, start=None, strata=None, subject_id=None)
```

<!-- signature: CoxPHCV.fit -->
```text
CoxPHCV.fit(X, time, event=None, entry=None, cluster=None, *, start=None, strata=None, subject_id=None)
```


### 拟合与选择

- `X`：有限实数 `(n,p)` 设计矩阵，不含截距或常数列。CoxPH 也接受单特征 `(n,)`；CoxPHCV 请使用矩阵。
- `time`：正且有限的 `(n,)` 终止/随访时间；`event`：二值 `(n,)` 指示，拟合需要观察到事件。省略 `event` 时，`time` 是 `(n,2)` 的 `[time,event]` 或 `(n,3)` 的 `[start,stop,event]` 打包目标。
- `entry` / `start`：互斥的 `(n,)` 别名，满足 `0 <= start < time`；省略时起始时间为零。不要与三列目标同时提供。
- `strata`：可选 `(n,)` 标签，分别定义风险集和基线，共享系数。`subject_id`：重复记录的受试者标签，也用于保持 CV 受试者完整。`cluster`：`cov_type="cluster"` 必需的聚类标签；它本身不会让 CV 按组划分。
- `init_coef`：仅 CoxPH 接受的有限 `(p,)` 初值。`formula`、`data`：仅 CoxPH 接受的公式接口，使用 pandas DataFrame 和可选 pandas/Patsy 依赖；响应可为 `Surv(time,event)` 或 `Surv(start,stop,event)`。删除缺失行时会对齐辅助标签；预测复用已保存的设计转换，不允许静默删除行。
- 两个 `fit` 均返回自身。Exact 并列事件与稳健推断组合抛出 `NotImplementedError`；Exact 仅估计拟合可用。输入或拟合失败不能被当作有效拟合结果。
- CV `penalties` 是非空、有限、非负向量；为 `None` 时由 `n_penalties` 和 `penalty_min_ratio` 控制自动网格，特征量纲应具有可比性。`cv_splits` 用非空、不相交的整数训练/验证索引对覆盖自动划分。提供 `subject_id` 时会拒绝受试者泄漏。`random_state` 控制自动划分。
- 选择目标是相同有效折上的平均未惩罚留出偏对数似然；各折贡献其偏对数似然总和，不除以行数或事件数，也不是 C-index。候选项须在每个有效折上得到有限分数并收敛；没有合格候选项时会报错。选定惩罚后在全部输入训练数据上重拟合；推断仅在最终重拟合进行，不校正调参不确定性。自定义网格中，数值上近似并列时优先较强惩罚，因此 `best_score_` 可能略小于候选平均分数的最大值。
- 自定义 `penalties` 拒绝布尔、字符串/字节和复数元素。候选按惩罚从强到弱评价，但 `penalties_` 及候选轴结果保留输入顺序。一次性 `cv_splits` 迭代器在首次拟合、`get_params`、克隆或序列化时被读取一次，之后复用；`get_params` 返回可重复使用的等价序列。不要另外消耗该迭代器。

### 预测、评分与摘要

<!-- signature: CoxPH.predict -->
```text
CoxPH.predict(X)
```

<!-- signature: CoxPH.predict_risk_score -->
```text
CoxPH.predict_risk_score(X)
```

<!-- signature: CoxPH.predict_hazard_ratio -->
```text
CoxPH.predict_hazard_ratio(X)
```

<!-- signature: CoxPH.predict_survival -->
```text
CoxPH.predict_survival(X, times=None, strata=None)
```

<!-- signature: CoxPH.score -->
```text
CoxPH.score(X, time, event=None, start=None, strata=None, subject_id=None)
```

<!-- signature: CoxPH.summary -->
```text
CoxPH.summary()
```

<!-- signature: CoxPHCV.predict -->
```text
CoxPHCV.predict(X)
```

<!-- signature: CoxPHCV.predict_risk_score -->
```text
CoxPHCV.predict_risk_score(X)
```

<!-- signature: CoxPHCV.predict_hazard_ratio -->
```text
CoxPHCV.predict_hazard_ratio(X)
```

<!-- signature: CoxPHCV.predict_survival -->
```text
CoxPHCV.predict_survival(X, times=None, strata=None)
```

<!-- signature: CoxPHCV.score -->
```text
CoxPHCV.score(X, time, event=None, start=None, strata=None, subject_id=None)
```

<!-- signature: CoxPHCV.summary -->
```text
CoxPHCV.summary()
```


| 方法 | 输入和返回值 |
|---|---|
| `predict_risk_score` | 特征顺序相同的 `(q,p)` 设计；返回后端 `(q,)` 对数风险 `X @ coef_`。 |
| `predict`、`predict_hazard_ratio` | 返回后端 `(q,)` 相对风险 `exp(X @ coef_)`，不是概率；无法有限表示的指数值会报错，不会截断。 |
| `predict_survival` | `times=None` 使用拟合事件时间的并集；否则接受训练时间单位下的有限标量或一维序列。返回后端数组元组 `(curves,times)`，形状分别为 `(q,n_times)`、`(n_times,)`；保留显式时间顺序。首次失败前生存概率为 1，末次失败后阶梯函数保持不变。 |
| 生存预测的 `strata` | 显式分层拟合时，每个查询必须提供已知标签，只有一个拟合分层也如此。缺失/未知标签会报错。没有事件的分层返回 1。基线需要 `compute_inference=True`。 |
| `score` | `X`、分开的或打包的目标，以及可选 `start`、`strata`、`subject_id`；没有 `entry` 别名。返回 Harrell 风格 C-index 标量。多分层拟合需要标签；提供的标签须已知。同一受试者内的样本对被排除。无可比较样本对时返回 `0.5`，不能据此认为模型已经得到充分验证。 |
| `summary` | 打印拟合摘要并返回 `None`；使用 `model.summary()`，无需外层 `print`。CoxPHCV 委托给 `estimator_`。 |

### 系数、推断与 CV 结果

| 字段 | 形状/类型与解释 |
|---|---|
| `coef_`、`hazard_ratios_` | NumPy `(p,)`，分别为系数和其指数；没有截距。 |
| `_bse`、`_zvalues`、`_pvalues` | 启用推断时的 NumPy `(p,)` 标准误、系数 z 统计量和双侧正态 p 值；否则为 `None`。这些已有结果字段以下划线开头。 |
| `_conf_int` | NumPy `(p,2)`，固定 **95% 逐系数、系数尺度**正态区间。`np.exp(model._conf_int)` 才是风险比区间，也是 `summary()` 展示的形式；这里没有置信水平参数。 |
| `log_likelihood` | CoxPH 属性，拟合系数处未惩罚偏对数似然标量；仅估计拟合后也可读。 |
| `aic`、`bic` | 无惩罚 CoxPH 的标量属性。令 `k=p`、`d` 为事件数，AIC 为 `-2*log_likelihood+2*k`，BIC 为 `-2*log_likelihood+log(d)*k`；正惩罚拟合后读取会抛出 `RuntimeError`。 |
| `concordance_index` | CoxPH 训练 C-index 属性；`compute_cindex=False` 时为 `None`。 |
| `converged_`、`n_iter_` | 布尔值和整数；还应查看 `termination_reason_`、`optimization_stop_reason_`、`final_kkt_inf_`、`final_kkt_normalized_`。 |
| `penalties_`、`penalty_`、`best_score_` | CoxPHCV 搜索的惩罚向量（保留输入顺序）、选定标量及所选平均留出偏对数似然；适用上述数值近似并列规则。 |
| `estimator_` | CoxPHCV 最终拟合的 CoxPH。通过它读取 CoxPH 属性，例如 `cv.estimator_.log_likelihood`；这些属性并非全部直接出现在 CV 对象上。 |
| `cv_results_` | 字典：`pl_path`、`converged_path`、`failure_path` 按惩罚×折组织；`mean_pl`、`effective_fold_counts` 为逐惩罚向量；`fold_valid` 标识有效折。其他后端/缓存诊断见 [Cox 文档](../models/coxph.md#输出)。 |

CoxPHCV 从最终估计器复制系数及推断数组。`inference_method_`、`inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_` 说明推断目标；`inference_backend_`、`inference_approximate_`、`inference_fallback_reason_` 和传输字段说明执行情况。联合检验可用性与逐系数区间分开报告。正惩罚推断以固定惩罚估计方程为目标，不是去偏或选择后校正推断；见[协方差与推断](../models/coxph.md#协方差与推断)。

<!-- example: cox-output-reference-cpu -->
```python
import numpy as np
from scipy.stats import norm
from statgpu.survival import CoxPH

rng = np.random.default_rng(53)
X = rng.normal(size=(160, 2))
event_time = rng.exponential(np.exp(-X @ np.array([0.5, -0.3])))
censor_time = rng.exponential(2.0, size=len(X))
time = np.maximum(np.round(np.minimum(event_time, censor_time), 1), 0.1)
event = (event_time <= censor_time).astype(int)
model = CoxPH(ties="efron", device="cpu").fit(X, time, event)
coefficient_ci = np.column_stack([
    model.coef_ - norm.ppf(0.975) * model._bse,
    model.coef_ + norm.ppf(0.975) * model._bse,
])
hazard_ratio_ci = np.exp(coefficient_ci)
query = X[:2]
curves, grid = model.predict_survival(query, times=[0.0, 0.5, 1.0])
failure_times = np.unique(time[event == 1])
increments = np.array([
    np.sum((time == t) & (event == 1)) / np.exp(X[time >= t] @ model.coef_).sum()
    for t in failure_times
])
baseline = np.array([increments[failure_times <= t].sum() for t in grid])
survival_from_formula = np.exp(-np.exp(query @ model.coef_)[:, None] * baseline)
print(coefficient_ci.shape, hazard_ratio_ci.shape, curves.shape)
print(np.allclose(coefficient_ci, model._conf_int))
print(np.allclose(curves, survival_from_formula))
summary_result = model.summary()
assert summary_result is None
```


前三行输出为 `(2, 2) (2, 2) (2, 3)`、`True`、`True`，之后是打印的摘要。即使系数使用 Efron 并列事件方法，显式求和仍按当前实现的 Breslow 基线重建生存概率。

## GAM

从 `statgpu.semiparametric` 导入。完整构造参数含义和允许值见 [GAM 参数表](../models/semiparametric.md#完整构造参数与输出参考)。

<!-- signature: GAM -->
```text
GAM(n_splines=20, degree=3, lam=None, penalty_order=2, knot_method='quantile', gamma=1.0, device='auto', n_jobs=None)
```

<!-- signature: GAM.fit -->
```text
GAM.fit(X, y=None, **fit_params)
```

<!-- signature: GAM.predict -->
```text
GAM.predict(X)
```

<!-- signature: GAM.summary -->
```text
GAM.summary()
```


- `fit`：有限数值 `(n,p)` 或单特征 `(n,)` 的 `X`；每行一个有限响应，单列 `y` 会展平。`y=None` 不代表可省略目标拟合。返回自身。当前只使用 `X` 和 `y`；额外 `fit_params`（包括 `sample_weight`）会被忽略，不能启用加权拟合，请勿传入。
- `predict`：接受 `(q,p)`，或按拟合特征数解释的向量；即使 GPU 拟合也返回 NumPy `(q,)`。复用节点、边界和中心化，不提供可靠外推。
- 多特征 Torch GAM 预测请显式使用 `(q,p)`，单个点为 `(1,p)`；当前一维向量的形状检查会抛出 `TypeError`。
- `summary`：打印并返回字典，含 `n_features`、`n_splines_per_feature`、`spline_degree`、`penalty_order`、`smoothing_parameter`、`effective_df`、`intercept`，仅自动选择时另有 `gcv_score`。
- 拟合属性：后端 `coef_`，长度为 `1+sum(n_basis_j)`；逐特征后端节点数组 `knots_`；标量 `intercept_`、`edf_`、`lam_`；自动选择时 `gcv_score_` 为浮点数，固定 lambda 时为 `None`；`n_features_` 为整数。系数对应中心化的基函数，不是原始特征斜率。
- `lam=None` 搜索内置 100 点网格；没有 GAM 自定义网格或专用 CV 估计器接口。其他设置可用外部验证选择。没有专用 `score`、样本加权目标、family/link、系数推断或置信带方法；继承的通用工具不会自动补齐这些能力。
- `set_params` 后始终重新拟合。部分 GAM 改参当前保留旧拟合数组，未重拟合便预测可能使用不一致状态。改变基函数设计时，创建新 GAM 实例最稳妥。重拟合若在构造基函数时失败，还可能把旧系数与新节点或特征数混在一起；此时应丢弃该实例，在新实例上成功拟合后再预测或解释 `summary()`。

自动选择平滑参数后，请检查 `gcv_score_` 是否有限。若全部候选的 GCV 都是无穷大，当前实现仍返回第一个网格值；仅仅返回已拟合对象不能证明选择成功。此时应重新检查设计，或采用经过外部验证的固定 `lam`。

## 核密度估计

以下名称均从 `statgpu.nonparametric` 导入。`KDE` 是 `KernelDensityEstimator` 的别名子类，构造参数和方法相同；构造参数须按名称传入。 使用 Torch 时，请先查看[Torch 参数限制](#torch-参数限制)。

<!-- signature: KernelDensityEstimator -->
```text
KernelDensityEstimator(*, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', device='auto', n_jobs=None, gpu_memory_cleanup=False)
```

<!-- signature: KernelDensityEstimator.fit -->
```text
KernelDensityEstimator.fit(X, y=None)
```

<!-- signature: KernelDensityEstimator.pdf -->
```text
KernelDensityEstimator.pdf(points, *, batch_size=1024)
```

<!-- signature: KernelDensityEstimator.logpdf -->
```text
KernelDensityEstimator.logpdf(points, *, batch_size=1024)
```

<!-- signature: KernelDensityEstimator.__call__ -->
```text
KernelDensityEstimator.__call__(points, *, batch_size=1024)
```

<!-- signature: KernelDensityEstimator.predict -->
```text
KernelDensityEstimator.predict(X)
```

<!-- signature: KernelDensityEstimator.score_samples -->
```text
KernelDensityEstimator.score_samples(X)
```

<!-- signature: KernelDensityEstimator.score -->
```text
KernelDensityEstimator.score(X, y=None)
```

<!-- signature: KernelDensityEstimator.to_numpy_metadata -->
```text
KernelDensityEstimator.to_numpy_metadata()
```

<!-- signature: fit_kde -->
```text
fit_kde(samples, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto')
```

<!-- signature: kde_pdf -->
```text
kde_pdf(samples, points, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', return_log=False, batch_size=1024)
```


| 参数 | 含义与限制 |
|---|---|
| `bandwidth` | 正且有限的标量因子，或 `scott`、`silverman`、`nrd0`、`nrd`、`ucv`、`bcv`、`sj`、`sj-ste`、`sj-dpi`。数值是协方差因子，不是原始单位下的宽度。 |
| `weights` | 构造/函数式拟合参数，不是 `fit` 关键字：有限、非负、长度 `n`，总和为正，内部归一化；`None` 为等权。协方差需要不止一个有有效权重的观测。 |
| `kernel` | 默认 `gaussian`，另有 `rectangular`、`triangular`、`epanechnikov`、`biweight`、`triweight`、`cosine`、`optcosine`；最后两种仅一维。 |
| `backend` | `auto`、`numpy`、`cupy`、`torch`，选择数组库；`auto` 参考设备/全局配置；显式设备请求的例外见下文。 |
| `device`、`n_jobs`、`gpu_memory_cleanup` | 仅估计器接受的设备控制、通用任务数选项和尽力执行的 GPU 缓存清理。核评价不会使用 `n_jobs` 建立并行查询池；函数式接口不接受这些参数。 |
| `batch_size` | `pdf`、`logpdf`、`__call__`、`kde_pdf` 的正查询批大小；不是 KDE 构造参数，也不是 `predict` / `score_samples` / `score` 参数。部分 NumPy 快速路径会一起评价较小任务。 |
| `return_log` | 仅 `kde_pdf`：`False` 返回密度，`True` 返回对数密度。 |

`fit(X,y=None)` 忽略 `y` 并返回自身；`fit_kde` 返回已拟合 `KDE`。样本为有限 `(n,)` 或 `(n,p)`，`n>=2`；查询为有限 `(q,p)`。向量对单特征表示多个点，对多特征可表示一个点。`pdf`、`predict`、`__call__` 返回后端 `(q,)` 密度；`logpdf`、`score_samples` 返回对数密度。`score` 返回 Python `float`，即未加权的查询平均对数密度，并忽略 `y`。紧支撑核可返回密度 0、对数密度 `-inf`。

原始坐标带有很大偏移量时，请先减去训练数据确定的偏移量，并对查询使用同一偏移量，再拟合/评价。当前二次距离计算可能在未中心化时损失精度，涉及 KDE 对数密度及多元密度/回归；这种共同平移不会改变目标统计估计量。

KDE 与核回归当前存在显式设备请求的例外。NumPy 或 Torch CPU 输入即使搭配 `device="torch"` 与 `backend="auto"` 或 `"torch"`，仍可能在 Torch CPU 上拟合和预测。显式 `backend="torch"` 搭配 `device="cuda"` 也可能在 CPU 上执行；`backend="numpy"` 会覆盖这两种加速器请求。应检查 `samples_` 与密度/预测数组的实际位置：Torch 查看 `.device`/`.is_cuda`，CuPy 查看 `.device`，NumPy 数组位于 CPU。仅查看配置的 `device` 与 `backend_` 不够。需要明确的 CPU 路径时，请用 NumPy 输入并设置 `device="cpu", backend="numpy"`。以上是严格设备约定的当前例外，并非设备参数的新含义，见[设备说明](../guides/device-and-memory.md)。

拟合属性：`samples_` `(n,p)`、归一化 `weights_` `(n,)`、标量 `bandwidth_factor_`、`bandwidth_info_`（选择结果；数值带宽时为 `None`）、`covariance_` 和 `inv_covariance_` `(p,p)`、标量 `norm_const_` 和 `inv_norm_const_`、`kernel_`、`backend_`、`n_samples_`、`n_features_`。`to_numpy_metadata()` 返回含 `bandwidth_factor`、`bandwidth_selection`、`n_samples`、`n_features`、`backend`、`kernel`、`covariance`、`inv_covariance`、`weights` 的字典，数组为主机 NumPy 数组。

`fit` 与 `score` 不使用可选 `y` 估计密度或计算分数，但传入非有限 `y` 仍会被通用输入校验拒绝；建议省略该参数。

使用加权高斯 KDE 的 `logpdf`、`score_samples` 或 `score` 时，请先删除零权重观测及其对应权重。虽然这些观测不贡献概率质量，但当前实现可能因它们的存在而将尾部对数密度错误地算成 `-inf`。删除后会自动重新归一化权重。如果分析希望根据正权重观测选择新带宽，应先筛选再选择。重新运行字符串选择器可能改变所选因子：`nrd`/`nrd0` 使用原始样本的尺度统计量，`ucv`、`bcv`、`sj-ste` 与 `sj-dpi` 则可能在删除后改变加权/重采样路径。若需保留已经选定的平滑因子，请保存 `original.bandwidth_factor_`，再用数值参数 `bandwidth=original.bandwidth_factor_`、正权重观测及其保留权重重新拟合。这会保留指定的核协方差与密度，但不代表受零权重行影响而选出的因子适合原本的科学问题。

## 核回归

`KernelRegressionRegressor` 是 `KernelRegression` 的别名子类，构造参数和方法相同。共同的核、带宽、权重、后端、设备、任务数和清理参数沿用上节含义。

<!-- signature: KernelRegression -->
```text
KernelRegression(*, bandwidth='scott', weights=None, kernel='gaussian', regression='nw', kernel_metric='full', bandwidth_per_feature=None, backend='auto', device='auto', n_jobs=None, batch_size=1024, min_effective_weight=1e-12, gpu_memory_cleanup=False)
```

<!-- signature: KernelRegression.fit -->
```text
KernelRegression.fit(X, y)
```

<!-- signature: KernelRegression.predict -->
```text
KernelRegression.predict(points, *, batch_size=None, min_effective_weight=None)
```

<!-- signature: KernelRegression.__call__ -->
```text
KernelRegression.__call__(points, *, batch_size=None, min_effective_weight=None)
```

<!-- signature: KernelRegression.score -->
```text
KernelRegression.score(X, y)
```

<!-- signature: KernelRegression.to_numpy_metadata -->
```text
KernelRegression.to_numpy_metadata()
```

<!-- signature: fit_kernel_regression -->
```text
fit_kernel_regression(samples, targets, *, bandwidth='scott', weights=None, kernel='gaussian', regression='nw', kernel_metric='full', bandwidth_per_feature=None, backend='auto')
```

<!-- signature: kernel_regression_predict -->
```text
kernel_regression_predict(samples, targets, points, *, bandwidth='scott', weights=None, kernel='gaussian', regression='nw', kernel_metric='full', bandwidth_per_feature=None, backend='auto', batch_size=1024, min_effective_weight=1e-12)
```


- `regression="nw"` 是 Nadaraya–Watson；`"local_linear"` 在每个查询点拟合局部截距和斜率，没有全局系数向量或系数推断。
- `kernel_metric="full"` 使用加权协方差，`"diagonal"` 去掉非对角项。`bandwidth_per_feature=None` 选择标量因子；正的逐特征绝对宽度（标量可广播）需要对角度量，并跳过 `bandwidth` 选择器。
- `fit(X,y)` 返回自身；`fit_kernel_regression(samples,targets,...)` 返回已拟合 `KernelRegression`。样本规则与 KDE 相同；目标为有限 `(n,)` 或 `(n,r)`。预测保留目标维数：`(q,)` 或 `(q,r)`，单列二维目标也返回 `(q,1)`。
- 构造参数 `batch_size=1024`、`min_effective_weight=1e-12` 成为预测默认值；方法参数为 `None` 时继承，显式正值仅覆盖本次调用。局部总权重太低时返回训练目标的加权均值；局部线性不稳定时可采用数值稳定化或 NW 回退。
- `kernel_regression_predict` 一次完成拟合和预测，并接受签名列出的评价覆盖参数。函数式接口接受 `backend`，不接受仅估计器使用的设备/任务数/清理参数。
- `score` 把所有目标列展平后计算一个主机 R-squared 标量，不是逐目标分数的平均；常数目标返回 `0.0`。目标量纲不同时应分别评价。
- 回归专用带宽别名 `cv`、`cv_ls`、`cv-nw`、`cv-ll` 搜索留一 MSE。选择阶段使用完整协方差，多元局部线性选择使用 NW 预测；对角或多元局部线性模型应另行验证实际目标配置。这些名称不适用于 KDE。

除共同拟合字段（不含 KDE 归一化常数）外，回归还有 `targets_` `(n,r)`、`n_targets_`、`target_mean_` `(r,)`、`target_was_1d_`、`regression_`、`kernel_metric_`、`bandwidth_per_feature_`（后端向量或 `None`）。元数据含 `bandwidth_factor`、`bandwidth_selection`、`bandwidth_per_feature`、`n_samples`、`n_features`、`n_targets`、`backend`、`kernel`、`kernel_metric`、`regression`、`covariance`、`inv_covariance`、`weights`、`target_mean`。

## 密度置信区间

即使 `method="normal"` 不进行自助抽样，`bootstrap_method` 也必须为 `"percentile"`。

<!-- signature: kde_confidence_interval -->
```text
kde_confidence_interval(samples, points, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', n_resamples=200, confidence_level=0.95, random_state=None, method='normal', bootstrap_method='percentile', return_bootstrap_samples=False, batch_size=1024)
```

<!-- signature: kde_bootstrap_confidence_interval -->
```text
kde_bootstrap_confidence_interval(samples, points, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', n_resamples=200, confidence_level=0.95, random_state=None, method='percentile', return_bootstrap_samples=False, batch_size=1024)
```

<!-- signature: KDEBootstrapResult.to_dict -->
```text
KDEBootstrapResult.to_dict()
```


共同的 `samples`、`points`、`bandwidth`、`weights`、`kernel`、`backend` 沿用 KDE 含义。`confidence_level` 应为严格位于 0 与 1 之间的有限数值，请勿传入 NaN。`n_resamples` 应为正整数，正态路径也会校验该参数。`random_state` 是 bootstrap 抽样种子；`batch_size` 控制密度评价，NumPy bootstrap 快速路径可能一起评价全部查询。

- `kde_confidence_interval(method="normal")`：仅一维高斯 KDE；绝对宽度来自拟合协方差。返回渐近、逐点、下界截断到零的正态区间。结果 `n_resamples=0`、`bootstrap_samples=None`，即使请求 `return_bootstrap_samples=True` 也如此。
- `kde_confidence_interval(method="bootstrap", bootstrap_method="percentile")`：百分位重采样；不支持其他 bootstrap 区间方法。
- `kde_bootstrap_confidence_interval(method="percentile")`：该 bootstrap 路径的便捷包装。这里的 `method` 指百分位方法，通用函数的 `method` 则区分正态与 bootstrap。
- `return_bootstrap_samples=True` 保留重复样本的估计矩阵，否则结果不含该矩阵。区间不校正平滑偏差，不是同时覆盖；[教程](../models/nonparametric.md#cpu-示例三逐点密度区间)解释了选择器重拟合，以及示例解释为何限定于独立等权观测。非均匀权重同时用于抽样及再次加权，需要根据研究设计判断其合理性。

`KDEBootstrapResult` 用于**两种**区间方法。数组均为 NumPy：一维 `points` 为 `(q,)`，多维为 `(q,p)`；`estimate`、`lower`、`upper` 均为 `(q,)`；可选 `bootstrap_samples` 为 `(n_resamples,q)`。另有 `confidence_level`、`n_resamples`、`random_state`、`kernel`、`backend`、`metadata`。后端标签记录拟合的数组库，不代表返回数组仍在该设备。元数据包括 `method`、`bandwidth`、`batch_size`、`n_features`，正态路径另有 `n_eff`，bootstrap 路径另有 `bootstrap_method`。`to_dict()` 把结果数组转为列表，没有重复样本时省略 `bootstrap_samples`。

<!-- example: kde-normal-reference-cpu -->
```python
import numpy as np
from scipy.stats import norm
from statgpu.nonparametric import fit_kde, kde_confidence_interval

samples = np.linspace(-2.0, 2.0, 60)
points = np.array([-1.0, 0.0, 1.0])
model = fit_kde(samples, bandwidth=0.4, backend="numpy")
ci = kde_confidence_interval(samples, points, bandwidth=0.4, backend="numpy")
h = np.sqrt(model.covariance_[0, 0])
n_eff = 1.0 / np.sum(model.weights_ ** 2)
se = np.sqrt(ci.estimate / (2 * np.sqrt(np.pi) * n_eff * h))
normal_lower = np.maximum(ci.estimate - norm.ppf(0.975) * se, 0.0)
normal_upper = ci.estimate + norm.ppf(0.975) * se
print(ci.estimate.shape, ci.n_resamples, ci.bootstrap_samples)
assert np.allclose(ci.lower, normal_lower)
assert np.allclose(ci.upper, normal_upper)
```


输出为 `(3,) 0 None`；这验证了计算公式，不代表验证了所有数据上的实际覆盖率。

### Torch 参数限制

在 Torch 上，向 KDE 或核回归传入 `weights` 当前会抛出 `TypeError`。
回归的 `bandwidth_per_feature` 无论为标量还是向量也会失败。即使用户没有提供权重，
Torch bootstrap 区间仍会失败，因为重采样拟合内部会显式传入权重。
这些需求请改用 CPU 数组与 `backend="numpy"`；改变权重或带宽并不是等价替代。
不加权的一维高斯 Torch KDE 仍可使用正态区间。

Torch 多元密度与回归的查询应为 `(q,p)`，单个查询也须为 `(1,p)`。
长度为 `p` 的一维向量当前会抛出 `TypeError`；`X[:1]` 可以保留所需行维度。

## 底层带宽选择接口

通常只需设置估计器的 `bandwidth`。以下公开函数面向已经准备好经过校验的后端数组和协方差的调用方。

<!-- signature: select_bandwidth -->
```text
select_bandwidth(bandwidth, *, n_eff, n_features, samples_2d, weights_1d, data_cov, xp, enable_r_selectors=True, weighted_r_selector_strategy='quantile_resample', multivariate_selector_strategy='projection_pca_1d', estimator='kde', targets=None, regression='nw', kernel='gaussian')
```

<!-- signature: select_bandwidth_factor -->
```text
select_bandwidth_factor(bandwidth, *, n_eff, n_features, samples_2d, weights_1d, data_cov, xp, enable_r_selectors=True, weighted_r_selector_strategy='quantile_resample', multivariate_selector_strategy='projection_pca_1d', estimator='kde', targets=None, regression='nw', kernel='gaussian')
```

<!-- signature: BandwidthSelectionResult.to_dict -->
```text
BandwidthSelectionResult.to_dict()
```


| 参数 | 所需输入或行为 |
|---|---|
| `bandwidth` | 所选估计器支持的名称/标量因子；回归 CV 名称需要 `estimator="kernel_regression"`。 |
| `n_eff`、`n_features` | 正的有效样本量和特征数 `p`；归一化权重对应 `n_eff=1/sum(weights_1d**2)`。 |
| `samples_2d`、`weights_1d`、`data_cov`、`xp` | `(n,p)` 后端样本、归一化 `(n,)` 权重、未按带宽缩放的 `(p,p)` 加权协方差，以及匹配的 NumPy/CuPy/Torch 数组模块。使用有限、兼容输入；传入模块不会自动把数组迁移到 GPU。 |
| `enable_r_selectors=True` | 控制 `ucv`、`bcv`、`sj`、`sj-ste`、`sj-dpi`；设为 `False` 时会拒绝这些选择器。正态参考规则 `nrd` 和 `nrd0` 不受此开关限制。算法不会启动 R 进程。 |
| `weighted_r_selector_strategy` | 非均匀权重下 R 风格选择支持 `"quantile_resample"` 策略。 |
| `multivariate_selector_strategy` | R 风格选择的多元扩展支持 `"projection_pca_1d"`；不等同于精确多元 R 方法。 |
| `estimator`、`targets`、`regression`、`kernel` | 默认 `"kde"`；回归 CV 选择使用 `"kernel_regression"` 和匹配目标。`regression` 为 `"nw"` 或 `"local_linear"`，`kernel` 选择回归核；默认值见签名。 |

`select_bandwidth` 返回 `BandwidthSelectionResult`；`select_bandwidth_factor` 返回其中的标量 `factor`。结果字段为 `factor`、`method`、`n_features`、`n_eff`、`used_r_selector`、`weighted`、`weighted_strategy`、`multivariate_strategy`、`selector_dimension`、`details`；`to_dict()` 返回相同名称。`details` 随方法变化，不应假定所有选择器都有相同键。`used_r_selector=True` 表示使用 `ucv`、`bcv` 或 `sj` 系列方法；`nrd`/`nrd0` 对应 `False`，该字段不代表调用了外部 R 进程；部分常数/稀疏样本会使选择失败。

<!-- example: bandwidth-selector-reference-cpu -->
```python
import numpy as np
from statgpu.nonparametric import fit_kde, select_bandwidth, select_bandwidth_factor

samples = np.linspace(-2.0, 2.0, 40)
base = fit_kde(samples, bandwidth=1.0, backend="numpy")
selector_inputs = dict(
    n_eff=1.0 / np.sum(base.weights_ ** 2),
    n_features=base.n_features_, samples_2d=base.samples_,
    weights_1d=base.weights_, data_cov=base.covariance_, xp=np,
)
selection = select_bandwidth("scott", **selector_inputs)
factor = select_bandwidth_factor("scott", **selector_inputs)
print(selection.method, round(factor, 6))
assert np.isclose(factor, 40 ** (-1 / 5))
assert factor == selection.factor
```


输出为 `scott 0.478176`。复用带宽因子为一的 KDE 可取得协方差和归一化权重，无需重复准备过程。

## 实现链接

查阅实现细节可用：[CoxPH](../../../statgpu/survival/_cox.py)、[CoxPHCV](../../../statgpu/survival/_cox_cv.py)、[GAM](../../../statgpu/semiparametric/_gam.py)、[KDE 与区间](../../../statgpu/nonparametric/kernel_smoothing/_kde.py)、[核回归](../../../statgpu/nonparametric/kernel_smoothing/_kernel_regression.py)、[选择器](../../../statgpu/nonparametric/kernel_smoothing/_bandwidth_selection.py)。统计学参考文献见对应教程。
