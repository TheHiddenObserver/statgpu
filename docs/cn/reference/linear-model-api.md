# 线性模型 API 参考

> 语言：中文  
> 最后更新：2026-10-06  
> 切换：[English](../../en/reference/linear-model-api.md)

本页各类可从 `statgpu` 或 `statgpu.linear_model` 导入，但 `SCADRegression` 和 `MCPRegression` 仅由 `statgpu.linear_model` 导出。`X` 为有限数值 `(n,p)` 矩阵，预测列顺序与训练一致。分析权重为有限非负 `(n,)` 向量，总和须为正。公式删行后的权重对齐见[公式输入](#formula-inputs)。显式 GPU 请求要求对应 CUDA 后端可用，详见[设备与内存](../guides/device-and-memory.md)。

这些类均继承 [get_params/set_params 和四个推断辅助方法](estimator-api.md)。通用辅助方法可以接收显式数据/p 值；类上有这些方法，不代表 CV 包装对象本身一定保存系数推断数组。CV 的推断结果应从 `estimator_` 读取。

这些直接估计器的公开 `coef_` 和成功计算的系数推断数组保存在 NumPy 中，包括 GPU 拟合后的结果；单目标截距为 Python 数值。这与后端原生预测/评估数组的存储方式不同。构造函数不提供 `dtype` 控制，也不保证始终保留输入浮点类型；与其他库配合时请查看实际返回数组的 dtype。

## LinearRegression

```text
LinearRegression(fit_intercept=True, device='auto', n_jobs=None, compute_inference=True, gpu_memory_cleanup=False, cov_type='nonrobust', hac_maxlags=None)
```
| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `fit_intercept` | `True` | 拟合截距；公式语法决定公式拟合的截距。 |
| `device` | `"auto"` | `cpu`/`cuda`（CuPy）/`torch`（Torch CUDA）/`auto`；显式 GPU 请求要求相应后端可用。 |
| `n_jobs` | `None` | 共享 CPU 工作线程配置；不选择求解器，也不保证这些包装类会并行拟合。 |
| `compute_inference` | `True` | 启用受支持的拟合后推断；`summary()` 要求推断成功。 |
| `gpu_memory_cleanup` | `False` | 拟合后尝试清理后端内存池。 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3`、`hac`；适用性见各模型推断说明。 |
| `hac_maxlags` | `None` | HAC 非负整数滞后阶；Linear/Logistic 自动值为 `floor(4*(n/100)**(2/9))`，上限 n−1。 |

`fit(X=None, y=None, sample_weight=None, formula=None, data=None)` 返回 `self`。使用数组，或 `formula` 与 `data`，不要同时提供两种表示。当前公式拟合即使同时收到数组 X/y，也直接从 `data` 取设计与响应，不会拒绝冲突。fit 会展平单列响应；真正多目标 `y` 的形状为 `(n,t)`。

| 方法/结果 | 约定 |
|---|---|
| `predict(X)` | 单目标 `(m,)`，多目标 `(m,t)`；CPU 返回 NumPy，GPU 返回受支持的原生后端数组。公式拟合后也可传预测 DataFrame。预测时会重新解析设备；如需避免后续全局设备设置改变输出位置，应显式指定设备。 |
| `score(X,y)` | 不加权 R²；多目标取各目标 R² 的均值。不接受 `sample_weight`。单目标 `y` 必须展平；当前单列响应会触发错误广播。常数目标返回 0，不宜解释为通常的方差解释比例。 |
| `summary()` | 打印单目标表格，返回 `None`。要求推断成功且残差自由度为正。 |
| `coef_`、`intercept_`、`rank_` | 单目标形状 `(p,)`/标量，多目标 `(t,p)`/`(t,)`；`rank_` 是拟合设计矩阵的秩。 |
| `rsquared`、`rsquared_adj` | 训练 R²（使用拟合权重；多目标为合并计算）及残差自由度修正。不同于留出数据的 `score`。 |
| `fvalue`、`f_pvalue` | 单目标的残差 F 诊断，不是 HC/HAC 联合 Wald 检验。加权多目标访问当前抛出 `TypeError`；无权重的合并输出不是联合多元检验。 |
| `llf`、`aic`、`bic` | 无权重单目标的高斯似然和信息准则。加权数值缺少 WLS 对数权重归一化项，会随权重尺度改变。多目标 AIC/BIC 为 `None`，合并 `llf` 不是联合多元似然。详见[诊断限制](../models/linear-regression.md#加权或多目标诊断的限制)。 |
| `_bse`、`_tvalues`、`_pvalues`、`_conf_int` | 拟合截距时置于首位。k 个参数时形状为 `(k,)`，区间为 `(k,2)`；CPU 多目标为 `(k,t)` 与 `(k,t,2)`。关闭或无法计算推断时为 `None`。 |

经典协方差采用 t 参考分布；HC/HAC 采用正态参考分布，尽管属性名仍为 `_tvalues`。CPU 支持多目标推断；GPU 多目标拟合需设 `compute_inference=False`。解释和假设见[入门页](../models/linear-regression.md)。

## LogisticRegression

```text
LogisticRegression(fit_intercept=True, C=1.0, max_iter=100, tol=0.0001, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', gpu_memory_cleanup=False, hac_maxlags=None)
```
| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `fit_intercept` | `True` | 拟合不受惩罚的截距；此类不提供公式接口。 |
| `device` | `"auto"` | `cpu`/`cuda`（CuPy）/`torch`（Torch CUDA）/`auto`；显式 GPU 请求要求相应后端可用。 |
| `n_jobs` | `None` | 共享 CPU 工作线程配置；不选择求解器，也不保证这些包装类会并行拟合。 |
| `compute_inference` | `True` | 启用受支持的拟合后推断；`summary()` 要求推断成功。 |
| `gpu_memory_cleanup` | `False` | 拟合后尝试清理后端内存池。 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3`、`hac`；适用性见各模型推断说明。 |
| `hac_maxlags` | `None` | HAC 非负整数滞后阶；Linear/Logistic 自动值为 `floor(4*(n/100)**(2/9))`，上限 n−1。 |
| `C` | `1.0` | 有限非负实数；正数为 L2 强度倒数，零表示完全去掉惩罚。 |
| `max_iter` | `100` | IRLS 最大迭代次数，正整数。 |
| `tol` | `1e-4` | 有限正数收敛容差。 |

`fit(X,y,sample_weight=None)` 返回 `self`；二分类 `y` 必须为 n 个 0/1 标签。此独立类没有公式参数、多分类模式、求解器选择、L1 或 Elastic Net 惩罚。`coef_` 为 `(p,)`，`intercept_` 为标量。`converged_` 与 `n_iter_` 描述数值优化，不代表统计模型合理。

| 方法 | 参数、默认值与返回值 |
|---|---|
| `predict_proba(X)` | `(m,2)` 概率，列依次为类别 0、1。 |
| `predict(X)` | `(m,)` 整数标签；概率 ≥0.5 时为 1。 |
| `predict_with_threshold(X,threshold=0.5)` | 使用 `[0,1]` 内有限实数阈值的标签；不接受布尔阈值。 |
| `score(X,y)` | Python 浮点数，不加权准确率；没有阈值或权重参数。 |
| `confusion_matrix(X,y,threshold=0.5)` | `(2,2)` 数组 `[[tn,fp],[fn,tp]]`；行为真实标签，列为预测标签。 |
| `classification_table(X,y,threshold=0.5)` | 字典：`tn`、`fp`、`fn`、`tp`、`accuracy`、`precision`、`recall`、`specificity`、`f1`、`support_negative`、`support_positive`。比例分母为零时返回 0。 |
| `roc_curve(X,y)` | `(fpr,tpr,thresholds)` 元组，数组等长；阈值递减且首项为无穷大。评估标签必须同时包含两类，否则抛出 `ValueError`。 |
| `roc_auc_score(X,y)` | 梯形积分 ROC 面积标量。 |
| `precision_recall_curve(X,y)` | `(precision,recall,thresholds)` 元组；本实现三数组等长，阈值递减，首项无穷大对应 precision=1、recall=0。不要套用其他库的长度约定。至少需要一个正标签；全零评价 y 抛出 `ValueError`，全一 y 则可以计算。 |
| `average_precision_score(X,y)` | 按召回率增量积分的平均精确率标量；与精确率—召回率曲线一样，至少需要一个正标签。 |
| `evaluate_classification(X,y,threshold=0.5,include_curves=True)` | 一次概率计算返回下述指标字典。 |
| `plot_roc_curve(X,y,ax=None,label=None)` | 需要 matplotlib；新建或使用传入 Axes 并返回它。默认标签含 AUC。 |
| `plot_precision_recall_curve(X,y,ax=None,label=None)` | 同样返回 Axes，默认标签含平均精确率。 |
| `summary()` | 打印推断报告，返回 `None`；要求开启且成功完成推断。 |

评估数组/标量在 CPU 使用 NumPy，在 GPU 使用相应 CuPy/Torch 后端；`score` 返回 Python 浮点数。绘图把小型结果转为 NumPy。这些评估方法均不接受 `sample_weight`；加权拟合不代表评估也加权。

`evaluate_classification` 始终返回 `threshold`、`confusion_matrix`、`classification_table`、`roc_auc`、`average_precision`。即使设置 `include_curves=False`，仍会计算标量 ROC AUC，因此评估 y 必须同时包含两类，否则抛出 `ValueError`。对单类别子集，可用 `classification_table` 或 `confusion_matrix` 计算阈值指标。`include_curves=True` 时再加入 `roc_curve={fpr,tpr,thresholds}` 与 `precision_recall_curve={precision,recall,thresholds}`。对外部概率可调用 `statgpu.metrics.evaluate_binary_classification(y_true,y_score,threshold=0.5,include_curves=True,backend="auto")` 或顶层别名 `statgpu.evaluate_binary_classification`；传入一维类别 1 概率。

训练属性包括 `loglikelihood`、`loglikelihood_null`、`aic`、`bic`、`pseudo_rsquared`、`accuracy`、`precision`、`recall`、`f1`、`auc`、`average_precision`。`pseudo_rsquared` 是 McFadden 的 `1-loglikelihood/loglikelihood_null`，不是 R² 或准确率。推断数组 `_bse`、`_zvalues`、`_pvalues` 为 `(k,)`，`_conf_int` 为 `(k,2)`，截距在首位。C>0 时采用围绕惩罚拟合系数的正态参考推断；关闭推断时这些数组不可用。训练指标不衡量泛化表现。

## Lasso

```text
Lasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=0.0001, stopping='coef_delta', inference_method='debiased', n_bootstrap=200, bootstrap_random_state=None, enable_simultaneous_inference=False, simultaneous_method='maxz_bootstrap', simultaneous_alpha=0.05, simultaneous_n_bootstrap=1000, simultaneous_random_state=None, simultaneous_include_intercept=False, device='auto', n_jobs=None, compute_inference=True, solver='fista', cpu_solver='coordinate_descent', lipschitz_L=None, admm_rho=1.0, gpu_memory_cleanup=False, *, nodewise_alpha=None)
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 平均平方损失下的非负 L1 正则化强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 优化最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | 保存 `coef_delta` / `kkt` 请求，但当前直接拟合忽略此选项，见下文直接拟合限制 |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`；普通 `auto` 解析为 `debiased`。设置 `enable_simultaneous_inference=True` 时应显式使用 `debiased`，构造函数当前拒绝 `auto`。`cpu_ols` 和 `gpu_ols` 暂时作为弃用别名接受 |
| `nodewise_alpha` | `None` | 去偏推断中逐节点 Lasso 的惩罚强度 |
| `n_bootstrap` | `200` | 残差自助法重拟合次数，至少为 2 |
| `bootstrap_random_state` | `None` | 残差自助法随机种子 |
| `enable_simultaneous_inference` | `False` | 是否启用同时推断（仅 `debiased`） |
| `simultaneous_method` | `"maxz_bootstrap"` | 同时推断方法；当前为 `maxz_bootstrap` |
| `simultaneous_alpha` | `0.05` | 同时推断的族错误率水平，必须严格位于 `(0,1)` |
| `simultaneous_n_bootstrap` | `1000` | max-\|Z\| 乘子自助法的抽样次数，必须为正整数 |
| `simultaneous_random_state` | `None` | 同时推断随机种子 |
| `simultaneous_include_intercept` | `False` | 是否把去偏截距纳入同时推断目标集合 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda`（CuPy）/ `torch`（Torch CUDA） |
| `n_jobs` | `None` | 共享 CPU 工作线程设置，不选择求解器，也不保证并行拟合 |
| `compute_inference` | `True` | 是否计算拟合后推断 |
| `solver` | `"fista"` | 与后端无关的直接拟合求解器；CPU 坐标下降使用 `coordinate_descent` |
| `cpu_solver` | `"coordinate_descent"` | **弃用兼容参数**；不再决定直接拟合算法，请改用 `solver` |
| `lipschitz_L` | `None` | 兼容迭代求解器可使用的显式 Lipschitz 常数 |
| `admm_rho` | `1.0` | 当前仅被保存，统一 ADMM 忽略此值，以 rho=1.0 启动。是否自适应调整取决于求解方式；直接 Cholesky 求解保持 rho 不变 |
| `gpu_memory_cleanup` | `False` | 在支持的路径上，拟合后是否请求释放可回收的 GPU 缓存内存 |

`nodewise_alpha` 只能按关键字传入。Lasso 是仅含 L1 惩罚的包装类，
不能照搬 ElasticNet 的完整构造参数：`l1_ratio`、`cov_type`、`hac_maxlags`、
`initial_coef` 都不是 Lasso 的构造/拟合控制参数。它采用默认非稳健推断配置。
目标函数、调参建议与统计解释见[入门页](../models/lasso.md)。

### Lasso 方法与形状

| 方法 | 输入与返回约定 |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | 返回 `self`。数组输入为有限数值 X `(n,p)`、一维 y `(n,)`；可选分析权重 `(n,)` 须有限、非负且总和为正。也可用 formula/data，详见[公式输入](#formula-inputs)。不要混用两种输入；当前公式解析会覆盖数组而不报告冲突。 |
| `predict(X,return_cpu=True)` | X 为按训练特征顺序排列的 `(m,p)`，公式拟合后也可传预测 DataFrame。返回 `(m,)`；默认返回 NumPy，包括 GPU 拟合后的预测。`return_cpu=False` 保留拟合所用 NumPy/CuPy/Torch 数值后端。 |
| `score(X,y,sample_weight=None)` | 评价数据上的 R²，返回 Python 浮点数。y 应为一维 `(m,)`，响应与权重使用 NumPy/主机数据。评价权重不会自动继承训练权重；请自行验证长度 m、有限、非负且总和为正。当前评分可能接受负权重并返回无效 R²。 |
| `summary()` | 打印系数/推断表并返回 `None`；需要成功计算推断，仅完成预测拟合还不够。 |
| `get_params(deep=True)`、`set_params(**params)` | 前者返回构造配置字典，后者更新并返回自身；非空合法更新会重置拟合状态，预测前须重新拟合。详见[参数管理](estimator-api.md)。 |
| `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` | 继承的辅助方法接收显式/缓存 p 值或显式数据。完整签名、参数、返回值与限制见[共享估计器参考](estimator-api.md)。这些方法不会自动重复 Lasso 调参或校正选择不确定性。 |

<a id="lasso-fitted-results"></a>

### Lasso 已拟合结果

拟合截距时 k=p+1，否则 k=p；公式中的截距语法优先于构造设置。
`coef_` 及下列系数报告数组均为 NumPy，包括 GPU 拟合后的结果。

| 结果 | 形状与含义 |
|---|---|
| `coef_`、`intercept_`、`n_iter_` | 惩罚预测斜率 `(p,)`、标量截距（无截距时为零）、迭代次数；均不能证明 KKT 最优性。 |
| `_params`、`_bse`、`_tvalues` / `_zvalues`、`_pvalues`、`_conf_int` | 推断成功后向量为 `(k,)`，区间为 `(k,2)`。有截距时第 0 行是报告截距，其后为斜率；无截距时每行均按特征顺序对应斜率。未计算推断时可能为 `None`。`_params` 可能是去偏或活跃集重拟合参数，不是惩罚预测参数。 |
| `_inference_result` | 包含 `params`、`bse`、`statistic`、`pvalues`、`conf_int`、`method`、`distribution`、`metadata`；据此判断实际程序与目标。 |
| `nodewise_alpha_` | 多特征去偏推断成功后解析出的逐节点惩罚；单特征或其他推断路径为 `None`。 |
| `_conf_int_simultaneous` | 启用去偏同时推断后的联合区间，采用 `(k,2)` 报告布局。有截距但未纳入目标集合时，该行仍为边际区间；纳入时截距参与 max-\|Z\| 校准。普通 `_conf_int` 始终为边际区间。 |
| `inference_requested_method_`、`inference_resolved_method_`、`inference_method_`、`inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_` | 支持的报告路径所记录的方法/目标属性。`post_selection_ols` 成功后部分值仍可能为 `None`，此时读取 `_inference_result.method` 与元数据。 |
| `rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic` | 可用时提供训练诊断。加权去偏推断再次中心化工作响应，可能使 R²/F 总离差不准确；优先用原始数据和已验证权重调用 `score`。似然/AIC/BIC/F 不能普遍解释为考虑惩罚有效自由度或选择后的推断标准；状态缺失时可能返回 `None`/NaN。 |

三种推断方法都保留惩罚预测系数。`post_selection_ols` 在已选列上重拟合作诊断；
`debiased` 修正系数并给出正态参考的边际推断；`bootstrap` 在固定 alpha 下
重采样经验残差、重新拟合完整惩罚设计。均不会自动校正调参/选择不确定性。
自助法仅支持无权重、非稳健的高斯模型推断，构造参数包括 `n_bootstrap` 与
`bootstrap_random_state`。

`fit_intercept=True` 时，同时去偏推断复用拟合所用 NumPy/CuPy/Torch 后端。
结果元数据可包含 `simultaneous_numerical_backend`、`simultaneous_numerical_device`、
`simultaneous_reporting_backend`、`simultaneous_reporting_boundary`。
`fit_intercept=False` 时，同时计算改用 NumPy 主机辅助程序，
即使边际推断在 GPU 上完成也如此。

当前直接拟合的 `stopping` 选项不生效：高斯 CPU FISTA/坐标下降与 GPU FISTA
检查系数变化，ADMM 检查原始/对偶残差。单独的 Lasso CV/路径辅助算法
不能为直接拟合或最终重拟合提供 KKT 认证。`admm_rho` 被保存但被统一 ADMM
忽略，实际以 rho=1.0 启动。是否自适应调整取决于求解方式；平方误差的直接
Cholesky 求解保持 rho 不变，详见 [Lasso 求解限制](../models/lasso.md)。

## ElasticNet

```text
ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=0.0001, stopping='coef_delta', device='auto', n_jobs=None, solver='fista', cpu_solver='fista', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='debiased', cov_type='nonrobust', hac_maxlags=None, *, nodewise_alpha=None)
```
| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `alpha` | `1.0` | 平均平方损失下的非负总正则化强度。 |
| `l1_ratio` | `0.5` | `[0,1]` 内混合比例；0 为 Ridge 目标，1 为 Lasso 目标。 |
| `fit_intercept` | `True` | 不惩罚截距；公式拟合由公式语法控制。 |
| `max_iter` | `1000` | 求解器正整数迭代预算。 |
| `tol` | `1e-4` | 正数收敛容差。 |
| `stopping` | `"coef_delta"` | 保存 `coef_delta` / `kkt` 请求；当前直接高斯拟合忽略此选项，FISTA/坐标下降检查系数变化，ADMM 检查原始/对偶残差。 |
| `device` | `"auto"` | `cpu`/`cuda`（CuPy）/`torch`（Torch CUDA）/`auto`；显式 GPU 请求要求相应后端可用。 |
| `n_jobs` | `None` | 共享 CPU 工作线程配置；不选择求解器，也不保证这些包装类会并行拟合。 |
| `solver` | `"fista"` | 与后端独立的求解器；其他取值依组合而定，见求解器与惩罚兼容矩阵。 |
| `cpu_solver` | `"fista"` | 弃用兼容参数，请使用 `solver`；显式旧参数可能警告或报错，详见迁移指南。 |
| `lipschitz_L` | `None` | 光滑损失梯度 Lipschitz 常数的可选正数上界；通常由求解器估计。 |
| `gpu_memory_cleanup` | `False` | 拟合后尝试清理后端内存池。 |
| `compute_inference` | `False` | 默认只估计与预测。 |
| `inference_method` | `"debiased"` | `debiased`、`post_selection_ols`、`bootstrap`；`auto` 解析为 `debiased`。弃用的 `cpu_ols`/`gpu_ols` 均表示 `post_selection_ols`。 |
| `nodewise_alpha` | `None` | 仅关键字参数：有限正数精度矩阵调参值，或使用设计矩阵自动规则；不改变预测系数。 |
| `cov_type` | `"nonrobust"` | 行为依方法而定，详见下文；纠偏路径不实现 HC/HAC。 |
| `hac_maxlags` | `None` | `post_selection_ols` 的非负 HAC 滞后阶；不会把纠偏推断变成 HAC 推断。 |

| 方法/结果 | 约定 |
|---|---|
| `fit(X=None,y=None,sample_weight=None,initial_coef=None,**kwargs)` | 返回 `self`；`y` 为一维。`kwargs` 接受 `formula=None,data=None`。`initial_coef` 是长度 p 的起始向量；后续拟合省略它时，当前实现仍保留旧向量。需要默认初始化，尤其改变 p 后，请新建估计器。 |
| `predict(X,return_cpu=True)` | `(m,)` 预测；即使 GPU 拟合，默认也返回 NumPy。`False` 保留拟合所用 NumPy/CuPy/Torch 后端。 |
| `score(X,y,sample_weight=None)` | Python 浮点 R²；可选评价权重用于加权均值与残差和，不自动沿用训练权重。须自行检查权重有限、非负、长度为 m 且总和为正：当前路径接受负权重，可能返回大于 1 的无效 R²。使用展平的 y，并把评价 y/权重放在 NumPy/主机上；此平方损失评分路径不提供 GPU 原生响应转换。 |
| `summary()` | 打印系数/推断报告并返回 `None`；要求开启且成功完成推断。 |
| `coef_`、`intercept_`、`n_iter_` | 惩罚预测斜率 `(p,)`、标量截距、迭代次数。达到迭代上限不代表已收敛。 |
| `_params`、`_bse`、`_tvalues`、`_zvalues`、`_pvalues`、`_conf_int` | 报告参数及不确定性，不一定等于预测系数。通常为 `(k,)`，区间为 `(k,2)`，截距优先；具体可用项依推断方法而定。 |
| `_inference_result`、`nodewise_alpha_` | 结构化结果（`params`、`bse`、`statistic`、`pvalues`、`conf_int`、`method`、`distribution`、`metadata`）与多特征时实际使用的逐节点调参值。 |
| `inference_requested_method_`、`inference_resolved_method_`、`inference_method_`、`inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_` | 去偏和 bootstrap 推断会记录公开的方法/目标字段。当前 `post_selection_ols` 即使推断成功，`inference_method_` 与 `inference_target_` 也可能仍为 `None`；应读取 `_inference_result.method` 和 `_inference_result.metadata`。这些属于报告属性，不是构造参数。 |
| `rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic` | 可用的拟合诊断。加权纠偏推断后的 `rsquared`/`rsquared_adj` 使用再次中心化的工作响应，不一定等于原始加权 R²；应验证权重后调用 `score(X,y,sample_weight=weights)`，见[加权诊断](../models/elastic-net.md#加权训练诊断)。AIC/BIC/F 为兼容性代入式汇总，不是通用的惩罚有效自由度或选择性推断准则；状态缺失可能返回 `None`/NaN。 |

<a id="covariance-and-inference-behavior"></a>

### 协方差与推断行为

| `inference_method` | `cov_type`/权重 | 解释 |
|---|---|---|
| `debiased`（及 `auto`） | 使用残差方差和逐节点精度矩阵的模型式公式。当前接受 HC/HAC 设置，但**不会因此改变协方差**，不能据此解释为稳健推断。 | 纠偏系数的正态参考边际区间，依赖设计、稀疏性与模型假设；公开预测系数仍是惩罚估计。 |
| `post_selection_ols` | `nonrobust`、HC0–HC3 或 HAC，可带分析权重 | 在活跃斜率上重拟合 OLS/WLS；经典协方差使用 t 参考，HC/HAC 使用正态参考。不作一般选择调整。 |
| `bootstrap` | 只支持 `cov_type="nonrobust"`，不接受样本权重 | 固定调参，对相同惩罚模型执行残差 bootstrap，返回百分位区间。 |

若需可复现的残差 bootstrap，在 fit **之前**设置 `model.bootstrap_random_state = seed`、`model.n_bootstrap = B`，B≥2，默认 200。这些是额外属性，不是构造/get_params/set_params 参数。ElasticNet 构造函数也不接受 `random_state`。报告方法为 `residual_bootstrap`，参考分布标记为 `bootstrap_percentile`；不会重新运行 CV 或调整惩罚参数。每次重采样都会在完整设计矩阵上重新拟合惩罚模型，因此非零系数集合可能变化；所得区间不代表一般意义上的选择后调整区间。通用 `bootstrap_statistic` 是另一种操作。

## ElasticNetCV

```text
ElasticNetCV(l1_ratio=0.5, *, alphas=None, n_alphas=100, alpha_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, device='auto', n_jobs=None, compute_inference=False, max_iter=1000, tol=0.0001, random_state=None, nodewise_alpha=None)
```
| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `l1_ratio` | `0.5` | `[0,1]` 内标量或候选序列；标量时只选择 alpha。比例为零或接近零时应显式提供 alphas，详见下文网格限制。 |
| `alphas` | `None` | 有限正数候选序列；省略时为每个混合比例生成自动网格。自动规则基于 L1，并非专门的 Ridge 搜索。 |
| `n_alphas` | `100` | `alphas=None` 时自动网格的正整数长度。 |
| `alpha_min_ratio` | `1e-3` | 自动对数网格最小/最大 alpha 的正比例，通常在 `(0,1]` 内。 |
| `cv` | `5` | ≥2 的整数折数；每个训练/验证划分都应有足够观测。 |
| `cv_splits` | `None` | 显式 `(训练索引, 验证索引)` 可迭代对象，替代自动划分。 |
| `fit_intercept` | `True` | 各折拟合和最终重拟合均使用的截距设置。 |
| `device` | `"auto"` | `cpu`/`cuda`（CuPy）/`torch`（Torch CUDA）/`auto`；显式 GPU 请求要求相应后端可用。 |
| `n_jobs` | `None` | 接受的配置项；此类尚未实现候选模型并行计算。 |
| `compute_inference` | `False` | 只在最终全数据重拟合上运行纠偏推断；没有推断方法选择参数。 |
| `max_iter` | `1000` | 单次拟合迭代预算。 |
| `tol` | `1e-4` | 单次拟合收敛容差。 |
| `random_state` | `None` | 自动随机 K 折划分的整数种子。 |
| `nodewise_alpha` | `None` | 只用于最终重拟合的精度矩阵调参，不改变 CV 网格或评分。 |


## LogisticRegressionCV

```text
LogisticRegressionCV(Cs=None, n_Cs=100, C_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, max_iter=100, tol=0.0001, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', gpu_memory_cleanup=False, random_state=None, gpu_cv_mixed_precision=True)
```
| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `Cs` | `None` | 有限正数 C 候选；省略时生成依赖数据的对数网格。`C=0` 不是 CV 候选。 |
| `n_Cs` | `100` | 自动网格正整数长度。 |
| `C_min_ratio` | `1e-3` | 最小/最大 C 的正比例，通常在 `(0,1]` 内。 |
| `cv` | `5` | ≥2 的整数折数；每个训练/验证划分都应有足够观测。 |
| `cv_splits` | `None` | 显式 `(训练索引, 验证索引)` 可迭代对象，替代自动划分。 |
| `fit_intercept` | `True` | 各折拟合和最终重拟合均使用的截距设置。 |
| `max_iter` | `100` | 单次拟合 IRLS 迭代预算。 |
| `tol` | `1e-4` | 单次拟合收敛容差。 |
| `device` | `"auto"` | `cpu`/`cuda`（CuPy）/`torch`（Torch CUDA）/`auto`；显式 GPU 请求要求相应后端可用。 |
| `n_jobs` | `None` | 共享 CPU 工作线程配置；不选择求解器，也不保证这些包装类会并行拟合。 |
| `compute_inference` | `True` | 只在最终重拟合推断，以选定的 C 为条件。 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3`、`hac`；适用性见各模型推断说明。 |
| `gpu_memory_cleanup` | `False` | 拟合后尝试清理后端内存池。 |
| `random_state` | `None` | 自动随机 K 折划分的整数种子。 |
| `gpu_cv_mixed_precision` | `True` | GPU CV 混合精度选项，不改变统计目标函数。 |


<a id="cv-methods-and-results"></a>

## 交叉验证方法与结果
`ElasticNetCV` 与 `LogisticRegressionCV` 均提供 `fit(X,y,sample_weight=None) -> self`、`predict(X)`、`score(X,y)` 及继承的 `summary()`。fit 使用数组/设计矩阵，不接受 `formula`/`data`。`score` 分别为不加权留出 R² 与准确率，**不是**选择时的 CV 损失。ElasticNetCV.predict 通过最终模型的默认行为返回 NumPy；LogisticRegressionCV 另有 `predict_proba(X) -> (m,2)`，使用最终模型后端。其他分类评价方法从 `estimator_` 调用。

自动划分为随机 K 折，不自动分层，也不识别分组/时间结构。此类数据请传入可复用的索引对列表。每组应使用一维整数索引，训练/验证集合均非空、互不重叠，且各自没有重复行。当前共享划分器会转成整数、展平数组并跳过空划分，但不会拒绝重叠或重复索引；无效划分可能把验证行泄漏进训练集。折内权重同时影响训练与验证损失，相关划分的权重总和须为正。预处理只能在每个训练折内学习；这些包装类不接收预处理流水线或评分函数。需要折内标准化时，用外部 CV 循环/流水线，不要先对全部数据标准化再执行内部 CV。

根据平均验证损失最小值选择后，在全部传入训练行上重新拟合。推断只在最终 `estimator_` 上运行，以选定调参为条件，不修正调参不确定性。预测与外层 `coef_`/`intercept_` 属于最终拟合；`summary()` 转交给最终模型。

### 网格与边界行为

请主动提供有限正数网格。当前 CV 会过滤无效/非正 alpha/C；过滤后为空则重新生成自动网格。ElasticNetCV 也会过滤 `[0,1]` 外的 l1_ratio，全部无效时使用 0.5。不要把这些回退当作科学调参方案的验证。直接 LogisticRegression 的特殊 `C=0` 不参与 CV。ElasticNet 自动 alpha 网格可随混合比例变化，应检查返回的映射。

ElasticNetCV 自动网格的最大值取加权中心化 X/y 交叉乘积的最大绝对值，除以
`sum(sample_weight) * max(l1_ratio, 1e-6)`（未传权重时均取 1），并设 `1e-6`
的下限。`l1_ratio=0` 时，这可能产生全部过大的惩罚；即使有用的 Ridge 拟合存在，
预测仍可能几乎为常数。比例为零或接近零时，请显式提供 `alphas`。自动规则还会忽略
`fit_intercept` 设置、始终中心化 X/y，因此无截距拟合也宜使用显式网格。
有限的 CV 评分仅说明在候选范围中选出了较好的值，不证明该范围足够合理。

至少使用四行数据，并保证各折有足够观测。当前 ElasticNetCV 少于四行时不能产生可用结果，应增加数据或使用直接估计器。LogisticRegressionCV 少于四行或只有一个 C 候选时，只完成最终拟合，没有有效 CV 比较；损失数组与 `best_score_` 为 NaN。

### CV 结果

记 r 为混合比例数，a 为最大 alpha 候选数，c 为 C 候选数，f 为折数。

| 类/结果 | 形状与含义 |
|---|---|
| ElasticNetCV `alpha_`、`l1_ratio_` | 选定的标量调参值。 |
| ElasticNetCV `cv_results_` | `mse_path` 为 `(r,a,f)`，`mean_mse`/`std_mse` 为 `(r,a)`；`alphas` 是 `{混合比例索引: alpha数组}` 字典，`l1_ratios` 为 `(r,)`，另有标量 `best_alpha`、`best_l1_ratio`。alpha 轴对应各自保存的网格。 |
| ElasticNetCV `best_score_` | 选定平均 MSE 的**负数**，越大越好，不是正损失或最终模型 R²。 |
| ElasticNetCV `nodewise_alpha_` | 最终模型实际精度矩阵调参值，或 `None`。 |
| LogisticRegressionCV `C_`、`Cs_` | 选定标量与实际 `(c,)` 候选数组。 |
| LogisticRegressionCV `cv_results_` | 只有 `(c,f)` 的 `loss_path`，没有 `mean_loss` 键。 |
| LogisticRegressionCV `mean_loss_`、`best_score_` | `(c,)` 平均对数损失与选定损失的**负数**；不是最终准确率。 |
| 两类：`coef_`、`intercept_`、`n_iter_`、`estimator_`、`cv_selected_device_` | 最终预测斜率 `(p,)`、标量截距、最终迭代数、已拟合直接估计器、最终设备选择。 |

ElasticNetCV 不接受直接拟合的 `solver`、`stopping`、`inference_method`、`cov_type` 参数。LogisticRegressionCV 不接受 `hac_maxlags`，HAC 使用最终估计器的自动滞后规则。不要把直接估计器构造参数整组复制到 CV。

### 可复现的 CPU 调参示例
<!-- api-example: linear-cv -->
```python
import numpy as np
from statgpu.linear_model import ElasticNetCV, LogisticRegressionCV

rng = np.random.default_rng(19)
X = rng.normal(size=(160, 3))
y = 1 + X @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.4, size=160)
enet = ElasticNetCV(
    l1_ratio=[0.3, 0.7], alphas=[0.03, 0.1], cv=3,
    device="cpu", random_state=7,
).fit(X[:120], y[:120])
print(enet.alpha_, enet.l1_ratio_, enet.score(X[120:], y[120:]))
assert np.isclose(enet.best_score_, -np.nanmin(enet.cv_results_["mean_mse"]))

probability = 1 / (1 + np.exp(-X[:, 0]))
y_binary = rng.binomial(1, probability)
logit = LogisticRegressionCV(
    Cs=[0.1, 1.0], cv=3, device="cpu", random_state=7,
    compute_inference=False,
).fit(X[:120], y_binary[:120])
report = logit.estimator_.evaluate_classification(X[120:], y_binary[120:], include_curves=False)
print(logit.C_, report["classification_table"]["accuracy"])
assert set(logit.cv_results_) == {"loss_path"}
assert np.isclose(logit.best_score_, -np.nanmin(logit.mean_loss_))
```

打印的评分使用最后 40 行独立留出数据。断言明确分数符号与结果结构，不保证所有数据都选择同一模型。模拟特征本来就有相同尺度，没有进行全数据学习的预处理。

<a id="formula-inputs"></a>

## 公式输入
本页 LinearRegression、直接 Lasso、ElasticNet、Ridge、SCADRegression 与 MCPRegression 支持公式。这些类均应只传 `formula`/`data`，不要同时传数组 X/y；当前公式解析会直接覆盖这些数组而不报告冲突。需要可选 pandas/patsy 依赖。`formula="y ~ x + C(group)"` 描述数值/分类项；`~ 0 + ...` 去掉截距，不受构造参数覆盖。交互项与转换遵循 Patsy 语法。公式拟合可能删除相关项缺失的行。权重可对应原始全部行或恰好保留的行，按位置对齐，不按任意 Series 标签对齐。

预测 DataFrame 会重建设计矩阵与原有分类水平。未知水平会报错。Lasso/ElasticNet/Ridge/SCAD/MCP 还会拒绝预测缺失值；普通 LinearRegression 和普通 GLM 却可能静默删除预测行，需遵守[缺失行警告](#missing-prediction-rows-in-ordinary-glms)。数组预测则须传入按训练顺序编码好的非截距列。转换与水平定义须保持一致。LogisticRegression、ElasticNetCV、LogisticRegressionCV、RidgeCV 与 LassoCV 没有公式参数，应先构建设计矩阵并防止预处理泄漏。

<!-- api-example: formula-models -->
```python
import numpy as np
import pandas as pd
from statgpu import LinearRegression, ElasticNet, Lasso

df = pd.DataFrame({"x": np.arange(12.0), "group": ["a", "b"] * 6})
df["y"] = 1.0 + 2.0 * df["x"] + (df["group"] == "b").astype(float)
for cls in (LinearRegression, ElasticNet, Lasso):
    model = cls(device="cpu", compute_inference=False)
    model.fit(formula="y ~ x + C(group)", data=df)
    assert model.predict(df.iloc[:3]).shape == (3,)
```

## GeneralizedLinearModel

```text
GeneralizedLinearModel(family='gaussian', fit_intercept=True, max_iter=100, tol=0.0001, C=1.0, device='auto', n_jobs=None, solver='auto', gpu_memory_cleanup=False, compute_inference=False, cov_type='nonrobust')
```

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `family` | `'gaussian'` | 支持 gaussian、binomial、poisson、gamma、inverse_gaussian、negative_binomial、tweedie；特定链接或离散度参数由相应专用模型提供。 |
| `fit_intercept` | `True` | 拟合不受惩罚的截距；公式语法优先。 |
| `max_iter` | `100` | 求解器迭代预算。 |
| `tol` | `0.0001` | 数值收敛容差。 |
| `C` | `1.0` | 普通 IRLS 在 C>0 时加入 ||beta||²/(4C)；C=0 取消惩罚。显式 newton/lbfgs/fista 不使用 C。 |
| `device` | `'auto'` | cpu、cuda（CuPy）、torch（Torch CUDA）或 auto；显式 GPU 请求要求对应 CUDA 后端可用。 |
| `n_jobs` | `None` | 共享 CPU 工作线程设置，不保证拟合并行执行。 |
| `solver` | `'auto'` | auto、irls、fista、newton、lbfgs；普通 auto 选择 IRLS，更换求解器可能改变 C 对应的惩罚目标。 |
| `gpu_memory_cleanup` | `False` | 尽力清理 GPU 内存池。 |
| `compute_inference` | `False` | 启用受支持的 M-估计系数推断。 |
| `cov_type` | `'nonrobust'` | 普通 GLM 推断支持 nonrobust、hc0、hc1；构造器没有 hac_maxlags 参数。 |


`fit(X=None,y=None,sample_weight=None,formula=None,data=None)` 返回 self。X 为有限数值矩阵 `(n,p)`，y 为标量响应 `(n,)`；单列响应会被展平。数组与 formula/data 应二选一：目前同时传入时，公式会直接覆盖数组而不报冲突。Gaussian 默认 identity 链接，binomial 默认 logit，其余上述分布族默认 log 链接。Binomial 要求 0/1 标签，Poisson/负二项响应须非负，Gamma/逆高斯响应须严格为正；Tweedie 默认 power=1.5，允许零值。特定分布族和链接还可能有其他限制。

| 方法/结果 | 约定 |
|---|---|
| `predict(X)` | 返回 `(m,)` 响应均值，设备在预测时解析。Binomial 返回概率，不是类别标签或两列矩阵。公式模式下的 DataFrame 会重建训练设计。 |
| `summary()` | 返回字符串，不主动打印；使用 `print(model.summary())`。未开启推断时仍报告系数与诊断，未拟合时返回提示字符串。 |
| `family_to_loss()` | 返回内部损失名，例如 gaussian → squared_error、binomial → logistic。 |
| `coef_`、`intercept_`、`n_iter_` | NumPy 斜率 `(p,)`、标量截距与迭代次数；迭代次数本身不能证明收敛。 |
| `_bse`、`_zvalues`、`_pvalues`、`_conf_int` | 推断成功后为 `(k,)` 与 `(k,2)`；有截距时排在首位，k=p+1，否则 k=p。 |
| `loglikelihood`、`llf`、`aic`、`bic` | 省略了与参数无关常数的伪似然诊断；加权 loglikelihood 为逐行损失加权平均的 −n 倍。不要跨软件直接比较绝对值，也不要混用响应、观测行或权重不一致的结果。 |

普通 GLM 对受支持的分布族统一使用正态（z）系数推断，包括 Gaussian。它不同于 nonrobust `LinearRegression` 和共享平方误差 L2/Ridge 的 Student-t 路径。可以检查 `model._inference_result.distribution`；`_zvalues` 对应这里的正态参考计算。

这个通用普通模型没有 `score` 或 `predict_proba` 方法。`get_params(deep=True)`、`set_params(**params)`、`adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` 遵循[共享 API](estimator-api.md)；其中重采样辅助方法不会自动重拟合 GLM。

<a id="failed-ordinary-glm-refits"></a>

### 普通 GLM 重拟合失败后的限制

当前 auto/IRLS/FISTA 重拟合报错后，对象可能仍标记为已拟合，却混用旧系数和新的观测数、公式或截距设置。因此，即使新拟合已经报错，预测以及似然/AIC/BIC 仍可能发生变化。不要继续使用该对象的输出；应新建估计器并完成一次成功拟合。共享普通 GLM 实现的专用模型也有此限制。显式 newton/lbfgs 目前会在失败后恢复原拟合状态，但恢复旧状态并不代表新数据拟合成功。


<a id="missing-prediction-rows-in-ordinary-glms"></a>

### 普通线性模型和 GLM 预测时的缺失行

公式拟合的 `LinearRegression`、`GeneralizedLinearModel`、`PoissonRegression`、`GammaRegression`、
`InverseGaussianRegression`、`NegativeBinomialRegression` 和 `TweedieRegression`
目前会删除预测 DataFrame 中公式预测变量缺失的行，并返回较短且不带行标签的
数组。例如五行查询有一行预测变量缺失时，可能只返回四个预测值，不附保留行
索引。不要将这些值按位置配给原始行。应先处理预测变量和变换产生的缺失，
并在关联结果前检查 `len(prediction) == len(query)`。如果有意筛行，应保留
筛选后 DataFrame 的索引，并明确对该 DataFrame 预测。
训练时允许的行过滤及其权重对齐是另一回事。`LinearRegression.score` 还可能
把唯一保留的预测值广播给多个评价响应，返回无效但有限的 R²。展平单列 y
不能解决此问题，评分前应核对公式预测行数。Lasso、ElasticNet、Ridge、
SCADRegression、MCPRegression 和已经测试的惩罚 GLM 专用类会拒绝预测缺失值。

## PenalizedGeneralizedLinearModel

```text
PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l1', alpha=1.0, l1_ratio=0.5, penalty_kwargs=None, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', n_jobs=None, cpu_solver='fista', solver='auto', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None, stopping='coef_delta', lla=True, max_lla_iters=50, lla_tol=1e-06, loss_kwargs=None, *, nodewise_alpha=None)
```

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `loss` | `'squared_error'` | 损失名包括 squared_error、logistic、poisson、gamma、inverse_gaussian、negative_binomial、tweedie 和 quantile。其他分布族及限制见[损失参考](../models/losses.md)。 |
| `penalty` | `'l1'` | none、l1、l2、elasticnet、scad、mcp、adaptive_l1、受支持的分组惩罚，或 Penalty 对象。 |
| `alpha` | `1.0` | 平均损失尺度上的惩罚强度；传入 Penalty 对象时使用该对象自身配置。 |
| `l1_ratio` | `0.5` | elasticnet 中 L1 部分的比例。 |
| `penalty_kwargs` | `None` | 惩罚项的其他构造参数，如分组或形状设置。 |
| `fit_intercept` | `True` | 不受惩罚的截距；公式语法优先。 |
| `max_iter` | `1000` | 单次求解的迭代预算。 |
| `tol` | `0.0001` | 数值容差。 |
| `device` | `'auto'` | cpu、cuda、torch、auto；详见设备指南。 |
| `n_jobs` | `None` | 支持时使用的共享 CPU 工作线程设置。 |
| `cpu_solver` | `'fista'` | 已弃用的兼容参数，不再选择直接拟合算法。显式传入非 None 值（包括历史默认值）会发出 FutureWarning；省略默认值或内部克隆重放不会。应改用 `solver`，详见[迁移指南](../guides/penalized-solver-api-migration.md)。 |
| `solver` | `'auto'` | 候选值包括 auto、fista、fista_bb、admm、irls、newton、lbfgs 和 exact；支持范围取决于损失与惩罚，不支持的显式组合会报错。共享 ADMM 不支持 Quantile，详见[兼容性矩阵](../guides/solver-penalty-matrix.md)。 |
| `lipschitz_L` | `None` | 兼容近端路径的可选 Lipschitz 上界。 |
| `gpu_memory_cleanup` | `False` | 尽力清理 GPU 内存池。 |
| `compute_inference` | `False` | 为 True 时执行受支持的拟合后推断。 |
| `inference_method` | `'auto'` | 按损失与惩罚解析受支持的方法，详见推断矩阵。 |
| `cov_type` | `'nonrobust'` | 协方差选择取决于推断方法；非高斯光滑路径支持 nonrobust/hc0/hc1。 |
| `hac_maxlags` | `None` | 仅在支持 HAC 的路径中控制最大滞后阶数。 |
| `stopping` | `'coef_delta'` | 停止条件请求；当前直接稀疏高斯拟合忽略 kkt 选项。 |
| `lla` | `True` | 为受支持的非凸惩罚启用局部线性近似。 |
| `max_lla_iters` | `50` | LLA 外层最大迭代次数。 |
| `lla_tol` | `1e-06` | LLA 外层收敛容差。 |
| `loss_kwargs` | `None` | 损失特定参数，如 link、负二项离散度 alpha 或 Tweedie power。 |
| `nodewise_alpha` | `None` | 仅限关键字；控制受支持的去偏精度估计，不是拟合惩罚强度。 |


`fit(X=None,y=None,sample_weight=None,formula=None,data=None)` 返回 self；标量响应形状及公式使用规则与上面的普通 GLM 相同。通用构造器没有 `initial_coef`、`n_bootstrap` 或 `bootstrap_random_state` 参数，不应照搬专用封装的控制参数。

`predict(X,return_cpu=True)` 返回 `(m,)`；默认返回 NumPy，False 则保留拟合时的原生后端。平方误差返回线性预测，其他 GLM 返回响应均值，但 logistic 返回 0/1 标签（仅当概率严格大于 0.5 时为 1）。此通用类没有 `predict_proba` 或 `summary`；需要这些附加方法时，应选用适当的专用封装。`score(X,y,sample_weight=None)` 返回响应尺度 R²，logistic 也对预测标签计算 R²，不是准确率或偏差伪 R²。y 应为一维；评估权重须有限、非负且总和为正，当前共享评分路径未完整检查这些条件。

拟合后的 `coef_` `(p,)` 和标量 `intercept_` 用于预测。受支持的推断结果应从 `_inference_result` 及对应数组读取，不要把预测斜率直接视为纠偏或重拟合后的推断参数。`_inference_result.to_dict()` 返回报告字典；`to_dataframe()` 需要 pandas。[推断指南](../guides/penalized-glm-inference.md) 说明各方法、推断目标及限制。上文列出的六个共享配置、p 值和重采样方法同样适用。


## PenalizedGLM_CV

```text
PenalizedGLM_CV(loss='squared_error', penalty='l2', alpha_grid=None, n_alphas=100, l1_ratio=0.5, cv=5, cv_splits=None, random_state=0, device='auto', max_iter=1000, tol=0.0001, solver='auto', cv_strategy='strict', acknowledge_approx=False, refine_top_k=3, loss_kwargs=None, penalty_kwargs=None, *, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None)
```

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `loss` | `'squared_error'` | 标量响应损失；cox_ph 使用独立的生存分析接口。 |
| `penalty` | `'l2'` | 受支持的可调惩罚；none 无需调参，因此会被拒绝。 |
| `alpha_grid` | `None` | 显式有限候选网格；常规可调惩罚要求正值。省略时按损失与惩罚生成。 |
| `n_alphas` | `100` | 自动网格的目标候选数。 |
| `l1_ratio` | `0.5` | 固定的 Elastic Net 混合比例，此类不搜索比例序列。 |
| `cv` | `5` | 自动生成的随机打乱 K 折数量。 |
| `cv_splits` | `None` | 显式、互不重叠、非空的整数训练/验证行索引；一次性迭代器会被保存为可重复使用的划分。 |
| `random_state` | `0` | 自动生成划分的随机种子。 |
| `device` | `'auto'` | 显式 cpu/cuda/torch，或按任务规模自动选择；查看 cv_selected_device_。 |
| `max_iter` | `1000` | 严格折内拟合及最终重拟合的迭代预算。 |
| `tol` | `0.0001` | 严格折内拟合及最终重拟合的容差。 |
| `solver` | `'auto'` | 兼容的求解器请求；开启推断不会更换调参网格。 |
| `cv_strategy` | `'strict'` | strict 或 two_stage；后者先近似筛选，再严格细化。 |
| `acknowledge_approx` | `False` | 确认接受两阶段近似并关闭对应警告。 |
| `refine_top_k` | `3` | 严格细化的优先候选数量；必要时会扩大细化范围。 |
| `loss_kwargs` | `None` | 传入拟合及验证损失的分布族/链接设置。 |
| `penalty_kwargs` | `None` | 惩罚项专用设置。 |
| `compute_inference` | `False` | 仅限关键字；对最终重拟合执行受支持的推断，不对各折执行。 |
| `inference_method` | `'auto'` | 仅限关键字；最终重拟合的推断方法请求。 |
| `cov_type` | `'nonrobust'` | 仅限关键字；最终重拟合的协方差选择。 |
| `hac_maxlags` | `None` | 仅限关键字；仅在最终推断方法支持 HAC 时适用。 |


标量 alpha 网格中的非法值或非正值会被过滤并发出警告；若没有候选值剩下，则警告后自动生成网格。应自行检查搜索范围，不要依赖这一替代行为。

此通用交叉验证类与 ElasticNetCV、LogisticRegressionCV 不同。`fit(X,y,sample_weight=None)` 返回 self，没有公式接口。标量响应 CV 始终拟合截距，此处不提供公开的 `fit_intercept` 选项；若要无截距拟合，应对合适的直接估计器自行建立交叉验证循环。`cox_ph` 分支则遵循[生存分析目标与无截距约定](../models/coxph.md)。

`predict(X)` 与 `score(X,y,sample_weight=None)` 委托给 `estimator_`；标量响应评分是 R²，logistic 也对预测标签计算。`alpha_` 按最小平均验证损失选择，`best_score_` 是该损失的负值，不是 score 返回的留出集 R²。最终推断以所选 alpha 为条件，不校正调参不确定性。

设候选数为 a、折数为 f：`alpha_grid_` 与 `cv_results_["alpha"]` 为 `(a,)`，`mean_score` 为 `(a,)`，`all_scores` 为 `(f,a)`。这些 score 数组存储损失，越小越好。其他键为 `device_sizing_fold_count`、`cv_strategy_`、`cv_selected_device_`、`mean_score_stage1`、`all_scores_stage1`、`refined_mask`。严格模式的第一阶段数组为 None；布尔细化掩码为 `(a,)`。`coef_`、`intercept_` 与 `estimator_` 对应最终全数据拟合。`cv_strategy_` 和 `cv_selected_device_` 也作为属性提供。

当前 `summary(*args,**kwargs)` 在通用最终重拟合推断成功后仍会抛出 AttributeError，因为它委托的估计器没有 summary 方法；未启用推断时则抛出 RuntimeError。可用 `estimator_._inference_result.to_dict()`，或安装 pandas 后用 `to_dataframe()` 读取受支持的推断。报告渲染失败本身不会改变拟合系数。可运行的结果示例见 [GLM 页面](../models/generalized-linear-model.md#reading-cv-inference-results)。共享配置、p 值及重采样方法同样适用；`get_params` 会将一次性自定义划分保存为可复用形式。


## Ridge

```text
Ridge(alpha=1.0, fit_intercept=True, device='auto', n_jobs=None, gpu_memory_cleanup=False, compute_inference=True, cov_type='nonrobust', hac_maxlags=None, max_iter=1000, tol=0.0001, solver='exact', cpu_solver='fista', lipschitz_L=None)
```

每个构造参数及含义均列于[完整 Ridge 参数表](../models/ridge.md#参数)。
此封装类只接受单一响应。默认 `solver="exact"` 拟合 L2 回归；
`compute_inference=True` 启用受支持的高斯协方差推断。
CPU 闭式优化路径存在已说明的[大偏移限制](../models/ridge.md#large-feature-offsets)。

| 方法 | 参数、默认值与返回值 |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | 返回 self。使用有限 X `(n,p)` 和一维 y `(n,)`，或 formula/data。可选分析权重 `(n,)` 须有限、非负且总和为正。公式语法决定截距。不要同时提供数组与公式：当前公式会直接替换数组，不会拒绝冲突。删行对齐及预测 DataFrame 见[公式输入](#formula-inputs)。 |
| `predict(X,return_cpu=True)` | 对 X `(m,p)` 或公式预测 DataFrame 返回 `(m,)`。默认返回 NumPy，包括 GPU 拟合后；False 保留拟合使用的数值后端。 |
| `score(X,y,sample_weight=None)` | 返回 Python 浮点 R²。响应应为一维；响应和权重使用主机/NumPy 数组。评估权重独立于训练权重；当前共享校验不完整，应自行确认权重有限、非负、长度为 m 且总和为正。 |
| `summary()` | 打印系数/推断表并返回 None；关闭或无法计算推断时抛出 RuntimeError。 |
| `get_params(deep=True)`、`set_params(**params)` | 返回配置字典，或更新配置并返回 self。非空有效更新会清除已拟合状态，之后须重新拟合。 |
| `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` | 全部参数、默认值与返回值见[共享估计器参考](estimator-api.md#inference-helpers)。这些方法不会自动重拟合本模型或重复选择、调参。 |

`coef_` 是 NumPy `(p,)` 数组，`intercept_` 为标量（不拟合截距时为零），
`n_iter_` 是迭代次数，不证明全局最优。
成功推断时，`_params`、`_bse`、`_tvalues`、`_pvalues` 形状为 `(k,)`，
`_conf_int` 为 `(k,2)`；拟合截距时 k=p+1，否则 k=p。截距位于首位。
`_inference_result` 记录方法、参考分布与结果元数据。非稳健推断使用 Student-t，
HC/HAC 使用正态参考。缺少所需推断状态时，诊断属性 `rsquared`、`rsquared_adj`、
`fvalue`、`f_pvalue`、`llf`、`aic`、`bic` 可以为 None。这些代入式诊断不是通用的
有效自由度或调参校正准则；稳健协方差也不会把 F 变为稳健 Wald 检验。


## SCADRegression

```text
SCADRegression(alpha=1.0, a=3.7, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', compute_inference=False, solver='auto', gpu_memory_cleanup=False)
```

每个构造参数及含义均列于[完整 SCADRegression 参数表](../models/scad.md#参数)。
应使用 `from statgpu.linear_model import SCADRegression`；顶层 `statgpu`
不导出该类。`alpha` 必须为有限正数，`a`
必须有限且大于 2。请保持 `compute_inference=False`：
此封装类不暴露 `inference_method`，开启推断后拟合会报错。
显式推断应使用[模型页](../models/scad.md#推断)介绍的通用带惩罚线性接口。

| 方法 | 参数、默认值与返回值 |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | 返回 self。使用有限 X `(n,p)` 和一维 y `(n,)`，或 formula/data。可选分析权重 `(n,)` 须有限、非负且总和为正。公式语法决定截距。不要同时提供数组与公式：当前公式会直接替换数组，不会拒绝冲突。删行对齐及预测 DataFrame 见[公式输入](#formula-inputs)。 |
| `predict(X,return_cpu=True)` | 对 X `(m,p)` 或公式预测 DataFrame 返回 `(m,)`。默认返回 NumPy，包括 GPU 拟合后；False 保留拟合使用的数值后端。 |
| `score(X,y,sample_weight=None)` | 返回 Python 浮点 R²。响应应为一维；响应和权重使用主机/NumPy 数组。评估权重独立于训练权重；当前共享校验不完整，应自行确认权重有限、非负、长度为 m 且总和为正。 |
| `summary()` | 打印系数/推断表并返回 None；关闭或无法计算推断时抛出 RuntimeError。 |
| `get_params(deep=True)`、`set_params(**params)` | 返回配置字典，或更新配置并返回 self。非空有效更新会清除已拟合状态，之后须重新拟合。 |
| `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` | 全部参数、默认值与返回值见[共享估计器参考](estimator-api.md#inference-helpers)。这些方法不会自动重拟合本模型或重复选择、调参。 |

`coef_` 是 NumPy `(p,)` 数组，`intercept_` 为标量（不拟合截距时为零），
`n_iter_` 是迭代次数，不证明全局最优。
关闭推断时，`_bse`、`_tvalues`、`_pvalues`、`_conf_int` 与继承的诊断属性
`rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic`
不可用，通常为 None。请用 `score` 或显式留出预测损失评估模型。
不要仅为使用 `summary()` 而开启推断：专用构造函数无法选择受支持的推断方法。


## MCPRegression

```text
MCPRegression(alpha=1.0, gamma=3.0, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', compute_inference=False, solver='auto', gpu_memory_cleanup=False)
```

每个构造参数及含义均列于[完整 MCPRegression 参数表](../models/mcp.md#参数)。
应使用 `from statgpu.linear_model import MCPRegression`；顶层 `statgpu`
不导出该类。`alpha` 必须为有限正数，`gamma`
必须有限且大于 1。请保持 `compute_inference=False`：
此封装类不暴露 `inference_method`，开启推断后拟合会报错。
显式推断应使用[模型页](../models/mcp.md#推断)介绍的通用带惩罚线性接口。

| 方法 | 参数、默认值与返回值 |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | 返回 self。使用有限 X `(n,p)` 和一维 y `(n,)`，或 formula/data。可选分析权重 `(n,)` 须有限、非负且总和为正。公式语法决定截距。不要同时提供数组与公式：当前公式会直接替换数组，不会拒绝冲突。删行对齐及预测 DataFrame 见[公式输入](#formula-inputs)。 |
| `predict(X,return_cpu=True)` | 对 X `(m,p)` 或公式预测 DataFrame 返回 `(m,)`。默认返回 NumPy，包括 GPU 拟合后；False 保留拟合使用的数值后端。 |
| `score(X,y,sample_weight=None)` | 返回 Python 浮点 R²。响应应为一维；响应和权重使用主机/NumPy 数组。评估权重独立于训练权重；当前共享校验不完整，应自行确认权重有限、非负、长度为 m 且总和为正。 |
| `summary()` | 打印系数/推断表并返回 None；关闭或无法计算推断时抛出 RuntimeError。 |
| `get_params(deep=True)`、`set_params(**params)` | 返回配置字典，或更新配置并返回 self。非空有效更新会清除已拟合状态，之后须重新拟合。 |
| `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` | 全部参数、默认值与返回值见[共享估计器参考](estimator-api.md#inference-helpers)。这些方法不会自动重拟合本模型或重复选择、调参。 |

`coef_` 是 NumPy `(p,)` 数组，`intercept_` 为标量（不拟合截距时为零），
`n_iter_` 是迭代次数，不证明全局最优。
关闭推断时，`_bse`、`_tvalues`、`_pvalues`、`_conf_int` 与继承的诊断属性
`rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic`
不可用，通常为 None。请用 `score` 或显式留出预测损失评估模型。
不要仅为使用 `summary()` 而开启推断：专用构造函数无法选择受支持的推断方法。


## RidgeCV

```text
RidgeCV(alphas=None, n_alphas=100, alpha_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', gpu_memory_cleanup=False, random_state=None, gpu_cv_mixed_precision=True)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `alphas` | `None` | 显式有限正数候选项；省略时生成数据相关网格。非法或非正项会被过滤；没有剩余候选项时自动生成网格。 |
| `n_alphas` | `100` | 省略 alphas 时的自动网格大小。 |
| `alpha_min_ratio` | `0.001` | 自动网格最小/最大值比例；应选正数，通常不大于 1。 |
| `cv` | `5` | 自动生成的随机打乱 K 折数量，至少为 2。 |
| `cv_splits` | `None` | 显式、可重复使用的训练/验证索引对列表；应自行确认索引为非空、不相交的整数子集。请参阅下文自定义训练子集的已知问题。 |
| `fit_intercept` | `True` | 在验证折和最终重拟合中拟合截距。 |
| `device` | `'auto'` | cpu、cuda（CuPy）、torch（Torch CUDA）、auto；显式 GPU 请求不可用时会报错。 |
| `n_jobs` | `None` | 共享工作线程配置；不保证候选项并行拟合。 |
| `compute_inference` | `True` | 只在最终全数据重拟合中计算受支持的推断，以已选 alpha 为条件。 |
| `cov_type` | `'nonrobust'` | 最终 Ridge 的协方差：nonrobust、hc0、hc1、hc2、hc3、hac。不接受 hac_maxlags 构造参数；HAC 使用自动规则。 |
| `gpu_memory_cleanup` | `False` | 请求在最终拟合后尽力清理可回收 GPU 缓存。 |
| `random_state` | `None` | 生成验证折的随机种子，不控制残差自助法。 |
| `gpu_cv_mixed_precision` | `True` | 在 GPU 交叉验证中启用混合精度；最终推断使用其自身数值路径。 |

`fit(X,y,sample_weight=None)` 返回 self，接受有限 X `(n,p)`、一维 y `(n,)`
和可选分析权重 `(n,)`，不提供公式接口。`predict(X)` 经最终估计器返回
NumPy `(m,)`。`score(X,y)` 返回不加权 R²，不接受权重参数；如需独立校验后的
评估权重，请用 `estimator_.score(X,y,sample_weight=...)`。
`summary()` 打印最终估计器的报告并返回 None，要求最终推断成功。
共享的 `get_params(deep=True)`、`set_params(**params)`、`adjust_pvalues`、
`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 见[估计器参考](estimator-api.md)。

`alpha_` 按平均验证 MSE 的最小值选择，`best_score_` 是该值的**相反数**，
不是正损失或最终模型 R²。a 个 alpha、f 折时，`alphas_`/`mean_mse_` 为 `(a,)`，
`cv_results_` 只含 `(a,f)` 的 `mse_path`，没有 `cv_results_["mean_mse"]`。
`coef_` `(p,)`、标量 `intercept_`、`n_iter_` 和 `estimator_` 描述全数据重拟合。
推断应从 `estimator_` 读取，以已选 alpha 为条件，不校正调参不确定性。
`cv_selected_device_` 记录最终设备。RidgeCV 不公开 solver、max_iter 或 tol 控制；
最终估计器采用 Ridge 的闭式路径。解释最终拟合时也应注意
[大偏移限制](../models/ridge.md#large-feature-offsets)。

自动生成的验证折是随机打乱的 K 折，不自动按组、分层或时间设计。
指定设计应使用可重复的索引对列表，并在拟合前自行校验；请求被接受不代表
完整校验。各相关加权折都需要正的权重总量。学习式预处理应在各训练折内拟合，
必要时使用外部交叉验证循环。实际调参比较应至少有四行、两个候选项和两个折；
小样本或单候选路径可能只重拟合，损失和 best_score_ 为 NaN。
所选网格仍可能遗漏有用的惩罚范围。

<a id="custom-ridgecv-training-subsets"></a>

### RidgeCV 自定义训练子集的已知问题

这是已知的实现问题（[#243](https://github.com/TheHiddenObserver/statgpu/issues/243)）。
未提供 `sample_weight` 且各验证集恰好覆盖每一行一次时，RidgeCV 当前会用每个验证集
的完整补集替换所给训练索引。因此，即使较小的训练子集合法且与验证集不相交，
也不会被忠实采用；验证损失和 alpha 选择可能改变。不要在此路径中使用刻意
排除或设置间隔的训练行。应改用外部循环，精确地在每个训练子集上拟合 Ridge，
再在对应验证行上评估。普通完整 K 折本来就使用互补训练集，不受这种替换影响。


## LassoCV

```text
LassoCV(alphas=None, n_alphas=12, alpha_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, device='auto', n_jobs=None, compute_inference=False, max_iter=3000, tol=0.0001, stopping='coef_delta', solver='fista', cpu_solver=None, method='standard', cd_kkt_check_every=None, inference_method='post_selection_ols', lipschitz_L=None, admm_rho=1.0, gpu_memory_cleanup=False, random_state=None, gpu_cv_mixed_precision=True, cv_solver='auto', *, nodewise_alpha=None)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `alphas` | `None` | 显式有限正数候选项；省略时生成数据相关网格。非法或非正项会被过滤；没有剩余候选项时自动生成网格。 |
| `n_alphas` | `12` | 省略 alphas 时的自动网格大小。 |
| `alpha_min_ratio` | `0.001` | 自动网格最小/最大值比例；应选正数，通常不大于 1。 |
| `cv` | `5` | 自动生成的随机打乱 K 折数量，至少为 2。 |
| `cv_splits` | `None` | 显式、可重复使用的训练/验证索引对列表；应自行确认索引为非空、不相交的整数子集。 |
| `fit_intercept` | `True` | 在验证折和最终重拟合中拟合截距。 |
| `device` | `'auto'` | cpu、cuda（CuPy）、torch（Torch CUDA）、auto；显式 GPU 请求不可用时会报错。 |
| `n_jobs` | `None` | 共享工作线程配置；不保证候选项并行拟合。 |
| `compute_inference` | `False` | 只在最终全数据重拟合中计算受支持的推断，以已选 alpha 为条件。 |
| `max_iter` | `3000` | 验证折求解和最终直接拟合的迭代预算。 |
| `tol` | `0.0001` | 验证折求解与最终拟合的收敛容差。 |
| `stopping` | `'coef_delta'` | 只传给最终重拟合；当前直接 Gaussian 拟合不采用 kkt 请求，也不借此选择验证路径的停止条件。 |
| `solver` | `'fista'` | 最终全数据 Lasso 的求解器，不选择验证折求解器。 |
| `cpu_solver` | `None` | 已弃用的 CPU 验证求解器别名，请用 cv_solver；CPU 显式冲突会报错。GPU 上发出警告但不替换 FISTA。 |
| `method` | `'standard'` | standard 或 glmnet 验证路径模式。CPU 上 glmnet 要求坐标下降；GPU 验证仍使用 FISTA。它不是推断方法。 |
| `cd_kkt_check_every` | `None` | CPU 验证坐标下降的 KKT 检查间隔，须为正整数。None 在 standard 下为 1，在 glmnet 下为 4；不证明最终重拟合满足 KKT。 |
| `inference_method` | `'post_selection_ols'` | 最终重拟合的 post_selection_ols、debiased、bootstrap 或受支持的 auto 请求；适用性见 Lasso 推断说明。弃用别名会被规范化。 |
| `lipschitz_L` | `None` | 适用最终重拟合的可选 Lipschitz 上界，不控制验证路径。 |
| `admm_rho` | `1.0` | 传给最终 Lasso；当前统一 ADMM 忽略此项，并从 rho=1.0 开始。 |
| `gpu_memory_cleanup` | `False` | 请求在最终拟合后尽力清理可回收 GPU 缓存。 |
| `random_state` | `None` | 生成验证折的随机种子，不控制残差自助法。 |
| `gpu_cv_mixed_precision` | `True` | 在 GPU 交叉验证中启用混合精度；最终推断使用其自身数值路径。 |
| `cv_solver` | `'auto'` | 验证阶段使用 auto、coordinate_descent 或 fista。auto 在 CPU 使用坐标下降，在 GPU 使用 FISTA；显式坐标下降仅支持 CPU。 |
| `nodewise_alpha` | `None` | 仅限关键字的最终去偏精度矩阵调参；不改变网格、验证损失或所选 alpha。 |

`fit(X,y,sample_weight=None)` 返回 self，接受有限 X `(n,p)`、一维 y `(n,)`
和可选分析权重 `(n,)`，不提供公式接口。`predict(X)` 经最终估计器返回
NumPy `(m,)`。`score(X,y)` 返回不加权 R²，不接受权重参数；如需独立校验后的
评估权重，请用 `estimator_.score(X,y,sample_weight=...)`。
`summary()` 打印最终估计器的报告并返回 None，要求最终推断成功。
共享的 `get_params(deep=True)`、`set_params(**params)`、`adjust_pvalues`、
`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 见[估计器参考](estimator-api.md)。

`alpha_` 按平均验证 MSE 的最小值选择，`best_score_` 是该值的**相反数**，
不是正损失或最终模型 R²。a 个 alpha、f 折时，`alphas_`/`mean_mse_` 为 `(a,)`，
`cv_results_` 只含 `(a,f)` 的 `mse_path`，没有 `cv_results_["mean_mse"]`。
`coef_` `(p,)`、标量 `intercept_`、`n_iter_` 和 `estimator_` 描述全数据重拟合。
推断应从 `estimator_` 读取，以已选 alpha 为条件，不校正调参不确定性。
`mse_path_` 也暴露 `(a,f)` 损失数组；`cv_solver_` 记录实际验证算法，
`solver` 只控制最终 Lasso。`nodewise_alpha_` 在适用时暴露最终精度矩阵调参。
源码静态签名可能不含安装后的关键字参数 `nodewise_alpha`；上面的构造器反映
实际公开 API。LassoCV 不暴露直接 Lasso 的同时推断或残差自助法次数/种子构造参数。

自动生成的验证折是随机打乱的 K 折，不自动按组、分层或时间设计。
指定设计应使用可重复的索引对列表，并在拟合前自行校验；请求被接受不代表
完整校验。各相关加权折都需要正的权重总量。学习式预处理应在各训练折内拟合，
必要时使用外部交叉验证循环。实际调参比较应至少有四行、两个候选项和两个折；
小样本或单候选路径可能只重拟合，损失和 best_score_ 为 NaN。
所选网格仍可能遗漏有用的惩罚范围。


<a id="ridgecv-and-lassocv-cpu-example"></a>

## RidgeCV 与 LassoCV 的 CPU 示例

<!-- api-example: ridge-lasso-cv -->
```python
import numpy as np
from statgpu import LassoCV, RidgeCV

rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
models = {}
for cls in (LassoCV, RidgeCV):
    model = cls(
        alphas=[0.03, 0.1, 0.3], cv=3, random_state=7,
        device="cpu", compute_inference=False,
    ).fit(X[:120], y[:120])
    models[cls.__name__] = model
    assert set(model.cv_results_) == {"mse_path"}
    assert model.cv_results_["mse_path"].shape == (3, 3)
    assert np.isclose(model.best_score_, -np.min(model.mean_mse_))
    print(cls.__name__, model.alpha_, round(model.score(X[120:], y[120:]), 3))
```

该数据下两者均选择 alpha=0.03；LassoCV 的留出集 R² 约为 0.965，
RidgeCV 约为 0.966。没有从留出行学习预处理。

<a id="typed-glm-constructors"></a>

## GLM 专用类构造参数

以下七个专用估计器也可从 `statgpu` 导入。下列运行时签名列出各类接受的全部
构造参数。同名共享参数的完整含义见所链接的通用参数表，专用差异逐项说明。
不要把仅属于通用类的参数直接传给专用类。

下列三个固定损失的惩罚类，`loss_kwargs` 仅接受 None 或空字典；通用参数表的
链接、离散参数和幂次示例不适用于 squared_error、logistic 或 Poisson 损失构造器。

四个普通专用类共享 `fit(X=None,y=None,sample_weight=None,formula=None,data=None)`、
`predict(X)`、`summary()`、`family_to_loss()`，以及
[GeneralizedLinearModel](#generalizedlinearmodel) 的配置和推断辅助方法。
该节还定义全部拟合字段、响应均值预测形状、NumPy 报告数组、正态参考推断及
重拟合失败限制。这四个类都没有 `score` 或 `predict_proba`。`family` 由类固定，
不是构造参数；formula/data 属于 fit 参数，沿用相同的训练行对齐规则。
预测时需遵守[普通 GLM 缺失行限制](#missing-prediction-rows-in-ordinary-glms)。

## GammaRegression

```text
GammaRegression(fit_intercept=True, max_iter=100, tol=0.0001, C=1.0, device='auto', n_jobs=None, link='log', solver='auto', compute_inference=False, cov_type='nonrobust', gpu_memory_cleanup=False)
```

除 `link` 外，全部参数及默认值沿用[普通 GLM 参数表](#generalizedlinearmodel)。
`link="log"` 对应均值 `exp(eta)`；`"inverse_power"` 对应 `1/eta`，要求线性
预测子为正。响应必须严格为正。后一链接的显式 Newton/L-BFGS 定义域行为见
[带权 GLM 说明](../models/generalized-linear-model.md)。该构造函数不提供形状、
离散参数或通用 `family` 参数。

## InverseGaussianRegression

```text
InverseGaussianRegression(fit_intercept=True, max_iter=100, tol=0.0001, C=1.0, device='auto', n_jobs=None, solver='auto', compute_inference=False, cov_type='nonrobust', gpu_memory_cleanup=False)
```

全部参数及默认值沿用[普通 GLM 参数表](#generalizedlinearmodel)，但不接受
`family`。该类固定 Inverse Gaussian 分布族和对数链接，响应必须严格为正。
没有公开 `link` 或形状参数；选择 Inverse Gaussian 并不意味着使用其规范的
逆平方链接。

## NegativeBinomialRegression

```text
NegativeBinomialRegression(alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, C=1.0, device='auto', n_jobs=None, solver='auto', compute_inference=False, cov_type='nonrobust', gpu_memory_cleanup=False)
```

共享参数及默认值沿用[普通 GLM 参数表](#generalizedlinearmodel)。额外的
`alpha=1.0` 是**固定、有限且为正的离散参数**，满足
`Var(Y | X) = mu + alpha*mu**2`。它不是正则化强度，也不会被自动估计。
均值采用对数链接，响应非负。`C` 单独控制普通 IRLS 岭惩罚；构造函数不接受
`family` 或 `link`。

## TweedieRegression

```text
TweedieRegression(power=1.5, fit_intercept=True, max_iter=100, tol=0.0001, C=1.0, device='auto', n_jobs=None, solver='auto', compute_inference=False, cov_type='nonrobust', gpu_memory_cleanup=False)
```

共享参数及默认值沿用[普通 GLM 参数表](#generalizedlinearmodel)。额外的
`power=1.5` 必须严格位于 1 和 2 之间，固定复合 Poisson–Gamma 方差幂次，
即 `Var(Y | X) = phi*mu**power`，不会自动调参。响应可以为零或正值，均值
采用对数链接。构造函数不提供 `family`、`link` 或自由的 phi 参数。
该封装不接受某些其他库支持的全部幂次范围。

## PenalizedLinearRegression

```text
PenalizedLinearRegression(penalty='l1', alpha=1.0, l1_ratio=0.5, penalty_kwargs=None, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', n_jobs=None, cpu_solver='fista', solver='auto', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None, stopping='coef_delta', lla=True, max_lla_iters=50, lla_tol=1e-06, loss_kwargs=None, *, nodewise_alpha=None)
```

该类固定 `loss="squared_error"`，不接受 `loss` 参数。全部已列参数的含义
和默认值见[通用惩罚 GLM 参数表](#penalizedgeneralizedlinearmodel)，包括默认
`penalty="l1"` 和仅限关键字的 `nodewise_alpha=None`。响应为单变量高斯模型，
支持公式及分析权重。

它共享 `fit`、`predict(X,return_cpu=True)`、`score(X,y,sample_weight=None)`、
`get_params`、`set_params`、四个继承推断辅助方法及通用系数/结果布局。
额外的 `summary()` 在成功启用推断后打印系数表并返回 None；还提供
`rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic` 诊断属性。
未做推断时这些诊断可能不可用。需遵守[高斯诊断限制](#lasso-fitted-results)，
包括带权纠偏 R² 的问题，以及信息准则不使用一般惩罚有效自由度的限制。
该类没有 `predict_proba`。

## PenalizedLogisticRegression

```text
PenalizedLogisticRegression(penalty='l2', alpha=1.0, l1_ratio=0.5, penalty_kwargs=None, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', n_jobs=None, cpu_solver='fista', solver='auto', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None, stopping='coef_delta', lla=True, max_lla_iters=50, lla_tol=1e-06, loss_kwargs=None)
```

该类固定 `loss="logistic"`，默认 `penalty="l2"`，与通用类的 L1 默认值不同。
全部已列控制项的含义见[通用惩罚 GLM 参数表](#penalizedgeneralizedlinearmodel)，
但不接受 `loss` 或 `nodewise_alpha`。没有 C 参数；alpha 使用平均损失尺度，
不同于独立 LogisticRegression 的求和损失 C 尺度。响应仅支持二元 0/1。

它继承通用惩罚类的 `fit`、`predict(X,return_cpu=True)`、
`score(X,y,sample_weight=None)`、配置/推断辅助方法及拟合结果字段。
`predict` 返回标签，概率严格大于 0.5 才取类别 1。额外方法
`predict_proba(X)` 返回 `(m,2)` 的 0/1 类别概率，使用已拟合的 NumPy/CuPy/Torch
后端，不接受 `return_cpu`。公式拟合后可用 DataFrame 预测。`score` 是标签的
响应尺度 R²，**不是准确率**。该类没有 `summary`、阈值控制、分类指标集合或
绘图接口；受支持的系数推断通过 `_inference_result` 读取。

## PenalizedPoissonRegression

```text
PenalizedPoissonRegression(penalty='l2', alpha=1.0, l1_ratio=0.5, penalty_kwargs=None, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', n_jobs=None, cpu_solver='fista', solver='auto', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None, stopping='coef_delta', lla=True, max_lla_iters=50, lla_tol=1e-06, loss_kwargs=None)
```

该类固定 `loss="poisson"`，默认 `penalty="l2"`。全部已列参数的含义见
[通用惩罚 GLM 参数表](#penalizedgeneralizedlinearmodel)，不接受 `loss`、
`nodewise_alpha` 或 C。alpha 是平均损失尺度的惩罚强度，响应非负，均值使用
对数链接。fit 不提供 offset/exposure 参数。

方法与拟合字段完全沿用通用惩罚类：`fit`、`predict(X,return_cpu=True)`、
`score(X,y,sample_weight=None)`、配置方法、四个继承推断辅助方法及系数/推断
结果布局。`predict` 返回响应均值；`score` 是响应尺度 R²，不是偏差或 Poisson
对数损失。没有 `summary` 或 `predict_proba` 方法。公式/data 和可选分析权重
遵循通用 fit 约定。
