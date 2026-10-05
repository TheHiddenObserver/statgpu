# 线性模型 API 参考

> 语言：中文  
> 最后更新：2026-10-05  
> 切换：[English](../../en/reference/linear-model-api.md)

本页各类可从 `statgpu` 或 `statgpu.linear_model` 导入。`X` 为有限数值 `(n,p)` 矩阵，预测列顺序与训练一致。分析权重为有限非负 `(n,)` 向量，总和须为正。公式删行后的权重对齐见[公式输入](#formula-inputs)。显式 GPU 请求要求对应 CUDA 后端可用，详见[设备与内存](../guides/device-and-memory.md)。

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
| `roc_curve(X,y)` | `(fpr,tpr,thresholds)` 元组，数组等长；阈值递减且首项为无穷大。有意义的 ROC 评价需要两类都出现。 |
| `roc_auc_score(X,y)` | 梯形积分 ROC 面积标量。 |
| `precision_recall_curve(X,y)` | `(precision,recall,thresholds)` 元组；本实现三数组等长，阈值递减，首项无穷大对应 precision=1、recall=0。不要套用其他库的长度约定。 |
| `average_precision_score(X,y)` | 按召回率增量积分的平均精确率标量。 |
| `evaluate_classification(X,y,threshold=0.5,include_curves=True)` | 一次概率计算返回下述指标字典。 |
| `plot_roc_curve(X,y,ax=None,label=None)` | 需要 matplotlib；新建或使用传入 Axes 并返回它。默认标签含 AUC。 |
| `plot_precision_recall_curve(X,y,ax=None,label=None)` | 同样返回 Axes，默认标签含平均精确率。 |
| `summary()` | 打印推断报告，返回 `None`；要求开启且成功完成推断。 |

评估数组/标量在 CPU 使用 NumPy，在 GPU 使用相应 CuPy/Torch 后端；`score` 返回 Python 浮点数。绘图把小型结果转为 NumPy。这些评估方法均不接受 `sample_weight`；加权拟合不代表评估也加权。

`evaluate_classification` 始终返回 `threshold`、`confusion_matrix`、`classification_table`、`roc_auc`、`average_precision`。`include_curves=True` 时再加入 `roc_curve={fpr,tpr,thresholds}` 与 `precision_recall_curve={precision,recall,thresholds}`。对外部概率可调用 `statgpu.metrics.evaluate_binary_classification(y_true,y_score,threshold=0.5,include_curves=True,backend="auto")` 或顶层别名 `statgpu.evaluate_binary_classification`；传入一维类别 1 概率。

训练属性包括 `loglikelihood`、`loglikelihood_null`、`aic`、`bic`、`pseudo_rsquared`、`accuracy`、`precision`、`recall`、`f1`、`auc`、`average_precision`。`pseudo_rsquared` 是 McFadden 的 `1-loglikelihood/loglikelihood_null`，不是 R² 或准确率。推断数组 `_bse`、`_zvalues`、`_pvalues` 为 `(k,)`，`_conf_int` 为 `(k,2)`，截距在首位。C>0 时采用围绕惩罚拟合系数的正态参考推断；关闭推断时这些数组不可用。训练指标不衡量泛化表现。

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
| `stopping` | `"coef_delta"` | `coef_delta` 或 `kkt`；按系数变化或 KKT 条件停止。 |
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
| `inference_requested_method_`、`inference_resolved_method_`、`inference_method_`、`inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_` | 成功推断记录请求/实际方法与条件化目标；属于报告属性，不是构造参数。 |
| `rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic` | 可用的拟合诊断。AIC/BIC/F 为兼容性代入式汇总，不是通用的惩罚有效自由度或选择性推断准则；状态缺失可能返回 `None`/NaN。 |

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
两类均提供 `fit(X,y,sample_weight=None) -> self`、`predict(X)`、`score(X,y)` 及继承的 `summary()`。fit 使用数组/设计矩阵，不接受 `formula`/`data`。`score` 分别为不加权留出 R² 与准确率，**不是**选择时的 CV 损失。ElasticNetCV.predict 通过最终模型的默认行为返回 NumPy；LogisticRegressionCV 另有 `predict_proba(X) -> (m,2)`，使用最终模型后端。其他分类评价方法从 `estimator_` 调用。

自动划分为随机 K 折，不自动分层，也不识别分组/时间结构。此类数据请传入可复用的索引对列表。折内权重同时影响训练与验证损失，相关划分的权重总和须为正。预处理只能在每个训练折内学习；这些包装类不接收预处理流水线或评分函数。需要折内标准化时，用外部 CV 循环/流水线，不要先对全部数据标准化再执行内部 CV。

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
本页只有 LinearRegression 与直接 ElasticNet 支持公式。两者均应只传 `formula`/`data`，不要同时传数组 X/y；当前公式解析会直接覆盖这些数组而不报告冲突。需要可选 pandas/patsy 依赖。`formula="y ~ x + C(group)"` 描述数值/分类项；`~ 0 + ...` 去掉截距，不受构造参数覆盖。交互项与转换遵循 Patsy 语法。公式拟合可能删除相关项缺失的行。权重可对应原始全部行或恰好保留的行，按位置对齐，不按任意 Series 标签对齐。

预测 DataFrame 会重建设计矩阵与原有分类水平。未知水平或导致预测删行的缺失值会报错；数组预测则须传入按训练顺序编码好的非截距列。转换与水平定义须保持一致。LogisticRegression 及两个 CV 包装类没有公式参数，应先构建设计矩阵并防止预处理泄漏。

<!-- api-example: formula-models -->
```python
import numpy as np
import pandas as pd
from statgpu import LinearRegression, ElasticNet

df = pd.DataFrame({"x": np.arange(12.0), "group": ["a", "b"] * 6})
df["y"] = 1.0 + 2.0 * df["x"] + (df["group"] == "b").astype(float)
for cls in (LinearRegression, ElasticNet):
    model = cls(device="cpu", compute_inference=False)
    model.fit(formula="y ~ x + C(group)", data=df)
    assert model.predict(df.iloc[:3]).shape == (3,)
```
