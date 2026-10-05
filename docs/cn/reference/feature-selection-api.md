# 特征选择 API 参考

> 语言：中文  
> 最后更新：2026-10-05  
> 切换：[English](../../en/reference/feature-selection-api.md)

本页覆盖 `statgpu.feature_selection` 的八个导出：`StepwiseSelector`、`stepwise_selection`、`KnockoffResult`、`knockoff_filter`、`fixed_x_knockoff_filter`、`model_x_knockoff_filter`、`KnockoffSelector`、`FixedXKnockoffSelector`。除 `KnockoffResult` 外均有顶层 `statgpu` 别名。这些选择器不继承 BaseEstimator 的 p 值/bootstrap 辅助方法。

## StepwiseSelector

```python
StepwiseSelector(model_class, criterion='aic', direction='both', max_features=None, n_jobs=None, verbose=False, **model_kwargs)
stepwise_selection(X, y, model_class=LinearRegression, criterion='aic', direction='both', **model_kwargs)
```

`model_class` 应为可调用对象/类；候选拟合需要兼容的 `fit`、AIC/BIC 诊断，以及你后续需要调用的方法。构造参数、全部七个公开方法、选择结果/历史字段、贪心搜索行为、替代评分与限制，完整列于[逐步选择入门页](../models/feature-selection.md#如何选择参数)。

`stepwise_selection` 返回已拟合选择器。关键字可包含 `max_features`、`n_jobs`、`verbose` 等选择器控制以及模型配置；选择器参数由 StepwiseSelector 消费，不会继续转交模型。`fit`/`score` 接受一维响应或会被展平的单列响应。StepwiseSelector 不提供 `fit_transform`、`get_support`、样本权重、公式输入或系数推断接口；转换使用 `fit(...).transform(...)`，索引读取 `selected_features_`。

`get_params(deep=True)` 仍返回扁平字典。`set_params` 把选择器控制项以外的名称
当作包装模型的构造关键字，未知模型参数可能直到 `fit` 才报错。返回选择器不代表
评分全部有效：初始无穷大评分可能阻止接受有限的改进结果；后向搜索失败时，特征数
可能超过 `max_features`。使用所选模型前，应检查准则历史全部有限且满足数量上限，
详见[非有限评分限制](../models/feature-selection.md#nonfinite-score-limitations)。

## Knockoff 函数与构造函数

```python
fixed_x_knockoff_filter(X, y, q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', Xk=None, compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off')

model_x_knockoff_filter(X, y, q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', Xk=None, compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off', modelx_covariance_shrinkage=0.2, modelx_s_scale=0.999, modelx_draws=None, modelx_shrinkage='ledoitwolf', modelx_smatrix_method='mvr', knockpy_sampler=None, knockpy_sampler_method=None)

knockoff_filter(X, y, knockoff_type='fixed_x', q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', Xk=None, compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off', modelx_covariance_shrinkage=0.2, modelx_s_scale=0.999, modelx_draws=None, modelx_shrinkage='ledoitwolf', modelx_smatrix_method='mvr', knockpy_sampler=None, knockpy_sampler_method=None)

KnockoffSelector(knockoff_type='fixed_x', q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off', modelx_covariance_shrinkage=0.2, modelx_s_scale=0.999, modelx_draws=None, modelx_shrinkage='ledoitwolf', modelx_smatrix_method='mvr', knockpy_sampler=None, knockpy_sampler_method=None)

FixedXKnockoffSelector(q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off')
```

fixed-X 函数/类只接受其签名中的共享参数子集。`modelx_*` 与采样器参数用于 model-X 分支，不会改变 fixed-X 结果。函数返回 `KnockoffResult`，选择器构造函数返回未拟合对象。

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `X, y` | `required` | 有限数值 X `(n,p)` 与 y `(n,)`，行数一致。 |
| `knockoff_type` | `"fixed_x"` | 仅统一函数/选择器：`fixed_x` 或 `model_x`。 |
| `q` | `0.1` | `(0,1)` 内目标错误率，不是系数置信水平。 |
| `method` | `"corr_diff"` | `corr_diff`、`ols_coef_diff`、`lasso_coef_diff`，比较原变量与 knockoff 变量重要性。 |
| `fdr_control` | `"knockoff_plus"` | `knockoff_plus` 偏移为 1，`knockoff` 为 0；后者在相应理论下针对不同的修正 FDR 保证。 |
| `random_state` | `None` | 构造或随机统计量拟合的整数种子。 |
| `backend` | `"auto"` | `numpy`、`cupy`、`torch` 或按数组推断的 auto；不是估计器的 `device` 参数。 |
| `Xk` | `None` | 可选外部 `(n,p)` knockoff 矩阵；传给函数或 selector.fit，不传给选择器构造函数。有效性由调用者负责，形状正确不代表可交换性成立。 |
| `compat_mode` | `"statgpu"` | `statgpu` 或 `knockpy`；兼容设置影响构造/统计量约定，可能需要可选包或 CPU 计算。 |
| `lasso_cv_impl` | `"auto"` | `statgpu` 或 `sklearn`；auto 在 knockpy 兼容时选 sklearn，否则选 statgpu。用于 Lasso 统计量。 |
| `lasso_fast_profile` | `"off"` | `off`、`auto`、`moderate`、`aggressive`；Lasso 计算配置。对照分析建议从 off 开始。 |
| `modelx_covariance_shrinkage` | `0.20` | 原生 model-X 协方差收缩比例，应在 `[0,1]` 内。 |
| `modelx_s_scale` | `0.999` | 原生 model-X 的 S 矩阵缩放，通常在 `(0,1]` 内。 |
| `modelx_draws` | `None` | 严格正整数或 None；OLS/Lasso 差默认 5 次，相关差默认 3 次。提供 Xk 时使用该矩阵，不重新抽取多个矩阵。 |
| `modelx_shrinkage` | `"ledoitwolf"` | 兼容协方差策略：`ledoitwolf`、`none`/`mle`、`graphicallasso`/`glasso`；实际选择及回退行为见下文。 |
| `modelx_smatrix_method` | `"mvr"` | 请求的兼容 S 矩阵方法；有 knockpy 时传给它，但可能回退为等相关构造，详见下文。 |
| `knockpy_sampler` | `None` | 可选 gaussian/fx/metro/artk 等分发名；当前分发实现为占位，可能抛出 NotImplementedError。使用已实现构造时保持 None。 |
| `knockpy_sampler_method` | `None` | 高斯分发子方法，如 mvr/sdp/maxent/equi/ci；不能使未实现的采样器可用。 |

### 兼容模式的实际方法与回退

使用 `compat_mode="knockpy"` 的内置 model-X 构造时，协方差估计在本地 CPU 上执行。`none`/`mle` 使用样本协方差，但最小特征值低于内部阈值时会改用 Ledoit–Wolf。Ledoit–Wolf 和图形 Lasso 使用 sklearn；导入失败时改用样本协方差，`metadata["modelx_covariance_estimator"]` 会报告 `"mle_fallback_no_sklearn"`。

S 矩阵构造会尝试调用 knockpy 中请求的方法。包缺失或**该调用抛出任何异常**时，都会回退为等相关构造；因此，方法名无效也可能返回结果而不是报错。`metadata["modelx_smatrix_method"]` 仅记录请求名称。要判断实际执行的协方差/S 矩阵方法，或与 knockpy 作对照，应检查 `modelx_covariance_estimator` 和 `modelx_smatrix_source`（`"knockpy"` 或 `"equicorrelated_fallback"`）。这些回退标记本身不证明 knockoff 有效，也不保证 FDR 控制。

### Lasso 实现的实际选择

当 sklearn 导入失败，或统计量本身使用 Torch 计算时，`lasso_cv_impl="sklearn"`
可能无提示地改用 statgpu。返回的 `metadata["lasso_cv_impl"]` 记录请求值或 auto
初次解析后的值，不一定是实际执行的实现。显式指定 `lasso_cv_impl="statgpu"`
可避免这种歧义。在 knockpy 兼容模式之外，两种实现的截距与交叉验证设置也不同，
不能仅凭统计量名称相同就认为结果数值等价。

<a id="repeated-lasso-statistic-calls"></a>

### 重复计算 Lasso 统计量

指定整数种子的 `method="lasso_coef_diff"` 调用，在原地修改 X、y 或 Xk 后可能
返回旧统计量：统计量缓存及原生 Lasso 调参依据输入的内存标识复用结果，而不是
当前数值。复用会跨越函数调用和选择器实例。数组内存被回收后再次使用也有相同
风险；仅新建选择器或删除旧数组都不能可靠解决。

需要对变化后的数据进行确定性分析时，应每次使用新的 Python 进程。对于**外部
提供且均为 float64 NumPy 数组的 X/y/Xk**，也可为三个输入都创建新副本，
同时保留全部旧输入且不修改它们。
下面的小例子显式保存这些副本；中心化且相互正交的 X/Xk 满足 fixed-X 的 Gram
矩阵约束。大规模分析时，保留旧数组可能消耗较多内存。内部生成 knockoff 时，
调用者不保留临时构造数组，因此应优先采用独立进程。

<!-- api-example: knockoff-fresh-inputs -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter

rng = np.random.default_rng(23)
# Orthogonal, centered pairs satisfy the fixed-X Gram constraints here.
n, p = 80, 4
basis, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
X, Xk = basis[:, 1:p+1], basis[:, p+1:2*p+1]
responses = [5 * X[:, 0], 5 * X[:, 1]]
inputs_kept_alive = []
results = []
for response in responses:
    snapshot = tuple(np.array(a, dtype=np.float64, copy=True) for a in (X, response, Xk))
    inputs_kept_alive.append(snapshot)
    x_fit, y_fit, xk_fit = snapshot
    results.append(fixed_x_knockoff_filter(
        x_fit, y_fit, Xk=xk_fit, method="lasso_coef_diff",
        random_state=17, backend="numpy", lasso_cv_impl="statgpu",
    ))
assert np.argmax(results[0].W) == 0
assert np.argmax(results[1].W) == 1
```

`random_state=None` 会禁用这种按种子复用的行为，但不再保证固定种子的可重复性。
`corr_diff` 和 `ols_coef_diff` 不使用这些 Lasso 缓存。更换统计量会改变方法，
因此应事先选择，而不是看到选择结果后再挑选更合意的统计量。

## 选择器方法

`KnockoffSelector` 与 `FixedXKnockoffSelector` 均提供：

| 方法 | 输入/返回值 |
|---|---|
| `fit(X,y,Xk=None)` | 执行选择并返回 **self**，不是结果对象；设置 `result_` 与从零开始的 NumPy `selected_features_`。 |
| `get_support()` | `(p,)` 布尔 NumPy 掩码；没有 `indices` 参数。 |
| `transform(X)` | 要求原始 `(m,p)` 特征布局，返回 `(m,s)` 所选列并保留 NumPy/CuPy/Torch 输入后端和 dtype；列宽不符报错，s 可以为零。 |
| `fit_transform(X,y,Xk=None)` | 在 X/y 上拟合后返回转换的 X，不是预测或留出评价。 |
| `get_params(deep=True)` | 构造配置字典；这些包装类没有嵌套估计器展开。 |
| `set_params(**params)` | 返回 self；有效非空更新清空拟合选择状态，未知参数报错。 |

当前失败的 `fit` 会保留先前成功的选择结果。重新拟合报错后，不要把 `result_`、`get_support()` 或 `transform()` 视为新数据的结果；请新建选择器并完成一次成功拟合。

这些选择器不拟合响应预测模型，没有 `predict`、`score` 或 `summary`。若要评价预测，应在选中训练列上拟合另一个估计器，再对留出数据使用相同列选择；每个训练折内必须重新选择。

## KnockoffResult

`KnockoffResult` 是报告 dataclass，不是拟合估计器。构造时需要 `knockoff_type`、`selected_features`、`W`、`threshold`、`q`、`estimated_fdr`、`q_trajectory`、`method`、`fdr_control`、`random_state`、`backend`；`metadata` 默认新建空字典。通常从过滤函数或 `selector.result_` 获取，无需自行构造。

| 字段 | 含义 |
|---|---|
| `selected_features` | 从零开始的 NumPy `int64` 索引 `(s,)`，允许空选择。 |
| `W` | `(p,)` NumPy `float64` 特征统计量，即使数值计算使用 GPU；较大正数更支持原始特征。 |
| `threshold` | 选择阈值；无合格阈值时可能为无穷大。 |
| `q`、`estimated_fdr` | 请求目标和阈值规则估计量，不是无法直接得知的实际误发现比例。 |
| `q_trajectory` | 按稳定排序后的每个秩记录字典，键为 `rank`、`threshold`、`fdr_hat`、`n_selected`。`fdr_hat` 上限截为 1，`n_selected` 为正统计量前缀计数且下限为 1；绝对统计量并列时，这些秩诊断不能证明最终整个选择集满足阈值规则。 |
| `knockoff_type`、`method`、`fdr_control`、`random_state`、`backend` | 方法与配置标签。 |
| `metadata` | 数据维数、knockoff 来源、抽样次数等构造/统计量细节；兼容路径应检查实际执行信息。 |
| `to_dict()` | 返回全部字段的字典，把 W/索引转为列表，不重新拟合。 |

## 统计与后端边界

理论 knockoff+ 阈值必须在每个不同的绝对统计量阈值处统计**全部**特征。当前并列 |W| 处理可能使用部分前缀计数，低估阈值处的估计 FDR，随后却选入整个并列组。W=[8,8,-8]、q=0.5 时，可能选择 [0,1] 并报告 estimated_fdr=0.5，而理论完整计数比为 1。此类并列阈值输出不能据此声称名义 FDR 控制，详见[入门页警告](../models/knockoff.md#统计量并列时的限制)。



生成 fixed-X knockoff 要求构造所需的样本量/秩条件，通常 n≥2p，标准化后列满秩。外部 Xk 绕过构造，调用者必须保证其确为匹配且有效的 knockoff 设计。Model-X 使用估计的高斯二阶特征模型：仅匹配估计矩并不能对任意分布提供无条件保证，多次抽样平均也不自动形成独立的 FDR 定理。应结合 [knockoff 指南](../models/knockoff.md)的构造与统计量假设解释结果。

Knockoff 构造与统计量计算把输入转换为 float64；选择器 `transform` 仅选列，保留输入的原始 dtype。存在原生 NumPy/CuPy/Torch 数值路径，但 `lasso_cv_impl="sklearn"` 或部分 `compat_mode="knockpy"` 构造涉及主机转换、CPU 或可选库。`backend` 不能保证兼容步骤全部留在 GPU。提供 Xk 并显式 `lasso_cv_impl="statgpu"` 的原生路径可避开通用 knockpy CPU 构造。不支持的采样分发会报错，不会自动实现另一个采样器。

<a id="runnable-fixed-x-example"></a>

## 可运行的 fixed-X 示例
<!-- api-example: knockoff-selector -->
```python
import numpy as np
from statgpu.feature_selection import FixedXKnockoffSelector

rng = np.random.default_rng(12)
X = rng.normal(size=(120, 5))
y = 3 * X[:, 0] + rng.normal(size=120)
selector = FixedXKnockoffSelector(backend="numpy", random_state=7)
assert selector.fit(X, y) is selector
selected = selector.transform(X[:10])
assert selected.shape == (10, int(selector.get_support().sum()))
assert selector.result_.W.shape == (5,)
print(selector.selected_features_.tolist())
```

小问题可能不选择任何特征，特别是 knockoff+ 配合严格 q 时。这是有效结果，不应在看过结果后只为得到发现而放宽阈值。
