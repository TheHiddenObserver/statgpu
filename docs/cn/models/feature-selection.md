# 特征选择

> 语言：中文  
> 最后更新：2026-10-09\
> 切换：[English](../../en/models/feature-selection.md)

## 哪些预测变量应该保留？

当多个测量变量都可能解释响应时，较小的模型更容易检查和使用。
`StepwiseSelector` 每次尝试增加或删除一个预测变量，在信息准则改善时接受改变。
它寻找的是有用的变量子集，并不能证明所选变量导致了响应变化。

候选变量数量适中、模型之间的似然评分可比较时，可以使用逐步选择。
如果目标是对大量相关变量进行预测，可考虑 [Ridge](ridge.md) 或 [Lasso](lasso.md)。
如果主要目标是控制错误发现率（FDR），请参阅 [knockoff 方法](knockoff.md)
及其设计矩阵和分布假设。AIC/BIC 逐步选择本身**不控制 FDR**。

## 搜索优化什么？

对于有似然定义的模型，信息准则在拟合程度与模型大小之间取舍：

$$
\mathrm{AIC}=-2\ell+2k,\qquad
\mathrm{BIC}=-2\ell+k\log n.
$$

其中 $\ell$ 是拟合后的对数似然，$n$ 是观测数，$k$ 是模型的参数计数
（OLS 使用包含截距的拟合设计矩阵秩）。只有在**响应、观测行及模型评分口径可比较**时，
才能认为数值越小越好；这些值不是准确率。在常见样本量下，BIC 通常比 AIC 更强调精简模型。

前向选择从空特征集开始，尝试加入变量；后向选择从全部变量开始，尝试删除变量。
`direction="both"` 从空特征集开始，每一步同时考虑加入和删除。
这是贪心的局部搜索，不会枚举全部子集，也不保证找到全局最优解。

## 可直接运行的 CPU 示例

下面生成六个候选变量，其中只有两个参与生成响应。按顺序运行各段代码，先导入所需对象。

<!-- example: feature-selection-basic -->
```python
import numpy as np
from statgpu import LinearRegression, StepwiseSelector
```

### 准备预测变量并保留测试行

`X` 的形状为 `(240, 6)`，每行是一条观测，每列是一个候选变量。
`y` 的形状为 `(240,)`，每行对应一个连续响应。前 180 行用于训练，
最后 60 行保留到评价时才使用，不参与变量选择。

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(240, 6))
y = 1.5 + 3.0 * X[:, 0] - 2.0 * X[:, 2] + rng.normal(scale=0.5, size=240)
X_train, X_test = X[:180], X[180:]
y_train, y_test = y[:180], y[180:]
```

### 在训练数据上选择变量

用 BIC 评价候选模型，最多保留三个预测变量。`LinearRegression` 是每次候选拟合
使用的模型类。这里关闭系数推断，因为普通重拟合的 P 值不包含变量选择的不确定性。

```python
selector = StepwiseSelector(
    LinearRegression,
    criterion="bic",
    direction="both",
    max_features=3,
    device="cpu",
    compute_inference=False,
).fit(X_train, y_train)
```

### 评价并检查选出的模型

调用选择器时仍保留原始六列。`transform` 返回保留的列，`predict` 和 `score`
会在内部选取这些列。

```python
X_selected = selector.transform(X_test)
prediction = selector.predict(X_test)
print("Selected columns:", selector.selected_features_)
print("Selected test shape:", X_selected.shape)
print("Slopes:", np.round(selector.best_model_.coef_, 3))
print("Test R2:", round(selector.score(X_test, y_test), 3))
```

最后查看 BIC 历史，了解搜索为何接受这些步骤。

```python
print("BIC path:", np.round(selector.bic_history_, 3))
```
<!-- example-end: feature-selection-basic -->

输出约为：

```text
Selected columns: [0, 2]
Selected test shape: (60, 2)
Slopes: [ 2.982 -2.049]
Test R2: 0.981
BIC path: [993.049 768.053 286.428]
```

索引从零开始，指向原矩阵的列。两个斜率依次对应第 0、2 列，接近模拟设定的
3 和 -2。历史记录包含初始的仅截距模型以及每次接受的改变，而不是所有尝试过的子集。
BIC 下降解释了为什么搜索接受这两次加入。本次模拟恢复了真实子集，但其他样本、
相关变量或更弱的信号不一定如此。测试集 $R^2$ 评价的是预测能力，不能据此认定每个
所选变量都是可靠的发现。

## 如何选择参数

| 参数 | 默认值 | 建议 |
|---|---|---|
| `model_class` | 必填 | 传入估计器类或可调用对象，例如 `LinearRegression`，而不是拟合好的实例。每个候选模型都重新拟合。 |
| `criterion` | `"aic"` | `"aic"` 或 `"bic"`；常见样本量下 BIC 更偏向较小模型。 |
| `direction` | `"both"` | `"forward"`、`"backward"` 或 `"both"`；起点与搜索路径不同，可能得到不同子集。 |
| `max_features` | `None` | 期望的特征数量上限，不是必须保留的数量。允许 0 到输入列数之间的整数；`None` 以全部列数为上限。后向搜索遇到非有限评分时可能无法满足上限，详见下文。 |
| `n_jobs` | `None` | 候选评分使用线程；`None`/`1` 为顺序执行，`-1` 按 joblib 约定使用全部可用工作线程。小例子或 GPU 内存有限时宜先顺序执行。 |
| `verbose` | `False` | 打印接受的搜索步骤。 |
| `**model_kwargs` | — | 转交给模型构造函数，例如 `device="cpu"`、`compute_inference=False`。后端与推断能力取决于该模型。 |

候选评分有限时，后向选择先通过删除变量满足 `max_features`，再要求信息准则改善，因此满足上限
期间准则值可能暂时变差。任一方向都允许仅截距/空模型胜出，不强制保留变量。
`max_features=0` 是合法设置。

## 输入、方法与结果

`fit(X, y)` 要求有限数值矩阵 `X`，形状为 `(n_samples, n_features)`，以及
单目标 `y`，形状为 `(n_samples,)`（单列数组会展平）。请传入数组/设计矩阵，
不要直接传入带列名或类别值的原始 DataFrame。选择器的 `fit` 不接受
`sample_weight`、`formula` 或 `data` 参数。

| API | 结果 |
|---|---|
| `fit(X, y)` | 返回 `self`；再次拟合会清空历史与缓存。 |
| `transform(X)` | 按原始列索引的升序返回选中列，形状可能为 `(n, 0)`。拟合后转换可写成 `fit(...).transform(...)`。 |
| `predict(X)` | 内部选择列后调用最终模型；传入时保留原始全部特征布局。 |
| `score(X, y)` | 调用最终模型评分；`LinearRegression` 返回 $R^2$。 |
| `summary()` | 打印准则、搜索方向、特征索引及最终 AIC/BIC，返回 `None`。 |
| `get_params(deep=True)`、`set_params(**params)` | 读取或更新选择器参数和转交的模型参数；成功执行非空 `set_params` 后清空拟合状态。 |
| `selected_features_` | 按升序排列、从零开始的特征列索引列表。 |
| `best_model_` | 在所选列上重新拟合的最终估计器；系数顺序与所选列顺序一致。 |
| `aic_history_`、`bic_history_` | 初始状态及每次接受后的信息准则值。 |
| `selection_history_` | 字典列表，包含 `action`、`feature`、`features`、`aic`、`bic`；初始记录为 `action="initial"`、`feature=None`。 |

调用 `transform`、`predict`、`score` 时，应保留训练时的原始列数与列顺序。
这些方法当前不检查原始列数；更宽的矩阵，或仍包含全部所选位置索引的更窄矩阵，
都可能被直接接受。不要把已经选过列的矩阵再次传给选择器。若列顺序与所选训练列
一致，可以把它直接传给 `best_model_`。

`StepwiseSelector` 和 `stepwise_selection` 可从 `statgpu` 或
`statgpu.feature_selection` 导入。便捷函数
`stepwise_selection(X, y, model_class=LinearRegression, criterion="aic", direction="both", **model_kwargs)`
返回的是**拟合好的选择器**，不是特征索引列表；`max_features` 等选择器控制项也可作为
关键字传入。完整构造函数和方法说明见
[`_stepwise.py`](../../../statgpu/feature_selection/_stepwise.py)。

`get_params(deep=True)` 返回扁平字典，`deep` 不会展开嵌套模型配置。
`set_params` 将选择器控制项以外的名称作为包装模型的构造关键字；
这些关键字直到 `fit` 创建模型时才会验证。因此，拼错模型参数名不一定会在
`set_params` 调用时就报错。

## 常见误区与统计边界

- 交叉验证时，应在每个训练折内重新选择变量。先在全部数据上筛选再切分，会向评价过程
  泄漏响应信息。
- `best_model_` 的普通 p 值不是选择后有效的推断。同一数据已经用于选择变量；
  仅重新拟合 OLS 或事后调整 p 值，不能消除搜索带来的不确定性。
- 高度相关的变量可能互相替代。样本稍有改变，就可能选出不同子集，而预测变化不大。
- 即使不是穷举，候选搜索仍可能需要很多次拟合。可先用领域知识缩小候选范围；
  单次拟合消耗较多 GPU 内存时，尤其应避免过多线程。
- 当所包装的模型是确定性的，选择结果也是确定性的。随机模型应通过其支持的参数固定随机种子。
- 包装模型应提供有限的 `aic` 和 `bic`。否则，若有有限的 `rsquared`，
  选择器会使用基于高斯回归形式的近似评分：AIC 为
  `n * log(max(1 - R2, tiny)) + 2*k`，BIC 改用 `k*log(n)` 惩罚，
  其中 `k = 所选列数 + 是否拟合截距`。这种替代评分不是适用于任意模型族的通用似然准则。
  只有 AIC 和 BIC 都有限时才使用模型提供的准则；即使所选准则本身有限，
  另一项缺失或非有限也会触发上述替代评分。

<a id="nonfinite-score-limitations"></a>

### 非有限评分的限制

当前实现不会在所有评分失效场景下主动报错。初始准则为无穷大时，前向或双向搜索
甚至不会接受评分有限的候选模型。如果所有候选删除操作的评分都非有限，后向搜索
可能返回超过 `max_features` 的特征数。最终模型仍可能拟合成功，不能把返回了选择器
当作搜索成功的证明。

解释结果前，应检查所选准则的历史值是否全部有限，以及选中数量是否满足上限。
任一检查未通过，都应弃用本次搜索结果，改用能够为起始子集及候选子集提供有限、
可比较评分的模型。候选拟合中的部分数值错误会被记为无穷大评分，其他错误继续抛出；
最终重新拟合也可能报错。

## 相关 FDR API 与参考文献

[特征选择 API 参考](../reference/feature-selection-api.md)列出命名空间全部导出、knockoff 完整签名/默认值、选择器方法、`KnockoffResult` 字段与可运行示例。

特征选择命名空间还导出 `KnockoffResult`、`knockoff_filter`、
`fixed_x_knockoff_filter`、`model_x_knockoff_filter`、`KnockoffSelector` 和
`FixedXKnockoffSelector`。完整参数与输出约定见 API 参考；fixed-X 要求以及高斯二阶 model-X 假设见 [knockoff 页面](knockoff.md)。这些方法解决的问题与贪心 AIC/BIC 搜索不同。

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723. [DOI](https://doi.org/10.1109/TAC.1974.1100705)
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464. [DOI](https://doi.org/10.1214/aos/1176344136)
