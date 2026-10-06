# LinearRegression

> 语言：中文  
> 最后更新：2026-10-05  
> 页面定位：模型文档  
> 切换：[English](../../en/models/linear-regression.md)

## 它解决什么问题？

当响应变量是连续数值，希望用简单的加性模型预测结果或描述变量间的关联时，
可以从普通最小二乘回归（OLS）开始。一个系数表示：在其他已纳入的预测变量
不变时，该变量增加一个单位，拟合的响应值改变多少。关联本身不等于因果效应。

`LinearRegression` 不使用正则化，适合候选变量不太多、线性关系较合理的场景。
如果变量很多或高度相关，可考虑用 [Ridge](ridge.md) 收缩系数，或用
[Lasso](lasso.md) 得到稀疏系数。二分类响应应使用分类模型，而不是本页的连续响应模型。

## 模型与直觉

带截距的拟合均值为 `intercept_ + X @ coef_`。OLS 寻找使预测误差平方和最小的直线或平面：

$$
\min_{b,\beta}\sum_{i=1}^n (y_i-b-x_i^\top\beta)^2.
$$

n 为观测数，$x_i$ 为 p 维预测变量，$b$ 为截距，$\beta$ 为 p 维斜率。带截距时记 $D=[\mathbf1,X]$、$\theta=(b,\beta^\top)^\top$，否则 $D=X$、$\theta=\beta$。令 $W=\operatorname{diag}(w_i)$，无权重时为单位矩阵，加权目标与正规方程为

$$
\min_\theta (y-D\theta)^\top W(y-D\theta),\qquad
D^\top W(y-D\hat\theta)=0.
$$

经典模型的加权设计矩阵满秩时，记 $r=\operatorname{rank}(W^{1/2}D)$，有

$$
\hat\sigma^2=\frac{(y-D\hat\theta)^\top W(y-D\hat\theta)}{n-r},\qquad
\widehat{\operatorname{Var}}(\hat\theta)=\hat\sigma^2(D^\top WD)^{-1}.
$$

第 j 个标准误是协方差矩阵第 j 个对角元的平方根。边际区间为 $\hat\theta_j\pm c\,\mathrm{SE}_j$，经典推断使用 t 临界值，HC/HAC 使用正态临界值。稳健选项改变协方差构造，不改变 OLS/WLS 拟合。无权重时，r 就是 D 的秩；零权重行可能降低加权设计矩阵的秩。上述逆矩阵表达式要求加权设计满秩且残差自由度为正；使用估计器时无需手工求逆。

## 可直接运行的 CPU 示例

安装 statgpu 后，以下示例只需 NumPy 和本包。它生成三个预测变量，用前 180 行
拟合，留出 60 行测试；真实系数是 `[2, -1, 0]`。

```python
import numpy as np
from statgpu import LinearRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(240, 3))
y = 1.5 + X @ np.array([2.0, -1.0, 0.0]) + rng.normal(scale=0.5, size=240)
X_train, X_test = X[:180], X[180:]
y_train, y_test = y[:180], y[180:]

model = LinearRegression(device="cpu", cov_type="hc3", compute_inference=True)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Intercept:", round(model.intercept_, 3))
print("Coefficients:", np.round(model.coef_, 3))
print("Prediction shape:", prediction.shape)
print("Test R2:", round(model.score(X_test, y_test), 3))
print("95% coefficient intervals:\n", np.round(model._conf_int, 3))
```

输出约为：

```text
Intercept: 1.484
Coefficients: [ 2.049 -0.966 -0.088]
Prediction shape: (60,)
Test R2: 0.934
95% coefficient intervals:
 [[ 1.410  1.557]
  [ 1.971  2.128]
  [-1.038 -0.894]
  [-0.167 -0.008]]
```

前两个斜率分别恢复了正向和负向关联。测试集 $R^2\approx0.934$ 表示测试残差平方和
约为测试响应相对其均值的离差平方和的 6.6%。它不是“模型正确的概率”；在新数据上，
$R^2$ 也可能为负。

区间的行顺序是**截距在先，随后按输入列顺序排列**。注意第三个斜率的真实值为零，
但这次抽样得到的区间恰好没有覆盖零。名义 95% 区间并不保证每次抽样都覆盖真值；
同时检查多个系数还会引入多重检验问题。HC3 改变的是协方差估计，不改变 OLS 拟合系数。
这些是逐个系数的边际置信区间，不是新观测值的预测区间。

## 输入与输出

- 拟合时的数组输入：`X` 为有限数值矩阵，形状为 `(n_samples, n_features)`；`y` 为
  `(n_samples,)` 或 `(n_samples, n_targets)`。`fit` 会将单列 `y` 展平后按单目标处理。
  预测时必须保持训练时的列顺序。
- `fit(X, y, sample_weight=None)` 返回拟合后的估计器。权重必须有限、非负，
  长度为 `n_samples` 且总和大于零。权重改变拟合目标，而不只是标准误。
- 公式输入：也可用 `fit(formula="y ~ x1 + x2", data=df)`，需要可选的
  pandas/patsy 依赖。公式语法决定截距（`~ 0 + ...` 去掉截距）；传入 DataFrame
  预测时会重建保存的设计矩阵。使用公式时需注意解析过程中删除的缺失数据行。
- 单目标：`coef_` 为 `(n_features,)`，`intercept_` 为标量，
  `predict(X_new)` 为 `(n_new,)`。
- 多目标：`coef_` 为 `(n_targets, n_features)`，`intercept_` 为
  `(n_targets,)`，预测值为 `(n_new, n_targets)`。`score` 返回各目标
  $R^2$ 的平均值，而不是分目标的评分数组。

单目标评分时，应向 `score(X, y)` 传入一维响应。与 `fit` 不同，当前 `score`
不会自动展平 `(n_samples, 1)` 响应；它与一维预测值相减时会广播成矩阵，
可能在不报错的情况下给出错误的 $R^2$。因此，评分前应展平单目标响应，
但不要展平真正的多目标数组。

<!-- learner-example: linear-column-target -->
```python
import numpy as np
from statgpu.linear_model import LinearRegression

X = np.arange(6.0)[:, None]
y_column = 2.0 + 3.0 * X
model = LinearRegression(device="cpu", compute_inference=False).fit(X, y_column)
r2 = model.score(X, y_column.ravel())
print("R2:", round(float(r2), 3))
```

输出为 `R2: 1.0`，因为这条直线恰好拟合了全部无噪声观测。

## 如何选择参数与协方差

| 参数 | 默认值 | 选择建议 |
|---|---|---|
| `fit_intercept` | `True` | 通常保留；只有零截距有实际依据时才关闭。不要另加一列常数。公式拟合由公式语法决定截距。 |
| `device` | `"auto"` | 示例使用 `"cpu"`；`"cuda"` 指 CuPy CUDA，`"torch"` 指 Torch CUDA。只有 `"auto"` 允许自动选择其他可用后端。 |
| `n_jobs` | `None` | 此模型接受该共享参数，但当前拟合过程不使用它控制并行任务数。 |
| `compute_inference` | `True` | 只需拟合和预测时可设为 `False`，此时 `summary()` 会报错。 |
| `gpu_memory_cleanup` | `False` | 尽力清理 GPU 内存；详见[设备与内存指南](../guides/device-and-memory.md)。 |
| `cov_type` | `"nonrobust"` | 根据误差结构选择，见下表。 |
| `hac_maxlags` | `None` | HAC 的非负最大滞后阶，结合采样间隔与相关性的持续时间选择；自动规则仅依据样本量。 |

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 经典协方差，使用 t 参考分布；常规 OLS 不确定性计算依赖同方差、独立误差等假设。 |
| `"hc0"` | White 异方差稳健三明治（sandwich）协方差。 |
| `"hc1"` | 在 HC0 基础上进行残差自由度修正。 |
| `"hc2"` | 利用杠杆值调整残差平方。 |
| `"hc3"` | 更强的杠杆值调整，适合关注高影响观测的场景。 |
| `"hac"` | 使用 Bartlett 权重的 Newey-West 协方差，考虑异方差与序列相关；必须保留有意义的观测/时间顺序。 |

HC/HAC 的系数 p 值与区间使用**正态参考分布**，尽管统计量的历史属性名仍是
`_tvalues`。当 `hac_maxlags=None` 时，采用
`floor(4 * (n / 100)**(2 / 9))`，并限制在 `[0, n - 1]` 内。
稳健协方差不能消除遗漏变量、非线性设定错误，也不能自动覆盖协方差模型之外的相关性。
总体诊断 `fvalue`/`f_pvalue` 是基于残差的 F 统计量，不是稳健联合 Wald 检验。

## 常见误区与支持边界

- 检查残差结构，并在留出数据上评价预测。仅看训练集 $R^2$ 不能判断泛化能力。
- 共线性会使单个斜率难以识别。`rank_` 记录拟合设计矩阵的秩；得到数值解并不意味着
  每个系数都有可靠解释。残差自由度非正时推断不可用，`summary()` 会报错。
- CPU 支持多目标推断。CuPy/Torch CUDA 上的多目标拟合需要
  `compute_inference=False`；请求 GPU 多目标推断会抛出 `NotImplementedError`。
  `summary()` 只支持单目标，多目标 `aic`/`bic` 为 `None`。
- 没有单独的公开 strict/approx 推断开关。显式设备请求和不支持的组合仍会明确报错；
  支持的后端之间可能有小幅浮点差异。
- 即使拟合时使用权重，`score(X, y)` 仍是不加权评分，也不接受 `sample_weight`。
  训练集 `rsquared` 属性则在加权拟合后使用拟合权重。
- 常规系数区间不包含利用同一响应数据筛选变量所带来的不确定性；参见
  [特征选择](feature-selection.md)。

### 加权或多目标诊断的限制

单目标加权拟合的 `llf` 当前只把加权残差平方和代入普通高斯表达式。对于独立高斯观测、方差与 `1 / sample_weight` 成比例且权重严格为正的模型，它缺少似然归一化项 `0.5 * sum(log(sample_weight))`。因此，将全部权重乘以同一个常数会改变 `llf`、`aic` 和 `bic`，尽管拟合系数和经典标准误不变。不要跨权重归一化方式比较这些数值，也不要将其视为已正确归一化的 WLS 似然；使用完全相同观测行和权重的模型之间，遗漏项是相同常数。

多目标 `fvalue`、`f_pvalue` 和 `llf` 不是联合多元检验或似然。有权重时，访问 `fvalue` 或 `f_pvalue` 当前会抛出 `TypeError`；无权重时的合并计算也不是各目标的单独 F 检验。需要这些诊断时应逐个目标拟合。训练 `rsquared` 即使在无截距模型中也使用中心化总离差，因此不同于某些参考实现的无截距非中心化 R²。

## API 清单与进阶参考

面向用户的 [LinearRegression API 参考](../reference/linear-model-api.md#linearregression)列出全部构造/方法参数、默认值、形状与返回值。[估计器共享辅助方法](../reference/estimator-api.md#inference-helpers)单独说明继承的推断方法及其与模块函数的区别。

可从 `statgpu` 或 `statgpu.linear_model` 导入。上表列出了全部构造参数。
完整实现与方法说明见
[`LinearRegression`](../../../statgpu/linear_model/wrappers/_linear.py)，
继承方法见 [`BaseEstimator`](../../../statgpu/_base.py)。

| 类别 | 名称与约定 |
|---|---|
| 拟合与报告 | `fit(X=None, y=None, sample_weight=None, formula=None, data=None)`、`predict(X)`、`score(X, y)`、`summary()`（打印表格） |
| 参数管理 | `get_params(deep=True)`、`set_params(**params)` |
| 拟合结果 | `coef_`、`intercept_`、`rank_` |
| 诊断统计 | `rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic` |
| 推断数组 | `_bse`、`_tvalues`、`_pvalues`、`_conf_int`；仅在推断可用时有相应结果 |
| 继承的推断辅助方法 | `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test`；完整参数见[推断 API](../guides/inference-api.md)及基类方法说明 |

单目标、共 `k` 个拟合参数时，前三个推断数组形状为 `(k,)`，`_conf_int` 为
`(k, 2)`。CPU 多目标推断对应为 `(k, n_targets)` 和 `(k, n_targets, 2)`。
拟合截距时，它位于参数首位。上述推断辅助方法不会自动修正变量选择带来的不确定性。

### 与外部实现的对照

[`dev/tests/test_external_consistency.py`](../../../dev/tests/test_external_consistency.py)
包含与 `statsmodels.OLS` 的估计、推断、稳健协方差、GPU 稳健协方差及 HAC 对照，
包括 `test_linear_estimation_and_inference_match_statsmodels`、
`test_linear_robust_covariance_matches_statsmodels`、
`test_linear_robust_covariance_gpu_matches_statsmodels` 和
`test_linear_hac_covariance_matches_statsmodels`。
比较时须对齐设计矩阵/截距、协方差类型、滞后阶和参考分布。

## 参考文献

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838. [DOI](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325. [DOI](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. [DOI](https://doi.org/10.2307/1913610)
