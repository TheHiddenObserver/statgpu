# Adaptive Lasso

> 语言：中文  
> 最后更新：2026-10-09<br>
> 页面定位：模型文档  
> 切换：[English](../../en/models/adaptive-lasso.md)

## 什么时候使用 Adaptive Lasso？

`AdaptiveLasso` 为连续响应拟合稀疏线性模型。普通 [Lasso](lasso.md) 对各个斜率使用同样的惩罚；Adaptive Lasso 则根据初始估计，对较小的初始系数施加较强惩罚，对较大的初始系数施加较弱惩罚。这有助于减少强信号的收缩，但效果取决于初始拟合与特征尺度。需要更简单的基线时先用普通 Lasso；相关特征使选择不稳定时可比较 [Elastic Net](elastic-net.md)。

Oracle 性质是在正则条件与调参条件下成立的渐近结论，不保证有限样本一定找回真实变量，也不代表入选变量显著或存在因果关系。

## 路径

`statgpu.linear_model.AdaptiveLasso`

## 目标函数

$$
\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \alpha \sum_{j=1}^p w_j |\beta_j|
$$

其中 $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$ 为自适应权重，由初始估计（默认为岭回归）计算。

这里 n 为观测数，X 含 p 列特征，b 是不受惩罚的截距，β 为斜率；`fit_intercept=False` 固定 b=0。坐标权重 $w_j$ 用于惩罚特征，不是观测层面的 `sample_weight`。`nu` 越大，惩罚权重对初始系数大小越敏感。用于预测的特征不会自动标准化。

## 完整 CPU 示例

请在同一个 Python 会话中按顺序运行。模拟特征已有相近尺度，这里无需额外学习缩放参数。

<!-- learner-example: adaptive-lasso-prediction -->
```python
import numpy as np
from statgpu.linear_model import AdaptiveLasso
```

<a id="cpu-data"></a>

### 准备观测与留出集

`X` 的形状为 `(160, 5)`，行对应观测、列对应预测变量；`y` 是形状 `(160,)` 的连续响应。只有前两个变量产生信号，最后 40 行不参与训练与调参。

```python
rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]
```

### 拟合自适应惩罚模型

估计器从训练数据计算初始系数与自适应权重。`alpha=0.1`、`nu=1.0` 仅作演示，推断保持关闭。

```python
model = AdaptiveLasso(
    alpha=0.1, nu=1.0, device="cpu", compute_inference=False,
    max_iter=5000, tol=1e-8,
)
model.fit(X_train, y_train)
```

### 预测与解释

测试特征的列顺序必须与训练时相同。结合留出 R² 观察系数的入选情况。

```python
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Selected columns:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("Test R2:", round(model.score(X_test, y_test), 3))
```
<!-- example-end: adaptive-lasso-prediction -->

该随机种子下，斜率约为 `[1.982, -0.957, 0, 0, 0]`，入选列为 `[0, 1]`，留出集 R² 约为 `0.966`。

`coef_` 的形状为 `(5,)`，`intercept_` 是单独的标量，`prediction` 的形状为 `(40,)`。零系数表示本次惩罚拟合排除了对应变量。换一个样本或调参值可能改变入选结果，较高的测试 R² 也不等于确认了科学发现。

## 选择参数并检查稳定性

- 在训练数据内通过验证选择 `alpha`，测试集单独保留。当前没有专用的 Adaptive Lasso 交叉验证类。
- 若同时选择 `nu`，应比较预测表现与入选集合稳定性；每个训练折都要重新计算初始拟合与自适应权重。
- 缩放参数也只能在训练折内学习，再用于验证或测试行；不要在验证之前利用全量数据计算自适应权重。
- 对数值精度有要求时，比较更严格的 `tol` 与更大的 `max_iter`。迭代次数本身不能证明最优性。

## 算法

1. **初始化**：通过岭惩罚坐标下降计算初始系数估计（匹配 R glmnet 的岭求解器）
2. **权重计算**：$w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$，默认 $\nu = 1$
3. **加权 L1 求解**：使用 FISTA 求解加权 Lasso 问题

## Oracle Property

在正则条件下（Zou 2006, Theorem 1）：
- **选择一致性**：$\Pr(\hat{S} = S_0) \to 1$
- **渐近正态性**：$\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

其中 $S_0$ 为真实非零变量集合，$\Sigma_0$ 为 oracle 估计量的极限协方差。结论依赖相应假设和调参条件。

## 协方差与推断限制

请保留默认的 `compute_inference=False`。`adaptive_l1` 不支持内置拟合后推断，开启时会报错，不能得到可用的系数区间；构造参数 `inference_method` 不会解除这一限制。

另行对入选列拟合 OLS 可以描述活跃集重拟合，但使用同一响应得到的常规 OLS 区间没有校正变量选择的不确定性。渐近 oracle 性质并不意味着这些区间普遍具有选择后覆盖保证，详见[推断模式](../guides/inference-modes.md)。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 正则化强度 |
| `nu` | `1.0` | 控制自适应权重对初始系数敏感程度的指数 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `False` | 保持关闭；该惩罚不支持内置推断 |
| `inference_method` | `"debiased"` | 保存推断请求，但不能启用不受支持的自适应 Lasso 推断 |
| `solver` | `"auto"` | 求解器选择 |
| `gpu_memory_cleanup` | `False` | CuPy 内存池清理 |

## 可选 GPU 使用

完成 [CPU 数据准备](#cpu-data)后，复用同一构造方式与训练数组，改用 `device="cuda"` 请求 CuPy CUDA，或 `device="torch"` 请求 Torch CUDA。显式后端不可用时会报错，只有 `auto` 可以选择其他可用后端。推断应继续关闭，详见[设备与内存](../guides/device-and-memory.md)。

## 输出与方法参考

上表列出了全部构造参数。请从 `statgpu.linear_model` 导入 `AdaptiveLasso`。`fit` 返回估计器，`predict` 为每行返回一个响应预测，`score` 返回 R²。预测使用 `coef_` 与 `intercept_`，`n_iter_` 记录数值迭代次数。`summary()` 要求推断，因此不能用于这里仅估计的模型。

继承的 `fit`、公式与权重参数、预测输出位置、加权 `score`、`get_params` 和 `set_params` 见 [PenalizedLinearRegression 方法参考](../reference/linear-model-api.md#penalizedlinearregression)。通用接口中的推断选项不代表自适应惩罚也受支持。

## 与其他实现比较

比较自适应 Lasso 拟合时，应对齐初始估计器及其正则化、自适应权重公式（包括 epsilon 与 `nu`）、特征尺度、截距处理、损失归一化、最终 `alpha` 与收敛精度。普通 Lasso 结果一致，并不能单独验证自适应权重计算。

## 参考文献

- Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418-1429.
- Wang, H., Li, B., & Leng, C. (2009). Shrinkage tuning parameter selection with a diverging number of parameters. *Journal of the Royal Statistical Society: Series B*, 71(3), 671-683.
