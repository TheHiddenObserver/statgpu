# 有序广义线性模型 (Ordered Logit/Probit)

> 语言: 中文  
> 最后更新: 2026-10-09<br>
> 切换: [English](../../en/models/ordered.md)

## 什么时候使用有序模型？

`OrderedLogitRegression` 与 `OrderedProbitRegression` 适合“低、中、高”等具有次序的类别。它们利用顺序，但不要求相邻类别之间等距。没有自然顺序的类别应使用无序分类模型；连续结果则应保留其连续信息。

模型通过同一组斜率与多个阈值描述累积概率。Logit 采用比例优势假设，Probit 使用标准正态累积分布函数；两者都要求不同阈值共享斜率。这一假设不合适时，简单有序模型可能不够灵活。

## 模型形式

$$
P(y \le j \mid X)=F(\theta_j-X\beta),\qquad j=0,\ldots,K-2.
$$

标签编码为 `0, ..., K-1`，`theta_j` 是严格递增的内部阈值。正斜率使结果更倾向于较高类别；Logit 中 `exp(beta)` 是跨每个阈值的“较高类别对较低类别”优势比，Probit 系数没有这一优势比解释。阈值承担位置参数的作用，不要向 `X` 添加常数列。

<a id="cpu-example"></a>

## 完整 CPU 示例

按顺序运行以下小节，先学习三分类 Logit 的拟合和预测。

### 1. 导入

<!-- learner-example: ordered-basic -->
```python
import numpy as np
from statgpu.linear_model import OrderedLogitRegression
```

### 2. 准备有序响应

`X` 为 `(400, 2)` 数值矩阵，每行是一条观测；`y` 为长度 400 的整数标签。这里把含 Logistic 噪声的潜在连续响应按两个阈值分成 0、1、2 三类。前 300 行训练，后 100 行评价。真实数据应先明确类别顺序、处理缺失值，并保持预测列顺序一致。

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(400, 2))
latent = X @ np.array([0.8, -0.5]) + rng.logistic(size=400)
y = np.digitize(latent, [-0.7, 0.8])
```

### 3. 拟合并读取系数

`n_categories=3` 必须与编码方案对应。暂不计算推断，先查看原始特征尺度的斜率和内部阈值。

```python
model = OrderedLogitRegression(
    n_categories=3, device="cpu", max_iter=200, tol=1e-8,
).fit(X[:300], y[:300])
print("Slopes:", np.round(model.coef_, 3))
print("Thresholds:", np.round(model._thresh_est, 3))
```

斜率约为 `[0.773, -0.458]`，方向与模拟设定一致。它们不是对整数标签求均值的线性回归系数；阈值也不是特征系数。

### 4. 预测类别概率

```python
probability = model.predict_proba(X[300:])
prediction = model.predict(X[300:])
print("First probabilities:", np.round(probability[:3], 3))
print("Held-out accuracy:", round(float(model.score(X[300:], y[300:])), 3))
```

概率数组为 `(100, 3)`，三列依次对应 0、1、2，每行之和为 1。`predict` 选择概率最大的类别，返回 `(100,)` 标签。该种子下准确率约为 `0.54`；准确率把所有误分类同等计数，不能反映错一个等级与错两个等级的不同代价。
<!-- example-end: ordered-basic -->

## 参数选择与常见问题

- 用有实际意义的顺序编码类别；不要按字母顺序自动推定高低。稀少或空类别可能使阈值估计不稳定。
- 检查共享斜率假设、各类别预测概率及留出表现，不能只看训练准确率。
- `n_iter_` 记录迭代次数，不是单独的收敛证明。遇到数值警告时，检查共线性、分离与类别样本量，再考虑增加 `max_iter`。
- 斜率反映条件关联，不自动具有因果解释。

## 目标函数

负对数似然（平均尺度）：

```
NLL = -(1/n) * Σ_i log P(y_i | X_i)
```

其中类别概率为：

```
P(y=k | X) = F(θ_k - Xβ) - F(θ_{k-1} - Xβ)
```

边界约定 `θ_{-1} = -∞`，`θ_{K-1} = ∞`。

## 优化算法

Newton-Raphson + 信赖域正则化（三端统一）：

| 后端 | 算法 | 说明 |
|------|------|------|
| numpy (CPU) | Newton-Raphson + 向量化解析 Hessian | NumPy `linalg.solve` |
| cupy (GPU) | Newton-Raphson + 向量化解析 Hessian | CuPy 数值计算，每次迭代有标量同步 |
| torch (GPU) | Newton-Raphson + 向量化解析 Hessian | Torch 原生，使用 `torch.linalg.solve` |

迭代预算与容差控制数值优化。数值正则化用于稳定 Newton 步，并不是普通 GLM 中由 C 控制的统计斜率惩罚。

**标准化**：X 内部标准化为均值=0，标准差=1。收敛后系数和阈值转换回原始（未标准化）尺度：
`β_raw = β_fit / X_std`，`θ_raw = θ_fit + X_mean @ β_raw`。

## 推断 (Inference)

### Hessian 矩阵

解析观测 Hessian 按总负对数似然计算，虽然优化目标使用平均损失。这一区别决定了协方差中正确的样本量尺度。

Hessian 具有分块结构：

```
H = [ H_{ββ}   H_{βθ} ]
    [ H_{θβ}   H_{θθ} ]
```

- `H_{ββ}` (p × p): 系数的二阶导数
- `H_{βθ}` (p × K-1): 系数与阈值的交叉导数
- `H_{θθ}` (K-1 × K-1): 阈值的二阶导数

### 协方差矩阵

协方差矩阵 = `H^{-1}`（在 MLE 处的观测 Hessian 逆矩阵）。

标准误：`bse = sqrt(diag(H^{-1}))`。

Wald z 统计量：`z = θ / bse`，双侧 p 值使用标准正态分布。

### 属性（设置 `compute_inference=True` 拟合后）

| 属性 | 形状 | 说明 |
|------|------|------|
| `coef_` | (p,) | 原始尺度系数估计值 |
| `_thresh_est` | (K-1,) | 原始尺度阈值估计值（内部使用，不含 -inf/+inf） |
| `thresholds_` | (K+1,) | 完整阈值向量 `[-inf, θ_1, ..., θ_{K-1}, +inf]` |
| `_bse` | (d,) | 标准误：`[bse_coef, bse_thresh]`，`d = p + K - 1`。用 `_bse[:p]` 取系数部分，`_bse[p:]` 取阈值部分 |
| `_zvalues` | (d,) | Wald z 统计量。用 `_zvalues[:p]` / `_zvalues[p:]` 分割 |
| `_pvalues` | (d,) | 双侧 p 值 |
| `_conf_int` | (d, 2) | 95% 置信区间 |
| `loglikelihood` | float | MLE 处的对数似然 |
| `aic` | float | AIC: `-2*loglik + 2*d` |
| `bic` | float | BIC: `-2*loglik + d*log(n)` |
| `n_iter_` | int | Newton-Raphson 迭代次数 |

### 当前限制

- **仅 nonrobust**：`cov_type='nonrobust'` 是唯一支持的协方差类型。
  HC0/HC1 sandwich、bootstrap、惩罚推断尚未支持。
- **不支持 sample_weight**：有序模型不支持样本权重。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_categories` | int | 3 | 序数类别数（>= 2） |
| `fit_intercept` | bool | True | 继承的参数；有序模型由阈值表示位置，不另加截距 |
| `max_iter` | int | 100 | Newton-Raphson 最大迭代数 |
| `tol` | float | 1e-4 | 收敛容差（NLL 绝对变化） |
| `C` | float | 1.0 | 逆正则化强度（未使用；继承自 GLM 基类） |
| `device` | str | 'auto' | 'auto' \| 'cpu' \| 'cuda' \| 'torch' |
| `compute_inference` | bool | False | 拟合后计算 SE、z 值、p 值、CI |
| `cov_type` | str | 'nonrobust' | 协方差估计类型（目前仅 nonrobust） |
| `n_jobs` | int 或 None | None | 共享配置，不用于选择有序求解器 |
| `gpu_memory_cleanup` | bool | False | 拟合后清理 GPU 显存 |

## 推断、Probit 与 GPU

### 可选：Logit 系数推断

先完成 [CPU 示例](#cpu-example)，再复用其中的导入与训练数据。启用 `compute_inference` 后重新拟合，计算基于模型的标准误与边际区间。当前仅支持 `cov_type="nonrobust"`；这些结果依赖模型设定和大样本近似。

<!-- example-requires: ordered-basic -->
<!-- learner-example: ordered-inference -->
```python
inference_model = OrderedLogitRegression(
    n_categories=3, device="cpu", max_iter=200, tol=1e-8,
    compute_inference=True, cov_type="nonrobust",
).fit(X[:300], y[:300])
print("Slope SE:", inference_model._bse[:2])
print("Threshold SE:", inference_model._bse[2:])
print(inference_model.summary())
```
<!-- example-end: ordered-inference -->

前两项对应斜率，后两项对应阈值；`_pvalues`、`_zvalues`、`_conf_int` 采用同一顺序，不能套用普通 GLM 的“截距在前”顺序。`aic` 与 `bic` 可用于相同响应、相同观测上的可比似然模型。

### 更换为 Probit 链接

先完成 [CPU 示例](#cpu-example)，再使用其中的 `X`、`y` 更换链接函数。下面不做推断；如需推断，按上一小节开启并使用同样的结果排列方式。

<!-- example-requires: ordered-basic -->
<!-- learner-example: ordered-probit -->
```python
from statgpu.linear_model import OrderedProbitRegression

probit_model = OrderedProbitRegression(
    n_categories=3, device="cpu", max_iter=200, tol=1e-8,
).fit(X[:300], y[:300])
probit_probability = probit_model.predict_proba(X[300:])
```
<!-- example-end: ordered-probit -->

### 请求 GPU

先完成 [CPU 示例](#cpu-example)，再复用其中的导入、`X` 与 `y`。下面请求 CuPy CUDA；Torch CUDA 使用 `device="torch"`。需要安装对应后端且有可用 CUDA 设备，显式请求不可用设备时会报错。

```python
gpu_model = OrderedLogitRegression(
    n_categories=3, device="cuda", max_iter=200, tol=1e-8,
).fit(X[:300], y[:300])
gpu_probability = gpu_model.predict_proba(X[300:])
```

系数、推断报告数组以及 `predict` / `predict_proba` 输出为 NumPy 数组。计算后端的安装与选择见[设备与内存](../guides/device-and-memory.md)。

## 数值差异与外部比较

该 API 没有独立的严格/近似模式开关。CPU、CuPy 和 Torch 使用相同的模型与解析 Hessian，但浮点运算、容差及问题条件数可能造成数值差异。启用推断后，数值计算使用所选后端，报告数组转换为 NumPy。

与 R `MASS::polr`、`ordinal::clm` 或 statsmodels `OrderedModel` 比较时，应对齐类别顺序、链接、设计矩阵、阈值约定和收敛设置。statgpu 优化平均负对数似然，而 `loglikelihood` 报告总对数似然；在同一参数点，平均负对数似然乘以观测数才对应总负对数似然。比较标准误时使用总负对数似然的观测 Hessian，并注意参数排列和原始尺度转换。

完整构造参数见上表；`fit(X, y)` 返回已拟合对象，`predict_proba` 返回类别概率，`predict` 返回标签，`score` 返回准确率。`summary()`、`aic`、`bic` 与推断属性按各节说明使用。更多签名见[公开实现](../../../statgpu/linear_model/_glm_base.py)与已安装版本的 `help(OrderedLogitRegression)`、`help(OrderedProbitRegression)`。

## 参考文献

- McCullagh, P. (1980). Regression models for ordinal data. *JRSS B*, 42(2), 109–142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley.
- Christensen, R. H. B. (2019). ordinal—Regression Models for Ordinal Data. R package.
