# LogisticRegression

> 语言： 中文  
> 最后更新： 2026-10-09
> 页面定位： 模型文档  
> 切换： [English](../../en/models/logistic-regression.md)

## 什么时候使用这个模型？

`LogisticRegression` 用于估计二分类结果发生的概率，例如某个事件是否发生。它先构造线性预测子，再通过逻辑函数把预测值映射到 0 到 1 之间。既可用于二分类预测，也可用于解释变量与结果的对数优势之间的关联；关联本身不代表因果关系。

这个独立估计器在 CPU/GPU 上采用 IRLS 求解，可加入 L2 正则化。不支持多分类、L1 或 Elastic Net 拟合。需要其他惩罚形式时，参见[带惩罚 GLM](generalized-linear-model.md)，但不要把不同估计器的 `C` 或求解器接口当作相同约定。

<a id="cpu-example"></a>

## 完整的 CPU 示例

按顺序运行以下小节。先拟合和预测，最后再选择是否计算系数区间。示例只需要 CPU。

### 1. 导入

<!-- learner-example: logistic-unpenalized -->
```python
import numpy as np
from statgpu.linear_model import LogisticRegression
```

### 2. 准备二分类数据

`X` 的形状为 `(800, 3)`：每行是一条观测，每列是一个数值特征。`y` 为长度 800 的 0/1 标签。模拟中的概率用于生成响应，实际拟合时不需要已知真实概率。前 600 行用于训练，后 200 行留作评价；预测时保持相同列顺序。

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(800, 3))
true_coef = np.array([0.8, -0.6, 0.3])
p = 1.0 / (1.0 + np.exp(-(-0.2 + X @ true_coef)))
y = rng.binomial(1, p)
```

### 3. 拟合并检查收敛

`C=0` 是这个估计器完全取消惩罚的特殊值；默认 `C=1.0` 会加入 L2 惩罚。先关闭推断，只关注拟合结果。

```python
model = LogisticRegression(
    C=0, device="cpu", compute_inference=False, max_iter=200, tol=1e-8,
).fit(X[:600], y[:600])
print("converged:", model.converged_)
print("coef:", np.round(model.coef_, 3))
print("odds ratios:", np.round(np.exp(model.coef_), 3))
```

该种子下系数约为 `[0.849, -0.501, 0.138]`，优势比约为 `[2.338, 0.606, 1.148]`。其他特征不变时，第一个特征每增加一单位，优势约乘以 2.338；这不是概率比。先确认 `converged_` 为真，不能仅凭存在系数就认为拟合可靠。

### 4. 在留出数据上预测

```python
probability = model.predict_proba(X[600:])
print("P(y=1):", np.round(probability[:3, 1], 3))
print("held-out accuracy:", round(float(model.score(X[600:], y[600:])), 3))
```

`predict_proba` 的一般形状为 `(n_samples, 2)`，本例返回 `(200, 2)` 数组，两列分别是类别 0 和 1 的概率。前三个正类概率约为 `[0.388, 0.306, 0.563]`，准确率约为 `0.65`。`predict` 使用 0.5 阈值，`predict_with_threshold` 可调整阈值。类别不平衡时还应检查精确率、召回率或精确率—召回率曲线。

### 5. 可选：系数不确定性

继续使用上述训练数据，开启推断后重新拟合。这里选择 HC1 得分稳健协方差；其适用假设及正 C 下的解释见后文“协方差与推断”。改变协方差约定不会改变本例的系数。

```python
model.set_params(compute_inference=True, cov_type="hc1")
model.fit(X[:600], y[:600])
print("95% coefficient intervals:", np.round(model._conf_int, 3))
```

`_conf_int` 的形状为 `(4, 2)`，首行为截距，其后按三个输入列的顺序排列；`_bse`、`_zvalues`、`_pvalues` 采用同一顺序。这些是系数的边际区间，不是个体发生事件的概率区间。
<!-- example-end: logistic-unpenalized -->

## 输入与模型定义

导入方式为 `from statgpu.linear_model import LogisticRegression`。
`fit(X, y, sample_weight=None)` 返回已拟合估计器。`X` 应为有限数值组成的 `(n_samples, n_features)` 矩阵，`y` 为形状 `(n_samples,)` 的 0/1 响应。预测时应保持特征列及其顺序一致。数组拟合前先编码分类变量并处理缺失值。

可选的 `sample_weight` 是长度为 n 的有限非负分析权重（analytic weights），权重总和必须为正。未传权重时，下式中各权重均为 1。这个独立估计器最小化**求和尺度**的加权负对数似然与斜率惩罚：

$$
\eta_i=b+x_i^\top\beta,\qquad p_i=\frac{1}{1+\exp(-\eta_i)},
$$

$$
Q(b,\beta)=-\sum_i w_i\{y_i\log p_i+(1-y_i)\log(1-p_i)\}
+\frac{\alpha_C}{2}\|\beta\|_2^2,
\qquad
\alpha_C=\begin{cases}1/C,&C>0,\\0,&C=0.\end{cases}
$$

截距不受惩罚。对于正 `C`，数值越大正则化越弱；很大的 `C` 只是近似无惩罚拟合。`C=0` 是保留的特殊约定，表示完全去掉惩罚，而不是无限强的正则化。这里不是平均损失：当 `C>0` 时，把全部权重同时乘以正常数会改变有效正则强度。与其他接口比较时，应先对齐目标函数尺度，不能直接照搬正则化参数。

## 拟合与估计方程

IRLS 求解以下加权估计方程：

$$
\sum_i w_i x_i(y_i-p_i)-\alpha_C\beta=0,
\qquad \sum_i w_i(y_i-p_i)=0\quad\text{(with intercept)}.
$$

`max_iter` 与 `tol` 控制迭代。默认 `C=1.0` 时，斜率方程含有惩罚项；无惩罚得分方程只适用于 `C=0`，或作为很大 `C` 下的近似。该估计器不提供 `solver` 选择参数。

## 协方差与推断

推断默认使用大样本正态近似（z 统计口径），支持：

- `cov_type="nonrobust"`：`C=0` 时为信息矩阵的逆；`C>0` 时为带惩罚曲率矩阵的逆
- `cov_type="hc0"`：White 三明治（sandwich）稳健协方差
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于杠杆值（leverage）修正
- `cov_type="hc3"`：更保守的 jackknife（留一法）风格修正
- `cov_type="hac"`：Newey-West（Bartlett 核）自相关稳健协方差
- `hac_maxlags`：仅 `cov_type="hac"` 生效

似然、AIC、BIC、伪 R² 与 `converged_` 在 `compute_inference=False` 时仍可用；协方差相关字段不可用。

正 `C` 下的推断围绕带惩罚拟合计算，不是针对无惩罚总体系数的纠偏推断。稳健协方差使用带惩罚曲率矩阵的逆作为三明治协方差估计两侧的矩阵（bread）。这些结果不修正收缩偏差，也不计入选择 `C` 的不确定性。若需要普通无惩罚 Logit 推断，使用 `C=0`，检查可识别性与收敛，并根据数据选择合理的协方差假设。`summary()` 要求 `compute_inference=True`。

按上述求和损失约定，令 $d_i=(1,x_i^\top)^\top$，$D$ 由行 $d_i^\top$ 组成，$P=\operatorname{diag}(0,1,\ldots,1)$；不拟合截距时去掉对应元素。惩罚信息矩阵与 HC0 三明治协方差为

$$
H=D^\top\operatorname{diag}\{w_i p_i(1-p_i)\}D+\alpha_C P,
\qquad
\widehat V_{\mathrm{HC0}}=H^{-1}\left(\sum_i s_i s_i^\top\right)H^{-1},
\quad s_i=w_i d_i(y_i-p_i).
$$

经典协方差为 $H^{-1}$；HC1–HC3 改变得分外积修正，HAC 加入滞后得分乘积。第 j 个系数的 $z_j=\hat\theta_j/\sqrt{\widehat V_{jj}}$，95% 边际区间使用正态 0.975 分位数。这一计算不会把正 C 下的推断变成无惩罚推断。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `C` | `1.0` | 正值为 L2 强度的倒数；`0` 是无惩罚的特殊约定 |
| `max_iter` | `100` | IRLS 最大迭代数 |
| `tol` | `1e-4` | 收敛阈值 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | 基类的 CPU 并行度设置，不用于选择 IRLS 求解器 |
| `compute_inference` | `True` | 是否计算推断统计 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy 内存池 |

## GPU 使用

先完成 [CPU 示例](#cpu-example)，再复用其中的导入与 `X`、`y`。下面用 CuPy CUDA 拟合同一个无惩罚预测模型：

```python
import cupy as cp

X_gpu = cp.asarray(X[:600])
y_gpu = cp.asarray(y[:600])
model_gpu = LogisticRegression(
    C=0, device="cuda", compute_inference=False, max_iter=200, tol=1e-8,
).fit(X_gpu, y_gpu)
p_gpu = model_gpu.predict_proba(cp.asarray(X[600:603]))[:, 1]
```

`device="torch"` 选择 Torch CUDA。显式请求 `cuda` 或 `torch` 需要相应软件包及可用的 CUDA 设备，不会静默回退到 CPU。预测数组使用所选的受支持后端；绘图函数会把结果转为 NumPy 以便渲染。安装与设备选择详见[设备与内存](../guides/device-and-memory.md)。

## 严格与近似模式的差别

该估计器没有单独的严格/近似计算开关。CPU 与 GPU 使用相同的统计定义，但由于浮点运算和底层线性代数实现不同，结果可能存在数值差异，其大小受数据类型、收敛容差及问题条件数影响。

## 输出

- `fit(X, y) -> self`
- 预测：`predict_proba(X)`、`predict(X)`、`predict_with_threshold(X, threshold)`
- 推断属性：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合与信息准则：`loglikelihood`, `loglikelihood_null`, `pseudo_rsquared`, `aic`, `bic`
- 常用属性：`coef_`, `intercept_`, `n_iter_`
- 汇总：`summary()`

分类评估 API：
- `confusion_matrix`、`classification_table`
- `roc_curve`、`roc_auc_score`
- `precision_recall_curve`、`average_precision_score`
- `evaluate_classification`
- `statgpu.metrics.evaluate_binary_classification`
- `plot_roc_curve`、`plot_precision_recall_curve`（依赖 `matplotlib`）

## 参数选择与常见问题

- 当变量单位不应决定 L2 惩罚大小时，先标准化连续特征。只在训练集上估计预处理参数，预测时使用相同转换。
- 用于预测时，利用验证集选择正 `C` 和分类阈值，不要按训练误差最小来调参；最终测试集应单独保留。
- 普通无惩罚推断可使用 `C=0`。完全或近乎完全分离会使无惩罚系数不稳定，甚至不存在有限的极值；只增加迭代次数不能修复分离或共线性。
- `converged_` 为假时，先检查尺度、分离与矩阵秩，再考虑增加 `max_iter` 或放宽 `tol`。不能仅因为推断字段存在，就认为 p 值可靠。
- HC 协方差用于相应的稳健分析；HAC 还依赖观测顺序与滞后阶数。打乱时间顺序会改变 HAC 的含义。
- `summary()` 提示推断不可用时，应以 `compute_inference=True` 重新拟合。关闭推断后仍可使用预测与似然诊断。
- `roc_curve`、`roc_auc_score` 和 `evaluate_classification` 要求评估标签同时包含两类。`include_curves=False` 仅省略曲线数组，组合评估仍会计算 ROC AUC，因而对单类别子集报错；此时可用 `classification_table` 或 `confusion_matrix` 计算阈值指标。
- `precision_recall_curve` 与 `average_precision_score` 至少需要一个正标签；对全零评价子集会抛出 `ValueError`。全一子集可以计算，但此时完美的精确率/AP 不能说明区分两类的能力。相应绘图方法与训练 `average_precision` 属性遵循同一限制。
- 二分类标签必须编码为 0/1。先处理缺失值、非有限输入和形状不匹配，再调用拟合。

## API 参考与验证

分类阈值范围、指标字典、曲线形状与绘图返回值见[完整 LogisticRegression 方法参考](../reference/linear-model-api.md#logisticregression)；[LogisticRegressionCV](../reference/linear-model-api.md#logisticregressioncv)另有构造参数与[选择/结果约定](../reference/linear-model-api.md#cv-methods-and-results)。继承方法见[估计器共享 API](../reference/estimator-api.md)。

上表覆盖构造参数，拟合接口为 `fit(X, y, sample_weight=None)`；继承的 `get_params` / `set_params` 用于估计器配置。`score(X, y)` 返回准确率，`summary()` 显示推断报告。`accuracy`、`precision`、`recall`、`f1`、`auc` 和 `average_precision` 属性描述训练数据；评价泛化表现应在独立数据上调用评估方法。

完整方法签名、返回值与估计器说明见[公开类的 API 源码](../../../statgpu/linear_model/wrappers/_logistic.py)，也可通过 `help(LogisticRegression)` 查看已安装版本的接口。该估计器只支持二分类，推断采用大样本 z 统计量；其 `fit` 不单独提供公式参数。

CPU/GPU 的估计与协方差在对齐设定后与统计参考实现进行数值对照，包括近乎无惩罚时与 `statsmodels.Logit` 的比较。这些比较以相应数据与协方差假设为前提，不是对所有问题的统一精度保证。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

