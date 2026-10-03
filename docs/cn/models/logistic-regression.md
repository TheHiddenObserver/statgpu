# LogisticRegression

> 语言: 中文  
> 最后更新: 2026-10-03
> 页面定位: 模型文档  
> 切换: [English](../../en/models/logistic-regression.md)

## 什么时候使用这个模型？

`LogisticRegression` 用于估计二分类结果发生的概率，例如某个事件是否发生。它先构造线性预测子，再通过逻辑函数把预测值映射到 0 到 1 之间。既可用于二分类预测，也可用于解释变量与结果的对数优势之间的关联；关联本身不代表因果关系。

这个独立估计器在 CPU/GPU 上采用 IRLS 求解，可加入 L2 正则化。不支持多分类、L1 或 Elastic Net 拟合。需要其他惩罚形式时，参见[带惩罚 GLM](generalized-linear-model.md)，但不要把不同估计器的 `C` 或求解器接口当作相同约定。

## 完整的 CPU 示例

下面生成不会完全分离的 0/1 数据，用前 600 行拟合，再评估其余 200 行。`C=0` 表示这个估计器的无惩罚拟合；默认 `C=1.0` 则加入 L2 正则化。

```python
import numpy as np
from statgpu.linear_model import LogisticRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(800, 3))
true_coef = np.array([0.8, -0.6, 0.3])
p = 1.0 / (1.0 + np.exp(-(-0.2 + X @ true_coef)))
y = rng.binomial(1, p)

model = LogisticRegression(
    C=0, device="cpu", cov_type="hc1",
    compute_inference=True, max_iter=200, tol=1e-8,
).fit(X[:600], y[:600])

print("converged:", model.converged_)
print("coef:", np.round(model.coef_, 3))
print("odds ratios:", np.round(np.exp(model.coef_), 3))
print("P(y=1):", np.round(model.predict_proba(X[600:603])[:, 1], 3))
print("held-out accuracy:", round(float(model.score(X[600:], y[600:])), 3))
print("95% coefficient intervals:", np.round(model._conf_int, 3))
```

该随机种子下，四舍五入后的部分输出如下；不同运行环境可能有轻微数值差异：

```text
converged: True
coef: [ 0.849 -0.501  0.138]
odds ratios: [2.338 0.606 1.148]
P(y=1): [0.388 0.306 0.563]
held-out accuracy: 0.65
```

### 如何读取结果？

- `coef_[j]` 表示其他特征不变时，第 j 个特征增加一个单位所对应的对数优势变化。`exp(coef_[j])` 是优势比，不是概率比，也不是概率的直接变化量。
- `predict_proba(X)` 的形状为 `(n_samples, 2)`，两列依次是类别 0 和 1 的概率。`predict(X)` 使用 0.5 阈值；业务需要其他阈值时，可调用 `predict_with_threshold(X, threshold=...)`。
- `score(X, y)` 返回分类准确率。应在留出数据上评估；类别不平衡时，还要查看精确率、召回率或精确率—召回率曲线。
- `_conf_int` 保存 95% 系数置信区间。拟合截距时，第一行及 `_bse`、`_zvalues`、`_pvalues` 的第一个元素对应截距，其余行按输入特征顺序排列。
- 解释系数与推断前先检查 `converged_`，迭代次数见 `n_iter_`。获得已拟合对象不等于已经收敛。

## 输入与模型定义

导入方式为 `from statgpu.linear_model import LogisticRegression`。
`fit(X, y, sample_weight=None)` 返回已拟合估计器。`X` 应为有限数值组成的 `(n_samples, n_features)` 矩阵，`y` 为形状 `(n_samples,)` 的 0/1 响应。预测时应保持特征列及其顺序一致。数组拟合前先编码分类变量并处理缺失值。

可选的 `sample_weight` 是长度为 n 的有限非负解析权重，权重总和必须为正。未传权重时，下式中各权重均为 1。这个独立估计器最小化**求和尺度**的加权负对数似然与斜率惩罚：

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
- `cov_type="hc0"`：White（sandwich）稳健
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于杠杆值（leverage）修正
- `cov_type="hc3"`：更保守的 jackknife（留一法）风格修正
- `cov_type="hac"`：Newey-West（Bartlett 核）自相关稳健协方差
- `hac_maxlags`：仅 `cov_type="hac"` 生效

似然、AIC、BIC、伪 R² 与 `converged_` 在 `compute_inference=False` 时仍可用；协方差相关字段不可用。

正 `C` 下的推断围绕带惩罚拟合计算，不是针对无惩罚总体系数的纠偏推断。稳健协方差使用带惩罚曲率矩阵的逆作为 sandwich 的 bread。这些结果不修正收缩偏差，也不计入选择 `C` 的不确定性。若需要普通无惩罚 Logit 推断，使用 `C=0`，检查可识别性与收敛，并根据数据选择合理的协方差假设。`summary()` 要求 `compute_inference=True`。

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

先运行 CPU 示例的数据生成部分，再用 CuPy 执行相同拟合：

```python
import cupy as cp
from statgpu.linear_model import LogisticRegression

X_gpu = cp.asarray(X[:600])
y_gpu = cp.asarray(y[:600])
model_gpu = LogisticRegression(C=0, device="cuda", cov_type="hc1").fit(X_gpu, y_gpu)
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
- `statgpu.evaluation.evaluate_binary_classification`
- `plot_roc_curve`、`plot_precision_recall_curve`（依赖 `matplotlib`）

## 参数选择与常见问题

- 当变量单位不应决定 L2 惩罚大小时，先标准化连续特征。只在训练集上估计预处理参数，预测时使用相同转换。
- 用于预测时，利用验证集选择正 `C` 和分类阈值，不要按训练误差最小来调参；最终测试集应单独保留。
- 普通无惩罚推断可使用 `C=0`。完全或近乎完全分离会使无惩罚系数不稳定，甚至不存在有限的极值；只增加迭代次数不能修复分离或共线性。
- `converged_` 为假时，先检查尺度、分离与矩阵秩，再考虑增加 `max_iter` 或放宽 `tol`。不能仅因为推断字段存在，就认为 p 值可靠。
- HC 协方差用于相应的稳健分析；HAC 还依赖观测顺序与滞后阶数。打乱时间顺序会改变 HAC 的含义。
- `summary()` 提示推断不可用时，应以 `compute_inference=True` 重新拟合。关闭推断后仍可使用预测与似然诊断。
- 二分类标签必须编码为 0/1。先处理缺失值、非有限输入和形状不匹配，再调用拟合。

## API 参考与验证

上表覆盖构造参数，拟合接口为 `fit(X, y, sample_weight=None)`；继承的 `get_params` / `set_params` 用于估计器配置。`score(X, y)` 返回准确率，`summary()` 显示推断报告。`accuracy`、`precision`、`recall`、`f1`、`auc` 和 `average_precision` 属性描述训练数据；评价泛化表现应在独立数据上调用评估方法。

完整方法签名、返回值与估计器说明见[公开类的 API 源码](../../../statgpu/linear_model/wrappers/_logistic.py)，也可通过 `help(LogisticRegression)` 查看已安装版本的接口。该估计器只支持二分类，推断采用大样本 z 统计量；其 `fit` 不单独提供公式参数。

CPU/GPU 的估计与协方差在对齐设定后与统计参考实现进行数值对照，包括近乎无惩罚时与 `statsmodels.Logit` 的比较。这些比较以相应数据与协方差假设为前提，不是对所有问题的统一精度保证。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

