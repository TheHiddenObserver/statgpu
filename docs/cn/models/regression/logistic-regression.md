# LogisticRegression

> 语言： 中文  
> 最后更新： 2026-10-14
> 页面定位： 模型文档  
> 切换： [English](../../../en/models/regression/logistic-regression.md)

## 什么时候使用这个模型？

`LogisticRegression` 用于估计二分类结果发生的概率，例如某个事件是否发生。它先构造线性预测子，再通过逻辑函数把预测值映射到 0 到 1 之间。既可用于二分类预测，也可用于解释变量与结果的对数优势之间的关联；对数优势就是 $p/(1-p)$ 的对数，其中 $p$ 是事件发生概率；关联本身不代表因果关系。

这个独立估计器在 CPU/GPU 上提供 IRLS 或 L-BFGS 求解，可加入 L2 正则化。默认 `solver="auto"` 选择 IRLS。`LogisticRegressionCV` 通过交叉验证选择正数 `C`，再使用全部传入的训练数据重拟合。两者均不支持多分类、L1 或 Elastic Net 拟合。需要其他惩罚形式时，参见[带惩罚 GLM](generalized-linear-model.md)，但不要把不同估计器的 `C` 或求解器接口当作相同约定。

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

该种子下系数约为 `[0.849, -0.501, 0.138]`，优势比约为 `[2.338, 0.606, 1.148]`。优势（odds）指发生概率与不发生概率之比，即 $p/(1-p)$。其他特征不变时，第一个特征增加一单位，会让这个比值乘以约 2.338。例如起始概率为 0.5 时，优势为 1；乘以 2.338 后，对应概率为 $2.338/(1+2.338)\approx0.70$，而不是把概率 0.5 直接乘以 2.338。不同起始概率对应的概率增量不同。先确认 `converged_` 为真，不能仅凭存在系数就认为拟合可靠。

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

截距不受惩罚。对于正 `C`，数值越大正则化越弱；很大的 `C` 只是近似无惩罚拟合。`C=0` 是保留的特殊约定，表示完全去掉惩罚，而不是无限强的正则化。负数或非有限的 `C` 无效。这里不是平均损失：当 `C>0` 时，把全部权重同时乘以正常数会改变有效正则强度。与其他接口比较时，应先对齐目标函数尺度，不能直接照搬正则化参数。

## 拟合与估计方程

IRLS 与 L-BFGS 求解相同的加权估计方程：

$$
\sum_i w_i x_i(y_i-p_i)-\alpha_C\beta=0,
\qquad \sum_i w_i(y_i-p_i)=0\quad\text{(with intercept)}.
$$

默认 `C=1.0` 时，斜率方程含有惩罚项；无惩罚得分方程只适用于 `C=0`，或作为很大 `C` 下的近似。在这两种求解器之间切换，不会改变目标函数或 `C` 的含义。

### 求解器选择与收敛

- `solver="auto"` 解析为 `"irls"`，保持默认拟合算法。
- `solver="irls"` 使用迭代加权最小二乘法，`tol` 检查参数步长的欧几里得范数。
- `solver="lbfgs"` 使用有限内存 BFGS。它将**整个目标函数（包括惩罚项）**除以样本数 n，传入权重时则除以 `sum(sample_weight)`，再将缩放后目标函数的梯度欧几里得范数与 `tol` 比较。参数步长很小本身不能证明收敛。
- `max_iter` 限制所选求解器的迭代次数。请检查 `converged_` 和 `n_iter_`；未达到停止准则时会发出 `ConvergenceWarning`。线搜索失败不会被报告为已收敛。
- `get_params()["solver"]` 保留用户指定值；拟合后的 `solver_` 记录 `"irls"` 或 `"lbfgs"`。开启推断时，`summary()` 显示指定/实际求解器与收敛状态。不支持的求解器名称会引发 `ValueError`。

整体缩放不改变最优解，但会改变梯度的数值尺度。因此，同一 `tol` 在 IRLS 和 L-BFGS 下不代表相同精度或迭代次数。合理缩放特征有助于两种算法收敛。L-BFGS 不必在每次优化迭代中求解 Hessian 线性系统；开启协方差推断仍需要计算曲率。

### CPU 示例：L-BFGS 与 C 选择

下面的独立示例可单独运行。它在前 300 行上拟合带权模型并选择 `C`，剩下 100 行用于预测。示例特征已有接近的尺度；若预处理需要从数据学习参数，应在每个 CV 训练折内分别拟合。

<!-- api-example: logistic-lbfgs -->
```python
import numpy as np
from statgpu.linear_model import LogisticRegression, LogisticRegressionCV

rng_lbfgs = np.random.default_rng(42)
X_lbfgs = rng_lbfgs.normal(size=(400, 4))
p_lbfgs = 1 / (1 + np.exp(-(0.3 + X_lbfgs @ [0.8, -0.5, 0.2, 0.0])))
y_lbfgs = rng_lbfgs.binomial(1, p_lbfgs)
weights_lbfgs = rng_lbfgs.uniform(0.5, 1.5, size=300)

model_lbfgs = LogisticRegression(
    solver="lbfgs", device="cpu", C=1.0,
    max_iter=300, tol=1e-7, compute_inference=False,
).fit(X_lbfgs[:300], y_lbfgs[:300], sample_weight=weights_lbfgs)
print(model_lbfgs.get_params()["solver"], model_lbfgs.solver_, model_lbfgs.converged_)
probabilities_lbfgs = model_lbfgs.predict_proba(X_lbfgs[300:303])
print(probabilities_lbfgs)
assert model_lbfgs.solver_ == "lbfgs" and model_lbfgs.converged_
assert probabilities_lbfgs.shape == (3, 2)
assert np.isfinite(probabilities_lbfgs).all()
assert np.allclose(probabilities_lbfgs.sum(axis=1), 1.0)

cv_lbfgs = LogisticRegressionCV(
    Cs=[0.1, 1.0, 10.0], cv=3, random_state=42,
    solver="lbfgs", device="cpu", max_iter=300, tol=1e-7,
    compute_inference=False,
).fit(X_lbfgs[:300], y_lbfgs[:300], sample_weight=weights_lbfgs)
print(cv_lbfgs.C_, cv_lbfgs.solver_, cv_lbfgs.estimator_.converged_)
print(cv_lbfgs.mean_loss_)
assert cv_lbfgs.solver_ == cv_lbfgs.estimator_.solver_ == "lbfgs"
assert cv_lbfgs.estimator_.converged_
assert cv_lbfgs.mean_loss_.shape == (3,)
assert cv_lbfgs.cv_results_["loss_path"].shape == (3, 3)
assert np.isfinite(cv_lbfgs.cv_results_["loss_path"]).all()
assert np.isclose(cv_lbfgs.best_score_, -np.min(cv_lbfgs.mean_loss_))
```
<!-- example-end: logistic-lbfgs -->

两次拟合的实际求解器均为 `"lbfgs"`。CV 选择平均验证对数损失最小的候选项；`mean_loss_` 按 `Cs_` 顺序保存各候选值的平均损失，`cv_results_["loss_path"]` 保存候选值与数据折构成的损失数组，`best_score_` 为所选平均损失的负数。`estimator_` 是全训练数据上的最终重拟合，`estimator_.converged_` 描述该拟合的收敛状态。将 `solver` 改为 `"auto"` 或 `"irls"`，即可比较 IRLS。

### 后端与交叉验证行为

下表所有组合均支持二元响应、可选截距、样本权重和 L2 惩罚。直接拟合还接受无惩罚的 `C=0`；CV 只搜索正数 `C` 候选值。

| 指定求解器 | 实际求解器 | NumPy CPU 拟合 / CV | CuPy CUDA 拟合 / CV | Torch CUDA 拟合 / CV |
|---|---|---|---|---|
| `"auto"` | `"irls"` | IRLS / 依次进行 IRLS 拟合 | IRLS / 批处理 IRLS | IRLS / 批处理 IRLS |
| `"irls"` | `"irls"` | IRLS / 依次进行 IRLS 拟合 | IRLS / 批处理 IRLS | IRLS / 批处理 IRLS |
| `"lbfgs"` | `"lbfgs"` | L-BFGS / 依次进行 L-BFGS 拟合 | L-BFGS / 依次进行 GPU L-BFGS 拟合 | L-BFGS / 依次进行 GPU L-BFGS 拟合 |

`LogisticRegressionCV` 对所有候选拟合与最终重拟合使用同一指定求解器，`solver_` 记录实际解析值。`gpu_cv_mixed_precision` 只影响 GPU 批处理 IRLS；L-BFGS 候选拟合在所选后端上使用 float64，不会为了执行 L-BFGS 而切换到 CPU。相关接口与数据折选择见[求解器与惩罚项兼容性](../../guides/solver-penalty-matrix.md)和[交叉验证指南](../../guides/cross-validation.md)。

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

两种求解器在拟合得到的估计值处使用相同的协方差计算。正 `C` 下的推断围绕带惩罚拟合计算，不是针对无惩罚总体系数的纠偏推断。稳健协方差使用带惩罚曲率矩阵的逆作为三明治协方差估计两侧的矩阵（bread）。这些结果不修正收缩偏差，也不计入选择 `C` 的不确定性。若需要普通无惩罚 Logit 推断，使用 `C=0`，检查可识别性与收敛，并根据数据选择合理的协方差假设。`summary()` 要求 `compute_inference=True`。CV 的推断结果从最终 `estimator_` 获取。

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
| `max_iter` | `100` | 所选求解器的最大迭代次数 |
| `tol` | `1e-4` | 有限正数停止阈值；IRLS 使用参数步长范数，L-BFGS 使用缩放后目标函数的梯度范数 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | 共享估计器配置；本类不使用它控制并行拟合，也不选择求解器 |
| `compute_inference` | `True` | 是否计算推断统计 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy 内存池 |
| `solver` | `"auto"` | `auto` / `irls` / `lbfgs`；`auto` 解析为 `irls` |

上表覆盖 `LogisticRegression`。[完整 CV 构造参数参考](../../reference/linear-model-api.md#logisticregressioncv)还包括网格、数据折和 `gpu_cv_mixed_precision`。CV 不接受 `hac_maxlags`，最终重拟合的 HAC 使用自动滞后阶规则。

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

`device="torch"` 选择 Torch CUDA。显式请求 `cuda` 或 `torch` 需要相应软件包及可用的 CUDA 设备，不会静默回退到 CPU。预测数组使用所选的受支持后端；绘图函数会把结果转为 NumPy 以便渲染。安装与设备选择详见[设备与内存](../../guides/device-and-memory.md)。

## 严格与近似模式的差别

该估计器没有单独的严格/近似计算开关。CPU 与 GPU 使用相同的统计定义，但由于浮点运算和底层线性代数实现不同，结果可能存在数值差异，其大小受数据类型、收敛容差及问题条件数影响。

## 输出

- `fit(X, y, sample_weight=None) -> self`
- 预测：`predict_proba(X)`、`predict(X)`、`predict_with_threshold(X, threshold)`
- 推断属性：`_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- 拟合与信息准则：`loglikelihood`, `loglikelihood_null`, `pseudo_rsquared`, `aic`, `bic`
- 系数与拟合状态：`coef_`、`intercept_`、`n_iter_`、`solver_`、`converged_`
- 汇总：`summary()`
- CV：`C_`、`Cs_`、`mean_loss_`、`cv_results_`、`best_score_`、`cv_selected_device_`、`solver_` 及最终 `estimator_`；最终收敛状态为 `estimator_.converged_`

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

<a id="api-参考与验证"></a>

## API 参考与结果比较

分类阈值范围、指标字典、曲线形状与绘图返回值见[完整 LogisticRegression 方法参考](../../reference/linear-model-api.md#logisticregression)；[LogisticRegressionCV](../../reference/linear-model-api.md#logisticregressioncv)另有构造参数与[选择/结果约定](../../reference/linear-model-api.md#cv-methods-and-results)。继承方法见[估计器共享 API](../../reference/estimator-api.md)。

上表覆盖构造参数，拟合接口为 `fit(X, y, sample_weight=None)`；继承的 `get_params` / `set_params` 用于估计器配置。`score(X, y)` 返回准确率，`summary()` 显示推断报告。`accuracy`、`precision`、`recall`、`f1`、`auc` 和 `average_precision` 属性描述训练数据；评价泛化表现应在独立数据上调用评估方法。

完整方法签名、返回值与估计器说明见[公开类的 API 源码](../../../../statgpu/linear_model/wrappers/_logistic.py)，也可通过 `help(LogisticRegression)` 查看已安装版本的接口。该估计器只支持二分类，推断采用大样本 z 统计量；其 `fit` 不单独提供公式参数。

与 `statsmodels.Logit` 比较普通无权重、无惩罚拟合时，使用 `C=0`，并对齐响应行、编码后的设计矩阵与截距、协方差约定（包括有限样本修正）以及收敛设置。确认两边都已收敛。对于正 `C`，应先找到等价的带惩罚参考实现：对齐求和损失的归一化方式、仅施加于斜率的 L2 惩罚及其强度，再比较系数或协方差。数值设置一致并不意味着不同的协方差假设等价。

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

