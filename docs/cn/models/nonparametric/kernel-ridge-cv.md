# 核岭回归交叉验证

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../../en/models/nonparametric/kernel-ridge-cv.md)

## 不使用测试集，选择正则化强度

正则化强度 `alpha` 决定核岭回归对拟合函数复杂度的惩罚力度。`KernelRidgeCV`
对每个候选值在训练折上拟合，再用对应验证折的预测误差比较候选值。选定后，
它用传给 `fit` 的全部数据拟合最终 `KernelRidge`。

如果核及其设置已经确定，且观测适合随机打乱后按行分折，该类可以简化调参。
`KernelRidgeCV` **只选择 alpha**，不选择核类型、`gamma` 或预处理方案。
`alpha` 已固定，或需要自行控制分折与预处理时，使用 [KernelRidge](kernel-ridge.md)。

<a id="selection"></a>

## 搜索实际比较什么

对每个候选 $\alpha$ 和训练折 $T$，先求解

$$
(K_{TT}+\alpha I)c_{\alpha,T}=y_T
$$

再通过 $K_{VT}c_{\alpha,T}$ 预测对应验证折 $V$。若有 $F$ 折和 $r$ 个响应列，
选择过程最小化

$$
L(\alpha)=\frac{1}{Fr}\sum_{f=1}^{F}\sum_{j=1}^{r}
\frac{1}{|V_f|}\sum_{i\in V_f}
  (y_{ij}-\hat y_{ij}^{(-f,\alpha)})^2.
$$

各折和各响应列都等权。当各折大小不同时，这不等于把所有验证观测合并后求整体
MSE。多响应时，即使按列等权平均，数值尺度较大的响应仍可能主导 MSE。

`alpha_` 保存选中的值。**`best_score_` 是该值对应的平均折 R²，而不是选择时
最小化的 MSE。** `estimator_` 是选择后用全部传入训练行重新拟合的最终模型。
它的训练得分和 `best_score_` 都不等于独立留出评价。

搜索中只有 $\alpha$ 变化，因此每折可以复用核矩阵的特征分解。
若 $K_{TT}=Q\Lambda Q^\top$，每个候选的系数可写为

$$
c_{\alpha,T}=Q\,\operatorname{diag}\!\left(\frac{1}{\lambda_i+\alpha}\right)
Q^\top y_T.
$$

这样就不必为每个候选重新求解线性系统。每次拟合均使用
[按损失求和的核岭目标函数](kernel-ridge.md#theory)。

<a id="example"></a>
<a id="kernel-cv-workflow"></a>

## 完整的 CPU 选择与评价示例

本示例可独立运行，240 条观测中有 60 条完全留在模型选择之外。两列特征按相同
尺度生成，不使用需要学习的预处理。核及 `gamma=0.7` 预先固定。

<!-- example: kernel-ridge-cv-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidgeCV
```

### 准备训练行与留出行

留出集不参与参数搜索。

```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### 在训练折中选择 alpha

用五折交叉验证比较四个正则化强度。所有候选值使用相同且固定的核与尺度。

```python
kr_cv = KernelRidgeCV(
    alphas=[0.01, 0.1, 1.0, 10.0], gamma=0.7,
    cv=5, random_state=42, device="cpu",
).fit(X_train, y_train)
```

### 检查搜索结果并预测

解释拟合前，先排除非有限的搜索结果与模型输出。

```python
assert np.isfinite(kr_cv.cv_results_["mean_mse"]).all()
assert np.isfinite(kr_cv.best_score_)
assert np.isfinite(kr_cv.dual_coef_).all()
prediction = kr_cv.predict(X_test)
assert np.isfinite(prediction).all()
held_out_r2 = kr_cv.score(X_test, y_test)
held_out_mse = np.mean((y_test - prediction) ** 2)
```

### 查看所选模型与留出表现

分别报告选择指标和最终评价，避免混淆。

```python
print("Selected alpha:", kr_cv.alpha_)
print("Mean validation MSE by alpha:", kr_cv.cv_results_["mean_mse"][:, 0])
print("Mean fold R2 at selected alpha:", kr_cv.best_score_)
print("Prediction shape:", prediction.shape)
print("CV model held-out R2 / MSE:", held_out_r2, held_out_mse)
```
<!-- example-end: kernel-ridge-cv-cpu -->

预测形状为 `(60,)`。MSE 越小，平方预测误差越小；R² 相对于当前评价行的响应均值
基线，1 为完美预测，0 与该基线相当，负值更差。`best_score_` 汇总训练折；最终
留出 R² 与 MSE 评价的是选定并重新拟合后的模型。不要持续利用同一批测试行调整
候选值、核类型或 `gamma`。

### 读取结果数组

若有 $a$ 个候选、$F$ 折、$r$ 个响应列：

- `cv_results_["alphas"]` 保存候选值。
- `mse_table` 和 `r2_table` 的形状为 `(a, F, r)`。
- `mean_mse` 和 `mean_r2` 对折求平均后，形状为 `(a, r)`。
- `best_alpha`、`best_score` 分别对应 `alpha_`、`best_score_`。

即使在 GPU 上拟合，这些结果数组也是 NumPy 数组。多响应时，对 `mean_mse` 每行
再沿响应列求平均，即得到用于选择的损失。`predict` 和 `score` 委托给
`estimator_`；后者对各响应 R² 等权平均。完整属性和形状见
[API 参考](../../reference/kernel-methods-api.md)。

## 选择候选值与分折方式

可像示例一样，先在对数尺度上选择**严格为正**的 `alphas`。该参数接受任意非空、
有限、非负的数组；`alphas=None` 时，根据训练核谱自动构造 100 个正的对数等距候选。

训练折核奇异时，显式零 alpha 可能导致除零。非有限 CV 得分仍可能被选中，因此
返回拟合对象本身不能证明搜索有效。使用正候选，并检查 `mean_mse`、`best_score_`、
最终系数和预测的有限性。最终重新拟合成功，也不能修复此前无效的选择过程。

`cv` 必须为 2 到传入行数之间的整数，使用打乱后的按行 K 折。`random_state`
让打乱可复现，但不会把它变成分组或时间分折。

一次搜索中，核类型及其控制参数保持固定。若要选择 `gamma` 等设置，需要额外的
训练内选择过程，评价完整拟合流程。

<a id="fold-local-preprocessing"></a>

## 何时必须使用外部分折循环

内置 `cv` 不接受自定义分折器、分组折或按时间排序的折，也不会为每折单独拟合
预处理。不能先在全部传入训练行上学习缩放参数，再将变换结果交给 `KernelRidgeCV`：
这样内层验证行已参与缩放参数的估计。

需要学习预处理时，应在外部循环中对原始训练数据分折，只用各折训练行拟合预处理，
再变换验证行，并为每个候选拟合 `KernelRidge`。候选之间应复用同一组划分。选择完成
后，用全部外层训练行重新拟合预处理与模型，再一次性评价未使用过的测试集。
分组或时间数据还应使用保持分组或时间方向的划分。

下面的独立示例使用尺度不同的特征，展示为何要在分折循环内学习缩放参数。
示例按独立观测进行行级分折；分组或时间数据需要替换相应划分方式。

<!-- example: kernel-ridge-fold-local-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidge

```

先从原始数据中留出测试集，再只对训练行分折。

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(160, 2)) * np.array([1.0, 10.0])
y = np.sin(X[:, 0]) + 0.02 * X[:, 1] ** 2 + rng.normal(0, 0.1, 160)
order = rng.permutation(len(X))
X_train, X_test = X[order[:120]], X[order[120:]]
y_train, y_test = y[order[:120]], y[order[120:]]
folds = np.array_split(rng.permutation(len(X_train)), 4)
alphas = np.array([0.01, 0.1, 1.0, 10.0])
losses = np.empty((len(alphas), len(folds)))

```

每折只用其训练行估计均值和尺度，再将同一变换应用到验证行，以比较各候选模型。

```python
for fi, validation in enumerate(folds):
    training = np.concatenate([fold for j, fold in enumerate(folds) if j != fi])
    mean = X_train[training].mean(axis=0)
    scale = X_train[training].std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    X_fit = (X_train[training] - mean) / scale
    X_validation = (X_train[validation] - mean) / scale
    for ai, alpha in enumerate(alphas):
        model = KernelRidge(alpha=alpha, gamma=0.7, device="cpu")
        model.fit(X_fit, y_train[training])
        pred = model.predict(X_validation)
        losses[ai, fi] = np.mean((y_train[validation] - pred) ** 2)

```

选定 alpha 后，用全部外层训练行重新学习预处理，供最终模型拟合和测试集预测使用。

```python
assert np.isfinite(losses).all()
selected_alpha = alphas[np.argmin(losses.mean(axis=1))]
mean = X_train.mean(axis=0)
scale = X_train.std(axis=0)
scale = np.where(scale > 0, scale, 1.0)
final_model = KernelRidge(alpha=selected_alpha, gamma=0.7, device="cpu")
final_model.fit((X_train - mean) / scale, y_train)
held_out_prediction = final_model.predict((X_test - mean) / scale)
assert np.isfinite(held_out_prediction).all()
print("Selected alpha:", selected_alpha)
print("Held-out MSE:", np.mean((y_test - held_out_prediction) ** 2))
```
<!-- example-end: kernel-ridge-fold-local-cpu -->

这里每个验证损失都对应一套仅用该折训练行拟合的缩放参数和模型。最终缩放参数
由全部 120 条外层训练行估计，并原样应用到 40 条测试行。

## 能力边界与计算成本

交叉验证不提供系数推断或置信区间。底层模型采用无权重拟合，本类没有样本权重参数。
[KernelRidge 注意事项](kernel-ridge.md#numerical-precautions) 同样适用：有限数据
可能产生溢出的核，RBF 坐标尺度不当可能丢失距离，重新拟合时也可能加入数值扰动。

完整核需要平方级内存，每个训练折还需稠密特征分解。跨 alpha 复用分解能节省工作，
但各折仍增加计算量。核及折内线性代数使用所选后端；显式请求不可用设备会报错。
[完整 API](../../reference/kernel-methods-api.md) 进一步说明了执行边界。

<a id="gpu-examples"></a>

## 可选 GPU 示例

如果已安装可用的 CUDA 环境，只需改变设备，就能在 GPU 上重复同一搜索。先运行
[CPU 选择示例](#kernel-cv-workflow)，定义 `KernelRidgeCV`、`X_train`、`y_train`、
`X_test` 和 `kr_cv`，再选择下面一个片段。两个片段都依赖这些定义，并需要对应的
CUDA 后端。

<a id="cupy"></a>

### CuPy CUDA

<!-- example-requires: kernel-ridge-cv-cpu -->
<!-- example: kernel-methods-cupy -->
```python
gpu_options = {**kr_cv.get_params(), "device": "cuda"}
kr_cupy = KernelRidgeCV(**gpu_options).fit(X_train, y_train)
prediction_cupy = kr_cupy.predict(X_test)
```
<!-- example-end: kernel-methods-cupy -->

<a id="torch-cuda"></a>

### Torch CUDA

<!-- example-requires: kernel-ridge-cv-cpu -->
<!-- example: kernel-methods-torch -->
```python
gpu_options = {**kr_cv.get_params(), "device": "torch"}
kr_torch = KernelRidgeCV(**gpu_options).fit(X_train, y_train)
prediction_torch = kr_torch.predict(X_test)
```
<!-- example-end: kernel-methods-torch -->

`device="cuda"` 请求 CuPy CUDA，`device="torch"` 请求 Torch CUDA。对设备上的
预测和系数使用对应后端的有限性检查；NumPy CV 结果仍按 CPU 教程检查。CPU 检查
不能证明 GPU 精度或速度。KernelPCA 与 Nystroem 的执行和输出位置不同，详见对应
教程以及[设备指南](../../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)。

## 延伸阅读

- [KernelRidge：目标函数、预测与数值检查](kernel-ridge.md)
- [核方法总览](kernel-methods.md)与[完整 API](../../reference/kernel-methods-api.md)
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
