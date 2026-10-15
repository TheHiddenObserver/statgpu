# 核岭回归

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../../en/models/nonparametric/kernel-ridge.md)

## 用观测之间的相似性预测非线性响应

如果要预测连续响应，且相似观测应有相近结果，但直线模型过于受限，可以考虑
`KernelRidge`。RBF 核允许拟合曲面弯曲，无需手工列出每个非线性特征与交互项。
模型学习一组权重，将新观测与各训练观测的相似度组合成预测值，同时用 `alpha`
惩罚拟合函数的复杂度。这些权重不是原始特征的斜率。

若线性关系已足够，或需要解释斜率，应先考虑[线性模型](../regression/linear-regression.md)。
如果主要是各特征独立平滑作用的相加，[GAM](semiparametric.md) 往往更易解释。
精确核岭适合交互关系重要、且训练核能放入内存的任务；
[Nystroem 特征](kernel-nystroem.md) 则提供更小的近似表示。

<a id="theory"></a>

## 从相似度到拟合函数

设训练观测为 $x_1,\ldots,x_n$，响应为 $y$，核矩阵为
$K_{ij}=k(x_i,x_j)$。默认 RBF 核为

$$
k(x,z)=\exp\{-\gamma\lVert x-z\rVert_2^2\}.
$$

`gamma` 越大，相似度随距离衰减得越快。特征单位会影响距离；尺度远大于其他列的
特征可能主导相似度。对于半正定核，拟合是在一类由核函数规定的函数中进行的。
这个函数空间称为再生核希尔伯特空间（RKHS），下式用 $\mathcal H$ 表示。
第一项要求预测接近观测响应；第二项按这个空间的范数限制拟合函数的复杂度：

$$
\min_{f\in\mathcal H}\sum_{i=1}^n(y_i-f(x_i))^2
    +\alpha\lVert f\rVert_{\mathcal H}^2.
$$

拟合函数可写为 $f(x)=\sum_i c_i k(x,x_i)$。代入后，函数的惩罚变成
$c^\top Kc$，不是简单地对权重平方求和，系数目标为

$$
\min_c\ \lVert y-Kc\rVert_2^2+\alpha c^\top Kc.
$$

这里的损失是**求和而非平均**。若改为惩罚 $\lVert c\rVert^2$，将得到另一种估计量。
对于对称 $K$，梯度为

$$
\nabla_c=2K(Kc-y)+2\alpha Kc
       =2K\{(K+\alpha I)c-y\}.
$$

当 $K$ 正定时，令梯度为零即可得到

$$
(K+\alpha I)c=y.
$$

当 $K$ 奇异但半正定且 $\alpha>0$ 时，该可逆线性系统仍能选出一组使上述目标函数最小的系数；
原目标中的最优系数表示不一定唯一。不定核不具有上述凸 RKHS
解释。多响应矩阵 $Y$ 使用同一系统和矩阵右端项求解，目标函数将各响应列的平方
损失与 RKHS 惩罚相加。

对 $q$ 条新观测，先构造 $q\times n$ 的查询到训练核，再预测：

$$
\hat y_{\mathrm{new}}=K(X_{\mathrm{new}},X_{\mathrm{train}})c.
$$

该 API 不单独拟合截距。若决定对响应中心化，应只用训练数据估计响应均值，拟合
中心化后的响应，并在预测时加回均值。选择模型设置时，这一步也要在各训练折内
重新进行。

<a id="example"></a>
<a id="kernel-cpu-workflow"></a>

## 完整的 CPU 训练与留出集示例

以下示例可独立运行。`X` 有 240 行、两列连续特征，`y` 为每行一个连续响应。
先留出 60 行用于最终评价，并在查看留出结果前固定 `alpha=1.0`、`gamma=0.7`。
若希望在训练折内选择 `alpha`，可使用 [KernelRidgeCV](kernel-ridge-cv.md)。

<!-- example: kernel-ridge-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidge
```

### 准备并划分数据

响应由正弦效应和二次效应组成，直线模型无法充分描述其形状。先留出测试行，
再选择模型设置。

```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### 用固定参数拟合

仅使用 180 条训练观测拟合，然后检查学到的权重是否有限。

```python
kr = KernelRidge(alpha=1.0, gamma=0.7, device="cpu").fit(X_train, y_train)
assert np.isfinite(kr.dual_coef_).all()
```

### 预测并评价

测试行保持相同的特征顺序，直接使用已拟合模型预测。

```python
fixed_prediction = kr.predict(X_test)
assert np.isfinite(fixed_prediction).all()
print("Prediction shape:", fixed_prediction.shape)
print("Fixed-alpha held-out R2:", kr.score(X_test, y_test))
print("Held-out MSE:", np.mean((y_test - fixed_prediction) ** 2))
```
<!-- example-end: kernel-ridge-cpu -->

输出 `(60,)` 表示每条留出观测各有一个预测值，单位与 `y` 相同。MSE 的单位是响应
单位的平方，越小越好。R² 为 1 表示完美预测，0 表示与留出响应均值这一常数基线
相当，负值表示比该基线更差。多响应的 `score` 对各列 R² 等权平均；常数响应在
预测近乎精确时记为 1，否则记为 0。

这里 `dual_coef_` 的形状为 `(180, 1)`，虽然预测输出会展平。它是在核空间中对
训练观测加权，不能解释为某个输入特征增加一单位的效应。本模块不提供系数级标准误、
假设检验或置信区间。

### 在小示例中检查定义方程

接着使用上面的 CPU 拟合结果。额外构造稠密训练核适合小规模诊断，但会重复占用
内存，正常预测并不需要这一步。

<!-- example-requires: kernel-ridge-cpu -->
<!-- example: kernel-ridge-system-check -->
```python
from statgpu.nonparametric.kernel_methods import pairwise_kernels

K = pairwise_kernels(X_train, metric="rbf", gamma=0.7, xp=np)
np.testing.assert_allclose(
    (K + kr.alpha * np.eye(len(X_train))) @ kr.dual_coef_,
    y_train[:, None], rtol=1e-9, atol=1e-9,
)
```
<!-- example-end: kernel-ridge-system-check -->

## 联合考虑核与正则化

对于 RBF 模型，`gamma` 控制多远范围内的观测仍有较高相似度，`alpha` 则控制求和
损失目标中的收缩程度。`gamma` 默认值为 `1 / n_features`。两者都应在训练过程中
选择；内置 CV 类**只选择 alpha**。

实践中可先使用有限、严格正的 `alpha`。零值虽被接受，但遇到奇异核时较脆弱。
与平均损失形式比较时，必须先将惩罚换算到相同尺度。

更换 `kernel` 会改变相似度的定义。多项式核还使用 `degree` 和 `coef0`；卡方核
要求输入非负。KernelRidge 的卡方核目前忽略构造参数 `gamma`，须通过
`kernel_params={"gamma": value}` 传入。

从数据学习的缩放、填补、响应中心化都必须在各训练折内拟合。需要此类预处理，
或观测有分组、时间顺序时，应使用
[外部分折循环](kernel-ridge-cv.md#fold-local-preprocessing)。
[完整核方法 API](../../reference/kernel-methods-api.md) 列出所有构造参数、公开方法、
拟合属性、核函数与后端边界。

<a id="numerical-precautions"></a>

## 数值问题与使用注意事项

`KernelRidge` 直接求解正则化线性系统。若线性代数操作抛出异常，会逐步增大对角
扰动后重试；得到的稳定化解未必精确满足最初显示的系统。优先选择尺度合理的
正 `alpha`。与 scikit-learn 或线性系统对照时，要对齐核、正则化、数据以及无权重
的求和损失目标。

`fit(X, y, sample_weight=None)` 目前**忽略样本权重**。应省略该参数，仅用于无权重
分析。删除行可以排除观测，但不能实现任意加权。

有限训练数据不保证计算出的核有限。多项式核可能溢出，NaN `gamma` 等无效控制项
也可能使拟合成功返回，但系数或预测非有限。像示例一样检查适用的参数、拟合数组
与输出；丢弃非有限拟合，调整缩放或核设置后，用新实例重新拟合。已经溢出的核
不能靠增大 `alpha` 修复。仅有有限结果也不代表条件数良好或预测有效。

RBF 的平方距离运算可能丢失具有巨大共同偏移坐标之间的小差异，例如坐标接近
`1e9` 时。可从训练数据学习同一偏移量，并同时从训练和查询特征中减去。该操作
保持理想的 RBF 相似度，但不适用于所有核，尤其不能机械用于卡方核。CV 时仍须在
各训练折内学习偏移量。

预测时保持特征数量和顺序一致。一维 `X` 表示单个特征的多行观测，而不是一条
多变量查询；单条多变量查询应写成二维行。修改参数后应重新拟合，不要反复根据
留出得分调整模型设置。

## 计算成本与设备

精确拟合存储 $n\times n$ 核矩阵，内存随样本数平方增长，直接求解最坏为立方级
算术成本。核构造与求解使用所选后端。显式请求不可用的 `"cuda"` 或 `"torch"`
会报错；`"auto"` 表示自动选择。小问题可能在 CPU 上更快，因为数据传输、启动与
同步也有开销。

[CuPy 与 Torch CUDA 示例](kernel-ridge-cv.md#gpu-examples)在 CPU 交叉验证示例的
基础上展示 GPU 拟合。CPU 检查不能证明 GPU 精度或速度。KernelPCA 与 Nystroem
的执行和输出设备边界不同，详见对应教程以及
[设备指南](../../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)。

## 延伸阅读

- [用 KernelRidgeCV 选择 alpha](kernel-ridge-cv.md)
- [用 Nystroem 构造更小的特征表示](kernel-nystroem.md)
- [核方法总览](kernel-methods.md)与[完整 API](../../reference/kernel-methods-api.md)
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
