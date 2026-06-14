# GAM（广义可加模型）

> 语言: 中文
> 最后更新: 2026-05-28
> 页面定位: 模型文档
> 切换: [English](../en/models/semiparametric.md)

语言切换：[English](../en/models/semiparametric.md)

## 概览（Overview）

`GAM` 使用惩罚 B 样条拟合广义可加模型，通过广义交叉验证（GCV）自动选择平滑参数。模型为：

$$
y = \alpha + \sum_j f_j(x_j) + \epsilon
$$

其中每个 $f_j$ 表示为惩罚 B 样条。GAM 是半参数模型：具有参数化的截距和每个特征的非参数光滑函数。

底层 B 样条基工具请参见 [样条基函数](splines.md)。

## 路径（Path）

`statgpu.semiparametric.GAM`

## 目标函数（Objective Function）

GAM 拟合惩罚最小二乘模型：

$$
\min_{\beta} \|y - B\beta\|_2^2 + \lambda \, \beta^\top S \, \beta
$$

其中 $B$ 为各特征样条基矩阵的列拼接（加一列截距），$S$ 为块对角差分惩罚矩阵，$\lambda$ 为平滑参数。默认惩罚阶数为 2（二阶差分），惩罚曲率。

## 估计方程（Estimating Equation）

惩罚目标的一阶条件给出如下系统：

$$
(B^\top B + \lambda S) \, \hat\beta = B^\top y
$$

通过 Cholesky 分解求解。

**GCV 选择 lambda**（当 `lam=None` 时）：

$$
\text{GCV} = \frac{n \cdot \text{RSS}}{(n - \text{edf})^2}
$$

其中有效自由度为：

$$
\text{edf} = \text{tr}\!\left((B^\top B + \lambda S)^{-1} B^\top B\right)
$$

通过对数间隔网格搜索最小化 GCV 来选择 lambda。

## 协方差与推断（Covariance/Inference）

- `edf_`：拟合模型的有效自由度。
- `gcv_score_`：GCV 得分（lambda 自动选择时可用）。
- `lam_`：最终拟合使用的平滑参数。
- 不产生系数层面的标准误或 p 值；GAM 是平滑器，而非参数推断工具。

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `n_splines` | `20` | 每个特征的样条基函数数量 |
| `degree` | `3` | 样条次数 |
| `lam` | `None` | 平滑参数；若为 `None` 则通过 GCV 自动选择 |
| `penalty_order` | `2` | 差分惩罚矩阵的阶数 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.semiparametric import GAM
import numpy as np

X = np.random.randn(500, 3)
y = np.sin(X[:, 0] * 3) + 0.1 * np.random.randn(500)

# GAM（CPU）
gam = GAM(n_splines=20, device='cpu')
gam.fit(X, y)
print(f"有效自由度: {gam.edf_:.1f}, GCV: {gam.gcv_score_:.6f}")
y_pred = gam.predict(X)

# GAM（GPU）
gam_gpu = GAM(n_splines=20, device='cuda')
gam_gpu.fit(X, y)
y_pred_gpu = gam_gpu.predict(X)
```

## strict/approx 差异（strict/approx difference）

- 当 `lam=None`（默认）时，GAM 在对数间隔网格（1e-10 到 1e10，100 个点）上使用 GCV 选择平滑参数。这是近似路径；网格较粗糙，可能在狭窄谷底遗漏最优 lambda。
- 当手动指定 `lam` 时，对该单一值计算精确的惩罚最小二乘解。这是精确路径。
- 若需在默认网格之外进行精细调整，可传入自定义 `lam` 值，该值可通过更窄的搜索或领域知识获得。

## 输出（Outputs）

**GAM 拟合属性**：

| 属性 | 类型 | 说明 |
|---|---|---|
| `coef_` | array，形状 `(1 + sum(n_basis_j),)` | 拼接的样条系数（含截距） |
| `intercept_` | float | 截距项 |
| `edf_` | float | 有效自由度 |
| `gcv_score_` | float | 所选 lambda 下的 GCV 得分 |
| `lam_` | float | 使用的平滑参数 |
| `knots_` | list of arrays | 各特征的节点位置 |
| `n_features_` | int | 输入特征数 |

**方法**：`fit(X, y)`、`predict(X)`、`summary()`。

## 常见问题（FAQ）

**节点数量如何选择？** `n_splines=20` 是一个合理的默认值。更多节点提供更大灵活性，但会增加有效自由度并有过拟合风险。

**惩罚阶数如何选择？** `penalty_order=2`（二阶差分）是光滑函数的标准选择。若需要分段线性拟合，使用 `penalty_order=1`。

**GPU 加速效果如何？** GAM 求解受 Cholesky 分解主导，对于大基维度可从 GPU 加速中获益。

## 外部验证（External Validation）

- GAM 预测在标准测试数据集上与 pyGAM 进行了验证。

## 参考文献（References）

- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman & Hall/CRC.
