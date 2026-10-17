# Lasso 推断：从去偏估计到同时置信区间

> 语言：中文  
> 最后更新：2026-10-15  
> 模型指南：[Lasso](lasso.md)  
> 切换：[English](../../../en/models/regression/lasso-inference.md)

Lasso 适合预测和稀疏特征选择，但惩罚会使系数收缩，变量选择又使用了同一份数据。
因此，不能直接把固定模型下的普通置信区间套用到惩罚系数上。本篇补充说明去偏估计量、
用于构造修正项的逐节点回归，以及计算同时置信区间的共享乘子自助法。
模型选择、求解器限制和替代方法可先阅读 [Lasso 指南](lasso.md)；
完整控制参数和输出见 [Lasso API 参考](../../reference/linear-model-api.md#lasso)。

本文介绍的公开 `nodewise_alpha` 接口及自动规则属于
[计划在 0.2.6 发布、目前尚未发布的变更](../../changelog.md)，0.2.5 版本不提供这些功能。

## 先明确需要哪一种推断

- `debiased`：通过一步修正估计高斯线性模型的系数。近似推断成立需要稀疏性、
  合适的设计与噪声条件，以及足够小的系数估计误差和精度矩阵估计误差。
- `post_selection_ols`：在入选列上重新拟合 OLS/WLS。应将其视为选择后的诊断，
  不能当作一般性的选择性推断方法。
- `bootstrap`：对残差重采样，并在固定调参配置下重新拟合惩罚估计器。
  当前仅支持无权重的高斯数据和非稳健协方差，也不能普遍校正变量选择的不确定性。

这里通过残差重采样并重拟合的 `bootstrap`，与下文**对推断得分使用的高斯乘子自助法**
不同。后者配合 `debiased` 使用，不会在每次乘子抽样时重新拟合 Lasso。
如果只关心预测，请设置 `compute_inference=False`。`LassoCV` 后的推断以选定的
惩罚强度为条件，不会单独校正交叉验证调参引入的不确定性。

## 先定义用于计算的工作模型

设原始设计矩阵为 $X\in\mathbb R^{n\times p}$，响应为 $y$。
当 `fit_intercept=True` 时，定义加权均值和逐行缩放因子：

$$
W=\sum_i w_i,\qquad
\bar x_w=\frac{1}{W}\sum_i w_i x_i,\qquad
\bar y_w=\frac{1}{W}\sum_i w_i y_i,\qquad
a_i=\sqrt{\frac{n w_i}{W}}.
$$

权重必须有限、非负且总和为正。无权重时取 $w_i=1$。
中心化并加权后的工作数组为

$$
X_{w,i}=a_i(x_i-\bar x_w),\qquad
y_{w,i}=a_i(y_i-\bar y_w),\qquad
\hat r_w=y_w-X_w\hat\beta.
$$

其中，$X_{w,i}$ 表示第 $i$ 行。设置 `fit_intercept=False` 时不做中心化：
$X_{w,i}=a_i x_i$，$y_{w,i}=a_i y_i$，且模型没有截距参数。
拟合的斜率使以下目标函数最小：

$$
\hat\beta=\arg\min_\beta
\left\{\frac{1}{2n}\lVert y_w-X_w\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1\right\}.
$$

因此，平均损失公式中的 $n$ 始终是观测行数。后文用于逐节点调参的有效样本量，
**不会**替换所有公式中的 $n$。将全部权重乘以同一正常数，不会改变这些工作数组。
统计推断是否有效，仍取决于使用权重的理由和相应的误差模型；
仅凭权重缩放不改变结果，不能证明推断有效。

## 为什么需要一步修正

Lasso 的最优性关系为

$$
\frac{X_w^\top\hat r_w}{n}=\alpha\hat\kappa,
\qquad \hat\kappa_j\in\partial|\hat\beta_j|.
$$

未加惩罚的得分在这里并不为零，这是惩罚作用的结果。
设 $M$ 近似于 $\widehat\Sigma_w=X_w^\top X_w/n$ 的逆矩阵。
去偏估计沿得分方向进行一步修正：

$$
\boxed{\hat\theta^{\rm db}=\hat\beta+
\frac{1}{n}MX_w^\top\hat r_w.}
$$

若线性工作模型 $y_w=X_w\beta^0+\varepsilon_w$ 设定正确，
则有以下精确分解：

$$
\hat\theta^{\rm db}-\beta^0
=\frac{1}{n}MX_w^\top\varepsilon_w
+(I-M\widehat\Sigma_w)(\hat\beta-\beta^0).
$$

第一项是噪声的主要贡献，第二项是余项。近似推断要求余项相对于标准误足够小。
求解器收敛，或拟合系数向量足够稀疏，都不能单独证明这一条件成立。

## 用逐节点 Lasso 构造 M

依次把每个特征当作临时响应，用其余特征预测它。
这些回归描述的是**设计矩阵内部的关系**，而不是原始响应 $y$。
首先对工作设计矩阵的各列进行标准化：

$$
d_j^2=\frac{1}{n}\sum_i X_{w,ij}^2,\qquad
D=\operatorname{diag}(d_1,\ldots,d_p),\qquad Z=X_wD^{-1}.
$$

对每个 $j$，求解

$$
\hat\gamma_j=\arg\min_{\gamma\in\mathbb R^{p-1}}
\left\{\frac{1}{2n}\lVert Z_j-Z_{-j}\gamma\rVert_2^2
+\lambda_{\rm nw}\lVert\gamma\rVert_1\right\}.
$$

主模型的 `alpha` 控制预测拟合中的收缩；`nodewise_alpha` 只控制这些用于估计
精度矩阵的辅助回归。显式提供一个有限正标量，就会在标准化设计矩阵上使用该值作为
$\lambda_{\rm nw}$。当 `nodewise_alpha=None` 且 $p\ge2$ 时，默认规则为

$$
\lambda_{\rm nw}=\sqrt{\frac{2\log(\max(p,2))}{n_{\rm nw}}},
\qquad
n_{\rm nw}=\begin{cases}
n,&\text{unweighted},\\
\dfrac{(\sum_i w_i)^2}{\sum_i w_i^2},&\text{analytic weights}.
\end{cases}
$$

等值正权重同样得到 $n_{\rm nw}=n$。$\sqrt{\log p/n}$ 的量级有理论依据；
具体常数以及加权有效样本量的定义方式，是库的默认选择，并不是某个定理规定的唯一形式。
该规则不依赖响应变量的尺度。$p=1$ 时没有需要控制的其他特征，因此不做辅助回归，
直接使用一维解析精度值，`nodewise_alpha_` 保持为 `None`。

令 $r_j=Z_j-Z_{-j}\hat\gamma_j$，定义

$$
\hat\tau_j^2=\frac{\lVert r_j\rVert_2^2}{n}
+\lambda_{\rm nw}\lVert\hat\gamma_j\rVert_1.
$$

$\Theta_Z$ 的第 $j$ 行中，对角元素为 $1/\hat\tau_j^2$，
其他列的元素为 $-\hat\gamma_{j,k}/\hat\tau_j^2$。再变换回原来的工作特征尺度：

$$
\boxed{M=D^{-1}\Theta_ZD^{-1}.}
$$

按行构造的近似矩阵不一定严格对称，因此在 $MX_w^\top\hat r_w$ 中，
矩阵乘法的方向很重要。软件会在求解后独立检查逐节点回归的 KKT 最优性条件。
若特征尺度无效、归一化量退化、结果含非有限值，或 KKT 检查失败，推断会报错，
而不会返回占位结果。通过这种数值检查，并不意味着统计假设也得到了验证。
参数校验与迁移细节见[逐节点调参](../../guides/nodewise-alpha-migration.md)。
在 `LassoCV` 中，该设置只影响全数据最终重拟合后的推断，不影响 `alpha` 网格或折内评分。

## 边际不确定性与用于报告的截距

高斯模型下使用

$$
V=M\widehat\Sigma_wM^\top,\qquad
\widehat{\mathrm{se}}_j^2=\hat\sigma^2V_{jj}/n.
$$

当前的残差尺度估计为
$\hat\sigma^2=\lVert\hat r_w\rVert_2^2/\max(n-\hat s,1)$，其中
$\hat s$ 是非零惩罚斜率的个数。这是一个尺度估计，并非一般性的选择校正自由度结论。
边际 z 统计量、p 值和置信区间以标准正态分布为参照。
得到的 `_conf_int` 是每个参数各自的边际区间，不能据此声明联合覆盖率。

公开属性 `coef_` 和 `intercept_` 仍属于惩罚预测拟合。
拟合截距时，`_params[1:]` 保存 $\hat\theta^{\rm db}$，`_params[0]` 保存与之对应、
位于原始坐标系中的报告截距：

$$
\hat\theta_0^{\rm db}=\bar y_w-\bar x_w^\top\hat\theta^{\rm db}.
$$

将该截距的影响向量写成作用于工作残差的形式，有

$$
q=a-X_wM^\top\bar x_w,\qquad h=q/n,\qquad
\widehat{\mathrm{se}}_0^2=\hat\sigma^2q^\top q/n^2.
$$

因此，截距的标准误和区间描述的是这个修正后的报告截距，而不是惩罚拟合的
`intercept_`。将所有原始特征平移同一个常向量 $c$，去偏斜率保持不变，
报告截距则变化 $-c^\top\hat\theta^{\rm db}$。

## 从边际误差到统一的同时临界值

设 $\mathcal J$ 是需要同时覆盖的参数坐标集合。
理想情况下，我们需要知道最大标准化估计误差的分布：

$$
T=\max_{j\in\mathcal J}
\left|\frac{\hat\theta_j^{\rm db}-\theta_j^0}
{\widehat{\mathrm{se}}_j}\right|.
$$

若 $c_{1-\alpha_{\rm sim}}$ 是该分布对应的分位数，
则区间 $\hat\theta_j^{\rm db}\pm c_{1-\alpha_{\rm sim}}\widehat{\mathrm{se}}_j$
的目标是满足

$$
\Pr\{\theta_j^0\in CI_j^{\rm sim}\text{ for all }j\in\mathcal J\}
\approx1-\alpha_{\rm sim}.
$$

这里关注的是整个参数集合上的最大值，而不是某一个服从标准正态参照分布的坐标。
给每个参数分别构造 95% 边际区间，并不能自动得到 95% 的联合覆盖率。

### 用共享高斯乘子近似最大值分布

在去偏估计的分解中，忽略足够小的余项后，主要项为
$MX_w^\top\varepsilon_w/n$。用 $\hat r_w$ 替代未知的工作误差，
在第 $b$ 次抽样中独立生成 $\xi_i^{(b)}\sim N(0,1)$，得到斜率扰动

$$
\boxed{S^{*(b)}=\frac{1}{n}MX_w^\top
(\xi^{(b)}\odot\hat r_w),\qquad
Z_j^{*(b)}=S_j^{*(b)}/\widehat{\mathrm{se}}_j.}
$$

同一次抽样中的每个坐标都使用**同一个乘子向量**。这样可以保留由观测、
设计矩阵和共同精度矩阵产生的坐标间依赖关系，而不是分别对各个系数独立做自助抽样。
若目标集合只包含斜率，则计算

$$
T^{*(b)}=\max_{j=1,\ldots,p}|Z_j^{*(b)}|.
$$

进行 $B=$ `simultaneous_n_bootstrap` 次抽样后，使用经验分位数

$$
\hat c_{1-\alpha_{\rm sim}}=
\widehat Q_{1-\alpha_{\rm sim}}(T^{*(1)},\ldots,T^{*(B)}),\qquad
CI_j^{\rm sim}=\hat\theta_j^{\rm db}\pm
\hat c_{1-\alpha_{\rm sim}}\widehat{\mathrm{se}}_j.
$$

这种 max-|Z| 方法考虑了坐标间的依赖关系，用来校准最大值分布；它既不是逐坐标独立自助法，
也不是机械地进行 Bonferroni 调整。其有效性仍依赖去偏 Lasso 和乘子自助法的假设，
并不能为任意设计矩阵、误差分布或权重提供精确的有限样本保证。
增加抽样次数可以减小蒙特卡洛误差，但无法弥补不成立的统计假设。

### 将截距纳入同一个最大值

当 `fit_intercept=True` 且 `simultaneous_include_intercept=True` 时，
使用与截距边际推断相同的 $h=q/n$：

$$
S_0^{*(b)}=\sum_i\xi_i^{(b)}\hat r_{w,i}h_i,\qquad
Z_0^{*(b)}=S_0^{*(b)}/\widehat{\mathrm{se}}_0,
$$

$$
T^{*(b)}=\max\{|Z_0^{*(b)}|,\max_{j=1,\ldots,p}|Z_j^{*(b)}|\}.
$$

截距会直接参与临界值校准，而不是先只根据斜率求出临界值，再额外附上一行截距区间。
默认的 `simultaneous_include_intercept=False` 只对斜率进行同时校准：
若模型拟合了截距，`_conf_int_simultaneous` 的第 0 行仍保留截距的**边际**区间。
`_inference_result.simultaneous_target_mask` 有值时，可用它识别联合校准的目标行。
当前 CPU 上只校准斜率的计算路径会将该字段留为 `None`；此时应根据
`fit_intercept` 和 `simultaneous_include_intercept` 确定目标行。
不拟合截距时，所有行都对应斜率。

## 完整的 CPU 示例

请在同一个会话中依次运行以下步骤。这里使用固定惩罚强度作示范，数据为模拟生成；
不能根据这一次拟合推断实际覆盖率。

<!-- learner-example: lasso-inference-companion -->
```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.normal(size=(400, 5)) + np.array([1.0, -0.5, 0.0, 0.8, -0.2])
beta = np.array([1.5, -1.0, 0.0, 0.7, 0.0])
y = 0.4 + X @ beta + rng.normal(size=400)
```

启用同时推断时，请显式选择 `debiased`；当前构造函数不接受这一组合中的 `auto`。
抽样次数须为正整数，族错误率水平须严格位于 0 与 1 之间。

```python
model = Lasso(
    alpha=0.05, nodewise_alpha=None, device="cpu",
    solver="coordinate_descent", max_iter=5000, tol=1e-10,
    compute_inference=True, inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap", simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=200, simultaneous_random_state=123,
    simultaneous_include_intercept=True,
).fit(X, y)
```

分别读取预测估计和推断估计，再查看区间数组及参与联合校准的参数集合：

```python
print("Prediction slopes:", model.coef_)
print("Reporting parameters:", model._params)
print("Node-wise penalty:", model.nodewise_alpha_)
print("Marginal/joint shapes:", model._conf_int.shape,
      model._conf_int_simultaneous.shape)
print("Joint target rows:", model._inference_result.simultaneous_target_mask)
print("Common critical value:", model._simultaneous_critical_value)
```
<!-- example-end: lasso-inference-companion -->

两个区间数组的形状都是 `(6, 2)`，第一行是截距，随后五行是斜率。
本例中，目标掩码的六个元素均为 `True`。200 次抽样只是为了让示例计算量较小；
若需要更稳定的蒙特卡洛校准，应增加抽样次数。随机种子控制同一后端内的抽样，
并不保证 CPU 与 GPU 生成完全相同的随机序列。

## 设备与限制

`inference_method` 选择统计方法，`device` 选择执行设备。
显式的 `device="cuda"` 请求 CuPy CUDA，`device="torch"` 请求 Torch CUDA；
若指定设备不可用，会直接报错，不会静默选择 CPU。
只有实际使用 `device="auto"` 时，才允许自动选择后端。
全局默认设置和后端原生输入的设备选择规则，见[设备指南](../../guides/device-and-memory.md)。

边际去偏推断使用成功拟合所采用的后端。`fit_intercept=True` 时，
同时校准也在该具体后端和设备上执行，最后返回用于报告的 NumPy 数组。
`fit_intercept=False` 时，即使拟合在 GPU 上完成，同时推断目前仍使用 NumPy 主机辅助程序；
这部分计算**不是 GPU 原生计算**。不能因为模型在 GPU 上拟合，
就认定之后每个辅助计算也都在 GPU 上进行。

若需要调整或合并边际 p 值，请参阅
[多重检验指南](../../guides/multiple-testing-combine-pvalues.md)。
这些操作属于另一层统计处理，需要输入的 p 值本身有效；
它们不能修复有偏估计，也不能弥补不适用的推断假设。

## 参考文献

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional
  parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On
  asymptotically optimal confidence regions and tests for high-dimensional
  models. *Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis
  testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
- Zhang, X., & Cheng, G. (2017). Simultaneous inference for high-dimensional
  linear models. *JASA*, 112(518), 757–768.
