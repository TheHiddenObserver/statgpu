# 逐节点 Lasso 推断调参迁移说明

> 状态：适用于 0.2.5 选择后 OLS API 迁移之后引入的逐节点调参契约。

## 改了什么

稀疏 Gaussian 纠偏推断现在公开 `nodewise_alpha` 参数，覆盖 `Lasso`、`ElasticNet`、相应的 Gaussian penalized 基础接口，以及 `LassoCV` / `ElasticNetCV` 的最终重拟合推断配置。

主模型 `alpha` 与 `nodewise_alpha` 是两个不同的参数：

- `alpha` 控制用于预测/变量选择的惩罚拟合；
- `nodewise_alpha` 只控制纠偏推断中用于近似设计精度矩阵的逐节点 Lasso 问题。

用户显式给出的有限正标量具有最高优先级；`nodewise_alpha=None` 才使用 statgpu 的自动规则。

## 有意修正默认行为

历史内部实现使用主响应模型的残差尺度来决定逐节点惩罚：

$$
\hat\sigma_y\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

这个规则从未作为公开调参契约承诺给用户，而且会使设计侧精度矩阵估计依赖 `y` 的计量单位。新的自动规则先对已经完成中心化/加权处理的规范工作设计进行标准化，再使用

$$
\lambda_{\mathrm{nw}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无分析权重时 `n_nw=n`；非均匀分析权重下使用 Kish 型有效样本量。$\sqrt{\log(p)/n}$ 的量级具有高维逐节点回归的理论动机，但具体的常数和加权有效样本量约定属于 statgpu 的默认选择，并不是某个定理规定的唯一形式。

本次迁移**不提供**旧的、依赖响应尺度的内部规则作为兼容选项。需要特定逐节点惩罚时，应在标准化逐节点尺度上显式设置 `nodewise_alpha=`。

## 精度矩阵构造

令 `X_w` 表示稀疏 Gaussian 推断路径已经定义好的中心化/加权平均损失工作设计，并令

$$
d_j^2=\frac{1}{n}\sum_iX_{w,ij}^2,
\qquad Z=X_wD^{-1}.
$$

statgpu 在 `Z` 上对每个特征求解逐节点 Lasso，随后独立重新检查完整 KKT 残差，并使用更接近原始文献记号的归一化量

$$
\hat\tau_j^2
=
\frac{\|r_j\|_2^2}{n}
+
\lambda_{\mathrm{nw}}\|\hat\gamma_j\|_1.
$$

标准化尺度上得到的近似精度矩阵最后再变换回原工作特征尺度。特征尺度退化、非有限精度状态、KKT 检验失败或归一化量无效时，推断会 fail closed，不再发布历史上的 identity-row 占位结果。

当 `p=1` 时不存在 nuisance 逐节点回归；statgpu 直接使用一维解析精度矩阵，并令 `nodewise_alpha_` 保持为 `None`。

## 内部求解与溯源信息

逐节点 FISTA 的内部迭代停止阈值故意设置得比最终 KKT 发布门槛更严格。目前内部使用 `coef_delta` 停止准则、`1e-8` 迭代容差和 3000 次最大迭代预算，随后还必须通过独立的 `1e-5` KKT 检验。这些属于内部数值设置，不是新的公开调参参数。

多特征纠偏推断成功后，实际使用的值通过 `nodewise_alpha_` 暴露。`_inference_result.metadata` 会记录请求值、解析值、来源、自动规则标识、加权有效样本量、逐节点求解设置、最大 KKT 残差、缓存信息以及实际数值后端/设备。

## 交叉验证

`LassoCV(nodewise_alpha=...)` 与 `ElasticNetCV(nodewise_alpha=...)` 都把该参数视为**最终全数据重拟合的推断配置**。它不会进入主正则化参数候选网格，也不会改变折内评分、最终 `alpha_` 或 `l1_ratio_` 的选择。

## 后端与权重契约

NumPy、CuPy 与 Torch 使用同一个标准化统计定义。显式 CUDA/Torch 推断不会在数值计算阶段静默回退到 CPU。分析权重继续满足既有平均损失约定、全局正权重缩放不变性、全 1 权重恒等性，以及自动逐节点调参对零权重行增删的不变性。

精度矩阵缓存可以为了缓存身份对后端驻留的工作设计做分块哈希，但逐节点求解、KKT 检验、精度矩阵回变换以及 GPU 同时推断的数值计算仍在所选后端/设备上完成。

## 参考文献

- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217-242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869-2909.
