# 逐节点 Lasso 推断调参迁移说明

> 版本说明：当前正式发布版为 **0.2.5**。本页所述逐节点调参新规则已合入当前 `master`，计划随 **0.2.6** 发布；正式版 0.2.5 尚不包含公开的 `nodewise_alpha` 参数及这里说明的新默认规则。

## 改了什么

稀疏 Gaussian 线性模型的纠偏推断现在公开 `nodewise_alpha` 参数，适用于 `Lasso`、`ElasticNet`、相应的 Gaussian 惩罚模型基础接口，以及 `LassoCV` / `ElasticNetCV` 最终全数据重拟合时的推断配置。

主模型的 `alpha` 与 `nodewise_alpha` 作用不同：

- `alpha` 控制用于预测和变量选择的惩罚拟合；
- `nodewise_alpha` 只控制纠偏推断中、用于估计设计矩阵近似精度矩阵的逐节点 Lasso。

如果用户显式给出有限的正实数，该值直接生效；只有 `nodewise_alpha=None` 时才使用 statgpu 的自动规则。

## 默认规则为何调整

历史内部实现会把主响应模型的残差尺度带入逐节点惩罚：

$$
\hat\sigma_y\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

这个旧规则从未作为公开参数约定承诺给用户，而且会使本应由设计矩阵决定的近似精度矩阵估计依赖响应变量 `y` 的计量单位。

从计划发布的 **0.2.6** 起，自动规则改为：先对已经完成中心化和加权处理的工作设计做标准化，再取

$$
\lambda_{\mathrm{nw}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

无分析权重时 `n_nw=n`；存在非均匀分析权重时，`n_nw` 使用 Kish 型有效样本量。$\sqrt{\log(p)/n}$ 的量级来自高维逐节点回归的常见理论尺度，但具体常数和加权有效样本量的定义是 statgpu 的默认选择，并非某篇定理规定的唯一形式。

旧的、依赖响应尺度的内部规则不会作为兼容选项继续保留。若需要固定某个逐节点惩罚，应在标准化后的逐节点问题上显式设置 `nodewise_alpha=`。

## 近似精度矩阵如何构造

令 `X_w` 表示稀疏 Gaussian 推断路径中已经完成中心化和加权处理、并采用平均损失尺度的工作设计，定义

$$
d_j^2=\frac{1}{n}\sum_iX_{w,ij}^2,
\qquad Z=X_wD^{-1}.
$$

statgpu 在标准化设计 `Z` 上对每个特征求解逐节点 Lasso，并使用与相关文献记号一致的归一化量

$$
\hat\tau_j^2
=
\frac{\|r_j\|_2^2}{n}
+
\lambda_{\mathrm{nw}}\|\hat\gamma_j\|_1.
$$

得到标准化尺度上的近似精度矩阵后，再把它变换回原工作特征尺度。

逐节点求解器自身的停止条件并不是最终的数值检查。求解结束后，statgpu 会**独立重新计算完整的 KKT 残差**；只有这项检查通过，才继续采用该精度矩阵并生成推断结果。若特征尺度退化、精度矩阵出现非有限值、归一化量无效或 KKT 检查失败，推断会直接中止并报错，而不会像旧实现那样返回单位行占位结果。

当 `p=1` 时没有需要拟合的辅助逐节点回归。statgpu 直接使用一维解析精度矩阵，此时 `nodewise_alpha_` 保持为 `None`。

## 内部求解设置与溯源信息

逐节点 FISTA 的迭代容差故意设得比最终 KKT 检查更严格。目前内部使用 `coef_delta` 作为停止准则、`1e-8` 的迭代容差和最多 3000 次迭代；求解结束后还必须通过独立的 `1e-5` KKT 残差检查。这些都是内部数值设置，不是新增的公开调参参数。

多特征纠偏推断成功后，实际采用的逐节点惩罚值会记录在 `nodewise_alpha_` 中。`_inference_result.metadata` 还会保存用户请求值、实际解析值及其来源、自动规则标识、加权有效样本量、逐节点求解设置、最大 KKT 残差、缓存命中信息，以及实际执行数值计算的后端和设备，便于复现和排查。

## 交叉验证中的含义

`LassoCV(nodewise_alpha=...)` 与 `ElasticNetCV(nodewise_alpha=...)` 都只把该参数用于**最终全数据重拟合后的纠偏推断**。它不会进入主正则化参数的候选网格，也不会改变各折评分、最终 `alpha_`，或 `ElasticNetCV` 的 `l1_ratio_` 选择。

换句话说，`nodewise_alpha` 调的是“推断时如何估计近似精度矩阵”，不是“交叉验证如何选择预测模型”。

## 后端与权重

NumPy、CuPy 与 Torch 使用同一套标准化统计定义。显式选择 CUDA 或 Torch 时，逐节点求解及其 KKT 检查不会在数值计算阶段静默转到 CPU。

分析权重继续沿用既有的平均损失约定：所有权重同时乘以同一个正数不会改变目标统计问题；全 1 权重与不加权情形一致；自动逐节点调参对零权重观测的增删保持不变。

为了判断精度矩阵缓存是否可复用，缓存层可以对后端上的工作设计分块计算哈希；这只是缓存身份识别。逐节点求解、KKT 检查、精度矩阵尺度变换以及 GPU 上的同时推断仍在所选后端和设备上完成。

## 版本迁移建议

- **使用 0.2.5：** 没有公开 `nodewise_alpha` 参数；逐节点惩罚仍属于旧的内部实现细节。
- **升级到计划发布的 0.2.6：** 如果不传 `nodewise_alpha`，将使用本文所述、与响应尺度无关的新自动规则。
- **需要固定旧实验的数值行为：** 不建议依赖旧内部公式；应在新版本中根据标准化逐节点尺度显式设置 `nodewise_alpha`，并记录实际解析后的 `nodewise_alpha_`。

## 参考文献

- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217-242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869-2909.
