# 无监督估计器 API 参考

> 语言：中文
> 最后更新：2026-10-06
> 页面定位：完整 API 参考
> 切换：[English](../../en/unsupervised/api-reference.md)

本页列出 `statgpu.unsupervised` 导出的十二个估计器的构造默认值、公开模型方法、返回值、拟合输出及重要限制。选择模型请先看[索引](README.md)；直觉、目标函数与统计文献见各模型指南。

## 公共输入、设备与状态

各类都可通过 `from statgpu.unsupervised import ClassName` 导入。下面用 `n` 表示训练行数，`p` 表示输入特征数，`m` 表示后续调用的行数，`k` 表示请求或实际采用的成分数、簇数。`X` 是非空、有限的稠密数值矩阵，形状为 `(n,p)` 或 `(m,p)`。这些估计器不提供公式或 dataframe 设计矩阵接口；分类编码和缺失值处理应在调用前完成。后续输入必须保持特征顺序与列数。`y=None` 是未参与拟合的 sklearn 兼容参数，不是监督目标。

- `device="cpu"` 使用 NumPy；`"cuda"` 请求 CuPy CUDA；`"torch"` 请求 Torch CUDA。显式请求的 GPU 不可用时会报错，不会悄悄改用 CPU。`"auto"` 通常结合全局设备配置及可用后端选择；AgglomerativeClustering 是例外，它即使在全局配置选择 GPU 时也保持 CPU 路径。要固定 CPU 示例请指定 `"cpu"`。参见[设备与内存](../guides/device-and-memory.md)。
- 数值计算通常采用 float64；UMAP 近邻搜索和 DBSCAN 的 GPU 距离计算在内部采用 float32。拟合数组与方法返回的数组通常留在所选后端。AgglomerativeClustering 即使在 GPU 上拟合，也发布 NumPy 标签与树数组；UMAP 的图输出为元组。整数标签是标识，不是连续预测值。
- 需要用于报告的 CPU 数组时，NumPy 数组用 `np.asarray(a)`，CuPy 用 `cupy.asnumpy(a)`，Torch 用 `a.detach().cpu().numpy()`。这些转换不一定创建独立副本：NumPy 数组及 CPU Torch 张量可能与结果共享存储。若需独立修改，应再对得到的 NumPy 数组调用 `.copy()`。GPU 转换可能引起数据传输及同步。评分与标量拟合诊断是主机端数值。数值超参数也必须有限；范围检查不一定能拒绝所有 NaN/Inf 设置。
- `fit(...)` 和支持的 `partial_fit(...)` 返回估计器自身。转换、预测和评分需在成功拟合后调用。普通 `fit` 从头拟合；只有下文说明的三个 `partial_fit` API 会累积批次。调用 `set_params(...)` 修改设置后，应重新拟合再使用结果。重新拟合失败不代表旧结果已被成功替换，实例中可能保留旧属性或只更新了一部分的属性；报错后应使用新估计器，并验证新的结果。
- 公共 `get_params(deep=True)` 返回配置字典；这些类的 `set_params(**params)` 返回 `self`，检查参数名并重置已拟合状态。完整继承签名、推断辅助方法限制及示例见[参数管理](../reference/estimator-api.md#parameter-management)和[通用推断辅助方法](../reference/estimator-api.md#inference-helpers)。继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic`、`permutation_test` 本身不能保证簇或成分推断有效。这十二个类均不提供模型专属 `summary()` 或系数标准误。
- 所有构造函数都接受 `n_jobs=None`。它保留为公共估计器配置，但当前无监督实现不会用它设置数值内核或线程并行度。因此这里的 `n_jobs=-1` 不意味着强制使用全部线程，也不构成加速保证。

返回数组的方法并不统一提供独立副本。`KMeans`、`MiniBatchKMeans`、`DBSCAN` 和 `AgglomerativeClustering` 的 `fit_predict` 直接返回其 `labels_` 对象；UMAP 与 TSNE 的 `fit_transform` 直接返回其 `embedding_` 对象；NMF 的 `fit_transform` 返回内部保存的联合拟合因子。应将这些数组及拟合属性视为只读；需要修改时，NumPy/CuPy 使用 `a.copy()`，Torch 使用 `a.clone()`。

## 随机性

整数种子会让随机拟合从该种子重新开始；NumPy `Generator` 和 `RandomState` 则持有可变状态，拟合可能推进它们的状态。重复使用同一个对象，不等于每次传入同一个整数。`None` 请求新的随机性，不代表固定种子。固定种子不保证不同后端或软件版本的结果完全相同。PCA 的 full/covariance 求解器与 TruncatedSVD 的 full 求解器不使用随机初始化。

PCA 的随机化求解器、NMF、MiniBatchNMF、随机化 TruncatedSVD、UMAP，以及采用随机初始化的 TSNE，都接受 NumPy `Generator` 和 `RandomState` 对象，并使用可跨后端传递的 32 位无符号整数种子。KMeans、MiniBatchKMeans 和 GaussianMixture 接受的种子形式则取决于当前版本的 [`numpy.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng)；例如，NumPy 2.3 可以转换旧式 `RandomState`，部分更早版本则不支持。需要兼容这些版本时，使用整数或 `Generator`。这些种子规则不会消除另行说明的 UMAP 谱初始化限制。

## 方法范围

某模型的方法表中未列出的操作，不属于它额外提供的模型能力。下方明确列出聚类或可视化模型中不支持的 `predict` / `transform` 占位方法。该模块没有专属 CV 估计器。应根据实际问题选择预处理、秩、簇数等设置，不应跨模型直接比较含义不同的 `score` 数值。

## PCA

[模型指南](pca.md)

```text
PCA(n_components=None, svd_solver='auto', whiten=False, copy=True, random_state=None, n_oversamples=10, iterated_power=2, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `None` | 取 `[1, min(n,p)]` 内的整数；`None` 保留 `min(n,p)` 个成分。 |
| `svd_solver` | `'auto'` | `"auto"`、`"full"`、`"covariance"` 或 `"randomized"`；`auto` 在 `n >= p` 时使用协方差特征分解，否则使用完整 SVD。 |
| `whiten` | `False` | 是否用已拟合成分的标准差缩放 PCA 坐标。 |
| `copy` | `True` | 兼容参数；即使为 `False`，实现也不会修改输入数据。 |
| `random_state` | `None` | `[0, 2**32-1]` 内的整数种子、`None`、NumPy `Generator` 或 NumPy `RandomState`；随机计算路径会使用它。见[随机性](#随机性)。 |
| `n_oversamples` | `10` | 非负整数，表示随机投影时额外使用的方向数。 |
| `iterated_power` | `2` | 随机化求解器的非负幂迭代次数。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 拟合中心化后的成分；至少需要两行。返回 `self`。 |
| `transform(X)` | 返回中心化后的成分坐标，形状 `(m,k)`；按配置进行白化。 |
| `fit_transform(X, y=None)` | 拟合并返回训练坐标 `(n,k)`。 |
| `inverse_transform(X)` | 输入坐标 `(m,k)`；撤销白化、投影回原空间并加回训练均值。返回 `(m,p)`。 |
| `predict(X)` | 等同于 `transform(X)`，不预测类别。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `components_` | 正交单位行方向 `(k,p)`；符号不唯一。 |
| `mean_` | 训练均值 `(p,)`。 |
| `explained_variance_`, `explained_variance_ratio_`, `singular_values_` | 三个 `(k,)` 向量：样本方差（除以 `n-1`）、占总方差的比例、奇异值。 |
| `n_components_`, `n_features_in_` | 实际成分数 `k` 与输入列数 `p`。 |

令 $W=\mathrm{components\_}$，$Z$ 为低维坐标，$\bar X=\mathrm{mean\_}$。不白化时，转换为 $(X-\bar X)W^\top$，逆转换为 $ZW+\bar X$。白化会将坐标除以 $\sqrt{\mathrm{explained\_variance\_}}$，重构时撤销该缩放。普通 PCA 保留零方差成分时，白化无定义，应避免保留这些成分。不提供 `partial_fit`、`score` 或概率推断方法。

## KMeans

[模型指南](kmeans.md)

```text
KMeans(n_clusters=8, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, random_state=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_clusters` | `8` | 不超过 `n` 的正整数。 |
| `init` | `'k-means++'` | `"k-means++"` 或 `"random"`；不支持显式中心数组及可调用初始化器。 |
| `n_init` | `'auto'` | 正整数或 `"auto"`：k-means++ 运行一次，random 运行十次。 |
| `max_iter` | `300` | 正整数迭代预算；迭代和整轮数据遍历的区别见对应模型。 |
| `tol` | `0.0001` | 非负收敛阈值；各模型采用的准则见下文。 |
| `random_state` | `None` | 整数种子、`None`、NumPy `Generator`，或当前 NumPy 版本的 `default_rng` 接受的其他种子；旧式输入随版本变化的说明见[随机性](#随机性)。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None, sample_weight=None)` | 拟合中心，返回 `self`；`sample_weight` 必须为 `None`。 |
| `fit_predict(X, y=None)` | 拟合并返回训练标签 `(n,)`。 |
| `transform(X)` | 返回到每个中心的欧氏距离 `(m,k)`，不是平方距离。 |
| `predict(X)` | 返回最近中心的标签 `(m,)`。 |
| `score(X, y=None)` | 返回 Python 浮点数：到最近中心的平方距离之和的负值；越大越好。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `cluster_centers_` | 中心 `(k,p)`。 |
| `labels_` | 训练整数标签 `(n,)`，范围 `0` 到 `k-1`。 |
| `inertia_` | 训练集簇内平方距离之和，为浮点数。 |
| `n_iter_`, `n_features_in_` | 所选运行的 Lloyd 迭代数与输入列数。 |

`tol` 限制中心移动的平方和，不是目标函数的相对误差。`max_iter` 分别限制每次重启。评分为 $-\sum_i\min_j\|x_i-c_j\|^2$，其中 $x_i$ 为观测，$c_j$ 为已拟合中心。比较时必须使用相同观测和特征缩放。不提供 `partial_fit`；增量更新请用 MiniBatchKMeans。

## DBSCAN

[模型指南](dbscan.md)

```text
DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', batch_size=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `eps` | `0.5` | 正且有限的欧氏邻域半径；非有限值当前不一定会被拒绝。 |
| `min_samples` | `5` | 正整数，最低邻居数包含观测本身。 |
| `metric` | `'euclidean'` | 仅支持 `"euclidean"`；不支持其他距离及预先计算的距离矩阵。 |
| `algorithm` | `'auto'` | `"auto"`、`"brute"`、`"ball_tree"` 或 `"kd_tree"`；用于高维 CPU 的 scikit-learn 近邻搜索，并非控制所有后端。 |
| `batch_size` | `None` | 正整数或 `None`；在使用分批距离计算的 GPU 路径上控制批量大小。`None` 自动选择。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 寻找训练数据的密度连通组，返回 `self`。 |
| `fit_predict(X, y=None)` | 拟合并返回训练标签 `(n,)`。 |
| `predict(X)` | 始终抛出 `NotImplementedError`；不提供新观测的归类规则。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `labels_` | 训练整数标签 `(n,)`；`-1` 表示噪声。 |
| `core_sample_indices_` | 核心训练观测的行索引 `(n_core,)`。 |
| `components_` | 核心观测 `(n_core,p)`，不是聚类中心。 |
| `n_features_in_` | 训练输入列数 `p`。 |

CPU 输入超过 12 个特征时使用 scikit-learn 的 `NearestNeighbors`，更低维时使用 SciPy 树搜索。GPU 距离采用 float32，核心样本采用 float64；两条 GPU 路径均包含主机传输或辅助处理。很大的共同偏移可能使 GPU float32 距离丢失样本间距；应先以 float64 减去训练数据确定的偏移，再拟合，保持 `eps` 不变。当前未编译扩展的 CPU 路径可能错误合并互不连通的孤立核心点，使用这类配置前请查看模型指南。不提供 `transform`、`score` 或 `partial_fit`。

## GaussianMixture

[模型指南](gaussian-mixture.md)

```text
GaussianMixture(n_components=1, covariance_type='diag', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', random_state=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `1` | 不超过 `n` 的正整数混合成分数。 |
| `covariance_type` | `'diag'` | `"diag"`、`"spherical"`、`"tied"` 或 `"full"`；决定下方协方差数组的形状。 |
| `tol` | `0.001` | 非负收敛阈值；各模型采用的准则见下文。 |
| `reg_covar` | `1e-06` | 非负协方差正则化量；为零时可能出现奇异协方差错误或 NaN 拟合结果，退化数据应保留正值。对角下限与完整矩阵加岭项的区别见模型指南。 |
| `max_iter` | `100` | 正整数迭代预算；迭代和整轮数据遍历的区别见对应模型。 |
| `n_init` | `1` | EM 重启的正整数次数；保留拟合下界最大的结果。 |
| `init_params` | `'kmeans'` | 均值初始化采用 `"kmeans"` 或 `"random"`。 |
| `random_state` | `None` | 整数种子、`None`、NumPy `Generator`，或当前 NumPy 版本的 `default_rng` 接受的其他种子；旧式输入随版本变化的说明见[随机性](#随机性)。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 用 EM 拟合混合模型，返回 `self`。 |
| `fit_predict(X, y=None)` | 拟合并返回训练数据中责任概率最大的成分标签 `(n,)`。 |
| `predict_proba(X)` | 返回责任概率 `(m,k)`；每行和为一。 |
| `predict(X)` | 返回责任概率最大的成分标签 `(m,)`。 |
| `score_samples(X)` | 返回每个观测的混合密度对数 `(m,)`。 |
| `score(X, y=None)` | 返回平均对数密度，为 Python 浮点数；越大越好。 |
| `aic(X)` | 返回所给观测的 AIC；输入需为具有 `.shape` 的数组，越小越好。 |
| `bic(X)` | 返回所给观测的 BIC；输入需为具有 `.shape` 的数组，越小越好。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `weights_`, `means_` | 混合权重 `(k,)` 与均值 `(k,p)`。 |
| `covariances_`, `precisions_cholesky_` | 协方差及精度因子：diag 为 `(k,p)`，spherical 为 `(k,)`，tied 为 `(p,p)`，full 为 `(k,p,p)`。diag/spherical 的精度因子是标准差的倒数；full/tied 的精度因子为下三角矩阵 `L`，满足 `L @ L.T = inv(covariance)`。 |
| `converged_`, `n_iter_`, `lower_bound_`, `n_features_in_` | 收敛标志、EM 迭代数、最后一次 M 步之前监测到的平均对数似然及特征数。解释评分前先检查收敛情况。 |

`tol` 控制平均对数似然变化的绝对值，`max_iter` 限制每次重启。令总对数似然 $L=m\,\mathrm{score}(X)$，自由参数数目为 $d$，则 $\mathrm{AIC}=2d-2L$，$\mathrm{BIC}=d\log m-2L$。其中 $d=kp+(k-1)+d_{\mathrm{cov}}$，协方差参数数目分别为 $kp$（diag）、$k$（spherical）、$p(p+1)/2$（tied）或 $kp(p+1)/2$（full）。不同候选项应使用相同数据比较。不提供 `transform`、`partial_fit` 或系数推断 API。

## NMF

[模型指南](nmf.md)

```text
NMF(n_components=None, init='random', solver='mu', beta_loss='frobenius', max_iter=200, tol=0.0001, random_state=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `None` | 正整数秩；`None` 采用 `min(n,p)`。 |
| `init` | `'random'` | 仅支持 `"random"`。 |
| `solver` | `'mu'` | 仅支持乘性更新 `"mu"`。 |
| `beta_loss` | `'frobenius'` | 仅支持 `"frobenius"`。 |
| `max_iter` | `200` | 正整数迭代预算；迭代和整轮数据遍历的区别见对应模型。 |
| `tol` | `0.0001` | 非负收敛阈值；各模型采用的准则见下文。 |
| `random_state` | `None` | `[0, 2**32-1]` 内的整数种子、`None`、NumPy `Generator` 或 NumPy `RandomState`；随机计算路径会使用它。见[随机性](#随机性)。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 拟合非负因子，返回 `self`。 |
| `fit_transform(X, y=None)` | 返回训练因子 `W`，形状 `(n,k)`。 |
| `transform(X)` | 固定已学到的成分，为新数据求非负因子，返回 `(m,k)`。 |
| `inverse_transform(X)` | 将输入因子 `(m,k)` 乘以成分 `(k,p)`，返回重构 `(m,p)`。允许负坐标，也可能产生负的重构值。 |
| `predict(X)` | 等同于 `transform(X)`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `components_` | 非负字典 `H`，形状 `(k,p)`。 |
| `reconstruction_err_` | 训练残差 `X-WH` 的 Frobenius 范数，为浮点数（不是平方损失）。 |
| `n_iter_`, `n_components_`, `n_features_in_` | 拟合迭代次数、秩与输入列数。 |

拟合和转换的数据必须非负。`tol` 在 `fit` 的定期检查中比较重构误差的相对变化；`transform` 固定成分执行 `max_iter` 次更新，不会按 `tol` 提前停止。`fit_transform` 返回联合拟合的 `W`；之后的 `transform(X)` 固定 `H` 重新求解 `W`，两者不保证相同。不提供 `score` 或 `partial_fit`。

输入采用极小的正数单位时，固定绝对稳定项可能主导更新并使因子塌缩，`tol=0` 也不能避免。应使用由训练数据确定的同一个正尺度处理所有特征、批次及后续转换，再把重构转回原始单位；见[极小单位的注意事项](nmf.md#极小数值单位)。除了绝对误差诊断，还应检查相对重构误差。

## AgglomerativeClustering

[模型指南](agglomerative-clustering.md)

```text
AgglomerativeClustering(n_clusters=2, linkage='single', metric='euclidean', device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_clusters` | `2` | 请求的簇数，为不超过 `n` 的正整数；CPU 按 maxclust 切分时，合并高度并列可能产生更少的组。 |
| `linkage` | `'single'` | `"single"`、`"complete"`、`"average"` 或 `"ward"`。 |
| `metric` | `'euclidean'` | 仅支持 `"euclidean"`；不支持其他距离及预先计算的距离矩阵。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 构建稠密层次结构并切分，返回 `self`。 |
| `fit_predict(X, y=None)` | 拟合并返回训练标签 `(n,)`。 |
| `predict(X)` | 不支持新观测预测，始终抛出 `NotImplementedError`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `labels_` | 训练整数标签 `(n,)`；即使在 GPU 上拟合，也返回 NumPy 数组。 |
| `children_`, `distances_` | NumPy 合并节点对 `(n-1,2)` 与合并距离 `(n-1,)`；叶节点编号为 `0..n-1`，第 `i` 行合并节点编号为 `n+i`。 |
| `n_features_in_` | 训练输入列数。 |

单个观测且 `n_clusters=1` 时合并树为空。GPU 路径使用稠密成对距离，估计距离矩阵超过 `STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES`（默认 1 GiB）时会报错；该值是配置上限，不是对可用显存的测量。GPU 路径的展开距离公式在共同偏移很大时可能丢失较小的样本间距；应先以 float64 中心化，再拟合。不提供稀疏连通约束、`transform`、`score` 或 `partial_fit`。

## TruncatedSVD

[模型指南](truncated-svd.md)

```text
TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, n_oversamples=10, random_state=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `2` | `[1,min(n,p)]` 范围内的整数秩。 |
| `algorithm` | `'randomized'` | `"randomized"` 或 `"full"`。 |
| `n_iter` | `5` | 随机化方法的非负幂迭代次数。 |
| `n_oversamples` | `10` | 非负整数，表示随机投影时额外使用的方向数。 |
| `random_state` | `None` | `[0, 2**32-1]` 内的整数种子、`None`、NumPy `Generator` 或 NumPy `RandomState`；随机计算路径会使用它。见[随机性](#随机性)。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 拟合不中心化的低秩成分，返回 `self`。 |
| `fit_transform(X, y=None)` | 拟合并返回训练坐标 `(n,k)`。 |
| `transform(X)` | 返回不中心化投影 `X @ components_.T`，形状 `(m,k)`。 |
| `inverse_transform(X)` | 由坐标 `(m,k)` 返回 `X @ components_`，形状 `(m,p)`；不加均值。 |
| `predict(X)` | 等同于 `transform(X)`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `components_`, `singular_values_` | 行成分 `(k,p)` 与奇异值 `(k,)`。 |
| `explained_variance_`, `explained_variance_ratio_` | 训练投影列的总体方差（除以 `n`）及其占原始各列方差和的比例，均为 `(k,)`。 |
| `n_components_`, `n_features_in_` | 实际秩与训练列数。 |

仅支持稠密输入，不能替代 scikit-learn TruncatedSVD 的稀疏矩阵用途。投影不中心化，但报告方差时会减去各列均值。不提供白化、`score` 或 `partial_fit`。

## MiniBatchKMeans

[模型指南](minibatch-kmeans.md)

```text
MiniBatchKMeans(n_clusters=8, init='k-means++', n_init='auto', batch_size=1024, max_iter=100, max_no_improvement=10, tol=0.0, random_state=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_clusters` | `8` | 正整数；普通 `fit` 至少需要这么多行。 |
| `init` | `'k-means++'` | `"k-means++"`、`"random"` 或全部有限的显式中心数组 `(k,p)`；不支持可调用初始化器。应自行检查数组有限性，无效中心可能使 `partial_fit` 结果错误而不报错。 |
| `n_init` | `'auto'` | 正整数或 `"auto"`：k-means++ 或显式中心运行一次，random 运行三次。用于 `fit`，不用于反复重启 `partial_fit`。 |
| `batch_size` | `1024` | `fit` 内每批的正整数上限；`partial_fit` 每次处理传入的整个批次。 |
| `max_iter` | `100` | 正整数迭代预算；迭代和整轮数据遍历的区别见对应模型。 |
| `max_no_improvement` | `10` | 连续未刷新最佳批次惯性的批次数上限，为非负整数；`None` 关闭该停止规则。 |
| `tol` | `0.0` | 非负收敛阈值；各模型采用的准则见下文。 |
| `random_state` | `None` | 整数种子、`None`、NumPy `Generator`，或当前 NumPy 版本的 `default_rng` 接受的其他种子；旧式输入随版本变化的说明见[随机性](#随机性)。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None, sample_weight=None)` | 从整个数组重新拟合，再用全量数据的 Lloyd 步骤改善中心；返回 `self`。`sample_weight` 必须为 `None`。 |
| `partial_fit(X, y=None, sample_weight=None)` | 用本批次更新一次中心及计数，返回 `self`。`sample_weight` 必须为 `None`。 |
| `fit_predict(X, y=None)` | 拟合并返回整个训练集的标签 `(n,)`。 |
| `transform(X)` | 返回到全部中心的距离 `(m,k)`。 |
| `predict(X)` | 返回最近中心的标签 `(m,)`。 |
| `score(X, y=None)` | 返回到最近中心的平方距离和的负值，为 Python 浮点数。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `cluster_centers_`, `counts_` | 中心 `(k,p)` 与各中心计数 `(k,)`；`fit` 返回最终全量分配计数，`partial_fit` 累积批次分配计数。 |
| `labels_`, `inertia_` | `fit` 后对应全部训练数据；`partial_fit` 后只对应最近一个批次。标签长度等于最近一次输入的行数。 |
| `n_iter_`, `n_steps_`, `n_features_in_` | `fit` 后为已进入的整轮数（最后一轮可能提前中止）、批次更新次数及特征数；每次 `partial_fit` 将前两个计数各加一。 |

使用字符串初始化方式时，首次 `partial_fit` 至少需要 `n_clusters` 行；提供显式中心则可使用更小的首批。后续批次可以更小，但特征列数及顺序必须一致。`max_iter`、`tol`（中心移动平方和）和 `max_no_improvement` 控制 `fit`，不会让单次 `partial_fit` 变成整轮迭代。`partial_fit` 不会执行全量数据的改善步骤。它的 `labels_` 与 `inertia_` 记录移动中心之前用于更新的批次分配；要评价更新后的中心，应重新调用 `predict(batch)` 和 `-score(batch)`。

## IncrementalPCA

[模型指南](incremental-pca.md)

```text
IncrementalPCA(n_components=None, batch_size=None, whiten=False, copy=True, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `None` | 不超过 `p` 的正整数秩；`None` 时，`fit` 使用 `min(n,p)`，而 `partial_fit` 由首批固定为 `min(首批行数,p)`。 |
| `batch_size` | `None` | 正整数或 `None`；默认 `fit` 一次处理全部行，会扩充不足以容纳实际采用秩的首批。不会切分传给 `partial_fit` 的数组。 |
| `whiten` | `False` | 是否用已拟合成分的标准差缩放 PCA 坐标。 |
| `copy` | `True` | 兼容参数；即使为 `False`，实现也不会修改输入数据。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 重置增量状态并分批处理本数据集，返回 `self`。 |
| `partial_fit(X, y=None)` | 用全部传入行更新分解，返回 `self`。 |
| `fit_transform(X, y=None)` | 拟合后用最终基转换全部训练行，返回 `(n,k)`。 |
| `transform(X)` | 减去累计均值后投影，可选白化；返回 `(m,k)`。 |
| `inverse_transform(X)` | 撤销白化，重构坐标 `(m,k)` 并加回累计均值；返回 `(m,p)`。 |
| `predict(X)` | 等同于 `transform(X)`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `components_`, `singular_values_` | 保留的行方向 `(k,p)` 与奇异值 `(k,)`。 |
| `mean_`, `var_` | 累计特征均值与总体方差 `(p,)`。 |
| `explained_variance_`, `explained_variance_ratio_` | 保留方差（可用时除以 `n_samples_seen_-1`）及比例，形状 `(k,)`。 |
| `n_samples_seen_`, `n_components_`, `n_features_in_` | 累计处理行数、保留秩与特征列数。 |

首次 `partial_fit` 的行数至少应达到显式指定的 `n_components`，后续批次可更小。特征列数必须固定。`n_components=None` 时，小首批也会限制保留秩，之后不会自动增加。截断和批次顺序会影响结果。白化对方差使用很小的下限以保持数值稳定。不提供 `score`。

## MiniBatchNMF

[模型指南](minibatch-nmf.md)

```text
MiniBatchNMF(n_components=None, init='random', batch_size=None, max_iter=200, tol=0.0001, random_state=None, device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `None` | 正整数秩；`None` 时，`fit` 使用 `min(n,p)`，`partial_fit` 则由首批确定秩。 |
| `init` | `'random'` | 仅支持 `"random"`。 |
| `batch_size` | `None` | 正整数或 `None`；控制 `fit` 内的批次。`None` 根据数据量选择，可能一次处理全部数据。`partial_fit` 处理所传入的批次。 |
| `max_iter` | `200` | 正整数迭代预算；迭代和整轮数据遍历的区别见对应模型。 |
| `tol` | `0.0001` | 非负收敛阈值；各模型采用的准则见下文。 |
| `random_state` | `None` | `[0, 2**32-1]` 内的整数种子、`None`、NumPy `Generator` 或 NumPy `RandomState`；随机计算路径会使用它。见[随机性](#随机性)。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 从头拟合非负因子，返回 `self`。 |
| `partial_fit(X, y=None)` | 累积本批次的因子统计量并更新成分，返回 `self`。 |
| `fit_transform(X, y=None)` | 拟合成分，再固定成分求训练因子 `(n,k)`。 |
| `transform(X)` | 固定已学成分，返回非负因子 `(m,k)`。 |
| `inverse_transform(X)` | 将因子 `(m,k)` 乘以成分，返回 `(m,p)`。允许负坐标，也可能产生负的重构值。 |
| `predict(X)` | 等同于 `transform(X)`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `components_` | 非负字典 `(k,p)`。 |
| `reconstruction_err_` | 拟合因子的残差 Frobenius 范数；`fit` 后对应全量数据，`partial_fit` 后对应最近批次。之后的 `transform` 可能进一步改善因子并产生不同误差。 |
| `n_iter_`, `n_components_`, `n_features_in_` | 拟合的整轮遍历次数（或增量更新次数）、实际秩与固定输入列数。 |

使用非负稠密数据，并保持特征列数与顺序。显式指定的正整数秩不要求小于首批行数；`None` 则由首批决定。`max_iter` 限制 `fit` 的整轮遍历次数，并影响固定成分后的转换求解；`tol` 检查拟合中成分的相对变化，而不是重构误差的相对变化。两者均不控制 `partial_fit` 内的收敛循环。首批中全零的特征可能使字典对应列永久为零，即使后续批次出现正值也无法恢复。应缓冲有代表性的初始化数据；字典全零列对应的特征后来出现正值时，需用代表性保留数据重新拟合。详见[首批注意事项](minibatch-nmf.md#首批中全零特征的限制)。不提供 `score` 或样本权重参数。

输入采用极小的正数单位时，固定绝对稳定项可能主导更新并使因子塌缩，`tol=0` 也不能避免。应使用由训练数据确定的同一个正尺度处理所有特征、批次及后续转换，再把重构转回原始单位；见[极小单位的注意事项](minibatch-nmf.md#极小数值单位)。除了绝对误差诊断，还应检查相对重构误差。

## UMAP

[模型指南](umap.md)

```text
UMAP(n_neighbors=15, n_components=2, metric='euclidean', min_dist=0.1, spread=1.0, n_epochs=None, learning_rate=1.0, init='spectral', negative_sample_rate=5, repulsion_strength=1.0, random_state=None, nn_method='auto', device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_neighbors` | `15` | `[2,n-1]` 内的整数，局部邻域大小。 |
| `n_components` | `2` | 小于 `n` 的正整数嵌入维数；当前 CPU 一维嵌入会失败，因此 CPU 请至少使用两个维度。 |
| `metric` | `'euclidean'` | 仅支持 `"euclidean"`；不支持其他距离及预先计算的距离矩阵。 |
| `min_dist` | `0.1` | 非负低维紧凑程度参数，应结合 `spread` 理解。 |
| `spread` | `1.0` | 低维吸引曲线的正尺度。 |
| `n_epochs` | `None` | 正整数或 `None`；当前自动规则为 `n<=2000` 时 500，`n<=10000` 时 200，否则 100。实际值见 `n_epochs_`。 |
| `learning_rate` | `1.0` | 优化的正初始步长。 |
| `init` | `'spectral'` | `"spectral"`（主机端 SciPy 特征求解）或 `"random"`；随机初始化可避开下文说明的稀疏谱初始化限制。 |
| `negative_sample_rate` | `5` | 正整数；每轮独立抽取 `n * negative_sample_rate` 对源点与目标点，并非每条吸引边抽取这么多对。 |
| `repulsion_strength` | `1.0` | 正的排斥力系数。 |
| `random_state` | `None` | `[0, 2**32-1]` 内的整数种子、`None`、NumPy `Generator` 或 NumPy `RandomState`；随机计算路径会使用它。见[随机性](#随机性)。 |
| `nn_method` | `'auto'` | `"auto"`、`"exact"` 或 `"nndescent"`；精确搜索使用稠密距离，NNDescent 为近似搜索，auto 当前始终选择精确搜索。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 拟合训练嵌入，返回 `self`。 |
| `fit_transform(X, y=None)` | 拟合并返回 `embedding_`，形状 `(n,k)`。 |
| `transform(X)` | 始终抛出 `NotImplementedError`，包括原始训练数据；原始行请读取 `embedding_`。 |
| `predict(X)` | 始终抛出 `NotImplementedError`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `embedding_` | 所选后端上的训练坐标 `(n,k)`。 |
| `graph_` | 元组 `(source_rows, target_rows, edge_weights, n_samples)`，不是 SciPy 邻接矩阵。前三项为长度相同的后端边数组。权重采用[图权重](umap.md#图权重)中说明的平均超出距离带宽与模糊并集，并非 umap-learn 的局部隶属度总和校准。 |
| `n_epochs_`, `n_features_in_` | 实际训练轮数与原始特征数。 |

近邻距离在内部采用 float32，嵌入优化采用 float64。很大的共同特征偏移可能在转为 float32 或计算展开距离时丢失细小间距；应在数据仍为 float64 时，先减去由训练数据确定的偏移，再拟合。即使输入已中心化，极大的成对距离仍可能使精确搜索选中样本自身，之后删除自环便会丢失邻居。中心化后应把所有特征除以由训练数据确定的同一个正尺度，使坐标适中，并对需要比较的数据复用此处理。统一缩放保持欧氏近邻顺序，与逐特征分别缩放不同。即使选择 GPU，图组装、谱初始化和吸引曲线拟合仍使用主机端 SciPy。带种子的随机初始化适合验证形状与接口，但图形质量仍需单独检查。当前力更新近似构造近邻布局，但不是标准 UMAP 交叉熵的精确梯度。在 NumPy 2 上，CPU 的 `nn_method="nndescent"` 当前会在后端分派时失败，可改用 `"exact"` 或 `"auto"`。稀疏谱初始化可能保留常量图特征向量，而漏掉一个有效方向；其特征求解器的起始向量也不受 `random_state` 控制。需要按种子初始化时，应使用 `init="random"`。不提供逆转换、评分或增量拟合。

## TSNE

[模型指南](tsne.md)

```text
TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, init='pca', random_state=None, metric='euclidean', device='auto', n_jobs=None)
```

| 参数 | 默认值 | 含义与可接受值 |
|---|---|---|
| `n_components` | `2` | 小于 `n` 的正整数嵌入维数；`init="pca"` 还要求不超过 `min(n,p)`。 |
| `perplexity` | `30.0` | 正的邻域困惑度目标，必须严格小于 `n`。这是 API 的范围检查；可达到的目标位于 `[1,n-1]`，并列最近邻还可能抬高下限，详见模型指南。 |
| `early_exaggeration` | `12.0` | 优化早期高维亲和度的正倍数。 |
| `learning_rate` | `'auto'` | 正数或 `"auto"`；自动值为 `max(n / early_exaggeration / 4, 10)`，不保证与其他库相同。 |
| `max_iter` | `1000` | 至少 250 的整数，表示优化总迭代数。 |
| `init` | `'pca'` | `"pca"` 或 `"random"`；不接受用户提供的初始嵌入数组。 |
| `random_state` | `None` | `[0, 2**32-1]` 内的整数种子、`None`、NumPy `Generator` 或 NumPy `RandomState`；随机计算路径会使用它。见[随机性](#随机性)。 |
| `metric` | `'euclidean'` | 仅支持 `"euclidean"`；不支持其他距离及预先计算的距离矩阵。 |
| `device` | `'auto'` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"`（Torch CUDA）；见本页公共设备说明。 |
| `n_jobs` | `None` | 公共 CPU 并行配置参数；这些实现目前不通过它控制数值计算内核，也不保证指定的线程数。 |

| 方法签名 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 拟合精确的稠密欧氏 t-SNE，返回 `self`。 |
| `fit_transform(X, y=None)` | 拟合并返回训练嵌入 `(n,k)`。 |
| `transform(X)` | 始终抛出 `NotImplementedError`；训练行请保留 `embedding_`。 |
| `predict(X)` | 始终抛出 `NotImplementedError`。 |

| 拟合输出 | 含义与形状 |
|---|---|
| `embedding_` | 所选后端上的训练坐标 `(n,k)`。 |
| `kl_divergence_` | 最终高低维亲和度的 KL 目标值，为 Python 浮点数；不是通用留出集评分。 |
| `n_iter_`, `n_features_in_` | 实际迭代次数（配置的预算）与原始输入列数。 |

该实现分配稠密成对数组，内存随样本数平方增长。不提供 Barnes–Hut/FFT 求解器选项、稀疏或预计算距离支持、逆转换、评分或增量拟合。不要未经核对就照搬其他库的学习率或困惑度设置。当前亲和度带宽搜索在特征尺度极大或极小时可能失败，甚至返回无效的负 KL 值。应先减去由训练数据确定的偏移，再缩放到适中的数值范围：共同偏移很大时，展开距离公式可能破坏亲和度，即使 KL 仍有限且非负。有限性与 KL 非负性是必要而不充分的检查；详见 [TSNE 数值注意事项](tsne.md)。

## 增量示例

下面的首批行数满足指定秩与簇数要求；后续批次保持相同特征顺序。

```python
# Example: incremental_unsupervised_api
import numpy as np
from statgpu.unsupervised import IncrementalPCA, MiniBatchKMeans, MiniBatchNMF

rng = np.random.default_rng(24)
X = rng.normal(size=(36, 4))
positive = np.abs(X) + 0.1
ipca = IncrementalPCA(n_components=2, device="cpu")
km = MiniBatchKMeans(n_clusters=3, random_state=24, device="cpu")
nmf = MiniBatchNMF(n_components=2, random_state=24, device="cpu")
for start in range(0, len(X), 12):
    ipca.partial_fit(X[start:start + 12])
    km.partial_fit(X[start:start + 12])
    nmf.partial_fit(positive[start:start + 12])
Z = ipca.transform(X[:5])
reconstructed = ipca.inverse_transform(Z)
labels = km.predict(X[:5])
factors = nmf.transform(positive[:5])
print("PCA coordinates/reconstruction:", Z.shape, reconstructed.shape)
print("Total rows / latest cluster labels:", ipca.n_samples_seen_, km.labels_.shape)
print("New labels / NMF factors:", labels.shape, factors.shape)
```

预期形状为 `(5,2)`、`(5,4)`；PCA 累计行数为 `36`，而 `km.labels_` 的形状是 `(12,)`，只覆盖最近一个批次。新标签为 `(5,)`，NMF 因子为 `(5,2)`。需要全部已处理观测的标签时，应调用 `km.predict(all_rows)`。这些形状检查说明接口行为，不衡量模型质量或 GPU 性能。
