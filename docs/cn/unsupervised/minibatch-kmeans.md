# MiniBatchKMeans

> 语言：中文
> 最后更新：2026-09-29
> 路径：`statgpu.unsupervised.MiniBatchKMeans`

## 概览

`MiniBatchKMeans` 使用小批量更新聚类中心，避免每一步 Lloyd 迭代都扫描完整数据集。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import MiniBatchKMeans
```

## 目标函数

目标仍是 KMeans 的惯性：

$$
\sum_i \min_j \|x_i - c_j\|_2^2.
$$

## 估计方程

对每一批中被分配到簇 `j` 的样本，中心更新为：

$$
c_j \leftarrow c_j + \eta_j(\bar{x}_{B_j} - c_j),
\qquad
\eta_j = \frac{|B_j|}{n_j + |B_j|}.
$$

`fit` 在小批量更新结束后，会对完整的稠密数据再做少量精确的 Lloyd 打磨；主体训练仍是小批量方式，但最终惯性会更接近按全量数据分配标签时的中心。

## 参数

`n_clusters`、`init`、`n_init`、`batch_size`、`max_iter`、`max_no_improvement`、`tol`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import MiniBatchKMeans

labels = MiniBatchKMeans(n_clusters=20, batch_size=4096, device="cpu").fit_predict(X)
labels_gpu = MiniBatchKMeans(n_clusters=20, batch_size=4096, device="torch").fit_predict(X_torch)
```

## 严格与近似模式的差别

该方法属于随机近似优化；公平比较时应固定相同的初始中心、批次顺序、收敛阈值与迭代预算。

## 输出

`cluster_centers_`、`labels_`、`inertia_`、`n_iter_`、`n_steps_`、`counts_`、`n_features_in_`。

## FAQ

Phase 3A 仅支持稠密的欧氏输入；不支持稀疏输入、`sample_weight` 和可调用的 `init`。

## 外部验证

测试脚本：`dev/tests/test_unsupervised_minibatch_kmeans.py`。
基准测试：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
对齐基线：sklearn 的 `MiniBatchKMeans`。

## References

- Sculley, D. (2010). Web-scale k-means clustering. *Proceedings of the 19th International Conference on World Wide Web*, 1177-1178.

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [MiniBatchKMeans API 参考](api-reference.md#minibatchkmeans)。
