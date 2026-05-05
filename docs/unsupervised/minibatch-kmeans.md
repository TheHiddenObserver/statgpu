# MiniBatchKMeans

> 语言：中文
> 最后更新：2026-05-04
> 路径：`statgpu.unsupervised.MiniBatchKMeans`

## 概览

`MiniBatchKMeans` 使用小批量更新聚类中心，避免每一步 Lloyd 迭代都扫描完整数据集。

## 目标函数

目标仍是 KMeans inertia：

$$
\sum_i \min_j \|x_i - c_j\|_2^2.
$$

## 估计方程

对 batch 中被分到 cluster `j` 的样本，中心更新为：

$$
c_j \leftarrow c_j + \eta_j(\bar{x}_{B_j} - c_j),
\qquad
\eta_j = \frac{|B_j|}{n_j + |B_j|}.
$$

`fit` 在 mini-batch 更新结束后，会对完整 dense 数据做少量 exact Lloyd polishing。这样主体仍是 mini-batch 训练，但最终 inertia 会更接近完整数据标签分配下的中心。

## 参数

`n_clusters`、`init`、`n_init`、`batch_size`、`max_iter`、`max_no_improvement`、`tol`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import MiniBatchKMeans

labels = MiniBatchKMeans(n_clusters=20, batch_size=4096, device="cpu").fit_predict(X)
labels_gpu = MiniBatchKMeans(n_clusters=20, batch_size=4096, device="torch").fit_predict(X_torch)
```

## Strict/Approx Difference

该方法是随机近似优化。公平比较时应固定相同初始中心、batch 顺序、收敛阈值和迭代预算。

## 输出

`cluster_centers_`、`labels_`、`inertia_`、`n_iter_`、`n_steps_`、`counts_`、`n_features_in_`。

## FAQ

Phase 3A 仅支持 dense Euclidean 输入；不支持 sparse、sample_weight 和 callable init。

## 外部验证

测试：`dev/tests/test_unsupervised_minibatch_kmeans.py`。
Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
Baseline：sklearn `MiniBatchKMeans`。

## References

- Sculley, D. (2010). Web-scale k-means clustering. *Proceedings of the 19th International Conference on World Wide Web*, 1177-1178.
