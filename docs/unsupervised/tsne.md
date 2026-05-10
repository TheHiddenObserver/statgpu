# TSNE

> 语言：中文
> 最后更新：2026-05-09
> 路径：`statgpu.unsupervised.TSNE`

## 概览

`TSNE` 通过匹配高维 Gaussian affinity 和低维 Student-t affinity 来学习 embedding。Phase 3A 实现 dense exact Euclidean t-SNE。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TSNE
```

## 目标函数

t-SNE 最小化 KL divergence：

$$
\operatorname{KL}(P \| Q)
= \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

## 估计方程

高维条件概率通过二分搜索 bandwidth，使每行概率满足目标 perplexity。低维 affinity 定义为：

$$
q_{ij} =
\frac{(1+\|y_i-y_j\|_2^2)^{-1}}
{\sum_{a \ne b}(1+\|y_a-y_b\|_2^2)^{-1}}.
$$

embedding 使用 early exaggeration、momentum 和逐坐标 adaptive gains 优化。

## 参数

`n_components`、`perplexity`、`early_exaggeration`、`learning_rate`、`max_iter`、`init`、`random_state`、`metric`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import TSNE

embedding = TSNE(perplexity=30, device="cpu").fit_transform(X)
embedding_gpu = TSNE(perplexity=30, device="torch").fit_transform(X_torch)
```

## Strict/Approx Difference

这里实现的是 exact dense t-SNE。Barnes-Hut、FFT/FIt-SNE 和 openTSNE 加速路径只作为外部 baseline。

## 输出

`embedding_`、`kl_divergence_`、`n_iter_`、`n_features_in_`。

## FAQ

Phase 3A 不支持 sparse、非 Euclidean metric、Barnes-Hut、FFT/FIt-SNE 和新样本 `transform`。

## 外部验证

测试：`dev/tests/test_unsupervised_tsne.py`。
Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
Baseline：sklearn exact `TSNE`、`openTSNE`，以及远程可用时的 cuML TSNE。

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
