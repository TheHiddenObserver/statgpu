# TSNE

> 语言：中文
> 最后更新：2026-09-29
> 路径：`statgpu.unsupervised.TSNE`

## 概览

`TSNE` 通过匹配高维高斯亲和度与低维 Student-t 亲和度来学习嵌入（embedding）。Phase 3A 实现稠密欧氏距离上的精确 t-SNE。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TSNE
```

## 目标函数

t-SNE 最小化 KL 散度：

$$
\operatorname{KL}(P \| Q)
= \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

## 估计方程

高维条件概率通过对带宽做二分搜索得到，使每行的概率满足目标困惑度（perplexity）。低维亲和度定义为：

$$
q_{ij} =
\frac{(1+\|y_i-y_j\|_2^2)^{-1}}
{\sum_{a \ne b}(1+\|y_a-y_b\|_2^2)^{-1}}.
$$

嵌入使用 early exaggeration、动量和逐坐标自适应增益进行优化。

## 参数

`n_components`、`perplexity`、`early_exaggeration`、`learning_rate`、`max_iter`、`init`、`random_state`、`metric`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import TSNE

embedding = TSNE(perplexity=30, device="cpu").fit_transform(X)
embedding_gpu = TSNE(perplexity=30, device="torch").fit_transform(X_torch)
```

## 严格与近似模式的差别

这里实现的是稠密数据上的精确 t-SNE；Barnes-Hut、FFT/FIt-SNE 与 openTSNE 等加速路径只作为外部对齐基线。

## 输出

`embedding_`、`kl_divergence_`、`n_iter_`、`n_features_in_`。

## FAQ

Phase 3A 不支持稀疏输入、非欧氏 `metric`、Barnes-Hut、FFT/FIt-SNE，也不支持对新样本调用 `transform`。

## 外部验证

测试脚本：`dev/tests/test_unsupervised_tsne.py`。
基准测试：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
对齐基线：sklearn 的精确 `TSNE`、`openTSNE`，以及远程环境可用时的 cuML TSNE。

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
