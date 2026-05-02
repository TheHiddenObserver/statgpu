# 无监督学习

> 语言：中文
> 最后更新：2026-05-02
> 本页：无监督模型文档兼容入口
> English: [English](../en/models/unsupervised.md)

## 概览

无监督学习的详细文档已经拆分到 [docs/unsupervised/](../unsupervised/README.md)。保留本页是为了让旧链接 `docs/models/unsupervised.md` 继续可用。

当前 public estimators：

- [PCA](../unsupervised/pca.md)
- [KMeans](../unsupervised/kmeans.md)
- [DBSCAN](../unsupervised/dbscan.md)
- [GaussianMixture](../unsupervised/gaussian-mixture.md)
- [NMF](../unsupervised/nmf.md)
- [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md)

## Phase 2 范围

`PCA`、`KMeans`、`DBSCAN`、`GaussianMixture(diag)` 和 `NMF` 支持 CPU、CuPy/CUDA、Torch CUDA 路径。`AgglomerativeClustering(single)` 仅支持 CPU；显式 GPU device 会报错。

这些无监督 estimator 没有 strict inference 模式，因为它们不报告 inference covariance、standard errors 或 p-values。这里需要区分的是算法精确性：PCA full/covariance 和 DBSCAN 支持的 CPU 路径是 exact；PCA randomized、KMeans、GMM、NMF 属于 iterative 或 approximate 行为，具体见逐模型页面。

## 外部验证

最新远程验证产物：

- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`

DBSCAN compact `n=5000` 最新结果：Cython CPU `219.62ms`，fallback CPU `379.80ms`，sklearn CPU `178.94ms`，CuPy `21.56ms`，Torch `21.07ms`；labels 的 ARI 为 `1.0`。这个结果接近 sklearn CPU，但没有严格达到 `<=1.2x` 目标，比例为 `1.23x`。
