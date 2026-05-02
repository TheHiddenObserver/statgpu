# Unsupervised Learning

> Language: English
> Last updated: 2026-05-02
> This page: compatibility entry for unsupervised model docs
> Switch: [Chinese](../../models/unsupervised.md)

## Overview

The detailed unsupervised documentation has moved to [docs/en/unsupervised/](../unsupervised/README.md). This page is kept so existing links to `docs/en/models/unsupervised.md` remain valid.

Current public estimators:

- [PCA](../unsupervised/pca.md)
- [KMeans](../unsupervised/kmeans.md)
- [DBSCAN](../unsupervised/dbscan.md)
- [GaussianMixture](../unsupervised/gaussian-mixture.md)
- [NMF](../unsupervised/nmf.md)
- [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md)

## Phase 2 Scope

`PCA`, `KMeans`, `DBSCAN`, `GaussianMixture(diag)`, and `NMF` support CPU, CuPy/CUDA, and Torch CUDA paths. `AgglomerativeClustering(single)` is CPU-only and raises for explicit GPU devices.

There is no strict inference mode for these unsupervised estimators because they do not report inference covariance, standard errors, or p-values. The relevant distinction is algorithmic exactness: PCA full/covariance and DBSCAN supported CPU paths are exact; PCA randomized, KMeans, GMM, and NMF are iterative or approximate where documented on each model page.

## External Validation

Latest remote artifacts:

- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`

DBSCAN compact `n=5000` latest result: Cython CPU `219.62ms`, fallback CPU `379.80ms`, sklearn CPU `178.94ms`, CuPy `21.56ms`, Torch `21.07ms`; labels match with ARI `1.0`. This is near sklearn CPU but not a strict pass for the `<=1.2x` target (`1.23x`).
