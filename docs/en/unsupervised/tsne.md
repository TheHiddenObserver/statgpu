# TSNE

> Language: English
> Last updated: 2026-05-04
> Path: `statgpu.unsupervised.TSNE`

## Overview

`TSNE` embeds dense data by matching high-dimensional Gaussian affinities with low-dimensional Student-t affinities. Phase 3A implements exact dense Euclidean t-SNE.

## Objective Function / Loss Function

t-SNE minimizes KL divergence:

$$
\operatorname{KL}(P \| Q)
= \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

## Estimating Equation

The high-dimensional conditional probabilities are calibrated by binary search so each row matches the target perplexity. The low-dimensional affinities use:

$$
q_{ij} =
\frac{(1+\|y_i-y_j\|_2^2)^{-1}}
{\sum_{a \ne b}(1+\|y_a-y_b\|_2^2)^{-1}}.
$$

The embedding is optimized with early exaggeration, momentum, and adaptive per-coordinate gains.

## Parameters

`n_components`, `perplexity`, `early_exaggeration`, `learning_rate`, `max_iter`, `init`, `random_state`, `metric`, and `device`.

## CPU+GPU Examples

```python
from statgpu.unsupervised import TSNE

embedding = TSNE(perplexity=30, device="cpu").fit_transform(X)
embedding_gpu = TSNE(perplexity=30, device="torch").fit_transform(X_torch)
```

## Strict/Approx Difference

This is exact dense t-SNE. Barnes-Hut, FFT/FIt-SNE, and openTSNE acceleration are external baselines only.

## Outputs

`embedding_`, `kl_divergence_`, `n_iter_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, Barnes-Hut, FFT/FIt-SNE, and new-data `transform` are not supported in Phase 3A.

## External Validation

Tests: `dev/tests/test_unsupervised_tsne.py`.
Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3.py`.
Baselines: sklearn exact `TSNE`, `openTSNE`, and cuML TSNE if available remotely.

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
