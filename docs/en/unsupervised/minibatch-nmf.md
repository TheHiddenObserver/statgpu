# MiniBatchNMF

> Language: English
> Last updated: 2026-05-07
> Switch: [Chinese](../../cn/unsupervised/minibatch-nmf.md)

## Overview

`MiniBatchNMF` fits a non-negative low-rank factorization from dense mini-batches. Phase 3C supports Frobenius loss with multiplicative-update style mini-batch updates on CPU, CuPy/CUDA, and Torch CUDA.

## Path

```python
from statgpu.unsupervised import MiniBatchNMF
```

## Objective Function / Loss Function

For non-negative `W` and `H`, the estimator minimizes mini-batch approximations to the Frobenius reconstruction loss:

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

## Estimating Equation

For each batch, `MiniBatchNMF` initializes or updates batch activations `W_batch` with fixed `H`, then updates `H` using multiplicative updates:

$$
W \leftarrow W \odot \frac{XH^\top}{WHH^\top + \epsilon},
\qquad
H \leftarrow H \odot \frac{W^\top X}{W^\top WH + \epsilon}.
$$

## Parameters

- `n_components`: factorization rank; `None` uses `min(n_samples, n_features)`.
- `init`: v1 supports `"random"`.
- `batch_size`, `max_iter`, `tol`, `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
from statgpu.unsupervised import MiniBatchNMF

nmf = MiniBatchNMF(n_components=8, batch_size=1024, max_iter=50, random_state=0, device="torch")
W = nmf.fit_transform(X)
X_hat = nmf.inverse_transform(W)
```

## Strict/Approx Difference

MiniBatchNMF is non-convex and mini-batch order dependent. It is intended for scalable approximate factorization, not strict statistical inference.

## Outputs

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**Does v1 support negative or sparse input?**
No. Inputs must be dense and non-negative.

**Does v1 support CD solver or other beta losses?**
No. Phase 3C supports MU-style updates and Frobenius loss only.

## External Validation

- Tests: `dev/tests/test_unsupervised_minibatch_nmf.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3c.py`.
- Latest remote artifact: `results/unsupervised_phase3c_opt7_20260507_185500.json`.
- Baseline: sklearn `MiniBatchNMF` with aligned rank, batch size, initialization, and iteration count.

## References

- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. *Advances in Neural Information Processing Systems*, 13.
- Cichocki, A., Zdunek, R., Phan, A. H., & Amari, S.-I. (2009). *Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation*. Wiley.
- scikit-learn Developers. `sklearn.decomposition.MiniBatchNMF`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
