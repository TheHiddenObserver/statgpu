# NMF

> Language: English
> Last updated: 2026-05-02
> Switch: [Chinese](../../cn/unsupervised/nmf.md)

## Overview

`NMF` factorizes non-negative dense data into non-negative factors `W` and `H`. Phase 2 supports multiplicative updates with Frobenius loss on CPU, CuPy/CUDA, and Torch CUDA.

## Path

```python
from statgpu.unsupervised import NMF
```

## Objective Function / Loss Function

The fitted factors solve the non-convex constrained problem:

$$
\min_{W \ge 0,\; H \ge 0}
\frac{1}{2}\left\|X - WH\right\|_F^2 .
$$

`components_` stores `H`; `fit_transform` returns `W`.

## Estimating Equation

The implementation uses multiplicative updates:

$$
W \leftarrow W \odot
\frac{XH^\top}{WHH^\top + \varepsilon}
$$

$$
H \leftarrow H \odot
\frac{W^\top X}{W^\top W H + \varepsilon}
$$

Factors are initialized from positive random values scaled by the mean of `X`. Reconstruction error is checked every 10 iterations and at the final iteration. `transform(X)` keeps fitted `H` fixed and updates a new `W` for the new data.

## Parameters

- `n_components`: latent dimension; `None` uses `min(n_samples, n_features)`.
- `init`: only `"random"` is supported.
- `solver`: only `"mu"` is supported.
- `beta_loss`: only `"frobenius"` is supported.
- `max_iter`, `tol`, `random_state`.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import NMF

X = np.abs(np.random.default_rng(0).normal(size=(1000, 32)))

nmf = NMF(n_components=8, random_state=0, device="cuda")
W = nmf.fit_transform(X)
X_hat = nmf.inverse_transform(W)
```

## Strict/Approx Difference

NMF has no strict inference mode. The objective is non-convex, and multiplicative updates converge to a local solution that depends on initialization and stopping criteria.

## Outputs

- `components_`
- `reconstruction_err_`
- `n_iter_`
- `n_components_`
- `n_features_in_`

## FAQ

**Can input contain negative values?**
No. NMF raises when `X` contains negative values.

**Is coordinate descent supported?**
No. Phase 2 supports only MU with Frobenius loss.

## External Validation

- Tests: `dev/tests/test_unsupervised_nmf.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase2.py`.
- Baseline: sklearn `NMF(solver="mu", beta_loss="frobenius")`.
- Latest remote matrix: CPU/CuPy/Torch reconstruction differences are at floating-point noise scale; sklearn reconstruction error matches the statgpu CPU scale.

## References

- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401(6755), 788-791. https://doi.org/10.1038/44565
- Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In T. K. Leen, T. G. Dietterich, & V. Tresp (Eds.), *Advances in Neural Information Processing Systems 13* (pp. 556-562). MIT Press.
