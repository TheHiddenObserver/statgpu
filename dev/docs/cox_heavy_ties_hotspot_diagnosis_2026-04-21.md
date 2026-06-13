# Heavy-Ties Efron GPU Hotspot Diagnosis (2026-04-21)

## Scope

This diagnosis focuses on why `ties="efron"` remains slower on GPU than CPU under heavy ties, especially at larger `p` and tie groups.

Code paths reviewed:

- `statgpu/survival/_cox_efron_cuda.py`
- `statgpu/survival/_cox.py`

Benchmark evidence source:

- `dev/docs/cox_heavy_ties_compare.json` (regenerated via `dev/scripts/run_cox_heavy_ties_compare.py`)

## Confirmed Hotspots

### 1) `p^2` atomic-update hotspot in fused backward scan kernel

In `efron_backward_scan` (`statgpu/survival/_cox_efron_cuda.py`), both risk-enter/risk-exit updates and failure-group updates perform repeated full-matrix `p x p` accumulation:

- `xp2[j*p+k] += ...`
- `xp2f[j*p+k] += ...`

For non-sequential branches (`nt > seq_thresh`, `m > seq_thresh`), these become `atomicAdd` to global or shared buffers in nested `j,k` loops. Under heavy ties and moderate-to-large `p`, this is effectively `O(m * p^2)` atomic traffic with strong contention.

### 2) Serial reduction region after parallel accumulation

After parallel accumulation, thread 0 executes Efron correction and Hessian combination loops, including:

- full `p*p` updates to `hess_acc`
- symmetric `j1,j2` nested loops for correction terms

This introduces a serial tail per unique failure time that grows with `p^2`.

### 3) Iteration-level dual-kernel structure in GPU fit loop

In `_fit_gpu` (`statgpu/survival/_cox.py`), each Newton iteration computes:

1. gradient/Hessian via `_compute_gradient_hessian_gpu(...)` (Efron fused backward path), then
2. log-likelihood via `_compute_log_likelihood_gpu(...)` (Efron loglik kernel path)

So Efron statistics are traversed twice per iteration. CSR caching reduces Python/packing overhead, but not this repeated per-iteration kernel work.

## Heavy-Ties Warm-Median Evidence (Efron)

From regenerated `dev/docs/cox_heavy_ties_compare.json`:

- `(n=2000, p=20, avg_tie=21.00)`
  - `statgpu_cpu`: `0.020818s`
  - `statgpu_cuda`: `0.072641s` (`3.49x` slower than CPU)
  - `statgpu_torch`: `0.066809s` (`3.21x` slower than CPU)
- `(n=5000, p=20, avg_tie=41.96)`
  - `statgpu_cpu`: `0.040348s`
  - `statgpu_cuda`: `0.110316s` (`2.73x` slower than CPU)
  - `statgpu_torch`: `0.107408s` (`2.66x` slower than CPU)
- `(n=10000, p=50, avg_tie=54.25)`
  - `statgpu_cpu`: `0.229228s`
  - `statgpu_cuda`: `0.972797s` (`4.24x` slower than CPU)
  - `statgpu_torch`: `0.974558s` (`4.25x` slower than CPU)

These results are consistent with a heavy `p^2` update bottleneck and atomic contention rather than cold-start-only effects (cold and warm are close for larger cases, but warm still trails CPU).

## Accuracy Check (Efron, vs statsmodels)

Numerical parity remains intact despite runtime gap:

- `statgpu_cuda`
  - coef max abs diff: `3.33e-16 ~ 1.11e-12`
  - loglik abs diff: `0 ~ 3.64e-12`
- `statgpu_torch`
  - coef max abs diff: `2.22e-16 ~ 1.11e-12`
  - loglik abs diff: `0 ~ 3.64e-12`

## Conclusion

The bottleneck diagnosis is confirmed:

- dominant `p^2` accumulation with atomics in heavy ties,
- serial `p^2` post-processing segments per tie group,
- and duplicated Efron statistics work across grad/Hessian and loglik kernels each iteration.

The next optimization should prioritize tiled/blocked Hessian accumulation to reduce global atomic contention before deeper kernel-fusion work.
