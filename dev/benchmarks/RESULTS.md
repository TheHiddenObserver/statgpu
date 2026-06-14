# Benchmark Results

GPU performance data for statgpu. Results are from remote GPU testing unless noted otherwise.

## Test Coverage (v23c — Latest)

**1043/1043 ALL PASS (100%)**

Full matrix: 7 GLM families × 13 penalties × 5 solvers × 3 backends × multiple scales.

## GPU Speedup Highlights

### Async FISTA (v22e) — Key Optimization

Eliminated per-iteration GPU→CPU synchronization in FISTA loop. Activates for all non-smooth penalties (L1, ElasticNet, SCAD, MCP, Adaptive L1, Group) on GPU.

#### n=5000, p=500 (vs CPU baseline)

| Model + Penalty | Before (v21d) | After (v22e) | Improvement |
|----------------|---------------|--------------|-------------|
| logistic + L1 | 2.22x | **5.41x** | 2.45x |
| logistic + ElasticNet | 2.18x | **5.17x** | 2.50x |
| Poisson + L1 | 1.90x | **4.55x** | 2.39x |
| Gamma + L1 | 1.95x | **4.32x** | 2.22x |
| Tweedie + L1 | 1.85x | **4.13x** | 2.23x |
| Inverse Gaussian + L1 | 1.81x | **4.16x** | 2.30x |

#### n=2000, p=200 (smaller scale)

| Model + Penalty | Before | After | Note |
|----------------|--------|-------|------|
| logistic + Adaptive L1 | 0.56x | **1.12x** | GPU now beats CPU |
| Tweedie + ElasticNet | 0.45x | **1.07x** | GPU now beats CPU |
| Gamma + ElasticNet | 0.45x | 0.96x | Near parity |

### What Doesn't Improve

- **SCAD/MCP**: Compute-bound (34k+ iterations), sync removal has no effect
- **Group penalties**: Kernel launch overhead per group operation
- **Smooth penalties** (L2, none): Backtracking path unchanged
- **Squared error + non-smooth**: Modest (0.54x → 0.70x), exact Lipschitz means little backtracking

## Optimization Techniques

1. **Async FISTA**: Fixed step + on-device gradient clipping + deferred convergence checks
2. **Deferred BB convergence**: Check every 5 iterations instead of every iteration
3. **Small problem auto-dispatch**: n×p < 200k → CPU (avoid GPU overhead)
4. **CuPy GPU-native cummin/cummax**: Custom CUDA RawKernel
5. **Kernel fusion + D2H batching** (v20b): Reduced kernel launch overhead
6. **Torch CUDA warmup**: First-use matmul to avoid lazy initialization

## Version History

| Version | Pass Rate | Key Change |
|---------|-----------|------------|
| v23c | 1043/1043 (100%) | L-BFGS fused penalty gradient fix |
| v22e/f | 816/816 | Async FISTA + precision fixes |
| v20b/c | 336/336 | Kernel fusion + D2H batching |
| v17f | 581/581 | Torch SCAD/MCP fix, GPU sync optimizations |
| v15 | 531/533 (99.6%) | 2 remaining FISTA+L2 edge cases |

## Data Files

- `dev/results/` — Knockoff benchmark results (JSON)
- `dev/benchmarks/results.json` — General benchmark results
- `dev/benchmarks/results_glm_gpu.json` — GLM GPU benchmarks
- `dev/benchmarks/results_knockoff.json` — Knockoff benchmarks
- `dev/comparisons/` — R vs Python comparison scripts and data
