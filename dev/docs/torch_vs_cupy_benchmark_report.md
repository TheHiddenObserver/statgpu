# StatGPU Torch vs CuPy Performance Benchmark Report

**Test Date**: 2026-04-17  
**Remote Server**: hz-4.matpool.com:27609  
**Conda Environment**: `myconda`  
**GPU Device**: Tesla P100-SXM2-16GB  

---

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.0.0+cu117 |
| CuPy | 13.6.0 |
| CUDA | 11.7 |
| NumPy | 1.24.2 |
| Driver | 460.32.03 |

---

## Benchmark Results

### Dataset: 10,000 samples × 500 features

| Method | Time (s) | Speedup vs NumPy | R² | Max Diff vs NumPy |
|--------|----------|------------------|-----|-------------------|
| NumPy CPU | 0.1718 | 1.00x | 0.9995 | - |
| CuPy GPU | 0.6881 | 0.25x | 0.9995 | 1.03e-14 |
| Torch CPU | 0.1794 | 0.96x | 0.9995 | 0.00e+00 |
| **Torch GPU** | **0.0606** | **2.83x** | 0.9995 | 1.03e-14 |

### Key Findings

1. **Torch GPU is fastest**: 2.83x speedup vs NumPy, 11x faster than CuPy
2. **CuPy GPU is slow**: Overhead from data transfer outweighs compute benefits for this workload
3. **Torch CPU ≈ NumPy**: Comparable performance on CPU path
4. **Numerical accuracy**: All methods achieve identical R² (0.9995) and coefficient precision (< 1e-14 diff)

---

## Performance Analysis

### Why Torch GPU Outperforms CuPy

1. **Efficient memory management**: Torch's CUDA memory allocator may be better optimized for this workload
2. **Linear algebra kernels**: Torch's cuSOLVER integration appears more efficient for OLS solve
3. **Less overhead**: Torch backend has simpler array conversion logic

### Why CuPy GPU is Slower

1. **Data transfer overhead**: Converting NumPy arrays to CuPy adds overhead
2. **Kernel launch overhead**: CuPy's dynamic kernel compilation may add latency
3. **Memory pool**: CuPy's memory pool management may be less efficient for moderate-sized arrays

---

## Scaling Behavior

### Small Dataset (2,000 × 50)

| Method | Time (s) | Speedup |
|--------|----------|---------|
| NumPy CPU | 0.0052 | 1.00x |
| CuPy GPU | 0.6577 | 0.01x |
| Torch CPU | 0.0054 | 0.96x |
| Torch GPU | 0.0087 | 0.59x |

For small datasets, CPU is faster due to GPU transfer overhead.

### Large Dataset (10,000 × 500)

| Method | Time (s) | Speedup |
|--------|----------|---------|
| NumPy CPU | 0.1718 | 1.00x |
| CuPy GPU | 0.6881 | 0.25x |
| Torch CPU | 0.1794 | 0.96x |
| **Torch GPU** | **0.0606** | **2.83x** |

For larger datasets, Torch GPU shows clear advantage.

---

## Recommendations

### When to Use Torch GPU

- Dataset size > 5,000 samples
- High-dimensional features (> 100)
- Repeated fitting (model selection, cross-validation)
- When PyTorch ecosystem integration is needed

### When to Use NumPy CPU

- Small datasets (< 2,000 samples)
- Single-fit scenarios
- When GPU memory is constrained

### When to Use CuPy

- Legacy code compatibility
- When CuPy-specific features are needed
- Not recommended for new Torch-compatible code

---

## Next Steps

### Phase 5: Documentation Updates

1. Update README.md with Torch backend installation instructions
2. Update USAGE.md with Torch usage examples
3. Add performance comparison to documentation
4. Document environment requirements (PyTorch 2.0+ recommended)

### Future Work

1. Test with even larger datasets (100K+ samples)
2. Benchmark other models (Ridge, Lasso, LogisticRegression)
3. Profile memory usage patterns
4. Investigate Torch compile (PyTorch 2.0+ feature) for additional speedup

---

**Benchmark Script**: `dev/scripts/torch_vs_cupy_benchmark.py`  
**Report Generated**: 2026-04-17
