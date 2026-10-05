# Historical multiple-testing timing records

These records were moved from the user guide at source snapshot
`c6bd229b62aff7c82cec92df9f2b57667e810ece`. They are historical transcription,
not a benchmark of the current PR head. The original text does not identify a
complete source/environment/timing contract for every table. No timing here
establishes a general backend ranking or crossover threshold. Rerun a maintained,
correctness-checked benchmark with exact source, hardware, synchronization and
transfer scope before making a current performance claim.

## Large-Scale Performance (p=50k-1M, Tesla P100)

Benchmark script: `dev/benchmarks/_bench_inference_timing_large.py`

### adjust_pvalues BH (sort + cummin, O(n log n))

| p | NumPy | CuPy | Torch | CuPy vs CPU |
|---|------:|-----:|------:|-----------:|
| 50,000 | 33.3 ms | 1.75 ms | 114 ms | 19.0x |
| 100,000 | 69.7 ms | 3.49 ms | 232 ms | 20.0x |
| 500,000 | 374 ms | 28.5 ms | 1.01 s | 13.1x |
| 1,000,000 | 799 ms | 76.7 ms | 1.96 s | **10.4x** |

CuPy excels at sort-heavy operations (adjust family).

### combine_pvalues Stouffer (norm.ppf + sum, O(n) compute-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 50,000 | 2.01 ms | 1.47 ms | 0.51 ms | 3.9x |
| 100,000 | 3.88 ms | 2.93 ms | 0.65 ms | 6.0x |
| 500,000 | 17.8 ms | 17.6 ms | 1.67 ms | 10.7x |
| 1,000,000 | 36.8 ms | 34.0 ms | 3.09 ms | **11.9x** |

Torch excels at compute-bound operations (norm.ppf).

### combine_pvalues Fisher (sum+log, O(n) bandwidth-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 1,000,000 | 5.43 ms | 6.93 ms | 2.04 ms | **2.7x** |

Bandwidth-bound operations (sum+log) see modest GPU speedup (~2-3x).

### combine_pvalues Cauchy (tan + sum, O(n) compute-bound)

| p | NumPy | CuPy | Torch | Torch vs CPU |
|---|------:|-----:|------:|-----------:|
| 1,000,000 | 49.5 ms | 42.6 ms | 11.2 ms | **4.4x** |

### GPU Speedup Summary

- **p < 10,000**: GPU kernel launch overhead (>300 us) dominates; CPU may be faster
- **p > 50,000**: GPU advantage becomes clear
- **Sort-heavy** (adjust BH): CuPy best at 10x+
- **Compute-bound** (Stouffer norm.ppf): Torch best at 12x
- **Bandwidth-bound** (Fisher sum+log): modest GPU speedup at 2-3x
- **Mixed** (Cauchy tan+sum): moderate GPU speedup at 4-5x

## Benchmark Interpretation Notes (old, p=4000x64)

Remote supplement artifact:
- JSON: `results/remote_fisher_cauchy_benchmark_2026-04-05.json`
- Summary: `results/remote_fisher_cauchy_benchmark_2026-04-05.md`

Workload used in that artifact:
- `n_groups=4000`, `group_size=64`, `axis=1`, `warmup=1`, `repeats=5`

Key runtime means:

| Method | statgpu NumPy (ms) | statgpu CuPy (ms) | SciPy (ms) |
|---|---:|---:|---:|
| Fisher | 3.979 | 0.816 | 350.721 |
| Cauchy | 4.052 | 0.874 | N/A |
| ACAT alias | 3.971 | N/A | N/A |

How to read these numbers:
- Fisher: SciPy is much slower than statgpu NumPy on this workload (`~88.15x`).
- NumPy to CuPy speedup in statgpu is about `~4.88x` (Fisher) and `~4.63x` (Cauchy).
- Cauchy and ACAT alias are numerically identical in this benchmark (`max abs p-value diff = 0.0`).
- NumPy/CuPy output differences are at floating-point noise level (about `1e-15` in p-values).

## Reproducibility

Primary local benchmark script:
- `dev/benchmarks/benchmark_inference_backends.py`

Example run:

```bash
python dev/benchmarks/benchmark_inference_backends.py --output-tag local_check
```

Generated output:
- `results/inference_backend_benchmark_<date>_local_check.json`