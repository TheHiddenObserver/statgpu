# Remote Fisher/Cauchy Benchmark (2026-04-05)

## Environment

- Host: hz-4.matpool.com:27613
- Remote workspace: /tmp/statgpu_inference_full_1775387750
- Python: /root/miniconda3/envs/myconda/bin/python
- CuPy available: true
- SciPy available: true

## Workload

- n_groups: 4000
- group_size: 64
- axis: 1
- warmup: 1
- repeats: 5

## Runtime (mean over 5 runs)

| Method | statgpu NumPy (ms) | statgpu CuPy (ms) | SciPy (ms) |
|---|---:|---:|---:|
| Fisher | 3.979 | 0.816 | 350.721 |
| Cauchy | 4.052 | 0.874 | N/A |
| ACAT alias | 3.971 | N/A | N/A |

## Derived Ratios

- Fisher SciPy vs statgpu NumPy: 88.152x slower
- Fisher statgpu NumPy vs CuPy: 4.879x
- Cauchy statgpu NumPy vs CuPy: 4.634x

## Precision Consistency

- Fisher vs SciPy:
  - max abs stat diff: 0.0
  - max abs p-value diff: 0.0
- Cauchy vs independent NumPy reference:
  - max abs stat diff: 0.0
  - max abs p-value diff: 0.0
- Cauchy vs ACAT alias:
  - max abs stat diff: 0.0
  - max abs p-value diff: 0.0
- CuPy vs NumPy:
  - Fisher max abs p-value diff: 1.3322676295501878e-15
  - Cauchy max abs p-value diff: 9.43689570931383e-16

## Notes

- This benchmark is a targeted remote supplement for p-value combination methods only.
- Result JSON source: results/remote_fisher_cauchy_benchmark_2026-04-05.json
