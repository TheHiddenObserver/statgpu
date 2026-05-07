# Unsupervised Phase 3C Validation Summary

| Model | Backend/framework | Mean ms | Quality metric | Status |
|---|---:|---:|---:|---|
| IncrementalPCA | statgpu_cpu | 9.561 | 18.016715 | ok |
| IncrementalPCA | statgpu_cuda | 32.131 | 18.016715 | ok |
| IncrementalPCA | statgpu_torch | 35.106 | 18.016715 | ok |
| IncrementalPCA | sklearn_cpu | 12.608 | 18.016715 | ok |
| MiniBatchNMF | statgpu_cpu | 23.892 | 61.511913 | ok |
| MiniBatchNMF | statgpu_cuda | 234.374 | 61.511913 | ok |
| MiniBatchNMF | statgpu_torch | 65.853 | 61.511913 | ok |
| MiniBatchNMF | sklearn_cpu | 28.967 | 69.182804 | ok |
