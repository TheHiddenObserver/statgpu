# Unsupervised Phase 3C Validation Summary

| Model | Backend/framework | Mean ms | Quality metric | Status |
|---|---:|---:|---:|---|
| IncrementalPCA | statgpu_cpu | 9.693 | 18.016715 | ok |
| IncrementalPCA | statgpu_cuda | 33.810 | 18.016715 | ok |
| IncrementalPCA | statgpu_torch | 35.290 | 18.016715 | ok |
| IncrementalPCA | sklearn_cpu | 13.051 | 18.016715 | ok |
| MiniBatchNMF | statgpu_cpu | 23.087 | 61.511913 | ok |
| MiniBatchNMF | statgpu_cuda | 259.404 | 61.511913 | ok |
| MiniBatchNMF | statgpu_torch | 66.839 | 61.511913 | ok |
| MiniBatchNMF | sklearn_cpu | 30.935 | 69.182804 | ok |
