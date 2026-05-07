# Unsupervised Phase 3C Validation Summary

| Model | Backend/framework | Mean ms | Quality metric | Status |
|---|---:|---:|---:|---|
| IncrementalPCA | statgpu_cpu | 40.584 | 52.871036 | ok |
| IncrementalPCA | statgpu_cuda | 35.283 | 52.871036 | ok |
| IncrementalPCA | statgpu_torch | 31.627 | 52.871036 | ok |
| IncrementalPCA | sklearn_cpu | 125.002 | 52.871036 | ok |
| MiniBatchNMF | statgpu_cpu | 61.313 | 230.656447 | ok |
| MiniBatchNMF | statgpu_cuda | 86.060 | 230.656447 | ok |
| MiniBatchNMF | statgpu_torch | 25.153 | 230.656447 | ok |
| MiniBatchNMF | sklearn_cpu | 86.045 | 233.221730 | ok |
