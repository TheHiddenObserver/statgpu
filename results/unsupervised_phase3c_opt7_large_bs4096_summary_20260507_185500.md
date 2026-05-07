# Unsupervised Phase 3C Validation Summary

| Model | Backend/framework | Mean ms | Quality metric | Status |
|---|---:|---:|---:|---|
| IncrementalPCA | statgpu_cpu | 40.861 | 52.871036 | ok |
| IncrementalPCA | statgpu_cuda | 35.251 | 52.871036 | ok |
| IncrementalPCA | statgpu_torch | 30.771 | 52.871036 | ok |
| IncrementalPCA | sklearn_cpu | 4277.993 | 52.871036 | ok |
| MiniBatchNMF | statgpu_cpu | 61.386 | 230.656447 | ok |
| MiniBatchNMF | statgpu_cuda | 85.808 | 230.656447 | ok |
| MiniBatchNMF | statgpu_torch | 23.868 | 230.656447 | ok |
| MiniBatchNMF | sklearn_cpu | 92.806 | 233.221730 | ok |
