# Unsupervised Phase 3C Validation Summary

| Model | Backend/framework | Mean ms | Quality metric | Status |
|---|---:|---:|---:|---|
| IncrementalPCA | statgpu_cpu | 288.841 | 120.372007 | ok |
| IncrementalPCA | statgpu_cuda | 126.955 | 120.372007 | ok |
| IncrementalPCA | statgpu_torch | 104.380 | 120.372007 | ok |
| IncrementalPCA | sklearn_cpu | 441.797 | 120.372007 | ok |
| MiniBatchNMF | statgpu_cpu | 290.677 | 672.845866 | ok |
| MiniBatchNMF | statgpu_cuda | 92.596 | 672.845866 | ok |
| MiniBatchNMF | statgpu_torch | 31.103 | 672.845866 | ok |
| MiniBatchNMF | sklearn_cpu | 354.712 | 689.613335 | ok |
