# Unsupervised Phase 3 Benchmark Summary

| Model | Backend/framework | Warm mean ms | Status |
|---|---:|---:|---|
| TruncatedSVD | statgpu_cpu | 1.0976046323776245 | ok |
| TruncatedSVD | statgpu_cuda | 4.99330461025238 | ok |
| TruncatedSVD | statgpu_torch | 2.7081966400146484 | ok |
| TruncatedSVD | sklearn_cpu | 1.1649131774902344 | ok |
| TruncatedSVD | statsmodels_cpu | 0.8820444345474243 | ok |
| TruncatedSVD | cuml_gpu |  | skipped |
| TruncatedSVD | R_cpu | 1.0 | ok |
| MiniBatchKMeans | statgpu_cpu | 5.472585558891296 | ok |
| MiniBatchKMeans | statgpu_cuda | 38.70433568954468 | ok |
| MiniBatchKMeans | statgpu_torch | 21.159619092941284 | ok |
| MiniBatchKMeans | sklearn_cpu | 4.517614841461182 | ok |
| MiniBatchKMeans | cuml_gpu |  | skipped |
| UMAP | statgpu_cpu | 78.65273952484131 | ok |
| UMAP | statgpu_cuda | 19.49070394039154 | ok |
| UMAP | statgpu_torch | 9.571626782417297 | ok |
| UMAP | umap_learn_cpu | 30.124202370643616 | ok |
| UMAP | cuml_gpu |  | skipped |
| TSNE | statgpu_cpu | 511.921226978302 | ok |
| TSNE | statgpu_cuda | 200.3563791513443 | ok |
| TSNE | statgpu_torch | 102.92665660381317 | ok |
| TSNE | sklearn_cpu | 442.5657391548157 | ok |
| TSNE | openTSNE_cpu | 782.2159379720688 | ok |
| TSNE | cuml_gpu |  | skipped |
