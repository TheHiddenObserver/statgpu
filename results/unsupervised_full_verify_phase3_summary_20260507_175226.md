# Unsupervised Phase 3 Benchmark Summary

| Model | Backend/framework | Warm mean ms | Status |
|---|---:|---:|---|
| TruncatedSVD | statgpu_cpu | 2.664536237716675 | ok |
| TruncatedSVD | statgpu_cuda | 6.208881735801697 | ok |
| TruncatedSVD | statgpu_torch | 5.032986402511597 | ok |
| TruncatedSVD | sklearn_cpu | 243.2074248790741 | ok |
| TruncatedSVD | statsmodels_cpu | 15.105396509170532 | ok |
| TruncatedSVD | cuml_gpu |  | skipped |
| TruncatedSVD | R_cpu | 3.0 | ok |
| MiniBatchKMeans | statgpu_cpu | 11.926516890525818 | ok |
| MiniBatchKMeans | statgpu_cuda | 94.24568712711334 | ok |
| MiniBatchKMeans | statgpu_torch | 67.032590508461 | ok |
| MiniBatchKMeans | sklearn_cpu | 1167.3319637775421 | ok |
| MiniBatchKMeans | cuml_gpu |  | skipped |
| UMAP | statgpu_cpu | 1405.9125930070877 | ok |
| UMAP | statgpu_cuda | 152.1603763103485 | ok |
| UMAP | statgpu_torch | 22.503063082695007 | ok |
| UMAP | umap_learn_cpu | 414.0029549598694 | ok |
| UMAP | cuml_gpu |  | skipped |
| TSNE | statgpu_cpu | 2365.5757904052734 | ok |
| TSNE | statgpu_cuda | 269.74573731422424 | ok |
| TSNE | statgpu_torch | 287.4186784029007 | ok |
| TSNE | sklearn_cpu | 3741.9478446245193 | ok |
| TSNE | openTSNE_cpu | 8362.318590283394 | ok |
| TSNE | cuml_gpu |  | skipped |
