# Unsupervised Phase 3 Benchmark Summary

| Model | Backend/framework | Warm mean ms | Status |
|---|---:|---:|---|
| TruncatedSVD | statgpu_cpu | 3.5829395055770874 | ok |
| TruncatedSVD | statgpu_cuda | 5.388826131820679 | ok |
| TruncatedSVD | statgpu_torch | 4.216477274894714 | ok |
| TruncatedSVD | sklearn_cpu | 373.7916201353073 | ok |
| TruncatedSVD | statsmodels_cpu | 7.738709449768066 | ok |
| TruncatedSVD | cuml_gpu |  | skipped |
| TruncatedSVD | R_cpu | 3.0 | ok |
| MiniBatchKMeans | statgpu_cpu | 17.690196633338928 | ok |
| MiniBatchKMeans | statgpu_cuda | 160.320445895195 | ok |
| MiniBatchKMeans | statgpu_torch | 73.32096993923187 | ok |
| MiniBatchKMeans | sklearn_cpu | 22.46403694152832 | ok |
| MiniBatchKMeans | cuml_gpu |  | skipped |
| UMAP | statgpu_cpu | 3401.776894927025 | ok |
| UMAP | statgpu_cuda | 347.84671664237976 | ok |
| UMAP | statgpu_torch | 50.711557269096375 | ok |
| UMAP | umap_learn_cpu | 522.662490606308 | ok |
| UMAP | cuml_gpu |  | skipped |
| TSNE | statgpu_cpu | 4078.0807733535767 | ok |
| TSNE | statgpu_cuda | 498.71091544628143 | ok |
| TSNE | statgpu_torch | 333.25745165348053 | ok |
| TSNE | sklearn_cpu | 7078.951880335808 | ok |
| TSNE | openTSNE_cpu | 11013.647854328156 | ok |
| TSNE | cuml_gpu |  | skipped |
