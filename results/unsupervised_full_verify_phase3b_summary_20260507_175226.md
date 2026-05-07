# Unsupervised Phase 3B Validation Summary

## Runtime Table

| Model | Variant | Backend/framework | Mean ms | Status |
|---|---|---:|---:|---|
| GaussianMixture | diag | statgpu_cpu | 2.614 | ok |
| GaussianMixture | diag | statgpu_cuda | 10.120 | ok |
| GaussianMixture | diag | statgpu_torch | 7.003 | ok |
| GaussianMixture | diag | sklearn_cpu | 56.257 | ok |
| GaussianMixture | spherical | statgpu_cpu | 3.944 | ok |
| GaussianMixture | spherical | statgpu_cuda | 10.994 | ok |
| GaussianMixture | spherical | statgpu_torch | 8.374 | ok |
| GaussianMixture | spherical | sklearn_cpu | 42.895 | ok |
| GaussianMixture | tied | statgpu_cpu | 5.787 | ok |
| GaussianMixture | tied | statgpu_cuda | 21.436 | ok |
| GaussianMixture | tied | statgpu_torch | 11.975 | ok |
| GaussianMixture | tied | sklearn_cpu | 147.975 | ok |
| GaussianMixture | full | statgpu_cpu | 6.880 | ok |
| GaussianMixture | full | statgpu_cuda | 30.991 | ok |
| GaussianMixture | full | statgpu_torch | 16.849 | ok |
| GaussianMixture | full | sklearn_cpu | 325.978 | ok |
| AgglomerativeClustering | single | statgpu_cpu | 15.175 | ok |
| AgglomerativeClustering | single | sklearn_cpu | 6.729 | ok |
| AgglomerativeClustering | single | scipy_cpu | 5.603 | ok |
| AgglomerativeClustering | single | R_cluster_agnes | 1270.000 | ok |
| AgglomerativeClustering | complete | statgpu_cpu | 16.608 | ok |
| AgglomerativeClustering | complete | sklearn_cpu | 15.293 | ok |
| AgglomerativeClustering | complete | scipy_cpu | 13.822 | ok |
| AgglomerativeClustering | complete | R_cluster_agnes | 1329.000 | ok |
| AgglomerativeClustering | average | statgpu_cpu | 15.517 | ok |
| AgglomerativeClustering | average | sklearn_cpu | 15.057 | ok |
| AgglomerativeClustering | average | scipy_cpu | 13.951 | ok |
| AgglomerativeClustering | average | R_cluster_agnes | 1218.000 | ok |
| AgglomerativeClustering | ward | statgpu_cpu | 15.902 | ok |
| AgglomerativeClustering | ward | sklearn_cpu | 15.869 | ok |
| AgglomerativeClustering | ward | scipy_cpu | 14.626 | ok |
| AgglomerativeClustering | ward | R_cluster_agnes | 1317.000 | ok |
