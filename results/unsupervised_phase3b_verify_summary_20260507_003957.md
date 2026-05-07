# Unsupervised Phase 3B Validation Summary

## Runtime Table

| Model | Variant | Backend/framework | Mean ms | Status |
|---|---|---:|---:|---|
| GaussianMixture | diag | statgpu_cpu | 2.439 | ok |
| GaussianMixture | diag | statgpu_cuda | 10.925 | ok |
| GaussianMixture | diag | statgpu_torch | 5.498 | ok |
| GaussianMixture | diag | sklearn_cpu | 8.482 | ok |
| GaussianMixture | spherical | statgpu_cpu | 2.832 | ok |
| GaussianMixture | spherical | statgpu_cuda | 9.806 | ok |
| GaussianMixture | spherical | statgpu_torch | 5.344 | ok |
| GaussianMixture | spherical | sklearn_cpu | 9.001 | ok |
| GaussianMixture | tied | statgpu_cpu | 4.731 | ok |
| GaussianMixture | tied | statgpu_cuda | 20.830 | ok |
| GaussianMixture | tied | statgpu_torch | 8.901 | ok |
| GaussianMixture | tied | sklearn_cpu | 33.204 | ok |
| GaussianMixture | full | statgpu_cpu | 4.722 | ok |
| GaussianMixture | full | statgpu_cuda | 29.981 | ok |
| GaussianMixture | full | statgpu_torch | 12.162 | ok |
| GaussianMixture | full | sklearn_cpu | 24.215 | ok |
| AgglomerativeClustering | single | statgpu_cpu | 4.685 | ok |
| AgglomerativeClustering | single | sklearn_cpu | 4.177 | ok |
| AgglomerativeClustering | single | scipy_cpu | 3.505 | ok |
| AgglomerativeClustering | single | R_cluster_agnes | 659.000 | ok |
| AgglomerativeClustering | complete | statgpu_cpu | 9.927 | ok |
| AgglomerativeClustering | complete | sklearn_cpu | 9.611 | ok |
| AgglomerativeClustering | complete | scipy_cpu | 8.575 | ok |
| AgglomerativeClustering | complete | R_cluster_agnes | 660.000 | ok |
| AgglomerativeClustering | average | statgpu_cpu | 9.892 | ok |
| AgglomerativeClustering | average | sklearn_cpu | 9.918 | ok |
| AgglomerativeClustering | average | scipy_cpu | 8.820 | ok |
| AgglomerativeClustering | average | R_cluster_agnes | 633.000 | ok |
| AgglomerativeClustering | ward | statgpu_cpu | 10.145 | ok |
| AgglomerativeClustering | ward | sklearn_cpu | 9.904 | ok |
| AgglomerativeClustering | ward | scipy_cpu | 8.992 | ok |
| AgglomerativeClustering | ward | R_cluster_agnes | 678.000 | ok |
