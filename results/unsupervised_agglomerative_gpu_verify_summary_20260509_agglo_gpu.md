# AgglomerativeClustering GPU Verification

| Case | Linkage | Backend/framework | ms | ARI vs CPU | ARI vs truth | Status |
|---|---|---:|---:|---:|---:|---|
| n50_p4 | single | statgpu_cpu | 1.002 | 1.000000 | 1.000000 | ok |
| n50_p4 | single | statgpu_cuda | 489.830 | 1.000000 | 1.000000 | ok |
| n50_p4 | single | statgpu_torch | 26.564 | 1.000000 | 1.000000 | ok |
| n50_p4 | single | sklearn_cpu | 0.648 |  | 1.000000 | ok |
| n50_p4 | single | scipy_cpu | 0.084 |  | 1.000000 | ok |
| n50_p4 | single | R_cluster_agnes | 1.000 |  |  | ok |
| n50_p4 | complete | statgpu_cpu | 0.404 | 1.000000 | 1.000000 | ok |
| n50_p4 | complete | statgpu_cuda | 20.167 | 1.000000 | 1.000000 | ok |
| n50_p4 | complete | statgpu_torch | 11.007 | 1.000000 | 1.000000 | ok |
| n50_p4 | complete | sklearn_cpu | 0.507 |  | 1.000000 | ok |
| n50_p4 | complete | scipy_cpu | 0.123 |  | 1.000000 | ok |
| n50_p4 | complete | R_cluster_agnes | 0.000 |  |  | ok |
| n50_p4 | average | statgpu_cpu | 0.257 | 1.000000 | 1.000000 | ok |
| n50_p4 | average | statgpu_cuda | 22.645 | 1.000000 | 1.000000 | ok |
| n50_p4 | average | statgpu_torch | 12.424 | 1.000000 | 1.000000 | ok |
| n50_p4 | average | sklearn_cpu | 0.439 |  | 1.000000 | ok |
| n50_p4 | average | scipy_cpu | 0.119 |  | 1.000000 | ok |
| n50_p4 | average | R_cluster_agnes | 0.000 |  |  | ok |
| n50_p4 | ward | statgpu_cpu | 0.248 | 1.000000 | 1.000000 | ok |
| n50_p4 | ward | statgpu_cuda | 32.085 | 1.000000 | 1.000000 | ok |
| n50_p4 | ward | statgpu_torch | 19.194 | 1.000000 | 1.000000 | ok |
| n50_p4 | ward | sklearn_cpu | 0.453 |  | 1.000000 | ok |
| n50_p4 | ward | scipy_cpu | 0.114 |  | 1.000000 | ok |
| n50_p4 | ward | R_cluster_agnes | 0.000 |  |  | ok |
| n50_p16 | single | statgpu_cpu | 1.318 | 1.000000 | 1.000000 | ok |
| n50_p16 | single | statgpu_cuda | 29.913 | 1.000000 | 1.000000 | ok |
| n50_p16 | single | statgpu_torch | 11.939 | 1.000000 | 1.000000 | ok |
| n50_p16 | single | sklearn_cpu | 0.718 |  | 1.000000 | ok |
| n50_p16 | single | scipy_cpu | 0.090 |  | 1.000000 | ok |
| n50_p16 | single | R_cluster_agnes | 1.000 |  |  | ok |
| n50_p16 | complete | statgpu_cpu | 0.292 | 1.000000 | 1.000000 | ok |
| n50_p16 | complete | statgpu_cuda | 18.570 | 1.000000 | 1.000000 | ok |
| n50_p16 | complete | statgpu_torch | 14.863 | 1.000000 | 1.000000 | ok |
| n50_p16 | complete | sklearn_cpu | 0.482 |  | 1.000000 | ok |
| n50_p16 | complete | scipy_cpu | 0.121 |  | 1.000000 | ok |
| n50_p16 | complete | R_cluster_agnes | 0.000 |  |  | ok |
| n50_p16 | average | statgpu_cpu | 0.279 | 1.000000 | 1.000000 | ok |
| n50_p16 | average | statgpu_cuda | 22.089 | 1.000000 | 1.000000 | ok |
| n50_p16 | average | statgpu_torch | 12.777 | 1.000000 | 1.000000 | ok |
| n50_p16 | average | sklearn_cpu | 0.454 |  | 1.000000 | ok |
| n50_p16 | average | scipy_cpu | 0.140 |  | 1.000000 | ok |
| n50_p16 | average | R_cluster_agnes | 0.000 |  |  | ok |
| n50_p16 | ward | statgpu_cpu | 0.263 | 1.000000 | 1.000000 | ok |
| n50_p16 | ward | statgpu_cuda | 29.788 | 1.000000 | 1.000000 | ok |
| n50_p16 | ward | statgpu_torch | 18.947 | 1.000000 | 1.000000 | ok |
| n50_p16 | ward | sklearn_cpu | 0.466 |  | 1.000000 | ok |
| n50_p16 | ward | scipy_cpu | 0.128 |  | 1.000000 | ok |
| n50_p16 | ward | R_cluster_agnes | 0.000 |  |  | ok |
| n200_p4 | single | statgpu_cpu | 2.924 | 1.000000 | 1.000000 | ok |
| n200_p4 | single | statgpu_cuda | 88.358 | 1.000000 | 1.000000 | ok |
| n200_p4 | single | statgpu_torch | 52.091 | 1.000000 | 1.000000 | ok |
| n200_p4 | single | sklearn_cpu | 1.122 |  | 1.000000 | ok |
| n200_p4 | single | scipy_cpu | 0.275 |  | 1.000000 | ok |
| n200_p4 | single | R_cluster_agnes | 10.000 |  |  | ok |
| n200_p4 | complete | statgpu_cpu | 1.062 | 1.000000 | 1.000000 | ok |
| n200_p4 | complete | statgpu_cuda | 77.049 | 1.000000 | 1.000000 | ok |
| n200_p4 | complete | statgpu_torch | 50.810 | 1.000000 | 1.000000 | ok |
| n200_p4 | complete | sklearn_cpu | 1.107 |  | 1.000000 | ok |
| n200_p4 | complete | scipy_cpu | 0.598 |  | 1.000000 | ok |
| n200_p4 | complete | R_cluster_agnes | 12.000 |  |  | ok |
| n200_p4 | average | statgpu_cpu | 0.926 | 1.000000 | 1.000000 | ok |
| n200_p4 | average | statgpu_cuda | 90.705 | 1.000000 | 1.000000 | ok |
| n200_p4 | average | statgpu_torch | 56.722 | 1.000000 | 1.000000 | ok |
| n200_p4 | average | sklearn_cpu | 1.150 |  | 1.000000 | ok |
| n200_p4 | average | scipy_cpu | 0.557 |  | 1.000000 | ok |
| n200_p4 | average | R_cluster_agnes | 10.000 |  |  | ok |
| n200_p4 | ward | statgpu_cpu | 0.986 | 1.000000 | 1.000000 | ok |
| n200_p4 | ward | statgpu_cuda | 128.304 | 1.000000 | 1.000000 | ok |
| n200_p4 | ward | statgpu_torch | 88.212 | 1.000000 | 1.000000 | ok |
| n200_p4 | ward | sklearn_cpu | 1.071 |  | 1.000000 | ok |
| n200_p4 | ward | scipy_cpu | 0.614 |  | 1.000000 | ok |
| n200_p4 | ward | R_cluster_agnes | 11.000 |  |  | ok |
| n200_p16 | single | statgpu_cpu | 3.122 | 1.000000 | 1.000000 | ok |
| n200_p16 | single | statgpu_cuda | 92.298 | 1.000000 | 1.000000 | ok |
| n200_p16 | single | statgpu_torch | 53.944 | 1.000000 | 1.000000 | ok |
| n200_p16 | single | sklearn_cpu | 1.176 |  | 1.000000 | ok |
| n200_p16 | single | scipy_cpu | 0.372 |  | 1.000000 | ok |
| n200_p16 | single | R_cluster_agnes | 11.000 |  |  | ok |
| n200_p16 | complete | statgpu_cpu | 1.122 | 1.000000 | 1.000000 | ok |
| n200_p16 | complete | statgpu_cuda | 76.031 | 1.000000 | 1.000000 | ok |
| n200_p16 | complete | statgpu_torch | 50.589 | 1.000000 | 1.000000 | ok |
| n200_p16 | complete | sklearn_cpu | 1.288 |  | 1.000000 | ok |
| n200_p16 | complete | scipy_cpu | 0.673 |  | 1.000000 | ok |
| n200_p16 | complete | R_cluster_agnes | 10.000 |  |  | ok |
| n200_p16 | average | statgpu_cpu | 1.054 | 1.000000 | 1.000000 | ok |
| n200_p16 | average | statgpu_cuda | 89.889 | 1.000000 | 1.000000 | ok |
| n200_p16 | average | statgpu_torch | 56.666 | 1.000000 | 1.000000 | ok |
| n200_p16 | average | sklearn_cpu | 1.157 |  | 1.000000 | ok |
| n200_p16 | average | scipy_cpu | 0.678 |  | 1.000000 | ok |
| n200_p16 | average | R_cluster_agnes | 9.000 |  |  | ok |
| n200_p16 | ward | statgpu_cpu | 1.052 | 1.000000 | 1.000000 | ok |
| n200_p16 | ward | statgpu_cuda | 125.514 | 1.000000 | 1.000000 | ok |
| n200_p16 | ward | statgpu_torch | 84.737 | 1.000000 | 1.000000 | ok |
| n200_p16 | ward | sklearn_cpu | 1.204 |  | 1.000000 | ok |
| n200_p16 | ward | scipy_cpu | 0.700 |  | 1.000000 | ok |
| n200_p16 | ward | R_cluster_agnes | 11.000 |  |  | ok |
| n500_p4 | single | statgpu_cpu | 8.354 | 1.000000 | 0.710445 | ok |
| n500_p4 | single | statgpu_cuda | 218.290 | 1.000000 | 0.710445 | ok |
| n500_p4 | single | statgpu_torch | 142.019 | 1.000000 | 0.710445 | ok |
| n500_p4 | single | sklearn_cpu | 2.130 |  | 0.710445 | ok |
| n500_p4 | single | scipy_cpu | 1.283 |  | 0.710445 | ok |
| n500_p4 | single | R_cluster_agnes | 153.000 |  |  | ok |
| n500_p4 | complete | statgpu_cpu | 4.345 | 1.000000 | 1.000000 | ok |
| n500_p4 | complete | statgpu_cuda | 206.966 | 1.000000 | 1.000000 | ok |
| n500_p4 | complete | statgpu_torch | 138.017 | 1.000000 | 1.000000 | ok |
| n500_p4 | complete | sklearn_cpu | 3.900 |  | 1.000000 | ok |
| n500_p4 | complete | scipy_cpu | 3.150 |  | 1.000000 | ok |
| n500_p4 | complete | R_cluster_agnes | 153.000 |  |  | ok |
| n500_p4 | average | statgpu_cpu | 4.079 | 1.000000 | 1.000000 | ok |
| n500_p4 | average | statgpu_cuda | 247.555 | 1.000000 | 1.000000 | ok |
| n500_p4 | average | statgpu_torch | 154.593 | 1.000000 | 1.000000 | ok |
| n500_p4 | average | sklearn_cpu | 3.950 |  | 1.000000 | ok |
| n500_p4 | average | scipy_cpu | 3.300 |  | 1.000000 | ok |
| n500_p4 | average | R_cluster_agnes | 144.000 |  |  | ok |
| n500_p4 | ward | statgpu_cpu | 4.103 | 1.000000 | 1.000000 | ok |
| n500_p4 | ward | statgpu_cuda | 345.653 | 1.000000 | 1.000000 | ok |
| n500_p4 | ward | statgpu_torch | 232.055 | 1.000000 | 1.000000 | ok |
| n500_p4 | ward | sklearn_cpu | 4.124 |  | 1.000000 | ok |
| n500_p4 | ward | scipy_cpu | 3.325 |  | 1.000000 | ok |
| n500_p4 | ward | R_cluster_agnes | 154.000 |  |  | ok |
| n500_p16 | single | statgpu_cpu | 9.509 | 1.000000 | 1.000000 | ok |
| n500_p16 | single | statgpu_cuda | 217.946 | 1.000000 | 1.000000 | ok |
| n500_p16 | single | statgpu_torch | 140.537 | 1.000000 | 1.000000 | ok |
| n500_p16 | single | sklearn_cpu | 3.281 |  | 1.000000 | ok |
| n500_p16 | single | scipy_cpu | 1.928 |  | 1.000000 | ok |
| n500_p16 | single | R_cluster_agnes | 164.000 |  |  | ok |
| n500_p16 | complete | statgpu_cpu | 5.294 | 1.000000 | 1.000000 | ok |
| n500_p16 | complete | statgpu_cuda | 208.042 | 1.000000 | 1.000000 | ok |
| n500_p16 | complete | statgpu_torch | 138.078 | 1.000000 | 1.000000 | ok |
| n500_p16 | complete | sklearn_cpu | 4.681 |  | 1.000000 | ok |
| n500_p16 | complete | scipy_cpu | 3.879 |  | 1.000000 | ok |
| n500_p16 | complete | R_cluster_agnes | 150.000 |  |  | ok |
| n500_p16 | average | statgpu_cpu | 4.629 | 1.000000 | 1.000000 | ok |
| n500_p16 | average | statgpu_cuda | 245.422 | 1.000000 | 1.000000 | ok |
| n500_p16 | average | statgpu_torch | 153.954 | 1.000000 | 1.000000 | ok |
| n500_p16 | average | sklearn_cpu | 4.633 |  | 1.000000 | ok |
| n500_p16 | average | scipy_cpu | 3.889 |  | 1.000000 | ok |
| n500_p16 | average | R_cluster_agnes | 139.000 |  |  | ok |
| n500_p16 | ward | statgpu_cpu | 4.819 | 1.000000 | 1.000000 | ok |
| n500_p16 | ward | statgpu_cuda | 342.211 | 1.000000 | 1.000000 | ok |
| n500_p16 | ward | statgpu_torch | 231.375 | 1.000000 | 1.000000 | ok |
| n500_p16 | ward | sklearn_cpu | 4.847 |  | 1.000000 | ok |
| n500_p16 | ward | scipy_cpu | 4.197 |  | 1.000000 | ok |
| n500_p16 | ward | R_cluster_agnes | 160.000 |  |  | ok |
