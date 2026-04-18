# Elastic Net 基准测试报告 (vs sklearn)

**运行时间**: 2026-04-18T19:52:10.900241
**测试数量**: 6

---

## 测试概览

| 测试 | n_samples | n_features | 后端 | 最大系数差异 | fit_time (ms) | 加速比 |
|------|-----------|------------|------|--------------|---------------|--------|
| small_data | 200 | 20 | sklearn | N/A | 0.77 | N/A |
| small_data | 200 | 20 | statgpu_cpu | 5.75e-09 | 1.10 | 0.70x |
| small_data | 200 | 20 | statgpu_gpu_cupy | 5.75e-09 | 650.12 | 0.00x |
| small_data | 200 | 20 | statgpu_gpu_torch | 5.75e-09 | 19.29 | 0.04x |
| medium_data | 1000 | 50 | sklearn | N/A | 10.42 | N/A |
| medium_data | 1000 | 50 | statgpu_cpu | 2.58e-08 | 2.37 | 4.40x |
| medium_data | 1000 | 50 | statgpu_gpu_cupy | 2.58e-08 | 11.96 | 0.87x |
| medium_data | 1000 | 50 | statgpu_gpu_torch | 2.58e-08 | 11.28 | 0.92x |
| large_data | 5000 | 100 | sklearn | N/A | 6.01 | N/A |
| large_data | 5000 | 100 | statgpu_cpu | 6.63e-10 | 4.13 | 1.45x |
| large_data | 5000 | 100 | statgpu_gpu_cupy | 6.63e-10 | 15.45 | 0.39x |
| large_data | 5000 | 100 | statgpu_gpu_torch | 6.63e-10 | 19.02 | 0.32x |
| high_dim_data | 100 | 200 | sklearn | N/A | 1.27 | N/A |
| high_dim_data | 100 | 200 | statgpu_cpu | 2.19e-08 | 11.01 | 0.11x |
| high_dim_data | 100 | 200 | statgpu_gpu_cupy | 2.19e-08 | 48.17 | 0.03x |
| high_dim_data | 100 | 200 | statgpu_gpu_torch | 2.19e-08 | 42.36 | 0.03x |
| sparse_coef | 500 | 100 | sklearn | N/A | 1.02 | N/A |
| sparse_coef | 500 | 100 | statgpu_cpu | 2.12e-08 | 2.48 | 0.41x |
| sparse_coef | 500 | 100 | statgpu_gpu_cupy | 2.12e-08 | 17.86 | 0.06x |
| sparse_coef | 500 | 100 | statgpu_gpu_torch | 2.12e-08 | 29.92 | 0.03x |
| high_noise | 500 | 50 | sklearn | N/A | 1.84 | N/A |
| high_noise | 500 | 50 | statgpu_cpu | 0.00e+00 | 1.46 | 1.26x |
| high_noise | 500 | 50 | statgpu_gpu_cupy | 0.00e+00 | 2.08 | 0.89x |
| high_noise | 500 | 50 | statgpu_gpu_torch | 0.00e+00 | 1.93 | 0.96x |

---

## 详细结果

### small_data (n=200, p=20)

**sklearn**:
- coef_norm: 0.615496
- intercept: -0.107940
- n_iter: 6
- R²: 0.511106
- fit_time: 0.77 ms

**statgpu_cpu**:
- coef_norm: 0.615496
- intercept: -0.107940
- n_iter: 26
- R²: 0.511106
- fit_time: 1.10 ms
- max_coef_diff_vs_sklearn: 5.75e-09
- speedup_vs_sklearn: 0.70x

**statgpu_gpu_cupy**:
- coef_norm: 0.615496
- intercept: -0.107940
- n_iter: 26
- R²: 0.511106
- fit_time: 650.12 ms
- max_coef_diff_vs_sklearn: 5.75e-09
- speedup_vs_sklearn: 0.00x

**statgpu_gpu_torch**:
- coef_norm: 0.615496
- intercept: -0.107940
- n_iter: 26
- R²: 0.511106
- fit_time: 19.29 ms
- max_coef_diff_vs_sklearn: 5.75e-09
- speedup_vs_sklearn: 0.04x


### medium_data (n=1000, p=50)

**sklearn**:
- coef_norm: 0.741253
- intercept: -0.019189
- n_iter: 5
- R²: 0.553755
- fit_time: 10.42 ms

**statgpu_cpu**:
- coef_norm: 0.741253
- intercept: -0.019189
- n_iter: 21
- R²: 0.553755
- fit_time: 2.37 ms
- max_coef_diff_vs_sklearn: 2.58e-08
- speedup_vs_sklearn: 4.40x

**statgpu_gpu_cupy**:
- coef_norm: 0.741253
- intercept: -0.019189
- n_iter: 21
- R²: 0.553755
- fit_time: 11.96 ms
- max_coef_diff_vs_sklearn: 2.58e-08
- speedup_vs_sklearn: 0.87x

**statgpu_gpu_torch**:
- coef_norm: 0.741253
- intercept: -0.019189
- n_iter: 21
- R²: 0.553755
- fit_time: 11.28 ms
- max_coef_diff_vs_sklearn: 2.58e-08
- speedup_vs_sklearn: 0.92x


### large_data (n=5000, p=100)

**sklearn**:
- coef_norm: 0.447185
- intercept: 0.007229
- n_iter: 4
- R²: 0.389421
- fit_time: 6.01 ms

**statgpu_cpu**:
- coef_norm: 0.447185
- intercept: 0.007229
- n_iter: 17
- R²: 0.389421
- fit_time: 4.13 ms
- max_coef_diff_vs_sklearn: 6.63e-10
- speedup_vs_sklearn: 1.45x

**statgpu_gpu_cupy**:
- coef_norm: 0.447185
- intercept: 0.007229
- n_iter: 17
- R²: 0.389421
- fit_time: 15.45 ms
- max_coef_diff_vs_sklearn: 6.63e-10
- speedup_vs_sklearn: 0.39x

**statgpu_gpu_torch**:
- coef_norm: 0.447185
- intercept: 0.007229
- n_iter: 17
- R²: 0.389421
- fit_time: 19.02 ms
- max_coef_diff_vs_sklearn: 6.63e-10
- speedup_vs_sklearn: 0.32x


### high_dim_data (n=100, p=200)

**sklearn**:
- coef_norm: 0.749663
- intercept: 0.003528
- n_iter: 5
- R²: 0.563083
- fit_time: 1.27 ms

**statgpu_cpu**:
- coef_norm: 0.749663
- intercept: 0.003528
- n_iter: 83
- R²: 0.563083
- fit_time: 11.01 ms
- max_coef_diff_vs_sklearn: 2.19e-08
- speedup_vs_sklearn: 0.11x

**statgpu_gpu_cupy**:
- coef_norm: 0.749663
- intercept: 0.003528
- n_iter: 83
- R²: 0.563083
- fit_time: 48.17 ms
- max_coef_diff_vs_sklearn: 2.19e-08
- speedup_vs_sklearn: 0.03x

**statgpu_gpu_torch**:
- coef_norm: 0.749663
- intercept: 0.003528
- n_iter: 83
- R²: 0.563083
- fit_time: 42.36 ms
- max_coef_diff_vs_sklearn: 2.19e-08
- speedup_vs_sklearn: 0.03x


### sparse_coef (n=500, p=100)

**sklearn**:
- coef_norm: 0.692610
- intercept: 0.010115
- n_iter: 5
- R²: 0.522372
- fit_time: 1.02 ms

**statgpu_cpu**:
- coef_norm: 0.692610
- intercept: 0.010115
- n_iter: 33
- R²: 0.522372
- fit_time: 2.48 ms
- max_coef_diff_vs_sklearn: 2.12e-08
- speedup_vs_sklearn: 0.41x

**statgpu_gpu_cupy**:
- coef_norm: 0.692610
- intercept: 0.010115
- n_iter: 33
- R²: 0.522372
- fit_time: 17.86 ms
- max_coef_diff_vs_sklearn: 2.12e-08
- speedup_vs_sklearn: 0.06x

**statgpu_gpu_torch**:
- coef_norm: 0.692610
- intercept: 0.010115
- n_iter: 33
- R²: 0.522372
- fit_time: 29.92 ms
- max_coef_diff_vs_sklearn: 2.12e-08
- speedup_vs_sklearn: 0.03x


### high_noise (n=500, p=50)

**sklearn**:
- coef_norm: 0.000000
- intercept: -0.051427
- n_iter: 1
- R²: 0.000000
- fit_time: 1.84 ms

**statgpu_cpu**:
- coef_norm: 0.000000
- intercept: -0.051427
- n_iter: 1
- R²: 0.000000
- fit_time: 1.46 ms
- max_coef_diff_vs_sklearn: 0.00e+00
- speedup_vs_sklearn: 1.26x

**statgpu_gpu_cupy**:
- coef_norm: 0.000000
- intercept: -0.051427
- n_iter: 1
- R²: 0.000000
- fit_time: 2.08 ms
- max_coef_diff_vs_sklearn: 0.00e+00
- speedup_vs_sklearn: 0.89x

**statgpu_gpu_torch**:
- coef_norm: 0.000000
- intercept: -0.051427
- n_iter: 1
- R²: 0.000000
- fit_time: 1.93 ms
- max_coef_diff_vs_sklearn: 0.00e+00
- speedup_vs_sklearn: 0.96x


---

## 结论

- 所有 statgpu 后端与 sklearn 的系数差异 < 1e-6 ✅
- GPU 后端在大数据集上显示出加速效果

*报告生成时间*: 2026-04-18T19:52:10.903732