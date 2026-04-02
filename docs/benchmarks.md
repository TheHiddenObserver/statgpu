# 基准脚本索引

## 推断相关

- `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
  - 对比 `cpu_ols_inference` vs `gpu_ols_inference`
  - 输出时间与 `coef/bse/t/p/conf_int` 差异

## 显存管理

- `examples/benchmark_gpu_memory_cleanup.py`
  - 对比 `gpu_memory_cleanup=False/True`
  - 输出 `fit_ms` 与 CuPy memory pool 指标

## 训练性能 / 停止准则

- `examples/benchmark_lasso_cpu_gpu_tol.py`
- `examples/compare_lasso_kkt_stopping.py`
