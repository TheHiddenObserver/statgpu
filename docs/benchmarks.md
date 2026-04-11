# 基准脚本索引

> 语言: 中文  
> 最后更新: 2026-04-10  
> 页面定位: 基准脚本索引  
> 切换: [English](en/benchmarks.md)

语言切换：[English](en/benchmarks.md)

## 推断相关

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
  - 对比 `cpu_ols_inference` vs `gpu_ols_inference`
  - 输出时间与 `coef/bse/t/p/conf_int` 差异

## 多重检验与全局 p 值合并

- `dev/benchmarks/benchmark_inference_backends.py`
  - 含 `combine_pvalues` 的 `fisher/cauchy/acat` 耗时基准
  - 含一致性检查：
    - Fisher vs `scipy.stats.combine_pvalues`
    - Cauchy vs 独立 NumPy 参考实现
    - statgpu NumPy vs CuPy
  - 统一输出结构化 JSON 到 `results/`

远端补充产物：
- `results/remote_fisher_cauchy_benchmark_2026-04-05.json`
- `results/remote_fisher_cauchy_benchmark_2026-04-05.md`

## 显存管理

- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
  - 对比 `gpu_memory_cleanup=False/True`
  - 输出 `fit_ms` 与 CuPy memory pool 指标

## 训练性能 / 停止准则

- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`

## 全方法大数据量耗时对比

- `dev/benchmarks/benchmark_all_methods_large_scale.py`
  - 覆盖：`LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
  - 支持 CPU/GPU 双设备、warmup/repeats、可选 inference 计时
  - 关键点：数据构造与 host->device 迁移在计时外，默认只统计 `fit()`

推荐运行命令：

```bash
python dev/benchmarks/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --include-external \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_all_large_results.json
```

若要把推断统计的计算时间也纳入计时，追加：

```bash
--compute-inference
```

## 外部框架对标（数值 + 时间）

- `dev/benchmarks/benchmark_external_frameworks.py`
  - 优先对标：`statsmodels`、`sklearn`
  - 可选对标：`R`（若系统有 `Rscript` 与对应包）
  - 输出：`fit_ms` + 系数/推断统计差异（可输出 JSON）

推荐运行命令（statsmodels + sklearn）：

```bash
python dev/benchmarks/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow \
  --skip-r
```

推荐运行命令（含 R）：

```bash
python dev/benchmarks/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow
```

对标门禁建议（务必统一口径）：
- 显式固定同一特征集合（避免 `y ~ .` 误包含目标/辅助列）
- 显式固定 `ties`（如 `cox-ties=breslow` 或 `efron`）
- 显式记录正则参数与迭代阈值（`alpha/C/max_iter/tol`）

## 协方差三方统一对比（statsmodels / statgpu CPU / statgpu GPU）

- 运行脚本：`tmp_remote_covariance_full_compare.py`
- 结果产物：`results/remote_covariance_full_compare_2026-04-10.json`
- 对齐设置：
  - `cov_type`: `hc2/hc3/hac`
  - `linear`: `n=8000, p=24`
  - `logistic`: `n=12000, p=16`
  - `timing_repeats=2`（含 warmup）

最新重跑快照（2026-04-10，同设定）：
- Linear-HAC: `statsmodels=9.9158ms`, `statgpu CPU=10.3402ms`, `statgpu GPU=3.8064ms`
- Logistic-HAC: `statsmodels=14.6619ms`, `statgpu CPU=10.2583ms`, `statgpu GPU=7.4366ms`
- Linear-HAC 精度：`statgpu CPU vs statsmodels` 的 `max_abs_bse_diff=1.3817e-09`

## Cox 协方差专项基准

- `dev/benchmarks/benchmark_cox_cluster.py`
  - 对比 `CoxPH cov_type=nonrobust/hc1/cluster` 的时间与数值差异
  - 覆盖 `statgpu CPU/GPU` 与 `statsmodels.PHReg`（可用时）

## Knockoff 特征选择

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
  - 在多个 `q` 水平下运行 fixed-X knockoff，并输出选择结果诊断指标。

- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
  - 对比 fixed-X/model-X knockoff 与基线选择器：
    - marginal-correlation top-k
    - statgpu lasso top-k
    - sklearn `LassoCV`（可用时）
    - `knockpy` Gaussian knockoff + lasso 统计量（可用时）
  - 支持通过 `config.knockoff_method` 配置 knockoff 统计量（当前默认：`ols_coef_diff`）。
  - model-X 路径采用协方差收缩 + 多次 knockoff 的 W 聚合（次数随统计量变化）。
  - 运行结果包含 model-X 校准元信息（`modelx_n_draws`、`modelx_covariance_shrinkage`）。
  - 在 `knockpy` 可用时，结果同时包含其可用性标记与 pairwise 差异字段。
  - 统一输出 precision/recall/FDP/F1/Jaccard 与耗时 JSON。
  - 额外支持环境变量：
    - `STATGPU_KNOCKOFF_COMPAT_MODE`：`statgpu` 或 `knockpy`
    - `STATGPU_KNOCKOFF_LASSO_CV_IMPL`：`auto` / `statgpu` / `sklearn`

- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`
  - 用同一个 `Xk`（由 knockpy 生成）对比 `statgpu` 与 `knockpy`。
  - 重点输出：`W` 相关系数、`W` 误差、阈值差、选择集合 Jaccard。
  - 适用于“先固定 knockoff 变量，再比较算法实现差异”的正确性诊断场景。
