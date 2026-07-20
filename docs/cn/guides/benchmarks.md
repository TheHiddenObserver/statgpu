# 基准脚本索引

> 语言：中文  
> 最后更新：2026-07-20  
> 页面定位：基准脚本索引  
> 切换：[English](../../en/guides/benchmarks.md)

## 交互式基准面板

- **打开面板**：[Benchmark Dashboard](../../assets/benchmarks/index.html)
- **使用说明**：[筛选、图表、指标与复现](../../en/guides/statgpu_benchmark_dashboard.md)

当前 canonical dashboard 只注册日期不早于 **2026-06-01** 的 benchmark source，共包含：

```text
8 个已注册 source
1,774 条 normalized runs
36 个 models
```

已覆盖的主要模块包括：惩罚 GLM、GLM、近期线性模型、稳健/分位数回归、生存分析、无监督学习、有序模型、非参数方法、面板模型、协方差估计和 ANOVA。Feature Selection 分类已经预留，但在出现 2026-06 或之后的结构化 benchmark 前保持为空。

2026 年 4 月的 ElasticNet、LassoCV、comprehensive validation、Cox package comparison 和 knockoff 结果不会接入当前 dashboard。已有的 6 月 distribution Markdown 汇总也不会直接转换成 measured rows；需要保留原始重复计时和精度元数据的结构化 JSON 或重新运行。

当前功能包括：

- Environment 与多分类导航；
- Metric scope：Fit、CV、Inference、Prediction、Selection；
- Model → Variant → Penalty → Solver → Scale 的渐进式筛选；
- NumPy、CuPy、Torch 后端筛选；
- 根据当前上下文显示 scikit-learn、SciPy、statsmodels、linearmodels、pyGAM 等 external reference；
- Focused 与 Full matrix 两种图表模式；
- Timing 与 Speedup 图，并区分 computed 和 runner-reported speedup；
- 带 Scope 列的可排序、可分页明细表；
- Validation、Accuracy、Inference、Prediction、Convergence、Selection 指标面板；
- Source provenance、parse report 与 source inventory。

生成并验证 canonical bundle：

```bash
python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
```

构建并测试前端：

```bash
cd frontend
npm ci
npm run typecheck
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

---

## 推断相关

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
  - 对比 `cpu_ols_inference` 与 `gpu_ols_inference`
  - 输出时间以及 `coef/bse/t/p/conf_int` 差异

当前 dashboard 中已接入的 inference 还包括 Ordered Logit/Probit、Quantile kernel/bootstrap、penalized-logistic HC0/oracle 和 penalized-linear bootstrap。CV 前端 contract 已经实现，但在新的合格 CV source 接入前显示为 `CV (0)`。

## 非参数方法

- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
  - 对比 `statgpu` 与 `statsmodels.nonparametric.kernel_regression.KernelReg`
  - 支持 `regression=nw/local_linear` 和多维设置
  - 可通过 `--kernel-metric diagonal` 使用可比口径
  - 输出 CPU/GPU、statsmodels 的精度和运行时间 JSON

- `dev/benchmarks/benchmark_kde_vs_scipy.py`
  - 对比 `statgpu` 与 `scipy.stats.gaussian_kde`
  - 输出 CPU/GPU 与 SciPy 的精度和时间结果

- `dev/benchmarks/benchmark_nonparametric_vs_r.py`
  - 对比 `statgpu` 与 R `density()`、`ksmooth()`、`KernSmooth::locpoly()`
  - 支持 `--statgpu-backend numpy/cupy`
  - 支持 `--ci-method normal/bootstrap`

## Multiple Testing 与全局 p-value 组合

- `dev/benchmarks/benchmark_inference_backends.py`
  - 覆盖 `combine_pvalues` 的 `fisher/cauchy/acat`
  - 对齐 SciPy、独立 NumPy reference 和 statgpu NumPy/CuPy
  - 输出结构化 JSON

历史远程补充结果：

- `results/remote_fisher_cauchy_benchmark_2026-04-05.json`
- `results/remote_fisher_cauchy_benchmark_2026-04-05.md`

这些历史文件不自动成为当前 canonical source。
