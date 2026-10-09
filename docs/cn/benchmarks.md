# 基准测试：理解结果与复现实验

> 语言：中文<br>
> 最后更新：2026-10-09<br>
> 切换：[English](../en/benchmarks.md)

## 先确认工作负载可比

通过[交互式面板](../assets/benchmarks/index.html)查找已记录的实验；筛选和指标的详细说明见[面板使用指南](../en/guides/statgpu_benchmark_dashboard.md)。基准结果描述的是特定工作负载和运行环境，不能当作模型或后端的通用排名。

比较时间前，请对齐：

- 统计任务、输入行列、预处理、样本权重和目标函数归一化。跨软件包时还要对齐惩罚尺度，参数名称相同不代表优化目标相同。
- 方法变体、求解器、收敛容差、数值精度及误差。Cox 模型还需统一并列事件处理方法；交叉验证还需统一划分、候选网格、评分及是否计入最终重拟合。
- 计时范围：仅拟合、拟合加推断、完整交叉验证、预测或验证。仅包含正确性验证的记录不代表测量过时间或加速比。
- 硬件与软件环境、预热、重复次数、GPU 同步和数据搬运口径。不同环境的结果应分开理解。

加速比是“参考实现用时 / 当前实现用时”：大于一表示更快，小于一表示更慢。先核对参考对象和精度结果，再解读比值；某一工作负载误差很小，不代表所有设置都具有同样精度。

## 测量自己的实际流程

同时选择有代表性的样本数和特征数，并使用实际需要的数据精度、稀疏度、求解器设置、交叉验证网格及推断选项。小问题可能主要消耗在启动或数据搬运上；即使硬件相同，不同算法的性能分界点也可能不同。

先预热再重复测量，同时报告代表性用时和离散程度。读取 GPU 用时前，应同步实际执行计算的具体设备。说明数组转换和主机/设备搬运是否计入时间；仅拟合计时与端到端计时回答的是不同问题。记录软件版本、CPU/GPU 型号、源码版本、数据形状、随机种子和数值差异，便于之后解释结果。显式选择后端的方法见[设备与显存指南](guides/device-and-memory.md)。

## 进阶附录：基准脚本

以下索引面向需要从源码复现或设计测量的读者，可能需要额外的 Python/R 包或 GPU。使用前请检查各脚本的选项和计时范围；列入索引不表示每份历史脚本都符合当前测量要求。面板数据接入、构建、测试和发布流程见[贡献者维护指南](../../frontend/docs/benchmark-dashboard-maintenance.md)。

旧远程实验的阶段产物清单保留在[历史产物索引](../../dev/guides/historical-benchmark-artifacts.md)。

### 推断相关

- [dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py](../../dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py)
  - 使用规范 `inference_method="post_selection_ols"`，对 NumPy CPU 与 CuPy CUDA 的完整拟合与推断事务做基准和数值对照；
  - 输出惩罚系数、活跃集重拟合参数、标准误/统计量/p 值/置信区间、活跃集一致性，以及推断后端与具体设备来源的 CPU/CuPy 对照结果；
  - 这不是仅推断阶段的加速基准：硬件由 `device` 选择，而不是由 `inference_method` 选择。

### 非参数方法

- [dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py](../../dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py)
  - 对比 `statgpu` 与 `statsmodels.nonparametric.kernel_regression.KernelReg`
  - 支持 `regression=nw/local_linear` 与多维设置
  - 支持通过 `--kernel-metric diagonal` 进行公平口径对齐
  - 结果包含 `statgpu CPU/GPU` 与 `statsmodels` 的精度和时间对比
  - 输出时间与精度 JSON 到 `results/`

- [dev/benchmarks/benchmark_kde_vs_scipy.py](../../dev/benchmarks/benchmark_kde_vs_scipy.py)
  - 对比 `statgpu` 与 `scipy.stats.gaussian_kde`
  - 结果包含 `statgpu CPU/GPU` 与 SciPy 的精度和时间对比

- [dev/benchmarks/benchmark_nonparametric_vs_r.py](../../dev/benchmarks/benchmark_nonparametric_vs_r.py)
  - 对比 `statgpu` 与 R 的 `density()` / `ksmooth()` / `KernSmooth::locpoly()`
  - 支持 `--statgpu-backend numpy/cupy`
  - 支持 `--ci-method normal/bootstrap`
  - 结果包含 `statgpu CPU/GPU`、R，以及 KDE CI 与 SciPy 的对照

### 无监督学习

- 详细模型文档：[docs/unsupervised/](unsupervised/README.md)
- [dev/benchmarks/benchmark_unsupervised.py](../../dev/benchmarks/benchmark_unsupervised.py)
  - 对比 `PCA` 与 `KMeans` 在 `statgpu` CPU/CuPy/Torch 和 sklearn（可用时）下的时间与数值差异。
  - 支持输出 JSON 结果。

第二阶段脚本：
- [dev/benchmarks/benchmark_unsupervised_phase2.py](../../dev/benchmarks/benchmark_unsupervised_phase2.py)
  - 对比 `DBSCAN`、`GaussianMixture`、`NMF` 和 `AgglomerativeClustering`，对齐 sklearn/SciPy/R 等可用基线。
  - 记录可选的 `umap-learn` 与 `openTSNE` 冒烟检查及运行时间，供比较参考。
- [dev/benchmarks/benchmark_unsupervised_dbscan_cython.py](../../dev/benchmarks/benchmark_unsupervised_dbscan_cython.py)
  - 验证可选的 statgpu 自有 DBSCAN Cython CPU 快速路径，并对照纯 Python 回退、sklearn CPU、CuPy 与 Torch。

DBSCAN CPU Cython 说明：
- 可选的 `_dbscan_cpu` 扩展由 statgpu 自有实现，并不是 sklearn 的封装。
- 紧凑的稠密 CPU 场景可以在扩展已编译并被选用时使用它；变密度、稀疏/全噪声或无编译器的环境会退回纯 Python 实现。

第三阶段脚本：
- [dev/benchmarks/benchmark_unsupervised_phase3.py](../../dev/benchmarks/benchmark_unsupervised_phase3.py)
  - 对比 `TruncatedSVD`、`MiniBatchKMeans`、`UMAP` 和 `TSNE` 的 statgpu CPU/CuPy/Torch 路径，以及可用的 sklearn、statsmodels、R、umap-learn、openTSNE、cuML 对齐基线。
  - 记录预热与重复运行的用时、精度或嵌入质量指标，以及外部框架不可用时的跳过原因。
  - GPU 主计时路径使用已在目标后端上的输入数组，避免把 NumPy 到 GPU 的搬运混入主结论。

第三阶段 B 组脚本：
- [dev/benchmarks/benchmark_unsupervised_phase3b.py](../../dev/benchmarks/benchmark_unsupervised_phase3b.py)
  - 对比 `GaussianMixture` 的 `diag/spherical/tied/full` 协方差形式。
  - 对比 `AgglomerativeClustering` 的 `single/complete/average/ward` 连接方式。
  - 覆盖 statgpu CPU/CuPy/Torch（适用时）、sklearn、SciPy，以及可用时的 R `cluster::agnes`。

第三阶段 C 组脚本：
- [dev/benchmarks/benchmark_unsupervised_phase3c.py](../../dev/benchmarks/benchmark_unsupervised_phase3c.py)
  - 对比 `IncrementalPCA` 和 `MiniBatchNMF` 的 statgpu CPU/CuPy/Torch 路径，以及可用时的 sklearn 参考实现。
  - 记录重构质量、解释方差、预热与重复运行用时，以及被跳过的可选外部框架。

### 多重检验与全局 p 值合并

- [dev/benchmarks/benchmark_inference_backends.py](../../dev/benchmarks/benchmark_inference_backends.py)
  - 含 `combine_pvalues` 的 `fisher/cauchy/acat` 耗时基准
  - 含一致性检查：
    - Fisher 与 `scipy.stats.combine_pvalues` 的对照
    - Cauchy 与独立 NumPy 参考实现的对照
    - statgpu 的 NumPy 与 CuPy 对照
  - 统一输出结构化 JSON 到 `results/`

### 显存管理

- [dev/benchmarks/benchmark_gpu_memory_cleanup.py](../../dev/benchmarks/benchmark_gpu_memory_cleanup.py)
  - 对比 `gpu_memory_cleanup=False/True`
  - 输出 `fit_ms` 与 CuPy 内存池指标

### 训练性能 / 停止准则

- [dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py](../../dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py)
- [dev/comparisons/compare_lasso_kkt_stopping.py](../../dev/comparisons/compare_lasso_kkt_stopping.py)

### 全方法大数据量耗时对比

- [dev/benchmarks/benchmark_all_methods_large_scale.py](../../dev/benchmarks/benchmark_all_methods_large_scale.py)
  - 覆盖：`LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
  - 支持 CPU/GPU 双设备、预热与重复运行，以及可选的推断计时
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

### 外部框架对标（数值 + 时间）

- [dev/benchmarks/benchmark_external_frameworks.py](../../dev/benchmarks/benchmark_external_frameworks.py)
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

比较时请统一以下口径：
- 显式固定同一特征集合（避免 `y ~ .` 误包含目标/辅助列）
- 显式固定 `ties`（如 `cox-ties=breslow` 或 `efron`）
- 显式记录正则参数与迭代阈值（`alpha/C/max_iter/tol`）

### Cox 协方差专项基准

- [dev/benchmarks/benchmark_cox_cluster.py](../../dev/benchmarks/benchmark_cox_cluster.py)
  - 对比 `CoxPH cov_type=nonrobust/hc1/cluster` 的时间与数值差异
  - 覆盖 `statgpu CPU/GPU` 与 `statsmodels.PHReg`（可用时）

### Elastic Net 基准测试

以下是 2026 年 4 月 18 日的历史实验，不能用作当前版本的预期加速比，也不能直接据此选择后端。

- [小规模实验脚本](../../dev/benchmarks/benchmark_elasticnet_sklearn.py)与[结果 JSON](../../results/benchmark_elasticnet_sklearn_2026-04-18.json)：六组合成数据改变样本数、特征数、稀疏程度和噪声。JSON 记录了系数向量及各行相对 sklearn 的差异；这些误差只适用于相应配置，不能推广到其他目标函数、容差或数据。
- [大规模实验脚本](../../dev/benchmarks/benchmark_large_scale.py)与[结果 JSON](../../results/large_scale/benchmark_elasticnet_large_scale_2026-04-18.json)：六组稠密高斯设计数据，`n=10,000–100,000`、`p=100 或 500`，生成模型有十个非零系数，随机种子为 42，噪声标准差为 0.5。对应脚本生成 NumPy float64 输入，sklearn 与 statgpu 均设置 `alpha=1.0`、`l1_ratio=0.5`、`max_iter=5000`、`tol=1e-8`。这些是脚本设置，不是完整的实际运行环境记录。
- [R glmnet 结果](../../results/benchmark_full/benchmark_glmnet_all.json)需结合[对应的 statgpu 结果](../../results/benchmark_full/benchmark_statgpu_all.json)理解。目标函数归一化、惩罚尺度、标准化或停止准则未对齐时，系数及计算量都可能变化；系数范数的差异不能证明两个优化问题等价。

例如，大规模 JSON 中标为 `statgpu_gpu_torch` 的一行，在 `n=100,000, p=500` 时记录了相对 sklearn 的 `615.59 ms / 141.05 ms ≈ 4.36×`；同一标签在 `n=10,000, p=100` 时的比值则小于一。**该标签不能证明实际执行的是 Torch：**对应脚本用 `device='cuda'` 构造这一行的估计器，计时段内也没有显式 GPU 同步。结果文件没有记录 CPU/GPU 型号、软件版本、源码提交或重复计时的离散程度，只保存系数范数，没有可逐元素比较的完整系数向量。因此，不能把这一比值归因于已确认的 Torch/硬件组合，也不能作为当前版本的性能保证。

请按上文的方法，在自己的模型、特征数、数值精度、收敛设置和数据搬运口径下测量。上述结果无法给出仅由样本数决定的 GPU 性能分界点。旧实验记录见[历史产物索引](../../dev/guides/historical-benchmark-artifacts.md)。

---

### Knockoff 特征选择

- [dev/benchmarks/benchmark_knockoff_fixedx.py](../../dev/benchmarks/benchmark_knockoff_fixedx.py)
  - 在多个 `q` 水平下运行 fixed-X knockoff，并输出选择结果诊断指标。

- [dev/benchmarks/benchmark_knockoff_vs_baselines.py](../../dev/benchmarks/benchmark_knockoff_vs_baselines.py)
  - 对比 fixed-X/model-X knockoff 与基线选择器：
    - marginal-correlation top-k
    - statgpu lasso top-k
    - sklearn `LassoCV`（可用时）
    - `knockpy` Gaussian knockoff + lasso 统计量（可用时）
  - 支持通过 `config.knockoff_method` 配置 knockoff 统计量（当前默认：`ols_coef_diff`）。
  - model-X 路径采用协方差收缩 + 多次 knockoff 的 W 聚合（次数随统计量变化）。
  - 运行结果包含 model-X 校准元信息（`modelx_n_draws`、`modelx_covariance_shrinkage`）。
  - 在 `knockpy` 可用时，结果同时包含其可用性标记与两两比较差异字段。
  - 统一输出 precision/recall/FDP/F1/Jaccard 与耗时 JSON。
  - 额外支持环境变量：
    - `STATGPU_KNOCKOFF_COMPAT_MODE`：`statgpu` 或 `knockpy`
    - `STATGPU_KNOCKOFF_LASSO_CV_IMPL`：`auto` / `statgpu` / `sklearn`

- [dev/benchmarks/benchmark_knockoff_same_xk_parity.py](../../dev/benchmarks/benchmark_knockoff_same_xk_parity.py)
  - 用同一个 `Xk`（由 knockpy 生成）对比 `statgpu` 与 `knockpy`。
  - 重点输出：`W` 相关系数、`W` 误差、阈值差、选择集合 Jaccard。
  - 适用于“先固定 knockoff 变量，再比较算法实现差异”的正确性诊断场景。
