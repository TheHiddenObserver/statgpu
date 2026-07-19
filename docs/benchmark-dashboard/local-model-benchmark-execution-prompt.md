# Local-Model Prompt for Completing Benchmark Data

本文档是一份可直接交给本地 coding model / agent 的执行提示词，用于补充 statgpu Benchmark Dashboard 尚缺失的 benchmark 数据。

目标不是让模型“根据已有结果推测数据”，而是让它：

1. 扫描当前 repo 与实现接口；
2. 编写可重复运行的 benchmark runner；
3. 在本地 CPU/GPU 环境中实际执行；
4. 保存原始 repeats、环境和失败信息；
5. 产出结构化 JSON；
6. 在数据稳定后，选择性完成 parser、manifest、前端和测试接入。

---

## 1. 最简单的调用方式

在 repo 根目录启动本地模型，并给它下面的短提示词：

```text
你正在维护 TheHiddenObserver/statgpu。

请先阅读并严格遵循：

- docs/benchmark-dashboard/local-model-benchmark-execution-prompt.md
- docs/benchmark-dashboard/remaining-module-audit.md
- docs/benchmark-dashboard/cv-inference-frontend-contract.md

本次只执行模块：{{MODULE_ID}}
执行模式：{{MODE}}

其中：
- MODULE_ID 从本文档第 9 节选择；
- MODE=compute-only 时，只实现 runner、实际运行并产出结构化 JSON，不修改 parser/manifest/frontend；
- MODE=full-integration 时，在 benchmark 结果通过验收后继续完成 parser、manifest、测试和前端资产更新。

必须先检查当前 git branch、工作树、Python/CUDA/CuPy/Torch 环境和现有实现，不得假设接口。
不得伪造、插值或手工补写任何 timing、accuracy、inference、selection 或 speedup 数据。
若某组合失败、OOM、缺少依赖或不支持，必须显式记录，不得静默删除。
开始工作前先给出一个不超过 20 行的执行计划；随后直接执行，不要只写 proposal。
```

推荐的 `MODULE_ID`：

```text
robust_losses
penalized_robust_quantile
cv_models
linear_inference
ordered_crossover
anova_crossover
covariance
nonparametric
feature_selection
penalized_survival
panel_extended
distributions
multiple_testing
```

首次运行建议使用：

```text
MODE=compute-only
```

数据检查通过后，再用同一个模型继续：

```text
MODE=full-integration
```

---

## 2. 本地模型的角色与硬约束

将下面整段作为 system-style task prompt 使用也可以：

```text
你是 statgpu 仓库的 benchmark implementation and integration agent。
你的工作结果必须可重复、可审计、可在前端解释，而不仅仅是“脚本能够跑通”。

硬约束：

1. 只能把实际运行得到的数据标为 measured。
2. 禁止根据 stdout、旧 Markdown、相邻规模或其他 backend 推算缺失数值。
3. 禁止为了让图表更好看而删除慢、失败、OOM 或 speedup < 1 的组合。
4. 禁止把不同 objective、tuning constant、scale estimator、initialization、ties method、covariance method、CV protocol 或 timing scope 合并成同一个 method identity。
5. 禁止在未验证数学定义一致时声称 external framework parity。
6. 禁止在没有显式 GPU synchronization 的情况下记录 GPU elapsed time。
7. 禁止将 host-to-device transfer 混入 fit-only timing；若需要，另记 transfer-inclusive timing。
8. 禁止全局升级或替换 CUDA、PyTorch、CuPy、NumPy 等核心依赖。缺少可选 reference package 时记录 unsupported/missing_dependency，并继续可执行部分。
9. 所有 JSON 数值必须有限；NaN/Infinity 必须转成 null，并附带 failure_reason 或 validation failure。
10. 所有随机数据必须由保存的 seed 完整复现。
11. 不修改已有 benchmark 测量值。
12. 不合并 PR，不 force-push，不删除用户已有工作。

在任何写入前：
- 读取当前实现和测试；
- 用最小 smoke case 验证 API；
- 明确 benchmark case identity、method identity 和 timing scope。

最终必须报告：
- 实际运行命令；
- 环境信息；
- 完成/失败/OOM 的矩阵；
- 输出文件和 SHA256；
- 验证结果；
- 未完成原因；
- 是否修改 parser/frontend。
```

---

## 3. 两阶段工作流

### 3.1 `compute-only`

此模式只完成：

```text
repo/API audit
→ benchmark design
→ smoke test
→ production runner
→ actual execution
→ raw structured JSON
→ result validation
```

允许修改：

```text
dev/benchmarks/run_<module>_benchmark.py
相关 benchmark helper/test
results/<module>_bench_YYYY-MM-DD.json
results/benchmark_frontend_sources/<module>_YYYYMMDD.json
相关执行说明文档
```

默认不要修改：

```text
dev/benchmarks/frontend_sources.json
dev/benchmarks/frontend_data/parsers/**
frontend/src/**
docs/assets/benchmarks/**
```

只有在用户明确选择 `full-integration` 后才继续接入。

### 3.2 `full-integration`

必须先重新读取并验证 compute-only artifact，再完成：

```text
parser
→ parser registry
→ source manifest + SHA256
→ exact matrix tests
→ generated canonical bundle
→ frontend scope/filter checks
→ deterministic assets
→ full CI
```

如果 artifact 不符合第 8 节验收标准，应停止接入并修复 runner，而不是在 parser 中猜测缺失字段。

---

## 4. Benchmark runner 通用协议

### 4.1 数据类型和设备

除非模型本身只支持其他精度，默认使用：

```text
dtype = float64
backend ∈ {numpy, cupy, torch}
```

必须保存：

```text
requested backend
resolved backend/device
CPU model
GPU model and memory
CUDA runtime/driver if available
Python version
NumPy/CuPy/Torch/statgpu version
external reference versions
git commit and dirty status
hostname
UTC timestamp
```

### 4.2 时间测量

使用单调高精度计时器，例如：

```python
start_ns = time.perf_counter_ns()
...
elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
```

GPU 同步至少应等价于：

```python
def synchronize(backend: str) -> None:
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

每次 GPU timing：

```text
synchronize
→ start timer
→ measured operation
→ synchronize
→ stop timer
```

不得仅同步 Torch 而忽略 CuPy。

### 4.3 Warmup、repeats 和 seeds

推荐默认协议：

```text
warmups = 2
measured repeats per seed = 5
data seeds = 3
```

对特别昂贵的 benchmark，可以降低 repeats，但必须：

- 在 artifact 中记录实际值；
- 对所有可比 backend 使用相同协议；
- 不把不同 repeat protocol 混成一个 method identity；
- 至少保留原始 repeat times，而不是只有 mean。

### 4.4 Timing scopes

至少区分：

```text
fit_only
transfer_plus_fit
predict_only
transform_only
cv_total
cv_path_only
cv_refit_only
inference_only
fit_plus_inference
```

用于 `fit_only` 的 backend array 必须在计时前准备完成。

### 4.5 失败和资源限制

每个预定组合都必须产生一个 observation，状态为：

```text
ok
unsupported
missing_dependency
failed
oom
timeout
```

记录：

```text
exception type
sanitized exception message
stage
elapsed time before failure
peak memory if available
fallback attempted or not
```

不得通过 `continue` 让失败组合从最终 JSON 消失。

---

## 5. 推荐的原始 JSON contract

新的 runner 优先采用统一的 observation-oriented artifact。Parser 可以将其转换为前端 Schema v1.1。

```json
{
  "artifact_version": "1.0",
  "benchmark_id": "cv_models_20260719",
  "source_date": "2026-07-19",
  "created_at_utc": "2026-07-19T00:00:00Z",
  "git": {
    "commit": "<40-char sha>",
    "dirty": false
  },
  "environment": {
    "env_id": "remote-p100",
    "hostname": "host",
    "python": "3.x",
    "cpu": "Intel ...",
    "gpu": "NVIDIA Tesla P100-SXM2-16GB",
    "gpu_memory_bytes": 17179869184,
    "cuda_driver": "...",
    "cuda_runtime": "...",
    "packages": {
      "statgpu": "...",
      "numpy": "...",
      "cupy": "...",
      "torch": "...",
      "sklearn": "..."
    }
  },
  "protocol": {
    "dtype": "float64",
    "warmups": 2,
    "repeats": 5,
    "seeds": [11, 23, 47],
    "synchronization": {
      "cupy": "cp.cuda.Stream.null.synchronize",
      "torch": "torch.cuda.synchronize"
    },
    "timer": "time.perf_counter_ns"
  },
  "observations": [
    {
      "observation_id": "stable-human-readable-or-hashed-id",
      "case": {
        "case_id": "stable-case-id",
        "category_ids": ["linear_models"],
        "model_id": "LassoCV",
        "n_samples": 20000,
        "n_features": 100,
        "data_seed": 11,
        "data_regime": "sparse_gaussian",
        "data_parameters": {
          "n_signal": 10,
          "noise_scale": 1.0,
          "rho": 0.2
        }
      },
      "method": {
        "method_config_id": "stable-method-id",
        "framework": "statgpu",
        "backend": "cupy",
        "loss": "squared_error",
        "penalty": "l1",
        "solver_requested": "auto",
        "solver_resolved": "fista",
        "variant": "path-cv",
        "parameters": {
          "metric_scope": "cross_validation",
          "timing_scope": "cv_total",
          "cv_folds": 5,
          "grid_size": 100,
          "warm_start": true,
          "refit": true
        }
      },
      "status": "ok",
      "failure": null,
      "raw_repeats": {
        "fit_time_ms": [120.1, 119.4, 121.3, 118.9, 120.7],
        "transfer_plus_fit_ms": [124.0, 123.1, 125.2, 122.8, 124.5]
      },
      "summary": {
        "fit_time_ms_mean": 120.08,
        "fit_time_ms_std": 0.84,
        "fit_time_ms_min": 118.9,
        "fit_time_ms_max": 121.3,
        "sample_count": 5,
        "std_ddof": 1
      },
      "metrics": {
        "selected_alpha": 0.01,
        "test_mse": 1.03,
        "coef_l2_rel_error": 0.08,
        "support_precision": 0.9,
        "support_recall": 0.8,
        "n_iter": 42,
        "converged": true,
        "peak_gpu_memory_bytes": 123456789
      },
      "validation": {
        "reference_observation_id": "matching-numpy-or-external-id",
        "checks": [
          {
            "metric": "coef_l2_rel_diff",
            "value": 1.2e-8,
            "tolerance": 1e-6,
            "status": "pass"
          }
        ]
      }
    }
  ]
}
```

规则：

- `case_id` 只包含数据生成、规模和任务条件；
- `method_config_id` 只包含算法、loss、penalty、solver、tuning 和统计方法；
- backend/framework 属于 implementation identity；
- 不把 seed 放进 method identity；
- 不把 contamination regime 放进 method identity；
- raw repeats 必须保留；
- summary 必须能从 raw repeats 重新计算；
- speedup 优先由 parser 基于 matched reference timing 计算；
- runner-reported speedup 必须明确标注语义，不能与 computed speedup 混用。

---

## 6. Compute-only 主提示词

下面这段可以直接复制给本地模型：

```text
任务：为 statgpu Benchmark Dashboard 补充 {{MODULE_ID}} 的真实 benchmark 数据。
模式：compute-only。

仓库：TheHiddenObserver/statgpu
当前工作目录：repo 根目录
目标 branch：先执行 `git branch --show-current` 确认，不要假设；不得切走用户当前未提交工作。

请严格完成：

A. Repo audit
1. 阅读 docs/benchmark-dashboard/local-model-benchmark-execution-prompt.md。
2. 阅读 remaining-module-audit.md 以及该模块对应的专项 plan。
3. 搜索实际实现类、公开 API、solver、backend dispatch、已有 tests 和旧 runner。
4. 输出实际支持矩阵，不要根据类名猜测。
5. 检查现有 Python/CUDA/CuPy/Torch 和 external reference 环境；不要全局升级依赖。

B. Benchmark design
1. 明确定义 case identity、method identity、implementation identity 和 timing scope。
2. 使用第 9 节中 {{MODULE_ID}} 的 required matrix。
3. 默认 float64、2 warmups、每 seed 5 repeats、3 seeds。
4. fit-only timing 前完成 backend array transfer。
5. 对 CuPy 和 Torch 分别显式同步。
6. 保存 raw repeats、summary、environment、versions、git SHA、seeds 和失败信息。
7. 不静默跳过 unsupported/failed/OOM 组合。

C. Implementation
1. 创建 `dev/benchmarks/run_{{MODULE_ID}}_benchmark.py`，使用 argparse。
2. 至少支持：
   --output
   --quick
   --seeds
   --repeats
   --warmups
   --backends
   --timeout-seconds（若适用）
3. `--quick` 只做 smoke validation，不能覆盖 production artifact。
4. production mode 输出符合本文第 5 节的 observation-oriented JSON。
5. 写入必须使用临时文件 + atomic rename，避免中断时留下半个 JSON。

D. Execution
1. 先运行 quick smoke case，检查每个 backend 的 API、有限值和同步。
2. 再运行 production matrix。
3. 原始结果写到：
   `results/{{MODULE_ID}}_bench_YYYY-MM-DD.json`
4. 验证后复制完全相同的字节到：
   `results/benchmark_frontend_sources/{{MODULE_ID}}_YYYYMMDD.json`
5. 不手工编辑 measured values。

E. Validation
1. JSON 可重新加载，且不存在 NaN/Infinity。
2. summary 可由 raw repeats 重算。
3. 每个预定组合都有 observation 或显式 failure record。
4. backend agreement 使用预先声明的 tolerance；失败必须保留。
5. 检查 timing 没有明显异步错误，例如 GPU 时间近似为零或第一次运行异常混入 repeats。
6. 计算并报告 SHA256：
   `sha256sum results/benchmark_frontend_sources/{{MODULE_ID}}_YYYYMMDD.json`

F. Deliverable
最终答复必须包含：
1. repo/API 扫描结论；
2. benchmark 完整矩阵；
3. 实际执行命令；
4. 成功、失败、OOM、unsupported 数量；
5. 每个 backend 的关键 timing/accuracy 摘要，但不要省略慢于 CPU 的结果；
6. 输出路径和 SHA256；
7. 新增/修改文件；
8. 尚未解决的问题；
9. 明确声明未修改 parser/manifest/frontend。

不要只输出代码建议。必须在当前环境中实际运行；若当前环境无 GPU，则只完成 runner 和 CPU smoke test，并明确停止，绝不能伪造 GPU 结果。
```

---

## 7. Full-integration 继续提示词

当 compute-only artifact 已经人工检查后，把下面这段交给同一模型：

```text
任务：将已确认的 {{MODULE_ID}} benchmark artifact 接入 statgpu Benchmark Dashboard。
模式：full-integration。

输入 artifact：
results/benchmark_frontend_sources/{{ARTIFACT_FILE}}
预期 SHA256：{{SHA256}}

先验证文件 SHA256、JSON contract、finite values、完整矩阵和 source_date；任何一项不满足都停止接入并报告，不得在 parser 中猜测。

然后完成：

1. Parser
- 在 `dev/benchmarks/frontend_data/parsers/` 新增或扩展模块 parser。
- 输出 Schema v1.1 runs、model entries 和 warnings。
- 保留 loss、penalty、solver、variant、parameters、timing_scope、replicate 和 provenance。
- 对 matched reference 计算 speedup，并填入 reference_run_id。
- 不为失败 observation 伪造成功 run；根据既有 dashboard policy 记录 parse issue 或显式状态。

2. Registry
- 更新 parsers/__init__.py。
- 更新 frontend_data/registry.py 的 PARSER_FUNCTIONS。

3. Manifest
- 在 `dev/benchmarks/frontend_sources.json` 中添加 framework、comparison 和 source entry。
- source_date 必须 >= 2026-06-01。
- source_id 包含日期和 SHA256 前 12 位。
- sha256 必须与文件完全一致。
- 不修改其他 source 的 hash 或 measured data。

4. Tests
- 在 Python contract tests 中断言完整 source matrix 的精确 row 数。
- 断言 scale、backend、framework、variant、timing_scope 和 parser_version。
- 添加错误/缺字段测试。
- 添加 Playwright 测试，验证 category/model/variant/scope/filter 可发现。
- CV 数据应使 `Metric scope → CV` 自动从 0 变为非零并可点击。
- Inference 数据应通过 `Metric scope → Inference` 可见。

5. Generation and validation
依次运行：

python -m pytest \
  dev/tests/test_benchmark_frontend_data.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_domain_coverage.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

cd frontend
npm ci
npm run typecheck
npm run build
npm run test:e2e

6. Assets and workflow
- 保证 `frontend/public/data` 与 `docs/assets/benchmarks` 是当前确定性结果。
- 永久 CI workflow 必须保持 read-only staleness gate，不保留 contents:write 或 self-commit 逻辑。

7. Documentation
- 更新 remaining-module-audit.md：将该项从 gap 移到 completed，并保留尚未覆盖的子矩阵。
- 更新相关专项 plan 和 PR 描述中的 run/source/model 数。

最终报告：
- parser contract；
- 新增 normalized run 数；
- category/model/framework 变化；
- 所有命令及结果；
- CI run；
- PR head；
- 未覆盖的 benchmark 子项。
```

---

## 8. Artifact 验收清单

在进入 full integration 前，逐项确认：

### Provenance

- [ ] source date 合格；
- [ ] git SHA 已记录；
- [ ] dirty state 已记录；
- [ ] 环境和 package versions 完整；
- [ ] artifact SHA256 已计算；
- [ ] canonical copy 与 original artifact 字节一致。

### Timing

- [ ] 使用高精度单调计时器；
- [ ] CuPy 显式同步；
- [ ] Torch 显式同步；
- [ ] warmup 未混入 repeats；
- [ ] raw repeats 已保存；
- [ ] mean/std/min/max 可重算；
- [ ] `std_ddof` 明确；
- [ ] fit-only 与 transfer-inclusive 分开；
- [ ] 没有近零异步假 timing。

### Statistical identity

- [ ] 数据 case 与 method 分开；
- [ ] loss/tuning/scale estimator/initialization 明确；
- [ ] penalty/alpha/path 明确；
- [ ] solver requested/resolved 明确；
- [ ] inference covariance/bootstrap/HAC 明确；
- [ ] CV folds/grid/warm-start/refit 明确；
- [ ] survival ties 明确；
- [ ] operation scope 明确。

### Correctness

- [ ] finite checks；
- [ ] convergence status；
- [ ] backend agreement；
- [ ] external-reference alignment 说明；
- [ ] failed/OOM/unsupported 未被省略；
- [ ] speedup 只来自 matched reference；
- [ ] 慢于 CPU 的结果保留。

---

## 9. 各模块 required matrix

### 9.1 `robust_losses`

专项文档：

```text
docs/benchmark-dashboard/robust-loss-comparison-plan.md
```

Required losses：

```text
squared_error
huber_mad, epsilon=1.35
bisquare_mad, epsilon=4.685
fair_mad, epsilon=1.35
```

Required scales：

```text
5K×50
20K×100
100K×50
5K×500
```

Required regimes：

```text
clean_gaussian
vertical_5pct
vertical_15pct
heavy_tail_t3
leverage_5pct
```

Required backends：

```text
numpy
cupy
torch
```

必须记录 Bisquare initialization、estimated scale、nonzero weight fraction 和 fully rejected fraction。

### 9.2 `penalized_robust_quantile`

先做稳定的 Phase 1：

```text
loss ∈ {huber, quantile}
penalty ∈ {l1, l2, elasticnet, scad, mcp}
backend ∈ {numpy, cupy, torch}
quantile ∈ {0.1, 0.5, 0.9}
```

Bisquare/Fair penalty path 只有在 unpenalized robust benchmark 通过后再加入。

记录：

```text
alpha/path grid
warm starts
selected or fixed alpha
solver requested/resolved
objective/stationarity
support metrics
prediction metrics
```

### 9.3 `cv_models`

前端 contract：

```text
docs/benchmark-dashboard/cv-inference-frontend-contract.md
```

Models：

```text
RidgeCV
LassoCV
ElasticNetCV
LogisticRegressionCV
PenalizedGLM_CV
CoxPHCV
```

至少使用三个规模，覆盖：

```text
small launch-overhead
medium crossover
large throughput or high-p
```

每个 model/backend 记录：

```text
cv_folds
grid_size/path length
warm_start
path construction time
fold fitting time
selection time
refit time
cv_total time
selected hyperparameter
prediction metric
selection metric where applicable
failed folds
```

External references 只在 objective、standardization、intercept、fold split 和 refit policy 对齐时比较。

### 9.4 `linear_inference`

Required methods：

```text
HC0
HC1
HC2
HC3
HAC with explicit kernel and bandwidth
debiased Lasso
bootstrap with multiple replicate counts
oracle inference
```

Required timing scopes：

```text
fit_only
inference_only
fit_plus_inference
```

记录：

```text
BSE
coverage in repeated simulation
Wald/p-value agreement
bootstrap replicate count
HAC kernel/bandwidth
support recovery for oracle/debiased methods
```

### 9.5 `ordered_crossover`

扩展 Ordered Logit/Probit 到能够观察 GPU crossover 的规模。

至少覆盖：

```text
n ∈ {500, 2K, 10K, 50K, 100K where feasible}
p ∈ {5, 20, 50}
number of ordered classes ∈ {3, 5, 10}
backend ∈ {numpy, cupy, torch}
```

分开记录 fit 和 inference；保留 thresholds/cutpoints correctness 与 probability normalization checks。

### 9.6 `anova_crossover`

Functions：

```text
OneWayANOVA
TwoWayANOVA
WelchANOVA
TukeyHSD
BonferroniCorrection
```

必须对每个 backend 显式同步。规模应覆盖 GPU launch-overhead、crossover 和 throughput 区间。记录 group count、observations per group、factor levels、interaction structure 和 number of pairwise contrasts。

### 9.7 `covariance`

Models：

```text
EmpiricalCovariance
ShrunkCovariance
LedoitWolf
OAS
GraphicalLasso
GraphicalLassoCV
MinCovDet
```

规模必须同时覆盖：

```text
n >> p
n ≈ p
p > n
```

记录 covariance/precision matrix error、log-likelihood 或 objective、positive-definiteness、condition number、iterations 和 failure/OOM。

### 9.8 `nonparametric`

Models/operations：

```text
KDE fit/score/sample where supported
Nadaraya-Watson predict
local-linear kernel regression predict
KernelRidge fit/predict
KernelRidgeCV cv/refit
KernelPCA fit/transform
natural cubic spline basis/fit/predict
pairwise kernels beyond RBF
```

不要把 fit、transform 和 predict 合并为一个 timing scope。

### 9.9 `feature_selection`

至少包括：

```text
Knockoff generation
feature statistic computation
threshold/selection
end-to-end
```

比较可对齐的 baselines，并记录 target FDR、estimated FDR、realized FDP、power/recall、precision、selected count 和 runtime stages。至少三个 seeds，推荐更多 seeds 以稳定 FDP/power。

### 9.10 `penalized_survival`

Models：

```text
PenalizedCoxPHModel
CoxPHCV
```

Matrix：

```text
penalty ∈ {l2, l1 if supported, scad, mcp}
ties ∈ {breslow, efron}
backend ∈ {numpy, cupy, torch}
```

记录 path/refit/CV timing、C-index、support recovery、objective、convergence、ties distribution 和 censoring fraction。

### 9.11 `panel_extended`

Models：

```text
PooledOLS
PanelOLS
RandomEffects
BetweenOLS
FirstDifferenceOLS
FamaMacBeth
```

明确记录：

```text
n_entities
n_times
n_features
entity/time/two-way effects
covariance type
cluster dimensions
balanced/unbalanced panel
```

不得再次产生类似 `n=5000p=` 的不完整 scale key。

### 9.12 `distributions`

覆盖已实现的 15 种分布及其适用操作：

```text
pdf/pmf
cdf
sf
ppf
isf
```

Required vector lengths：

```text
10K
100K
1M
```

记录 distribution parameters、domain construction、tail probabilities、raw timing repeats，以及与 SciPy 的 max absolute/relative error。不要从旧 Markdown 反向填 measured 数据；必须重新运行或将恢复值标为 reported。

### 9.13 `multiple_testing`

覆盖：

```text
adjust_pvalues
combine_pvalues
permutation_test
bootstrap utilities where maintained
```

记录 number of hypotheses、correlation/dependence regime、number of permutations/bootstrap replicates、correction method、p-value agreement、rejection count 和 timing scope。

---

## 10. 推荐执行顺序

为了优先补前端最明显的缺口，建议：

```text
1. robust_losses
2. cv_models
3. feature_selection
4. ordered_crossover
5. anova_crossover
6. linear_inference
7. distributions
8. covariance
9. nonparametric
10. penalized_survival
11. panel_extended
12. multiple_testing
13. penalized_robust_quantile
```

每次只让本地模型处理一个 `MODULE_ID`。不要用一个长任务同时运行所有模块，否则容易出现：

- GPU memory 累积；
- benchmark protocol 漂移；
- dependency 交叉污染；
- artifact 不完整；
- local model 为完成任务而静默缩减矩阵。

---

## 11. 本地模型最终答复模板

要求模型按以下格式返回：

```text
## Scope
MODULE_ID:
MODE:
Branch:
Git SHA:
Dirty before / after:

## Environment
Python:
CPU:
GPU:
CUDA driver/runtime:
NumPy/CuPy/Torch/statgpu:
External references:

## Implemented matrix
Expected observations:
OK:
Unsupported:
Missing dependency:
Failed:
OOM:
Timeout:

## Timing protocol
Warmups:
Repeats:
Seeds:
Synchronization:
Timing scopes:

## Commands executed
<exact commands>

## Result artifact
Original path:
Canonical copy:
SHA256:
JSON finite check:
Raw-repeat reconstruction check:

## Main findings
<do not omit GPU slower-than-CPU cases>

## Validation
Backend agreement:
External alignment:
Convergence/failure summary:

## Files changed
<paths>

## Integration status
Parser modified: yes/no
Manifest modified: yes/no
Frontend modified: yes/no
Tests modified: yes/no

## Remaining issues
<explicit blockers and reduced matrix, if any>
```

---

## 12. Repo integration reference

当前 dashboard contract 的关键位置：

```text
dev/benchmarks/frontend_sources.json
dev/benchmarks/benchmark_frontend_schema.json
dev/benchmarks/generate_benchmark_data.py
dev/benchmarks/frontend_data/registry.py
dev/benchmarks/frontend_data/parsers/
dev/tests/test_benchmark_frontend_data.py
dev/tests/test_frontend_contracts.py
dev/tests/test_frontend_domain_coverage.py
frontend/src/data.ts
frontend/src/components/FilterBar.ts
frontend/src/components/OverviewTable.ts
frontend/e2e/domain-coverage.spec.ts
```

当前 source policy：

```text
minimum_source_date = 2026-06-01
```

常用最终验证命令：

```bash
python -m pytest \
  dev/tests/test_benchmark_frontend_data.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_domain_coverage.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

cd frontend
npm ci
npm run typecheck
npm run build
npm run test:e2e
```

若本地环境不能运行浏览器，可以先完成 Python、typecheck 和 build，但必须明确把 Playwright 标为未执行，不能声称 CI 已通过。
