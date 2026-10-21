# statgpu 0.3.0 全能力 benchmark 执行计划

> 下方结果快照属于修复前的 v2 历史记录。随后授权的未发布修复、当前验证结果与适用边界见 `benchmark_failure_repairs_v030.md` 及新的修复包验证记录；不要把 v2 的失败状态或来源用于认证修复后的数值源码。

基线：`efbfb80570eb1632e0605d68d410715dac4d9be8`。日期：2026-10-20。

本计划取代上一版仅含两个先导 runner 的实现交付。此次交付目标是所有主要公共统计能力都有可审计的执行入口，而不是把两个示例当作完整 benchmark 更新。

## 怎样判断覆盖

覆盖分为三层，不能互相替代：

1. **能力/操作有可执行 case**：每个主要模块和维护能力，有具体 runner、CLI、oracle 或明确边界。aliases 不重复计为独立统计方法。
2. **配置维度已实测**：backend、n/p、dtype、penalty/solver、weights、CV grid/folds、inference 等维度必须看各 case 声明的实际范围。一个模块的 smoke 通过，不代表该模块所有组合通过。
3. **性能证据可发表**：在锁定源码与环境的物理设备运行，数值和收敛先通过，再审查计时/传输/设备归属、原始重复值，最后注册 dashboard。CPU smoke 不能替代这层。

此次补齐第一层并尽可能展开第二层的可配置入口；未运行物理 GPU，因此不能宣称完成第三层。任何 blocked、failed、skipped 都不得计入通过数。

## 代码结构

- `full_coverage_manifest.json`：逐能力/操作清单、执行入口、现有 runner 复用、依赖和剩余维度，另外保留不可自动执行的 legacy 目录。
- `run_coverage_benchmarks.py`：统一入口，默认只列清单；显式 `--execute` 才启动本机子进程。无 SSH、无凭据读取、无自动安装、无 push/upload。超时记录失败，失败不中断其他 suite，最终 summary 保留每个状态。既检查退出码，也检查结果 JSON 中嵌套的失败/不可用状态。
- `benchmark_release_refresh.py`：OLS/Ridge、无权重/解析权重、输入 float32/64、fit/predict。
- `benchmark_classification_metrics.py`：七个 binary evaluation API，不平衡/并列分数/常数分数。
- `benchmark_workflow_coverage.py`：diagnostics、stepwise、multi-target、formula。
- `benchmark_statistical_coverage.py`：分布、multiple testing、resampling、covariance、ANOVA、ordered。
- `benchmark_model_coverage.py`：GLM、penalties、robust/quantile、Cox、Panel、inference。
- `benchmark_learning_coverage.py`：nonparametric、kernel/spline、GAM、unsupervised、Knockoff。
- `benchmark_cv_coverage.py` 适配并复用 `benchmark_cv_models.py`：六类 CV 的 selected parameter / fold / final refit 对齐及计时，保持原统计契约；原 canonical schema 要求多后端记录，不能直接拿单后端结果冒充该 schema。CV 细分计时包含 profiler overhead，仅作明确标注的 instrumented candidate。

以实际 manifest 和各 runner `--help` 为准；suite 数量不能当作覆盖数量。详细方法、操作和边界在 manifest 的 capabilities 中逐项列出。

## 执行方法

在基线仓库根目录复制代码后，先安装/确认对应依赖。推荐独立 venv；本包不安装依赖。CPU 基本环境需要 numpy、scipy、scikit-learn、pytest、jsonschema；相关 oracle 还需要 pandas、patsy、statsmodels，其他按 manifest 单列。GPU 环境另需版本匹配的 CuPy/Torch/CUDA；R 只在启用对应外部比较时需要。

```bash
# 默认不运行任何 benchmark，只显示执行目录
python -m dev.benchmarks.run_coverage_benchmarks

# 首先验证整个执行链；结果目录必须新建或为空，避免覆盖证据
python -m dev.benchmarks.run_coverage_benchmarks --execute --smoke --output-dir results/coverage-cpu-smoke

# 实际规模的 CPU 候选测量
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m dev.benchmarks.run_coverage_benchmarks --execute --backend numpy --output-dir results/coverage-numpy

# 在准备好的物理 GPU 主机，分别运行；不可用/未验证路径会明确 partial/blocked
python -m dev.benchmarks.run_coverage_benchmarks --execute --backend cupy --output-dir results/coverage-cupy
python -m dev.benchmarks.run_coverage_benchmarks --execute --backend torch --output-dir results/coverage-torch
```

`--suites` 可选部分 suite；具体 ID 用默认 inventory 输出查询。`--timeout` 为每个子进程的上限；默认 1800 秒，不是所有规模的保证。`--allow-incomplete` 仅放宽部分未覆盖状态的退出码，不把记录改成 pass，也不能掩盖数值失败。正式验收不要用该选项来宣称全通过。

性能实验需要分别固定 CPU 线程预算、记录 BLAS/CUDA/硬件。上面的单线程示例是一种公平预算，不代表唯一推荐的最佳性能配置；多线程结果单独作为环境分组。

## 运行顺序与扩展轴

1. 全部 tiny CPU smoke：验证 runner/oracle/输出协议，定位不能执行的 API，而不是先跑数小时 GPU。
2. 物理三后端的小规模 correctness：本机不可确认设备归属的路径在 GPU 端仍需通过原生数组/执行位置检查。host-only 诊断必须标注 CPU 阶段。
3. 每个系列做 small/medium/large 和关键交互：n/p/targets、预测 batch、dtype、权重、tol、grid/folds、covariance。条件不合法的组合记录不适用，不把它们伪造为成功。
4. 增加 numerical-difficulty 和 accuracy/time 扫描：病态/秩亏、大偏移、零权重/极端权重、类别分离、尾部 quantile；复用维护 validator 的正确性 fixture。已知 bug 用 failed/blocked 留档，不修改科学目标求通过。
5. 增加真实内存峰值、cold-start、transfer-inclusive 和多硬件环境；不能把现有 allocator cleanup snapshot 当峰值。
6. 所有合格结果再进入 parser/manifest/dashboard，历史来源不覆盖，source SHA 与数值实现不偷换。

## 原有 issue 与新增范围

继续使用 #101–109、#117、#118，避免重新开重复任务：Distribution、Robust/Quantile、Feature Selection、Ordered/ANOVA、inference、Nonparametric/Covariance、Penalized Cox、Panel、Unsupervised/utilities、dtype、CV peak memory。

额外明确的范围包括 binary metrics、host regression diagnostics、发布版本回归、多响应、监督预测吞吐、formula/input pipeline、六类 CV 的多尺度。Stepwise 属于 #103 的具体子矩阵，不是新的独立 tracker。

现有 dashboard 为 21 sources / 2,580 rows，其中 1,136 timing rows；六类 CV 的历史数据规模仅 n=240、p=16。新脚本存在不等于旧 dashboard 已更新；发布仍需新的真实 artifact 和注册测试。

## 准入与安全边界

- 不自动执行带硬编码远程连接/过时 API 的 legacy runner。目录中留其历史用途与阻断理由，不复制其敏感配置。
- 不改生产数值代码来让 benchmark 通过；只修 runner/oracle/参数对齐问题。若发现生产 bug，保留失败证据并单列处理。
- 每次记录 exact source、runner 和 helper hashes、工作树覆盖、数据身份、环境、重复次数、计时范围、误差和状态。
- 不能从一次统计 simulation 的 finite output 宣称 FDR/coverage 已验证。Monte Carlo calibration 需要独立 seeds、足够 repetitions、误差条及明确 target。
- 有限参数/solver 组合不叫穷尽所有组合。最终交付报告必须写明实际能力数、CPU pass/blocked 数、GPU 未测范围，不能只报“全覆盖 PASS”。
- 此次任务只交付代码、计划和本地验证；未推送、未提交 PR、未发布测量结果。

## 最终核验

实际测试数、执行结果和源码文件哈希见随包 `VERIFICATION.txt`、`CONTENTS.sha256` 和 smoke evidence。以最终文件版本为准，不引用开发中的旧结果证明后续修改。

## 本次实现范围与结果快照

执行目录覆盖 124 个顶层公开导出，并加入分类指标、分布族、NNDescent、低层 kernel 和工作流条目，共 151 条能力登记。基础类型/配置对象单列为非独立数值 benchmark，aliases 与间接消费覆盖明确标注，不能把登记条数当作独立统计方法数。

七个 suite 的 CPU smoke 共 316 个 case：Gaussian 8、metrics 42、model 74、statistical 138、learning 44、workflow 4、CV 6。最终状态与原始证据以 `VERIFICATION.txt` 和统一 summary 为准。

目前实际发现并保留的边界：
- `distributions/f/rvs`：NumPy 参数名错误，已有 [#173](https://github.com/TheHiddenObserver/statgpu/issues/173)。失败，无 timing。
- `cyclic_spline`：周期边界约束不满足，已有 [#216](https://github.com/TheHiddenObserver/statgpu/issues/216)。blocked，无 timing。
- `umap_nndescent`：NumPy 2 数组的 device 属性导致错误 Torch 路由，已有 [#191](https://github.com/TheHiddenObserver/statgpu/issues/191)。blocked，不冒充可选依赖缺失。
- inverse-Gaussian HC0 与 negative-binomial HC0：对齐 statsmodels 时 SE 最大差分别约 0.004205 与 0.013751。独立 observed-Hessian sandwich 与 statsmodels covariance 相差 <1.8e-17，而当前结果匹配 Fisher-bread sandwich。这两个 case 保留为参考不一致失败，无 timing；尚未发布新 issue，也未改生产推断。

422 项 harness/目录/协议测试通过不表示上述失败 case 通过：测试刻意要求已复现的失败不得被吞掉或附带可用性能时间。CV 有 14 条已知 deprecation FutureWarnings，不能把它们说成 zero-warning。

当前仍未完成的配置与证据范围包括：全部 loss×penalty×solver×CV、所有 covariance/fold/grid/weights/condition 组合、足量 Monte Carlo calibration、物理 CUDA/MPS/多卡、内存峰值、cold-start/transfer-inclusive、多版本同机回归与 dashboard 注册。这些保留为明确计划项，不声称已经闭合。

GPU 输出数组的设备归属不自动等于全部运算发生在 GPU。对于只能确认 ownership 的 learning 路径，性能资格标为 false，保留诊断记录，不认证 GPU 加速。
