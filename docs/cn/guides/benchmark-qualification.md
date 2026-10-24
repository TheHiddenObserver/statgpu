# 可审计的 benchmark qualification campaign

七套维护中的覆盖 runner 是**候选证据生成器**，不是所有公共 API、所有 CUDA
操作或可发布加速比的认证。`dev/benchmarks/run_qualification_campaign.py`
提供显式启动的串行 campaign：展开到每一行的预期清单、分开的 CPU 线程配置，以及
失败后仍保留的证据归档。它不修改数值实现，也不自动写入 canonical dashboard。

## 先看计划：默认不运行 benchmark

在源码仓库根目录执行：

```bash
python -m dev.benchmarks.run_qualification_campaign > /tmp/statgpu-plan.json
```

不带 `--execute` 时，只使用 Python 标准库读取 registry 和 CPU 配额并输出计划。
不会导入 statgpu、NumPy、CuPy 或 Torch，不启动子进程、不查询 GPU，也不创建输出
目录。仅指定 `--output-dir` 不会触发执行。

默认 `representative` 配置保留既有七套 fullscan 的形状、case、输入 dtype、场景、
随机种子和各 runner 的默认 warmup/repeats，不是缩减后的 smoke：

| Suite | 默认维度 | 每个 backend 行数 |
| --- | --- | ---: |
| Gaussian refresh | 1024×16、8192×64、32768×128；OLS/Ridge；float32/float64 输入；无权重/analytic 权重 | 24 |
| Classification metrics | n=1000/10000/100000；7 个操作；2 种输入 dtype；3 个场景 | 126 |
| Model families | 全部 74 个 case；名义 n=500、p=8 | 74 |
| Statistical families | 全部 138 个 case；n=128/1024、p=4 | 276 |
| Learning families | 全部 44 个 case；n=256、p=4；knockoff p=20 | 44 |
| Workflow families | 4 个 case；n=256/1024、p=6、targets=4 | 8 |
| Six-family CV | 6 个模型；n=240/1024、p=16；3 折和固定 grid | 12 |

默认是七套 × NumPy/CuPy/Torch，共 21 次串行调用，每次请求一个 CPU 线程。
精确预期为 **1692 行**，其中 16 行是 CUDA statistical suite 中允许的 host-only
`not-applicable`，不是 GPU 执行成功。当前 producer 不给这些 NA 行写入 problem
维度，因此 assessor 保留这一缺失并核对精确重复次数，不伪造实测形状；计划仍记录
全部选定的名义形状。平衡 panel 的有效 n 和一维 learning fixture 按实际 producer
规则归一化。

## 在目标 benchmark 机器显式执行

使用经过审查的源码 checkout，预先准备 runner 依赖、`threadpoolctl` 和请求的实际
backend。GPU/数值 benchmark 应遵循项目配置的远程环境规则，本地 harness 测试
不能冒充物理 CUDA 证据。

**先完成定向 regression，再启动 campaign**：

```bash
python -m pytest -q -rs \
  dev/tests/test_cv_clean_timing.py \
  dev/tests/test_learning_compute_qualification.py
```

当前物理单 GPU 矩阵包含 12 个 CV 节点（六模型 × CuPy/Torch）和 88 个 learning
节点（44 case × CuPy/Torch）。这是物理矩阵维度，不是两个文件全部测试的固定总数。
在目标机器上只检查或执行这些物理 gate，可使用：

```bash
python -m pytest -q -rs \
  dev/tests/test_cv_clean_timing.py::test_physical_cv_clean_lane_matches_observed_cuda_workload \
  dev/tests/test_learning_compute_qualification.py::test_physical_gpu_complete_learning_compute_contract
```

本地没有 GPU 时，可以验证 harness，但不能把 skip 当成物理 gate 通过。即使机器
上有 GPU，也必须检查每个 skip 原因，确认目标 backend 的节点确实执行；仍有 skip
或 unavailable，就不能宣称相应物理 closure 已完成。

```bash
OUT="/tmp/statgpu-qualification-$(date -u +%Y%m%dT%H%M%SZ)"
python -m dev.benchmarks.run_qualification_campaign \
  --execute --output-dir "$OUT" --timeout 1800
```

输出目录必须尚不存在、父目录已经存在，解析符号链接后的路径必须在 checkout
之外。已有目录、归档或校验文件都拒绝覆盖，空目录也不例外。超时必须大于零且
不超过 1800 秒，按**每次调用**计时，包含 import/preflight，不是整个 campaign
的总预算。21 次均跑满上限可达 10.5 小时。没有 `--allow-incomplete` 逃生选项。
需要缩小范围时，用 `--suites` 和 `--backends` 明确选择。

普通非零退出、缺少依赖、JSON 缺失/损坏、超时、数值检查失败和不完整产物都会被
保留，然后继续下一次调用。源码发生变化或触发资源 guard 时停止剩余 campaign。
资源 guard 检查当前 CPU affinity/quota、512 MiB 输出磁盘余量，以及可读 cgroup
下的 256 MiB 内存余量。这是保守的停止条件，不保证能预防所有主机/GPU OOM。
需要终止时，只终止本 campaign 启动的当前子进程组，绝不处理其他 GPU 用户的进程。
即使 leader 先退出，也会清理自己的进程组；仍存活的后代进程会令本次调用失败，
不能在证据冻结后继续写入。

只有选定的全部展开行、报告检查、进程退出、源码身份和线程验证均通过，campaign
才以 complete 成功退出。dirty/未知源码保持不完整。成功不代表公共 API 全覆盖。

## CPU 线程配置与有界 scaling

额外线程数是独立的 **NumPy-only** 配置，不会暗中重跑所有 GPU 作业：

```bash
python -m dev.benchmarks.run_qualification_campaign --cpu-threads 1 2 4 8
```

必须包含基线 1。可选线程数不能超过 scheduler affinity 和实际 cgroup CPU quota
向下取整后的上限。quota 小于 1 时，单线程基线可能被节流，但不允许额外线程。Linux quota 无法读取时也只允许单线程基线。
线程环境变量在子解释器启动前设置；子进程导入 runner 后和工作结束后，分别检查
实际 `threadpoolctl` pools。Torch 配置还设置并验证 intra-op/inter-op 数量。
请求线程数是资源上限，不声称每个 pool 都恰好用了该数量；实际数量与是否相等的
标志都会保存。不能暗中挑选最快的 CPU 配置混成一个未说明的对照基线。

medium/large 必须显式指定 suite，每套最多选择八个精确 case ID。CV runner
没有逐模型选择器，因此保留六模型契约。下面仍然只生成计划：

```bash
python -m dev.benchmarks.run_qualification_campaign \
  --scale large --suites learning-families six-family-cv \
  --cases learning-families=pca,kde --backends numpy cupy torch
```

较大 learning 配置只允许 KDE、NW kernel regression、PCA、incremental PCA、
truncated SVD、k-means、minibatch k-means、Nystroem、B-spline、natural/cyclic
spline，以及 Scott/Silverman/NRD/NRD0 带宽规则；参数使用 registry 中精确的 case
ID。medium/large 的 learning n 为 1024/2000，显式 pairwise cap 为 4,000,000
元素，不提供自动运行全部 44 个 large case 的入口。model/statistical/workflow
的 n 上限为 2048；CV 为 1024/2048，不会默认启动 n=5000/10000 的 CV。
Gaussian 为 8192×64/32768×128，classification 为 n=10000/100000。
部分点与 fullscan 重合，不能据此宣称每个默认 case 都完成了 scaling。
可选 `--warmup`、`--repeats` 覆盖是显式且有上限的，清单会记录实际选择。

## 证据可以、不能说明什么

每次调用保存原始产物和独立 qualification sidecar。sidecar 比较展开后的身份
**多重集合**，不是只比较 case 名称：backend、形状、输入 dtype、seed，以及适用的
权重/场景和每一项重复次数；timing 样本数也必须符合实际 CLI 的 repeats。
成功的进程退出不能覆盖 failed/partial。不会重写原始 JSON 来“修好”结果。

float32 输入不等于 float32 计算。分类率的提升精度、Gaussian dtype promotion、
host orchestration、diagnostic accessor、learning host/hybrid 阶段和 CV 显式
float64 精度基线，必须分别说明。输出在 CUDA 上不证明整个算法是 native CUDA；
要结合各 producer 契约审阅数值阶段 trace 和具体同步设备。

CV 的 clean public-fit timing 与带 instrumentation 的 selector/refit profile
分开。子进程 peak RSS 按平台单位归一化为字节，覆盖 import、reference 和 runner
全进程高水位，不是某个算法的独立内存占用。**GPU allocator peak 为
`not_measured`**，pairwise 安全估算也不是实测 VRAM。若子进程在 finalizer 前被
终止，peak RSS 可能缺失，不从父进程读数伪造。

所有性能数据保持 `diagnostic_only` 和 `UNVERIFIED_EXCLUSIVITY`，不生成加速比，
不自动晋升 canonical dashboard。未知或被 Docker namespace 隐藏的 GPU PID
不会阻止 correctness，也不会证明独占。campaign 不 kill NVML 列出的 PID、不挂载
主机 namespace，也不通过特权操作推断进程归属。

## 失败也保留的证据结构

新输出目录包含：

- `plan.json`：精确命令、形状、展开身份和资源限制；
- `source-before.json`、`source-after.json`，以及有 hash 的源码/脚本副本；
- `child_bootstrap.py` 与父进程 runtime 信息；
- 每次调用的目录：`invocation.json`、生成时的原始 `raw.json`、stdout/stderr、
  实际退出/超时、源码快照、可获得的实际线程/runtime 报告、`qualification.json`；
- `summary.json` 与 `SHA256SUMS.json`。

正常完成、普通失败、源码变更、资源停止和捕获到的异常，都会收尾保存已有证据。
文件设为只读，并生成相邻的 `OUT.tar.gz` 和 `OUT.tar.gz.sha256`，不覆盖历史证据。
Hash 使后续修改可检测；只读权限不是对管理员也不可改的加密存储保证。机器断电、
父进程被强制 kill、磁盘故障或归档写入失败，仍可能导致无法收尾。

不运行数值/GPU campaign 的 harness 验证命令：

```bash
python -m pytest dev/tests/test_qualification_campaign.py -q
```

这些测试使用合成产物和纯 Python 子进程替身。真实 correctness/performance 证据
仍需在目标源码与硬件上另行授权执行。相关背景见 [benchmark guide](../benchmarks.md)
和 [设备与内存指南](device-and-memory.md)。
