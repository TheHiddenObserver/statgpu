# 可审计的 benchmark qualification campaign

Benchmark 覆盖有三种不同的分母：公开能力、已注册 workload，以及某次运行实际
选定的有限配置。完成一种分母不等于完成另外两种。维护中的 runner 是**候选证据
生成器**，不是所有公共 API、所有 CUDA 操作或可发布加速比的认证。

`dev/benchmarks/run_full_benchmark_matrix.py` 是已注册配置与独立协议的统一
先规划入口。`dev/benchmarks/run_qualification_campaign.py` 提供七套已注册 suite 的显式串行
执行，保留精确 workload 清单、独立 CPU 线程 profile，以及失败后仍保存的证据。
默认的 **21 次调用、1692 行，是已注册的代表性矩阵，不是整个 benchmark 的覆盖
范围**。更完整的计划还必须纳入独立 runner 和默认 fixture 没有表示的配置轴。

## 先明确覆盖分母

| 分母 | 实际统计对象 | 不能证明的内容 |
| --- | --- | --- |
| 覆盖审计中的 151 项公开能力映射 | direct、runner-smoke、indirect、alias 和 infrastructure 关系 | 151 项独立数值通过，或物理 GPU 证明 |
| 275 个已注册 case 身份 | 2 Gaussian + 7 classification + 74 model + 138 statistical + 44 learning + 4 workflow + 6 CV | 完整 solver、dtype、权重、形状或 backend 矩阵 |
| 默认展开的 1692 行 | 默认形状、种子、输入 dtype、场景和三后端下的精确 case | 穷尽 API；其中 16 行还是预期的 host-only CUDA `not-applicable` |
| 仓库审计中的 41 个额外参数化 runner 文件 | 七套之外的独立计时入口 | 41 套互不重叠且可安全运行的 suite，或额外 41 项通过 |
| 选定的有限矩阵 | 生成计划中声明的精确配置和证据身份 | 所有可能输入/配置，或任何未执行、被阻断的维度 |

仓库审计检查了 202 个 tracked Python 文件，含顶层 158 个、嵌套 44 个。这是源码
清单的一个快照，不是实时文件总数，也不是 suite 数量。其中还包括 helper、wrapper、
validator、diagnostic、legacy 程序和远程 launcher。wrapper 与它调用的子程序不能
重复计数；indirect/alias 能力映射也不能变成额外的独立数值证据。

有效的完成报告应区分**计划、已执行、通过、失败、blocked、不适用和未运行**。
被阻断的轴仍属于覆盖分母，不能为了全绿而静默删除。完成全部选定的可执行作业，
不等于所有声明的轴都已闭合。比较或去重时，保留 case、配置、backend、形状、
输入/计算 dtype、seed、线程 profile 和计时协议的精确身份。representative、
medium 和 large 中重叠的点，不会因为分属不同批次就增加独立覆盖。

## 从统一计划入口开始

统一入口组合已注册配置计划与独立 runner 清单：

```bash
python -m dev.benchmarks.run_full_benchmark_matrix \
  > /tmp/statgpu-full-plan.json
```

默认只规划：包含不变的注册代表性矩阵，以及全部 41 个独立入口和各自 disposition。
不会启动 benchmark workload、导入数值 runner 或查询 GPU。“full”表示统一核算
范围，不代表每个入口都可运行，也不代表所有 API 已覆盖。先检查计划，再选执行 lane。

选定计划时，将以下 JSON 保存到仓库外配置文件，例如
`/tmp/statgpu-matrix-config.json`：

```json
{
  "schema_version": "statgpu-full-benchmark-config-v1",
  "registered": {
    "argv": ["--suites", "six-family-cv", "--seeds", "17", "42"]
  },
  "independent": {
    "lanes": ["public-native-kde-bootstrap"],
    "configurations": {
      "public-native-kde-bootstrap": [
        {"n":128,"dimension":1,"bandwidth":0.4,"weights":"none","seed":17},
        {"n":256,"dimension":3,"bandwidth":0.4,"weights":"unequal","seed":42}
      ]
    },
    "validation_tier": "remote-full",
    "environment_id": "my-benchmark-machine",
    "expected_sha": "<full-commit-sha>"
  }
}
```

执行前，将 `<full-commit-sha>` 替换为已审查 checkout 的完整 40 字符 commit，并
填写实际环境标识；示例占位符不是有效执行值。`remote-full` 只标记证据 tier，
不会连接服务器或启动 SSH，命令必须在目标环境中运行。示例选择两组有界公开
KDE 配置，每组都使用三后端；未填字段沿用 adapter 默认值。`registered.argv`
中的 backend 选项不会筛选独立 lane。

```bash
python -m dev.benchmarks.run_full_benchmark_matrix \
  --config /tmp/statgpu-matrix-config.json > /tmp/statgpu-selected-plan.json

# 审查选定 workload、依赖与资源预算之后，才显式执行：
python -m dev.benchmarks.run_full_benchmark_matrix \
  --config /tmp/statgpu-matrix-config.json --execute \
  --output-dir /tmp/statgpu-selected-evidence-NEW --timeout 1800
```

省略 `registered` 或 `independent` 即只选择另一条 lane。registered 的 argv
使用下文 campaign CLI，不允许嵌套 `--execute` 或 `--output-dir`，执行与证据
位置由 coordinator 管理。独立 lane 执行时必须显式选择不重复的 lane ID；全部
inventory 的默认计划不能直接执行。未知/重复 JSON key 会被拒绝。输出必须是
checkout 外尚不存在的路径。coordinator 保存配置 hash 和源码快照，将各 lane
证据放入 `registered/`、`independent/`，并冻结汇总与归档。
`complete_for_selected_protocols` 不代表整个库已通过认证。

## 先看计划：默认不运行 benchmark

在源码仓库根目录执行：

```bash
python -m dev.benchmarks.run_qualification_campaign > /tmp/statgpu-plan.json
```

不带 `--execute` 时，只使用 Python 标准库读取 registry 和 CPU 配额并输出计划。
不会导入 statgpu、NumPy、CuPy 或 Torch，不启动子进程、不查询 GPU，也不创建输出
目录。仅指定 `--output-dir` 不会触发执行。

默认 `representative` 配置保留既有七套 fullscan 的形状、case、输入 dtype、场景、
随机种子和各 runner 的默认 warmup/repeats，不是缩减后的 smoke，但仍是一组固定 fixture：

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
的总预算。21 次默认调用的 timeout 之和为 10.5 小时，还不含调度和收尾开销；
这既不是耗时预测，也不是整个 campaign 的严格 wall-clock 上界。没有
`--allow-incomplete` 逃生选项。
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

全部 44 个 learning 身份现在都可以选择显式形状，但不是共用一个规模上限。
超出 case 的资源/oracle 契约时会拒绝，不会静默缩小输入：

| Learning 工作类型 | 最大 n | 最大名义 p | 典型成员 |
| --- | ---: | ---: | --- |
| Linear/streamed fixture | 2000 | 64 | PCA、k-means、Nystroem 等有界 fixture |
| Dense factorization | 1024 | 16 | Kernel ridge、KernelPCA、thin-plate spline、agglomerative、DBSCAN |
| Iterative embedding | 512 | 16 | t-SNE、UMAP、UMAP NNDescent |
| Iterative factorization | 2000 | 32 | NMF、minibatch NMF |
| Bandwidth search | 1024 | 4 | UCV、BCV、SJ 系列与 bandwidth CV |
| Kernel CV | 512 | 16 | KernelRidgeCV |
| Bootstrap | 2000 | 4 | 已注册 KDE bootstrap fixture |
| Selection | 2000 | 64 | fixed/model knockoff；p 由 `--knockoff-p` 选择 |

这些是 harness 的已审查边界，不是 estimator 可扩展性结论，也不是峰值内存硬保证。
Learning 还要求 n ≥ 2p，且 n² 不超过 4,000,000 元素；
`--max-pairwise-elements` 只能收紧，不能提高这一上限。一维 fixture 的行身份仍
记录实际有效 p。model/statistical/workflow 的 n 上限为 2048，quantile 参考为
2000；CV 上限为 2048，不会默认启动 n=5000/10000。Gaussian 和 classification
分别保留 n=32768 与 n=100000 的既有上限。可选 `--warmup`、`--repeats` 有明确
边界，并写入清单。

### 扩展 profile、显式 seed 与形状

`--profile expanded` 为每套 suite 规划三个候选形状，以及三个独立 seed
（17、42、20261020）；非 CV case 分成每个作业最多八项的批次，CV 仍保持六模型
契约。精确作业/行数以计划为准，不能直接把默认 1692 行相乘。超出边界的候选
case/shape 组合写入 `capability_exclusions`，不是通过。显式请求不支持的形状或
case 会失败，不会静默丢弃用户选择的工作。

```bash
python -m dev.benchmarks.run_qualification_campaign \
  --profile expanded > /tmp/statgpu-expanded-plan.json

python -m dev.benchmarks.run_qualification_campaign \
  --suites learning-families --cases learning-families=tsne,umap \
  --shapes learning-families=256x4,512x4 --seeds 17 42 \
  > /tmp/statgpu-embedding-plan.json
```

`--seeds` 最多接受 32 个不同种子，`--shapes` 每个选定 suite 最多接受八个不同
`NxP`；classification 使用 `Nx0`。显式形状不能与 `--scale medium` 或
`--scale large` 混用。dtype、权重、resamples、targets、Gaussian alpha、
knockoff 特征数/selection trials 和 device ordinal 都有逐 suite 契约；存在
一个选项不代表所有 suite 都支持该轴，不支持的请求会被拒绝。knockoff selection
seeds 是独立 trial，不是 timing repeat，也不能再与多个 campaign seed 重复相乘。

确定性分区使用相同选项并添加 `--batch-count K` 和从零开始的 `--batch-index I`。
每批同时记录本批总数和所有批次的分母，job 身份跨分区保持稳定。每批使用新的输出
目录，保留所有计划，并核对全部 expected slot 恰好一次；单批成功不代表整个矩阵
完成。每个 expected slot 都带有行、配置和源码内容身份。没有 `--execute`，以上
计划仍不会执行 benchmark。

### 命名 runner 配置

Campaign 可重复使用 `--config SUITE:NAME=JSON`，每个 suite 最多枚举 16 个命名
profile。这里的 JSON 对象是一个 CLI 参数，不是统一 coordinator 的配置文件选项。
profile 名在 suite 内必须唯一；值按每个选定 case 的 fixture/oracle 支持范围检查。
未知 key、重复 key 和不支持的组合在执行前就会失败。命名配置中的 dtype/weights
不能同时与相同的全局 `--dtypes`/`--weights` 轴混用。

```bash
python -m dev.benchmarks.run_qualification_campaign \
  --suites model-families --cases model-families=glm:gaussian,glm:poisson \
  --config 'model-families:baseline64={"dtype":"float64","solver":"irls"}' \
  --config 'model-families:weighted64={"dtype":"float64","weights":"analytic","solver":"irls"}' \
  --seeds 17 42 > /tmp/statgpu-glm-config-plan.json

python -m dev.benchmarks.run_qualification_campaign \
  --suites six-family-cv \
  --config 'six-family-cv:fivefold={"folds":5,"grid_size":3}' \
  > /tmp/statgpu-cv-config-plan.json
```

当前命名 profile 的 key 与边界：

- Model families：`dtype`、`weights`、`solver`、`sparse`。float32 输入限于
  GLM/GLM-inference/OLS-inference fixture。analytic weights 对齐普通 GLM
  的系数目标与 OLS/WLS inference，不代表带权 GLM covariance 已通过认证。
  显式 solver 菜单取决于 case 与 penalty。
- Learning families：`dtype`、`weights`、`solver`、`sparse`、`covariance_type`。
  float32 fixture 覆盖 PCA、incremental PCA、truncated SVD、k-means/minibatch
  k-means、DBSCAN、agglomerative 与 Gaussian mixture。PCA 显式 solver 为
  full/covariance/randomized；Gaussian mixture covariance 为
  diag/full/tied/spherical。本 runner 不接纳 analytic weights，公开 KDE 的
  加权工作流使用独立协议。
- Six-family CV：`dtype`、`weights`、`sparse`、`folds`、`grid_size`。仍是
  float64、无权重、dense 精度基线。folds 为 2–5，grid size 为 2–9，最终 refit
  之前最多 27 次 candidate-fold fit。改变 folds/grid 不会产生混合精度默认性能 lane。

这些配置 adapter 目前都要求 dense 输入。请求 sparse 会明确列为不支持，不会
暗中转 dense，也不宣称 CSR/CSC 覆盖。输入精度与观察到的计算精度始终分开。
这些菜单扩展了选定 fixture，不代表所有公开 solver、covariance、family 和权重
组合都已覆盖。

## 独立 runner 与配置缺口

审计中的 41 个独立文件当前对应 25 个可执行 adapter（14 ready、11 依赖外部前置），
9 个 blocked admission、6 个 unsupported contract，以及一个已由六模型 campaign 复用的 CV-models 去重项。“Ready”只表示存在执行/
校验契约，不代表已在目标硬件通过，更不代表整个参数空间都覆盖。
`independent_benchmark_catalog.json` 保存审计 hash 与 wrapper 关系；实际运行
重新获取源码 fingerprint，不把历史审计 hash 当作新运行证据。新增 support 文件
单列在 `new_since_audit` 中，不加入历史 202 文件分母。

只检查全部独立 disposition 或某条 lane，不执行 workload：

```bash
python -m dev.benchmarks.run_independent_benchmarks \
  > /tmp/statgpu-independent-plan.json
python -m dev.benchmarks.run_independent_benchmarks \
  --lanes public-native-kde-bootstrap > /tmp/statgpu-kde-pilot-plan.json
```

独立执行必须显式提供 `--lanes`、`--execute`、新的仓库外 `--output-root`、
包含已审查完整 40 字符 commit 的 `--expected-sha`，以及 `--environment-id`。
`--validation-tier` 可为 `local-minimal`、`local-full` 或 `remote-full`，
只记录上下文，不会连接远程。所需包与物理 backend 必须事先可用；executor 不会
安装依赖、使用 SSH 或静默回退 CPU。

统一 JSON 的 `independent.configurations` 将选定 lane ID 映射到有界 override
列表，如上例所示。独立 CLI 的等价选项是
`--configurations /path/to/configurations.json`，文件中只放这一映射。每条 lane
接受 1–128 个配置，拒绝精确重复；每个归一化配置有自己的身份与 expected cells。
默认值是有界 pilot，不会自动展开整个参数空间。

公开 KDE override 包括 n、query/bootstrap 数、维度、kernel、数值/字符串
bandwidth、权重模式、seed 和 warmup/repeats。其中权重标签 `sparse` 表示含零
权重，不是稀疏输入矩阵。其他 adapter 只提供各自审查过的轴，例如 Exact ties
的 sizes/scenario、strata 数、panel shape、hybrid case、compile shape/case。
未知 key、不支持的值和超过组合资源边界的工作量都会被拒绝，没有任意命令或
extra-arguments 透传。更大的计划仍需审查总工作量和统计/oracle 含义。

Adapter 使用单进程串行、一个 CPU 线程。每条 lane 预算为 900 秒，compile/
Fama–MacBeth 协议为 1800 秒，还会受请求的 `--timeout` 进一步限制。除既有
cgroup/disk guard 外，还检查 1 GiB GPU 剩余显存与 host 可用内存余量。这些是
准入/停止条件，不是峰值内存或不发生 OOM 的保证。要检查各 adapter 的实际形状与
协议；有界 Fama–MacBeth lane 并不是 tiny KDE 的工作量。

每条 lane 保留生成时的 `raw.json`、`qualified.json`、进程日志、源码快照与可用
runtime 报告。仅计时协议即使 complete，仍可能是 `correctness_status=unavailable`；
设备可用也不等于数值操作确实在该设备执行。blocked/failed 项保留在汇总中，全部
结果仍不具备性能发布资格。不会重写 raw artifact 来获得通过。

独立 lane 包括公开 KDE bootstrap、hybrid phase 诊断、Cox ties/strata/cluster
与稳定性、panel covariance 与 Fama–MacBeth scaling、inference/quantile、
group/penalized GLM 对比、无监督学习与外部包对照、Torch compile 冷启动/稳态、
memory cleanup，以及重复 seed 的统计校准。接入统一 adapter 前，每条 lane
都必须声明输出契约、依赖、预期行和资源边界。有 CLI 或 `main` guard，并不等于
可安全运行或覆盖完整。

不要遍历整个 benchmark 目录执行文件，不要通过导入 legacy 程序探测 CLI，也不要
把默认 SSH launcher 当作本地 runner。固定输出路径、import-time workload、
无界默认值、缺少结构化产物或未经审查的子进程行为，需要适配或明确保留 blocked。
外部 R/第三方包对照、multi-GPU/MPS 检查也需要各自的依赖与受支持硬件。

配置覆盖应是有限、逐算法定义的矩阵，不是无差别笛卡尔积。合法组合与明确不支持
的情况分别登记，尤其注意：

- solver、loss/family、penalty、covariance/inference、权重、稀疏格式、
  folds/grid/refit、backend/device、seed 和规模都需要显式轴。固定注册 case
  不能代表所有公开参数组合。
- 多 seed 可检查数值与随机行为，但少量 seed 不是 coverage、power 或 FDR
  的正式统计校准。
- dense workload 不能证明 CSR/CSC 支持；含零权重不是稀疏矩阵覆盖；float32
  输入不能证明 float32 计算。
- 规模限制与 oracle 限制不同。例如 large quantile 不能静默超过 dense-LP
  参考的 n=2000 上限，必须有适用 oracle，否则保持 blocked。
- 未纳入较大 profile 的 learning 算法仍是覆盖缺口。pairwise、quadratic、
  bootstrap 和优化 workload 需要逐算法预算，不能只共享一个 n 上限。更大的
  CV、query 数、bootstrap 数和 Monte Carlo 重复数都需要单独决定 workload。

### 公开 KDE 与实验协议不可混用

公开 native KDE bootstrap runner 测量公开区间工作流。controlled-index 开发
kernel、包含 RNG 的实验和 hybrid phase 测量各有不同范围，计时不能互换，也不能
当作对同一公开 API 的三份独立证明。不能通过加减 phase 时间拼出没有实测的 E2E。

kernel、维度、数值/字符串 bandwidth、权重模式、query 数、bootstrap 数和
workspace 策略都是独立轴。cosine/optcosine 只支持一维。数值 bandwidth 是协方差
因子（H = b²Σ）；native 字符串 selector 每次 draw 重新拟合，而 NumPy 一维高斯
快速路径固定初始因子。跨 backend 的 RNG 即使 seed 相同，也不是配对 draw。
selector 与后端算法差异应作为不同协议，而不是相同工作量下的加速比。

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
仍需在目标源码与硬件上另行授权执行。历史成功运行只对应记录中的源码与环境，
不能验证改过的代码、更大的计划、新 adapter 或未执行的 backend。相关背景见 [benchmark guide](../benchmarks.md)
和 [设备与内存指南](device-and-memory.md)。

### 可调度不等于数值通过

新增配置入口不是所有配置均通过的承诺。randomized PCA仍使用原exact rank-2重建误差门禁，高维近似结果可能达不到该精度；加权solver组合即便系数误差通过，收敛警告仍会使计时失效。此类结果保留failed，不在看到结果后移出分母，也不放宽原容差。

## 配置扩展的回归门禁

执行新配置协议前先跑控制与oracle回归：

```bash
python -m pytest -q -rs dev/tests/test_campaign_configuration_matrix.py dev/tests/test_registered_benchmark_axes.py dev/tests/test_full_benchmark_matrix.py dev/tests/test_independent_benchmark_registry.py
```

新增物理测试有58个单GPU参数节点，覆盖CuPy/Torch加权模型输入精度、PCA solver、GMM covariance，以及2 folds/4 candidates下六个CV家族（每个backend测试调用内部核对六家）。其中新增20个推断组合，覆盖协方差类型、输入精度、权重与后端；既有4个加权HC3组合仍保留在原组中：

```bash
python -m pytest -q -rs -x dev/tests/test_registered_benchmark_axes_physical.py
```

目标双后端CUDA环境中58个call phase均须通过；本地缺后端skip不能记为物理验收通过。这些测试补充此前设备/clean timing门禁，不替代它们，也不证明所有配置。执行前按实际source固定collect节点清单；该测试文件通过不代表expanded配置扫描或独立runner已执行。

## 显式调用次数预算

plan-only可以枚举大矩阵，执行默认最多21次调用，单次dispatch预算不能超过64。完整expanded必须分批，而不是一次全部启动：

```bash
python -m dev.benchmarks.run_qualification_campaign --profile expanded \
  --batch-count 48 --batch-index 0 --execute \
  --output-dir /tmp/statgpu-expanded-batch-00
```

每批用新输出目录，最后按全局slot身份核对无遗漏/重复。--max-invocations只确认调用次数，不证明耗时或内存安全；独立dispatcher同样默认21、上限64，统一JSON使用independent.max_invocations控制该lane。各child仍受计划timeout/资源门禁约束。statistical resamples范围2–1000；CV n5000/10000及未审查的大型quadratic任务不会因增加调用次数而解锁。
