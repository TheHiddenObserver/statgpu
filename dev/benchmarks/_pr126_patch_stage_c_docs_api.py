from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    i = text.index(start)
    j = text.index(end, i)
    p.write_text(text[:i] + replacement + text[j:], encoding="utf-8")


# Public top-level export mirrors the existing top-level hac_covariance export.
replace_once(
    "statgpu/__init__.py",
    "    hac_covariance,\n)\nfrom .backends",
    "    hac_covariance,\n    driscoll_kraay_covariance,\n)\nfrom .backends",
)
replace_once(
    "statgpu/__init__.py",
    '    "hac_covariance",\n    # Backends',
    '    "hac_covariance",\n    "driscoll_kraay_covariance",\n    # Backends',
)

# English model page.
en = "docs/en/models/panel.md"
replace_once(en, "> Last updated: 2026-08-08", "> Last updated: 2026-08-09")
replace_once(
    en,
    "Stage B of the Tier-1 panel roadmap adds parameter-based fit statistics and three structured specification tests without changing the Stage-A coefficient, prediction, covariance-normalization, or legacy inference contracts.",
    "Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. Final physical CUDA acceptance for this PR remains a separate evidence gate until the exact-head artifact is recorded.",
)
replace_once(
    en,
    "    hac_covariance,\n)",
    "    hac_covariance,\n    driscoll_kraay_covariance,\n)",
)
replace_between(
    en,
    "## Model Summary\n",
    "## Core Estimating Equations\n",
    """## Model Summary

| Model | Transformation | Main inference choices | Standardized fit statistics |
|---|---|---|---|
| `PanelOLS` | Entity/time within transformation | nonrobust; HC0/HC1/HC2/HC3; one-/two-way clustered; Driscoll-Kraay | within/between/overall R², adjusted R², classical model F, pooling F |
| `RandomEffects` | Swamy-Arora feasible GLS | nonrobust; HC0/HC1/HC2/HC3; one-/two-way clustered; Driscoll-Kraay | within/between/overall R², adjusted R², classical model F, Hausman input when nonrobust |
| `PooledOLS` | Stacked OLS | nonrobust; HC0/HC1/HC2/HC3; clustered; legacy row-HAC; Driscoll-Kraay | overall R² always; within/between R² and BP-LM with `entity_ids`; adjusted R² and classical model F |
| `BetweenOLS` | Entity means | nonrobust; HC0/HC1/HC2/HC3 | within/between/overall R², adjusted R², classical model F |
| `FirstDifferenceOLS` | Within-entity first differences | nonrobust; HC0/HC1/HC2/HC3 | within/between/overall R², adjusted R² on the differenced fit space, classical model F |
| `FamaMacBeth` | Cross-sectional regressions by period | nonrobust, Newey-West | parameter-based within/between/overall R²; no residual-OLS adjusted R² or model F |

""",
)
replace_between(
    en,
    "## Covariance and Existing Inference\n",
    "### PooledOLS HAC ordering\n",
    r'''## Stage-C Covariance and Inference

Stage C is additive: coefficient estimation, Stage-B fit statistics, and the historical default inference remain unchanged. Covariance names are normalized as follows.

| `cov_type` | Behavior |
|---|---|
| `"nonrobust"` | Classical fit-space OLS covariance with Student-t inference |
| `"robust"`, `"hc1"` | The historical statgpu HC1 sandwich with asymptotic normal inference; `hc1` normalizes to canonical `robust` |
| `"hc0"` | Unscaled Eicker-White sandwich on the estimator's actual fit-space regression |
| `"hc2"`, `"hc3"` | Leverage-adjusted sandwich on the estimator's actual fit-space regression |
| `"clustered"` | One- or two-way clustered sandwich; `group_debias=True` opt-in correction where supported |
| `"driscoll-kraay"`, `"dk"`, `"kernel"` | Time-aggregated Driscoll-Kraay covariance with Bartlett, Parzen, or quadratic-spectral kernels |
| `"hac"` | Historical row-order Bartlett/Newey-West covariance for `PooledOLS`; deliberately distinct from Driscoll-Kraay |
| `"newey-west"` | Existing HAC on the `FamaMacBeth` coefficient path; not routed through the residual-OLS Stage-C layer |

### HC0/HC2/HC3 fit-space definition

For the numerical regression actually used by an estimator, let `Z` be the fit-space design, `e` the fit-space residual vector, and `B=(Z'Z)^+`. The leverage is

$$
h_i=z_i^\top Bz_i.
$$

HC0 uses the meat $\sum_i z_i z_i^\top e_i^2$; HC2 divides each squared residual by $1-h_i$; HC3 divides by $(1-h_i)^2$. The implementation computes leverage rowwise and never materializes an `n x n` hat matrix. A numerically unit leverage makes HC2/HC3 undefined and raises rather than being clipped into a valid-looking covariance.

The fit space is model-specific: pooled level design for `PooledOLS`, fixed-effect transformed slopes for `PanelOLS`, quasi-demeaned `X_star` for `RandomEffects`, entity means for `BetweenOLS`, and retained first differences for `FirstDifferenceOLS`. Consequently Panel HC2/HC3 is documented as **transformed-fit-space HC2/HC3**; it is not silently redefined as HC2/HC3 from a literal full dummy regression.

### Cluster covariance and `group_debias`

One-way clustering aggregates score vectors within a cluster. Two-way clustering uses inclusion-exclusion of cluster 1, cluster 2, and the exact paired-label intersection. The default `group_debias=False` preserves the historical statgpu clustered covariance. With `group_debias=True`, each component's meat is multiplied by

$$
\frac{G}{G-1}\frac{n-1}{n},
$$

using that component's own group count before two-way inclusion-exclusion. This changes covariance magnitude only; Stage C does not silently switch coefficient tests to a finite-group t reference. String/categorical cluster labels are metadata and are factorized without moving the numerical score matrix to CPU.

### Driscoll-Kraay covariance

Driscoll-Kraay first aggregates fit-space scores by observed time,

$$
g_t=\sum_i z_{it}e_{it},
$$

then applies a kernel HAC to the ordered `g_t` series. `PooledOLS` uses `time_index=`, while `PanelOLS` and `RandomEffects` use aligned `time_ids=`. Unbalanced panels are supported because each time aggregate contains only observed rows.

For a full-rank fit-space design with `k` columns, Stage C uses the `linearmodels==7.0`-compatible debiased scale

$$
\mathrm{scale}_{DK}=\frac{n}{n-\mathrm{extra\_df}-k}.
$$

`PooledOLS` and `RandomEffects` use `extra_df=0`. `PanelOLS` uses the Stage-B standard fixed-effect nuisance rank (`N`, `T`, or `N+T-C`). If statgpu validly reaches a rank-deficient fit, the documented extension replaces `k` by the numerical rank and uses a pseudoinverse; this corner is not claimed to be a `linearmodels` equality case.

`bandwidth=None` uses `floor(4*(T/100)^(2/9))`, where `T` is the number of distinct observed periods. Bartlett/Newey-West and Parzen/Gallant are truncated at the bandwidth. Quadratic Spectral (`qs`, Andrews) treats bandwidth as a smoothing scale and applies weights to **all observed lags** when bandwidth is positive; it is not truncated at `bw`.

### RandomEffects covariance

Stage C does not alter Swamy-Arora variance-component or coefficient estimation. Robust, HC, cluster, and Driscoll-Kraay covariance are computed from the quasi-demeaned GLS design `X_star` and residuals. Therefore changing `cov_type` changes only inference. The classical Stage-B Hausman test requires **both** the FE and RE fits to use nonrobust covariance; robust auxiliary Hausman remains out of scope and returns a structured inapplicable result.

### Backend and validation status

HC leverage, row scores, grouped cluster/time scores, lag products, bread/meat matrices, and covariance accumulation remain on NumPy/CuPy/Torch. CPU transfers are restricted to labels/group codes, small configuration, and scalar audit reductions. Explicit GPU devices never silently fall back to CPU.

Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. The exact-head CuPy/Torch physical correctness and performance artifacts are a separate acceptance gate and must be recorded before PR #126 is promoted from Draft.

''',
)
replace_once(
    en,
    "- the FE coefficient covariance must be classical/nonrobust;\n- FE and RE must be fitted",
    "- the FE coefficient covariance must be classical/nonrobust;\n- the RE coefficient covariance must also be classical/nonrobust; Stage-C robust/HC/cluster/DK RE fits are not inputs to this classical test;\n- FE and RE must be fitted",
)
replace_once(
    en,
    '''PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",
    device="auto",
)''',
    '''PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",  # robust/hc0/hc1/hc2/hc3/clustered/dk also supported
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)''',
)
replace_once(
    en,
    '''PooledOLS(
    cov_type="nonrobust",
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    device="auto",
)''',
    '''PooledOLS(
    cov_type="nonrobust",  # includes HC, clustered, legacy hac, and dk
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)''',
)
replace_once(
    en,
    '''RandomEffects(device="auto")
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")''',
    '''RandomEffects(
    device="auto",
    cov_type="nonrobust",  # robust/hc0/hc1/hc2/hc3/clustered/dk
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
)
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # HC0/1/2/3 supported
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # HC0/1/2/3 supported''',
)

# Chinese model page.
cn = "docs/cn/models/panel.md"
replace_once(cn, "> 最后更新：2026-08-08", "> 最后更新：2026-08-09")
replace_once(
    cn,
    "Tier-1 Panel 路线的 Stage B 在不改变 Stage-A 系数、预测、协方差归一化和 legacy inference 契约的前提下，新增参数型 fit statistics 和三类结构化 specification test。",
    "Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。当前 PR 的最终 physical CUDA acceptance 仍需由 exact-head 机器产物单独闭合。",
)
replace_once(cn, "    hac_covariance,\n)", "    hac_covariance,\n    driscoll_kraay_covariance,\n)")
replace_between(
    cn,
    "## 模型汇总\n",
    "## 核心估计方程\n",
    """## 模型汇总

| 模型 | 变换 | 主要推断选项 | 标准化 fit statistics |
|---|---|---|---|
| `PanelOLS` | entity/time 组内变换 | nonrobust；HC0/HC1/HC2/HC3；one-/two-way clustered；Driscoll-Kraay | within/between/overall R²、adjusted R²、classical model F、pooling F |
| `RandomEffects` | Swamy-Arora 可行 GLS | nonrobust；HC0/HC1/HC2/HC3；one-/two-way clustered；Driscoll-Kraay | within/between/overall R²、adjusted R²、classical model F；nonrobust 时可作为 Hausman 输入 |
| `PooledOLS` | 堆叠 OLS | nonrobust；HC0/HC1/HC2/HC3；clustered；legacy row-HAC；Driscoll-Kraay | overall R² 始终可用；提供 `entity_ids` 后增加 within/between R² 与 BP-LM；另有 adjusted R² 和 classical model F |
| `BetweenOLS` | entity 均值 | nonrobust；HC0/HC1/HC2/HC3 | within/between/overall R²、adjusted R²、classical model F |
| `FirstDifferenceOLS` | entity 内一阶差分 | nonrobust；HC0/HC1/HC2/HC3 | within/between/overall R²、差分拟合空间 adjusted R²、classical model F |
| `FamaMacBeth` | 逐期横截面回归 | nonrobust、Newey-West | 参数型 within/between/overall R²；不定义 residual-OLS adjusted R² 或 model F |

""",
)
replace_between(
    cn,
    "## 协方差与现有推断\n",
    "### PooledOLS HAC 时间排序\n",
    r'''## Stage-C 协方差与推断

Stage C 是增量式扩展：系数估计、Stage-B fit statistics 与历史默认推断不改变。协方差名称规范如下。

| `cov_type` | 行为 |
|---|---|
| `"nonrobust"` | 经典拟合空间 OLS covariance 与 Student-t 推断 |
| `"robust"`、`"hc1"` | 历史 statgpu HC1 sandwich 与渐近正态推断；`hc1` 规范化为 canonical `robust` |
| `"hc0"` | estimator 实际拟合空间上的未缩放 Eicker-White sandwich |
| `"hc2"`、`"hc3"` | estimator 实际拟合空间上的 leverage-adjusted sandwich |
| `"clustered"` | one-/two-way cluster sandwich；支持时可用 `group_debias=True` 显式修正 |
| `"driscoll-kraay"`、`"dk"`、`"kernel"` | 按 time 聚合 score 后计算 Driscoll-Kraay，可选 Bartlett、Parzen、Quadratic Spectral kernel |
| `"hac"` | `PooledOLS` 历史 row-order Bartlett/Newey-West；与 Driscoll-Kraay 明确区分 |
| `"newey-west"` | `FamaMacBeth` 系数路径已有 HAC；不进入 Stage-C residual-OLS covariance 层 |

### HC0/HC2/HC3 的拟合空间定义

设 estimator 实际数值回归使用的 fit-space design 为 `Z`，残差为 `e`，`B=(Z'Z)^+`。leverage 为

$$
h_i=z_i^\top Bz_i.
$$

HC0 的 meat 为 $\sum_i z_i z_i^\top e_i^2$；HC2 将每个残差平方除以 $1-h_i$；HC3 除以 $(1-h_i)^2$。实现逐行计算 leverage，不会构造 `n x n` hat matrix。当 leverage 在数值上等于 1 时，HC2/HC3 本身未定义，因此显式报错而不是裁剪成看似有效的 covariance。

不同模型使用不同 fit space：`PooledOLS` 为含 fitted intercept 的 level design；`PanelOLS` 为 fixed-effect transformed slope design；`RandomEffects` 为 quasi-demeaned `X_star`；`BetweenOLS` 为 entity-mean design；`FirstDifferenceOLS` 为保留的一阶差分 design。因此 Panel HC2/HC3 明确称为 **transformed-fit-space HC2/HC3**，不会暗中改成完整 dummy regression 的 HC2/HC3。

### Cluster covariance 与 `group_debias`

one-way clustering 在 cluster 内汇总 score vector。two-way clustering 对 cluster 1、cluster 2 与精确 paired-label intersection 做 inclusion-exclusion。默认 `group_debias=False` 完全保留历史 statgpu clustered covariance。设某一 component 有 `G` 个 group，则 `group_debias=True` 会在 inclusion-exclusion 前将该 component 的 meat 乘以

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

这只改变 covariance 大小；Stage C 不会静默把 coefficient test 改成 finite-group t reference。字符串/分类 cluster label 属于 metadata，可在 CPU factorize，但数值 score matrix 不会搬回 CPU。

### Driscoll-Kraay covariance

Driscoll-Kraay 首先按 observed time 聚合 fit-space score：

$$
g_t=\sum_i z_{it}e_{it},
$$

再对有序 `g_t` 序列计算 kernel HAC。`PooledOLS` 使用 `time_index=`；`PanelOLS` 与 `RandomEffects` 使用对齐后的 `time_ids=`。unbalanced panel 直接按每个 time 实际存在的观测聚合。

对 full-rank fit-space design（列数 `k`），Stage C 使用与 `linearmodels==7.0` 对齐的 debiased scale：

$$
\mathrm{scale}_{DK}=\frac{n}{n-\mathrm{extra\_df}-k}.
$$

`PooledOLS` 与 `RandomEffects` 的 `extra_df=0`；`PanelOLS` 使用 Stage-B standard fixed-effect nuisance rank（`N`、`T` 或 `N+T-C`）。若 statgpu 合法到达 rank-deficient fit，则记录为扩展：用 numerical rank 替代 `k` 并配合 pseudoinverse；该 corner 不宣称与 `linearmodels` 完全相等。

`bandwidth=None` 使用 `floor(4*(T/100)^(2/9))`，其中 `T` 是 observed distinct time period 数。Bartlett/Newey-West 与 Parzen/Gallant 在 bandwidth 处截断。Quadratic Spectral（`qs`、Andrews）把 bandwidth 当作平滑尺度；bandwidth 为正时对**所有 observed lag**赋权，而不是在 `bw` 截断。

### RandomEffects covariance

Stage C 不改变 Swamy-Arora variance component 或 coefficient estimate。robust、HC、cluster 与 Driscoll-Kraay 都基于 quasi-demeaned GLS design `X_star` 与相应 residual 计算，因此改变 `cov_type` 只改变 inference。Stage-B classical Hausman 要求 **FE 与 RE 两端都使用 nonrobust covariance**；robust auxiliary Hausman 不在 Stage C 范围内，并返回结构化 inapplicable 结果。

### Backend 与验证状态

HC leverage、row score、cluster/time grouped score、lag product、bread/meat/covariance 都保留在 NumPy/CuPy/Torch 数值后端。CPU transfer 只允许 label/group code、小型配置和 scalar audit reduction。显式 GPU device 不静默回退 CPU。

hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。exact-head CuPy/Torch physical correctness 与 performance artifact 是独立 acceptance gate；PR #126 在这些产物闭合前继续保持 Draft。

''',
)
replace_once(
    cn,
    "- FE coefficient covariance 必须是 classical/nonrobust；\n- FE 与 RE 必须来自",
    "- FE coefficient covariance 必须是 classical/nonrobust；\n- RE coefficient covariance 同样必须是 classical/nonrobust；Stage-C robust/HC/cluster/DK RE fit 不属于该 classical test 的输入；\n- FE 与 RE 必须来自",
)
replace_once(
    cn,
    '''PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",
    device="auto",
)''',
    '''PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",  # 同时支持 robust/hc0/hc1/hc2/hc3/clustered/dk
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)''',
)
replace_once(
    cn,
    '''PooledOLS(
    cov_type="nonrobust",
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    device="auto",
)''',
    '''PooledOLS(
    cov_type="nonrobust",  # 包含 HC、clustered、legacy hac 与 dk
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)''',
)
replace_once(
    cn,
    '''RandomEffects(device="auto")
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")''',
    '''RandomEffects(
    device="auto",
    cov_type="nonrobust",  # robust/hc0/hc1/hc2/hc3/clustered/dk
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
)
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # 支持 HC0/1/2/3
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # 支持 HC0/1/2/3''',
)

# Changelogs: describe implementation and hosted definitions, but keep physical gate pending.
root = "CHANGELOG.md"
root_block = """## 2026-08-09

### PR #126 — Complete Panel Tier-1 Stage C covariance
- Added HC0/HC2/HC3, robust RandomEffects inference, cluster group debiasing, and Driscoll-Kraay covariance with NumPy/CuPy/Torch-native accumulation.
- Preserved historical HC1 (`robust`), Pooled row-HAC, default clustered covariance, coefficient estimates, and Stage-B diagnostics.
- Added pinned statsmodels/linearmodels covariance checks plus exact-head physical GPU and performance validators; final P100 acceptance remains pending.

"""
replace_once(root, "## 2026-08-08\n", root_block + "## 2026-08-08\n")

en_changelog = "docs/en/changelog.md"
en_block = """## 2026-08-09 — Panel Stage C covariance completion (PR #126)

Stage C extends the Panel Tier-1 inference layer without changing estimator coefficients or the Stage-B diagnostic definitions. `robust` remains the historical HC1 contract; new `hc0`, `hc2`, and `hc3` use each estimator's actual transformed fit space. `RandomEffects` now supports robust/HC, clustered, and Driscoll-Kraay covariance on the quasi-demeaned GLS scores. One-/two-way clustering gains opt-in `group_debias=True`; the default clustered result is unchanged. `PooledOLS(cov_type=\"hac\")` remains the legacy row-order Bartlett/Newey-West path and is not reinterpreted as Driscoll-Kraay.

Driscoll-Kraay follows the pinned `linearmodels==7.0` full-rank scaling and time-score aggregation, with Bartlett, Parzen, and quadratic-spectral kernels. QS uses all observed lags when bandwidth is positive. HC2/HC3 are checked against analytic/statsmodels fit-space calculations, and cluster/DK definitions are checked against `linearmodels==7.0`. A separate maintained Torch 2.0 CPU gate covers Stage-C covariance primitives and estimator integrations.

The physical CUDA gate is intentionally separate: `dev/benchmarks/validate_panel_stage_c_gpu.py` and `dev/benchmarks/benchmark_panel_stage_c_covariance.py` must be run on the final exact clean implementation head before PR #126 can leave Draft. No GPU speedup or final physical-acceptance claim is made here yet.

"""
replace_once(en_changelog, "## 2026-08-08", en_block + "## 2026-08-08")

cn_changelog = "docs/cn/changelog.md"
cn_block = """## 2026-08-09 — Panel Stage C 协方差补齐（PR #126）

Stage C 在不改变 estimator coefficient 与 Stage-B diagnostic definition 的前提下扩展 Panel Tier-1 inference。`robust` 继续表示历史 HC1；新增 `hc0`、`hc2`、`hc3` 按各 estimator 的实际 transformed fit space 计算。`RandomEffects` 在 quasi-demeaned GLS score 上新增 robust/HC、clustered 与 Driscoll-Kraay covariance。one-/two-way cluster 新增显式 `group_debias=True`；默认 clustered result 保持不变。`PooledOLS(cov_type=\"hac\")` 仍是 legacy row-order Bartlett/Newey-West，不会被重解释为 Driscoll-Kraay。

Driscoll-Kraay 的 full-rank scale 与 time-score aggregation 固定对齐 `linearmodels==7.0`，并支持 Bartlett、Parzen 与 Quadratic Spectral；QS 在 bandwidth 为正时对所有 observed lag 赋权。HC2/HC3 通过 analytic/statsmodels fit-space 结果检查，cluster/DK definition 通过 `linearmodels==7.0` 检查；另有 maintained Torch 2.0 CPU gate 覆盖 Stage-C covariance primitive 与 estimator integration。

physical CUDA gate 与 hosted definition gate 明确分离：`dev/benchmarks/validate_panel_stage_c_gpu.py` 与 `dev/benchmarks/benchmark_panel_stage_c_covariance.py` 仍需在最终 exact clean implementation head 上执行。当前不宣称 GPU speedup，也不宣称最终 physical acceptance 已完成。

"""
replace_once(cn_changelog, "## 2026-08-08", cn_block + "## 2026-08-08")
