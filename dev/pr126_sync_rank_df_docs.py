from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: replacement count={count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


# Plan: separate historical full-rank preservation from the rank-deficient extension.
plan = "dev/plans/panel_p1_stage_c_covariance_plan.md"
replace_once(
    plan,
    "Stage C completes the covariance/inference work promised by Issue #93 while preserving all Stage-A/Stage-B coefficient estimates, fitted values, predictions, default covariance behavior, diagnostic definitions, and strict-device semantics.",
    "Stage C completes the covariance/inference work promised by Issue #93 while preserving all historical full-column-rank Stage-A/Stage-B coefficient estimates, fitted values, predictions, default covariance behavior, diagnostic definitions, and strict-device semantics. The supported rank-deficient extension is additionally required to depend on the identified numerical fit space rather than the raw number of redundant columns.",
)
replace_once(
    plan,
    "8. Coefficients, fitted values, variance components, theta, R², specification tests, formula parsing, and prediction semantics do not change solely because Stage C exists.",
    "8. On historical full-column-rank fits, coefficients, fitted values, variance components, theta, R², specification tests, formula parsing, and prediction semantics do not change solely because Stage C exists.\n9. On supported rank-deficient fits, residual and auxiliary-regression degrees of freedom use the identified numerical rank. Adding an exactly redundant column must not change identified fitted values, Swamy-Arora variance components/theta, fit-space covariance, or inference merely by increasing the raw column count.",
)
replace_once(
    plan,
    "- preserve each estimator's historical full-column-rank coefficient solver; enter the shared SVD minimum-norm solve only when the explicit rank policy reports `rank < k` or the historical solver raises a linear-algebra failure, and reuse that fit-space rank in covariance/diagnostics;",
    "- preserve each estimator's historical full-column-rank coefficient solver; enter the shared SVD minimum-norm solve only when the explicit rank policy reports `rank < k` or the historical solver raises a linear-algebra failure, and reuse that fit-space rank in covariance/diagnostics and in rank-deficient residual/auxiliary degrees of freedom while leaving full-rank formulas unchanged;\n- strict inference rejects every strictly negative final covariance diagonal; IEEE signed zero may be normalized to zero, but no absolute or row/column-scale tolerance may turn a genuinely negative variance into a valid-looking standard error because such tolerances are not invariant to outcome/regressor units;",
)
replace_once(
    plan,
    "Swamy–Arora estimation is untouched. All Stage-C covariance uses the already-computed quasi-demeaned regression:",
    "Historical full-rank Swamy–Arora estimation is untouched. For the supported rank-deficient extension, the between, within, and quasi-demeaned residual degrees of freedom use the corresponding identified numerical ranks so redundant columns cannot change variance components or theta. All Stage-C covariance uses the already-computed quasi-demeaned regression:",
)

# English model page.
en = "docs/en/models/panel.md"
replace_once(en, "> Last updated: 2026-08-11", "> Last updated: 2026-08-12")
replace_once(
    en,
    "The repaired numerical-rank and FirstDifference chronology implementation is physically accepted on exact-clean Tesla P100 measurement `3dc7df19...`: CuPy 13.6.0 and Torch each pass 27 estimator integrations + 12 direct public primitives (**39/39 per backend**), including the numerical-rank boundary, and the paired synchronized performance artifact contains all 58 maintained rows. The implementation uses one backend-native SVD mask with cutoff `max(n,k) * eps64 * s_max` for pseudoinverse/rank decisions and preserves ordered-categorical FirstDifference chronology. The earlier `ec511f53...` 32/32 source remains immutable historical evidence and is not overwritten.",
    "The previous exact-clean Tesla P100 measurement `3dc7df19...` (CuPy/Torch **39/39 per backend**, plus 58 synchronized performance rows) remains immutable historical evidence, but it is no longer current acceptance evidence after the 2026-08-12 rank-deficient-df and strict covariance-validity fixes. The current implementation still uses one backend-native SVD mask with cutoff `max(n,k) * eps64 * s_max`, now also uses identified rank for supported rank-deficient residual/auxiliary degrees of freedom, and rejects every strictly negative final covariance diagonal. Fresh physical acceptance is pending on the current exact source with an expanded target of **35 estimator integrations + 12 public primitives = 47/47 per backend**, plus the unchanged 58-row performance matrix.",
)
replace_once(
    en,
    "Stage C is additive: coefficient estimation, Stage-B fit statistics, and the historical default inference remain unchanged. Covariance names are normalized as follows.",
    "Stage C is additive on historical full-column-rank fits: coefficient estimation, Stage-B fit statistics, and the historical default inference remain unchanged. The supported rank-deficient extension uses identified numerical rank so adding redundant columns does not change fit-space degrees of freedom or identified inference. Covariance names are normalized as follows.",
)
replace_once(
    en,
    "Stage C does not alter Swamy-Arora variance-component or coefficient estimation. Robust, HC, cluster, and Driscoll-Kraay covariance are computed from the quasi-demeaned GLS design `X_star` and residuals. Therefore changing `cov_type` changes only inference. The classical Stage-B Hausman test requires **both** the FE and RE fits to use nonrobust covariance; robust auxiliary Hausman remains out of scope and returns a structured inapplicable result.",
    "Historical full-rank Swamy-Arora variance-component and coefficient estimation is unchanged. In the supported rank-deficient extension, the between/within/quasi-demeaned residual degrees of freedom use their identified numerical ranks, making variance components, theta, identified fitted values, and fit-space inference invariant to exactly redundant columns. Robust, HC, cluster, and Driscoll-Kraay covariance are computed from `X_star` and its residuals, so changing `cov_type` still changes inference only. The classical Stage-B Hausman test requires **both** the FE and RE fits to use nonrobust covariance; robust auxiliary Hausman remains out of scope and returns a structured inapplicable result.",
)
replace_once(
    en,
    "Current `remote-full` acceptance is the exact-clean `3dc7df19...` Tesla P100 run: CuPy and Torch each pass 27 estimator integrations + 12 public primitives (**39/39 per backend**) and the synchronized performance matrix contains 58/58 rows. The six new rank-boundary primitives and `panel_rank_boundary_dk` integration pass on both GPU backends with requested/executed backend identity. The older `ec511f53...` source is historical only. Explicit GPU devices continue to forbid silent CPU fallback.",
    "Current validation status is **PARTIAL_REMOTE_PENDING**. The `3dc7df19...` Tesla P100 run remains historical evidence only because production numerical behavior changed afterward. Fresh correctness acceptance must run the expanded matrix at the current exact-clean source: CuPy and Torch each require **47/47 = 35 estimator integrations + 12 public primitives**, including eight rank-deficient nonrobust/HC1 estimator cases that record `fit_rank < parameter_count`; the synchronized performance target remains 58/58 rows. Explicit GPU devices continue to forbid silent CPU fallback.",
)

# Chinese model page.
cn = "docs/cn/models/panel.md"
replace_once(cn, "> 最后更新：2026-08-11", "> 最后更新：2026-08-12")
replace_once(
    cn,
    "修复后的 numerical-rank 与 FirstDifference chronology 实现已在 exact-clean `3dc7df19...` Tesla P100 上完成 physical acceptance：CuPy 13.6.0 与 Torch 各通过 27 个 estimator integration + 12 个 direct public primitive（**每个 backend 39/39**），包括 numerical-rank boundary；配套同步 performance artifact 也包含全部 58 个 maintained row。实现使用同一个 backend-native SVD mask、cutoff `max(n,k) * eps64 * s_max` 统一 pseudoinverse/rank 决策，并保留 ordered-categorical FirstDifference chronology。旧 `ec511f53...` 32/32 source 继续作为不可变历史证据保留，不会被覆盖。",
    "此前 exact-clean `3dc7df19...` Tesla P100 结果（CuPy/Torch **每个 backend 39/39**，以及 58 行同步 performance）继续作为不可变历史证据保留，但在 2026-08-12 的 rank-deficient df 与 strict covariance-validity 修复后已不再是当前 acceptance evidence。当前实现继续使用同一个 backend-native SVD mask 与 cutoff `max(n,k) * eps64 * s_max`，并进一步让受支持的秩亏 residual/auxiliary df 使用 identified rank，同时对任何严格为负的最终 covariance diagonal fail closed。新的 physical acceptance 尚待在当前 exact source 上执行扩展矩阵：**35 个 estimator integration + 12 个 public primitive = 每个 backend 47/47**，performance 目标仍为 58 行。",
)
replace_once(
    cn,
    "Stage C 是增量式扩展：系数估计、Stage-B fit statistics 与历史默认推断不改变。协方差名称规范如下。",
    "对于历史 full-column-rank 拟合，Stage C 仍是增量式扩展：系数估计、Stage-B fit statistics 与历史默认推断不改变。对受支持的 rank-deficient extension，则使用 identified numerical rank，使冗余列不会改变 fit-space df 或已识别推断。协方差名称规范如下。",
)
replace_once(
    cn,
    "Stage C 不改变 Swamy-Arora variance component 或 coefficient estimate。robust、HC、cluster 与 Driscoll-Kraay 都基于 quasi-demeaned GLS design `X_star` 与相应 residual 计算，因此改变 `cov_type` 只改变 inference。Stage-B classical Hausman 要求 **FE 与 RE 两端都使用 nonrobust covariance**；robust auxiliary Hausman 不在 Stage C 范围内，并返回结构化 inapplicable 结果。",
    "历史 full-rank Swamy-Arora variance component 与 coefficient estimate 保持不变。对受支持的 rank-deficient extension，between/within/quasi-demeaned residual df 使用各自 identified numerical rank，从而使 variance component、theta、identified fitted value 与 fit-space inference 对精确冗余列保持不变。robust、HC、cluster 与 Driscoll-Kraay 仍基于 `X_star` 与相应 residual 计算，因此改变 `cov_type` 只改变 inference。Stage-B classical Hausman 要求 **FE 与 RE 两端都使用 nonrobust covariance**；robust auxiliary Hausman 不在 Stage C 范围内，并返回结构化 inapplicable 结果。",
)
replace_once(
    cn,
    "当前 `remote-full` acceptance 来自 exact-clean `3dc7df19...` Tesla P100：CuPy 与 Torch 各通过 27 个 estimator integration + 12 个 public primitive（**每个 backend 39/39**），同步 performance matrix 为 58/58 行。新增的六个 rank-boundary primitive 与 `panel_rank_boundary_dk` integration 均在两个 GPU backend 上通过，且 requested/executed backend 一致。旧 `ec511f53...` source 仅作为历史证据保留；显式 GPU device 仍禁止静默回退 CPU。",
    "当前验证状态为 **PARTIAL_REMOTE_PENDING**。由于后续 production numerical behavior 已改变，`3dc7df19...` Tesla P100 结果仅保留为历史证据。新的 correctness acceptance 必须在当前 exact-clean source 上执行扩展矩阵：CuPy 与 Torch 各要求 **47/47 = 35 个 estimator integration + 12 个 public primitive**，其中新增 8 个 rank-deficient nonrobust/HC1 estimator case 并记录 `fit_rank < parameter_count`；同步 performance 目标仍为 58/58 行。显式 GPU device 继续禁止静默回退 CPU。",
)

# Changelogs: preserve old evidence but stop calling it current acceptance.
root = "CHANGELOG.md"
replace_once(
    root,
    "- Accepted the repaired numerical-rank/FirstDifference implementation on exact-clean `3dc7df19...` Tesla P100: CuPy 13.6.0 and Torch each pass 27 estimator integrations + 12 public primitives (39/39), including all new rank-boundary cases, and the paired synchronized performance source contains 58/58 rows. The older `ec511f53...` source remains immutable historical evidence and is not overwritten.",
    "- Retained the exact-clean `3dc7df19...` P100 result (CuPy/Torch 39/39 and 58 synchronized performance rows) as immutable historical evidence after a 2026-08-12 review reopened rank-deficient df and covariance-validity issues; the fixes now make supported rank-deficient df depend on identified rank and fail closed on every strictly negative final variance, with fresh 47/47-per-backend physical acceptance pending.",
)

en_ch = "docs/en/changelog.md"
replace_once(en_ch, "> Last updated: 2026-08-11", "> Last updated: 2026-08-12")
replace_once(
    en_ch,
    "The repaired implementation is now physically accepted on exact-clean `3dc7df19...` Tesla P100: CuPy 13.6.0 and Torch each pass 27 estimator integrations + 12 direct public primitives (**39/39 per backend**), including the numerical-rank boundary, and the paired synchronized performance source contains all 58 maintained rows. The older `ec511f53...` run remains immutable historical evidence and is not overwritten.",
    "The exact-clean `3dc7df19...` Tesla P100 result (CuPy/Torch **39/39 per backend**, plus all 58 synchronized performance rows) remains immutable historical evidence. A 2026-08-12 strict re-review subsequently reopened the rank-deficient residual/Swamy-Arora df contract and a unit-dependent negative-variance guard. The local fixes now use identified rank for supported rank-deficient df while preserving historical full-rank formulas, and strict inference rejects every truly negative final variance. Because production numerical behavior changed, fresh physical acceptance is pending on the expanded **47/47 per backend** correctness matrix; the performance target remains 58 rows.",
)
cn_ch = "docs/cn/changelog.md"
replace_once(cn_ch, "> 最后更新：2026-08-11", "> 最后更新：2026-08-12")
replace_once(
    cn_ch,
    "修复后的实现现已在 exact-clean `3dc7df19...` Tesla P100 上完成 physical acceptance：CuPy 13.6.0 与 Torch 各通过 27 个 estimator integration + 12 个 direct public primitive（**每个 backend 39/39**），包括 numerical-rank boundary，配套同步 performance source 也包含全部 58 个 maintained row。旧 `ec511f53...` 运行继续作为不可变历史证据保留，不会被覆盖。",
    "exact-clean `3dc7df19...` Tesla P100 结果（CuPy/Torch **每个 backend 39/39**，以及全部 58 行同步 performance）继续作为不可变历史证据保留。2026-08-12 的 strict re-review 随后重新发现 rank-deficient residual/Swamy-Arora df 契约以及依赖单位的 negative-variance guard 问题；本地修复现在让受支持的秩亏 df 使用 identified rank，同时保持历史 full-rank 公式不变，并对任何真实负的最终 variance fail closed。由于 production numerical behavior 已改变，新的 physical acceptance 尚待执行扩展后的 **每个 backend 47/47** correctness matrix；performance 目标仍为 58 行。",
)

# Replace the authoritative current physical status record; historical identities stay explicit.
Path("dev/reviews/pr126_physical_gpu_validation.md").write_text("""# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PARTIAL_REMOTE_PENDING / LOCAL FIX VALIDATED / NOT MERGE-READY**

Validation tier: local gates complete; fresh remote CUDA correctness/performance pending.

A 2026-08-12 strict re-review found that the previously accepted implementation did not use identified rank consistently for rank-deficient residual/Swamy-Arora degrees of freedom and used a unit-dependent negative-variance tolerance. Production behavior has therefore changed after the previously accepted numerical measurement `3dc7df19176f8fb881a8d37e9d75b4f75e71b058`; that P100 evidence and its canonical v2 sources remain immutable **historical** evidence, not current acceptance evidence.

## Current local fix contract

- historical full-column-rank coefficient solvers and df formulas remain algebraically unchanged;
- supported rank-deficient `PanelOLS`, `BetweenOLS`, `FirstDifferenceOLS`, and `RandomEffects` use identified fit-space rank for residual degrees of freedom;
- `RandomEffects` between/within auxiliary df are rank-aware, so exact redundant columns do not change `sigma2_e`, `sigma2_a`, theta, identified fitted values, or fit-space inference;
- strict inference rejects every strictly negative final covariance diagonal; only IEEE signed zero may be normalized to zero;
- maintained column-space invariance tests cover nonrobust and HC1/`robust`, including unbalanced RandomEffects and entity-effect PanelOLS;
- the physical correctness runner now records each estimator case's `fit_rank` and `parameter_count`.

Focused validation completed before commit:

- rank/inference/covariance/external review matrix: **86 passed**;
- strict-variance/entity-FE re-review matrix: passed;
- expanded physical-runner/Torch contract matrix: **45 passed**;
- syntax and `git diff --check`: passed.

## Fresh remote acceptance target

Fresh evidence must be measured on an exact-clean post-fix numerical head.

Correctness/backend provenance target:

- CuPy: **47/47** = 35 estimator integrations + 12 direct public primitives;
- Torch: **47/47** = 35 estimator integrations + 12 direct public primitives;
- eight dedicated rank-deficient estimator cases cover PanelOLS(entity effects), BetweenOLS, FirstDifferenceOLS, and RandomEffects under both `nonrobust` and historical HC1/`robust`;
- each of those eight cases must record `fit_rank < parameter_count`;
- requested backend must equal executed backend for every estimator/primitive;
- all prior numerical-rank-boundary primitives and `panel_rank_boundary_dk` remain required;
- no numerical CPU fallback is accepted.

Synchronized performance target:

- **58/58** maintained rows = 54 base + 4 high-T QS;
- synchronized end-to-end estimator fit;
- CuPy/Torch × PooledOLS/PanelOLS QS at `N=10,000,k=2,T=200` remains included;
- raw samples must be finite/positive and stored medians must match raw samples;
- the updated performance runner records CUDA-suffixed CuPy package distributions (`cupy-cuda11x`/`cupy-cuda12x`) when the unsuffixed distribution is absent;
- no speedup or CPU-baseline claim is made.

## Historical immutable evidence

The following remain registered exactly as historical audit evidence and must not be overwritten:

- measurement `3dc7df19...`, raw commit `be679c13...`: CuPy/Torch 39/39 and 58 performance rows;
- canonical v2 correctness source `panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f`;
- canonical v2 performance source `panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55`;
- earlier `ec511f53...` canonical v1 correctness/performance sources.

Any fresh post-fix artifacts require new immutable source/parser identities rather than mutating v1/v2 registrations.

## Exit rule

Current technical status remains **PARTIAL_REMOTE_PENDING / NOT MERGE-READY** until fresh exact-head P100 correctness and performance evidence is audited, promoted under new immutable identities, the promoted head passes the permanent hosted matrix, and a final strict re-review finds no unresolved CRITICAL/HIGH/relevant MEDIUM issue. PR #126 remains Draft and unmerged unless the user explicitly requests a lifecycle transition.
""", encoding="utf-8")

# Mark the old terminal checkpoint as historical rather than silently rewriting its facts.
post = "dev/reviews/pr126_post_promotion_review_2026-08-11.md"
p = Path(post)
text = p.read_text(encoding="utf-8")
notice = """> **Historical checkpoint, superseded on 2026-08-12.** A later independent strict review reopened rank-deficient df and covariance-validity issues. The `3dc7df19...` evidence and v2 promotion below remain immutable historical facts, but they are no longer current acceptance evidence. See `pr126_rank_df_inference_autofix_2026-08-12.md` and `pr126_physical_gpu_validation.md` for current status.\n\n"""
if "Historical checkpoint, superseded on 2026-08-12" not in text:
    text = text.replace("Standard: `.claude/skills/code-review.md` (`auto-fix` mode)\n\n", "Standard: `.claude/skills/code-review.md` (`auto-fix` mode)\n\n" + notice, 1)
p.write_text(text, encoding="utf-8")

# Current review/fix record.
Path("dev/reviews/pr126_rank_df_inference_autofix_2026-08-12.md").write_text("""# PR #126 rank-deficient df / inference auto-fix review — 2026-08-12

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Current verdict

**PARTIAL_REMOTE_PENDING / LOCAL REVIEW IN PROGRESS / NOT MERGE-READY**

Active axes: public API/presentation, inference, NumPy/CuPy/Torch backend behavior, formula/panel metadata, benchmark/performance, and docs/artifacts. Loss, penalty, generic solver framework, and CV remain inactive.

## Findings reopened by independent review

### [CRITICAL][BUG][fixed] Rank-deficient residual and Swamy-Arora df used raw column count

The supported minimum-norm rank-deficient paths already used explicit numerical rank for coefficient solves, but RandomEffects auxiliary/final df and adjacent Between/FirstDifference/Panel legacy-df paths still counted redundant columns. Equivalent identified column spaces could therefore produce different variance components, theta, scales, HC1 corrections, or inference.

Fix:

- historical full-rank formulas are preserved exactly;
- rank-deficient PanelOLS, BetweenOLS, FirstDifferenceOLS, and RandomEffects residual df use identified fit-space rank;
- RandomEffects within/between auxiliary df use `rank_within`/`rank_between` with the original nuisance-effect parameterization;
- column-space invariance tests compare `[x]` with `[x,2x]` under nonrobust and HC1/robust, including unbalanced RE and entity-effect PanelOLS.

### [HIGH][INFER][fixed] Negative-variance validity depended on measurement units

The prior `4096*eps*max(1, |V_jj|)` guard contained an absolute unit-dependent floor. A row/column covariance scale was considered during the first fix but rejected on re-review because off-diagonal covariance entries transform differently under regressor reparameterization.

Final fix: strict inference rejects every strictly negative final covariance diagonal. IEEE `-0.0` compares equal to zero and may be normalized to zero. Tests cover extreme outcome scaling, large off-diagonal entries, and signed zero.

### [HIGH][MATRIX][fixed locally / needs remote GPU] Physical matrix did not exercise the repaired df branches

The physical runner now adds eight exact-collinearity estimator integrations: PanelOLS(entity FE), BetweenOLS, FirstDifferenceOLS, and RandomEffects under both nonrobust and historical HC1/robust. Each case persists `fit_rank` and `parameter_count`; fresh GPU audit must verify `fit_rank < parameter_count` and requested/executed backend identity.

Fresh target: **47/47 per backend = 35 estimator integrations + 12 public primitives**. Performance remains 58 rows. The timing runner now discovers CUDA-suffixed CuPy distribution versions.

## Validation completed so far

- first focused source/external matrix: **86 passed**;
- strict variance/entity-FE re-review matrix: success;
- expanded runner + maintained Torch contract matrix: **45 passed**;
- syntax / compile checks: success;
- `git diff --check`: success.

## Remaining gate

After docs/status synchronization, run the seven permanent hosted workflows on a connector-authored local-review checkpoint and perform another independent strict review. If no local blocker remains, exit as `PARTIAL_REMOTE_PENDING` and collect fresh exact-head P100 correctness/performance artifacts. Old `3dc7df19...` and all v1/v2 canonical identities remain immutable historical evidence.
""", encoding="utf-8")
