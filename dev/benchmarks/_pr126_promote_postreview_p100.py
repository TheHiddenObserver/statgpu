from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_SHA = "c151550ab17bd9533a51599f86b6a4ea12a292e9"
OLD_MEASUREMENT_SHA = "9c0b3050dd143c43a06bb6393d69f4f83e861637"
OLD_VALIDATION_ID = "panel-stage-c-validation-pr126-20260810-a0d258f6d6b8"
OLD_PERFORMANCE_ID = "panel-stage-c-performance-pr126-20260810-214284f02a5e"
CORRECTNESS = ROOT / "results/pr126_p100/panel_stage_c_gpu_validation_c151550a.json"
PERFORMANCE = ROOT / "results/pr126_p100/panel_stage_c_performance_c151550a.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected anchor missing in {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


corr_sha = sha256(CORRECTNESS)
perf_sha = sha256(PERFORMANCE)
corr_blob = git_output("hash-object", str(CORRECTNESS.relative_to(ROOT)))
perf_blob = git_output("hash-object", str(PERFORMANCE.relative_to(ROOT)))
artifact_commit = git_output("log", "-1", "--format=%H", "--", str(CORRECTNESS.relative_to(ROOT)))
if artifact_commit != git_output("log", "-1", "--format=%H", "--", str(PERFORMANCE.relative_to(ROOT))):
    raise SystemExit("correctness/performance artifacts were not introduced by the same repository commit")

corr_id = f"panel-stage-c-validation-pr126-20260810-{corr_sha[:12]}"
perf_id = f"panel-stage-c-performance-pr126-20260810-{perf_sha[:12]}"

for path in (CORRECTNESS, PERFORMANCE):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("git_sha") != MEASUREMENT_SHA:
        raise SystemExit(f"{path.name}: measurement SHA mismatch")
    if data.get("working_tree_clean") is not True:
        raise SystemExit(f"{path.name}: physical tree was not clean")

# Pin canonical parser to the post-review exact measurement head.
parser = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c.py"
replace_once(parser, f'_MEASUREMENT_SHA = "{OLD_MEASUREMENT_SHA}"', f'_MEASUREMENT_SHA = "{MEASUREMENT_SHA}"')

# Point parser contracts at the new current evidence while retaining old raw files as historical audit evidence.
test_source = ROOT / "dev/tests/test_panel_stage_c_frontend_source.py"
replace_once(test_source, 'panel_stage_c_gpu_validation_9c0b3050.json', 'panel_stage_c_gpu_validation_c151550a.json')
replace_once(test_source, 'panel_stage_c_performance_9c0b3050.json', 'panel_stage_c_performance_c151550a.json')

# Replace the two current canonical manifest entries in place; source count remains stable.
manifest_path = ROOT / "dev/benchmarks/frontend_sources.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for source in manifest["sources"]:
    if source.get("source_id") == OLD_VALIDATION_ID:
        source.update({
            "source_id": corr_id,
            "path": "results/pr126_p100/panel_stage_c_gpu_validation_c151550a.json",
            "sha256": corr_sha,
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                f"Current PR #126 Stage-C post-review exact-clean-head P100 correctness/backend-provenance evidence. "
                f"Artifact commit {artifact_commit}; Git blob {corr_blob}; 26 estimator cases plus two direct public covariance primitives "
                "per CuPy/Torch backend, with requested/executed backend identity and no CPU fallback. The earlier 9c0b3050 measurement "
                "remains immutable historical evidence after the ordered-categorical chronology production fix."
            ),
        })
    elif source.get("source_id") == OLD_PERFORMANCE_ID:
        source.update({
            "source_id": perf_id,
            "path": "results/pr126_p100/panel_stage_c_performance_c151550a.json",
            "sha256": perf_sha,
            "measurement_git_sha": MEASUREMENT_SHA,
            "raw_git_sha": MEASUREMENT_SHA,
            "provenance_note": (
                f"Current PR #126 Stage-C post-review synchronized end-to-end P100 timing evidence. Artifact commit {artifact_commit}; "
                f"Git blob {perf_blob}; includes the bounded N=10000, k=2, T=200 QS all-lag scenario. No speedup claim or CPU baseline "
                "is encoded. The earlier 9c0b3050 timing artifact remains immutable historical evidence."
            ),
        })
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

# Coverage must reference only the current accepted Stage-C sources.
coverage_path = ROOT / "dev/benchmarks/benchmark_coverage_matrix.json"
coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
panel = next(item for item in coverage["capabilities"] if item["capability_id"] == "panel-estimation")
panel["source_ids"] = [
    corr_id if item == OLD_VALIDATION_ID else perf_id if item == OLD_PERFORMANCE_ID else item
    for item in panel["source_ids"]
]
panel["disposition"] = (
    "June timing rows cover aligned PanelOLS and RandomEffects; PR #122 provides canonical Stage-B diagnostic/physical validation. "
    "PR #126 current post-review P100 evidence at c151550a adds canonical Stage-C CuPy/Torch correctness for 26 estimator covariance "
    "integrations plus two direct public primitives per backend, and synchronized timing rows for three base scales plus a bounded "
    "N=10,000, k=2, T=200 QS all-lag scenario. The pre-review 9c0b3050 artifacts are retained only as historical evidence. "
    "No Stage-C speedup claim is made."
)
coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

# Rewrite the physical audit record so current vs historical status is unambiguous.
review = ROOT / "dev/reviews/pr126_physical_gpu_validation.md"
review.write_text(f'''# PR #126 Panel Stage C physical GPU validation

## Physical acceptance status

**PHYSICAL_GPU_ACCEPTED** for the Stage-C correctness and performance runners measured at post-review exact clean implementation head `{MEASUREMENT_SHA}`.

The immutable rerun artifacts were added by repository commit `{artifact_commit}`. Comparing `{MEASUREMENT_SHA}` to `{artifact_commit}` changes only the two new raw JSON evidence files; no numerical implementation, physical validator, performance runner, test, or documentation file changed before measurement was recorded. Therefore the rerun evidence is applicable under `RELEASING.md`.

The earlier exact-clean measurement `{OLD_MEASUREMENT_SHA}` and artifact commit `85d710bddf633134624501a9e27f03c30bc04ead` remain immutable **historical** evidence. They were superseded for current acceptance because the Ready-triggered ordered-categorical Driscoll-Kraay chronology fix changed `statgpu/panel/_covariance.py` and `statgpu/panel/_formula.py` after that run.

## Current correctness/backend-provenance artifact

- path: `results/pr126_p100/panel_stage_c_gpu_validation_c151550a.json`
- measurement SHA: `{MEASUREMENT_SHA}`
- artifact repository commit: `{artifact_commit}`
- Git blob: `{corr_blob}`
- SHA-256: `{corr_sha}`
- schema version: 1
- working tree clean: true
- top-level status: success
- GPU: Tesla P100-SXM2-16GB
- Python: 3.9.16
- NumPy: 1.24.2
- SciPy: 1.10.1
- Torch: 2.0.0

For both CuPy and Torch, all 26 estimator cases passed with requested/executed backend identity. Each backend also passed the two direct public primitive calls (`cluster_group_debias` and `driscoll_kraay_qs`) with `xp` omitted, physically validating public backend auto-detection rather than only estimator-mediated routing.

Direct primitive maximum absolute differences versus NumPy are `8.673617379884035e-19` for `cluster_group_debias` and `4.336808689942018e-19` for `driscoll_kraay_qs` on both GPU backends. The QS estimator path records `n_periods=8`, `bandwidth=2`, `max_weighted_lag=7`, and `all_observed_lags_weighted=true`; legacy Pooled HAC remains `row_order_hac=true`.

The ordered-categorical chronology regression is a CPU metadata-to-time-code contract and is maintained separately in hosted tests: ordered categories such as `t1,t2,t10` are shown equivalent to explicit numeric chronology before the compact codes enter backend-native grouped-score accumulation. The physical runner then validates those backend-native covariance operations on both CuPy and Torch.

## Current synchronized performance artifact

- path: `results/pr126_p100/panel_stage_c_performance_c151550a.json`
- measurement SHA: `{MEASUREMENT_SHA}`
- artifact repository commit: `{artifact_commit}`
- Git blob: `{perf_blob}`
- SHA-256: `{perf_sha}`
- schema version: 2
- working tree clean: true
- timing scope: synchronized end-to-end estimator fit
- input residency: X/y/entity/time preloaded on selected GPU backend; cluster labels remain CPU metadata
- rows: 58 = 54 base rows + 4 high-T QS rows
- repeats per row: 3

High-T `N=10,000`, `k=2`, `T=200` medians:

- CuPy: Pooled QS `23.4445 ms`; Panel entity-FE QS `32.3719 ms`.
- Torch: Pooled QS `16.1504 ms`; Panel entity-FE QS `23.0404 ms`.

The performance artifact is accepted as bounded synchronized timing evidence. It does **not** contain or imply a speedup claim or CPU/external baseline.

## Canonical benchmark promotion

The current raw rerun artifacts are registered directly as SHA-256-protected canonical frontend sources; no normalized duplicate is created.

- validation source id: `{corr_id}` — 56 validation-only rows = `(26 estimator + 2 public primitive) x 2 backends`; no timing or speedup.
- performance source id: `{perf_id}` — 58 timing rows; no speedup.
- source date: `2026-08-10`
- environment: `remote-p100-pr126-20260810`

The Stage-C parsers fail closed on measurement-SHA drift, backend/case identity drift, public-primitive matrix drift, non-finite/out-of-tolerance correctness differences, malformed timing samples, median/sample inconsistency, exact 54-row base matrix drift, and exact four-row high-T QS matrix drift.

## Historical artifacts retained

The following raw files remain in the repository for audit history but are no longer current required canonical sources:

- `results/pr126_p100/panel_stage_c_gpu_validation_9c0b3050.json` — SHA-256 `a0d258f6d6b8243e82684a29305606e5f6bd91bbe271c3ed335b32b5ec973665`.
- `results/pr126_p100/panel_stage_c_performance_9c0b3050.json` — SHA-256 `214284f02a5e21e775e58deaf2fa3cc9b6384d392b96c6f300f31f4a02953b1c`.

## Physical conclusion

Stage-C post-review physical correctness/backend provenance and bounded synchronized performance gates are **ACCEPTED**. Exact-final-head hosted CI and a fresh `.claude/skills/code-review.md` review remain post-promotion lifecycle gates. Any later change to `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py` must be audited for whether a new physical run is required.
''', encoding="utf-8")

# EN/CN model docs: post-review physical gate is now closed on c151550a.
en = ROOT / "docs/en/models/panel.md"
replace_once(
    en,
    "Tesla P100 correctness and synchronized performance evidence were recorded for the pre-review exact-clean head `9c0b3050...`; the subsequent ordered-categorical chronology fix changes production covariance input handling, so those artifacts are now historical evidence and a fresh exact-head P100 rerun is required before PR #126 returns to Ready.",
    "The Ready-triggered ordered-categorical chronology fix was revalidated on exact-clean head `c151550a...` using Tesla P100: CuPy and Torch each pass all 26 estimator covariance cases plus two direct public covariance primitives, and the synchronized performance rerun includes the bounded `N=10,000`, `k=2`, `T=200` QS scenario. The earlier `9c0b3050...` artifacts remain immutable historical evidence."
)
replace_once(
    en,
    "The previously recorded Tesla P100 artifacts passed 26/26 estimator covariance cases plus 2/2 direct public covariance primitives per backend and included synchronized `N=10,000`, `k=2`, `T=200` QS timing. Because the post-review ordered-categorical chronology fix modifies production covariance metadata handling, those artifacts remain historical evidence but no longer close the exact-head physical gate; the updated candidate must be rerun on CuPy and Torch before Ready/merge consideration.",
    "Post-review exact-clean-head Tesla P100 acceptance is complete on `c151550a...`: CuPy and Torch each pass 26/26 estimator covariance cases plus 2/2 direct public covariance primitives with requested/executed backend identity and no CPU fallback. The synchronized performance rerun covers the three base scales and explicit `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it records timing only and makes no speedup claim. The superseded `9c0b3050...` artifacts remain historical audit evidence."
)

cn = ROOT / "docs/cn/models/panel.md"
replace_once(
    cn,
    "pre-review exact-clean head `9c0b3050...` 已记录 Tesla P100 correctness 与同步 performance evidence；随后 ordered-categorical chronology 修复改变了 production covariance input handling，因此这些产物现作为历史证据保留，PR #126 回到 Ready 前必须在新的 exact head 上重新完成 P100 验证。",
    "Ready-triggered ordered-categorical chronology 修复已在 exact-clean head `c151550a...` 上重新完成 Tesla P100 验证：CuPy 与 Torch 各自通过全部 26 个 estimator covariance case 和 2 个 direct public covariance primitive，同步 performance rerun 也覆盖有界的 `N=10,000`、`k=2`、`T=200` QS 场景。此前 `9c0b3050...` 产物继续作为不可变历史证据保留。"
)
replace_once(
    cn,
    "此前 Tesla P100 产物已在每个 backend 上通过 26/26 estimator covariance case 与 2/2 direct public covariance primitive，并包含同步 `N=10,000`、`k=2`、`T=200` QS timing。由于 post-review ordered-categorical chronology 修复修改了 production covariance metadata handling，这些产物继续作为历史证据保留，但不再闭合新的 exact-head physical gate；更新后的 candidate 必须在 CuPy 与 Torch 上重跑后才能恢复 Ready/merge-ready 结论。",
    "post-review exact-clean head `c151550a...` 的 Tesla P100 acceptance 已闭合：CuPy 与 Torch 每个 backend 均通过 26/26 estimator covariance case 与 2/2 direct public covariance primitive，requested/executed backend 一致且无 CPU fallback。同步 performance rerun 覆盖三个 base scale 以及显式 `N=10,000`、`k=2`、`T=200` QS all-lag 场景，只记录 timing，不声明 speedup；旧 `9c0b3050...` 产物继续作为历史审计证据。"
)

# Root changelog: identify the post-review rerun as the current physical acceptance.
root_changelog = ROOT / "CHANGELOG.md"
replace_once(
    root_changelog,
    f"Tesla P100 acceptance passed on clean implementation head `{OLD_MEASUREMENT_SHA}`: both CuPy and Torch passed all 26 estimator covariance cases plus two direct public primitives, and synchronized performance evidence includes the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario without making a speedup claim.",
    f"After the Ready-triggered ordered-categorical Driscoll-Kraay chronology fix, Tesla P100 acceptance was rerun and passed on exact clean implementation head `{MEASUREMENT_SHA}`: both CuPy and Torch passed all 26 estimator covariance cases plus two direct public primitives, and synchronized performance evidence includes the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario without making a speedup claim. The earlier `{OLD_MEASUREMENT_SHA}` artifacts remain immutable historical evidence."
)

# EN changelog current acceptance.
en_changelog = ROOT / "docs/en/changelog.md"
replace_once(en_changelog, "> Last updated: 2026-08-09<br>", "> Last updated: 2026-08-10<br>")
replace_once(
    en_changelog,
    f"Physical CUDA acceptance is complete on exact clean implementation head `{OLD_MEASUREMENT_SHA}` using Tesla P100-SXM2-16GB. CuPy and Torch each pass all 26 estimator covariance cases plus two direct public covariance primitives without CPU fallback. The separate synchronized performance artifact covers the three base scales and the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it records timing only and makes no speedup claim.",
    f"After the Ready-triggered ordered-categorical Driscoll-Kraay chronology fix, physical CUDA acceptance was rerun and completed on exact clean implementation head `{MEASUREMENT_SHA}` using Tesla P100-SXM2-16GB. CuPy and Torch each pass all 26 estimator covariance cases plus two direct public covariance primitives without CPU fallback. The synchronized performance rerun covers the three base scales and bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it records timing only and makes no speedup claim. The earlier `{OLD_MEASUREMENT_SHA}` artifacts remain immutable historical evidence."
)

# CN changelog: remove the stale contradictory pending claim and pin the current rerun.
cn_changelog = ROOT / "docs/cn/changelog.md"
replace_once(cn_changelog, "> 最后更新：2026-08-09<br>", "> 最后更新：2026-08-10<br>")
replace_once(
    cn_changelog,
    f"PR #126 的物理 CUDA 验收已在精确且干净的实现提交 `{OLD_MEASUREMENT_SHA}` 上使用 Tesla P100-SXM2-16GB 完成：CuPy 与 Torch 均通过 26 个估计器协方差案例和 2 个直接公共协方差 primitive，且没有 CPU fallback。独立的同步性能证据覆盖三个基础规模及 `N=10,000`、`k=2`、`T=200` 的 QS all-lag 场景；该证据只记录 timing，不声明 speedup。",
    f"Ready-triggered ordered-categorical Driscoll-Kraay chronology 修复后，PR #126 的物理 CUDA 验收已在精确且干净的实现提交 `{MEASUREMENT_SHA}` 上使用 Tesla P100-SXM2-16GB 重新执行并通过：CuPy 与 Torch 均通过 26 个估计器协方差案例和 2 个直接公共协方差 primitive，且没有 CPU fallback。同步性能 rerun 覆盖三个基础规模及 `N=10,000`、`k=2`、`T=200` 的 QS all-lag 场景；该证据只记录 timing，不声明 speedup。旧 `{OLD_MEASUREMENT_SHA}` 产物继续作为不可变历史证据保留。"
)
replace_once(
    cn_changelog,
    "physical CUDA gate 与 hosted definition gate 明确分离：`dev/benchmarks/validate_panel_stage_c_gpu.py` 与 `dev/benchmarks/benchmark_panel_stage_c_covariance.py` 仍需在最终 exact clean implementation head 上执行。当前不宣称 GPU speedup，也不宣称最终 physical acceptance 已完成。",
    "physical CUDA gate 与 hosted definition gate 保持分离；post-review exact-clean-head P100 correctness 与 synchronized performance 已完成，因此当前 physical acceptance 已闭合。性能证据仍只表示对应工作负载的 bounded timing，不宣称通用 GPU speedup。"
)

print(json.dumps({
    "measurement_sha": MEASUREMENT_SHA,
    "artifact_commit": artifact_commit,
    "correctness_sha256": corr_sha,
    "performance_sha256": perf_sha,
    "correctness_blob": corr_blob,
    "performance_blob": perf_blob,
    "correctness_source_id": corr_id,
    "performance_source_id": perf_id,
}, indent=2))
