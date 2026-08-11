from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path.cwd()
MEASUREMENT = "ec511f539adeaaedf310f92248200d0868577532"
ARTIFACT_COMMIT = "e1a155bf77b416e0873a037015aaafd22371ab11"
OLD_MEASUREMENT = "5ed763be2a331e6dc988ac133e79f0484d4cdebd"
SOURCE_DATE = "2026-08-11"
OLD_ENV, NEW_ENV = "remote-p100-pr126-20260810", "remote-p100-pr126-20260811"
OLD_VCOMP, NEW_VCOMP = "panel-stage-c-pr126-validation-20260810", "panel-stage-c-pr126-validation-20260811"
OLD_PCOMP, NEW_PCOMP = "panel-stage-c-pr126-performance-20260810", "panel-stage-c-pr126-performance-20260811"
OLD_VID = "panel-stage-c-validation-pr126-20260810-7d8777fabe32"
OLD_PID = "panel-stage-c-performance-pr126-20260810-75da75c0405c"
OLD_VPATH = "results/pr126_p100/panel_stage_c_gpu_validation_5ed763be.json"
OLD_PPATH = "results/pr126_p100/panel_stage_c_performance_5ed763be.json"
NEW_VPATH = "results/pr126_p100/panel_stage_c_gpu_validation_ec511f53.json"
NEW_PPATH = "results/pr126_p100/panel_stage_c_performance_ec511f53.json"
VP, PP = ROOT / NEW_VPATH, ROOT / NEW_PPATH

EXPECTED_CASES = {
    "pooled_hc0", "pooled_hc2", "pooled_hc3",
    "pooled_cluster_one_way", "pooled_cluster_two_way_group_debias",
    "pooled_dk_bartlett", "pooled_dk_qs", "pooled_legacy_hac",
    "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3",
    "panel_two_way_hc3", "panel_two_way_cluster_group_debias", "panel_two_way_dk",
    "random_effects_explicit_constant_robust", "random_effects_explicit_constant_hc0",
    "random_effects_explicit_constant_hc2", "random_effects_explicit_constant_hc3",
    "random_effects_cluster_two_way", "random_effects_dk",
    "between_hc0", "between_hc2", "between_hc3",
    "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
}
EXPECTED_PRIMITIVES = {
    "cluster_group_debias", "driscoll_kraay_qs", "ill_conditioned_hc0",
    "ill_conditioned_hc2", "ill_conditioned_hc3", "ill_conditioned_dk",
}
BASE_CASES = {
    "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",
    "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",
    "random_effects_nonrobust", "random_effects_hc3",
}
BASE_SCALES = {(10000, 2, 20), (100000, 2, 20), (100000, 10, 20)}
HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}

def sh(*args):
    return subprocess.check_output(list(args), text=True).strip()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected one marker, got {text.count(old)}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

def replace_line(path, needle, new_line):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    idx = [i for i, line in enumerate(lines) if needle in line]
    if len(idx) != 1:
        raise RuntimeError(f"{path}: expected one line containing {needle!r}, got {len(idx)}")
    lines[idx[0]] = new_line
    path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")

def replace_paragraph(path, needles, new_paragraph):
    text = path.read_text(encoding="utf-8")
    parts = text.split("\n\n")
    hits = [i for i, part in enumerate(parts) if all(x in part for x in needles)]
    if len(hits) != 1:
        raise RuntimeError(f"{path}: expected one paragraph for {needles}, got {len(hits)}")
    parts[hits[0]] = new_paragraph
    path.write_text("\n\n".join(parts), encoding="utf-8")

def no_secret_keys(obj, where="$"):
    bad = ("token", "secret", "password", "passwd", "api_key", "apikey", "private_key")
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm = str(key).lower().replace("-", "_")
            if any(term in norm for term in bad):
                raise AssertionError(f"credential-like key at {where}.{key}")
            no_secret_keys(value, f"{where}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            no_secret_keys(value, f"{where}[{i}]")

changed = set(sh("git", "diff", "--name-only", f"{MEASUREMENT}..{ARTIFACT_COMMIT}").splitlines())
assert changed == {NEW_VPATH, NEW_PPATH}, changed
validation = json.loads(VP.read_text(encoding="utf-8"))
performance = json.loads(PP.read_text(encoding="utf-8"))
no_secret_keys(validation)
no_secret_keys(performance)

assert validation["schema_version"] == 1
assert validation["git_sha"] == MEASUREMENT
assert validation["working_tree_clean"] is True
assert validation["status"] == "success"
assert validation["environment"]["gpu"] == "Tesla P100-SXM2-16GB"
assert validation["case_count_per_backend"] == 26
assert validation["public_primitive_count_per_backend"] == 6
assert set(validation["backends"]) == {"cupy", "torch"}
for backend in ("cupy", "torch"):
    payload = validation["backends"][backend]
    assert payload["status"] == "success"
    assert payload["requested_backend"] == backend
    assert set(payload["cases"]) == EXPECTED_CASES
    assert set(payload["public_primitives"]) == EXPECTED_PRIMITIVES
    for name, item in payload["cases"].items():
        assert item["status"] == "success", (backend, name)
        assert item["executed_backend"] == backend, (backend, name)
        diffs = item["max_abs_differences"]
        assert diffs
        assert all(math.isfinite(float(x)) and float(x) >= 0 for x in diffs.values())
    for name, item in payload["public_primitives"].items():
        assert item["status"] == "success", (backend, name)
        assert item["executed_backend"] == backend, (backend, name)
        diff = float(item["max_abs_difference"])
        assert math.isfinite(diff) and diff >= 0

assert performance["schema_version"] == 2
assert performance["git_sha"] == MEASUREMENT
assert performance["working_tree_clean"] is True
assert performance["benchmark"] == "panel_stage_c_covariance_fit_overhead"
assert performance["timing_scope"] == "synchronized end-to-end estimator fit"
assert performance["high_t_scale"] == "10000x2x200"
assert set(performance["environment"]["gpu_by_backend"]) == {"cupy", "torch"}
assert set(performance["environment"]["gpu_by_backend"].values()) == {"Tesla P100-SXM2-16GB"}
rows = performance["rows"]
assert len(rows) == 58
assert {r["backend"] for r in rows} == {"cupy", "torch"}
base = [r for r in rows if r["scenario"] == "base"]
high = [r for r in rows if r["scenario"] == "high_t_qs"]
expected_base = {(b, c, n, k, t) for b in ("cupy", "torch") for c in BASE_CASES for n, k, t in BASE_SCALES}
actual_base = {(r["backend"], r["case"], int(r["n_samples"]), int(r["n_features"]), int(r["n_times"])) for r in base}
assert len(base) == 54 and actual_base == expected_base
expected_high = {(b, c, 10000, 2, 200) for b in ("cupy", "torch") for c in HIGH_T_CASES}
actual_high = {(r["backend"], r["case"], int(r["n_samples"]), int(r["n_features"]), int(r["n_times"])) for r in high}
assert len(high) == 4 and actual_high == expected_high
for row in rows:
    samples = [float(x) for x in row["samples_seconds"]]
    median = float(row["median_seconds"])
    assert int(row["repeats"]) == len(samples) > 0
    assert all(math.isfinite(x) and x > 0 for x in samples)
    assert math.isfinite(median) and median > 0
    assert math.isclose(median, statistics.median(samples), rel_tol=1e-12, abs_tol=1e-15)

vsha, psha = digest(VP), digest(PP)
vblob, pblob = sh("git", "hash-object", str(VP)), sh("git", "hash-object", str(PP))
VID = f"panel-stage-c-validation-pr126-20260811-{vsha[:12]}"
PID = f"panel-stage-c-performance-pr126-20260811-{psha[:12]}"

parser = ROOT / "dev/benchmarks/frontend_data/parsers/panel_stage_c.py"
replace_once(parser, '_SOURCE_DATE = "2026-08-10"', '_SOURCE_DATE = "2026-08-11"')
replace_once(parser, f'_MEASUREMENT_SHA = "{OLD_MEASUREMENT}"', f'_MEASUREMENT_SHA = "{MEASUREMENT}"')

manifest_path = ROOT / "dev/benchmarks/frontend_sources.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
env = manifest["environments"].pop(OLD_ENV)
env["label"] = "Tesla P100 PR #126 Panel Stage C — 2026-08-11"
manifest["environments"][NEW_ENV] = env
vcomp = manifest["comparisons"].pop(OLD_VCOMP)
pcomp = manifest["comparisons"].pop(OLD_PCOMP)
vcomp.update(label="Panel Stage C physical validation — PR #126 — 2026-08-11", env_id=NEW_ENV)
pcomp.update(label="Panel Stage C synchronized performance — PR #126 — 2026-08-11", env_id=NEW_ENV)
manifest["comparisons"][NEW_VCOMP] = vcomp
manifest["comparisons"][NEW_PCOMP] = pcomp
stage = {s["parser"]: s for s in manifest["sources"] if s["parser"].startswith("panel_stage_c_")}
assert set(stage) == {"panel_stage_c_physical_validation", "panel_stage_c_performance"}
vsrc, psrc = stage["panel_stage_c_physical_validation"], stage["panel_stage_c_performance"]
assert (vsrc["source_id"], vsrc["path"]) == (OLD_VID, OLD_VPATH)
assert (psrc["source_id"], psrc["path"]) == (OLD_PID, OLD_PPATH)
vsrc.update(
    source_id=VID, comparison_id=NEW_VCOMP, path=NEW_VPATH, sha256=vsha,
    env_id=NEW_ENV, source_date=SOURCE_DATE, measurement_git_sha=MEASUREMENT,
    raw_git_sha=MEASUREMENT,
    provenance_note=(
        f"Final PR #126 post-re-audit exact-clean P100 correctness/backend-provenance evidence. "
        f"Raw artifact commit {ARTIFACT_COMMIT}; Git blob {vblob}; SHA-256 {vsha}. "
        "CuPy and Torch each pass 26 estimator covariance cases plus six direct public covariance primitives "
        "(32/32 per backend), including full-rank ill-conditioned HC0/HC2/HC3/Driscoll-Kraay, with "
        "requested/executed backend identity and no numerical CPU fallback. Earlier 5ed763be, aad53587, "
        "c151550a, and 9c0b3050 measurements remain immutable historical evidence."
    ),
)
psrc.update(
    source_id=PID, comparison_id=NEW_PCOMP, path=NEW_PPATH, sha256=psha,
    env_id=NEW_ENV, source_date=SOURCE_DATE, measurement_git_sha=MEASUREMENT,
    raw_git_sha=MEASUREMENT,
    provenance_note=(
        f"Final PR #126 post-re-audit synchronized end-to-end P100 timing evidence. "
        f"Raw artifact commit {ARTIFACT_COMMIT}; Git blob {pblob}; SHA-256 {psha}. "
        "58 synchronized rows cover three base scales plus bounded N=10000, k=2, T=200 QS all-lag "
        "cases on both CuPy and Torch. No speedup or CPU-baseline claim is encoded. Earlier 5ed763be, "
        "aad53587, c151550a, and 9c0b3050 artifacts remain immutable historical evidence."
    ),
)
write_json(manifest_path, manifest)

coverage_path = ROOT / "dev/benchmarks/benchmark_coverage_matrix.json"
coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
panel = next(x for x in coverage["capabilities"] if x["capability_id"] == "panel-estimation")
panel["source_ids"] = [VID if x == OLD_VID else PID if x == OLD_PID else x for x in panel["source_ids"]]
panel["disposition"] = (
    "June timing rows cover aligned PanelOLS and RandomEffects; PR #122 provides canonical Stage-B "
    "diagnostic/physical validation. PR #126 fresh post-re-audit exact-clean P100 evidence at ec511f53 "
    "provides canonical Stage-C CuPy/Torch correctness for 26 estimator covariance integrations plus six "
    "direct public primitives per backend, including ill-conditioned HC0/HC2/HC3/DK, and synchronized timing "
    "for three base scales plus a bounded N=10,000, k=2, T=200 QS all-lag scenario. The 5ed763be, aad53587, "
    "c151550a, and 9c0b3050 artifacts are historical only. No Stage-C speedup claim is made."
)
write_json(coverage_path, coverage)

frontend_test = ROOT / "dev/tests/test_panel_stage_c_frontend_source.py"
replace_once(frontend_test, "panel_stage_c_gpu_validation_5ed763be.json", "panel_stage_c_gpu_validation_ec511f53.json")
replace_once(frontend_test, "panel_stage_c_performance_5ed763be.json", "panel_stage_c_performance_ec511f53.json")
text = frontend_test.read_text(encoding="utf-8")
if OLD_ENV not in text:
    raise RuntimeError("old Stage-C env missing from frontend test")
frontend_test.write_text(text.replace(OLD_ENV, NEW_ENV), encoding="utf-8")

catalog_test = ROOT / "dev/tests/test_benchmark_catalog.py"
replace_once(catalog_test, OLD_VPATH, NEW_VPATH)
replace_once(catalog_test, OLD_PPATH, NEW_PPATH)
replace_once(catalog_test, OLD_VID, VID)
replace_once(catalog_test, OLD_PID, PID)
text = catalog_test.read_text(encoding="utf-8")
marker = '    assert aad_performance["registered"] is False\n'
if text.count(marker) != 1:
    raise RuntimeError("Stage-C historical marker drifted")
extra = marker + "\n".join([
    "",
    "    five_validation = next(",
    "        entry for entry in entries",
    f'        if entry["path"] == "{OLD_VPATH}"',
    "    )",
    "    five_performance = next(",
    "        entry for entry in entries",
    f'        if entry["path"] == "{OLD_PPATH}"',
    "    )",
    '    assert five_validation["classification"] == "historical_or_excluded"',
    '    assert five_validation["registered"] is False',
    '    assert five_performance["classification"] == "historical_or_excluded"',
    '    assert five_performance["registered"] is False',
    "",
])
catalog_test.write_text(text.replace(marker, extra, 1), encoding="utf-8")

replace_line(
    ROOT / "CHANGELOG.md", "pre-re-audit exact-clean Tesla P100 run",
    "- Fresh post-re-audit physical acceptance was completed on exact-clean head `ec511f539adeaaedf310f92248200d0868577532` using Tesla P100-SXM2-16GB: CuPy and Torch each passed 32/32 correctness checks, and synchronized performance passed all 58 rows including the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario. Earlier `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` runs remain immutable historical evidence."
)
replace_paragraph(
    ROOT / "docs/en/changelog.md", ["pre-re-audit source", "fresh exact-source physical validation is pending"],
    "After the 2026-08-11 strict review fixed `PanelOLS.summary()` formula term naming, fresh exact-source physical acceptance was rerun on clean head `ec511f539adeaaedf310f92248200d0868577532` using Tesla P100-SXM2-16GB. CuPy and Torch each passed all 26 estimator covariance cases plus 6 direct public covariance primitives (32/32 per backend), including full-rank ill-conditioned HC0/HC2/HC3 and Driscoll-Kraay cases, with requested/executed backend identity and no numerical CPU fallback. The synchronized performance run passed 58 rows across three base scales plus the bounded `N=10,000`, `k=2`, `T=200` QS all-lag scenario and makes no speedup or CPU-baseline claim. Earlier `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` artifacts remain immutable historical evidence."
)
replace_paragraph(
    ROOT / "docs/cn/changelog.md", ["re-audit 之前的 source", "fresh exact-source physical validation 仍待完成"],
    "2026-08-11 的严格 review 修复 `PanelOLS.summary()` 的 formula term naming 后，已在精确且干净的提交 `ec511f539adeaaedf310f92248200d0868577532` 上使用 Tesla P100-SXM2-16GB 重新完成 fresh exact-source physical acceptance。CuPy 与 Torch 各自通过 26 个 estimator covariance case 和 6 个 direct public covariance primitive（每个 backend 32/32），包括 full-rank ill-conditioned HC0/HC2/HC3 与 Driscoll-Kraay，并验证 requested/executed backend 一致且无数值 CPU fallback。同步 performance 共通过 58 行，覆盖三个基础规模以及有界的 `N=10,000`、`k=2`、`T=200` QS all-lag 场景，不声明 speedup 或 CPU baseline。此前 `5ed763be...`、`aad53587...`、`c151550a...` 与 `9c0b3050...` 产物继续作为不可变历史证据保留。"
)
replace_paragraph(
    ROOT / "docs/en/models/panel.md", ["Stage C of the Tier-1 panel roadmap", "fresh exact-head physical acceptance"],
    "Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. After the 2026-08-11 strict re-audit fixed formula/inference presentation in `PanelOLS.summary()`, the final source was physically revalidated on exact-clean head `ec511f53...` using Tesla P100: CuPy and Torch each passed all 26 estimator covariance cases plus six direct public covariance primitives (32/32 per backend), including full-rank ill-conditioned HC0/HC2/HC3/DK. The synchronized performance run passed 58 rows including the bounded `N=10,000`, `k=2`, `T=200` QS scenario. Earlier `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` artifacts remain immutable historical evidence."
)
replace_paragraph(
    ROOT / "docs/en/models/panel.md", ["Hosted Stage-C tests", "fresh physical validation on the final review-fix head is pending"],
    "Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. Fresh exact-clean Tesla P100 acceptance on `ec511f53...` passed 26/26 estimator covariance cases plus 6/6 direct public covariance primitives on each of CuPy and Torch (32/32 per backend), including full-rank ill-conditioned HC0/HC2/HC3/DK regressions, with requested/executed backend identity and no CPU fallback. The synchronized performance run passed all 58 rows across the three base scales plus the explicit `N=10,000`, `k=2`, `T=200` QS all-lag scenario. It records timing only and makes no speedup or CPU-baseline claim."
)
replace_paragraph(
    ROOT / "docs/cn/models/panel.md", ["Tier-1 Panel 路线的 Stage C", "重新完成 exact-head physical acceptance"],
    "Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。2026-08-11 的严格 re-audit 修复 `PanelOLS.summary()` 的 formula/inference 展示语义后，最终 source 已在 exact-clean head `ec511f53...` 上重新完成 Tesla P100 physical acceptance：CuPy 与 Torch 各自通过全部 26 个 estimator covariance case 和 6 个 direct public covariance primitive（每个 backend 32/32），包括 full-rank ill-conditioned HC0/HC2/HC3/DK；同步 performance 共通过 58 行，并覆盖有界的 `N=10,000`、`k=2`、`T=200` QS 场景。此前 `5ed763be...`、`aad53587...`、`c151550a...` 与 `9c0b3050...` 产物继续作为不可变历史证据保留。"
)
replace_paragraph(
    ROOT / "docs/cn/models/panel.md", ["hosted Stage-C tests", "fresh physical validation 仍待完成"],
    "hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。fresh exact-clean Tesla P100 acceptance `ec511f53...` 在 CuPy 与 Torch 每个 backend 上均通过 26/26 estimator covariance case 与 6/6 direct public covariance primitive（每个 backend 32/32），包括 full-rank ill-conditioned HC0/HC2/HC3/DK regressions，requested/executed backend 一致且无 CPU fallback。同步 performance 共通过 58 行，覆盖三个 base scale 以及显式 `N=10,000`、`k=2`、`T=200` QS all-lag 场景；只记录 timing，不声明 speedup 或 CPU baseline。"
)

review = ROOT / "dev/reviews/pr126_physical_gpu_validation.md"
review.write_text(f'''# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PHYSICAL_GPU_ACCEPTED / POST_PROMOTION_REVIEW_PENDING**

Validation tier: `remote-full`.

Fresh Tesla P100 evidence was measured from exact clean strict-review checkpoint `{MEASUREMENT}`. The raw artifact commit `{ARTIFACT_COMMIT}` differs from the measurement checkpoint only by `{NEW_VPATH}` and `{NEW_PPATH}`. Canonical promotion changes only parser/source metadata, coverage/tests/docs/review records, and deterministic frontend benchmark assets; it does not change `statgpu/panel/**`, `dev/benchmarks/validate_panel_stage_c_gpu.py`, or `dev/benchmarks/benchmark_panel_stage_c_covariance.py`. Therefore the physical measurement remains applicable under `RELEASING.md`.

## Correctness and backend-provenance evidence

- path: `{NEW_VPATH}`
- measurement SHA: `{MEASUREMENT}`
- raw artifact commit: `{ARTIFACT_COMMIT}`
- Git blob: `{vblob}`
- SHA-256: `{vsha}`
- GPU: Tesla P100-SXM2-16GB
- result: CuPy **32/32**, Torch **32/32** = 26 estimator covariance cases + 6 direct public primitives per backend
- direct primitives: `cluster_group_debias`, `driscoll_kraay_qs`, `ill_conditioned_hc0`, `ill_conditioned_hc2`, `ill_conditioned_hc3`, `ill_conditioned_dk`
- every estimator case and public primitive records the requested backend as the executed backend; no numerical CPU fallback was observed

## Synchronized performance evidence

- path: `{NEW_PPATH}`
- measurement SHA: `{MEASUREMENT}`
- raw artifact commit: `{ARTIFACT_COMMIT}`
- Git blob: `{pblob}`
- SHA-256: `{psha}`
- GPU: Tesla P100-SXM2-16GB for both CuPy and Torch
- rows: **58** = 54 base + 4 high-T QS
- timing scope: synchronized end-to-end estimator fit
- high-T matrix: CuPy/Torch × PooledOLS/PanelOLS QS at `N=10,000,k=2,T=200`
- every timing sample and stored median is finite and positive; each stored median equals the median of its raw samples
- no speedup claim or CPU-baseline claim is made

## Canonical promotion

- correctness source id: `{VID}`
- performance source id: `{PID}`
- source date: `{SOURCE_DATE}`
- environment: `{NEW_ENV}`

The parser fails closed on measurement SHA, clean-tree flag, exact 26+6 correctness identity, requested/executed backend identity, exact 58-row base/high-T matrix, and positive finite synchronized timing samples.

## Superseded historical evidence

The `5ed763be...`, `aad53587...`, `c151550a...`, and `9c0b3050...` Stage-C artifacts remain immutable audit history and are not current canonical acceptance sources.

## Remaining merge-readiness boundary

The physical gate is closed. Merge readiness still requires permanent hosted workflows green on the final post-promotion checkpoint and a final `.claude/skills/code-review.md` re-review with no new CRITICAL/HIGH/relevant-MEDIUM finding. PR #126 remains Draft until an explicit Ready transition is requested.
''', encoding="utf-8")

print("AUDIT_OK")
print("validation_sha256", vsha)
print("performance_sha256", psha)
print("validation_source_id", VID)
print("performance_source_id", PID)
print("validation_git_blob", vblob)
print("performance_git_blob", pblob)
