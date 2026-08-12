#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one target, found {count}")
    path.write_text(text.replace(old, new, 1))


def harden_parser() -> None:
    path = Path("dev/benchmarks/frontend_data/parsers/panel_stage_c_identifiability.py")
    replace_once(
        path,
        "historical v1/v2 sources without overwriting or colliding with them.",
        "historical v1/v2/v3 sources without overwriting or colliding with them.",
        "parser history docstring",
    )
    replace_once(
        path,
        '''        repeats = int(row.get("repeats", 0))
        samples = row.get("samples_seconds")
        if repeats <= 0 or not isinstance(samples, list) or len(samples) != repeats:
            raise ValueError("PR126 identifiability Stage-C timing samples/repeats contract failed")''',
        '''        repeats = int(row.get("repeats", 0))
        samples = row.get("samples_seconds")
        if repeats != 3 or not isinstance(samples, list) or len(samples) != 3:
            raise ValueError(
                "PR126 identifiability Stage-C timing requires exactly three raw samples"
            )''',
        "exact three-sample contract",
    )
    replace_once(
        path,
        '''        expected_median = float(statistics.median(numeric_samples))
        if not math.isclose(median, expected_median, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("PR126 identifiability Stage-C reported median does not match raw samples")''',
        '''        expected_median = float(statistics.median(numeric_samples))
        if median != expected_median:
            raise ValueError(
                "PR126 identifiability Stage-C reported median must exactly match raw samples"
            )''',
        "exact median contract",
    )
    replace_once(
        path,
        '''                            {"metric": "synchronized_timing", "status": "pass"},
                            {"metric": "raw_samples_finite_positive", "status": "pass"},
                            {"metric": "median_matches_raw_samples", "status": "pass"},''',
        '''                            {"metric": "synchronized_timing", "status": "pass"},
                            {"metric": "exactly_three_raw_samples", "status": "pass"},
                            {"metric": "raw_samples_finite_positive", "status": "pass"},
                            {"metric": "median_exactly_matches_raw_samples", "status": "pass"},''',
        "performance validation checks",
    )


def harden_tests() -> None:
    path = Path("dev/tests/test_panel_stage_c_identifiability_frontend_source.py")
    text = path.read_text()
    if "import math\n" not in text:
        text = text.replace("import json\n", "import json\nimport math\n", 1)
    old = '''def test_v4_performance_parser_rejects_matrix_or_median_drift(tmp_path):
    payload = json.loads(PERFORMANCE.read_text())
    broken = copy.deepcopy(payload)
    broken["rows"] = broken["rows"][:-1]
    with pytest.raises(ValueError, match="60 rows"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-matrix.json", broken), ENV
        )

    broken = copy.deepcopy(payload)
    broken["rows"][0]["median_seconds"] *= 2.0
    with pytest.raises(ValueError, match="median"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-median.json", broken), ENV
        )
'''
    new = '''def test_v4_performance_parser_rejects_matrix_repeat_or_median_drift(tmp_path):
    payload = json.loads(PERFORMANCE.read_text())
    broken = copy.deepcopy(payload)
    broken["rows"] = broken["rows"][:-1]
    with pytest.raises(ValueError, match="60 rows"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-matrix.json", broken), ENV
        )

    broken = copy.deepcopy(payload)
    broken["rows"][0]["repeats"] = 2
    broken["rows"][0]["samples_seconds"] = broken["rows"][0]["samples_seconds"][:2]
    broken["rows"][0]["median_seconds"] = sum(broken["rows"][0]["samples_seconds"]) / 2.0
    with pytest.raises(ValueError, match="exactly three raw samples"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-repeats.json", broken), ENV
        )

    broken = copy.deepcopy(payload)
    current = float(broken["rows"][0]["median_seconds"])
    broken["rows"][0]["median_seconds"] = math.nextafter(current, math.inf)
    with pytest.raises(ValueError, match="exactly match"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-median.json", broken), ENV
        )
'''
    if text.count(old) != 1:
        raise RuntimeError(f"v4 performance test anchor drifted: {text.count(old)}")
    path.write_text(text.replace(old, new, 1))


def update_review_record() -> None:
    path = Path("dev/reviews/pr126_round4_autofix_review_2026-08-12.md")
    text = path.read_text()
    text = text.replace(
        "`PHYSICAL_GPU_ACCEPTED / CANONICAL_PROMOTION_PENDING / NOT MERGE-READY`",
        "`PHYSICAL_GPU_ACCEPTED / CANONICAL_PROMOTED / HOSTED_FINAL_PENDING / NOT MERGE-READY`",
        1,
    )
    if "[MEDIUM][PARSER][fixed]" not in text:
        text += '''\n\n## Post-promotion independent parser review\n\n[MEDIUM][PARSER][fixed] The first v4 performance parser accepted any positive `repeats` count and used a tolerance-based median comparison, while the immutable source contract requires exactly three raw timing samples and an exactly persisted median. The parser now requires `repeats == 3`, exactly three samples, and exact equality with `statistics.median(samples)`. Corruption tests cover a two-sample row and a one-ULP median drift.\n\n[MEDIUM][ARTIFACT][fixed] The checkpoint header still said `CANONICAL_PROMOTION_PENDING` after canonical promotion commit `72bc21d3d0a1afd23467ecb1ff176d42df709cb4` had passed the dedicated v4 promotion gate. The record now reflects `CANONICAL_PROMOTED / HOSTED_FINAL_PENDING`.\n\nThe post-promotion parser hardening changes only parser/test/review artifacts. It does not touch `statgpu/panel/**`, the correctness runner, the performance runner, or either immutable raw JSON file, so the exact-clean `a99726e1...` P100 evidence remains applicable.\n'''
    path.write_text(text)


def main() -> None:
    harden_parser()
    harden_tests()
    update_review_record()
    print("PR126 v4 final parser hardening applied")


if __name__ == "__main__":
    main()
