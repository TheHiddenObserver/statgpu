from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCE_COUNT = 17
NEW_RUN_COUNT = 2272


def replace_all(path: Path, old: str, new: str, expected: int) -> None:
    full = ROOT / path
    text = full.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} occurrences of {old!r}, found {count}")
    full.write_text(text.replace(old, new), encoding="utf-8")


def update_literal_counts() -> None:
    replace_all(
        Path("dev/tests/test_benchmark_frontend_data.py"),
        "== 15",
        f"== {NEW_SOURCE_COUNT}",
        6,
    )
    replace_all(
        Path("dev/tests/test_benchmark_catalog.py"),
        "== 15",
        f"== {NEW_SOURCE_COUNT}",
        3,
    )
    replace_all(
        Path("dev/tests/test_benchmark_inventory_v2.py"),
        "== 15",
        f"== {NEW_SOURCE_COUNT}",
        2,
    )
    replace_all(
        Path("dev/tests/test_frontend_domain_coverage.py"),
        "== 15",
        f"== {NEW_SOURCE_COUNT}",
        1,
    )
    replace_all(
        Path("dev/tests/test_frontend_domain_coverage.py"),
        "== 2120",
        f"== {NEW_RUN_COUNT}",
        1,
    )


def update_panel_coverage() -> tuple[str, str]:
    manifest_path = ROOT / "dev/benchmarks/frontend_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_parser = {source["parser"]: source["source_id"] for source in manifest["sources"]}
    validation_id = by_parser["panel_stage_c_rank_df_physical_validation"]
    performance_id = by_parser["panel_stage_c_rank_df_performance"]

    coverage_path = ROOT / "dev/benchmarks/benchmark_coverage_matrix.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    panel = next(
        row for row in coverage["capabilities"] if row["capability_id"] == "panel-estimation"
    )
    for source_id in (validation_id, performance_id):
        if source_id in panel["source_ids"]:
            raise RuntimeError(f"coverage already contains fresh source {source_id}")
        panel["source_ids"].append(source_id)
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    test_path = ROOT / "dev/tests/test_benchmark_catalog.py"
    text = test_path.read_text(encoding="utf-8")
    marker = '        "panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55",\n'
    if text.count(marker) != 1:
        raise RuntimeError("panel coverage test insertion marker drifted")
    insertion = marker + f'        "{validation_id}",\n        "{performance_id}",\n'
    test_path.write_text(text.replace(marker, insertion, 1), encoding="utf-8")
    return validation_id, performance_id


def main() -> None:
    update_literal_counts()
    validation_id, performance_id = update_panel_coverage()
    print("v3_validation_source_id", validation_id)
    print("v3_performance_source_id", performance_id)


if __name__ == "__main__":
    main()
