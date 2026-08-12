#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_exact(path: str, old: str, new: str, *, expected: int = 1) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"{path}: expected {expected} occurrences of {old!r}, found {count}"
        )
    p.write_text(text.replace(old, new))


def normalize_promotion_helper() -> None:
    path = Path("dev/pr126_v4_promote.py")
    text = path.read_text()

    old = '''    text = replace_once(
        text,
        'if int(data.get("schema_version", -1)) != 1:',
        'if int(data.get("schema_version", -1)) != 2:',
        "correctness schema",
    )
    text = replace_once(
        text,
        'requires schema_version=1',
        'requires schema_version=2',
        "correctness schema message",
    )
    text = replace_once(
        text,
        'if int(data.get("schema_version", -1)) != 2:',
        'if int(data.get("schema_version", -1)) != 3:',
        "performance schema",
    )
    text = replace_once(
        text,
        'performance requires schema_version=2',
        'performance requires schema_version=3',
        "performance schema message",
    )'''
    new = '''    text = replace_once(
        text,
        'if int(data.get("schema_version", -1)) != 2:',
        'if int(data.get("schema_version", -1)) != 3:',
        "performance schema",
    )
    text = replace_once(
        text,
        'performance requires schema_version=2',
        'performance requires schema_version=3',
        "performance schema message",
    )
    text = replace_once(
        text,
        'if int(data.get("schema_version", -1)) != 1:',
        'if int(data.get("schema_version", -1)) != 2:',
        "correctness schema",
    )
    text = replace_once(
        text,
        'requires schema_version=1',
        'requires schema_version=2',
        "correctness schema message",
    )'''
    if text.count(old) != 1:
        raise RuntimeError(f"schema helper anchor drifted: {text.count(old)}")
    text = text.replace(old, new, 1)

    helper_anchor = '''def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, found {count}")
    return text.replace(old, new, 1)
'''
    helper_replacement = helper_anchor + '''

def replace_first(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"{label}: replacement target is missing")
    return text.replace(old, new, 1)
'''
    if text.count(helper_anchor) != 1:
        raise RuntimeError("replace_once helper anchor drifted")
    text = text.replace(helper_anchor, helper_replacement, 1)

    case_call_anchor = '''    text = replace_once(
        text,
        ''' + "'''" + '''                _bool_check("recorded_numpy_difference_finite", diff_ok),\\n            ]''' + "'''" + ''','''
    if text.count(case_call_anchor) != 1:
        raise RuntimeError(
            f"case transform call anchor drifted: {text.count(case_call_anchor)}"
        )
    text = text.replace(
        case_call_anchor,
        case_call_anchor.replace("replace_once", "replace_first", 1),
        1,
    )
    path.write_text(text)


def sync_fixtures() -> None:
    p = "dev/tests/test_benchmark_frontend_data.py"
    for old, new in (
        ('assert len(manifest["sources"]) == 17', 'assert len(manifest["sources"]) == 19'),
        ('assert report["files_seen"] == 17', 'assert report["files_seen"] == 19'),
        ('assert report["files_parsed"] == 17', 'assert report["files_parsed"] == 19'),
        ('assert inventory["registered_sources"] == 17', 'assert inventory["registered_sources"] == 19'),
        ('assert inventory["available_sources"] == 17', 'assert inventory["available_sources"] == 19'),
        ('assert inventory["parsed_sources"] == 17', 'assert inventory["parsed_sources"] == 19'),
    ):
        replace_exact(p, old, new)

    p = "dev/tests/test_benchmark_catalog.py"
    source_anchor = '        "panel-stage-c-rank-df-performance-pr126-20260812-09337cc62c94",\n'
    source_replacement = source_anchor + (
        '        "panel-stage-c-identifiability-validation-pr126-20260812-2d929bccf1c7",\n'
        '        "panel-stage-c-identifiability-performance-pr126-20260812-2238002d491f",\n'
    )
    replace_exact(p, source_anchor, source_replacement)
    for old, new in (
        ('assert inventory["registered_sources"] == len(manifest["sources"]) == 17', 'assert inventory["registered_sources"] == len(manifest["sources"]) == 19'),
        ('assert inventory["available_registered_sources"] == 17', 'assert inventory["available_registered_sources"] == 19'),
        ('assert inventory["parsed_registered_sources"] == 17', 'assert inventory["parsed_registered_sources"] == 19'),
    ):
        replace_exact(p, old, new)

    p = "dev/tests/test_benchmark_inventory_v2.py"
    for old, new in (
        ('assert len(registered) == inventory["registered_sources"] == 17', 'assert len(registered) == inventory["registered_sources"] == 19'),
        ('assert serialized["available_registered_sources"] == 17', 'assert serialized["available_registered_sources"] == 19'),
        ('assert serialized["parsed_registered_sources"] == 17', 'assert serialized["parsed_registered_sources"] == 19'),
        ('assert inventory["available_sources"] == 17', 'assert inventory["available_sources"] == 19'),
        ('assert inventory["parsed_sources"] == 17', 'assert inventory["parsed_sources"] == 19'),
    ):
        replace_exact(p, old, new)

    p = "dev/tests/test_frontend_domain_coverage.py"
    for old, new in (
        ('assert len(manifest["sources"]) == 17', 'assert len(manifest["sources"]) == 19'),
        ('assert len(output["runs"]) == 2272', 'assert len(output["runs"]) == 2426'),
        ('assert report["runs_generated"] == 2272', 'assert report["runs_generated"] == 2426'),
    ):
        replace_exact(p, old, new)


def main() -> None:
    normalize_promotion_helper()
    sync_fixtures()
    print("PR126 v4 promotion preflight synchronized")


if __name__ == "__main__":
    main()
