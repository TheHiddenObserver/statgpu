from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ID = "panel-stage-c-final-validation-pr126-20260813-62fbf89e58fb"
PERFORMANCE_ID = "panel-stage-c-final-performance-pr126-20260813-980ffe9bd392"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


catalog = ROOT / "dev/tests/test_benchmark_catalog.py"
old_list_tail = '''        "panel-stage-c-identifiability-validation-pr126-20260812-2d929bccf1c7",
        "panel-stage-c-identifiability-performance-pr126-20260812-2238002d491f",
    ]'''
new_list_tail = f'''        "panel-stage-c-identifiability-validation-pr126-20260812-2d929bccf1c7",
        "panel-stage-c-identifiability-performance-pr126-20260812-2238002d491f",
        "{VALIDATION_ID}",
        "{PERFORMANCE_ID}",
    ]'''
replace_once(catalog, old_list_tail, new_list_tail, "panel coverage source snapshot")
text = catalog.read_text(encoding="utf-8")
for old, new in [
    ('assert inventory["registered_sources"] == len(manifest["sources"]) == 19',
     'assert inventory["registered_sources"] == len(manifest["sources"]) == 21'),
    ('assert inventory["available_registered_sources"] == 19',
     'assert inventory["available_registered_sources"] == 21'),
    ('assert inventory["parsed_registered_sources"] == 19',
     'assert inventory["parsed_registered_sources"] == 21'),
]:
    if text.count(old) != 1:
        raise SystemExit(f"catalog count anchor drifted: {old}")
    text = text.replace(old, new, 1)
catalog.write_text(text, encoding="utf-8")

inventory = ROOT / "dev/tests/test_benchmark_inventory_v2.py"
text = inventory.read_text(encoding="utf-8")
replacements = {
    'assert len(registered) == inventory["registered_sources"] == 19':
        'assert len(registered) == inventory["registered_sources"] == 21',
    'assert serialized["available_registered_sources"] == 19':
        'assert serialized["available_registered_sources"] == 21',
    'assert serialized["parsed_registered_sources"] == 19':
        'assert serialized["parsed_registered_sources"] == 21',
    'assert inventory["available_sources"] == 19':
        'assert inventory["available_sources"] == 21',
    'assert inventory["parsed_sources"] == 19':
        'assert inventory["parsed_sources"] == 21',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"inventory count anchor drifted: {old}")
    text = text.replace(old, new, 1)
inventory.write_text(text, encoding="utf-8")

frontend = ROOT / "dev/tests/test_benchmark_frontend_data.py"
text = frontend.read_text(encoding="utf-8")
replacements = {
    'assert len(manifest["sources"]) == 19': 'assert len(manifest["sources"]) == 21',
    'assert report["files_seen"] == 19': 'assert report["files_seen"] == 21',
    'assert report["files_parsed"] == 19': 'assert report["files_parsed"] == 21',
    'assert inventory["registered_sources"] == 19': 'assert inventory["registered_sources"] == 21',
    'assert inventory["available_sources"] == 19': 'assert inventory["available_sources"] == 21',
    'assert inventory["parsed_sources"] == 19': 'assert inventory["parsed_sources"] == 21',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"frontend count anchor drifted: {old}")
    text = text.replace(old, new, 1)
frontend.write_text(text, encoding="utf-8")

print("updated canonical-source snapshot tests from 19/v4 to 21/v5")
