from pathlib import Path

path = Path("dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_df.py")
text = path.read_text(encoding="utf-8")
replacements = {
    "ID namespaces so the fresh ``3dc7df19`` evidence can coexist with the\nhistorical source without overwriting or colliding with it.":
        "ID namespaces so the fresh ``f1546476`` evidence can coexist with the\nhistorical v1/v2 sources without overwriting or colliding with them.",
    'parse_panel_stage_c_rank_df_physical_validation_v2':
        'parse_panel_stage_c_rank_df_physical_validation_v3',
    'parse_panel_stage_c_rank_df_performance_v2':
        'parse_panel_stage_c_rank_df_performance_v3',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}, found {count}")
    text = text.replace(old, new, 1)
if "3dc7df19" in text or "rank_df_physical_validation_v2" in text or "rank_df_performance_v2" in text:
    raise RuntimeError("stale v2/rank-policy provenance label remains in v3 parser")
if "f154647665788df2570439a1cc154a43f509aa45" not in text:
    raise RuntimeError("v3 parser lost exact measurement SHA")
path.write_text(text, encoding="utf-8")
