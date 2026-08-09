from pathlib import Path

path = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = path.read_text(encoding="utf-8")
old = '''    required_prefixes = {
        "pooled_hc0", "pooled_hc2", "pooled_hc3", "pooled_dk_bartlett", "pooled_dk_qs",
        "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3", "panel_two_way_hc3",
        "panel_two_way_cluster_group_debias", "panel_two_way_dk",
        "random_effects_explicit_constant_robust", "random_effects_explicit_constant_hc0",
        "random_effects_explicit_constant_hc2", "random_effects_explicit_constant_hc3",
        "random_effects_cluster_two_way", "random_effects_dk",
        "between_hc0", "between_hc2", "between_hc3",
        "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
    }
    for backend, payload in results.items():
        missing = sorted(required_prefixes - set(payload["cases"]))
        if missing:
            raise AssertionError(f"{backend}: missing required Stage-C physical cases: {missing}")

    output = {
'''
new = '''    required_cases = {
        "pooled_hc0", "pooled_hc2", "pooled_hc3",
        "pooled_cluster_one_way", "pooled_cluster_two_way_group_debias",
        "pooled_dk_bartlett", "pooled_dk_qs", "pooled_legacy_hac",
        "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3", "panel_two_way_hc3",
        "panel_two_way_cluster_group_debias", "panel_two_way_dk",
        "random_effects_explicit_constant_robust", "random_effects_explicit_constant_hc0",
        "random_effects_explicit_constant_hc2", "random_effects_explicit_constant_hc3",
        "random_effects_cluster_two_way", "random_effects_dk",
        "between_hc0", "between_hc2", "between_hc3",
        "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
    }
    if set(reference) != required_cases:
        missing = sorted(required_cases - set(reference))
        unexpected = sorted(set(reference) - required_cases)
        raise AssertionError(
            "NumPy reference Stage-C physical matrix drifted: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(reference) != 26:
        raise AssertionError(f"expected 26 Stage-C physical cases, got {len(reference)}")
    for backend, payload in results.items():
        if set(payload["cases"]) != required_cases:
            missing = sorted(required_cases - set(payload["cases"]))
            unexpected = sorted(set(payload["cases"]) - required_cases)
            raise AssertionError(
                f"{backend}: Stage-C physical matrix drifted: "
                f"missing={missing}, unexpected={unexpected}"
            )

    output = {
'''
if old not in text:
    raise SystemExit("expected Stage-C required case block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
