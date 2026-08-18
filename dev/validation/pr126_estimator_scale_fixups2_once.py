from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Restoring an actually unrepresentable squared quantity should yield Inf
# deliberately, not via an overflowing multiplication warning.
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    root = float(np.sqrt(value))\n    scaled_root = root * scale\n    return float(scaled_root * scaled_root)\n''',
    '''    root = float(np.sqrt(value))\n    scaled_root = root * scale\n    if scaled_root > np.sqrt(np.finfo(np.float64).max):\n        return float("inf")\n    return float(scaled_root * scaled_root)\n''',
    "explicit squared-scale overflow",
)

# The RE invariance regression is about Swamy-Arora transformation under extreme
# scale. Use robust covariance so Hausman diagnostic fingerprint moments (whose
# true squared scale is outside float64) do not add unrelated audit warnings.
replace_once(
    "dev/tests/test_panel_stage_c_final_review_fixes.py",
    '''    reference = RandomEffects(cov_type="nonrobust").fit(\n        x, y, entity_ids=entity\n    )\n    scale = 1.0e200\n    candidate = RandomEffects(cov_type="nonrobust").fit(\n        scale * x, scale * y, entity_ids=entity\n    )\n''',
    '''    reference = RandomEffects(cov_type="robust").fit(\n        x, y, entity_ids=entity\n    )\n    scale = 1.0e200\n    candidate = RandomEffects(cov_type="robust").fit(\n        scale * x, scale * y, entity_ids=entity\n    )\n''',
    "RE invariance covariance mode",
)

print("PR126 estimator-scale overflow cleanup staged")
