from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Keep the additive FE null-direction gauge centered only near the float64 range
# boundary. Ordinary scales retain the historical weighted-zero gauge bitwise;
# the extreme path uses a deterministic min-infinity-norm representation.
anchor = '''def _recover_two_way_effects(\n'''
helper = '''def _range_center_two_way_effects(entity_effects, time_effects, xp):\n    """Range-center an additive-effect gauge only near the float64 boundary.\n\n    ``entity + time`` is invariant under ``entity += c`` and ``time -= c``.\n    For ordinary magnitudes the returned shift is exactly zero, preserving the\n    established low-order cancellation path.  Once a compact effect exceeds an\n    eighth of DBL_MAX, center ``[entity, -time]`` at its midrange; this minimizes\n    the largest absolute stored effect and leaves a margin for subsequent ALS\n    subtraction.  Compact concatenation needs only two backend reductions.\n    """\n    coordinates = (\n        xp.cat([entity_effects, -time_effects], dim=0)\n        if getattr(xp, "__name__", "") == "torch"\n        else xp.concatenate([entity_effects, -time_effects], axis=0)\n    )\n    lower = xp.min(coordinates)\n    upper = xp.max(coordinates)\n    max_abs = xp.maximum(xp.abs(lower), xp.abs(upper))\n    activate = max_abs > float(np.finfo(np.float64).max / 8.0)\n    candidate = -(0.5 * lower + 0.5 * upper)\n    shift = xp.where(activate, candidate, xp.zeros_like(candidate))\n    return entity_effects + shift, time_effects - shift, activate\n\n\n'''
replace_once("statgpu/panel/_utils.py", anchor, helper + anchor)

old_init = '''    converged = False\n    final_metric = float("inf")\n    for _iteration in range(int(max_iter)):\n'''
new_init = '''    converged = False\n    final_metric = float("inf")\n    range_mode = xp.zeros_like(level_scale, dtype=xp.bool)\n    for _iteration in range(int(max_iter)):\n'''
replace_once("statgpu/panel/_utils.py", old_init, new_init)

old_loop = '''    for _iteration in range(int(max_iter)):\n        entity_effects = _compact_group_means(\n            values - time_effects[t_idx], entity_projection, xp, stable=stable\n        )\n        time_effects = _compact_group_means(\n            values - entity_effects[e_idx], time_projection, xp, stable=stable\n        )\n\n        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n'''
new_loop = '''    for _iteration in range(int(max_iter)):\n        entity_effects = _compact_group_means(\n            values - time_effects[t_idx], entity_projection, xp, stable=stable\n        )\n        entity_effects, time_effects, activated = _range_center_two_way_effects(\n            entity_effects, time_effects, xp\n        )\n        range_mode = range_mode | activated\n        time_effects = _compact_group_means(\n            values - entity_effects[e_idx], time_projection, xp, stable=stable\n        )\n        entity_effects, time_effects, activated = _range_center_two_way_effects(\n            entity_effects, time_effects, xp\n        )\n        range_mode = range_mode | activated\n\n        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n'''
replace_once("statgpu/panel/_utils.py", old_loop, new_loop)

old_final = '''    # The entity/time decomposition has a common-shift null direction.  Apply\n    # the public normalization once, after convergence, so the iterative path\n    # never materializes ``time_effect * group_count``.  Expanding by ``t_idx``\n    # expresses the observation-weighted mean directly and lets the shared\n    # stable reducer preserve cancellation without an overflowing product.\n    shift = stable_mean(time_effects[t_idx], xp)\n    time_effects = time_effects - shift\n    entity_effects = entity_effects + shift\n    return entity_effects, time_effects\n'''
new_final = '''    # Preserve the historical observation-weighted-zero time gauge when no\n    # range protection was needed.  Once the extreme path has activated, keep\n    # the range-centered gauge: forcing the weighted-zero shift back onto an\n    # otherwise finite decomposition can itself exceed float64 range.\n    weighted_shift = stable_mean(time_effects[t_idx], xp)\n    final_shift = xp.where(\n        range_mode, xp.zeros_like(weighted_shift), weighted_shift\n    )\n    time_effects = time_effects - final_shift\n    entity_effects = entity_effects + final_shift\n    return entity_effects, time_effects\n'''
replace_once("statgpu/panel/_utils.py", old_final, new_final)

# Maintained Torch/NumPy regression: a connected sparse incidence graph has an
# exactly additive level fit, but the old weighted-zero gauge grows to 3.727*M.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_two_way_effect_range_gauge_keeps_finite_exact_fit():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_two_way_effect_range_gauge_keeps_finite_exact_fit():\n    amplitude = 5.0e307\n    edges = np.asarray(\n        [[0, 1], [2, 4], [1, 4], [2, 1], [4, 0], [5, 2],\n         [5, 4], [3, 0], [3, 1], [5, 3], [0, 4]],\n        dtype=np.int64,\n    )\n    signs = np.asarray([-1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1], dtype=np.float64)\n    entity = np.repeat(edges[:, 0], 2)\n    time = np.repeat(edges[:, 1], 2)\n    values_np = np.repeat(amplitude * signs, 2)\n\n    for xp, values, entity_ids, time_ids in (\n        (np, values_np, entity, time),\n        (\n            torch,\n            torch.as_tensor(values_np, dtype=torch.float64),\n            torch.as_tensor(entity, dtype=torch.int64),\n            torch.as_tensor(time, dtype=torch.int64),\n        ),\n    ):\n        entity_effect, time_effect = _recover_two_way_effects(\n            values, entity_ids, time_ids, xp, max_iter=500, tol=1e-12\n        )\n        entity_np = np.asarray(\n            entity_effect.detach().cpu().numpy() if hasattr(entity_effect, "detach") else entity_effect\n        )\n        time_np = np.asarray(\n            time_effect.detach().cpu().numpy() if hasattr(time_effect, "detach") else time_effect\n        )\n        assert np.all(np.isfinite(entity_np))\n        assert np.all(np.isfinite(time_np))\n        assert float(np.max(np.abs(entity_np))) <= 3.01 * amplitude\n        assert float(np.max(np.abs(time_np))) <= 2.01 * amplitude\n        reconstructed = entity_np[entity] + time_np[time]\n        np.testing.assert_allclose(reconstructed, values_np, rtol=3e-12, atol=0.0)\n\n    # Public fit/predict must inherit the same finite gauge. Duplicate edges give\n    # positive residual df while a constant regressor is fully absorbed by FE.\n    X = np.ones((values_np.size, 1), dtype=np.float64)\n    model = PanelOLS(entity_effects=True, time_effects=True, cov_type="hc0").fit(\n        X, values_np, entity_ids=entity, time_ids=time\n    )\n    prediction = model.predict(X, entity_ids=entity, time_ids=time)\n    assert np.all(np.isfinite(prediction))\n    np.testing.assert_allclose(prediction, values_np, rtol=3e-12, atol=0.0)\n'''
    p.write_text(text, encoding="utf-8")

# Physical runner mirrors the exact sparse graph for CuPy/Torch CUDA.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
audit_marker = "def _two_way_effect_range_gauge_audit(backend):"
if audit_marker not in text:
    insert_before = "def _nonfinite_covariance_guard_audit(backend):\n"
    audit = '''def _two_way_effect_range_gauge_audit(backend):\n    amplitude = 5.0e307\n    edges = np.asarray(\n        [[0, 1], [2, 4], [1, 4], [2, 1], [4, 0], [5, 2],\n         [5, 4], [3, 0], [3, 1], [5, 3], [0, 4]],\n        dtype=np.int64,\n    )\n    signs = np.asarray([-1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1], dtype=np.float64)\n    entity_np = np.repeat(edges[:, 0], 2)\n    time_np = np.repeat(edges[:, 1], 2)\n    values_np = np.repeat(amplitude * signs, 2)\n    dummy = np.ones((values_np.size, 1), dtype=np.float64)\n    _X, values, entity, time = _to_backend(dummy, values_np, entity_np, time_np, backend)\n    xp = __import__("torch") if backend == "torch" else __import__("cupy")\n    entity_effect, time_effect = _recover_two_way_effects(\n        values, entity, time, xp, max_iter=500, tol=1e-12\n    )\n    entity_effect_np = _array(entity_effect)\n    time_effect_np = _array(time_effect)\n    if not np.all(np.isfinite(entity_effect_np)) or not np.all(np.isfinite(time_effect_np)):\n        raise AssertionError(f"{backend}: range-centered two-way effects are non-finite")\n    if np.max(np.abs(entity_effect_np)) > 3.01 * amplitude:\n        raise AssertionError(f"{backend}: entity FE gauge exceeded range bound")\n    if np.max(np.abs(time_effect_np)) > 2.01 * amplitude:\n        raise AssertionError(f"{backend}: time FE gauge exceeded range bound")\n    reconstructed = entity_effect_np[entity_np] + time_effect_np[time_np]\n    np.testing.assert_allclose(reconstructed, values_np, rtol=3e-12, atol=0.0)\n    return {\n        "status": "success",\n        "backend": backend,\n        "max_abs_entity_effect": float(np.max(np.abs(entity_effect_np))),\n        "max_abs_time_effect": float(np.max(np.abs(time_effect_np))),\n        "max_abs_reconstruction_error": _max_abs(reconstructed, values_np),\n    }\n\n\n'''
    if insert_before not in text:
        raise RuntimeError("physical runner insertion anchor not found")
    text = text.replace(insert_before, audit + insert_before, 1)
register_anchor = '''            "fixed_effect_map_range": _fixed_effect_map_range_audit(backend),\n'''
register_new = register_anchor + '''            "two_way_effect_range_gauge": _two_way_effect_range_gauge_audit(backend),\n'''
if register_new not in text:
    if register_anchor not in text:
        raise RuntimeError("physical runner registration anchor not found")
    text = text.replace(register_anchor, register_new, 1)
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
contract_marker = "def test_stage_c_runner_registers_two_way_effect_range_gauge_audit():"
if contract_marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_registers_two_way_effect_range_gauge_audit():\n    source = inspect.getsource(_MOD._two_way_effect_range_gauge_audit)\n    for token in ("5.0e307", "max_abs_entity_effect", "max_abs_reconstruction_error"):\n        assert token in source\n    main_source = inspect.getsource(_MOD.main)\n    assert '"two_way_effect_range_gauge": _two_way_effect_range_gauge_audit(backend)' in main_source\n'''
    p.write_text(text, encoding="utf-8")

# Record the numerical representation policy in the changelog.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
needle = "- **Panel fixed-effect prediction extreme-scale correctness**:"
idx = text.find(needle)
if idx < 0:
    raise RuntimeError("CHANGELOG FE numerical bullet not found")
line_end = text.find("\n", idx)
line = text[idx:line_end]
if "range-minimizing common-shift gauge" not in line:
    replacement = line + " Two-way additive-effect recovery conditionally uses a range-minimizing common-shift gauge near the float64 boundary so sparse incidence graphs cannot drive an otherwise representable entity/time decomposition out of range; ordinary-scale recovery preserves the historical weighted-zero gauge."
    text = text[:idx] + replacement + text[line_end:]
    p.write_text(text, encoding="utf-8")
