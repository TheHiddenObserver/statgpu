from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Normalize two-way effects only once after ALS convergence.  The normalization
# is algebraically irrelevant to the fitted sum during iteration, while the old
# count-weighted product could overflow before a finite cancellation.
replace_once(
    "statgpu/panel/_utils.py",
    '''from statgpu.panel._reductions import (\n    stable_group_means_preindexed,\n    stable_reduction_flags,\n)''',
    '''from statgpu.panel._reductions import (\n    stable_group_means_preindexed,\n    stable_mean,\n    stable_reduction_flags,\n)''',
)

replace_once(
    "statgpu/panel/_utils.py",
    '''        shift = xp.sum(time_effects * t_counts) / float(values.shape[0])\n        time_effects = time_effects - shift\n        entity_effects = entity_effects + shift\n\n        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n''',
    '''        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n''',
)

replace_once(
    "statgpu/panel/_utils.py",
    '''    if not converged:\n        raise RuntimeError(\n            "two-way fixed-effect recovery did not converge within "\n            f"max_iter={int(max_iter)}; final normalized group-mean violation="\n            f"{final_metric:.6e}"\n        )\n    return entity_effects, time_effects\n''',
    '''    if not converged:\n        raise RuntimeError(\n            "two-way fixed-effect recovery did not converge within "\n            f"max_iter={int(max_iter)}; final normalized group-mean violation="\n            f"{final_metric:.6e}"\n        )\n\n    # The entity/time decomposition has a common-shift null direction.  Apply\n    # the public normalization once, after convergence, so the iterative path\n    # never materializes ``time_effect * group_count``.  Expanding by ``t_idx``\n    # expresses the observation-weighted mean directly and lets the shared\n    # stable reducer preserve cancellation without an overflowing product.\n    shift = stable_mean(time_effects[t_idx], xp)\n    time_effects = time_effects - shift\n    entity_effects = entity_effects + shift\n    return entity_effects, time_effects\n''',
)

# The tuple no longer needs t_counts after normalization is deferred.
replace_once(
    "statgpu/panel/_utils.py",
    '''    t_idx, n_times, _t_labels, t_counts, _t_inv, _t_codes_np = time_projection\n''',
    '''    t_idx, n_times, _t_labels, _t_counts, _t_inv, _t_codes_np = time_projection\n''',
)

# Hosted Torch CPU regression.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_torch_cpu_two_way_effect_normalization_avoids_weight_overflow():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_torch_cpu_two_way_effect_normalization_avoids_weight_overflow():\n    amplitude = 1.0e308\n    entity = np.repeat(np.arange(2, dtype=np.int64), 4)\n    time = np.tile(np.arange(4, dtype=np.int64), 2)\n    values_np = np.tile(\n        np.asarray([amplitude, amplitude, -amplitude, -amplitude], dtype=np.float64),\n        2,\n    )\n    values = torch.as_tensor(values_np, dtype=torch.float64)\n    entity_t = torch.as_tensor(entity, dtype=torch.int64)\n    time_t = torch.as_tensor(time, dtype=torch.int64)\n\n    entity_effect, time_effect = _recover_two_way_effects(\n        values, entity_t, time_t, torch, max_iter=20, tol=1e-12\n    )\n    reconstructed = entity_effect[entity_t] + time_effect[time_t]\n    assert bool(torch.all(torch.isfinite(entity_effect)))\n    assert bool(torch.all(torch.isfinite(time_effect)))\n    assert bool(torch.all(torch.isfinite(reconstructed)))\n    np.testing.assert_allclose(\n        reconstructed.detach().cpu().numpy(), values_np, rtol=0.0, atol=0.0\n    )\n'''
    p.write_text(text, encoding="utf-8")

# Physical CuPy/Torch audit.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
marker = "def _two_way_effect_normalization_overflow_audit(backend):"
if marker not in text:
    anchor = text.index("\ndef _nonfinite_covariance_guard_audit")
    audit = '''\n\ndef _two_way_effect_normalization_overflow_audit(backend):\n    if backend == "numpy":\n        xp = np\n    elif backend == "cupy":\n        import cupy as cp\n        xp = cp\n    elif backend == "torch":\n        import torch\n        xp = torch\n    else:\n        raise ValueError(backend)\n\n    amplitude = 1.0e308\n    entity = np.repeat(np.arange(2, dtype=np.int64), 4)\n    time = np.tile(np.arange(4, dtype=np.int64), 2)\n    values_np = np.tile(\n        np.asarray([amplitude, amplitude, -amplitude, -amplitude], dtype=np.float64),\n        2,\n    )\n    dummy_X = np.arange(8, dtype=np.float64)[:, None]\n    _X, values, entity_b, time_b = _to_backend(\n        dummy_X, values_np, entity, time, backend\n    )\n    entity_effect, time_effect = _recover_two_way_effects(\n        values, entity_b, time_b, xp, max_iter=20, tol=1e-12\n    )\n    reconstructed = _array(entity_effect[entity_b] + time_effect[time_b])\n    entity_np = _array(entity_effect)\n    time_np = _array(time_effect)\n    if not np.all(np.isfinite(entity_np)) or not np.all(np.isfinite(time_np)):\n        raise AssertionError(f"{backend}: two-way FE normalization produced non-finite effects")\n    np.testing.assert_allclose(\n        reconstructed, values_np, rtol=0.0, atol=0.0,\n        err_msg=f"{backend}: two-way FE normalization overflow audit",\n    )\n    return {\n        "status": "success",\n        "backend": backend,\n        "max_abs_reconstruction_error": _max_abs(reconstructed, values_np),\n    }\n'''
    text = text[:anchor] + audit + text[anchor:]

old = '            "fixed_effect_recovery_cancellation": _fixed_effect_recovery_cancellation_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
new = '            "fixed_effect_recovery_cancellation": _fixed_effect_recovery_cancellation_audit(backend),\n            "two_way_effect_normalization_overflow": _two_way_effect_normalization_overflow_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
if new not in text:
    if old not in text:
        raise RuntimeError("physical normalization audit registry anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Hosted physical-runner registration contract.
p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_runner_registers_two_way_effect_normalization_overflow_audit():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_registers_two_way_effect_normalization_overflow_audit():\n    audit_source = inspect.getsource(_MOD._two_way_effect_normalization_overflow_audit)\n    for token in (\n        "1.0e308",\n        "_recover_two_way_effects",\n        "two-way FE normalization overflow audit",\n    ):\n        assert token in audit_source\n    main_source = inspect.getsource(_MOD.main)\n    assert (\n        '\"two_way_effect_normalization_overflow\": _two_way_effect_normalization_overflow_audit(backend)'\n        in main_source\n    )\n'''
    p.write_text(text, encoding="utf-8")

# Release note.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
old = "Two-way additive-effect recovery follows the same normalization rule and certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range."
new = "Two-way additive-effect recovery follows the same normalization rule, certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range, and defers its common-shift normalization until convergence; the final observation-weighted shift uses the stable reducer directly rather than multiplying huge effects by group counts."
if new not in text:
    if old not in text:
        raise RuntimeError("CHANGELOG normalization anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
