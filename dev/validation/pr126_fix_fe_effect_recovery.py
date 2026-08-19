from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1. Reuse the stable group-mean primitive when storing one-way FE maps.
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''from statgpu.panel._utils import (\n    _recover_two_way_effects,\n    _scatter_add,\n    demean_variables,\n    factorize_panel_labels,\n)''',
    '''from statgpu.panel._utils import (\n    _compact_group_means,\n    _prepare_group_projection,\n    _recover_two_way_effects,\n    demean_variables,\n    factorize_panel_labels,\n)''',
)

replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''        else:\n            if self.entity_effects and entity_arr is not None:\n                ent_sums = _scatter_add(\n                    xp, entity_arr, resid_centered, len(entity_labels)\n                )\n                ent_counts = _scatter_add(\n                    xp,\n                    entity_arr,\n                    xp.ones_like(resid_centered),\n                    len(entity_labels),\n                )\n                ent_effects = _to_numpy(\n                    ent_sums / xp_maximum(ent_counts, 1.0, xp)\n                ).ravel()\n                for i, eid in enumerate(entity_labels):\n                    self._entity_effects_map[eid] = float(ent_effects[i])\n\n            if self.time_effects and time_arr is not None:\n                time_sums = _scatter_add(\n                    xp, time_arr, resid_centered, len(time_labels)\n                )\n                time_counts = _scatter_add(\n                    xp,\n                    time_arr,\n                    xp.ones_like(resid_centered),\n                    len(time_labels),\n                )\n                time_effect_values = _to_numpy(\n                    time_sums / xp_maximum(time_counts, 1.0, xp)\n                ).ravel()\n                for i, tid in enumerate(time_labels):\n                    self._time_effects_map[tid] = float(time_effect_values[i])\n''',
    '''        else:\n            if self.entity_effects and entity_arr is not None:\n                entity_projection = _prepare_group_projection(entity_arr, xp)\n                ent_effects = _to_numpy(\n                    _compact_group_means(\n                        resid_centered, entity_projection, xp\n                    )\n                ).ravel()\n                for i, eid in enumerate(entity_labels):\n                    self._entity_effects_map[eid] = float(ent_effects[i])\n\n            if self.time_effects and time_arr is not None:\n                time_projection = _prepare_group_projection(time_arr, xp)\n                time_effect_values = _to_numpy(\n                    _compact_group_means(\n                        resid_centered, time_projection, xp\n                    )\n                ).ravel()\n                for i, tid in enumerate(time_labels):\n                    self._time_effects_map[tid] = float(time_effect_values[i])\n''',
)

# 2. Certify two-way FE recovery with the stable reducer before accepting a
# fast-path convergence decision.  This adds one packed risk check after the
# first full ALS cycle and, when still on the fast path, one final certification
# only when the fast metric says it is ready to stop.
p = Path("statgpu/panel/_utils.py")
text = p.read_text(encoding="utf-8")
start = text.index("def _recover_two_way_effects(")
end = text.index("\n\ndef group_means", start)
new_func = '''def _recover_two_way_effects(\n    values,\n    entity_ids,\n    time_ids,\n    xp,\n    *,\n    max_iter=1_000_000,\n    tol=1e-10,\n):\n    \"\"\"Recover additive two-way effects by backend-native alternating least squares.\n\n    The returned compact entity/time effects reproduce the joint least-squares\n    projection on observed cells.  The time effects are normalized to have zero\n    observation-weighted mean, with the compensating shift applied to entity\n    effects so fitted values are unchanged.  Ordinary-scale iterations retain\n    the fast group scatter path, but convergence is certified with the shared\n    cancellation-safe reducer whenever a projection creates new dynamic range.\n    \"\"\"\n    values = xp_asarray(values, dtype=xp.float64, xp=xp).ravel()\n    entity_projection = _prepare_group_projection(entity_ids, xp)\n    time_projection = _prepare_group_projection(time_ids, xp)\n    e_idx, n_entities, _e_labels, _e_counts, _e_inv, _e_codes_np = entity_projection\n    t_idx, n_times, _t_labels, t_counts, _t_inv, _t_codes_np = time_projection\n    entity_effects = xp_zeros(n_entities, xp.float64, xp, values)\n    time_effects = xp_zeros(n_times, xp.float64, xp, values)\n    level_scale = xp.max(xp.abs(values))\n    stable = bool(stable_reduction_flags(values, xp)[0])\n\n    def _violation(residual, *, stable_mode):\n        entity_means = _compact_group_means(\n            residual, entity_projection, xp, stable=stable_mode\n        )\n        time_means = _compact_group_means(\n            residual, time_projection, xp, stable=stable_mode\n        )\n        return xp.maximum(\n            xp.max(xp.abs(entity_means)), xp.max(xp.abs(time_means))\n        )\n\n    def _metric(residual, violation):\n        residual_scale = xp.max(xp.abs(residual))\n        allowance = _convergence_allowance(\n            residual_scale, level_scale, tol, xp\n        )\n        return _to_float_scalar(\n            violation\n            / xp_maximum(allowance, np.finfo(np.float64).tiny, xp)\n        )\n\n    converged = False\n    final_metric = float(\"inf\")\n    for _iteration in range(int(max_iter)):\n        entity_effects = _compact_group_means(\n            values - time_effects[t_idx], entity_projection, xp, stable=stable\n        )\n        time_effects = _compact_group_means(\n            values - entity_effects[e_idx], time_projection, xp, stable=stable\n        )\n\n        shift = xp.sum(time_effects * t_counts) / float(values.shape[0])\n        time_effects = time_effects - shift\n        entity_effects = entity_effects + shift\n\n        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n\n        # The first full pair of projections can create a cancellation scale\n        # absent from the level values.  Classify that transformed residual once\n        # so later ALS updates use the stable reducer immediately when needed.\n        if _iteration == 0 and not stable:\n            stable = bool(stable_reduction_flags(residual, xp)[0])\n\n        violation = _violation(residual, stable_mode=stable)\n        final_metric = _metric(residual, violation)\n        if final_metric <= 1.0:\n            if not stable:\n                # A late cancellation can emerge only after several alternating\n                # projections.  Before accepting fast-path convergence, perform\n                # one packed risk classification and recompute the stopping\n                # criterion with the stable reducer if that risk is present.\n                certify_stable = bool(stable_reduction_flags(residual, xp)[0])\n                if certify_stable:\n                    stable = True\n                    violation = _violation(residual, stable_mode=True)\n                    final_metric = _metric(residual, violation)\n                    if final_metric > 1.0:\n                        continue\n            converged = True\n            break\n\n    if not converged:\n        raise RuntimeError(\n            \"two-way fixed-effect recovery did not converge within \"\n            f\"max_iter={int(max_iter)}; final normalized group-mean violation=\"\n            f\"{final_metric:.6e}\"\n        )\n    return entity_effects, time_effects\n'''
text = text[:start] + new_func + text[end:]
p.write_text(text, encoding="utf-8")

# 3. Hosted Torch/NumPy regressions: public one-way prediction and two-way
# projection-created recovery.
replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    'from statgpu.panel._utils import _zero_safe_statistic_ratio',
    '''from statgpu.panel._utils import (\n    _recover_two_way_effects,\n    _zero_safe_statistic_ratio,\n    demean_variables,\n)''',
)

test_path = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
test_text = test_path.read_text(encoding="utf-8")
marker = "def test_stage_c_torch_cpu_fe_effect_recovery_preserves_cancellation_tail():"
if marker not in test_text:
    test_text += '''\n\n\ndef test_stage_c_torch_cpu_fe_effect_recovery_preserves_cancellation_tail():\n    # Public one-way prediction: the within fit is cancellation-safe, so the\n    # stored FE map must not reintroduce a raw scatter-add that loses the +1 tail.\n    amplitude = float(2.0 ** 55)\n    entity = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)\n    X = np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0], dtype=np.float64)[:, None]\n    y = np.asarray([amplitude, 1.0, -amplitude, 0.0, 0.0, 0.0], dtype=np.float64)\n    expected = np.asarray([1.0 / 3.0] * 3 + [0.0] * 3, dtype=np.float64)\n\n    numpy_model = PanelOLS(entity_effects=True, cov_type=\"hc0\").fit(\n        X, y, entity_ids=entity\n    )\n    np.testing.assert_allclose(\n        numpy_model.predict(X, entity_ids=entity), expected, rtol=0.0, atol=2e-15\n    )\n\n    X_t = torch.as_tensor(X, dtype=torch.float64)\n    y_t = torch.as_tensor(y, dtype=torch.float64)\n    entity_t = torch.as_tensor(entity, dtype=torch.int64)\n    torch_model = PanelOLS(entity_effects=True, cov_type=\"hc0\").fit(\n        X_t, y_t, entity_ids=entity_t\n    )\n    np.testing.assert_allclose(\n        torch_model.predict(X_t, entity_ids=entity_t),\n        expected,\n        rtol=0.0,\n        atol=2e-15,\n    )\n\n    # Two-way recovery: raw values are all the same order, but the first entity\n    # projection creates +/-1 beside +/-2**49.  The recovered FE residual must\n    # match the already hardened two-way within projection on the low-order rows.\n    level = float(2.0 ** 50)\n    values_np = np.asarray(\n        [1.5 * level, 0.5 * level, level + 1.0, level - 1.0,\n         0.5 * level, 1.5 * level],\n        dtype=np.float64,\n    )\n    entity2 = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)\n    time2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)\n    values_t = torch.as_tensor(values_np, dtype=torch.float64)\n    entity2_t = torch.as_tensor(entity2, dtype=torch.int64)\n    time2_t = torch.as_tensor(time2, dtype=torch.int64)\n    dummy_X_t = torch.arange(6, dtype=torch.float64)[:, None]\n    reference, _ = demean_variables(\n        values_t, dummy_X_t, entity2_t, time2_t, xp=torch, max_iter=200, tol=1e-12\n    )\n    entity_effect, time_effect = _recover_two_way_effects(\n        values_t, entity2_t, time2_t, torch, max_iter=200, tol=1e-12\n    )\n    recovered_residual = values_t - entity_effect[entity2_t] - time_effect[time2_t]\n    np.testing.assert_allclose(\n        recovered_residual[2:4].detach().cpu().numpy(),\n        reference[2:4].detach().cpu().numpy(),\n        rtol=0.0,\n        atol=2e-12,\n    )\n'''
    test_path.write_text(test_text, encoding="utf-8")

# 4. Register the same behavior in the physical CuPy/Torch validator.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''from statgpu.panel._utils import (\n    _zero_safe_statistic_ratio,\n    demean_variables,\n    within_transform,\n)''',
    '''from statgpu.panel._utils import (\n    _recover_two_way_effects,\n    _zero_safe_statistic_ratio,\n    demean_variables,\n    within_transform,\n)''',
)

runner = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
runner_text = runner.read_text(encoding="utf-8")
audit_marker = "def _fixed_effect_recovery_cancellation_audit(backend):"
if audit_marker not in runner_text:
    insert_at = runner_text.index("\ndef _nonfinite_covariance_guard_audit", runner_text.index("def _projection_created_dynamic_range_audit"))
    audit = '''\n\ndef _fixed_effect_recovery_cancellation_audit(backend):\n    \"\"\"Keep public FE prediction maps on cancellation-safe group means.\"\"\"\n    if backend == \"numpy\":\n        xp = np\n    elif backend == \"cupy\":\n        import cupy as cp\n        xp = cp\n    elif backend == \"torch\":\n        import torch\n        xp = torch\n    else:\n        raise ValueError(backend)\n\n    amplitude = float(2.0 ** 55)\n    entity = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)\n    time = np.arange(6, dtype=np.int64)\n    X_np = np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0], dtype=np.float64)[:, None]\n    y_np = np.asarray([amplitude, 1.0, -amplitude, 0.0, 0.0, 0.0], dtype=np.float64)\n    X, y, entity_b, _time_b = _to_backend(X_np, y_np, entity, time, backend)\n    model = PanelOLS(entity_effects=True, cov_type=\"hc0\", device=_device(backend)).fit(\n        X, y, entity_ids=entity_b\n    )\n    prediction = _array(model.predict(X, entity_ids=entity_b))\n    expected = np.asarray([1.0 / 3.0] * 3 + [0.0] * 3, dtype=np.float64)\n    np.testing.assert_allclose(\n        prediction, expected, rtol=0.0, atol=2e-15,\n        err_msg=f\"{backend}: one-way FE prediction cancellation tail\",\n    )\n    if _backend_name(model) != backend or getattr(model, \"_predict_backend_name\", None) != backend:\n        raise AssertionError(f\"{backend}: FE prediction backend provenance drifted\")\n\n    level = float(2.0 ** 50)\n    values_np = np.asarray(\n        [1.5 * level, 0.5 * level, level + 1.0, level - 1.0,\n         0.5 * level, 1.5 * level],\n        dtype=np.float64,\n    )\n    entity2 = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)\n    time2 = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)\n    dummy_X = np.arange(6, dtype=np.float64)[:, None]\n    X2, values, entity2_b, time2_b = _to_backend(\n        dummy_X, values_np, entity2, time2, backend\n    )\n    reference, _ = demean_variables(\n        values, X2, entity2_b, time2_b, xp=xp, max_iter=200, tol=1e-12\n    )\n    entity_effect, time_effect = _recover_two_way_effects(\n        values, entity2_b, time2_b, xp, max_iter=200, tol=1e-12\n    )\n    recovered = values - entity_effect[entity2_b] - time_effect[time2_b]\n    recovered_np = _array(recovered)\n    reference_np = _array(reference)\n    np.testing.assert_allclose(\n        recovered_np[2:4], reference_np[2:4], rtol=0.0, atol=2e-12,\n        err_msg=f\"{backend}: two-way FE recovery projection-created risk\",\n    )\n    return {\n        \"status\": \"success\",\n        \"backend\": backend,\n        \"prediction_backend\": getattr(model, \"_predict_backend_name\", None),\n        \"max_abs_prediction_error\": _max_abs(prediction, expected),\n        \"max_abs_two_way_low_order_error\": _max_abs(\n            recovered_np[2:4], reference_np[2:4]\n        ),\n    }\n'''
    runner_text = runner_text[:insert_at] + audit + runner_text[insert_at:]

old_registry = '            "projection_created_dynamic_range": _projection_created_dynamic_range_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
new_registry = '            "projection_created_dynamic_range": _projection_created_dynamic_range_audit(backend),\n            "fixed_effect_recovery_cancellation": _fixed_effect_recovery_cancellation_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
if new_registry not in runner_text:
    if old_registry not in runner_text:
        raise RuntimeError("physical audit registry anchor not found")
    runner_text = runner_text.replace(old_registry, new_registry, 1)
runner.write_text(runner_text, encoding="utf-8")

contract = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
contract_text = contract.read_text(encoding="utf-8")
contract_marker = "def test_stage_c_runner_registers_fe_effect_recovery_gpu_audit():"
if contract_marker not in contract_text:
    contract_text += '''\n\n\ndef test_stage_c_runner_registers_fe_effect_recovery_gpu_audit():\n    audit_source = inspect.getsource(_MOD._fixed_effect_recovery_cancellation_audit)\n    for token in (\n        \"PanelOLS\",\n        \"_recover_two_way_effects\",\n        \"demean_variables\",\n        \"one-way FE prediction cancellation tail\",\n        \"two-way FE recovery projection-created risk\",\n    ):\n        assert token in audit_source\n    main_source = inspect.getsource(_MOD.main)\n    assert (\n        '\"fixed_effect_recovery_cancellation\": _fixed_effect_recovery_cancellation_audit(backend)'\n        in main_source\n    )\n'''
    contract.write_text(contract_text, encoding="utf-8")

# 5. Record the prediction-level numerical fix in the release note.
changelog = Path("CHANGELOG.md")
changelog_text = changelog.read_text(encoding="utf-8")
needle = "- **Panel diagnostics extreme-scale correctness**:"
addition = "- **Panel fixed-effect prediction extreme-scale correctness**: one-way entity/time effect maps now reuse the shared cancellation-safe group-mean reducer instead of raw scatter sums, and two-way additive-effect recovery certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range. This keeps known-label `PanelOLS.predict()` level effects consistent with the already hardened within transformation across NumPy, CuPy, and Torch.\n"
if addition not in changelog_text:
    if needle not in changelog_text:
        raise RuntimeError("CHANGELOG insertion anchor not found")
    changelog_text = changelog_text.replace(needle, addition + needle, 1)
    changelog.write_text(changelog_text, encoding="utf-8")
