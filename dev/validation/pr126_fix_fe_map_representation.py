from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Store fitted FE maps directly on their level scale.  Centering a finite group
# mean by a finite grand mean can itself overflow even though the final level
# effect/prediction is representable.
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''            # Normalize only the compact recovered effect vector. Subtracting\n            # the grand mean observation-by-observation can erase a recoverable\n            # low-order group contribution beside very large residual levels.\n            ent_effects_dev = ent_effects_dev - float(grand_mean)\n            ent_effects = np.asarray(_to_numpy(ent_effects_dev)).ravel()\n''',
    '''            # Store the recovered level effects directly.  A centered\n            # representation ``effect - grand_mean`` can overflow even when both\n            # operands and the final level prediction are finite.\n            ent_effects = np.asarray(_to_numpy(ent_effects_dev)).ravel()\n''',
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''                    _compact_group_means(\n                        resid_orig, entity_projection, xp\n                    ) - float(grand_mean)\n''',
    '''                    _compact_group_means(\n                        resid_orig, entity_projection, xp\n                    )\n''',
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''                    _compact_group_means(\n                        resid_orig, time_projection, xp\n                    ) - float(grand_mean)\n''',
    '''                    _compact_group_means(\n                        resid_orig, time_projection, xp\n                    )\n''',
)

# Prediction now consumes level-scale maps directly.  _grand_mean remains stored
# for the established diagnostic/golden contract, but is not part of the map
# representation.
p = Path("statgpu/panel/_fixed_effects.py")
text = p.read_text(encoding="utf-8")
old = '''        n_rows = int(prediction.shape[0])\n        uses_fitted_effect = np.zeros(n_rows, dtype=bool)\n\n        def _effect_values(ids_np, mapping):\n'''
new = '''        def _effect_values(ids_np, mapping):\n'''
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise RuntimeError("predict effect tracking anchor not found")

text = text.replace("            uses_fitted_effect |= entity_known\n", "", 1)
text = text.replace("            uses_fitted_effect |= time_known\n", "", 1)
old_block = '''\n        # Fixed-effect maps are recovered after centering the level residual by\n        # its grand mean.  Whenever a row actually uses a stored fitted effect,\n        # restore that common level component exactly once.  Rows whose labels\n        # are wholly unseen preserve the documented linear-only fallback.\n        if np.any(uses_fitted_effect):\n            grand_mean = xp_asarray(\n                uses_fitted_effect.astype(np.float64) * float(self._grand_mean),\n                dtype=xp.float64,\n                xp=xp,\n                ref_arr=prediction,\n            )\n            prediction = prediction + grand_mean\n'''
new_block = '''\n        # Stored fixed-effect maps are already on the original level-residual\n        # scale.  Rows with unseen labels therefore retain the documented\n        # linear-only fallback without constructing a centered effect that may\n        # exceed float64 range.\n'''
if old_block in text:
    text = text.replace(old_block, new_block, 1)
elif new_block not in text:
    raise RuntimeError("predict grand-mean restoration block not found")
p.write_text(text, encoding="utf-8")

# Public NumPy/Torch regression for a finite prediction whose centered private
# representation would overflow.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_torch_cpu_fe_prediction_avoids_centered_map_overflow():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_torch_cpu_fe_prediction_avoids_centered_map_overflow():\n    amplitude = 1.0e308\n    entity = np.asarray([0] + [1] * 10, dtype=np.int64)\n    X = np.concatenate(([0.0], np.linspace(-1.0, 1.0, 10)))[:, None]\n    y = np.asarray([amplitude] + [-amplitude] * 10, dtype=np.float64)\n    expected = y.copy()\n\n    numpy_model = PanelOLS(entity_effects=True, cov_type=\"hc0\").fit(\n        X, y, entity_ids=entity\n    )\n    numpy_prediction = numpy_model.predict(X, entity_ids=entity)\n    assert np.all(np.isfinite(numpy_prediction))\n    np.testing.assert_allclose(numpy_prediction, expected, rtol=0.0, atol=0.0)\n    assert np.isfinite(numpy_model._grand_mean)\n    assert np.isfinite(numpy_model._entity_effects_map[0])\n\n    X_t = torch.as_tensor(X, dtype=torch.float64)\n    y_t = torch.as_tensor(y, dtype=torch.float64)\n    entity_t = torch.as_tensor(entity, dtype=torch.int64)\n    torch_model = PanelOLS(entity_effects=True, cov_type=\"hc0\").fit(\n        X_t, y_t, entity_ids=entity_t\n    )\n    torch_prediction = torch_model.predict(X_t, entity_ids=entity_t)\n    assert np.all(np.isfinite(torch_prediction))\n    np.testing.assert_allclose(torch_prediction, expected, rtol=0.0, atol=0.0)\n    assert np.isfinite(torch_model._entity_effects_map[0])\n'''
    p.write_text(text, encoding="utf-8")

# Physical CuPy/Torch audit.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
marker = "def _fixed_effect_map_range_audit(backend):"
if marker not in text:
    anchor = text.index("\ndef _nonfinite_covariance_guard_audit")
    audit = '''\n\ndef _fixed_effect_map_range_audit(backend):\n    amplitude = 1.0e308\n    entity = np.asarray([0] + [1] * 10, dtype=np.int64)\n    time = np.arange(11, dtype=np.int64)\n    X_np = np.concatenate(([0.0], np.linspace(-1.0, 1.0, 10)))[:, None]\n    y_np = np.asarray([amplitude] + [-amplitude] * 10, dtype=np.float64)\n    X, y, entity_b, _time_b = _to_backend(X_np, y_np, entity, time, backend)\n    model = PanelOLS(entity_effects=True, cov_type=\"hc0\", device=_device(backend)).fit(\n        X, y, entity_ids=entity_b\n    )\n    prediction = _array(model.predict(X, entity_ids=entity_b))\n    if not np.all(np.isfinite(prediction)):\n        raise AssertionError(f\"{backend}: FE map range audit produced non-finite prediction\")\n    np.testing.assert_allclose(\n        prediction, y_np, rtol=0.0, atol=0.0,\n        err_msg=f\"{backend}: FE map centered-range overflow\",\n    )\n    if not np.isfinite(float(model._entity_effects_map[0])):\n        raise AssertionError(f\"{backend}: FE map itself is non-finite\")\n    return {\n        \"status\": \"success\",\n        \"backend\": backend,\n        \"max_abs_prediction_error\": _max_abs(prediction, y_np),\n        \"grand_mean_finite\": bool(np.isfinite(float(model._grand_mean))),\n        \"effect_map_finite\": True,\n    }\n'''
    text = text[:anchor] + audit + text[anchor:]
old = '            "common_scale_product_range_guard": _common_scale_product_range_guard_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
new = '            "common_scale_product_range_guard": _common_scale_product_range_guard_audit(backend),\n            "fixed_effect_map_range": _fixed_effect_map_range_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
if new not in text:
    if old not in text:
        raise RuntimeError("physical FE map registry anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Hosted physical-runner contract.
p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_runner_registers_fixed_effect_map_range_audit():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_registers_fixed_effect_map_range_audit():\n    audit_source = inspect.getsource(_MOD._fixed_effect_map_range_audit)\n    for token in (\n        "1.0e308",\n        "FE map centered-range overflow",\n        "_entity_effects_map",\n        "effect_map_finite",\n    ):\n        assert token in audit_source\n    main_source = inspect.getsource(_MOD.main)\n    assert '\"fixed_effect_map_range\": _fixed_effect_map_range_audit(backend)' in main_source\n'''
    p.write_text(text, encoding="utf-8")

# Release note: clarify level-scale storage.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
old = "- **Panel fixed-effect prediction extreme-scale correctness**: one-way entity/time effect maps now recover stable group means from uncentered level residuals and apply the grand-mean normalization only on the compact effect vector, avoiding observation-level centering that can erase a recoverable low-order group contribution beside huge residuals. Two-way additive-effect recovery follows the same normalization rule, certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range, and defers its common-shift normalization until convergence; the final observation-weighted shift uses the stable reducer directly rather than multiplying huge effects by group counts. This keeps known-label `PanelOLS.predict()` level effects consistent with the already hardened within transformation across NumPy, CuPy, and Torch."
new = "- **Panel fixed-effect prediction extreme-scale correctness**: one-way entity/time effect maps recover stable group means from uncentered level residuals and are stored directly on the level-residual scale, avoiding both observation-level centering that can erase a recoverable tail and a centered compact effect whose subtraction from a finite grand mean can itself overflow. Two-way additive-effect recovery uses the same level-scale map representation, certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range, and defers its common-shift normalization until convergence; the final observation-weighted shift uses the stable reducer directly rather than multiplying huge effects by group counts. `_grand_mean` remains available for the established diagnostic contract but is no longer required to reconstruct a stored fitted effect in `PanelOLS.predict()`. This keeps known-label level predictions consistent with the hardened within transformation across NumPy, CuPy, and Torch."
if new not in text:
    if old not in text:
        raise RuntimeError("CHANGELOG FE map representation anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
