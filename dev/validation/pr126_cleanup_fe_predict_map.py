from pathlib import Path

p = Path("statgpu/panel/_fixed_effects.py")
text = p.read_text(encoding="utf-8")
old = '''        def _effect_values(ids_np, mapping):\n            if not mapping or ids_np is None:\n                return None, None\n            known = np.fromiter(\n                (value in mapping for value in ids_np),\n                dtype=bool,\n                count=ids_np.shape[0],\n            )\n            values = np.fromiter(\n                (float(mapping.get(value, 0.0)) for value in ids_np),\n                dtype=np.float64,\n                count=ids_np.shape[0],\n            )\n            return (\n                xp_asarray(values, dtype=xp.float64, xp=xp, ref_arr=prediction),\n                known,\n            )\n\n        entity_effect, entity_known = _effect_values(\n            entity_ids_np, self._entity_effects_map\n        )\n        if entity_effect is not None:\n            prediction = prediction + entity_effect\n        time_effect, time_known = _effect_values(\n            time_ids_np, self._time_effects_map\n        )\n        if time_effect is not None:\n            prediction = prediction + time_effect\n'''
new = '''        def _effect_values(ids_np, mapping):\n            if not mapping or ids_np is None:\n                return None\n            values = np.fromiter(\n                (float(mapping.get(value, 0.0)) for value in ids_np),\n                dtype=np.float64,\n                count=ids_np.shape[0],\n            )\n            return xp_asarray(\n                values, dtype=xp.float64, xp=xp, ref_arr=prediction\n            )\n\n        entity_effect = _effect_values(entity_ids_np, self._entity_effects_map)\n        if entity_effect is not None:\n            prediction = prediction + entity_effect\n        time_effect = _effect_values(time_ids_np, self._time_effects_map)\n        if time_effect is not None:\n            prediction = prediction + time_effect\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("FE prediction effect-value anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Lock that the obsolete known-mask scan is not reintroduced now that map values
# are already level-scale.
p = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = p.read_text(encoding="utf-8")
marker = "def test_panel_predict_level_effects_do_not_build_obsolete_known_mask():"
if marker not in text:
    text += '''\n\n\ndef test_panel_predict_level_effects_do_not_build_obsolete_known_mask():\n    import inspect\n    from statgpu.panel import PanelOLS\n\n    source = inspect.getsource(PanelOLS.predict)\n    assert "uses_fitted_effect" not in source\n    assert "entity_known" not in source\n    assert "time_known" not in source\n    assert "value in mapping for value in ids_np" not in source\n'''
    p.write_text(text, encoding="utf-8")
