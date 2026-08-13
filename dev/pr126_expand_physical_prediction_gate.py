from pathlib import Path


def replace_exact(path, old, new, expected=1):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected}, found {count}: {old!r}")
    p.write_text(text.replace(old, new, expected), encoding="utf-8")


runner = "dev/benchmarks/validate_panel_stage_c_gpu.py"

replace_exact(
    runner,
    '''        "prediction": _optional_array(getattr(model, "_physical_prediction", None)),\n        "prediction_backend": getattr(model, "_predict_backend_name", None),''',
    '''        "prediction": _optional_array(getattr(model, "_physical_prediction", None)),\n        "prediction_backend": getattr(model, "_predict_backend_name", None),\n        "prediction_contract": getattr(model, "_physical_prediction_contract", None),''',
)

replace_exact(
    runner,
    '''    cases["panel_two_way_hc3"] = PanelOLS(\n        entity_effects=True, time_effects=True, cov_type="hc3", device=device\n    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)\n    cases["panel_two_way_cluster_group_debias"] = PanelOLS(''',
    '''    cases["panel_two_way_hc3"] = PanelOLS(\n        entity_effects=True, time_effects=True, cov_type="hc3", device=device\n    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)\n    cases["panel_two_way_hc3"]._physical_prediction = cases[\n        "panel_two_way_hc3"\n    ].predict(Xb[:8], entity_ids=eb[:8], time_ids=tb[:8])\n    cases["panel_two_way_hc3"]._physical_prediction_contract = (\n        "two_way_effect_prediction"\n    )\n    cases["panel_two_way_cluster_group_debias"] = PanelOLS(''',
)

replace_exact(
    runner,
    '''        if cov == "hc0":\n            cases[f"panel_entity_{cov}"]._physical_prediction = cases[\n                f"panel_entity_{cov}"\n            ].predict(Xb[:8], entity_ids=eb[:8])''',
    '''        if cov == "hc0":\n            cases[f"panel_entity_{cov}"]._physical_prediction = cases[\n                f"panel_entity_{cov}"\n            ].predict(Xb[:8], entity_ids=eb[:8])\n            cases[f"panel_entity_{cov}"]._physical_prediction_contract = (\n                "entity_effect_prediction"\n            )''',
)

replace_exact(
    runner,
    '''        if cov == "hc0":\n            cases[f"random_effects_explicit_constant_{cov}"]._physical_prediction = cases[\n                f"random_effects_explicit_constant_{cov}"\n            ].predict(Xcb[:8])''',
    '''        if cov == "hc0":\n            # Deliberately omit the fitted explicit constant.  This exercises the\n            # shared backend-native constant restoration path rather than only\n            # predicting with the already-complete design matrix.\n            cases[f"random_effects_explicit_constant_{cov}"]._physical_prediction = cases[\n                f"random_effects_explicit_constant_{cov}"\n            ].predict(Xb[:8])\n            cases[f"random_effects_explicit_constant_{cov}"]._physical_prediction_contract = (\n                "omitted_explicit_constant"\n            )''',
)

marker = '''\ndef _max_abs(actual, expected):\n'''
audit_fn = '''\ndef _disconnected_two_way_prediction_audit(backend):\n    """Exercise disconnected two-way prediction identifiability on one backend."""\n    rng = np.random.default_rng(20260820)\n    entity = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)\n    time = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)\n    X = rng.normal(size=(entity.size, 1)).astype(np.float64)\n    alpha = np.array([0.5, -0.2, 1.1, -0.7], dtype=np.float64)\n    tau = np.array([0.25, -0.15, 0.6, -0.4], dtype=np.float64)\n    y = (0.8 * X[:, 0] + alpha[entity] + tau[time]).astype(np.float64)\n    Xb, yb, eb, tb = _to_backend(X, y, entity, time, backend)\n    model = PanelOLS(\n        entity_effects=True, time_effects=True, cov_type="hc0", device=_device(backend)\n    ).fit(Xb, yb, entity_ids=eb, time_ids=tb)\n    executed = _backend_name(model)\n    if executed != backend:\n        raise AssertionError(\n            f"disconnected prediction audit requested {backend}, executed {executed}"\n        )\n\n    observed = model.predict(Xb, entity_ids=eb, time_ids=tb)\n    same_component = model.predict(\n        Xb[:1], entity_ids=np.array([1]), time_ids=np.array([1])\n    )\n\n    def guarded(label, **kwargs):\n        try:\n            model.predict(Xb[:1], **kwargs)\n        except ValueError as exc:\n            if "disconnected incidence graph" not in str(exc):\n                raise AssertionError(\n                    f"{label}: wrong disconnected-prediction failure: {exc}"\n                ) from exc\n            return True\n        raise AssertionError(f"{label}: disconnected prediction did not fail closed")\n\n    guards = {\n        "cross_component": guarded(\n            "cross_component", entity_ids=np.array([0]), time_ids=np.array([2])\n        ),\n        "entity_only": guarded("entity_only", entity_ids=np.array([0])),\n        "time_only": guarded("time_only", time_ids=np.array([0])),\n        "known_entity_unknown_time": guarded(\n            "known_entity_unknown_time",\n            entity_ids=np.array([0]),\n            time_ids=np.array([99]),\n        ),\n        "unknown_entity_known_time": guarded(\n            "unknown_entity_known_time",\n            entity_ids=np.array([99]),\n            time_ids=np.array([0]),\n        ),\n    }\n    both_unseen = model.predict(\n        Xb[:1], entity_ids=np.array([98]), time_ids=np.array([99])\n    )\n    prediction_backend = getattr(model, "_predict_backend_name", None)\n    if prediction_backend != backend:\n        raise AssertionError(\n            "disconnected prediction audit did not persist requested prediction backend: "\n            f"{prediction_backend!r} != {backend!r}"\n        )\n    return {\n        "executed_backend": executed,\n        "prediction_backend": prediction_backend,\n        "observed": _array(observed),\n        "same_component": _array(same_component),\n        "both_unseen": _array(both_unseen),\n        "guards": guards,\n    }\n\n'''
replace_exact(runner, marker, audit_fn + marker)

replace_exact(
    runner,
    '''    for field in ("coefficient_inference_applicable", "coefficient_inference_reason", "prediction_backend"):\n''',
    '''    for field in (\n        "coefficient_inference_applicable",\n        "coefficient_inference_reason",\n        "prediction_backend",\n        "prediction_contract",\n    ):\n''',
)

replace_exact(
    runner,
    '''    reference_models = _fit_cases(X, y, entity, time, clusters, "numpy")\n    reference = {name: _snapshot(model) for name, model in reference_models.items()}\n    primitive_reference = {''',
    '''    reference_models = _fit_cases(X, y, entity, time, clusters, "numpy")\n    reference = {name: _snapshot(model) for name, model in reference_models.items()}\n    prediction_reference = _disconnected_two_way_prediction_audit("numpy")\n    primitive_reference = {''',
)

replace_exact(
    runner,
    '''        payload = {\n            "status": "success",\n            "requested_backend": backend,\n            "cases": {},\n            "public_primitives": {},\n        }''',
    '''        payload = {\n            "status": "success",\n            "requested_backend": backend,\n            "cases": {},\n            "public_primitives": {},\n            "prediction_contracts": {},\n        }''',
)

replace_exact(
    runner,
    '''            if name in {\n                "panel_entity_hc0",\n                "random_effects_explicit_constant_hc0",\n            } and snapshot["prediction_backend"] != backend:''',
    '''            if name in {\n                "panel_entity_hc0",\n                "panel_two_way_hc3",\n                "random_effects_explicit_constant_hc0",\n            } and snapshot["prediction_backend"] != backend:''',
)

replace_exact(
    runner,
    '''                "prediction_backend": snapshot["prediction_backend"],\n            }\n        primitive_values = _public_primitive_cases(''',
    '''                "prediction_backend": snapshot["prediction_backend"],\n                "prediction_contract": snapshot["prediction_contract"],\n            }\n\n        prediction_audit = _disconnected_two_way_prediction_audit(backend)\n        if prediction_audit["executed_backend"] != backend:\n            raise AssertionError("disconnected prediction fit backend provenance drifted")\n        if prediction_audit["prediction_backend"] != backend:\n            raise AssertionError("disconnected prediction execution backend provenance drifted")\n        if not all(prediction_audit["guards"].values()):\n            raise AssertionError("disconnected prediction guard audit did not fail closed")\n        prediction_diffs = {}\n        for field in ("observed", "same_component", "both_unseen"):\n            np.testing.assert_allclose(\n                prediction_audit[field],\n                prediction_reference[field],\n                rtol=args.rtol,\n                atol=args.atol,\n                err_msg=f"two_way_disconnected_prediction.{field}",\n            )\n            prediction_diffs[field] = _max_abs(\n                prediction_audit[field], prediction_reference[field]\n            )\n        payload["prediction_contracts"]["two_way_disconnected"] = {\n            "status": "success",\n            "executed_backend": prediction_audit["executed_backend"],\n            "prediction_backend": prediction_audit["prediction_backend"],\n            "guards": dict(prediction_audit["guards"]),\n            "max_abs_differences": prediction_diffs,\n        }\n\n        primitive_values = _public_primitive_cases(''',
)

# Hosted regression: execute the physical prediction audit on NumPy and ensure
# the fixed 35-case runner now labels the three intended prediction contracts.
test_path = "dev/tests/test_panel_stage_c_final_review_fixes.py"
append_marker = '''\ndef test_torch_cpu_two_way_projection_and_prediction_match_numpy():\n'''
new_test = '''\ndef test_physical_stage_c_runner_covers_new_prediction_contracts_on_numpy():\n    import importlib.util\n    from pathlib import Path\n\n    runner_path = Path(__file__).parents[1] / "benchmarks" / "validate_panel_stage_c_gpu.py"\n    spec = importlib.util.spec_from_file_location("stage_c_gpu_validation_review", runner_path)\n    assert spec is not None and spec.loader is not None\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n\n    X, y, entity, time, clusters = module._dataset()\n    models = module._fit_cases(X, y, entity, time, clusters, "numpy")\n    assert models["panel_entity_hc0"]._physical_prediction_contract == (\n        "entity_effect_prediction"\n    )\n    assert models["panel_two_way_hc3"]._physical_prediction_contract == (\n        "two_way_effect_prediction"\n    )\n    re_model = models["random_effects_explicit_constant_hc0"]\n    assert re_model._physical_prediction_contract == "omitted_explicit_constant"\n    assert re_model._predict_constant_index == 0\n    assert_allclose(\n        re_model._physical_prediction,\n        re_model.predict(X[:8]),\n        rtol=0,\n        atol=3e-12,\n    )\n\n    audit = module._disconnected_two_way_prediction_audit("numpy")\n    assert audit["executed_backend"] == "numpy"\n    assert audit["prediction_backend"] == "numpy"\n    assert all(audit["guards"].values())\n    assert np.all(np.isfinite(audit["observed"]))\n    assert np.all(np.isfinite(audit["same_component"]))\n    assert np.all(np.isfinite(audit["both_unseen"]))\n\n'''
replace_exact(test_path, append_marker, new_test + append_marker)
