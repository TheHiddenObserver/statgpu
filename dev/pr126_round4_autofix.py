from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one literal match, got {text.count(old)}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def regex_once(path: str, pattern: str, repl: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    new, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one regex match, got {count}")
    p.write_text(new, encoding="utf-8")


# 1) Two-way FE alternating projection: convergence must cover both y and X,
# and max_iter exhaustion must fail closed.
regex_once(
    "statgpu/panel/_utils.py",
    r"    if time_ids is not None:\n        for _iteration in range\(max_iter\):.*?\n\n    return y_d, X_d",
    '''    if time_ids is not None:\n        converged = False\n        max_change = float("inf")\n        for _iteration in range(max_iter):\n            y_d_old = y_d.copy() if hasattr(y_d, "copy") else y_d.clone()\n            X_d_old = X_d.copy() if hasattr(X_d, "copy") else X_d.clone()\n            if entity_ids is not None:\n                y_d = within_transform(y_d, entity_ids, xp)\n                X_d = _within_transform_matrix(X_d, entity_ids, xp)\n            y_d = within_transform(y_d, time_ids, xp)\n            X_d = _within_transform_matrix(X_d, time_ids, xp)\n            y_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))\n            X_change = _to_float_scalar(xp.max(xp.abs(X_d - X_d_old)))\n            max_change = max(float(y_change), float(X_change))\n            if max_change < tol:\n                converged = True\n                break\n        if not converged:\n            raise RuntimeError(\n                "two-way fixed-effect demeaning did not converge within "\n                f"max_iter={max_iter}; final max change={max_change:.6e}, tol={tol:.6e}"\n            )\n\n    return y_d, X_d''',
)

# 2) Rank-deficient fits remain estimable in the identified fit space, but
# ordinary coordinate-wise coefficient inference is unavailable.
replace_once(
    "statgpu/panel/_base.py",
    "        from statgpu.inference._results import ParameterInferenceResult\n",
    "        from statgpu.inference._results import BaseInferenceResult, ParameterInferenceResult\n",
)
replace_once(
    "statgpu/panel/_base.py",
    "        hc1_correction=None,\n        distribution_df=None,\n        diag_floor=1e-30,\n",
    "        hc1_correction=None,\n        distribution_df=None,\n        fit_rank=None,\n        diag_floor=1e-30,\n",
)
replace_once(
    "statgpu/panel/_base.py",
    '''        if not np.all(np.isfinite(cov_np)):\n            raise ValueError(\n                "covariance contains non-finite values; inference is not numerically valid"\n            )\n        diag_np = np.diag(cov_np).astype(np.float64, copy=False)\n''',
    '''        if not np.all(np.isfinite(cov_np)):\n            raise ValueError(\n                "covariance contains non-finite values; inference is not numerically valid"\n            )\n\n        if fit_rank is None:\n            from statgpu.panel._linalg import panel_matrix_rank\n\n            fit_rank = panel_matrix_rank(X, xp)\n        fit_rank = int(fit_rank)\n        design_columns = int(X.shape[1])\n        if fit_rank <= 0 or fit_rank > design_columns:\n            raise ValueError(\n                "fit_rank must identify a positive subspace no larger than the design"\n            )\n        rank_deficient = fit_rank < design_columns\n        self._coefficient_inference_available = not rank_deficient\n        self._coefficient_inference_reason = None\n        self._covariance_metadata["design_rank"] = fit_rank\n        self._covariance_metadata["design_columns"] = design_columns\n        self._covariance_metadata["coefficient_inference_applicable"] = not rank_deficient\n\n        if rank_deficient:\n            reason = (\n                "coefficient-level inference is unavailable because the fit-space design "\n                f"is rank deficient (rank={fit_rank}, columns={design_columns}); "\n                "fitted values and identified fit-space quantities remain available"\n            )\n            self._coefficient_inference_reason = reason\n            self._covariance_metadata["coefficient_inference_reason"] = reason\n            self.coef_ = np.asarray(_to_numpy(params), dtype=np.float64).ravel()\n            self.bse_ = None\n            self.tvalues_ = None\n            self.pvalues_ = None\n            self.conf_int_ = None\n            self._params = self.coef_.copy()\n            self._bse = None\n            self._tvalues = None\n            self._zvalues = None\n            self._pvalues = None\n            self._conf_int = None\n            feature_names = getattr(self, "_feature_names", None)\n            if feature_names is not None and len(feature_names) != len(self.coef_):\n                feature_names = None\n            BaseInferenceResult(\n                method="panel_ols_unavailable",\n                feature_names=feature_names,\n                metadata={\n                    "applicable": False,\n                    "reason": reason,\n                    "covariance": dict(covariance_metadata),\n                    "fit_rank": fit_rank,\n                    "design_columns": design_columns,\n                },\n            ).apply_to(self)\n            return cov_params\n\n        diag_np = np.diag(cov_np).astype(np.float64, copy=False)\n''',
)
replace_once(
    "statgpu/panel/_base.py",
    '''        self._check_is_fitted()\n        from statgpu.panel._formula import _get_feature_names\n\n        coef_np = np.asarray(_to_numpy(self.coef_)).ravel()\n''',
    '''        self._check_is_fitted()\n        from statgpu.panel._formula import _get_feature_names\n\n        if getattr(self, "_coefficient_inference_available", True) is False:\n            raise ValueError(\n                getattr(\n                    self,\n                    "_coefficient_inference_reason",\n                    "coefficient-level inference is unavailable for this fit",\n                )\n            )\n\n        coef_np = np.asarray(_to_numpy(self.coef_)).ravel()\n''',
)

for path, rank_name in [
    ("statgpu/panel/_pooled.py", "rank"),
    ("statgpu/panel/_fixed_effects.py", "fit_rank"),
    ("statgpu/panel/_between.py", "rank_mean"),
    ("statgpu/panel/_first_diff.py", "rank_diff"),
    ("statgpu/panel/_random_effects.py", "rank_star"),
]:
    replace_once(
        path,
        "            backend=backend,\n            cov_type=self._cov_type,\n",
        f"            backend=backend,\n            fit_rank={rank_name},\n            cov_type=self._cov_type,\n",
    )

# 3) PanelOLS prediction must execute the linear algebra on the selected backend.
replace_once(
    "statgpu/panel/_fixed_effects.py",
    "    xp_cholesky_solve,\n    xp_maximum,\n)",
    "    xp_cholesky_solve,\n    xp_maximum,\n    xp_asarray,\n)",
)
regex_once(
    "statgpu/panel/_fixed_effects.py",
    r"    def predict\(self, X, entity_ids=None, time_ids=None\):.*?\n\n    def summary\(self\):",
    '''    def predict(self, X, entity_ids=None, time_ids=None):\n        """Predict on the selected numerical backend and return NumPy output."""\n        self._check_is_fitted()\n        backend = self._get_backend(backend="auto")\n        xp = backend.xp\n        prediction = self._panel_predict_linear(\n            X,\n            model_has_intercept=False,\n            add_intercept=False,\n            return_numpy=False,\n        )\n        self._predict_backend_name = backend.name\n\n        def _effect_values(ids, mapping, name):\n            if not mapping or ids is None:\n                return None\n            ids_np = np.asarray(_to_numpy(ids)).ravel()\n            if ids_np.shape[0] != int(prediction.shape[0]):\n                raise ValueError(\n                    f"{name} must have one value per prediction row"\n                )\n            values = np.fromiter(\n                (float(mapping.get(value, 0.0)) for value in ids_np),\n                dtype=np.float64,\n                count=ids_np.shape[0],\n            )\n            return xp_asarray(\n                values, dtype=xp.float64, xp=xp, ref_arr=prediction\n            )\n\n        entity_effect = _effect_values(\n            entity_ids, self._entity_effects_map, "entity_ids"\n        )\n        if entity_effect is not None:\n            prediction = prediction + entity_effect\n        time_effect = _effect_values(\n            time_ids, self._time_effects_map, "time_ids"\n        )\n        if time_effect is not None:\n            prediction = prediction + time_effect\n\n        return np.asarray(_to_numpy(prediction), dtype=np.float64)\n\n    def summary(self):''',
)

# 4) First differences cannot silently difference duplicate entity-time rows.
replace_once(
    "statgpu/panel/_first_diff.py",
    '''        _time_labels, time_codes = factorize_panel_metadata(\n            time_ids, name="time_ids", expected_n=eids_np.shape[0]\n        )\n        sort_idx_np = np.lexsort((time_codes, eids_np))\n''',
    '''        _time_labels, time_codes = factorize_panel_metadata(\n            time_ids, name="time_ids", expected_n=eids_np.shape[0]\n        )\n        pairs = np.column_stack(\n            [np.asarray(eids_np, dtype=np.int64), np.asarray(time_codes, dtype=np.int64)]\n        )\n        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:\n            raise ValueError(\n                "FirstDifferenceOLS requires unique (entity_id, time_id) observations"\n            )\n        # Differences are taken between consecutive observed times within each\n        # entity. Internal calendar gaps are therefore allowed and are not filled.\n        sort_idx_np = np.lexsort((time_codes, eids_np))\n''',
)

# 5) Existing rank-deficient parity tests should verify fit-space quantities and
# explicit inference unavailability, not ordinary coefficient t/z inference.
replace_once(
    "dev/tests/test_panel_stage_c_rank_deficient_matrix.py",
    '''    for expected, actual in model_pairs:\n        _assert_inference(actual, expected)\n        expected_meta = expected._covariance_metadata\n''',
    '''    for expected, actual in model_pairs:\n        assert_allclose(actual.coef_, expected.coef_, rtol=2e-7, atol=2e-9)\n        assert_allclose(\n            actual._panel_cov_params_raw,\n            expected._panel_cov_params_raw,\n            rtol=2e-7,\n            atol=2e-9,\n        )\n        for model in (expected, actual):\n            assert model._coefficient_inference_available is False\n            assert model.bse_ is None\n            assert model.tvalues_ is None\n            assert model.pvalues_ is None\n            assert model.conf_int_ is None\n            assert model._inference_result.metadata["applicable"] is False\n            with pytest.raises(ValueError, match="rank deficient"):\n                model.summary()\n        expected_meta = expected._covariance_metadata\n''',
)

# 6) Physical runner: rank-deficient cases remain success cases, but the raw
# artifact must prove coefficient inference is unavailable. Also exercise
# PanelOLS prediction backend provenance on an entity-FE case.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''def _array(value):\n    return np.asarray(_to_numpy(value), dtype=np.float64)\n\n\ndef _array_backend_name(value):\n''',
    '''def _array(value):\n    return np.asarray(_to_numpy(value), dtype=np.float64)\n\n\ndef _optional_array(value):\n    return None if value is None else _array(value)\n\n\ndef _array_backend_name(value):\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        "coef": _array(model.coef_).ravel(),\n        "bse": _array(model.bse_).ravel(),\n        "tvalues": _array(model.tvalues_).ravel(),\n        "pvalues": _array(model.pvalues_).ravel(),\n        "conf_int": _array(model.conf_int_),\n        "covariance": _array(model._panel_cov_params_raw),\n''',
    '''        "coef": _array(model.coef_).ravel(),\n        "bse": None if model.bse_ is None else _array(model.bse_).ravel(),\n        "tvalues": None if model.tvalues_ is None else _array(model.tvalues_).ravel(),\n        "pvalues": None if model.pvalues_ is None else _array(model.pvalues_).ravel(),\n        "conf_int": _optional_array(model.conf_int_),\n        "covariance": _array(model._panel_cov_params_raw),\n        "coefficient_inference_applicable": bool(\n            getattr(model, "_coefficient_inference_available", True)\n        ),\n        "coefficient_inference_reason": getattr(\n            model, "_coefficient_inference_reason", None\n        ),\n        "prediction": _optional_array(getattr(model, "_physical_prediction", None)),\n        "prediction_backend": getattr(model, "_predict_backend_name", None),\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        cases[f"panel_entity_{cov}"] = PanelOLS(\n            entity_effects=True, cov_type=cov, device=device\n        ).fit(Xb, yb, entity_ids=eb)\n''',
    '''        cases[f"panel_entity_{cov}"] = PanelOLS(\n            entity_effects=True, cov_type=cov, device=device\n        ).fit(Xb, yb, entity_ids=eb)\n        if cov == "hc0":\n            cases[f"panel_entity_{cov}"]._physical_prediction = cases[\n                f"panel_entity_{cov}"\n            ].predict(Xb[:8], entity_ids=eb[:8])\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    differences = {}\n    for field in ("coef", "bse", "tvalues", "pvalues", "conf_int", "covariance"):\n        np.testing.assert_allclose(\n            candidate[field], reference[field], rtol=rtol, atol=atol, err_msg=f"{label}.{field}"\n        )\n        differences[field] = _max_abs(candidate[field], reference[field])\n''',
    '''    differences = {}\n    for field in ("coef", "bse", "tvalues", "pvalues", "conf_int", "covariance", "prediction"):\n        actual = candidate[field]\n        expected = reference[field]\n        if expected is None or actual is None:\n            if expected is not None or actual is not None:\n                raise AssertionError(\n                    f"{label}.{field}: None contract differs between candidate and reference"\n                )\n            differences[field] = 0.0\n            continue\n        np.testing.assert_allclose(\n            actual, expected, rtol=rtol, atol=atol, err_msg=f"{label}.{field}"\n        )\n        differences[field] = _max_abs(actual, expected)\n    for field in ("coefficient_inference_applicable", "coefficient_inference_reason", "prediction_backend"):\n        actual = candidate[field]\n        expected = reference[field]\n        if field == "prediction_backend" and expected == "numpy" and actual is not None:\n            # NumPy reference predicts on NumPy; GPU candidates must persist the\n            # requested execution backend instead of matching the reference label.\n            continue\n        if actual != expected:\n            raise AssertionError(f"{label}.{field}: {actual!r} != {expected!r}")\n        differences[field] = 0.0\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            payload["cases"][name] = {\n                "status": "success",\n                "executed_backend": executed,\n                "max_abs_differences": differences,\n                "covariance_metadata": snapshot["covariance_metadata"],\n                "fit_rank": _fit_rank(model),\n                "parameter_count": int(snapshot["coef"].size),\n            }\n''',
    '''            if name == "panel_entity_hc0" and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n            payload["cases"][name] = {\n                "status": "success",\n                "executed_backend": executed,\n                "max_abs_differences": differences,\n                "covariance_metadata": snapshot["covariance_metadata"],\n                "fit_rank": _fit_rank(model),\n                "parameter_count": int(snapshot["coef"].size),\n                "coefficient_inference_applicable": snapshot[\n                    "coefficient_inference_applicable"\n                ],\n                "coefficient_inference_reason": snapshot[\n                    "coefficient_inference_reason"\n                ],\n                "prediction_backend": snapshot["prediction_backend"],\n            }\n''',
)

# 7) Add adversarial regression coverage for the new public contracts.
new_test = ROOT / "dev/tests/test_panel_stage_c_review_round4.py"
new_test.write_text('''from __future__ import annotations\n\nimport numpy as np\nimport pytest\nfrom numpy.testing import assert_allclose\n\nfrom statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects\nfrom statgpu.panel._utils import demean_variables\n\n\ndef _explicit_two_way_residual(values, entity, time):\n    entity_levels = np.unique(entity)\n    time_levels = np.unique(time)\n    cols = [np.ones(len(entity), dtype=np.float64)]\n    cols.extend((entity == level).astype(np.float64) for level in entity_levels[1:])\n    cols.extend((time == level).astype(np.float64) for level in time_levels[1:])\n    design = np.column_stack(cols)\n    coef = np.linalg.lstsq(design, values, rcond=None)[0]\n    return values - design @ coef\n\n\ndef test_two_way_demeaning_waits_for_x_even_when_y_is_already_projected():\n    entity = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)\n    time = np.array([0, 1, 3, 0, 2, 1, 2, 3, 0, 2, 3], dtype=np.int64)\n    rng = np.random.default_rng(12920)\n    raw_y = rng.normal(size=len(entity))\n    y = _explicit_two_way_residual(raw_y, entity, time)\n    X = rng.normal(size=(len(entity), 2))\n\n    y_d, X_d = demean_variables(y, X, entity, time, xp=np, max_iter=200, tol=1e-12)\n    expected_y = _explicit_two_way_residual(y, entity, time)\n    expected_X = np.column_stack(\n        [_explicit_two_way_residual(X[:, j], entity, time) for j in range(X.shape[1])]\n    )\n    assert_allclose(y_d, expected_y, rtol=0, atol=2e-11)\n    assert_allclose(X_d, expected_X, rtol=0, atol=2e-11)\n\n    with pytest.raises(RuntimeError, match="did not converge"):\n        demean_variables(y, X, entity, time, xp=np, max_iter=1, tol=1e-12)\n\n\ndef test_rank_deficient_fits_keep_fit_space_but_disable_coordinate_inference():\n    rng = np.random.default_rng(12921)\n    n_entities, n_times = 12, 4\n    entity = np.repeat(np.arange(n_entities), n_times)\n    time = np.tile(np.arange(n_times), n_entities)\n    x = rng.normal(size=entity.size)\n    X = np.column_stack([x, 2.0 * x])\n    y = 0.4 + 0.8 * x + np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)\n    y += rng.normal(scale=0.15, size=entity.size)\n\n    models = [\n        PooledOLS(cov_type="hc0").fit(X, y, entity_ids=entity),\n        PanelOLS(entity_effects=True, cov_type="hc0").fit(X, y, entity_ids=entity),\n        BetweenOLS(cov_type="hc0").fit(X, y, entity_ids=entity),\n        FirstDifferenceOLS(cov_type="hc0").fit(X, y, entity_ids=entity, time_ids=time),\n        RandomEffects(cov_type="hc0").fit(\n            np.column_stack([np.ones(len(y)), X]), y, entity_ids=entity\n        ),\n    ]\n\n    for model in models:\n        assert model.coef_ is not None\n        assert model._coefficient_inference_available is False\n        assert model.bse_ is None\n        assert model.tvalues_ is None\n        assert model.pvalues_ is None\n        assert model.conf_int_ is None\n        assert model._inference_result.metadata["applicable"] is False\n        assert "rank deficient" in model._inference_result.metadata["reason"]\n        with pytest.raises(ValueError, match="rank deficient"):\n            model.summary()\n\n\ndef test_full_rank_inference_contract_is_unchanged():\n    rng = np.random.default_rng(12922)\n    X = rng.normal(size=(60, 2))\n    y = 0.4 + X @ np.array([0.7, -0.2]) + rng.normal(scale=0.2, size=60)\n    model = PooledOLS(cov_type="hc0").fit(X, y)\n    assert model._coefficient_inference_available is True\n    assert model.bse_ is not None\n    assert model.pvalues_ is not None\n    assert model.conf_int_ is not None\n\n\ndef test_first_difference_rejects_duplicate_entity_time_pairs():\n    X = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [3.0]])\n    y = np.array([0.0, 1.0, 2.0, 0.0, 1.5, 3.0])\n    entity = np.array([0, 0, 0, 1, 1, 1])\n    time = np.array([0, 0, 2, 0, 1, 3])\n    with pytest.raises(ValueError, match="unique \\(entity_id, time_id\\)"):\n        FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time)\n\n\ndef test_first_difference_keeps_consecutive_observed_gap_semantics():\n    X = np.array([[0.0], [2.0], [5.0], [1.0], [4.0], [8.0]])\n    y = np.array([0.0, 2.0, 5.0, 1.0, 4.0, 8.0])\n    entity = np.array([0, 0, 0, 1, 1, 1])\n    time = np.array([1, 3, 7, 1, 4, 9])\n    model = FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time)\n    assert_allclose(model.coef_, np.array([1.0]), rtol=0, atol=1e-12)\n\n\ndef test_panel_predict_full_rank_effect_semantics_remain_numpy_visible():\n    rng = np.random.default_rng(12923)\n    entity = np.repeat(np.arange(8), 4)\n    X = rng.normal(size=(entity.size, 2))\n    alpha = np.repeat(np.linspace(-0.3, 0.4, 8), 4)\n    y = X @ np.array([0.6, -0.25]) + alpha\n    model = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)\n    pred = model.predict(X[:8], entity_ids=entity[:8])\n    assert isinstance(pred, np.ndarray)\n    assert model._predict_backend_name == "numpy"\n    assert pred.shape == (8,)\n''', encoding="utf-8")

# 8) Durable documentation: preserve immutable evidence facts but remove lifecycle
# claims from long-lived model pages. Record the new fixes in all changelogs.
for path in ("docs/en/models/panel.md",):
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    text = re.sub(
        r"The previous exact-clean Tesla P100 measurement `3dc7df19\.\.\.`.*?plus the unchanged 58-row performance matrix\.",
        "The exact-clean Tesla P100 measurements remain immutable audit evidence. The `f1546476...` source recorded CuPy/Torch 47/47 per backend and all 58 synchronized performance rows and was promoted under immutable v3 source identities. Subsequent 2026-08-12 hardening changed two-way FE convergence, rank-deficient coefficient-inference applicability, PanelOLS prediction backend execution, and duplicate-time FirstDifference validation, so that v3 numerical evidence is historical rather than evidence for the newer implementation. Long-lived model documentation records evidence lineage but does not carry PR lifecycle state.",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"The `3dc7df19\.\.\.` Tesla P100 run is historical evidence only.*?Explicit GPU devices continue to forbid silent CPU fallback\.",
        "The accepted `f1546476...` P100 run remains immutable historical evidence: both CuPy and Torch completed 47/47 correctness cases and the synchronized performance matrix contained 58/58 rows. Later implementation hardening requires a new exact-head physical lineage before those measurements can describe the current numerical tree. PR-specific acceptance state is intentionally tracked only in repository review records. Explicit GPU devices continue to forbid silent CPU fallback.",
        text,
        count=1,
        flags=re.S,
    )
    p.write_text(text, encoding="utf-8")

p = ROOT / "docs/cn/models/panel.md"
text = p.read_text(encoding="utf-8")
text = re.sub(
    r"此前 exact-clean `3dc7df19\.\.\.` Tesla P100 结果.*?performance 目标仍为 58 行。",
    "Tesla P100 的 exact-clean 测量继续作为不可变审计证据保留。`f1546476...` source 已记录 CuPy/Torch 每个 backend 47/47 correctness case 以及全部 58 行同步 performance，并按新的 immutable v3 source identity 完成 promotion。随后 2026-08-12 的 hardening 又修改了 two-way FE 收敛、秩亏 coefficient-inference applicability、PanelOLS prediction backend execution 与 FirstDifference duplicate-time validation，因此该 v3 数值证据现在只描述历史实现，而不代表更新后的 numerical tree。长期 model 文档只记录稳定的 evidence lineage，不记录 PR lifecycle 状态。",
    text,
    count=1,
    flags=re.S,
)
p.write_text(text, encoding="utf-8")

replace_once(
    "CHANGELOG.md",
    "- Retained the exact-clean `3dc7df19...` P100 result (CuPy/Torch 39/39 and 58 synchronized performance rows) as immutable historical evidence after a 2026-08-12 review reopened rank-deficient df and covariance-validity issues; the fixes now make supported rank-deficient df depend on identified rank and fail closed on every strictly negative final variance, with fresh 47/47-per-backend physical acceptance pending.\n",
    "- Hardened two-way FE convergence to require both outcome and design projections to converge, made rank-deficient coefficient inference explicitly unavailable while preserving identified fit-space results, kept `PanelOLS.predict()` on the selected NumPy/CuPy/Torch backend, and rejected duplicate entity-time rows in `FirstDifferenceOLS`; prior P100 lineages remain immutable historical evidence after these numerical changes.\n",
)
replace_once(
    "docs/en/changelog.md",
    "The exact-clean `3dc7df19...` Tesla P100 result (CuPy/Torch **39/39 per backend**, plus all 58 synchronized performance rows) remains immutable historical evidence. A 2026-08-12 strict re-review subsequently reopened the rank-deficient residual/Swamy-Arora df contract and a unit-dependent negative-variance guard. The local fixes now use identified rank for supported rank-deficient df while preserving historical full-rank formulas, and strict inference rejects every truly negative final variance. Because production numerical behavior changed, fresh physical acceptance is pending on the expanded **47/47 per backend** correctness matrix; the performance target remains 58 rows.\n",
    "The P100 validation lineages remain immutable audit evidence, including the later exact-clean `f1546476...` run with CuPy/Torch **47/47 per backend** and all 58 synchronized performance rows. A subsequent strict re-review hardened two-way FE alternating projection so both `y` and `X` must converge, made coordinate-wise coefficient inference explicitly unavailable for exact rank-deficient fits while preserving identified fit-space results, moved `PanelOLS.predict()` onto the selected numerical backend before returning its historical NumPy output, and rejected duplicate `(entity,time)` observations in `FirstDifferenceOLS` while retaining consecutive-observed gap semantics. These numerical changes require a new exact-head physical lineage before the historical P100 results can describe the current implementation.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "exact-clean `3dc7df19...` Tesla P100 结果（CuPy/Torch **每个 backend 39/39**，以及全部 58 行同步 performance）继续作为不可变历史证据保留。2026-08-12 的 strict re-review 随后重新发现 rank-deficient residual/Swamy-Arora df 契约以及依赖单位的 negative-variance guard 问题；本地修复现在让受支持的秩亏 df 使用 identified rank，同时保持历史 full-rank 公式不变，并对任何真实负的最终 variance fail closed。由于 production numerical behavior 已改变，新的 physical acceptance 尚待执行扩展后的 **每个 backend 47/47** correctness matrix；performance 目标仍为 58 行。\n",
    "P100 validation lineage 继续作为不可变审计证据保留，其中较新的 exact-clean `f1546476...` run 已完成 CuPy/Torch **每个 backend 47/47** correctness case 和全部 58 行同步 performance。随后新的 strict re-review 又强化了 two-way FE alternating projection：只有 `y` 与 `X` 都收敛才接受；exact rank-deficient fit 保留 identified fit-space 结果但不再发布普通逐系数 inference；`PanelOLS.predict()` 先在所选 numerical backend 上计算再返回历史 NumPy output；`FirstDifferenceOLS` 拒绝重复 `(entity,time)` 观测，同时继续采用 consecutive-observed gap 语义。由于这些修改再次改变 production numerical behavior，历史 P100 结果不能作为当前 implementation 的 exact-head evidence。\n",
)

print("PR126 round4 autofix patch applied")
