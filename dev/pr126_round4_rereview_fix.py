from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Convergence must be invariant to response/regressor units.  Normalize each
# alternating-projection change by the corresponding ORIGINAL variable scale;
# using the current residualized magnitude would fail for columns absorbed by
# the FE space, while a single global X scale would not be column-scale invariant.
replace_once(
    "statgpu/panel/_utils.py",
    '''    y_d = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()\n    X_d = (\n        X.copy()\n        if hasattr(X, "copy")\n        else X.clone()\n        if hasattr(X, "clone")\n        else X - 0.0\n    )\n\n    if entity_ids is not None:\n''',
    '''    y_d = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()\n    X_d = (\n        X.copy()\n        if hasattr(X, "copy")\n        else X.clone()\n        if hasattr(X, "clone")\n        else X - 0.0\n    )\n    y_scale_ref = _to_float_scalar(xp.max(xp.abs(y_d)))\n    if getattr(xp, "__name__", "") == "torch":\n        X_scale_ref = xp.max(xp.abs(X), dim=0).values\n    else:\n        X_scale_ref = xp.max(xp.abs(X), axis=0)\n    X_scale_ref = xp_maximum(\n        X_scale_ref, np.finfo(np.float64).tiny, xp\n    )\n    y_scale_ref = max(float(y_scale_ref), np.finfo(np.float64).tiny)\n\n    if entity_ids is not None:\n''',
)
replace_once(
    "statgpu/panel/_utils.py",
    '''            y_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))\n            X_change = _to_float_scalar(xp.max(xp.abs(X_d - X_d_old)))\n            max_change = max(float(y_change), float(X_change))\n            if max_change < tol:\n                converged = True\n                break\n        if not converged:\n            raise RuntimeError(\n                "two-way fixed-effect demeaning did not converge within "\n                f"max_iter={max_iter}; final max change={max_change:.6e}, tol={tol:.6e}"\n            )\n''',
    '''            y_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))\n            if getattr(xp, "__name__", "") == "torch":\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), dim=0).values\n            else:\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), axis=0)\n            y_relative_change = float(y_change) / y_scale_ref\n            X_relative_change = _to_float_scalar(\n                xp.max(X_change_columns / X_scale_ref)\n            )\n            max_change = max(float(y_relative_change), float(X_relative_change))\n            if max_change < tol:\n                converged = True\n                break\n        if not converged:\n            raise RuntimeError(\n                "two-way fixed-effect demeaning did not converge within "\n                f"max_iter={max_iter}; final max relative change={max_change:.6e}, "\n                f"tol={tol:.6e}"\n            )\n''',
)

# Preserve PanelOLS's historical array prediction convenience: if the fitted
# coefficient vector has exactly one more coordinate than the supplied design,
# retry with a leading constant, but keep both attempts on the selected backend.
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''        prediction = self._panel_predict_linear(\n            X,\n            model_has_intercept=False,\n            add_intercept=False,\n            return_numpy=False,\n        )\n        self._predict_backend_name = backend.name\n''',
    '''        try:\n            prediction = self._panel_predict_linear(\n                X,\n                model_has_intercept=False,\n                add_intercept=False,\n                return_numpy=False,\n            )\n        except ValueError as exc:\n            if str(exc) != "X has an incompatible feature count":\n                raise\n            prediction = self._panel_predict_linear(\n                X,\n                model_has_intercept=False,\n                add_intercept=True,\n                return_numpy=False,\n            )\n        self._predict_backend_name = backend.name\n''',
)

# Add regression coverage for unit-equivariant two-way convergence and the
# omitted-explicit-constant prediction compatibility path.
p = ROOT / "dev/tests/test_panel_stage_c_review_round4.py"
text = p.read_text(encoding="utf-8")
text += '''\n\ndef test_two_way_demeaning_convergence_is_unit_equivariant():\n    entity = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)\n    time = np.array([0, 1, 3, 0, 2, 1, 2, 3, 0, 2, 3], dtype=np.int64)\n    rng = np.random.default_rng(12924)\n    raw_y = rng.normal(size=len(entity))\n    y = _explicit_two_way_residual(raw_y, entity, time)\n    X = rng.normal(size=(len(entity), 2))\n\n    y_base, X_base = demean_variables(\n        y, X, entity, time, xp=np, max_iter=200, tol=1e-12\n    )\n    y_scale = 1.0e-12\n    X_scales = np.array([1.0e-9, 1.0e9])\n    y_scaled, X_scaled = demean_variables(\n        y * y_scale,\n        X * X_scales,\n        entity,\n        time,\n        xp=np,\n        max_iter=200,\n        tol=1e-12,\n    )\n    assert_allclose(y_scaled / y_scale, y_base, rtol=2e-10, atol=2e-11)\n    assert_allclose(X_scaled / X_scales, X_base, rtol=2e-10, atol=2e-11)\n\n    with pytest.raises(RuntimeError, match="did not converge"):\n        demean_variables(\n            y * 1.0e-12,\n            X * 1.0e-12,\n            entity,\n            time,\n            xp=np,\n            max_iter=1,\n            tol=1e-10,\n        )\n\n\ndef test_panel_predict_preserves_omitted_explicit_constant_compatibility():\n    rng = np.random.default_rng(12925)\n    x = rng.normal(size=50)\n    X_full = np.column_stack([np.ones(len(x)), x])\n    y = 0.6 + 0.9 * x + rng.normal(scale=0.1, size=len(x))\n    model = PanelOLS().fit(X_full, y)\n    expected = X_full[:12] @ model.coef_\n    actual = model.predict(x[:12, None])\n    assert_allclose(actual, expected, rtol=0, atol=2e-12)\n    assert model._predict_backend_name == "numpy"\n'''
p.write_text(text, encoding="utf-8")

print("PR126 round4 re-review fixes applied")
