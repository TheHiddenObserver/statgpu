from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def regex_once(path: str, pattern: str, repl: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    new, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one regex match, got {count}")
    p.write_text(new, encoding="utf-8")


# RandomEffects has the same public linear-prediction backend contract as the
# other residual-OLS panel families. Preserve the legacy optional-leading-
# constant convenience, but execute both attempts on the selected backend.
regex_once(
    "statgpu/panel/_random_effects.py",
    r"    def predict\(self, X\):.*?\n\n    def summary\(self\):",
    '''    def predict(self, X):\n        """Predict on the selected numerical backend and return NumPy output."""\n        self._check_is_fitted()\n        backend = self._get_backend(backend="auto")\n        try:\n            prediction = self._panel_predict_linear(\n                X,\n                model_has_intercept=False,\n                add_intercept=False,\n                return_numpy=False,\n            )\n        except ValueError as exc:\n            if str(exc) != "X has an incompatible feature count":\n                raise\n            prediction = self._panel_predict_linear(\n                X,\n                model_has_intercept=False,\n                add_intercept=True,\n                return_numpy=False,\n            )\n        self._predict_backend_name = backend.name\n        return np.asarray(_to_numpy(prediction), dtype=np.float64)\n\n    def summary(self):''',
)

# Exercise prediction on a full-rank RandomEffects physical case and persist
# the executed prediction backend through the already-versioned schema-v2
# snapshot contract.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    for cov in ("robust", "hc0", "hc2", "hc3"):\n        cases[f"random_effects_explicit_constant_{cov}"] = RandomEffects(\n            cov_type=cov, device=device\n        ).fit(Xcb, ycb, entity_ids=ecb)\n''',
    '''    for cov in ("robust", "hc0", "hc2", "hc3"):\n        cases[f"random_effects_explicit_constant_{cov}"] = RandomEffects(\n            cov_type=cov, device=device\n        ).fit(Xcb, ycb, entity_ids=ecb)\n        if cov == "hc0":\n            cases[f"random_effects_explicit_constant_{cov}"]._physical_prediction = cases[\n                f"random_effects_explicit_constant_{cov}"\n            ].predict(Xcb[:8])\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            if name == "panel_entity_hc0" and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n''',
    '''            if name in {\n                "panel_entity_hc0",\n                "random_effects_explicit_constant_hc0",\n            } and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n''',
)

# Add local full-rank prediction parity + omitted-constant compatibility.
p = ROOT / "dev/tests/test_panel_stage_c_review_round4.py"
text = p.read_text(encoding="utf-8")
text += '''\n\ndef test_random_effects_predict_preserves_backend_helper_and_constant_compatibility():\n    rng = np.random.default_rng(12926)\n    n_entities, n_times = 10, 4\n    entity = np.repeat(np.arange(n_entities), n_times)\n    x = rng.normal(size=entity.size)\n    X_full = np.column_stack([np.ones(len(x)), x])\n    alpha = np.repeat(rng.normal(scale=0.25, size=n_entities), n_times)\n    y = 0.5 + 0.85 * x + alpha + rng.normal(scale=0.15, size=len(x))\n    model = RandomEffects(cov_type="hc0").fit(X_full, y, entity_ids=entity)\n\n    expected_full = X_full[:12] @ model.coef_\n    actual_full = model.predict(X_full[:12])\n    assert_allclose(actual_full, expected_full, rtol=0, atol=2e-12)\n    assert model._predict_backend_name == "numpy"\n\n    actual_omitted = model.predict(x[:12, None])\n    assert_allclose(actual_omitted, expected_full, rtol=0, atol=2e-12)\n    assert model._predict_backend_name == "numpy"\n'''
p.write_text(text, encoding="utf-8")

print("PR126 RandomEffects prediction backend fix applied")
