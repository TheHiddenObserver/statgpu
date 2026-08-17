from pathlib import Path

# Re-review follow-up: prediction must use the fitted backend's strict
# cross-container converter just like fit does.
p = Path('statgpu/panel/_fama_macbeth.py')
text = p.read_text(encoding='utf-8')
old = '''        xp = self._xp
        X_arr = xp_asarray(X_data, dtype=xp.float64, xp=xp, ref_arr=self._fit_ref_)
'''
new = '''        xp = self._xp
        X_source = self._to_array(X_data, backend=self._backend_name)
        X_arr = xp_asarray(
            X_source, dtype=xp.float64, xp=xp, ref_arr=self._fit_ref_
        )
'''
if old not in text:
    raise RuntimeError('FamaMacBeth prediction conversion anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Make the hosted CPU regression prove that prediction traverses the shared
# backend converter instead of passing only because NumPy happens to accept a
# particular foreign container.
p = Path('dev/tests/test_panel_pr126_fmb_exact_period.py')
text = p.read_text(encoding='utf-8')
old = '''    prediction = actual.predict(torch.as_tensor(X[:3], dtype=torch.float64))
    assert isinstance(prediction, np.ndarray)
    assert_allclose(prediction, expected.predict(X[:3]), rtol=0.0, atol=0.0)
'''
new = '''    original_to_array = actual._to_array
    conversions = []

    def tracked_to_array(value, backend=None):
        conversions.append((type(value).__module__, backend))
        return original_to_array(value, backend=backend)

    actual._to_array = tracked_to_array
    prediction = actual.predict(torch.as_tensor(X[:3], dtype=torch.float64))
    assert conversions == [("torch", "numpy")]
    assert isinstance(prediction, np.ndarray)
    assert_allclose(prediction, expected.predict(X[:3]), rtol=0.0, atol=0.0)
'''
if old not in text:
    raise RuntimeError('explicit CPU prediction regression anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Physical cross-container acceptance must exercise foreign-container predict,
# not only fit.  Reference predict remains NumPy; candidate predict receives the
# opposite GPU container and must still return the fitted requested backend.
p = Path('dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py')
text = p.read_text(encoding='utf-8')
old = '''        "max_abs_differences": _assert_snapshot(
            _snapshot(reference, X[:3]), _snapshot(actual, X[:3])
        ),
'''
new = '''        "max_abs_differences": _assert_snapshot(
            _snapshot(reference, X[:3]), _snapshot(actual, foreign_X[:3])
        ),
'''
# Restrict replacement to the cross-container case by anchoring from its def.
start = text.find('def _explicit_device_cross_container_case(backend: str):')
if start < 0:
    raise RuntimeError('cross-container physical helper not found')
pos = text.find(old, start)
if pos < 0:
    raise RuntimeError('cross-container prediction snapshot anchor not found')
text = text[:pos] + new + text[pos + len(old):]
p.write_text(text, encoding='utf-8')

# Tighten docs so fit and subsequent prediction share one explicit-device rule.
for path, old_doc, new_doc in [
    (
        'docs/en/panel/fama-macbeth.md',
        'statgpu converts the input to the requested backend, and an unavailable explicitly requested GPU backend raises instead of silently switching execution.',
        'statgpu converts fit and prediction inputs to the requested/fitted backend, and an unavailable explicitly requested GPU backend raises instead of silently switching execution.'
    ),
    (
        'docs/cn/panel/fama-macbeth.md',
        'statgpu 也会将其转换到请求的后端执行。若显式请求的 GPU 后端不可用，`.fit()` 会报错，而不会静默切换执行后端。',
        'statgpu 也会将拟合与预测输入转换到请求/已拟合的后端执行。若显式请求的 GPU 后端不可用，`.fit()` 会报错，而不会静默切换执行后端。'
    ),
]:
    p = Path(path)
    doc = p.read_text(encoding='utf-8')
    if old_doc not in doc:
        raise RuntimeError(f'prediction doc anchor not found: {path}')
    p.write_text(doc.replace(old_doc, new_doc, 1), encoding='utf-8')
