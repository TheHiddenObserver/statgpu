from pathlib import Path

# Production: explicit device must own backend selection. AUTO retains input-native dispatch.
p = Path('statgpu/panel/_fama_macbeth.py')
text = p.read_text(encoding='utf-8')
old = '''    def _prepare_backend_arrays(self, X, y, *, validate_finite=True):
        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        ref = None
        if backend_name == "torch":
            import torch

            if isinstance(X, torch.Tensor):
                ref = X
            else:
                dev = self._get_compute_device()
                target = "cuda" if dev.value in ("torch", "cuda") else "cpu"
                ref = torch.empty(0, dtype=torch.float64, device=target)
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=ref)
        y_arr = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
'''
new = '''    def _prepare_backend_arrays(self, X, y, *, validate_finite=True):
        # AUTO preserves an already backend-native input. An explicit device is
        # authoritative instead: convert heterogeneous inputs to the requested
        # NumPy/CuPy/Torch backend rather than silently letting container type
        # override the public execution request.
        if self._device == Device.AUTO:
            backend_name = _detect_backend(X, self._get_compute_device())
            xp = _get_xp(backend_name)
            X_source = X
            y_source = y
            ref = None
            if backend_name == "torch":
                import torch

                if isinstance(X, torch.Tensor):
                    ref = X
                else:
                    dev = self._get_compute_device()
                    target = "cuda" if dev.value in ("torch", "cuda") else "cpu"
                    ref = torch.empty(0, dtype=torch.float64, device=target)
        else:
            backend = self._get_backend(backend="auto")
            backend_name = backend.name
            xp = backend.xp
            X_source = self._to_array(X, backend=backend_name)
            y_source = self._to_array(y, backend=backend_name)
            ref = X_source if backend_name in {"cupy", "torch"} else None

        X_arr = xp_asarray(X_source, dtype=xp.float64, xp=xp, ref_arr=ref)
        y_arr = xp_asarray(y_source, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
'''
if old not in text:
    raise RuntimeError('FamaMacBeth backend preparation anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Maintained regression already belongs to the Stage-C Torch CPU workflow.
p = Path('dev/tests/test_panel_pr126_fmb_exact_period.py')
text = p.read_text(encoding='utf-8')
anchor = '''def test_fama_macbeth_exactly_identified_period_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    X, y, time, _ = _exact_period_fixture()
    expected = FamaMacBeth(cov_type="newey-west", bandwidth=1, device="cpu").fit(X, y, time_ids=time)
    actual = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert actual.n_periods == expected.n_periods == 3
    for name in ("betas_", "coef_", "cov_params_", "bse_", "pvalues_", "conf_int_"):
        av = getattr(actual, name)
        if hasattr(av, "detach"):
            av = av.detach().cpu().numpy()
        ev = getattr(expected, name)
        assert_allclose(np.asarray(av), np.asarray(ev), rtol=3e-10, atol=3e-12)


'''
if anchor not in text:
    raise RuntimeError('Torch exact-period test anchor not found')
addition = anchor + '''def test_fama_macbeth_explicit_cpu_overrides_torch_input_container():
    torch = pytest.importorskip("torch")
    X, y, time, _ = _exact_period_fixture()
    expected = FamaMacBeth(cov_type="newey-west", bandwidth=1, device="cpu").fit(
        X, y, time_ids=time
    )
    actual = FamaMacBeth(cov_type="newey-west", bandwidth=1, device="cpu").fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert actual._backend_name == "numpy"
    assert actual._inference_backend_name == "numpy"
    assert isinstance(actual.coef_, np.ndarray)
    assert isinstance(actual.cov_params_, np.ndarray)
    assert_allclose(actual.coef_, expected.coef_, rtol=0.0, atol=0.0)
    assert_allclose(actual.cov_params_, expected.cov_params_, rtol=0.0, atol=0.0)
    prediction = actual.predict(torch.as_tensor(X[:3], dtype=torch.float64))
    assert isinstance(prediction, np.ndarray)
    assert_allclose(prediction, expected.predict(X[:3]), rtol=0.0, atol=0.0)


'''
p.write_text(text.replace(anchor, addition, 1), encoding='utf-8')

# Extend the physical validator with cross-container requests for both GPU backends.
p = Path('dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py')
text = p.read_text(encoding='utf-8')
anchor = '''def _exact_period_case(backend: str):
    X, y, time_ids, expected_betas = _exact_period_fixture()
    reference = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device="cpu"
    ).fit(X, y, time_ids=time_ids)
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device=_device(backend)
    ).fit(Xb, yb, time_ids=time_ids)
'''
if anchor not in text:
    raise RuntimeError('physical exact-period anchor not found')
# Keep the existing function intact; insert helper before square-rank rejection.
insert_at = '''def _square_rank_rejection(backend: str):
'''
if insert_at not in text:
    raise RuntimeError('physical square-rank anchor not found')
helper = '''def _explicit_device_cross_container_case(backend: str):
    X, y, time_ids, _expected_betas = _exact_period_fixture()
    reference = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device="cpu"
    ).fit(X, y, time_ids=time_ids)
    if backend == "cupy":
        import torch

        foreign_X = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        foreign_y = torch.as_tensor(y, dtype=torch.float64, device="cuda")
    elif backend == "torch":
        import cupy as cp

        foreign_X = cp.asarray(X, dtype=cp.float64)
        foreign_y = cp.asarray(y, dtype=cp.float64)
    else:
        raise ValueError("cross-container physical case requires cupy or torch")

    actual = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device=_device(backend)
    ).fit(foreign_X, foreign_y, time_ids=time_ids)
    if actual._backend_name != backend or actual._inference_backend_name != backend:
        raise AssertionError(
            f"explicit {backend} request was overridden by foreign input container: "
            f"fit={actual._backend_name}, inference={actual._inference_backend_name}"
        )
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(reference), _inference_descriptor(actual)
    )
    return {
        "status": "success",
        "foreign_input_backend": "torch" if backend == "cupy" else "cupy",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(reference, X[:3]), _snapshot(actual, X[:3])
        ),
    }


'''
text = text.replace(insert_at, helper + insert_at, 1)
old_payload = '''            "exactly_identified_full_rank_period": _exact_period_case(backend),
            "square_rank_deficient_retained_period_rejected": _square_rank_rejection(backend),
'''
new_payload = '''            "exactly_identified_full_rank_period": _exact_period_case(backend),
            "explicit_device_overrides_foreign_input_container": _explicit_device_cross_container_case(backend),
            "square_rank_deficient_retained_period_rejected": _square_rank_rejection(backend),
'''
if old_payload not in text:
    raise RuntimeError('physical payload anchor not found')
p.write_text(text.replace(old_payload, new_payload, 1), encoding='utf-8')

# Docs: make device precedence explicit.
for path, old_doc, new_doc in [
    (
        'docs/en/panel/fama-macbeth.md',
        'If an explicitly requested GPU backend is unavailable, `.fit()` raises an error rather than switching to CPU.',
        'With `device="auto"`, an already NumPy/CuPy/Torch-native input may keep its native backend. An explicit `device="cpu"`, `device="cuda"`, or `device="torch"` request is authoritative even when the input container belongs to another backend: statgpu converts the input to the requested backend, and an unavailable explicitly requested GPU backend raises instead of silently switching execution.'
    ),
    (
        'docs/cn/panel/fama-macbeth.md',
        '如果显式请求的 GPU 后端不可用，`.fit()` 会报错，而不会切换到 CPU。',
        '当 `device="auto"` 时，已经是 NumPy/CuPy/Torch 原生数组的输入可以保留其原生后端；但显式 `device="cpu"`、`device="cuda"` 或 `device="torch"` 请求具有最高优先级，即使输入容器属于另一个后端，statgpu 也会将其转换到请求的后端执行。若显式请求的 GPU 后端不可用，`.fit()` 会报错，而不会静默切换执行后端。'
    ),
]:
    p = Path(path)
    doc = p.read_text(encoding='utf-8')
    if old_doc not in doc:
        raise RuntimeError(f'doc anchor not found: {path}')
    p.write_text(doc.replace(old_doc, new_doc, 1), encoding='utf-8')

# Changelog entry next to current panel/FMB notes.
p = Path('CHANGELOG.md')
text = p.read_text(encoding='utf-8')
marker = '- **Fama-MacBeth exact-period eligibility**'
pos = text.find(marker)
if pos < 0:
    raise RuntimeError('CHANGELOG Fama-MacBeth marker not found')
line_end = text.find('\n', pos)
if line_end < 0:
    line_end = len(text)
entry = '\n- **Fama-MacBeth explicit-device backend authority**: explicit `cpu`/`cuda`/`torch` requests now override heterogeneous input container types; only `device="auto"` uses input-native backend dispatch, preventing silent execution on a backend different from the public request.'
text = text[:line_end] + entry + text[line_end:]
p.write_text(text, encoding='utf-8')
