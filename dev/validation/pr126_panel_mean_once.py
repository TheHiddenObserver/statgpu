from pathlib import Path


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"{label} anchor missing")
    return text.replace(old, new, 1)


# Fama-MacBeth coefficient average: scale only when a T-term reduction can
# overflow.  Dividing every coordinate by its data magnitude destroys a small
# but representable cancellation remainder.
p = Path("statgpu/panel/_fama_macbeth.py")
text = p.read_text(encoding="utf-8")
old = '''        # Scale before averaging so a finite common coefficient level does not
        # overflow merely because the raw reduction sums T copies first.
        if xp.__name__ == "torch":
            beta_scale = xp.max(xp.abs(betas), dim=0).values
        else:
            beta_scale = xp.max(xp.abs(betas), axis=0)
        safe_beta_scale = xp.where(
            beta_scale > 0.0, beta_scale, xp.ones_like(beta_scale)
        )
        avg_beta = xp.mean(betas / safe_beta_scale, axis=0) * safe_beta_scale
'''
new = '''        # Protect the T-term reduction with the minimum scale needed for
        # overflow safety.  Magnitude-normalizing each coordinate can underflow
        # a small but representable remainder when large period coefficients
        # cancel (e.g. +1e154, -1e154, 1e-170).
        if xp.__name__ == "torch":
            beta_scale = xp.max(xp.abs(betas), dim=0).values
        else:
            beta_scale = xp.max(xp.abs(betas), axis=0)
        reduction_limit = np.finfo(np.float64).max / float(T)
        mean_scale = xp.where(
            beta_scale > reduction_limit,
            xp.full_like(beta_scale, float(T)),
            xp.ones_like(beta_scale),
        )
        avg_beta = xp.sum(betas / mean_scale, axis=0) * (
            mean_scale / float(T)
        )
'''
text = replace_once(text, old, new, "FMB average")
p.write_text(text, encoding="utf-8")

# Shared parameter-R2 diagnostics: same minimal reduction scaling for ordinary
# means, and group-specific count scaling only in groups whose raw sum could
# overflow.  Safe/tiny groups are not scaled merely because another group has a
# huge value.
p = Path("statgpu/panel/_diagnostics.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from statgpu.panel._utils import group_means\n",
    "from statgpu.panel._utils import group_means, group_sizes\n",
    "diagnostic group_sizes import",
)
old = '''def _scaled_mean(values, xp):
    """Return a backend-native mean without overflowing the raw sum."""
    scale = xp.max(xp.abs(values))
    safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
    return xp.mean(values / safe_scale) * safe_scale


def _scaled_group_means(values, groups, xp):
    """Return group means after one global scaling to protect group sums."""
    scale = xp.max(xp.abs(values))
    safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
    return group_means(values / safe_scale, groups, xp=xp) * safe_scale
'''
new = '''def _scaled_mean(values, xp):
    """Return a mean with only the reduction-length scaling needed for safety."""
    n = int(values.shape[0])
    max_abs = xp.max(xp.abs(values))
    limit = np.finfo(np.float64).max / float(max(n, 1))
    factor = xp.where(
        max_abs > limit,
        xp.full_like(max_abs, float(n)),
        xp.ones_like(max_abs),
    )
    return xp.sum(values / factor) * (factor / float(n))


def _scaled_group_means(values, groups, xp):
    """Return group means without globally magnitude-normalizing safe groups."""
    sizes = group_sizes(groups, xp=xp)
    limit = np.finfo(np.float64).max / sizes
    dangerous = (xp.abs(values) > limit) * 1.0
    dangerous_group = group_means(dangerous, groups, xp=xp) > 0.0
    factor = xp.where(dangerous_group, sizes, xp.ones_like(sizes))
    # For a dangerous group of size m, group_means(values / m) * m equals
    # the original mean but the same-sign accumulation is bounded by max|x|.
    return group_means(values / factor, groups, xp=xp) * factor
'''
text = replace_once(text, old, new, "diagnostic mean helpers")
p.write_text(text, encoding="utf-8")

# Maintained regression coverage.
p = Path("dev/tests/test_fama_macbeth_review_fixes.py")
text = p.read_text(encoding="utf-8")
if "test_fama_macbeth_preserves_small_cancellation_remainder_numpy" not in text:
    text += r'''


def _cancellation_mean_fixture():
    x_period = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    slopes = np.asarray([-1.0e154, 1.0e154, 1.0e-170], dtype=np.float64)
    X = np.tile(x_period, 3)[:, None]
    y = np.concatenate([slope * x_period for slope in slopes])
    time = np.repeat(np.arange(3), x_period.size)
    return X, y, time, float(1.0e-170 / 3.0)


def test_fama_macbeth_preserves_small_cancellation_remainder_numpy():
    X, y, time, expected = _cancellation_mean_fixture()
    model = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(X, y, time_ids=time)
    actual = float(_to_numpy(model.coef_)[1])
    assert actual != 0.0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_fama_macbeth_preserves_small_cancellation_remainder_torch_cpu():
    torch = pytest.importorskip("torch")
    X, y, time, expected = _cancellation_mean_fixture()
    model = FamaMacBeth(cov_type="nonrobust").fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=time,
    )
    actual = float(_to_numpy(model.coef_)[1])
    assert actual != 0.0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_parameter_r2_mean_helpers_preserve_cancellation_and_safe_tiny_groups_numpy():
    from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean

    values = np.asarray([1.0e154, -1.0e154, 1.0e-170], dtype=np.float64)
    np.testing.assert_allclose(
        float(_scaled_mean(values, np)), 1.0e-170 / 3.0, rtol=1e-15, atol=0.0
    )

    grouped = np.asarray(
        [1.0e154, -1.0e154, 1.0e-170, 1.0e-320, 1.0e-320],
        dtype=np.float64,
    )
    groups = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    actual = np.asarray(_scaled_group_means(grouped, groups, np))
    expected = np.asarray(
        [1.0e-170 / 3.0] * 3 + [1.0e-320, 1.0e-320], dtype=np.float64
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)


def test_parameter_r2_mean_helpers_preserve_cancellation_and_safe_tiny_groups_torch_cpu():
    torch = pytest.importorskip("torch")
    from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean

    values = torch.tensor([1.0e154, -1.0e154, 1.0e-170], dtype=torch.float64)
    mean = float(_scaled_mean(values, torch).detach().cpu())
    np.testing.assert_allclose(mean, 1.0e-170 / 3.0, rtol=2e-15, atol=0.0)

    grouped = torch.tensor(
        [1.0e154, -1.0e154, 1.0e-170, 1.0e-320, 1.0e-320], dtype=torch.float64
    )
    groups = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64)
    actual = _scaled_group_means(grouped, groups, torch).detach().cpu().numpy()
    expected = np.asarray(
        [1.0e-170 / 3.0] * 3 + [1.0e-320, 1.0e-320], dtype=np.float64
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0.0)
'''
p.write_text(text, encoding="utf-8")

# Physical GPU FMB audit: retain a cancellation remainder on the coefficient
# average.  Stage-C physical audit: exercise the shared diagnostic mean helpers
# on both GPU backends.
p = Path("dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py")
text = p.read_text(encoding="utf-8")
needle = '''    if not np.all(np.isfinite(_public_array(mixed.pvalues_))):
        raise AssertionError("zero-variance coefficient leaked non-finite p-value")

'''
addition = '''    cancel_betas = np.asarray(
        [[0.0, -1.0e154, 0.0], [0.0, 1.0e154, 0.0], [0.0, 1.0e-170, 0.0]],
        dtype=np.float64,
    )
    y_cancel = np.concatenate([design @ beta for beta in cancel_betas])
    X_cancel_b, y_cancel_b = _arrays(X_mixed, y_cancel, backend)
    cancellation = FamaMacBeth(cov_type="nonrobust", device=_device(backend)).fit(
        X_cancel_b, y_cancel_b, time_ids=time_mixed
    )
    cancellation_mean = float(_public_array(cancellation.coef_)[1])
    expected_cancellation_mean = 1.0e-170 / 3.0
    if cancellation_mean == 0.0:
        raise AssertionError("small coefficient cancellation remainder underflowed to zero")
    np.testing.assert_allclose(
        cancellation_mean, expected_cancellation_mean, rtol=3e-11, atol=0.0
    )

'''
if needle not in text:
    raise RuntimeError("FMB physical cancellation anchor missing")
text = text.replace(needle, needle + addition, 1)
old = '''        "mixed_scale_zero_variance_statistic": float(
            _public_array(mixed.tvalues_)[0]
        ),
'''
new = '''        "mixed_scale_zero_variance_statistic": float(
            _public_array(mixed.tvalues_)[0]
        ),
        "cancellation_safe_average": cancellation_mean,
'''
text = replace_once(text, old, new, "FMB physical payload")
p.write_text(text, encoding="utf-8")

p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
if "from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean" not in text:
    text = replace_once(
        text,
        "from statgpu.panel._covariance import ols_covariance\n",
        "from statgpu.panel._covariance import ols_covariance\nfrom statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean\n",
        "Stage-C diagnostic import",
    )
if "def _cancellation_safe_mean_audit" not in text:
    anchor = "def _tiny_design_lstsq_audit(backend):\n"
    helper = '''def _cancellation_safe_mean_audit(backend):
    values = np.asarray([1.0e154, -1.0e154, 1.0e-170], dtype=np.float64)
    grouped = np.asarray(
        [1.0e154, -1.0e154, 1.0e-170, 1.0e-320, 1.0e-320], dtype=np.float64
    )
    groups = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    dummy = np.arange(values.size, dtype=np.float64)[:, None]
    entity = np.arange(values.size, dtype=np.int64)
    time = np.arange(values.size, dtype=np.int64)
    _dummy_b, values_b, _eb, _tb = _to_backend(dummy, values, entity, time, backend)
    dummy_g = np.arange(grouped.size, dtype=np.float64)[:, None]
    entity_g = np.arange(grouped.size, dtype=np.int64)
    time_g = np.arange(grouped.size, dtype=np.int64)
    _dg, grouped_b, groups_b, _tg = _to_backend(dummy_g, grouped, groups, time_g, backend)
    xp = __import__("torch") if backend == "torch" else __import__("cupy")
    mean = float(_array(_scaled_mean(values_b, xp)))
    group_result = _array(_scaled_group_means(grouped_b, groups_b, xp))
    expected_mean = 1.0e-170 / 3.0
    expected_group = np.asarray(
        [expected_mean] * 3 + [1.0e-320, 1.0e-320], dtype=np.float64
    )
    np.testing.assert_allclose(mean, expected_mean, rtol=3e-11, atol=0.0)
    np.testing.assert_allclose(group_result, expected_group, rtol=3e-11, atol=0.0)
    return {"status": "success", "backend": backend, "mean": mean}


'''
    text = replace_once(text, anchor, helper + anchor, "Stage-C mean audit insertion")
old = '''        payload["numerical_primitives"] = {
            "tiny_design_lstsq": _tiny_design_lstsq_audit(backend),
            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),
        }
'''
new = '''        payload["numerical_primitives"] = {
            "tiny_design_lstsq": _tiny_design_lstsq_audit(backend),
            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),
            "cancellation_safe_mean": _cancellation_safe_mean_audit(backend),
        }
'''
text = replace_once(text, old, new, "Stage-C numerical payload")
p.write_text(text, encoding="utf-8")

print("PR126 cancellation-safe mean patch applied")
