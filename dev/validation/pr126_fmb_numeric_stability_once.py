from pathlib import Path

# 1) Certified Gram fast path: non-finite RHS/solution is never certified.
p = Path('statgpu/panel/_linalg.py')
text = p.read_text(encoding='utf-8')
old = '''    rhs_input = y[..., None] if getattr(y, "ndim", None) == 2 else y
    rhs = xp.matmul(transpose, rhs_input)

    eigenvalues = xp.linalg.eigvalsh(gram)
'''
new = '''    rhs_input = y[..., None] if getattr(y, "ndim", None) == 2 else y
    rhs = xp.matmul(transpose, rhs_input)
    rhs_finite_view = xp.isfinite(rhs).reshape(int(rhs.shape[0]), -1)
    rhs_finite = (
        xp.all(rhs_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(rhs_finite_view, axis=1)
    )

    eigenvalues = xp.linalg.eigvalsh(gram)
'''
if old not in text:
    raise RuntimeError('Gram RHS anchor not found')
text = text.replace(old, new, 1)
old = '''        & (largest > 0.0)
        & (smallest > largest * ratio)
    )
'''
new = '''        & (largest > 0.0)
        & (smallest > largest * ratio)
        & rhs_finite
    )
'''
if old not in text:
    raise RuntimeError('Gram certification anchor not found')
text = text.replace(old, new, 1)
old = '''    else:
        params = xp.linalg.solve(safe_gram, safe_rhs)

    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, certified
'''
new = '''    else:
        params = xp.linalg.solve(safe_gram, safe_rhs)

    params_finite_view = xp.isfinite(params).reshape(int(params.shape[0]), -1)
    params_finite = (
        xp.all(params_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(params_finite_view, axis=1)
    )
    certified = certified & params_finite

    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, certified
'''
if old not in text:
    raise RuntimeError('Gram solution anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# 2) Fama-MacBeth: stable coefficient mean, scaled covariance reductions,
# and fail-closed covariance validation.
p = Path('statgpu/panel/_fama_macbeth.py')
text = p.read_text(encoding='utf-8')
old = '''        avg_beta = xp.mean(betas, axis=0)
        beta_centered = betas - avg_beta
        effective_bandwidth = None
        if self._cov_type == "nonrobust":
            covariance = (beta_centered.T @ beta_centered) / float(T - 1)
            cov_params = covariance / float(T)
        else:
            bandwidth = self.bandwidth
            if bandwidth is None:
                bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
            bandwidth = max(0, min(int(bandwidth), T - 1))
            effective_bandwidth = bandwidth
            long_run = beta_centered.T @ beta_centered / float(T)
            for lag in range(1, bandwidth + 1):
                weight = 1.0 - lag / float(bandwidth + 1)
                gamma_lag = beta_centered[lag:].T @ beta_centered[:-lag] / float(T)
                long_run = long_run + weight * (gamma_lag + gamma_lag.T)
            cov_params = long_run / float(T)

        diagonal = xp.diag(cov_params)
        bse = xp.sqrt(xp.maximum(diagonal, xp.zeros_like(diagonal)))
'''
new = '''        # Scale before averaging so a finite common coefficient level does not
        # overflow merely because the raw reduction sums T copies first.
        if xp.__name__ == "torch":
            beta_scale = xp.max(xp.abs(betas), dim=0).values
        else:
            beta_scale = xp.max(xp.abs(betas), axis=0)
        safe_beta_scale = xp.where(
            beta_scale > 0.0, beta_scale, xp.ones_like(beta_scale)
        )
        avg_beta = xp.mean(betas / safe_beta_scale, axis=0) * safe_beta_scale
        if not _finite_all(avg_beta, xp):
            raise ValueError(
                "FamaMacBeth average coefficient is non-finite; the retained-period "
                "coefficient scale exceeds float64 numerical range"
            )

        beta_centered = betas - avg_beta
        if not _finite_all(beta_centered, xp):
            raise ValueError(
                "FamaMacBeth centered period coefficients are non-finite; covariance "
                "cannot be represented reliably in float64"
            )
        centered_scale = xp.max(xp.abs(beta_centered))
        safe_centered_scale = xp.where(
            centered_scale > 0.0,
            centered_scale,
            xp.ones_like(centered_scale),
        )
        beta_centered_scaled = beta_centered / safe_centered_scale

        effective_bandwidth = None
        if self._cov_type == "nonrobust":
            cov_scaled = (
                beta_centered_scaled.T @ beta_centered_scaled
            ) / float(T * (T - 1))
        else:
            bandwidth = self.bandwidth
            if bandwidth is None:
                bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
            bandwidth = max(0, min(int(bandwidth), T - 1))
            effective_bandwidth = bandwidth
            cov_scaled = (
                beta_centered_scaled.T @ beta_centered_scaled
            ) / float(T * T)
            for lag in range(1, bandwidth + 1):
                weight = 1.0 - lag / float(bandwidth + 1)
                gamma_lag = (
                    beta_centered_scaled[lag:].T
                    @ beta_centered_scaled[:-lag]
                ) / float(T * T)
                cov_scaled = cov_scaled + weight * (gamma_lag + gamma_lag.T)

        # Multiplying by the common scale one factor at a time avoids the
        # avoidable overflow of forming scale**2 before the normalized
        # covariance has reduced the magnitude. If the final covariance itself
        # is outside float64 range, fail closed below rather than publishing
        # Inf/NaN inference.
        cov_params = (cov_scaled * safe_centered_scale) * safe_centered_scale
        if not _finite_all(cov_params, xp):
            raise ValueError(
                "FamaMacBeth covariance contains non-finite values; inference is "
                "not numerically representable in float64"
            )

        diagonal = xp.diag(cov_params)
        diagonal_min = _to_float_scalar(xp.min(diagonal))
        if diagonal_min < 0.0:
            raise ValueError(
                "FamaMacBeth covariance has negative diagonal variance; inference "
                "is not numerically valid"
            )
        bse = xp.sqrt(xp.maximum(diagonal, xp.zeros_like(diagonal)))
'''
if old not in text:
    raise RuntimeError('Fama-MacBeth covariance block anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# 3) Maintained tests: exercise the Torch Gram path and representable large covariance.
p = Path('dev/tests/test_fama_macbeth_inference_matrix.py')
text = p.read_text(encoding='utf-8')
addition = r'''


def _large_common_intercept_fixture(n_periods=4):
    x_period = np.asarray([-1.0, 0.0, 1.0])
    X = np.tile(x_period, n_periods)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time_ids = np.repeat(np.arange(n_periods), x_period.size)
    return X, y, time_ids


def test_fama_macbeth_torch_gram_certificate_rejects_nonfinite_rhs_and_falls_back():
    torch = pytest.importorskip("torch")
    X, y, time_ids = _large_common_intercept_fixture()
    expected = FamaMacBeth(bandwidth=0, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    actual = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time_ids, dtype=torch.int64),
    )

    # X' y overflows in the Gram fast path even though the SVD projection and
    # the true period coefficients are finite. Every period must therefore be
    # routed to the rank-revealing fallback rather than certified with Inf beta.
    assert actual._backend_name == "torch"
    assert actual._period_svd_fallbacks == actual.n_periods == 4
    assert np.all(np.isfinite(_to_numpy(actual.betas_)))
    assert np.all(np.isfinite(_to_numpy(actual.coef_)))
    assert np.all(np.isfinite(_to_numpy(actual.cov_params_)))
    np.testing.assert_allclose(
        _to_numpy(actual.betas_)[:, 0],
        np.asarray(expected.betas_)[:, 0],
        rtol=5e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _to_numpy(actual.coef_)[0],
        np.asarray(expected.coef_)[0],
        rtol=5e-14,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "cov_type,expected_slope_variance",
    [
        ("newey-west", (2.0 / 9.0) * 1.0e308),
        ("nonrobust", (1.0 / 3.0) * 1.0e308),
    ],
)
def test_fama_macbeth_scaled_coefficient_covariance_avoids_representable_overflow(
    cov_type, expected_slope_variance
):
    x_period = np.asarray([-1.0, 0.0, 1.0])
    slopes = np.asarray([-1.0e154, 0.0, 1.0e154])
    X = np.tile(x_period, slopes.size)[:, None]
    y = np.concatenate([slope * x_period for slope in slopes])
    time_ids = np.repeat(np.arange(slopes.size), x_period.size)

    model = FamaMacBeth(cov_type=cov_type, bandwidth=0, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    cov = np.asarray(model.cov_params_)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(
        cov[1, 1], expected_slope_variance, rtol=5e-13, atol=0.0
    )


def test_fama_macbeth_unrepresentable_coefficient_covariance_fails_closed():
    x_period = np.asarray([-1.0, 0.0, 1.0])
    slopes = np.asarray([-1.0e155, 0.0, 1.0e155])
    X = np.tile(x_period, slopes.size)[:, None]
    y = np.concatenate([slope * x_period for slope in slopes])
    time_ids = np.repeat(np.arange(slopes.size), x_period.size)

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="covariance contains non-finite values"):
            FamaMacBeth(bandwidth=0, device="cpu").fit(
                X, y, time_ids=time_ids
            )
'''
if '_large_common_intercept_fixture' in text:
    raise RuntimeError('numeric stability tests already present')
p.write_text(text + addition, encoding='utf-8')

# 4) Physical GPU validator: the new valid-input backend path must be exercised
# on CuPy and Torch CUDA before acceptance.
p = Path('dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py')
text = p.read_text(encoding='utf-8')
anchor = '''def _square_rank_rejection(backend: str):\n'''
if anchor not in text:
    raise RuntimeError('physical insertion anchor not found')
helper = r'''def _numeric_stability_case(backend: str):
    x_period = np.asarray([-1.0, 0.0, 1.0])
    n_periods = 4
    X = np.tile(x_period, n_periods)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time_ids = np.repeat(np.arange(n_periods), x_period.size)

    reference = FamaMacBeth(bandwidth=0, device="cpu").fit(
        X, y, time_ids=time_ids
    )
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(bandwidth=0, device=_device(backend)).fit(
        Xb, yb, time_ids=time_ids
    )
    if actual._backend_name != backend:
        raise AssertionError(
            f"numeric stability case requested {backend}, executed {actual._backend_name}"
        )
    if int(actual._period_svd_fallbacks) != n_periods:
        raise AssertionError(
            "non-finite Gram RHS must force every retained period to SVD fallback: "
            f"fallbacks={actual._period_svd_fallbacks}, periods={n_periods}"
        )
    for label, value in (
        ("betas", actual.betas_),
        ("coef", actual.coef_),
        ("cov_params", actual.cov_params_),
    ):
        if not np.all(np.isfinite(_public_array(value))):
            raise AssertionError(f"numeric stability {label} contains non-finite values")
    np.testing.assert_allclose(
        _public_array(actual.betas_)[:, 0],
        _public_array(reference.betas_)[:, 0],
        rtol=5e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _public_array(actual.coef_)[0],
        _public_array(reference.coef_)[0],
        rtol=5e-13,
        atol=0.0,
    )

    # Separately exercise the scaled coefficient covariance at a magnitude
    # where naive beta' beta overflows but the final covariance is representable.
    slopes = np.asarray([-1.0e154, 0.0, 1.0e154])
    X_cov = np.tile(x_period, slopes.size)[:, None]
    y_cov = np.concatenate([slope * x_period for slope in slopes])
    time_cov = np.repeat(np.arange(slopes.size), x_period.size)
    ref_cov = FamaMacBeth(bandwidth=0, device="cpu").fit(
        X_cov, y_cov, time_ids=time_cov
    )
    X_cov_b, y_cov_b = _arrays(X_cov, y_cov, backend)
    actual_cov = FamaMacBeth(bandwidth=0, device=_device(backend)).fit(
        X_cov_b, y_cov_b, time_ids=time_cov
    )
    if not np.all(np.isfinite(_public_array(actual_cov.cov_params_))):
        raise AssertionError("scaled coefficient covariance is non-finite")
    np.testing.assert_allclose(
        _public_array(actual_cov.cov_params_),
        _public_array(ref_cov.cov_params_),
        rtol=2e-11,
        atol=0.0,
    )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "gram_rhs_overflow_svd_fallbacks": int(actual._period_svd_fallbacks),
        "n_periods": n_periods,
        "common_intercept": float(_public_array(actual.coef_)[0]),
        "scaled_covariance_slope_variance": float(
            _public_array(actual_cov.cov_params_)[1, 1]
        ),
    }


'''
text = text.replace(anchor, helper + anchor, 1)
old_payload = '''            "explicit_device_overrides_foreign_input_container": _explicit_device_cross_container_case(backend),\n            "square_rank_deficient_retained_period_rejected": _square_rank_rejection(backend),\n'''
new_payload = '''            "explicit_device_overrides_foreign_input_container": _explicit_device_cross_container_case(backend),\n            "numeric_stability_and_gram_fallback": _numeric_stability_case(backend),\n            "square_rank_deficient_retained_period_rejected": _square_rank_rejection(backend),\n'''
if old_payload not in text:
    raise RuntimeError('physical payload anchor not found')
p.write_text(text.replace(old_payload, new_payload, 1), encoding='utf-8')

# 5) Documentation and changelog.
for path in ('docs/en/panel/fama-macbeth.md', 'docs/cn/panel/fama-macbeth.md'):
    p = Path(path)
    doc = p.read_text(encoding='utf-8')
    if path.startswith('docs/en/'):
        marker = 'With `device="auto"`, an already NumPy/CuPy/Torch-native input may keep its native backend.'
        note = ('\n\nNumerical safety: the certified Gram fast path treats a non-finite batched right-hand side or solution as unsafe and routes that period through the rank-revealing SVD fallback. The period-coefficient mean and covariance use scaled reductions to avoid avoidable float64 overflow; if the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed instead of publishing clipped or non-finite standard errors.')
    else:
        marker = '当 `device="auto"` 时，已经是 NumPy/CuPy/Torch 原生数组的输入可以保留其原生后端；'
        note = ('\n\n数值安全性：经认证的 Gram 快速路径若发现批量右端项或求解结果为非有限值，会将对应时期转入秩揭示 SVD 回退路径。时期系数均值与协方差采用缩放后的归约以避免可避免的 float64 溢出；若最终协方差本身仍为非有限值或出现负的对角方差，则推断会直接报错，而不会发布截断后或非有限的标准误。')
    if marker not in doc:
        raise RuntimeError(f'doc marker not found: {path}')
    if 'Numerical safety: the certified Gram fast path' in doc or '数值安全性：经认证的 Gram 快速路径' in doc:
        raise RuntimeError(f'numeric stability note already present: {path}')
    p.write_text(doc + note + '\n', encoding='utf-8')

p = Path('CHANGELOG.md')
text = p.read_text(encoding='utf-8')
marker = '- **Fama-MacBeth explicit-device backend authority**:'
pos = text.find(marker)
if pos < 0:
    raise RuntimeError('CHANGELOG PR126 marker not found')
line_end = text.find('\n', pos)
entry = ('\n- **Fama-MacBeth numerical stability**: certified Gram batching now rejects non-finite RHS/solutions and falls back to the shared SVD policy; coefficient-series means/covariances use scaled reductions to avoid representable float64 overflow, while genuinely non-finite or negative-variance covariance fails closed before inference.')
text = text[:line_end] + entry + text[line_end:]
p.write_text(text, encoding='utf-8')
