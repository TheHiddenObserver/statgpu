from pathlib import Path


def replace_between(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    text = text[:start] + replacement + text[end:]
    p.write_text(text, encoding="utf-8")


# 1) Shared SVD RHS scaling: scale only as much as the orthogonal projection
# bound requires. Max-normalizing every RHS can underflow an unrelated small
# component even for X=I.
p = Path("statgpu/panel/_linalg.py")
text = p.read_text(encoding="utf-8")
start = text.index("def _scaled_lstsq_rhs")
end = text.index("\ndef panel_svd_pseudoinverse", start)
replacement = '''def _scaled_lstsq_rhs(y, xp, *, batched: bool):
    """Scale responses only enough to keep ``U.T @ y`` in float64 range.

    For an orthonormal SVD factor ``U``, every projected coordinate obeys
    ``|u.T @ y| <= sqrt(n_obs) * max(abs(y))``.  Use this bound to apply only
    the dimensionless down-scaling needed to keep the projection below half of
    the float64 maximum.  Unlike normalizing every response by its maximum, the
    scale factor is at most ``2 * sqrt(n_obs)`` for finite float64 input, so a
    small but representable component is not needlessly collapsed merely
    because another observation is very large.
    """
    namespace = getattr(xp, "__name__", "")
    ndim = int(getattr(y, "ndim", 0))
    n_obs = int(y.shape[1] if batched else y.shape[0])
    projection_limit = np.finfo(np.float64).max / (
        2.0 * np.sqrt(float(max(n_obs, 1)))
    )

    def _safe_scale(max_abs):
        required = max_abs / float(projection_limit)
        return xp.where(required > 1.0, required, xp.ones_like(required))

    if batched:
        if ndim == 2:
            max_abs = (
                xp.max(xp.abs(y), dim=1).values
                if namespace == "torch"
                else xp.max(xp.abs(y), axis=1)
            )
            safe_scale = _safe_scale(max_abs)
            return y / safe_scale[:, None], safe_scale[:, None]
        if ndim == 3:
            max_abs = (
                xp.max(xp.abs(y), dim=1).values
                if namespace == "torch"
                else xp.max(xp.abs(y), axis=1)
            )
            safe_scale = _safe_scale(max_abs)
            return y / safe_scale[:, None, :], safe_scale[:, None, :]
        raise ValueError("batched panel response must have shape (batch, n_obs[, n_targets])")

    if ndim == 1:
        max_abs = xp.max(xp.abs(y))
        safe_scale = _safe_scale(max_abs)
        return y / safe_scale, safe_scale
    if ndim == 2:
        max_abs = (
            xp.max(xp.abs(y), dim=0).values
            if namespace == "torch"
            else xp.max(xp.abs(y), axis=0)
        )
        safe_scale = _safe_scale(max_abs)
        return y / safe_scale[None, :], safe_scale[None, :]
    raise ValueError("panel response must be one- or two-dimensional")

'''
text = text[:start] + replacement + text[end+1:]
p.write_text(text, encoding="utf-8")

# 2) Fama-MacBeth: use the same certified Gram/SVD policy on NumPy, CuPy and
# Torch; scale coefficient covariance per coordinate; avoid 0/0 statistics.
p = Path("statgpu/panel/_fama_macbeth.py")
text = p.read_text(encoding="utf-8")
text = text.replace("def _gpu_certified_period_betas(", "def _certified_period_betas(")
text = text.replace(
    '    """Solve GPU periods with a conservative Gram fast path and SVD fallback.\n',
    '    """Solve retained periods with a conservative Gram fast path and SVD fallback.\n',
)
start = text.index('        if backend_name in {"torch", "cupy"}:')
end = text.index("        T = int(betas.shape[0])", start)
new_block = '''        betas, solver_batches, rank_syncs, svd_fallbacks = (
            _certified_period_betas(
                X_design,
                y_arr,
                time_codes,
                counts,
                time_labels,
                min_obs_per_period=self.min_obs_per_period,
                backend_name=backend_name,
                xp=xp,
            )
        )
        if betas is None:
            raise ValueError("No time periods with enough observations")
        self._period_solver_mode = (
            "gram-certified"
            if int(svd_fallbacks) == 0
            else "gram-certified+svd-fallback"
        )
        self._period_solver_batches = int(solver_batches)
        self._period_rank_syncs = int(rank_syncs)
        self._period_svd_fallbacks = int(svd_fallbacks)

'''
text = text[:start] + new_block + text[end:]
old = '''        centered_scale = xp.max(xp.abs(beta_centered))
        safe_centered_scale = xp.where(
            centered_scale > 0.0,
            centered_scale,
            xp.ones_like(centered_scale),
        )
        beta_centered_scaled = beta_centered / safe_centered_scale
'''
new = '''        # Scale each coefficient coordinate independently. A single global
        # scale can underflow a small coordinate's quadratic variation to zero
        # when another coefficient varies at a much larger magnitude.
        if xp.__name__ == "torch":
            centered_scale = xp.max(xp.abs(beta_centered), dim=0).values
        else:
            centered_scale = xp.max(xp.abs(beta_centered), axis=0)
        safe_centered_scale = xp.where(
            centered_scale > 0.0,
            centered_scale,
            xp.ones_like(centered_scale),
        )
        beta_centered_scaled = beta_centered / safe_centered_scale
'''
if old not in text:
    raise RuntimeError("centered covariance scale anchor not found")
text = text.replace(old, new, 1)
old = '''        cov_params = (cov_scaled * safe_centered_scale) * safe_centered_scale
'''
new = '''        cov_params = (
            cov_scaled * safe_centered_scale[:, None]
        ) * safe_centered_scale[None, :]
'''
if old not in text:
    raise RuntimeError("covariance rescale anchor not found")
text = text.replace(old, new, 1)
old = '''        bse = xp.sqrt(xp.maximum(diagonal, xp.zeros_like(diagonal)))
        tvalues = avg_beta / bse
'''
new = '''        bse = xp.sqrt(xp.maximum(diagonal, xp.zeros_like(diagonal)))
        # Match the shared panel inference convention at an exactly zero
        # estimated variance: 0/0 should not leak NaN into the public result
        # surface. Positive standard errors are unchanged.
        bse_for_stat = xp.where(
            bse > 0.0,
            bse,
            xp.full_like(bse, np.finfo(np.float64).tiny),
        )
        tvalues = avg_beta / bse_for_stat
'''
if old not in text:
    raise RuntimeError("FMB t-stat anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# 3) Shared linalg regression: identity design with huge and small RHS entries.
p = Path("dev/tests/test_fama_macbeth_batched_solver.py")
text = p.read_text(encoding="utf-8")
anchor = '''def test_panel_lstsq_batched_scales_large_finite_rhs_before_projection():\n'''
insert = '''def test_panel_lstsq_preserves_mixed_dynamic_range_identity_rhs():
    X = np.eye(2, dtype=np.float64)
    y = np.asarray([1.0e308, 1.0e-20], dtype=np.float64)
    params, rank = panel_lstsq(X, y, np)
    assert rank == 2
    np.testing.assert_allclose(params[0], 1.0e308, rtol=5e-15, atol=0.0)
    np.testing.assert_allclose(params[1], 1.0e-20, rtol=5e-15, atol=0.0)


def test_panel_lstsq_deferred_rank_preserves_mixed_dynamic_range_identity_rhs():
    from statgpu.panel._linalg import panel_lstsq_deferred_rank

    X = np.eye(2, dtype=np.float64)
    y = np.asarray([1.0e308, 1.0e-20], dtype=np.float64)
    params, rank = panel_lstsq_deferred_rank(X, y, np)
    assert int(rank) == 2
    np.testing.assert_allclose(params[0], 1.0e308, rtol=5e-15, atol=0.0)
    np.testing.assert_allclose(params[1], 1.0e-20, rtol=5e-15, atol=0.0)


def test_panel_lstsq_batched_preserves_mixed_dynamic_range_identity_rhs():
    torch = pytest.importorskip("torch")
    X = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    y = torch.tensor(
        [[1.0e308, 1.0e-20], [-1.0e308, -1.0e-20]],
        dtype=torch.float64,
    )
    params, ranks = panel_lstsq_batched(X, y, torch)
    assert ranks.tolist() == [2, 2]
    expected = np.asarray(
        [[1.0e308, 1.0e-20], [-1.0e308, -1.0e-20]], dtype=np.float64
    )
    np.testing.assert_allclose(
        params.detach().cpu().numpy(), expected, rtol=5e-15, atol=0.0
    )


'''
if anchor not in text:
    raise RuntimeError("batched solver test anchor not found")
text = text.replace(anchor, insert + anchor, 1)
old = '''    assert expected._period_solver_mode == "serial"
    assert expected._period_solver_batches == expected.n_periods
    assert expected._period_rank_syncs == expected.n_periods
    assert expected._period_svd_fallbacks == 0
'''
new = '''    assert expected._period_solver_mode == "gram-certified"
    assert expected._period_solver_batches == 1
    assert expected._period_rank_syncs == 1
    assert expected._period_svd_fallbacks == 0
'''
if old not in text:
    raise RuntimeError("NumPy solver provenance test anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# 4) FMB mixed coefficient scales + exact-zero-variance inference, and update
# the old NumPy-SVD ownership assertion to the unified certified path.
p = Path("dev/tests/test_fama_macbeth_review_fixes.py")
text = p.read_text(encoding="utf-8")
old = '''def test_fama_macbeth_reuses_one_rank_revealing_svd_per_retained_period(monkeypatch):
    x, y, _labels, numeric = _chronology_fixture()
    calls = []
    original = panel_linalg._svd_inverse_factors

    def tracked(X, xp):
        calls.append((int(X.shape[0]), int(X.shape[1])))
        return original(X, xp)

    monkeypatch.setattr(panel_linalg, "_svd_inverse_factors", tracked)
    model = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=numeric
    )

    assert model.n_periods == 3
    assert calls == [(5, 2), (5, 2), (5, 2)]
'''
new = '''def test_fama_macbeth_numpy_uses_certified_batch_before_svd_fallback(monkeypatch):
    x, y, _labels, numeric = _chronology_fixture()
    calls = []
    original = panel_linalg._svd_inverse_factors

    def tracked(X, xp):
        calls.append((int(X.shape[0]), int(X.shape[1])))
        return original(X, xp)

    monkeypatch.setattr(panel_linalg, "_svd_inverse_factors", tracked)
    model = FamaMacBeth(bandwidth=1, device="cpu").fit(
        x[:, None], y, time_ids=numeric
    )

    assert model.n_periods == 3
    assert model._period_solver_mode == "gram-certified"
    assert model._period_solver_batches == 1
    assert model._period_rank_syncs == 1
    assert model._period_svd_fallbacks == 0
    assert calls == []
'''
if old not in text:
    raise RuntimeError("old NumPy SVD ownership test not found")
text = text.replace(old, new, 1)
append = r'''


def _mixed_coefficient_scale_fixture():
    X_period = np.asarray(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    period_betas = np.asarray(
        [
            [0.0, -1.0e154, -1.0e-10],
            [0.0, 0.0, 0.0],
            [0.0, 1.0e154, 1.0e-10],
        ],
        dtype=np.float64,
    )
    design = np.column_stack([np.ones(X_period.shape[0]), X_period])
    X = np.tile(X_period, (period_betas.shape[0], 1))
    y = np.concatenate([design @ beta for beta in period_betas])
    time = np.repeat(np.arange(period_betas.shape[0]), X_period.shape[0])
    expected_cov = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0e308 / 3.0, 1.0e144 / 3.0],
            [0.0, 1.0e144 / 3.0, 1.0e-20 / 3.0],
        ],
        dtype=np.float64,
    )
    return X, y, time, period_betas, expected_cov


def _assert_mixed_scale_fmb_result(model, period_betas, expected_cov):
    np.testing.assert_allclose(
        _to_numpy(model.betas_), period_betas, rtol=2e-13, atol=0.0
    )
    np.testing.assert_allclose(
        _to_numpy(model.cov_params_), expected_cov, rtol=3e-13, atol=0.0
    )
    assert _to_numpy(model.cov_params_)[2, 2] > 0.0
    assert np.all(np.isfinite(_to_numpy(model.tvalues_)))
    assert np.all(np.isfinite(_to_numpy(model.pvalues_)))
    assert _to_numpy(model.tvalues_)[0] == pytest.approx(0.0)
    assert _to_numpy(model.pvalues_)[0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        _to_numpy(model.conf_int_)[0], np.asarray([0.0, 0.0]), rtol=0.0, atol=0.0
    )


def test_fama_macbeth_preserves_coordinatewise_covariance_scale_numpy():
    X, y, time, period_betas, expected_cov = _mixed_coefficient_scale_fixture()
    model = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(
        X, y, time_ids=time
    )
    assert model._period_solver_mode == "gram-certified"
    _assert_mixed_scale_fmb_result(model, period_betas, expected_cov)


def test_fama_macbeth_preserves_coordinatewise_covariance_scale_torch_cpu():
    torch = pytest.importorskip("torch")
    X, y, time, period_betas, expected_cov = _mixed_coefficient_scale_fixture()
    model = FamaMacBeth(cov_type="nonrobust").fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=time,
    )
    assert model._backend_name == "torch"
    assert model._period_solver_mode == "gram-certified"
    _assert_mixed_scale_fmb_result(model, period_betas, expected_cov)
'''
text += append
p.write_text(text, encoding="utf-8")

# 5) Physical validator: add an analytic mixed-scale case to the existing
# numerical stability payload.
p = Path("dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py")
text = p.read_text(encoding="utf-8")
needle = '''    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "gram_rhs_overflow_svd_fallbacks": int(actual._period_svd_fallbacks),
'''
insert = '''    # A second, analytic fixture gives different coefficient coordinates
    # radically different period-series scales while keeping the design exactly
    # orthogonal. The small variance is representable and must not underflow to
    # zero just because another coefficient varies near 1e154.
    X_period = np.asarray(
        [[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    expected_betas = np.asarray(
        [[0.0, -1.0e154, -1.0e-10], [0.0, 0.0, 0.0], [0.0, 1.0e154, 1.0e-10]],
        dtype=np.float64,
    )
    design = np.column_stack([np.ones(X_period.shape[0]), X_period])
    X_mixed = np.tile(X_period, (3, 1))
    y_mixed = np.concatenate([design @ beta for beta in expected_betas])
    time_mixed = np.repeat(np.arange(3), X_period.shape[0])
    expected_cov = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 1.0e308 / 3.0, 1.0e144 / 3.0], [0.0, 1.0e144 / 3.0, 1.0e-20 / 3.0]],
        dtype=np.float64,
    )
    X_mixed_b, y_mixed_b = _arrays(X_mixed, y_mixed, backend)
    mixed = FamaMacBeth(cov_type="nonrobust", device=_device(backend)).fit(
        X_mixed_b, y_mixed_b, time_ids=time_mixed
    )
    np.testing.assert_allclose(
        _public_array(mixed.betas_), expected_betas, rtol=3e-12, atol=0.0
    )
    np.testing.assert_allclose(
        _public_array(mixed.cov_params_), expected_cov, rtol=3e-12, atol=0.0
    )
    if _public_array(mixed.cov_params_)[2, 2] <= 0.0:
        raise AssertionError("small coordinate variance underflowed to zero")
    if not np.all(np.isfinite(_public_array(mixed.tvalues_))):
        raise AssertionError("zero-variance coefficient leaked non-finite statistic")
    if not np.all(np.isfinite(_public_array(mixed.pvalues_))):
        raise AssertionError("zero-variance coefficient leaked non-finite p-value")

'''
if needle not in text:
    raise RuntimeError("physical numeric return anchor not found")
text = text.replace(needle, insert + needle, 1)
old = '''        "scaled_covariance_slope_variance": float(
            _public_array(actual_cov.cov_params_)[1, 1]
        ),
'''
new = '''        "scaled_covariance_slope_variance": float(
            _public_array(actual_cov.cov_params_)[1, 1]
        ),
        "mixed_scale_small_variance": float(
            _public_array(mixed.cov_params_)[2, 2]
        ),
        "mixed_scale_zero_variance_statistic": float(
            _public_array(mixed.tvalues_)[0]
        ),
'''
if old not in text:
    raise RuntimeError("physical numeric payload anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

print("PR126 scale-axis review patch applied")
