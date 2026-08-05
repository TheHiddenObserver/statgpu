from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# GLM sample_weight follows the existing analytic-weight convention.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        self._nobs = None
        self._effective_nobs = None
        self._df_resid = None
''',
    '''        self._nobs = None
        self._df_resid = None
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        self._effective_nobs = (
            float(weight_sum) if sample_weight is not None else float(self._nobs)
        )

        family = self._get_family()
''',
    '''        family = self._get_family()
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        # Keep displayed/inference degrees of freedom consistent with the
        # frequency-weight likelihood convention used by diagnostics.
        parameter_count = int(np.asarray(self._params).shape[0])
        self._df_resid = self._effective_nobs - parameter_count
''',
    '''        # Parameter counts are backend-neutral; reading ``shape`` must not
        # trigger an implicit CuPy/Torch-to-NumPy transfer.
        parameter_count = int(self._params.shape[0])
        self._df_resid = float(self._nobs - parameter_count)
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        lines.append(f"  No. Observations: {self._nobs}")
        if (
            self._effective_nobs is not None
            and not np.isclose(self._effective_nobs, float(self._nobs))
        ):
            lines.append(f"  Effective Observations: {self._effective_nobs:g}")
        lines.append(f"  Df Residuals: {self._df_resid:g}")
''',
    '''        lines.append(f"  No. Observations: {self._nobs}")
        lines.append(f"  Df Residuals: {self._df_resid:g}")
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        values = self._loss.per_sample_value(eta, self._y_inf)
        if self._sample_weight_inf is not None:
            weights = xp_asarray(
                self._sample_weight_inf, xp=xp, ref_arr=self._X_design
            )
            values = values * weights
        return -float(xp.sum(values))
''',
    '''        values = self._loss.per_sample_value(eta, self._y_inf)
        if self._sample_weight_inf is not None:
            weights = xp_asarray(
                self._sample_weight_inf, xp=xp, ref_arr=self._X_design
            )
            weight_sum = xp.sum(weights)
            # Analytic weights define a weighted average objective.  Report
            # its n-observation pseudo-loglikelihood so multiplying every
            # weight by a constant does not change diagnostics.
            return -float(self._nobs * xp.sum(values * weights) / weight_sum)
        return -float(xp.sum(values))
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        n = self._effective_nobs if self._effective_nobs is not None else self._nobs
        return -2.0 * ll + k * np.log(max(float(n or 0), 1.0))
''',
    '''        n = self._nobs if self._nobs else 0
        return -2.0 * ll + k * np.log(max(float(n), 1.0))
''',
)

# ---------------------------------------------------------------------------
# Dispersion uses analytic-weight residual sums with row-count degrees of freedom.
# Covariance still divides by sum(weights), cancelling global weight rescaling.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        dispersion = _default_dispersion(
            loss, X, y, coef, n_eff, k, sample_weight=sample_weight
        )
''',
    '''        dispersion = _default_dispersion(
            loss, X, y, coef, X.shape[0], k, sample_weight=sample_weight
        )
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''def _default_dispersion(
    loss, X, y, coef, n_eff, k, *, sample_weight=None
):
''',
    '''def _default_dispersion(
    loss, X, y, coef, n_obs, k, *, sample_weight=None
):
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        return rss / max(n_eff - k, 1)
''',
    '''        return rss / max(n_obs - k, 1)
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        df = max(n_eff - k, 1)
''',
    '''        df = max(n_obs - k, 1)
''',
)

# ---------------------------------------------------------------------------
# Narrow inference solve error handling; do not turn OOM/device errors into
# singular-Hessian messages or silent NaN Wald statistics.
# ---------------------------------------------------------------------------
sandwich = Path("statgpu/inference/_sandwich.py")
text = sandwich.read_text(encoding="utf-8")
anchor = '''def _infer_covariance_convention(cov_type: str, has_curvature: bool) -> str:
    """Map (cov_type, has_curvature) to a covariance convention label."""
    if cov_type == "nonrobust":
        return "penalized_information" if has_curvature else "model_based_nonrobust"
    else:
        return "penalized_sandwich" if has_curvature else "robust_sandwich"


# ---------------------------------------------------------------------------
'''
replacement = '''def _infer_covariance_convention(cov_type: str, has_curvature: bool) -> str:
    """Map (cov_type, has_curvature) to a covariance convention label."""
    if cov_type == "nonrobust":
        return "penalized_information" if has_curvature else "model_based_nonrobust"
    else:
        return "penalized_sandwich" if has_curvature else "robust_sandwich"


def _runtime_error_is_singular(exc: RuntimeError) -> bool:
    """Return whether a backend RuntimeError specifically reports singularity."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "singular",
            "not invertible",
            "zero pivot",
            "rank deficient",
        )
    )


# ---------------------------------------------------------------------------
'''
if anchor not in text:
    raise RuntimeError("sandwich error classifier anchor missing")
text = text.replace(anchor, replacement, 1)
old = '''    try:
        bread_avg = xp.linalg.solve(H_avg, eye)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Singular Hessian in compute_bread_avg. "
            "The design matrix may be rank-deficient or the penalty is too weak. "
            "Consider adding ridge regularization or checking for collinear features."
        ) from e
    except RuntimeError as e:
        # torch raises RuntimeError for singular matrices AND other errors.
        # Only re-wrap if the message suggests a linalg issue.
        msg = str(e).lower()
        if any(kw in msg for kw in ("singular", "linalg", "solve", "lapack")):
            raise np.linalg.LinAlgError(
                "Singular Hessian in compute_bread_avg. "
                "The design matrix may be rank-deficient or the penalty is too weak. "
                "Consider adding ridge regularization or checking for collinear features."
            ) from e
        raise
    except Exception as e:
        # CuPy may raise bare Exception for cuSOLVER failures.
        # Re-wrap but note it may also be OOM or device errors.
        raise np.linalg.LinAlgError(
            "Hessian solve failed in compute_bread_avg. "
            "This may indicate singularity, GPU out-of-memory, or cuSOLVER error. "
            "Consider adding ridge regularization or checking for collinear features."
        ) from e
'''
new = '''    try:
        bread_avg = xp.linalg.solve(H_avg, eye)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Singular Hessian in compute_bread_avg. "
            "The design matrix may be rank-deficient or the penalty is too weak. "
            "Consider adding ridge regularization or checking for collinear features."
        ) from exc
    except RuntimeError as exc:
        if not _runtime_error_is_singular(exc):
            raise
        raise np.linalg.LinAlgError(
            "Singular Hessian in compute_bread_avg. "
            "The design matrix may be rank-deficient or the penalty is too weak. "
            "Consider adding ridge regularization or checking for collinear features."
        ) from exc
'''
if old not in text:
    raise RuntimeError("bread solve exception anchor missing")
text = text.replace(old, new, 1)
old = '''    try:
        # wald = coef' @ cov^{-1} @ coef via solve
        wald_vec = xp.linalg.solve(cov, coef)
        wald_stat = float(xp.dot(coef, wald_vec))
    except (np.linalg.LinAlgError, RuntimeError):
        wald_stat = float("nan")
'''
new = '''    try:
        # wald = coef' @ cov^{-1} @ coef via solve
        wald_vec = xp.linalg.solve(cov, coef)
        wald_stat = float(xp.dot(coef, wald_vec))
    except np.linalg.LinAlgError:
        wald_stat = float("nan")
    except RuntimeError as exc:
        if not _runtime_error_is_singular(exc):
            raise
        wald_stat = float("nan")
'''
if old not in text:
    raise RuntimeError("Wald solve exception anchor missing")
text = text.replace(old, new, 1)
sandwich.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Replace the frequency-weight regression with analytic-weight invariance.
# ---------------------------------------------------------------------------
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
start = text.index("def test_glm_frequency_weights_match_expanded_data_diagnostics():")
end = len(text)
# The function is currently the final test in the file.
replacement = '''def test_glm_analytic_weight_diagnostics_are_scale_invariant():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [2.0], [4.0]], dtype=np.float64)
    y = np.array([-0.4, 0.5, 2.2, 5.1], dtype=np.float64)
    weights = np.array([0.5, 1.5, 2.0, 4.0], dtype=np.float64)

    def fit(current_weights, cov_type="nonrobust"):
        return GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            max_iter=100, tol=1e-12, device="cpu",
            compute_inference=True, cov_type=cov_type,
        ).fit(X, y, sample_weight=current_weights)

    weighted = fit(weights)
    scaled = fit(23.0 * weights)
    np.testing.assert_allclose(weighted.coef_, scaled.coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.intercept_, scaled.intercept_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.loglikelihood, scaled.loglikelihood, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.aic, scaled.aic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.bic, scaled.bic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.bse_, scaled.bse_, rtol=1e-11, atol=1e-11)
    assert weighted._df_resid == X.shape[0] - (X.shape[1] + 1)

    robust = fit(weights, cov_type="hc0")
    robust_scaled = fit(23.0 * weights, cov_type="hc0")
    np.testing.assert_allclose(robust.bse_, robust_scaled.bse_, rtol=1e-11, atol=1e-11)


def test_glm_weighted_loglikelihood_uses_normalized_analytic_weights():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [3.0]], dtype=np.float64)
    y = np.array([-0.2, 0.3, 1.4, 4.0], dtype=np.float64)
    weights = np.array([0.5, 2.0, 1.0, 6.0], dtype=np.float64)
    model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    eta = model.intercept_ + X @ model.coef_
    expected = -X.shape[0] * np.sum(weights * 0.5 * (y - eta) ** 2) / np.sum(weights)
    np.testing.assert_allclose(model.loglikelihood, expected, rtol=1e-12, atol=1e-12)


def test_inference_solve_errors_only_downgrade_true_singularity(monkeypatch):
    import statgpu.inference._sandwich as sandwich
    from statgpu.glm_core._squared import SquaredErrorLoss

    X = np.column_stack([np.ones(5), np.arange(5.0)])
    y = np.arange(5.0)
    coef = np.array([0.0, 1.0])

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(np.linalg, "solve", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        sandwich.compute_bread_avg(SquaredErrorLoss(), X, y, coef)

    assert sandwich._runtime_error_is_singular(
        RuntimeError("matrix is singular")
    )
    assert not sandwich._runtime_error_is_singular(
        RuntimeError("CUDA out of memory")
    )


def test_glm_parameter_count_does_not_use_numpy_array_conversion():
    from pathlib import Path

    source = Path("statgpu/linear_model/_glm_base.py").read_text(encoding="utf-8")
    block = source.split("# Parameter counts are backend-neutral", 1)[1].split(
        "# ---- Store design/loss", 1
    )[0]
    assert "self._params.shape[0]" in block
    assert "np.asarray(self._params)" not in block
'''
tests.write_text(text[:start] + replacement, encoding="utf-8")
