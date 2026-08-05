from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Single response-domain contract for every GLM loss/backend/solver.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/glm_core/_base.py",
    '''    def _mu_from_eta(self, eta):
        """Link inverse: μ = g⁻¹(η). Override for clipping."""
        return eta  # default: identity link

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
''',
    '''    def _mu_from_eta(self, eta):
        """Link inverse: μ = g⁻¹(η). Override for clipping."""
        return eta  # default: identity link

    def validate_response(self, y):
        """Validate the response domain on its current array backend.

        ``y_type`` is the shared public contract for every solver.  Only the
        final scalar boolean is synchronized; Torch/CuPy response arrays are
        never copied to NumPy for validation.
        """
        from statgpu.backends._array_ops import _xp

        xp = _xp(y)
        invalid = xp.any(~xp.isfinite(y))
        y_type = str(getattr(self, "y_type", "continuous")).lower()
        if y_type == "binary":
            invalid = invalid | xp.any(y < 0) | xp.any(y > 1)
            requirement = "values in [0, 1]"
        elif y_type in ("count", "nonnegative"):
            invalid = invalid | xp.any(y < 0)
            requirement = "non-negative values"
        elif y_type == "positive":
            invalid = invalid | xp.any(y <= 0)
            requirement = "strictly positive values"
        else:
            requirement = "finite values"

        if bool(invalid.item() if hasattr(invalid, "item") else invalid):
            raise ValueError(
                f"{self.name} response requires finite {requirement}."
            )
        return y

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
''',
)

# Public GLM fit validates before dispatch, so IRLS/FISTA/Newton/LBFGS and
# formula/direct paths share exactly the same response-domain behavior.
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        family = self._get_family()
        _solver_lower = self._solver.lower() if isinstance(self._solver, str) else self._solver
''',
    '''        family = self._get_family()
        fit_loss = self._resolve_loss_for_inference()
        fit_loss.validate_response(y_arr)
        _solver_lower = self._solver.lower() if isinstance(self._solver, str) else self._solver
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        self._loss = self._resolve_loss_for_inference()
''',
    '''        self._loss = fit_loss
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        Computed as -sum(loss.per_sample_value(eta, y)).  Additive constants
        that do not depend on the parameters (e.g. -log(y!) for Poisson,
        -n log(2πσ²)/2 for Gaussian) are omitted.  ΔAIC / ΔBIC comparisons
        between nested models on the same data remain valid; absolute values
        should not be compared with statsmodels or R.
''',
    '''        Without sample weights this is ``-sum(per_sample_loss)``.  With
        analytic sample weights it is the negative weighted-average loss
        multiplied by the original row count, so globally rescaling all
        weights leaves loglikelihood, AIC, and BIC unchanged.  Additive
        constants independent of the parameters are omitted; absolute values
        should therefore not be compared directly with statsmodels or R.
''',
)

# Direct IRLSSolver users receive the same loss-owned validation rather than
# the previous incomplete family-name switch.
irls = Path("statgpu/glm_core/_irls.py")
text = irls.read_text(encoding="utf-8")
start = text.index("    if backend == \"torch\":\n        import torch\n        invalid_y")
end_marker = '''        raise ValueError(
            f"{family_name} IRLS requires finite, {requirement} y values."
        )
'''
end = text.index(end_marker, start) + len(end_marker)
text = text[:start] + '''    objective_loss.validate_response(y_work)
''' + text[end:]
irls.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Regression matrix: family x solver, formula, direct IRLS, and physical GPU.
# ---------------------------------------------------------------------------
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_GLM_RESPONSE_DOMAIN_MATRIX_TESTS"
if marker not in text:
    text += '''

# PR87_GLM_RESPONSE_DOMAIN_MATRIX_TESTS
@pytest.mark.parametrize("solver", ["irls", "fista", "newton", "lbfgs"])
@pytest.mark.parametrize(
    "family,bad_y,message",
    [
        ("binomial", np.array([0.0, 1.0, -0.1, 0.5]), r"logistic response.*\\[0, 1\\]"),
        ("binomial", np.array([0.0, 1.0, 1.1, 0.5]), r"logistic response.*\\[0, 1\\]"),
        ("poisson", np.array([0.0, 1.0, -1.0, 2.0]), "poisson response.*non-negative"),
        ("gamma", np.array([1.0, 2.0, 0.0, 3.0]), "gamma response.*strictly positive"),
        ("inverse_gaussian", np.array([1.0, 2.0, -0.1, 3.0]), "inverse_gaussian response.*strictly positive"),
        ("negative_binomial", np.array([0.0, 1.0, -1.0, 2.0]), "negative_binomial response.*non-negative"),
        ("tweedie", np.array([0.0, 1.0, -0.1, 2.0]), "tweedie response.*non-negative"),
    ],
)
def test_glm_response_domain_is_validated_before_every_solver(family, bad_y, message, solver):
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.arange(8.0, dtype=np.float64).reshape(4, 2)
    with pytest.raises(ValueError, match=message):
        GeneralizedLinearModel(
            family=family,
            solver=solver,
            C=0.0,
            device="cpu",
            compute_inference=False,
        ).fit(X, bad_y)


def test_binomial_glm_accepts_fractional_responses_in_unit_interval():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    model = GeneralizedLinearModel(
        family="binomial", solver="irls", C=0.0,
        max_iter=100, device="cpu", compute_inference=False,
    ).fit(X, y)
    assert np.isfinite(model.coef_).all()


def test_formula_response_domain_validation_occurs_after_patsy_row_selection():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [0.0, 1.0, -2.0, 3.0], "x": [0.0, 1.0, np.nan, 3.0]}
    )
    # The negative response belongs to the row Patsy removes.  Retained rows
    # are valid Poisson responses and must fit successfully.
    model = GeneralizedLinearModel(
        family="poisson", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(formula="y ~ x", data=data)
    assert np.isfinite(model.coef_).all()

    data.loc[1, "y"] = -1.0
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        ).fit(formula="y ~ x", data=data)


def test_direct_irls_solver_uses_loss_owned_response_validation():
    from statgpu.glm_core._family import Poisson
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        IRLSSolver(Poisson()).fit(
            np.ones((3, 1)), np.array([0.0, -1.0, 2.0]), backend="numpy"
        )


def test_torch_glm_response_domain_validation_stays_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X = torch.arange(8.0, dtype=torch.float64, device="cuda").reshape(4, 2)
    y = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_glm_response_domain_validation_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model import GeneralizedLinearModel

    X = cp.arange(8.0, dtype=cp.float64).reshape(4, 2)
    y = cp.asarray([1.0, 2.0, 0.0, 3.0], dtype=cp.float64)
    with pytest.raises(ValueError, match="gamma response.*strictly positive"):
        GeneralizedLinearModel(
            family="gamma", solver="irls", C=0.0,
            device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
'''
    tests.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Maintained user-facing change records.
# ---------------------------------------------------------------------------
replace_once(
    "CHANGELOG.md",
    '''  corrected Gaussian GLM FISTA to use weighted centering and the intended
  weighted squared-loss intercept.
''',
    '''  corrected Gaussian GLM FISTA to use weighted centering and the intended
  weighted squared-loss intercept.
- Unified analytic-weight GLM semantics across IRLS ridge scaling, line search,
  pseudo-loglikelihood, AIC/BIC, dispersion, and sandwich inference; centralized
  active GLM Torch compilation; narrowed singular-system fallbacks; and added
  backend-native response-domain validation for every supported GLM family.
''',
)
replace_once(
    "docs/en/changelog.md",
    '''- Gaussian GLM FISTA now profiles the intercept with weighted feature and
  response means, matching the declared weighted squared-loss objective and
  closed-form weighted least squares when the penalty is zero.
''',
    '''- Gaussian GLM FISTA now profiles the intercept with weighted feature and
  response means, matching the declared weighted squared-loss objective and
  closed-form weighted least squares when the penalty is zero.
- GLM sample weights now follow one analytic-weight convention across IRLS
  ridge scaling, line search, normalized pseudo-loglikelihood, AIC/BIC,
  dispersion, and sandwich inference. Globally rescaling weights leaves fitted
  parameters and reported diagnostics unchanged.
- Every supported GLM family now enforces its response domain before any solver
  dispatch, using NumPy, Torch, or CuPy reductions on the selected backend.
  Active IRLS/FISTA helper compilation uses the centralized compile policy, and
  unrelated linear-algebra/device failures are no longer masked as fallback.
''',
)
replace_once(
    "docs/cn/changelog.md",
    '''- Gaussian GLM 的 FISTA 路径改用加权的特征均值与响应均值 profile intercept；
  在零惩罚时与闭式 weighted least squares 一致，不再优化错误的未加权中心化目标。
''',
    '''- Gaussian GLM 的 FISTA 路径改用加权的特征均值与响应均值 profile intercept；
  在零惩罚时与闭式 weighted least squares 一致，不再优化错误的未加权中心化目标。
- GLM 的 sample weight 统一采用 analytic-weight 语义，覆盖 IRLS ridge scaling、
  line search、归一化 pseudo-loglikelihood、AIC/BIC、dispersion 与 sandwich inference；
  对全部权重作统一倍数缩放不会改变估计量或报告的诊断量。
- 所有支持的 GLM family 都在 solver dispatch 之前执行 backend-native response-domain
  validation；active IRLS/FISTA 编译统一走 centralized compile policy，且不再把无关的
  线性代数、显存或 device 错误伪装成 fallback。
''',
)
