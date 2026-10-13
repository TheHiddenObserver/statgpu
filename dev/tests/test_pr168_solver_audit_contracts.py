"""Small CPU mathematical counterexamples and bilingual solver-doc guards.

These tests check statements repaired after the document audit. They do not
certify solver convergence, comparative speed, inference, or physical GPU runs.
"""
from pathlib import Path
import inspect
import re

import numpy as np
import pytest

from statgpu.losses import BisquareLoss, QuantileLoss
from statgpu.penalties import ElasticNetPenalty

ROOT = Path(__file__).resolve().parents[2]


def doc(language, path):
    return (ROOT / "docs" / language / (path + ".md")).read_text(encoding="utf-8")


def test_bisquare_returns_clipped_curvature_not_true_hessian():
    # REG-02 / ALG-02: the public Hessian interface deliberately drops negative
    # curvature. A finite difference of the actual gradient independently
    # verifies the analytical second derivative at this non-boundary point.
    loss = BisquareLoss(delta=1.0)
    X, y, beta = np.ones((1, 1)), np.array([0.8]), np.zeros(1)
    true_curvature = (1 - 0.8**2) * (1 - 5 * 0.8**2)
    step = 1e-5
    numerical = (loss.gradient(X, y, beta + step)
                 - loss.gradient(X, y, beta - step)) / (2 * step)
    np.testing.assert_allclose(true_curvature, -0.792, atol=1e-14)
    np.testing.assert_allclose(numerical, [true_curvature], atol=1e-8)
    np.testing.assert_array_equal(loss.hessian(X, y, beta), [[0.0]])
    gradient, curvature = loss.fused_gradient_and_hessian(X, y, beta)
    np.testing.assert_allclose(gradient, loss.gradient(X, y, beta), atol=1e-14)
    np.testing.assert_array_equal(curvature, [[0.0]])


def test_weighted_l1_does_not_convexify_bisquare_loss():
    # REG-01 / ALG-01: all beta values are positive, so the weighted-L1 term
    # is locally linear and cannot remove the Bisquare loss's negative curvature.
    loss = BisquareLoss(delta=1.0)
    X, y = np.ones((1, 1)), np.array([1.8])
    def surrogate(beta):
        return loss.value(X, y, np.array([beta])) + 0.2 * abs(beta)
    assert surrogate(1.0) > (surrogate(0.99) + surrogate(1.01)) / 2


def test_bisquare_fixed_delta_ignores_scale_options():
    # REG-03: delta is a public constructor option, with precedence over both
    # automatic-scale settings even when their values would otherwise be invalid.
    assert tuple(inspect.signature(BisquareLoss).parameters) == ("delta", "epsilon", "method")
    loss = BisquareLoss(delta=2.0, epsilon=-1.0, method="unused_in_fixed_mode")
    reference = BisquareLoss(delta=2.0)
    X, y, beta = np.ones((1, 1)), np.array([0.8]), np.zeros(1)
    assert loss.delta == 2.0
    np.testing.assert_allclose(loss.value(X, y, beta), reference.value(X, y, beta))
    for method in ("MAD", "huber_prop2"):
        assert BisquareLoss(method=method).epsilon == 4.685


@pytest.mark.parametrize("tau,old_residual,trial_residual", [(0.9, -1., 1.), (0.1, 1., -1.)])
def test_quantile_irls_tangent_quadratic_is_not_a_global_majorizer(tau, old_residual, trial_residual):
    # ALG-03: no epsilon truncation occurs at either point. The claim fails
    # because the weight was fixed using the OLD residual sign.
    loss = QuantileLoss(quantile=tau)
    epsilon = 1e-8
    weight = (tau if old_residual >= 0 else 1 - tau) / max(abs(old_residual), epsilon)
    def rho(residual):
        return float(loss.per_sample_value(np.zeros(1), np.array([residual]))[0])
    constant = rho(old_residual) - 0.5 * weight * old_residual**2
    def quadratic(residual):
        return 0.5 * weight * residual**2 + constant
    np.testing.assert_allclose(quadratic(old_residual), rho(old_residual), atol=1e-14)
    # per_sample_gradient differentiates with respect to eta, not the residual.
    residual_derivative = -loss.per_sample_gradient(np.zeros(1), np.array([old_residual]))[0]
    np.testing.assert_allclose(weight * old_residual, residual_derivative, atol=1e-14)
    np.testing.assert_allclose(quadratic(trial_residual), 0.1, atol=1e-14)
    np.testing.assert_allclose(rho(trial_residual), 0.9, atol=1e-14)
    assert quadratic(trial_residual) < rho(trial_residual)


def test_quantile_one_irls_step_uses_documented_sign_weights():
    # Tie the mathematical counterexample's weight rule to the actual public
    # IRLS implementation, rather than testing only a standalone formula.
    from statgpu.solvers._convergence import ConvergenceWarning
    X, y = np.ones((2, 1)), np.array([-1., 1.])
    epsilon = 1e-8
    with pytest.warns(ConvergenceWarning):
        beta, count = QuantileLoss(quantile=0.9).irls(
            X, y, init_coef=np.zeros(1), max_iter=1, tol=1e-12, eps=epsilon)
    expected = np.linalg.solve(np.array([[1. + epsilon]]), np.array([0.8]))
    np.testing.assert_allclose(beta, expected, atol=1e-14)
    assert count == 1


def test_gram_diagonal_is_not_a_matrix_majorizer_and_matches_jacobi_update():
    # ALG-03: exact positive-definite Gram matrix, with an indefinite D - Gram.
    from statgpu.solvers._proximal_irls_quantile import _parallel_majorization_step
    gram = np.array([[1., 0.9], [0.9, 1.]])
    X = np.linalg.cholesky(gram).T
    diagonal = np.diag(np.diag(gram))
    np.testing.assert_allclose(X.T @ X, gram, atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(diagonal - gram), [-0.9, 0.9], atol=1e-14)
    y, beta = np.array([0.3, -0.7]), np.array([0.2, -0.1])
    actual = _parallel_majorization_step(
        X, X**2, y, np.ones(2), beta, np.zeros(2), 2, 1e-12, np, "numpy")
    expected = beta + np.linalg.solve(diagonal, X.T @ (y - X @ beta))
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_quadratic_bb_secants_need_not_give_identical_steps():
    # ALG-05: a constant Hessian does not imply BB1 == BB2 or that secant
    # directions have identical curvature. This makes no performance claim.
    H, s = np.diag([1., 2.]), np.ones(2)
    v = H @ s
    bb1, bb2 = (s @ s) / (s @ v), (s @ v) / (v @ v)
    np.testing.assert_allclose([bb1, bb2], [2 / 3, 3 / 5], atol=1e-14)
    assert bb1 != bb2
    directional_steps = [(d @ d) / (d @ H @ d) for d in np.eye(2)]
    np.testing.assert_allclose(directional_steps, [1., 0.5], atol=1e-14)


def test_ordinary_glm_and_shared_penalized_auto_are_separate(monkeypatch):
    # ALG-04: check a real ordinary fit and the shared direct-fit selector.
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized._fit_mixin import _preferred_penalized_glm_solver
    original = GeneralizedLinearModel._fit_irls
    calls = []
    def observed_irls(self, *args, **kwargs):
        calls.append("irls")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(GeneralizedLinearModel, "_fit_irls", observed_irls)
    X = np.arange(-2., 4.).reshape(-1, 1)
    GeneralizedLinearModel(family="binomial", C=0., device="cpu", solver="auto").fit(
        X, np.array([0., 1., 1., 0., 1., 0.]))
    assert calls == ["irls"]
    assert _preferred_penalized_glm_solver("logistic", "l2", "numpy") == "newton"


@pytest.mark.parametrize("language", ["en", "cn"])
def test_elastic_net_teaching_step_runs_and_matches_proximal_api(language):
    # REG-06: execute the teaching pseudocode with an explicit z and step; a
    # missing beta_tilde assignment or a gradient at an unrelated point fails.
    text = doc(language, "models/regression/elastic-net")
    block = next(block for block in re.findall(r"```text\n(.*?)```", text, re.S)
                 if "soft_threshold" in block)
    X = np.array([[1., 0.], [0., 2.], [-1., -2.]])
    y, z = np.array([1., 0.5, -1.5]), np.array([0.4, -0.3])
    step, alpha, l1_ratio = 0.1, 0.7, 0.6
    namespace = dict(X=X, y=y, z=z, n=len(y), step=step, alpha=alpha,
                     l1_ratio=l1_ratio,
                     soft_threshold=lambda a, t: np.sign(a) * np.maximum(np.abs(a) - t, 0))
    exec(compile(block, "elastic-net-teaching-step", "exec"), namespace)
    gradient = X.T @ (X @ z - y) / len(y)
    expected = ElasticNetPenalty(alpha=alpha, l1_ratio=l1_ratio).proximal(z - step * gradient, step)
    np.testing.assert_allclose(namespace["beta"], expected, atol=1e-14)
    assert "beta_tilde = z - step * grad" in block


@pytest.mark.parametrize("language", ["en", "cn"])
def test_docs_distinguish_penalty_convexity_and_loss_curvature(language):
    algorithms = doc(language, "guides/solver-algorithms")
    robust = doc(language, "models/regression/robust")
    reference = doc(language, "reference/losses")
    assert 'id="loss-provided-curvature"' in algorithms
    assert "H_k=B_k+\\alpha I" in algorithms
    assert "H_k=B_k+\\nabla^2P" in algorithms
    for text in (algorithms, robust, reference):
        assert "Bisquare" in text
        assert ("clip" if language == "en" else "截") in text
    assert ("can remain nonconvex" if language == "en" else "仍可能非凸") in robust
    assert ("can remain nonconvex" if language == "en" else "仍可能非凸") in algorithms
    for stale in ("next weighted convex surrogate", "weighted convex surrogate", "实际梯度和 Hessian", "形成加权凸子问题", "转换成局部加权凸问题"):
        assert stale not in algorithms + robust
    parameters = reference.split("### `BisquareLoss`")[1].split("### `FairLoss`")[0]
    assert all(term in parameters for term in ("`delta`", "`None`", "`epsilon`", "`method`", '"MAD"', '"huber_prop2"'))
    assert ("ignored" if language == "en" else "忽略") in parameters


@pytest.mark.parametrize("language", ["en", "cn"])
def test_quantile_docs_do_not_claim_global_mm_or_diagonal_majorization(language):
    algorithms = doc(language, "guides/solver-algorithms")
    quantile_section = algorithms.split("## 1. Proximal IRLS-CD")[1].split("## 2.")[0]
    texts = [quantile_section, doc(language, "models/regression/quantile"),
             doc(language, "guides/solver-penalty-matrix")]
    for text in texts:
        assert ("approximation" if language == "en" else "近似") in text
        assert "MM" in text
        assert "IRLS/MM" not in text
        assert "IRLS majorization" not in text
        assert "IRLS 上界" not in text
        assert "IRLS 二次上界" not in text
        assert "diagonal-majorization update" not in text
        assert "对角上界更新" not in text
    for term in ("Q(r)=0.05r^2+0.05", "0.1<", "0.9", "Jacobi", "Gram"):
        assert term in quantile_section
    assert ("epsilon floor is inactive" if language == "en" else "没有触发 epsilon 下限") in quantile_section


@pytest.mark.parametrize("language", ["en", "cn"])
def test_auto_scope_and_quadratic_policy_are_explicit(language):
    algorithms = doc(language, "guides/solver-algorithms")
    matrix = doc(language, "guides/solver-penalty-matrix")
    for text in (algorithms, matrix):
        assert "`PenalizedGLM`" in text and "`GeneralizedLinearModel`" in text
        assert "`LogisticRegression`" in text
    assert ("fixed Lipschitz step for quadratic losses" if language == "en" else "二次损失使用固定 Lipschitz 步长") in algorithms
    assert "do not gain useful adaptation" not in algorithms
    assert "不从 BB 曲率获得额外收益" not in algorithms


@pytest.mark.parametrize("language", ["en", "cn"])
def test_regression_wording_preserves_limitations(language):
    logistic = doc(language, "models/regression/logistic-regression")
    row = next(line for line in logistic.splitlines() if line.startswith("| `n_jobs`"))
    assert ("does not use it to control parallel fitting" if language == "en" else "不使用它控制并行拟合") in row
    for model in ("mcp", "scad"):
        text = doc(language, "models/regression/" + model)
        assert ("This wrapper has no built-in CV" if language == "en" else "该封装类不提供") in text
        assert "These wrappers" not in text and "这两个封装类" not in text
    glm = doc(language, "models/regression/generalized-linear-model")
    assert "reserved structured penalties" not in glm
    row = next(line for line in glm.splitlines() if line.startswith("| `penalty`"))
    assert "solver-penalty-matrix.md" in row


def test_ols_interval_is_identified_by_slope_not_table_row():
    text = doc("en", "models/regression/linear-regression")
    assert "third slope (the fourth row)" in text
    assert "the third interval happens" not in text


def test_cn_analytic_weight_terminology_and_legacy_anchors():
    for path in ("guides/solver-algorithms", "guides/solver-penalty-matrix",
                 "models/regression/quantile", "reference/losses"):
        text = re.sub(r'<a[^>]*></a>', '', doc("cn", path))
        assert "解析权重" not in text and "解析训练权重" not in text
        assert "解析 `sample_weight`" not in text
        assert "分析权重" in text or "分析训练权重" in text
    algorithms = doc("cn", "guides/solver-algorithms")
    for anchor in ("凸近似问题", "解析-sample_weight", "l-bfgs-的解析-sample_weight"):
        assert f'id="{anchor}"' in algorithms
    assert 'id="convex-surrogate"' in doc("en", "guides/solver-algorithms")


@pytest.mark.parametrize("language", ["en", "cn"])
def test_fista_bb_restart_docs_use_old_evaluation_point_and_accepted_step(language):
    # ALG-A1 is coordinated with the separately authorized runtime repair.
    text = doc(language, "guides/solver-algorithms").split("## 4. FISTA-BB")[1].split("## 5.")[0]
    assert r"\left(y_k-\beta_{k+1}\right)^\top" in text
    assert r"\left(y_{k+1}-\beta_{k+1}\right)^\top" not in text
    assert "use_restart=True" in text
    assert ("final accepted point" if language == "en" else "最终接受的点") in text
    assert ("second iteration" if language == "en" else "第二次迭代") in text
    assert ("bypass this restart test" if language == "en" else "不使用上述重启判据") in text
    assert ("Inverse-Gaussian losses disable momentum" if language == "en" else "Inverse-Gaussian 损失禁用动量") in text
