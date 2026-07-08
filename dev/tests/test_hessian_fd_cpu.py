"""Finite-difference Hessian verification for ordered logit AND probit (CPU)."""
import numpy as np, pytest
from statgpu.linear_model._ordered_logit import OrderedLogitRegression
from statgpu.linear_model._ordered_probit import OrderedProbitRegression
from statgpu.glm_core._family import Binomial, LogitLink, ProbitLink


def _numerical_hessian(fn, theta, h=1e-5):
    d = len(theta)
    H = np.zeros((d, d))
    for i in range(d):
        ei = np.zeros(d); ei[i] = h
        for j in range(i, d):
            ej = np.zeros(d); ej[j] = h
            Hij = (fn(theta + ei + ej) - fn(theta + ei - ej)
                   - fn(theta - ei + ej) + fn(theta - ei - ej)) / (4 * h * h)
            H[i, j] = Hij; H[j, i] = Hij
    return H


def _check(model_cls, link_cls, n=200, p=4, K=4, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    beta_true = np.linspace(0.5, -0.3, p)
    y = np.digitize(0.5 + X @ beta_true + 0.5 * np.random.randn(n),
                     np.linspace(-1, 1, K - 1))
    m = model_cls(n_categories=K, compute_inference=False, max_iter=50)
    m.fit(X, y)
    fam = Binomial(link=link_cls())
    prob = m._ordered_category_probs(X, m.coef_, m._thresh_est, fam, K)
    pc = np.clip(prob, 1e-15, None)
    H_a = np.asarray(m._ordered_hessian_analytical(
        X, y, m.coef_, m._thresh_est, fam, K, prob, pc)) / n
    d = len(m.coef_) + len(m._thresh_est)
    theta = np.concatenate([m.coef_, m._thresh_est])
    def nll_fn(t):
        b = t[:p]; th = t[p:]
        pr = m._ordered_category_probs(X, b, th, fam, K)
        pc2 = np.clip(pr, 1e-15, None)
        return -np.sum(np.log(pc2[y, np.arange(n)])) / n
    H_n = _numerical_hessian(nll_fn, theta)
    diff = np.max(np.abs(H_a - H_n))
    assert diff < 1e-4, f"max|H_diff|={diff:.2e}"


class TestOrderedHessianFD:
    def test_logit(self): _check(OrderedLogitRegression, LogitLink)
    def test_probit(self): _check(OrderedProbitRegression, ProbitLink)
    def test_logit_k4(self): _check(OrderedLogitRegression, LogitLink, K=4, n=300, seed=123)
    def test_probit_k4(self): _check(OrderedProbitRegression, ProbitLink, K=4, n=300, seed=123)
