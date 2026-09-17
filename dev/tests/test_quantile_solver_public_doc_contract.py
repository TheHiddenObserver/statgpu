"""Runtime-help contract for guarded public Quantile solver boundaries."""

import inspect
from pathlib import Path

from statgpu import glm_core, solvers
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


_ROOT = Path(__file__).resolve().parents[2]


def test_guarded_public_solver_docstrings_expose_quantile_boundary():
    fista_doc = inspect.getdoc(solvers.fista_bb_solver) or ""
    admm_doc = inspect.getdoc(solvers.admm_solver) or ""
    fista_text = " ".join(fista_doc.split())
    admm_text = " ".join(admm_doc.split())

    assert "Quantile" in fista_text
    assert "does not support Quantile" in fista_text
    assert "alternating BB1/BB2 steps" in fista_text
    assert "Supports numpy / cupy / torch backends" in fista_text

    assert "Nesterov-accelerated gradient descent" in admm_text
    assert "does not support Quantile" in admm_text
    assert "cg_max_iter" in admm_doc

    # glm_core re-exports the same guarded public callables, so runtime help
    # must remain identical across both public import paths.
    assert inspect.getdoc(glm_core.fista_bb_solver) == fista_doc
    assert inspect.getdoc(glm_core.admm_solver) == admm_doc

    assert "maintained" not in fista_text.lower()
    assert "maintained" not in admm_text.lower()
    assert "fail closed" not in fista_text.lower()
    assert "fail closed" not in admm_text.lower()


def test_penalized_glm_runtime_help_lists_public_admm_solver():
    doc = inspect.getdoc(PenalizedGeneralizedLinearModel) or ""
    assert "'admm'" in doc
    assert "Support depends on the loss and penalty" in doc
    assert "unsupported explicit combinations raise an error" in doc


def test_typed_quantile_runtime_help_names_ordinary_fista_boundary():
    doc = " ".join((inspect.getdoc(PenalizedQuantileRegression) or "").split())
    assert "L1/ElasticNet objectives use ordinary FISTA" in doc
    assert "explicit ordinary ``solver='fista'`` is also supported" in doc
    assert "executes the generic FISTA engine rather than being silently substituted by IRLS" in doc
    assert "IRLS remains the default automatic choice" in doc
    assert "FISTA-BB and shared ADMM do not support Quantile" in doc
    assert "shared L-BFGS implementation assumes a smooth loss gradient" in doc
    assert "maintained" not in doc.lower()
    assert "fail closed" not in doc.lower()


def test_typed_quantile_runtime_help_documents_loss_kwargs_precedence():
    doc = " ".join((inspect.getdoc(PenalizedQuantileRegression) or "").split())
    assert "loss_kwargs={'quantile': q}" in doc
    assert "takes precedence over the typed ``quantile=`` argument" in doc
    assert "public ``quantile`` attribute retains the outer constructor value" in doc


def test_solver_algorithm_pages_preserve_explicit_quantile_fista_contract():
    en = (_ROOT / "docs/en/guides/solver-algorithms.md").read_text(encoding="utf-8")
    cn = (_ROOT / "docs/cn/guides/solver-algorithms.md").read_text(encoding="utf-8")

    assert "explicit `solver=\"fista\"` executes ordinary FISTA" in en
    assert "not a claim that textbook smooth-gradient FISTA convergence theory applies" in en
    assert "only an explicit `solver=\"fista\"` request selects this ordinary-FISTA path" in en
    assert "an explicit `solver=\"fista\"` request remains authoritative for both CV child fits" in en
    assert "explicit `solver=\"fista\"` fails rather than being silently substituted by IRLS" not in en
    assert "maintained" not in en.lower()
    assert "fail closed" not in en.lower()

    assert "显式 `solver=\"fista\"` 则真正执行普通 FISTA" in cn
    assert "并不声称 pinball loss 满足教科书式 smooth-gradient FISTA 的收敛假设" in cn
    assert "只有显式 `solver=\"fista\"` 才选择这条普通 FISTA 路径" in cn
    assert "该请求对 CV 子拟合和最终全数据重拟合都保持有效并执行普通 FISTA" in cn
    assert "显式 `solver=\"fista\"` 会失败而不是被静默替换成 IRLS" not in cn
    assert "维护" not in cn
    assert "fail closed" not in cn.lower()
