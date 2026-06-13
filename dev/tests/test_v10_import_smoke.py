"""V10 import compatibility smoke tests.

These tests intentionally avoid numerical accuracy checks.  Accuracy, GPU, and
external framework comparisons are run on the remote myconda environment.
"""


def test_glm_penalized_formula_public_imports():
    from statgpu import (
        ElasticNet,
        Lasso,
        LinearRegression,
        PoissonRegression,
        Ridge,
    )
    from statgpu.linear_model import (
        GeneralizedLinearModel,
        OrderedLogitRegression,
        OrderedProbitRegression,
        PenalizedGeneralizedLinearModel,
        PenalizedLinearRegression,
        PenalizedLogisticRegression,
        PenalizedPoissonRegression,
    )
    from statgpu.core.formula import FormulaParser, parse_formula
    from statgpu.glm_core import (
        GLMLoss,
        GLMFamily,
        fista_solver,
        get_glm_loss,
        lbfgs_solver,
        list_glm_losses,
        newton_solver,
        register_glm_loss,
    )

    assert LinearRegression is not None
    assert GeneralizedLinearModel is not None
    assert PenalizedGeneralizedLinearModel is not None
    assert PenalizedLinearRegression is not None
    assert PenalizedLogisticRegression is not None
    assert PenalizedPoissonRegression is not None
    assert Ridge is not None
    assert Lasso is not None
    assert ElasticNet is not None
    assert PoissonRegression is not None
    assert OrderedLogitRegression is not None
    assert OrderedProbitRegression is not None
    assert FormulaParser is not None
    assert parse_formula is not None
    assert GLMLoss is not None
    assert GLMFamily is not None
    assert register_glm_loss is not None
    assert fista_solver is not None
    assert newton_solver is not None
    assert lbfgs_solver is not None
    assert "logistic" in list_glm_losses()
    assert get_glm_loss("poisson").name == "poisson"


def test_old_losses_namespace_is_not_a_compatibility_entrypoint():
    import importlib.util

    assert importlib.util.find_spec("statgpu.losses") is None


def test_penalized_glm_auto_solver_is_backend_aware():
    from statgpu.linear_model import (
        PenalizedLinearRegression,
        PenalizedLogisticRegression,
        PenalizedPoissonRegression,
    )

    logit = PenalizedLogisticRegression(penalty="l2", solver="auto")
    logit._penalty = logit._resolve_penalty()
    logit_loss = logit._resolve_loss()
    # Smooth L2 GLMs dispatch to IRLS on all backends
    assert logit._select_solver(logit_loss, backend_name="numpy") == "irls"
    assert logit._select_solver(logit_loss, backend_name="cupy") == "irls"
    assert logit._select_solver(logit_loss, backend_name="torch") == "irls"

    poisson = PenalizedPoissonRegression(penalty="l2", solver="auto")
    poisson._penalty = poisson._resolve_penalty()
    poisson_loss = poisson._resolve_loss()
    # Smooth L2 GLMs dispatch to IRLS on all backends
    assert poisson._select_solver(poisson_loss, backend_name="numpy") == "irls"
    assert poisson._select_solver(poisson_loss, backend_name="cupy") == "irls"
    assert poisson._select_solver(poisson_loss, backend_name="torch") == "irls"

    ridge = PenalizedLinearRegression(penalty="l2", solver="auto")
    ridge._penalty = ridge._resolve_penalty()
    ridge_loss = ridge._resolve_loss()
    assert ridge._select_solver(ridge_loss, backend_name="numpy") == "exact"
    assert ridge._select_solver(ridge_loss, backend_name="cupy") == "exact"


def test_explicit_solvers_do_not_change_backend_choice():
    from statgpu.linear_model import PenalizedLogisticRegression

    for solver in ("irls", "newton", "lbfgs", "fista"):
        model = PenalizedLogisticRegression(
            penalty="l2",
            solver=solver,
            device="cpu",
            max_iter=2,
        )
        model._penalty = model._resolve_penalty()
        loss = model._resolve_loss()
        assert model._select_solver(loss, backend_name="numpy") == solver
        assert model._select_solver(loss, backend_name="cupy") == solver
        assert model._select_solver(loss, backend_name="torch") == solver


def test_smooth_solvers_reject_non_smooth_penalties():
    import pytest
    from statgpu.linear_model import PenalizedLogisticRegression

    for solver in ("newton", "lbfgs"):
        model = PenalizedLogisticRegression(
            penalty="elasticnet",
            solver=solver,
        )
        model._penalty = model._resolve_penalty()
        with pytest.raises(ValueError):
            model._validate_solver_penalty()
