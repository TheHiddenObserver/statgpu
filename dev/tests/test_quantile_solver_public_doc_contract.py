"""Runtime-help contract for guarded public Quantile solver boundaries."""

import inspect

from statgpu import glm_core, solvers
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def test_guarded_public_solver_docstrings_expose_quantile_boundary():
    fista_doc = inspect.getdoc(solvers.fista_bb_solver) or ""
    admm_doc = inspect.getdoc(solvers.admm_solver) or ""

    assert "Quantile" in fista_doc
    assert "not a maintained FISTA-BB route" in fista_doc
    assert "bb_burn_in" in fista_doc

    assert "Nesterov-accelerated gradient descent" in admm_doc
    assert "not a maintained ADMM route" in admm_doc
    assert "cg_max_iter" in admm_doc

    # glm_core re-exports the same guarded public callables, so runtime help
    # must remain identical across both maintained public import paths.
    assert inspect.getdoc(glm_core.fista_bb_solver) == fista_doc
    assert inspect.getdoc(glm_core.admm_solver) == admm_doc


def test_penalized_glm_runtime_help_lists_public_admm_solver():
    doc = inspect.getdoc(PenalizedGeneralizedLinearModel) or ""
    assert "'admm'" in doc
    assert "Support is loss/penalty dependent" in doc
    assert "unsupported explicit combinations fail closed" in doc
