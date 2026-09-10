import numpy as np
import pytest

from statgpu.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
)


class DerivedLassoCV(LassoCV):
    pass


class DerivedElasticNetCV(ElasticNetCV):
    pass


def _factories(value):
    return [
        lambda: PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l1", nodewise_alpha=value,
            compute_inference=False, device="cpu"
        ),
        lambda: PenalizedLinearRegression(
            penalty="l1", nodewise_alpha=value,
            compute_inference=False, device="cpu"
        ),
        lambda: Lasso(nodewise_alpha=value, compute_inference=False, device="cpu"),
        lambda: ElasticNet(nodewise_alpha=value, compute_inference=False, device="cpu"),
        lambda: LassoCV(nodewise_alpha=value, compute_inference=False, device="cpu"),
        lambda: ElasticNetCV(nodewise_alpha=value, compute_inference=False, device="cpu"),
    ]


def test_all_public_surfaces_preserve_requested_nodewise_alpha_in_get_params_and_clone():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    requested = np.float64(0.081)
    for factory in _factories(requested):
        model = factory()
        params = model.get_params(deep=False)
        assert "nodewise_alpha" in params
        assert params["nodewise_alpha"] is requested
        cloned = clone(model)
        cloned_param = cloned.get_params(deep=False)["nodewise_alpha"]
        assert isinstance(cloned_param, np.float64)
        assert float(cloned_param) == pytest.approx(float(requested))
        assert cloned.nodewise_alpha is cloned_param
        assert cloned.nodewise_alpha_ is None


def test_all_public_surfaces_transactionally_accept_set_params_and_clear_resolved_state():
    for factory in _factories(0.071):
        model = factory()
        model.set_params(nodewise_alpha=0.093)
        assert model.nodewise_alpha == pytest.approx(0.093)
        assert model.get_params(deep=False)["nodewise_alpha"] == pytest.approx(0.093)
        assert model.nodewise_alpha_ is None


def test_all_public_surfaces_reject_invalid_set_params_transactionally():
    for factory in _factories(0.071):
        model = factory()
        before = model.get_params(deep=False)["nodewise_alpha"]
        with pytest.raises(ValueError, match="nodewise_alpha"):
            model.set_params(nodewise_alpha=0.0)
        assert model.get_params(deep=False)["nodewise_alpha"] is before
        assert model.nodewise_alpha_ is None


def test_cv_subclasses_preserve_read_only_resolved_nodewise_state_contract():
    for cls in (DerivedLassoCV, DerivedElasticNetCV):
        requested = np.float64(0.082)
        model = cls(nodewise_alpha=requested, compute_inference=False, device="cpu")
        assert model.get_params(deep=False)["nodewise_alpha"] is requested
        assert model.nodewise_alpha_ is None


@pytest.mark.parametrize("bad", [False, True, 0.0, -0.1, np.nan, np.inf, 1 + 1j, [0.1]])
def test_all_public_surfaces_reject_the_same_invalid_nodewise_alpha_contract(bad):
    for factory in _factories(bad):
        with pytest.raises(ValueError, match="nodewise_alpha"):
            factory()
