"""Preserve the utility contract while removing its import-time installer."""

import importlib
import pickle
from types import SimpleNamespace

import pytest

from statgpu.solvers import _fista, _fista_bb, _utils


@pytest.mark.parametrize(
    "name, expected",
    [
        ("adaptive_group_lasso", 0.0),
        (" Adaptive_Group_Lasso ", 0.0),
        ("none", 0.0),
        (None, 0.0),
        ("l1", 0.0),
        ("adaptive_l1", 0.0),
        ("group_lasso", 0.0),
        ("group_mcp", 0.0),
        ("group_scad", 0.0),
        ("en", 0.0),
        ("l2", 0.6),
        ("elasticnet", 0.6),
        ("unknown", 0.6),
        (" group_lasso ", 0.6),
    ],
)
def test_smooth_lipschitz_name_and_curvature_characterization(name, expected):
    penalty = SimpleNamespace(name=name, alpha=0.8, l1_ratio=0.25)
    for helper in (
        _utils._smooth_penalty_lipschitz,
        _fista._smooth_penalty_lipschitz,
        _fista_bb._smooth_penalty_lipschitz,
    ):
        assert helper(penalty) == pytest.approx(expected)


def test_none_and_missing_penalty_attributes_preserve_zero_curvature():
    assert _utils._smooth_penalty_lipschitz(None) == 0.0
    assert _utils._smooth_penalty_lipschitz(object()) == 0.0


def test_adaptive_group_short_circuit_never_reads_smooth_attributes():
    class Penalty:
        name = " adaptive_group_lasso "

        @property
        def alpha(self):
            raise AssertionError("a nonsmooth penalty has no smooth curvature")

    assert _utils._smooth_penalty_lipschitz(Penalty()) == 0.0


def test_nonadaptive_penalty_preserves_historical_name_lookup_order():
    class Penalty:
        reads = 0

        @property
        def name(self):
            self.reads += 1
            return "unknown" if self.reads == 1 else "l1"

    penalty = Penalty()
    assert _utils._smooth_penalty_lipschitz(penalty) == 0.0
    assert penalty.reads == 2


def test_invalid_smooth_attribute_error_is_not_swallowed():
    penalty = SimpleNamespace(name="l2", alpha="invalid", l1_ratio=0.25)
    with pytest.raises(ValueError, match="could not convert string to float"):
        _utils._smooth_penalty_lipschitz(penalty)


def test_historical_callable_pickle_path_remains_resolvable():
    # Protocol-0 global emitted for the former module-level wrapper. Unlike a
    # round trip of the new function, this checks an already serialized path.
    old_global = (
        b"cstatgpu.solvers._adaptive_group_lipschitz_contract\n"
        b"_smooth_penalty_lipschitz_with_adaptive_group\n."
    )
    restored = pickle.loads(old_global)
    assert restored(SimpleNamespace(name="adaptive_group_lasso")) == 0.0


def test_compatibility_module_reload_never_rebinds_consumers():
    original = _utils._smooth_penalty_lipschitz
    compat = importlib.import_module(
        "statgpu.solvers._adaptive_group_lipschitz_contract"
    )
    for _ in range(2):
        importlib.reload(compat)
        assert _utils._smooth_penalty_lipschitz is original
        assert _fista._smooth_penalty_lipschitz is original
        assert _fista_bb._smooth_penalty_lipschitz is original


def test_solver_consumers_bind_the_owning_helper_without_wrapper_links():
    helper = _utils._smooth_penalty_lipschitz
    assert helper.__module__ == "statgpu.solvers._utils"
    assert not hasattr(helper, "_statgpu_original")
    assert _fista._smooth_penalty_lipschitz is helper
    assert _fista_bb._smooth_penalty_lipschitz is helper
