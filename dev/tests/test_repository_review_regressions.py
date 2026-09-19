
import sys
import types

import numpy as np
import pytest

from statgpu.penalties import AdaptiveL1Penalty
import statgpu.penalties._adaptive_l1 as _adaptive_l1_impl
from statgpu.unsupervised import UMAP
from statgpu.unsupervised._nndescent import nndescent_numpy
import statgpu.unsupervised._umap as umap_module


def test_adaptive_l1_gradient_numpy():
    penalty = AdaptiveL1Penalty(
        alpha=2.0, weights=np.array([1.0, 3.0]), normalize=False
    )
    np.testing.assert_array_equal(
        penalty.gradient(np.array([-4.0, 5.0])), np.array([-2.0, 6.0])
    )


def test_adaptive_l1_cached_alpha_weights_refresh_on_alpha_change_and_cupy_device(
    monkeypatch,
):
    penalty = AdaptiveL1Penalty(
        alpha=0.1,
        weights=np.array([1.0, 2.0]),
        normalize=False,
    )
    calls = []

    class FakeDevice:
        def __init__(self, device_id):
            self.id = int(device_id)

    class FakeArray:
        __module__ = "cupy._core.core"

        def __init__(self, device_id, scale=1.0):
            self.device = FakeDevice(device_id)
            self.dtype = np.dtype("float64")
            self.scale = float(scale)

        def __rmul__(self, value):
            return FakeArray(self.device.id, self.scale * float(value))

    def fake_align(value, dtype, ref):
        calls.append(int(ref.device.id))
        return FakeArray(ref.device.id)

    monkeypatch.setattr(_adaptive_l1_impl, "_xp_asarray", fake_align)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(float64=np.float64),
    )

    ref0 = FakeArray(0)
    first = penalty._cached_alpha_weights(ref0, "cupy")
    assert first.device.id == 0
    assert first.scale == pytest.approx(0.1)
    assert calls == [0]

    # Same source weights and device, but a new alpha must invalidate the
    # threshold cache rather than silently reusing the old regularization.
    penalty.alpha = 0.25
    second = penalty._cached_alpha_weights(ref0, "cupy")
    assert second.device.id == 0
    assert second.scale == pytest.approx(0.25)
    assert calls == [0, 0]

    # Reusing the same public penalty on another CUDA ordinal must migrate the
    # cached threshold tensor to the operand device.
    ref1 = FakeArray(1)
    third = penalty._cached_alpha_weights(ref1, "cupy")
    assert third.device.id == 1
    assert third.scale == pytest.approx(0.25)
    assert calls == [0, 0, 1]


def test_adaptive_l1_lla_weights_follow_operand_device(monkeypatch):
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=np.array([1.0, 3.0]),
        normalize=False,
    )
    targets = []

    class FakeDevice:
        id = 4

    class FakeCoef:
        __module__ = "cupy._core.core"
        device = FakeDevice()
        dtype = np.dtype("float64")

    fake_xp = types.SimpleNamespace(__name__="cupy")

    def fake_align(value, dtype, ref):
        targets.append(int(ref.device.id))
        return ("aligned", int(ref.device.id), dtype)

    monkeypatch.setattr(_adaptive_l1_impl, "_xp", lambda value: fake_xp)
    monkeypatch.setattr(_adaptive_l1_impl, "_xp_asarray", fake_align)

    result = penalty.lla_weights(FakeCoef())
    assert result[0] == "aligned"
    assert result[1] == 4
    assert targets == [4]


def test_adaptive_l1_proximal_preserves_operand_float_dtype():
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=np.array([1.0, 2.0]),
        normalize=False,
    )

    w32 = np.array([0.8, -0.5], dtype=np.float32)
    prox32 = penalty.proximal(w32, 0.1, backend="numpy")
    assert prox32.dtype == np.dtype("float32")
    assert penalty.lla_weights(w32).dtype == np.dtype("float32")

    w_int = np.array([1, -2], dtype=np.int64)
    prox_int = penalty.proximal(w_int, 0.1, backend="numpy")
    assert prox_int.dtype == np.dtype("float64")
    assert penalty.lla_weights(w_int).dtype == np.dtype("float64")


def test_adaptive_l1_torch_cache_refreshes_on_dtype_change():
    torch = pytest.importorskip("torch")
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=np.array([1.0, 2.0]),
        normalize=False,
    )

    w32 = torch.tensor([0.8, -0.5], dtype=torch.float32)
    prox32 = penalty.proximal(w32, 0.1, backend="torch")
    assert prox32.dtype == torch.float32
    assert penalty._alpha_w_torch.dtype == torch.float32

    w64 = w32.to(torch.float64)
    prox64 = penalty.proximal(w64, 0.1, backend="torch")
    assert prox64.dtype == torch.float64
    assert penalty._alpha_w_torch.dtype == torch.float64


def test_adaptive_l1_weight_dimension_check_does_not_host_materialize():
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=[1.0, 2.0],
        normalize=False,
    )

    class FakeDeviceWeights:
        size = 2

        def __array__(self, *args, **kwargs):
            raise AssertionError("device weights must not use NumPy array protocol")

    fake = FakeDeviceWeights()
    penalty._weights = fake
    assert penalty._require_weights(np.zeros(2, dtype=np.float64)) is fake


@pytest.mark.parametrize("method", ["value", "gradient", "proximal", "lla_weights"])
def test_adaptive_l1_weight_dimension_must_match_coefficients(method):
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=[1.0],
        normalize=False,
    )
    coef = np.array([0.8, -0.5, 0.3], dtype=np.float64)

    with pytest.raises(ValueError, match="same length"):
        if method == "proximal":
            penalty.proximal(coef, 0.1, backend="numpy")
        else:
            getattr(penalty, method)(coef)


@pytest.mark.parametrize("method", ["value", "gradient", "proximal", "lla_weights"])
def test_adaptive_l1_uninitialized_weights_fail_closed(method):
    penalty = AdaptiveL1Penalty(alpha=0.2, weights=None, normalize=False)
    coef = np.array([0.8, -0.5], dtype=np.float64)

    with pytest.raises(RuntimeError, match="weights are not initialized"):
        if method == "proximal":
            penalty.proximal(coef, 0.1, backend="numpy")
        else:
            getattr(penalty, method)(coef)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"alpha": -0.1}, ValueError, "alpha.*non-negative"),
        ({"alpha": np.nan}, ValueError, "alpha.*finite"),
        ({"alpha": True}, TypeError, "alpha.*numeric"),
        ({"nu": 0.0}, ValueError, "nu.*positive"),
        ({"nu": np.inf}, ValueError, "nu.*finite"),
        ({"eps": 0.0}, ValueError, "eps.*positive"),
        ({"eps": -1e-8}, ValueError, "eps.*positive"),
        ({"init_method": "bad"}, ValueError, "init_method"),
        ({"init_method": 1}, TypeError, "init_method"),
        ({"normalize": "False"}, TypeError, "normalize.*boolean"),
    ],
)
def test_adaptive_l1_public_controls_fail_closed(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        AdaptiveL1Penalty(**kwargs)


@pytest.mark.parametrize(
    ("weights", "error_type", "message"),
    [
        ([], ValueError, "non-empty one-dimensional"),
        ([[1.0, 2.0]], ValueError, "one-dimensional"),
        ([1.0, np.nan], ValueError, "finite"),
        ([1.0, np.inf], ValueError, "finite"),
        ([1.0, -0.1], ValueError, "non-negative"),
        ([True, False], TypeError, "real numeric"),
        (["1.0", "2.0"], TypeError, "real numeric"),
    ],
)
def test_adaptive_l1_external_weights_fail_closed(weights, error_type, message):
    with pytest.raises(error_type, match=message):
        AdaptiveL1Penalty(
            alpha=0.2,
            weights=weights,
            normalize=False,
        )


def test_adaptive_l1_shallow_params_are_complete_and_clone_safe():
    penalty = AdaptiveL1Penalty(
        alpha=0.3,
        nu=2.0,
        eps=1e-5,
        init_method="ridge",
        normalize=True,
        weights=[1.0, 3.0],
    )
    params = penalty.get_params(deep=False)

    assert set(params) == {
        "alpha",
        "nu",
        "eps",
        "init_method",
        "normalize",
        "weights",
    }
    assert params["weights"] == (1.0, 3.0)
    np.testing.assert_allclose(
        penalty._weights,
        np.array([0.5, 1.5], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )

    reconstructed = AdaptiveL1Penalty(**params)
    reconstructed_params = reconstructed.get_params(deep=False)
    assert reconstructed_params["weights"] is params["weights"]
    assert reconstructed_params["eps"] == pytest.approx(1e-5)
    assert reconstructed_params["init_method"] == "ridge"
    assert reconstructed_params["normalize"] is True
    np.testing.assert_allclose(
        reconstructed._weights,
        penalty._weights,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("coef", "error_type", "message"),
    [
        (np.zeros((2, 1), dtype=np.float64), ValueError, "one-dimensional"),
        (np.array([True, False]), TypeError, "real numeric"),
        (np.array([1.0 + 1.0j, 0.5 + 0.0j]), TypeError, "real numeric"),
        (np.array([1.0, np.nan]), ValueError, "finite"),
        (np.array([1.0, np.inf]), ValueError, "finite"),
        (np.array(["1.0", "2.0"]), TypeError, "real numeric"),
    ],
)
def test_adaptive_l1_set_weights_rejects_invalid_initializer_output(
    coef, error_type, message
):
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=None,
        normalize=False,
    )
    with pytest.raises(error_type, match=message):
        penalty.set_weights(coef)
    assert penalty._weights is None


def test_adaptive_l1_learned_weights_remain_fit_local_constructor_state():
    penalty = AdaptiveL1Penalty(
        alpha=0.2,
        weights=None,
        normalize=False,
    )
    assert penalty.get_params(deep=False)["weights"] is None

    penalty.set_weights(np.array([2.0, 1.0], dtype=np.float64))
    assert penalty.get_params(deep=False)["weights"] is None
    assert penalty.weights is None
    assert penalty._weights is not None


def test_nndescent_numpy_unique_and_validated():
    X = np.random.default_rng(123).normal(size=(24, 4))
    indices, distances = nndescent_numpy(X, k=5, max_iter=3, seed=7)
    assert indices.shape == distances.shape == (24, 5)
    assert np.all(np.isfinite(distances))
    for i, row in enumerate(indices):
        assert i not in row
        assert len(np.unique(row)) == 5
    with pytest.raises(ValueError, match="k must"):
        nndescent_numpy(X, k=24)


def test_umap_none_seed_draws_once_per_fit(monkeypatch):
    seeds = iter([101, 202])
    monkeypatch.setattr(umap_module, "draw_random_seed", lambda state: next(seeds))
    X = np.arange(60.0).reshape(20, 3)
    params = dict(
        n_neighbors=4,
        n_components=2,
        n_epochs=1,
        init="random",
        random_state=None,
        device="cpu",
    )
    first, second = UMAP(**params).fit(X), UMAP(**params).fit(X)
    assert (first._fit_random_seed_, second._fit_random_seed_) == (101, 202)
    assert not np.allclose(first.embedding_, second.embedding_)
