import numpy as np

from statgpu.linear_model import _nodewise_alpha_inference_contract as runtime
from statgpu.linear_model import _nodewise_precision_cache_contract as cache


def _design(seed=51, n=80, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X - X.mean(axis=0)


def test_auto_then_explicit_same_alpha_reuses_numerics_not_request_provenance():
    X = _design()
    cache._CACHE.clear()
    M1, alpha, meta1 = runtime.build_nodewise_precision_numpy(
        X,
        requested_alpha=None,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert meta1["precision_cache_hit"] is False
    assert meta1["nodewise_alpha_source"] == "auto"

    M2, alpha2, meta2 = runtime.build_nodewise_precision_numpy(
        X,
        requested_alpha=alpha,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert alpha2 == alpha
    assert meta2["precision_cache_hit"] is True
    assert meta2["nodewise_alpha_source"] == "user"
    np.testing.assert_allclose(M2, M1, rtol=0.0, atol=0.0)


def test_changing_nodewise_alpha_forces_cache_miss():
    X = _design(seed=52)
    cache._CACHE.clear()
    _, _, first = runtime.build_nodewise_precision_numpy(
        X,
        requested_alpha=0.07,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    _, _, second = runtime.build_nodewise_precision_numpy(
        X,
        requested_alpha=0.09,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert first["precision_cache_hit"] is False
    assert second["precision_cache_hit"] is False
