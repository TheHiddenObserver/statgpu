import numpy as np

from statgpu.linear_model.penalized import _nodewise_precision_cache as cache


def _design(seed=51, n=80, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X - X.mean(axis=0)


def test_auto_then_explicit_same_alpha_reuses_numerics_not_request_provenance():
    X = _design()
    cache._CACHE.clear()
    M1, alpha, meta1 = cache.build_nodewise_precision_numpy(
        X,
        requested_alpha=None,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert meta1["precision_cache_hit"] is False
    assert meta1["nodewise_alpha_source"] == "auto"

    M2, alpha2, meta2 = cache.build_nodewise_precision_numpy(
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
    _, _, first = cache.build_nodewise_precision_numpy(
        X,
        requested_alpha=0.07,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    _, _, second = cache.build_nodewise_precision_numpy(
        X,
        requested_alpha=0.09,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert first["precision_cache_hit"] is False
    assert second["precision_cache_hit"] is False


def test_p1_bypass_records_explicit_cache_miss_without_entry():
    X = _design(seed=53, p=1)
    cache._CACHE.clear()
    _, resolved, meta = cache.build_nodewise_precision_numpy(
        X,
        requested_alpha=0.08,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert resolved is None
    assert meta["precision_method"] == "analytic_univariate"
    assert meta["precision_cache_hit"] is False
    assert len(cache._CACHE) == 0


def test_disabled_cache_keeps_cache_provenance_explicit(monkeypatch):
    X = _design(seed=54)
    cache._CACHE.clear()
    monkeypatch.setattr(cache, "_CACHE_MAXSIZE", 0)
    _, resolved, meta = cache.build_nodewise_precision_numpy(
        X,
        requested_alpha=0.08,
        effective_n=float(X.shape[0]),
        weighted=False,
    )
    assert resolved == 0.08
    assert meta["precision_cache_hit"] is False
    assert len(cache._CACHE) == 0
