"""Stage-C covariance alias behavior under sklearn-style set_params."""

from statgpu.panel import PanelOLS, PooledOLS, RandomEffects


def _assert_alias_refresh(model, raw, canonical):
    returned = model.set_params(cov_type=raw)
    assert returned is model
    assert model.cov_type == raw
    assert model._cov_type == canonical
    assert model.get_params()["cov_type"] == raw


def test_hc1_set_params_preserves_raw_value_and_refreshes_runtime_alias():
    for model in (PooledOLS(), PanelOLS(entity_effects=True), RandomEffects()):
        _assert_alias_refresh(model, "hc1", "robust")


def test_driscoll_kraay_aliases_refresh_runtime_dispatch_after_set_params():
    for raw in ("dk", "kernel", "driscoll-kraay"):
        for model in (PooledOLS(), PanelOLS(entity_effects=True), RandomEffects()):
            _assert_alias_refresh(model, raw, "driscoll-kraay")
