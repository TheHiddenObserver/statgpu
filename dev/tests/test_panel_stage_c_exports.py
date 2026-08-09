"""Public export contract for Panel Tier-1 Stage C covariance."""

from statgpu import driscoll_kraay_covariance as top_level_dk
from statgpu.panel import driscoll_kraay_covariance as panel_dk
from statgpu.panel._covariance import driscoll_kraay_covariance as internal_dk


def test_driscoll_kraay_covariance_is_publicly_exported():
    assert top_level_dk is internal_dk
    assert panel_dk is internal_dk
