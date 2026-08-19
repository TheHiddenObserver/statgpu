import numpy as np
import pytest

from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean


def test_scaled_mean_preserves_small_term_after_huge_cancellation_numpy():
    values = np.asarray([1.0e308, 1.0, -1.0e308], dtype=np.float64)
    actual = float(_scaled_mean(values, np))
    np.testing.assert_allclose(actual, 1.0 / 3.0, rtol=0.0, atol=0.0)


def test_scaled_group_means_preserve_small_term_after_huge_cancellation_numpy():
    values = np.asarray(
        [1.0e308, 1.0, -1.0e308, 5.0, 5.0, 5.0], dtype=np.float64
    )
    groups = np.asarray([7, 7, 7, 3, 3, 3], dtype=np.int64)
    actual = np.asarray(_scaled_group_means(values, groups, np))
    expected = np.asarray([1.0 / 3.0] * 3 + [5.0] * 3, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_scaled_mean_preserves_smallest_subnormal_numpy():
    tiny = np.nextafter(0.0, 1.0)
    values = np.asarray([tiny, tiny, tiny], dtype=np.float64)
    actual = float(_scaled_mean(values, np))
    assert actual == tiny


def test_scaled_group_means_preserve_smallest_subnormal_numpy():
    tiny = np.nextafter(0.0, 1.0)
    values = np.asarray([tiny, tiny, tiny, 1.0e308, 1.0e308], dtype=np.float64)
    groups = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    actual = np.asarray(_scaled_group_means(values, groups, np))
    assert np.all(actual[:3] == tiny)
    np.testing.assert_allclose(actual[3:], np.asarray([1.0e308, 1.0e308]))


def test_scaled_mean_and_group_means_preserve_cancellation_torch_cpu():
    torch = pytest.importorskip("torch")
    values = torch.tensor(
        [1.0e308, 1.0, -1.0e308, 5.0, 5.0, 5.0], dtype=torch.float64
    )
    groups = torch.tensor([7, 7, 7, 3, 3, 3], dtype=torch.int64)

    mean = _scaled_mean(values[:3], torch)
    grouped = _scaled_group_means(values, groups, torch)

    assert mean.device.type == "cpu"
    assert grouped.device.type == "cpu"
    np.testing.assert_allclose(float(mean), 1.0 / 3.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        grouped.detach().cpu().numpy(),
        np.asarray([1.0 / 3.0] * 3 + [5.0] * 3, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_scaled_mean_and_group_means_preserve_subnormal_torch_cpu():
    torch = pytest.importorskip("torch")
    tiny = np.nextafter(0.0, 1.0)
    values = torch.tensor([tiny, tiny, tiny], dtype=torch.float64)
    groups = torch.tensor([0, 0, 0], dtype=torch.int64)
    assert float(_scaled_mean(values, torch)) == tiny
    grouped = _scaled_group_means(values, groups, torch)
    assert np.all(grouped.detach().cpu().numpy() == tiny)
