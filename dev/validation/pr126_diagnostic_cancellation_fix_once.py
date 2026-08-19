from pathlib import Path


def replace_function(path: str, name: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    marker = f"def {name}("
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"function {name} not found in {path}")
    next_def = text.find("\ndef ", start + len(marker))
    if next_def < 0:
        next_def = len(text)
    else:
        next_def += 1
    updated = text[:start] + replacement.rstrip() + "\n\n" + text[next_def:]
    p.write_text(updated, encoding="utf-8")


# The compact fit-space mean is always representable for finite inputs, but
# DBL_MAX / m may round upward. At equality, an unscaled same-sign reduction can
# therefore overflow even though the mean itself is finite. Treat equality as
# dangerous so the existing count prescale is activated.
utils_path = Path("statgpu/panel/_utils.py")
utils_text = utils_path.read_text(encoding="utf-8")
old_guard = "dangerous_obs = (xp.abs(values) > limit) * 1.0"
new_guard = "dangerous_obs = (xp.abs(values) >= limit) * 1.0"
if old_guard not in utils_text and new_guard not in utils_text:
    raise RuntimeError("compact group-mean overflow guard anchor not found")
if old_guard in utils_text:
    utils_text = utils_text.replace(old_guard, new_guard, 1)
utils_path.write_text(utils_text, encoding="utf-8")


replace_function(
    "statgpu/panel/_diagnostics.py",
    "_scaled_mean",
    r'''def _scaled_mean(values, xp):
    """Return a cancellation- and range-safe backend-native mean.

    A uniform ``1/n`` prescale prevents same-sign overflow, but applying it to
    every input would erase finite subnormal values before reduction. Scale only
    when an observation is large enough that an unscaled same-sign sum could
    overflow, then reuse the covariance layer's magnitude-tiered grouped sum so
    low-order terms survive later cancellation. Equality is included because
    ``DBL_MAX / n`` can round upward enough that ``n`` equal terms overflow.
    """
    n = int(values.shape[0])
    if n <= 0:
        raise ValueError("mean requires at least one observation")
    from statgpu.panel._covariance import _grouped_score_sums

    max_abs = xp.max(xp.abs(values))
    limit = float(np.finfo(np.float64).max) / float(n)
    dangerous = max_abs >= float(limit)
    factor = xp.where(
        dangerous,
        xp.full_like(max_abs, float(n)),
        xp.ones_like(max_abs),
    )
    scaled = values / factor
    codes_np = np.zeros(n, dtype=np.int64)
    total = _grouped_score_sums(
        scaled.reshape(-1, 1), codes_np, n_groups=1, xp=xp
    )[0, 0]
    return xp.where(dangerous, total, total / float(n))
''',
)

replace_function(
    "statgpu/panel/_diagnostics.py",
    "_scaled_group_means",
    r'''def _scaled_group_means(values, groups, xp):
    """Return cancellation- and range-safe group means aligned to observations.

    Statistical accumulation remains on the selected backend.  Only compact
    integer group codes cross to the host, matching the panel metadata policy.
    Groups whose same-sign sum could overflow are divided by their own count
    before reduction; safe groups remain on their original scale so finite
    subnormal means are not erased.  The covariance layer's magnitude-tiered
    grouped sum then preserves low-order terms through large cancellation.
    """
    from statgpu.panel._covariance import _grouped_score_sums

    codes_raw = np.asarray(_to_numpy(groups), dtype=np.int64).ravel()
    if codes_raw.shape[0] != int(values.shape[0]):
        raise ValueError("groups must match the number of observations")
    _labels, codes_np = np.unique(codes_raw, return_inverse=True)
    n_groups = int(codes_np.max()) + 1 if codes_np.size else 0
    if n_groups <= 0:
        raise ValueError("group means require at least one observation")

    counts_np = np.bincount(codes_np, minlength=n_groups).astype(np.float64)
    codes = xp_asarray(
        codes_np, dtype=xp.int64, xp=xp, ref_arr=values
    )
    counts = xp_asarray(
        counts_np, dtype=xp.float64, xp=xp, ref_arr=values
    )
    sizes = counts[codes]
    limit = float(np.finfo(np.float64).max) / sizes
    dangerous_obs = (xp.abs(values) >= limit) * 1.0
    dangerous_aligned = group_means(dangerous_obs, groups, xp=xp) > 0.0
    factor = xp.where(dangerous_aligned, sizes, xp.ones_like(sizes))

    compact = _grouped_score_sums(
        (values / factor).reshape(-1, 1),
        codes_np,
        n_groups=n_groups,
        xp=xp,
    )[:, 0]
    aligned = compact[codes]
    return xp.where(dangerous_aligned, aligned, aligned / sizes)
''',
)


test_path = Path("dev/tests/test_panel_diagnostic_cancellation_precision.py")
test_path.write_text(
    r'''import numpy as np
import pytest

from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean
from statgpu.panel._utils import group_means, within_transform


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


def test_shared_group_mean_scales_at_exact_overflow_boundary_numpy():
    value = float(np.finfo(np.float64).max / 3.0)
    values = np.asarray([value, value, value], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, np.full(3, value))
    demeaned = np.asarray(within_transform(values, groups, xp=np))
    np.testing.assert_array_equal(demeaned, np.zeros(3, dtype=np.float64))


def test_scaled_mean_preserves_exact_overflow_boundary_numpy():
    value = float(np.finfo(np.float64).max / 3.0)
    values = np.asarray([value, value, value], dtype=np.float64)
    actual = float(_scaled_mean(values, np))
    assert np.isfinite(actual)
    assert actual == value


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


def test_shared_group_mean_overflow_boundary_torch_cpu():
    torch = pytest.importorskip("torch")
    value = float(np.finfo(np.float64).max / 3.0)
    values = torch.full((3,), value, dtype=torch.float64)
    groups = torch.zeros(3, dtype=torch.int64)
    actual = group_means(values, groups, xp=torch)
    assert bool(torch.all(torch.isfinite(actual)))
    np.testing.assert_array_equal(
        actual.detach().cpu().numpy(), np.full(3, value, dtype=np.float64)
    )


def test_scaled_mean_and_group_means_preserve_subnormal_torch_cpu():
    torch = pytest.importorskip("torch")
    tiny = np.nextafter(0.0, 1.0)
    values = torch.tensor([tiny, tiny, tiny], dtype=torch.float64)
    groups = torch.tensor([0, 0, 0], dtype=torch.int64)
    assert float(_scaled_mean(values, torch)) == tiny
    grouped = _scaled_group_means(values, groups, torch)
    assert np.all(grouped.detach().cpu().numpy() == tiny)
''',
    encoding="utf-8",
)
