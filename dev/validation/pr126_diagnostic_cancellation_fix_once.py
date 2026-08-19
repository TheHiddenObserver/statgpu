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


replace_function(
    "statgpu/panel/_diagnostics.py",
    "_scaled_mean",
    r'''def _scaled_mean(values, xp):
    """Return a cancellation-safe backend-native mean.

    Scaling alone prevents overflow but does not preserve a small finite term
    between nearly cancelling O(DBL_MAX) observations. Reuse the covariance
    layer's magnitude-tiered grouped reduction after dividing each observation
    by ``n``. The reduced quantity is the mean itself, so same-sign groups never
    require an unrepresentable intermediate group sum.
    """
    n = int(values.shape[0])
    if n <= 0:
        raise ValueError("mean requires at least one observation")
    from statgpu.panel._covariance import _grouped_score_sums

    codes_np = np.zeros(n, dtype=np.int64)
    scaled = values / float(n)
    grouped = _grouped_score_sums(
        scaled.reshape(-1, 1), codes_np, n_groups=1, xp=xp
    )
    return grouped[0, 0]
''',
)

replace_function(
    "statgpu/panel/_diagnostics.py",
    "_scaled_group_means",
    r'''def _scaled_group_means(values, groups, xp):
    """Return cancellation-safe group means aligned to observations.

    The numerical accumulation stays on the selected backend; only compact
    integer group codes are materialized on the host, matching the panel
    metadata policy. Each observation is divided by its group size before the
    magnitude-tiered grouped reduction, so the reduced quantity is already a
    mean and therefore remains representable for finite inputs.
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
    scaled = values / counts[codes]
    compact = _grouped_score_sums(
        scaled.reshape(-1, 1), codes_np, n_groups=n_groups, xp=xp
    )[:, 0]
    return compact[codes]
''',
)


test_path = Path("dev/tests/test_panel_diagnostic_cancellation_precision.py")
test_path.write_text(
    r'''import numpy as np
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
''',
    encoding="utf-8",
)
