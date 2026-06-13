"""Test debiased Lasso inference on all three backends."""
import numpy as np
import sys


def test_cpu():
    from statgpu.linear_model import Lasso
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)
    model = Lasso(alpha=0.05, compute_inference=True, device='cpu')
    model.fit(X, y)
    assert model._bse is not None, "CPU: _bse is None"
    assert model._pvalues is not None, "CPU: _pvalues is None"
    assert model._conf_int is not None, "CPU: _conf_int is None"
    assert len(model._bse) == 6, f"CPU: bse length {len(model._bse)}"
    print(f"CPU:     bse={[f'{x:.4f}' for x in model._bse]}")
    print(f"CPU:     pval={[f'{x:.4f}' for x in model._pvalues]}")
    return model._bse, model._pvalues


def test_cupy():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= 0:
            print("CuPy: SKIP (no CUDA device)")
            return None, None
    except Exception:
        print("CuPy: SKIP (cupy not available)")
        return None, None

    from statgpu.linear_model import Lasso
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)

    model = Lasso(alpha=0.05, compute_inference=True, device='cuda')
    model.fit(X, y)
    assert model._bse is not None, "CuPy: _bse is None"
    assert model._pvalues is not None, "CuPy: _pvalues is None"
    assert model._conf_int is not None, "CuPy: _conf_int is None"
    print(f"CuPy:    bse={[f'{x:.4f}' for x in model._bse]}")
    print(f"CuPy:    pval={[f'{x:.4f}' for x in model._pvalues]}")
    return model._bse, model._pvalues


def test_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            print("Torch: SKIP (no CUDA device)")
            return None, None
    except Exception:
        print("Torch: SKIP (torch not available)")
        return None, None

    from statgpu.linear_model import Lasso
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.0]) + 0.1 * np.random.randn(100)

    model = Lasso(alpha=0.05, compute_inference=True, device='torch')
    model.fit(X, y)
    assert model._bse is not None, "Torch: _bse is None"
    assert model._pvalues is not None, "Torch: _pvalues is None"
    assert model._conf_int is not None, "Torch: _conf_int is None"
    print(f"Torch:   bse={[f'{x:.4f}' for x in model._bse]}")
    print(f"Torch:   pval={[f'{x:.4f}' for x in model._pvalues]}")
    return model._bse, model._pvalues


if __name__ == "__main__":
    print("=" * 50)
    print("Debiased Lasso: 3-backend test")
    print("=" * 50)

    cpu_bse, cpu_pval = test_cpu()
    cupy_bse, cupy_pval = test_cupy()
    torch_bse, torch_pval = test_torch()

    # Compare GPU vs CPU if available
    print("\n--- Comparison vs CPU ---")
    if cupy_bse is not None:
        diff = max(abs(a - b) for a, b in zip(cupy_bse, cpu_bse))
        print(f"CuPy vs CPU bse max diff: {diff:.2e}")
        assert diff < 0.01, f"CuPy vs CPU bse diff too large: {diff}"
    if torch_bse is not None:
        diff = max(abs(a - b) for a, b in zip(torch_bse, cpu_bse))
        print(f"Torch vs CPU bse max diff: {diff:.2e}")
        assert diff < 0.01, f"Torch vs CPU bse diff too large: {diff}"

    print("\nAll passed!")
