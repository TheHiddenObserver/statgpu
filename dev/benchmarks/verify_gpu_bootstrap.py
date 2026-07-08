import numpy as np, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)
from statgpu.linear_model import QuantileRegression

np.random.seed(42)
X = np.random.randn(200, 3)
y = 1.0 + X @ [0.5, -0.3, 0.8] + 0.5 * np.random.randn(200)

# CPU reference
mc = QuantileRegression(quantile=0.5, compute_inference=True,
                         inference_method='bootstrap', n_bootstrap=50, device='cpu')
mc.fit(X, y)
bse_cpu = mc._bse[0]
print(f"CPU  bse0={bse_cpu:.6f}")

for dev, label in [("cuda", "CuPy"), ("torch", "Torch")]:
    try:
        mg = QuantileRegression(quantile=0.5, compute_inference=True,
                                 inference_method='bootstrap', n_bootstrap=50, device=dev)
        mg.fit(X, y)
        bse_gpu = mg._bse[0]
        ratio = bse_gpu / bse_cpu
        ok = 0.5 < ratio < 2.0
        print(f"{label} bse0={bse_gpu:.6f} ratio={ratio:.3f} {'OK' if ok else 'FAIL'}")
    except Exception as e:
        print(f"{label} FAIL: {e}")
