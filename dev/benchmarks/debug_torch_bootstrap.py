import numpy as np, traceback, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)
from statgpu.linear_model import Lasso
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ [1, -0.5, 0.3] + 0.1 * np.random.randn(100)
try:
    m = Lasso(alpha=0.1, compute_inference=True, inference_method='bootstrap',
              n_bootstrap=5, max_iter=50, device='torch')
    m.fit(X, y)
    print(f'OK: bse0={m._bse[0]:.6f}')
except Exception:
    traceback.print_exc()
