import numpy as np, traceback, sys, os
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
np.random.seed(42)
X = np.random.randn(200, 5)
y = X @ [1, -0.5, 0.3, 0, 0] + 0.5 * np.random.randn(200)
try:
    m = PenalizedLinearRegression(penalty='scad', alpha=0.1,
                                   compute_inference=True,
                                   inference_method='oracle',
                                   max_iter=50, device='cpu')
    m.fit(X, y)
    print(f'OK bse0={m._bse[0]:.6f}')
except Exception:
    traceback.print_exc()
