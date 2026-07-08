"""Fix Torch Bootstrap on Tesla P100 by suppressing dynamo errors."""
import os, sys
os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT)

import torch
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
os.environ.setdefault("TORCHDYNAMO_SUPPRESS_ERRORS", "1")

import numpy as np
from statgpu.linear_model import Lasso
np.random.seed(42)
X = np.random.randn(100, 3)
y = X @ [1, -0.5, 0.3] + 0.1 * np.random.randn(100)

m = Lasso(alpha=0.1, compute_inference=True, inference_method='bootstrap',
          n_bootstrap=5, max_iter=50, device='torch')
m.fit(X, y)
print(f'OK bse0={m._bse[0]:.6f}')
