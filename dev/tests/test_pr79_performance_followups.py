import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from statgpu.covariance import GraphicalLasso


def _cv_cache_digests_for_seed(seed):
    script = (
        'import numpy as np\n'
        'from statgpu.cross_validation import CVCache, hash_cv_data\n'
        'X = np.array([[0.0, 1.0], [2.0, 3.0]])\n'
        'y = np.array([0.0, 1.0])\n'
        'key = {1: {3, 2}, 4: (2, 2)}\n'
        'print(hash_cv_data(X, y, cache_key=key).hex())\n'
        'print(CVCache.make_key(key))\n'
    )
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = str(seed)
    output = subprocess.check_output(
        [sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
    )
    return tuple(output.splitlines())


def test_cv_cache_keys_are_stable_across_python_hash_seeds():
    assert _cv_cache_digests_for_seed(1) == _cv_cache_digests_for_seed(98765)


def test_graphical_lasso_batches_inner_convergence_checks():
    rng = np.random.default_rng(714)
    X = rng.normal(size=(80, 6))
    model = GraphicalLasso(alpha=0.08, max_iter=25, tol=1e-7, device='cpu').fit(X)

    assert model._inner_iterations_ > 0
    assert model._inner_convergence_checks_ > 0
    assert model._inner_convergence_checks_ * 16 >= model._inner_iterations_
    assert model._inner_convergence_checks_ < model._inner_iterations_
    np.testing.assert_allclose(model.covariance_, model.covariance_.T, atol=1e-12)
    assert np.all(np.isfinite(model.precision_))
