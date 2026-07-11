"""Temporary patch script for repository review batch 2."""
from pathlib import Path
from textwrap import dedent
import re


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: match count={text.count(old)} for {old[:70]!r}")
    p.write_text(text.replace(old, new))


# Cross-validation utility contracts.
replace_once(
    "statgpu/cross_validation/_base.py",
    '''    def __init__(self, maxsize: int = 64):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = __import__('threading').Lock()
''',
    '''    def __init__(self, maxsize: int = 64):
        if not isinstance(maxsize, (int, np.integer)) or int(maxsize) < 0:
            raise ValueError("maxsize must be a non-negative integer")
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = int(maxsize)
        self._lock = __import__('threading').Lock()
''',
)

p = Path("statgpu/cross_validation/_base.py")
text = p.read_text()
pattern = re.compile(
    r"def detect_gpu_input\(X, y\) -> Tuple\[str, Any, Any\]:.*?\n\n# ---------------------------------------------------------------------------\n# Batch MSE computation",
    re.S,
)
replacement = dedent('''
    def detect_gpu_input(X, y) -> Tuple[str, Any, Any]:
        """Detect a common input backend, converting mixed inputs safely.

        Matching CuPy or Torch inputs are preserved. Any mixture of NumPy and
        GPU arrays, or CuPy and Torch arrays, is converted to NumPy so callers
        never receive ``backend='numpy'`` alongside an unconverted GPU object.
        """
        import warnings as _warnings

        def array_type(value):
            try:
                import cupy as cp
                if isinstance(value, cp.ndarray):
                    return "cupy"
            except ImportError:
                pass
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    return "torch"
            except ImportError:
                pass
            return "numpy"

        x_type = array_type(X)
        y_type = array_type(y)
        if x_type == y_type:
            return x_type, X, y

        _warnings.warn(
            f"Mixed backend detected: X is {x_type} but y is {y_type}. "
            "Converting both arrays to NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "numpy", _to_numpy(X), _to_numpy(y)


    # ---------------------------------------------------------------------------
    # Batch MSE computation''')
text, count = pattern.subn(replacement, text)
if count != 1:
    raise RuntimeError(f"detect_gpu_input block count={count}")
p.write_text(text)

replace_once(
    "statgpu/cross_validation/_base.py",
    '''    if not np.all(np.isfinite(sw_np)):
        raise ValueError("sample_weight must be finite")
    # Return the original array (preserves CuPy/Torch backend)
    return sample_weight
''',
    '''    if not np.all(np.isfinite(sw_np)):
        raise ValueError("sample_weight must be finite")
    if float(np.sum(sw_np)) <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    # Return the original array (preserves CuPy/Torch backend)
    return sample_weight
''',
)

replace_once(
    "statgpu/cross_validation/_base.py",
    '''    n_models = coefs.shape[0]

    if intercepts is not None:
        intercepts = _to_numpy(intercepts)

    if sample_weight is not None:
        sw = _to_numpy(sample_weight).ravel()
        sw_sum = float(np.sum(sw))
    else:
        sw = None
        sw_sum = 0.0
''',
    '''    n_models = coefs.shape[0]
    if not isinstance(chunk_size, (int, np.integer)) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)

    if intercepts is not None:
        intercepts = _to_numpy(intercepts).ravel()
        if intercepts.shape[0] != n_models:
            raise ValueError(
                f"intercepts length {intercepts.shape[0]} != n_models {n_models}"
            )
        if not np.all(np.isfinite(intercepts)):
            raise ValueError("intercepts must be finite")

    if sample_weight is not None:
        sw = _to_numpy(sample_weight).ravel().astype(np.float64, copy=False)
        if sw.shape[0] != X_val.shape[0]:
            raise ValueError(
                f"sample_weight length {sw.shape[0]} != n_samples {X_val.shape[0]}"
            )
        if not np.all(np.isfinite(sw)):
            raise ValueError("sample_weight must be finite")
        if np.any(sw < 0):
            raise ValueError("sample_weight must be non-negative")
        sw_sum = float(np.sum(sw))
        if sw_sum <= 0.0:
            raise ValueError("sample_weight must have a positive sum")
    else:
        sw = None
        sw_sum = 0.0
''',
)
replace_once(
    "statgpu/cross_validation/_base.py",
    '''        if sw is not None:
            if sw_sum > 0:
                mse[start:end] = np.sum(residuals ** 2 * sw[None, :], axis=1) / sw_sum
            else:
                mse[start:end] = np.nan
        else:
''',
    '''        if sw is not None:
            mse[start:end] = np.sum(residuals ** 2 * sw[None, :], axis=1) / sw_sum
        else:
''',
)

# KMeans score should enforce the same input contract as predict/transform.
replace_once(
    "statgpu/unsupervised/_kmeans.py",
    '''    def score(self, X, y=None):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        distances = self._squared_distances(backend, X_arr, self.cluster_centers_)
''',
    '''    def score(self, X, y=None):
        self._check_is_fitted()
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in KMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        distances = self._squared_distances(backend, X_arr, self.cluster_centers_)
''',
)

# UMAP spectral initialization must return the requested dimension for small n.
p = Path("statgpu/unsupervised/_umap.py")
text = p.read_text()
old = '''        n_components = min(int(self.n_components) + 1, n_samples - 2)
        _, eigenvectors = eigsh(laplacian, k=n_components, which='SM', tol=1e-4)
        jitter = backend_random_normal(backend, self._fit_random_seed_, size=(n_samples, int(self.n_components)), scale=1e-4)
        return backend.asarray(eigenvectors[:, 1:int(self.n_components)+1], dtype=backend.float64) + jitter
'''
new = '''        requested = int(self.n_components)
        if requested + 1 >= n_samples:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
            order = np.argsort(np.abs(eigenvalues))
            embedding_np = eigenvectors[:, order[1 : requested + 1]]
        else:
            _, eigenvectors = eigsh(
                laplacian, k=requested + 1, which="SM", tol=1e-4
            )
            embedding_np = eigenvectors[:, 1 : requested + 1]
        if embedding_np.shape[1] != requested:
            raise RuntimeError(
                f"spectral initialization returned {embedding_np.shape[1]} components, "
                f"expected {requested}"
            )
        jitter = backend_random_normal(
            backend,
            self._fit_random_seed_,
            size=(n_samples, requested),
            scale=1e-4,
        )
        return backend.asarray(embedding_np, dtype=backend.float64) + jitter
'''
if text.count(old) != 1:
    raise RuntimeError(f"UMAP spectral block count={text.count(old)}")
p.write_text(text.replace(old, new))

# Optional GPU dependencies must not break CPU-only test collection, and GPU
# failures must not be swallowed as a passing test.
p = Path("dev/tests/test_elasticnet_cv.py")
text = p.read_text()
text = text.replace(
    "import numpy as np\nimport torch\nfrom statgpu.linear_model import ElasticNetCV, ElasticNet\nfrom statgpu import get_backend, Device\n\nimport warnings\nwarnings.filterwarnings('ignore')\n",
    "import numpy as np\nimport pytest\nfrom statgpu.linear_model import ElasticNetCV\nfrom statgpu import get_backend\n",
    1,
)
pattern = re.compile(
    r"def test_elasticnetcv_gpu_backend\(\):.*?\n\n\ndef test_elasticnetcv_predict",
    re.S,
)
replacement = dedent('''
    def test_elasticnetcv_gpu_backend():
        """Compare CPU and explicit CuPy results when CUDA is available."""
        X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)
        cpu_model = ElasticNetCV(
            l1_ratio=0.5, n_alphas=20, cv=3, random_state=42, device="cpu"
        ).fit(X, y)
        if not get_backend("cupy").is_available():
            pytest.skip("working CuPy CUDA backend is unavailable")
        cuda_model = ElasticNetCV(
            l1_ratio=0.5, n_alphas=20, cv=3, random_state=42, device="cuda"
        ).fit(X, y)
        np.testing.assert_allclose(
            cpu_model.coef_, cuda_model.coef_, rtol=5e-4, atol=5e-5
        )


    def test_elasticnetcv_predict''')
text, count = pattern.subn(replacement, text)
if count != 1:
    raise RuntimeError(f"ElasticNet GPU test block count={count}")
p.write_text(text)

src = Path("dev/tests/remote_gpu_test.py")
dst = Path("dev/manual/remote_gpu_runner.py")
if not src.exists() or dst.exists():
    raise RuntimeError("remote GPU runner move precondition failed")
dst.parent.mkdir(parents=True, exist_ok=True)
src.rename(dst)
remote = dst.read_text()
remote = remote.replace(
    "db = DBSCAN(eps=0.5, min_samples=5)",
    'db = DBSCAN(eps=0.5, min_samples=5, device="cuda")',
    1,
)
remote = remote.replace(
    "db2 = DBSCAN(eps=0.5, min_samples=5)",
    'db2 = DBSCAN(eps=0.5, min_samples=5, device="torch")',
    1,
)
remote = remote.replace(
    "UMAP(n_neighbors=5, n_epochs=2, device='cuda')",
    "UMAP(n_neighbors=5, n_epochs=2, device='torch')",
)
dst.write_text(remote)

Path("dev/tests/test_repository_review_batch2.py").write_text(
    dedent(
        '''
        import numpy as np
        import pytest
        from scipy import sparse

        from statgpu.cross_validation import (
            CVCache,
            batch_mse,
            detect_gpu_input,
            validate_cv_sample_weight,
        )
        from statgpu.unsupervised import KMeans, UMAP


        def test_cv_cache_rejects_invalid_size_and_zero_disables_storage():
            with pytest.raises(ValueError, match="maxsize"):
                CVCache(-1)
            cache = CVCache(0)
            cache.put("key", 1)
            assert cache.get("key") is None


        def test_sample_weight_and_batch_mse_validation():
            with pytest.raises(ValueError, match="positive sum"):
                validate_cv_sample_weight(np.zeros(3), 3)
            X = np.eye(2)
            y = np.ones(2)
            coefs = np.ones((2, 2))
            with pytest.raises(ValueError, match="chunk_size"):
                batch_mse(X, y, coefs, chunk_size=0)
            with pytest.raises(ValueError, match="intercepts length"):
                batch_mse(X, y, coefs, intercepts=np.zeros(1))
            with pytest.raises(ValueError, match="sample_weight length"):
                batch_mse(X, y, coefs, sample_weight=np.ones(1))
            with pytest.raises(ValueError, match="positive sum"):
                batch_mse(X, y, coefs, sample_weight=np.zeros(2))


        def test_detect_gpu_input_numpy_pair_is_unchanged():
            X, y = np.ones((3, 2)), np.ones(3)
            backend, X_out, y_out = detect_gpu_input(X, y)
            assert backend == "numpy"
            assert X_out is X and y_out is y


        def test_kmeans_score_validates_input_shape_and_sparsity():
            X = np.arange(24.0).reshape(8, 3)
            model = KMeans(n_clusters=2, random_state=0, device="cpu").fit(X)
            with pytest.raises(ValueError, match="2D"):
                model.score(X[:, 0])
            with pytest.raises(ValueError, match="features"):
                model.score(np.ones((2, 4)))
            with pytest.raises(NotImplementedError, match="sparse"):
                model.score(sparse.csr_matrix(X))


        def test_umap_small_spectral_initialization_has_requested_dimension():
            X = np.array([[0.0], [1.0], [2.0]])
            embedding = UMAP(
                n_neighbors=2,
                n_components=2,
                n_epochs=1,
                init="spectral",
                random_state=0,
                device="cpu",
            ).fit_transform(X)
            assert embedding.shape == (3, 2)
            assert np.all(np.isfinite(embedding))
        '''
    )
)
