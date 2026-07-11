"""Temporary patch script for repository review batch 3."""
from pathlib import Path
from textwrap import dedent

path = Path("statgpu/_base.py")
text = path.read_text()
marker = "    def adjust_pvalues(\n"
helper = dedent('''
    def _resolve_inference_backend(self, backend: str) -> str:
        """Resolve model-context inference backend from the estimator device."""
        backend_name = str(backend).strip().lower()
        if backend_name == "auto":
            compute_device = self._get_compute_device()
            if compute_device == Device.CUDA:
                return "cupy"
            if compute_device == Device.TORCH:
                return "torch"
        return backend_name

    def _cast_inference_array(self, value, backend_name: str):
        """Cast an inference input to the explicitly resolved backend."""
        if backend_name == "cupy":
            return self._to_array(value, Device.CUDA, backend="cupy")
        if backend_name == "torch":
            return self._to_array(value, Device.TORCH, backend="torch")
        if backend_name == "numpy":
            return self._to_numpy(value)
        return value

    def adjust_pvalues(
''')
if text.count(marker) != 1:
    raise RuntimeError(f"adjust_pvalues marker count={text.count(marker)}")
text = text.replace(marker, helper, 1)

resolver = '''        backend_name = str(backend).strip().lower()
        if backend_name == "auto" and self._get_compute_device() == Device.CUDA:
            backend_name = "cupy"
'''
if text.count(resolver) != 4:
    raise RuntimeError(f"inference resolver count={text.count(resolver)}")
text = text.replace(resolver, "        backend_name = self._resolve_inference_backend(backend)\n")

old = '''        if backend_name == "cupy":
            pvals = self._to_array(source, Device.CUDA)
        else:
            pvals = self._to_numpy(source)
'''
new = '''        pvals = self._cast_inference_array(source, backend_name)
'''
if text.count(old) != 1:
    raise RuntimeError(f"adjust cast block count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        if backend_name == "cupy":
            pvals = self._to_array(source, Device.CUDA)
            w_cast = None if weights is None else self._to_array(weights, Device.CUDA)
        elif backend_name == "numpy":
            pvals = self._to_numpy(source)
            w_cast = None if weights is None else self._to_numpy(weights)
        else:
            pvals = source
            w_cast = weights
'''
new = '''        pvals = self._cast_inference_array(source, backend_name)
        w_cast = (
            None
            if weights is None
            else self._cast_inference_array(weights, backend_name)
        )
'''
if text.count(old) != 1:
    raise RuntimeError(f"combine cast block count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        if backend_name == "cupy":
            arrays_cast = tuple(self._to_array(a, Device.CUDA) for a in arrays_use)
            strata_cast = None if strata is None else self._to_array(strata, Device.CUDA)
            clusters_cast = None if clusters is None else self._to_array(clusters, Device.CUDA)
        elif backend_name == "numpy":
            arrays_cast = tuple(self._to_numpy(a) for a in arrays_use)
            strata_cast = None if strata is None else self._to_numpy(strata)
            clusters_cast = None if clusters is None else self._to_numpy(clusters)
        else:
            arrays_cast = arrays_use
            strata_cast = strata
            clusters_cast = clusters
'''
new = '''        arrays_cast = tuple(
            self._cast_inference_array(a, backend_name) for a in arrays_use
        )
        strata_cast = (
            None
            if strata is None
            else self._cast_inference_array(strata, backend_name)
        )
        clusters_cast = (
            None
            if clusters is None
            else self._cast_inference_array(clusters, backend_name)
        )
'''
if text.count(old) != 1:
    raise RuntimeError(f"bootstrap cast block count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        if backend_name == "cupy":
            X_cast = self._to_array(X, Device.CUDA)
            y_cast = self._to_array(y, Device.CUDA)
            strata_cast = None if strata is None else self._to_array(strata, Device.CUDA)
            groups_cast = None if groups is None else self._to_array(groups, Device.CUDA)
        elif backend_name == "numpy":
            X_cast = self._to_numpy(X)
            y_cast = self._to_numpy(y)
            strata_cast = None if strata is None else self._to_numpy(strata)
            groups_cast = None if groups is None else self._to_numpy(groups)
        else:
            X_cast = X
            y_cast = y
            strata_cast = strata
            groups_cast = groups
'''
new = '''        X_cast = self._cast_inference_array(X, backend_name)
        y_cast = self._cast_inference_array(y, backend_name)
        strata_cast = (
            None
            if strata is None
            else self._cast_inference_array(strata, backend_name)
        )
        groups_cast = (
            None
            if groups is None
            else self._cast_inference_array(groups, backend_name)
        )
'''
if text.count(old) != 1:
    raise RuntimeError(f"permutation cast block count={text.count(old)}")
text = text.replace(old, new, 1)

text = text.replace(
    "backend : {'auto', 'numpy', 'cupy'}, default='auto'",
    "backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'",
)
text = text.replace(
    "Compute backend. ``'auto'`` uses CuPy when estimator device is CUDA.",
    "Compute backend. ``'auto'`` follows the estimator's resolved device.",
)
path.write_text(text)

Path("dev/tests/test_repository_review_batch3.py").write_text(
    dedent(
        '''
        import numpy as np

        from statgpu._base import BaseEstimator
        from statgpu._config import Device


        class DummyEstimator(BaseEstimator):
            def fit(self, X, y=None, **fit_params):
                self._fitted = True
                return self

            def predict(self, X):
                return X


        def test_model_context_resolves_torch_backend():
            model = DummyEstimator(device=Device.TORCH)
            assert model._resolve_inference_backend("auto") == "torch"
            assert model._resolve_inference_backend("numpy") == "numpy"


        def test_inference_cast_helper_preserves_numpy(monkeypatch):
            model = DummyEstimator(device=Device.CPU)
            value = np.array([0.1, 0.2])
            out = model._cast_inference_array(value, "numpy")
            assert out is value

            calls = []
            monkeypatch.setattr(
                model,
                "_to_array",
                lambda x, device, backend=None: calls.append((device, backend)) or x,
            )
            model._cast_inference_array(value, "torch")
            model._cast_inference_array(value, "cupy")
            assert calls == [
                (Device.TORCH, "torch"),
                (Device.CUDA, "cupy"),
            ]
        '''
    )
)
