'''One-shot applicator for the PR #80 review fixes.

This file and its workflow delete themselves after applying the reviewed changes.
'''

from __future__ import annotations

import base64
from pathlib import Path
import re
import subprocess
import zlib


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def regex_once(
    text: str,
    pattern: str,
    replacement: str,
    *,
    label: str,
    flags: int = 0,
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return updated


# 1. Stable public Cox loss over the audited shared risk-set engine.
cox_loss = zlib.decompress(base64.b64decode('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' )).decode("utf-8")
(ROOT / "statgpu/losses/_cox_ph.py").write_text(cox_loss, encoding="utf-8")


# 2. Preserve the active backend in penalized Cox concordance scoring.
path = ROOT / "statgpu/linear_model/penalized/_penalized_cox.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from statgpu.backends._utils import _to_numpy",
    "from statgpu.backends._utils import _to_float_scalar, _to_numpy",
    label="penalized Cox scalar import",
)
text = regex_once(
    text,
    r"    def score\(self, X, y, sample_weight=None\):\n.*\Z",
    '    def score(self, X, y, sample_weight=None):\n        """Return the backend-native Harrell concordance index.\n\n        ``sample_weight`` is accepted for sklearn compatibility but is ignored\n        because the shared concordance definition is pair-based.\n        """\n        if sample_weight is not None:\n            import warnings\n\n            warnings.warn(\n                "sample_weight is not supported for C-index (ranking metric), "\n                "ignoring.",\n                UserWarning,\n                stacklevel=2,\n            )\n        if self.coef_ is None:\n            raise RuntimeError("Model has not been fitted yet.")\n\n        from statgpu.survival._risk_sets import counting_process_concordance\n\n        X = self._prepare_predict_X(X)\n        backend_name = self._prediction_backend_name()\n\n        if backend_name == "cupy":\n            import cupy as cp\n\n            Xb = cp.asarray(self._to_array(X, Device.CUDA), dtype=cp.float64)\n            if isinstance(y, dict):\n                if "time" not in y or "event" not in y:\n                    raise ValueError(\n                        "survival y dict must contain time and event"\n                    )\n                time = cp.asarray(y["time"], dtype=cp.float64).reshape(-1)\n                event = cp.asarray(y["event"], dtype=cp.float64).reshape(-1)\n            else:\n                yb = cp.asarray(y, dtype=cp.float64)\n                if yb.ndim != 2 or int(yb.shape[1]) != 2:\n                    raise ValueError(\n                        "y must be (n, 2) array with columns [time, event]"\n                    )\n                time, event = yb[:, 0], yb[:, 1]\n            coef = cp.asarray(self.coef_, dtype=cp.float64)\n        elif backend_name == "torch":\n            import torch\n\n            Xb = self._to_array(\n                X, Device.TORCH, backend="torch"\n            ).to(dtype=torch.float64)\n            if isinstance(y, dict):\n                if "time" not in y or "event" not in y:\n                    raise ValueError(\n                        "survival y dict must contain time and event"\n                    )\n                time = torch.as_tensor(\n                    y["time"],\n                    dtype=torch.float64,\n                    device=Xb.device,\n                ).reshape(-1)\n                event = torch.as_tensor(\n                    y["event"],\n                    dtype=torch.float64,\n                    device=Xb.device,\n                ).reshape(-1)\n            else:\n                yb = torch.as_tensor(\n                    y, dtype=torch.float64, device=Xb.device\n                )\n                if yb.ndim != 2 or int(yb.shape[1]) != 2:\n                    raise ValueError(\n                        "y must be (n, 2) array with columns [time, event]"\n                    )\n                time, event = yb[:, 0], yb[:, 1]\n            coef = torch.as_tensor(\n                self.coef_, dtype=Xb.dtype, device=Xb.device\n            )\n        else:\n            Xb = np.asarray(X, dtype=np.float64)\n            if isinstance(y, dict):\n                if "time" not in y or "event" not in y:\n                    raise ValueError(\n                        "survival y dict must contain time and event"\n                    )\n                time = np.asarray(\n                    _to_numpy(y["time"]), dtype=np.float64\n                ).reshape(-1)\n                event = np.asarray(\n                    _to_numpy(y["event"]), dtype=np.float64\n                ).reshape(-1)\n            else:\n                yb = np.asarray(_to_numpy(y), dtype=np.float64)\n                if yb.ndim != 2 or yb.shape[1] != 2:\n                    raise ValueError(\n                        "y must be (n, 2) array with columns [time, event]"\n                    )\n                time, event = yb[:, 0], yb[:, 1]\n            coef = np.asarray(self.coef_, dtype=np.float64)\n\n        if Xb.ndim == 1:\n            Xb = Xb.reshape(-1, 1)\n        if (\n            int(time.shape[0]) != int(event.shape[0])\n            or int(Xb.shape[0]) != int(time.shape[0])\n        ):\n            raise ValueError(\n                "X, time, and event must contain the same number of rows"\n            )\n\n        return _to_float_scalar(\n            counting_process_concordance(coef, Xb, time, event)\n        )\n' + "\n",
    label="penalized Cox backend-native score",
    flags=re.DOTALL,
)
path.write_text(text, encoding="utf-8")


# 3. Unified inference distributions and position-safe formula intercept removal.
path = ROOT / "statgpu/survival/_cox.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from scipy import stats\n",
    "",
    label="remove scipy stats import",
)
text = replace_once(
    text,
    "from statgpu.inference._distributions_backend import chi2",
    "from statgpu.inference._distributions_backend import chi2, norm",
    label="unified distribution imports",
)
text = text.replace("stats.norm.sf", "norm.sf")
text = text.replace("stats.chi2.sf", "chi2.sf")
if "stats." in text:
    remaining = [
        f"{line_number}: {line.strip()}"
        for line_number, line in enumerate(text.splitlines(), start=1)
        if "stats." in line
    ]
    raise RuntimeError(
        "unconverted scipy.stats use remains in _cox.py:\n"
        + "\n".join(remaining)
    )
text = replace_once(
    text,
    '            if "Intercept" in self._feature_names:\n                self._feature_names.remove("Intercept")\n                X_arr = X_arr[:, 1:]\n',
    '            if "Intercept" in self._feature_names:\n                intercept_index = self._feature_names.index("Intercept")\n                X_arr = np.delete(X_arr, intercept_index, axis=1)\n                self._feature_names = [\n                    name\n                    for index, name in enumerate(self._feature_names)\n                    if index != intercept_index\n                ]\n',
    label="formula intercept position",
)
path.write_text(text, encoding="utf-8")


# 4. Reuse the shared thread-safe CV cache and splitter.
path = ROOT / "statgpu/cross_validation/_base.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '            while len(self._cache) > self._maxsize:\n                self._cache.popitem(last=False)\n\n    @staticmethod\n',
    '            while len(self._cache) > self._maxsize:\n                self._cache.popitem(last=False)\n\n    def pop(self, key, default=None):\n        """Remove and return one cached value under the cache lock."""\n        with self._lock:\n            return self._cache.pop(key, default)\n\n    def clear(self) -> None:\n        """Remove every cached value under the cache lock."""\n        with self._lock:\n            self._cache.clear()\n\n    def __len__(self) -> int:\n        with self._lock:\n            return len(self._cache)\n\n    @staticmethod\n',
    label="CVCache mapping operations",
)
path.write_text(text, encoding="utf-8")

path = ROOT / "statgpu/survival/_cox_cv.py"
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from collections import OrderedDict\n",
    "",
    label="remove CoxCV OrderedDict import",
)
text = replace_once(
    text,
    "from statgpu.cross_validation._base import CVEstimatorBase",
    "from statgpu.cross_validation._base import CVCache, CVEstimatorBase, kfold_indices",
    label="shared CoxCV imports",
)
text = replace_once(
    text,
    '_COXPH_CV_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()',
    "_COXPH_CV_CACHE = CVCache(maxsize=_COXPH_CV_CACHE_MAXSIZE)",
    label="shared CoxCV cache instance",
)
text = regex_once(
    text,
    'def _coxcv_cache_get\\(cache_key: Optional\\[str\\]\\) -> Optional\\[Dict\\[str, Any\\]\\]:\n    """Get cached CoxPH CV results\\."""\n    if cache_key is None:\n        return None\n    val = _COXPH_CV_CACHE\\.get\\(cache_key\\)\n    if val is not None:\n        _COXPH_CV_CACHE\\.move_to_end\\(cache_key\\)\n        return copy\\.deepcopy\\(val\\)\n    return None\n',
    'def _coxcv_cache_get(cache_key: Optional[str]) -> Optional[Dict[str, Any]]:\n    """Get an isolated copy of cached CoxPH CV results."""\n    if cache_key is None:\n        return None\n    value = _COXPH_CV_CACHE.get(cache_key)\n    return None if value is None else copy.deepcopy(value)\n',
    label="shared CoxCV cache get",
)
text = regex_once(
    text,
    'def _coxcv_cache_put\\(cache_key: Optional\\[str\\], value: Dict\\[str, Any\\]\\) -> None:\n    """Put cached CoxPH CV results\\."""\n    if cache_key is None:\n        return\n    _COXPH_CV_CACHE\\[cache_key\\] = copy\\.deepcopy\\(value\\)\n    _COXPH_CV_CACHE\\.move_to_end\\(cache_key\\)\n    while len\\(_COXPH_CV_CACHE\\) > _COXPH_CV_CACHE_MAXSIZE:\n        _COXPH_CV_CACHE\\.popitem\\(last=False\\)\n',
    'def _coxcv_cache_put(cache_key: Optional[str], value: Dict[str, Any]) -> None:\n    """Store an isolated copy in the shared thread-safe CV cache."""\n    if cache_key is not None:\n        _COXPH_CV_CACHE.put(cache_key, copy.deepcopy(value))\n',
    label="shared CoxCV cache put",
)
text = regex_once(
    text,
    'def _kfold_indices\\(n_samples: int, n_splits: int, random_state: Optional\\[int\\] = None\\):\n    """Generate K-fold train/test indices\\."""\n    rng = np\\.random\\.RandomState\\(random_state\\)\n    indices = np\\.arange\\(n_samples\\)\n    rng\\.shuffle\\(indices\\)\n    fold_sizes = np\\.full\\(n_splits, n_samples // n_splits, dtype=np\\.int64\\)\n    fold_sizes\\[: n_samples % n_splits\\] \\+= 1\n    current = 0\n    folds = \\[\\]\n    for fold_size in fold_sizes:\n        start, stop = current, current \\+ fold_size\n        test_idx = indices\\[start:stop\\]\n        train_idx = np\\.concatenate\\(\\[indices\\[:start\\], indices\\[stop:\\]\\]\\)\n        folds\\.append\\(\\(train_idx, test_idx\\)\\)\n        current = stop\n    return folds\n',
    'def _kfold_indices(\n    n_samples: int,\n    n_splits: int,\n    random_state: Optional[int] = None,\n):\n    """Generate folds through the shared CV splitter."""\n    return kfold_indices(\n        n_samples,\n        n_splits=n_splits,\n        random_state=random_state,\n        shuffle=True,\n    )\n',
    label="shared CoxCV splitter",
)
path.write_text(text, encoding="utf-8")


# 5. Keep the root changelog concise.
path = ROOT / "CHANGELOG.md"
text = path.read_text(encoding="utf-8")
text = regex_once(
    text,
    r"## 2026-07-26\n\n### PR #80.*?(?=\n## 2026-07-25)",
    '## 2026-07-26\n\n### PR #80 — Complete GPU Cox phase one\n- Added Breslow, Efron, and Exact Cox risk sets with delayed entry, start-stop rows, strata, robust inference, and subject-grouped CV across NumPy, CuPy, and Torch.\n- Hardened penalized Cox estimation, formula handling, sklearn compatibility, numerical stability, and backend-preserving prediction and scoring.\n- Added synchronized GPU and R validation artifacts for coefficients, likelihood, covariance, convergence, and performance.\n',
    label="concise primary PR80 changelog entry",
    flags=re.DOTALL,
)
text = regex_once(
    text,
    r"\n### PR #80 — Cox survival Phase-1 completion and 0\.2\.2 compatibility review\n.*?(?=\n## 2026-07-24)",
    "",
    label="remove duplicate PR80 changelog entry",
    flags=re.DOTALL,
)
path.write_text(text, encoding="utf-8")


# 6. Focused regression tests for the concrete review findings.
(ROOT / "dev/tests/test_pr80_post_review_fixes.py").write_text(
    '"""Regression tests for the final PR #80 review fixes."""\n\nfrom __future__ import annotations\n\nimport inspect\n\nimport numpy as np\nimport pytest\nfrom numpy.testing import assert_allclose\n\nfrom statgpu.cross_validation._base import CVCache\nfrom statgpu.linear_model import PenalizedCoxPHModel\nfrom statgpu.losses import CoxPartialLikelihoodLoss\nfrom statgpu.survival import _cox_counting as counting_module\nfrom statgpu.survival import _cox_cv as cox_cv_module\nfrom statgpu.survival._cox_counting import fit_counting_process_cox\nfrom statgpu.survival._risk_sets import cox_counting_process_objective\n\n\n@pytest.mark.parametrize("ties", ["breslow", "efron"])\ndef test_penalized_cox_uses_failure_time_local_risk_scaling(ties):\n    # The maximum linear predictor leaves before the tied failures.  A single\n    # global max shift makes every later risk weight underflow to zero.\n    X = np.array([[1000.0], [0.0], [-1.0], [-2.0]])\n    time = np.array([1.0, 2.0, 2.0, 3.0])\n    event = np.array([0.0, 1.0, 1.0, 0.0])\n    y = np.column_stack([time, event])\n    coef = np.array([1.0])\n\n    reference = cox_counting_process_objective(\n        coef, X, time, event, ties=ties\n    )\n    loss = CoxPartialLikelihoodLoss(ties=ties)\n\n    value = loss.value(X, y, coef)\n    gradient = np.asarray(loss.gradient(X, y, coef))\n    hessian = np.asarray(loss.hessian(X, y, coef))\n\n    n = X.shape[0]\n    assert np.isfinite(value)\n    assert np.all(np.isfinite(gradient))\n    assert np.all(np.isfinite(hessian))\n    assert value == pytest.approx(\n        -float(reference["log_likelihood"]) / n, rel=1e-12, abs=1e-12\n    )\n    assert_allclose(\n        gradient,\n        -np.asarray(reference["score"]) / n,\n        rtol=1e-12,\n        atol=1e-12,\n    )\n    assert_allclose(\n        hessian,\n        np.asarray(reference["information"]) / n,\n        rtol=1e-12,\n        atol=1e-12,\n    )\n\n\ndef test_penalized_cox_first_order_path_avoids_information_matrix(monkeypatch):\n    import statgpu.losses._cox_ph as cox_loss_module\n\n    X = np.array([[1000.0], [0.0], [-1.0], [-2.0]])\n    y = np.array([[1.0, 0.0], [2.0, 1.0], [2.0, 1.0], [3.0, 0.0]])\n    coef = np.array([1.0])\n    loss = CoxPartialLikelihoodLoss(ties="efron")\n\n    def fail_shared_derivatives(*args, **kwargs):\n        raise AssertionError("first-order path requested the shared p-by-p information")\n\n    monkeypatch.setattr(\n        cox_loss_module,\n        "cox_counting_process_objective",\n        fail_shared_derivatives,\n    )\n    value, gradient = loss.fused_value_and_gradient(X, y, coef)\n    assert np.isfinite(value)\n    assert np.all(np.isfinite(np.asarray(gradient)))\n\n\ndef test_counting_solver_reports_line_search_failure_without_discarding_iterate(\n    monkeypatch,\n):\n    def objective(beta, X, stop, event, **kwargs):\n        beta_value = float(np.asarray(beta)[0])\n        return {\n            "log_likelihood": np.asarray(-(beta_value**2)),\n            "score": np.array([1.0]),\n            "information": np.array([[1.0]]),\n        }\n\n    monkeypatch.setattr(\n        counting_module, "cox_counting_process_objective", objective\n    )\n    result = fit_counting_process_cox(\n        np.ones((3, 1)),\n        np.array([1.0, 2.0, 3.0]),\n        np.array([1.0, 0.0, 0.0]),\n        ties="breslow",\n        max_iter=2,\n        compute_baseline=False,\n        compute_score_residuals=False,\n    )\n\n    assert result["converged"] is False\n    assert result["stop_reason"] == "line_search_failed"\n    assert_allclose(result["coef"], np.zeros(1))\n    assert len(result["objective_history"]) == 1\n\n\ndef test_cox_cv_reuses_thread_safe_shared_cache():\n    assert isinstance(cox_cv_module._COXPH_CV_CACHE, CVCache)\n    cox_cv_module._COXPH_CV_CACHE.clear()\n    cox_cv_module._COXPH_CV_CACHE.put("key", {"value": 1})\n    assert cox_cv_module._COXPH_CV_CACHE.get("key") == {"value": 1}\n    assert cox_cv_module._COXPH_CV_CACHE.pop("key") == {"value": 1}\n\n\ndef test_cox_inference_uses_unified_distribution_backend():\n    import statgpu.survival._cox as cox_module\n\n    source = inspect.getsource(cox_module)\n    assert "from scipy import stats" not in source\n    assert "stats.norm" not in source\n    assert "stats.chi2" not in source\n\n\n@pytest.mark.parametrize("device", ["cuda", "torch"])\ndef test_penalized_cox_score_preserves_explicit_gpu_backend(device, monkeypatch):\n    if device == "cuda":\n        cp = pytest.importorskip("cupy")\n        try:\n            if cp.cuda.runtime.getDeviceCount() < 1:\n                pytest.skip("CuPy CUDA device is unavailable")\n        except Exception as exc:\n            pytest.skip(f"CuPy CUDA backend is unavailable: {exc}")\n        X = cp.asarray([[1.0], [0.0], [-1.0]], dtype=cp.float64)\n        y = cp.asarray([[1.0, 1.0], [1.0, 0.0], [2.0, 0.0]])\n        backend_name = "cupy"\n    else:\n        torch = pytest.importorskip("torch")\n        if not torch.cuda.is_available():\n            pytest.skip("Torch CUDA device is unavailable")\n        X = torch.tensor(\n            [[1.0], [0.0], [-1.0]], dtype=torch.float64, device="cuda"\n        )\n        y = torch.tensor(\n            [[1.0, 1.0], [1.0, 0.0], [2.0, 0.0]],\n            dtype=torch.float64,\n            device="cuda",\n        )\n        backend_name = "torch"\n\n    model = PenalizedCoxPHModel(\n        device=device, compute_inference=False\n    )\n    model.coef_ = np.ones(1)\n    model._selected_backend_name = backend_name\n\n    import statgpu.linear_model.penalized._penalized_cox as module\n\n    def reject_host_transfer(*args, **kwargs):\n        raise AssertionError("full GPU score input was transferred to NumPy")\n\n    monkeypatch.setattr(module, "_to_numpy", reject_host_transfer)\n    assert model.score(X, y) == pytest.approx(1.0)\n',
    encoding="utf-8",
)


# Restore the maintained workflow, then remove every one-shot helper.
subprocess.run(
    ["git", "checkout", "origin/master", "--", ".github/workflows/test.yml"],
    cwd=ROOT,
    check=True,
)
(ROOT / "dev/_apply_pr80_review_fixes.py").unlink()
(ROOT / ".github/workflows/pr80-review-fix.yml").unlink()
