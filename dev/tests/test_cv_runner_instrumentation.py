from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_case_specs_cover_six_initial_families() -> None:
    from dev.benchmarks.benchmark_cv_models import CASE_SPECS

    assert {spec.model_id for spec in CASE_SPECS} == {
        "RidgeCV",
        "LassoCV",
        "ElasticNetCV",
        "LogisticRegressionCV",
        "PenalizedGLM_CV",
        "CoxPHCV",
    }


def test_git_sha_honors_explicit_remote_provenance(monkeypatch) -> None:
    from dev.benchmarks.benchmark_cv_models import _git_sha

    expected = "ad2cf88d1d443a53eeb5207c33c4ee4f25de2400"
    monkeypatch.setenv("STATGPU_BENCHMARK_GIT_SHA", expected)
    assert _git_sha() == expected


def test_package_version_falls_back_to_importable_module(monkeypatch) -> None:
    import types

    import dev.benchmarks.benchmark_cv_models as benchmark_cv_models

    def missing_distribution(name: str) -> str:
        raise benchmark_cv_models.importlib.metadata.PackageNotFoundError(name)

    fake_cupy = types.SimpleNamespace(__version__="13.6.0")
    monkeypatch.setattr(benchmark_cv_models.importlib.metadata, "version", missing_distribution)
    monkeypatch.setattr(
        benchmark_cv_models.importlib,
        "import_module",
        lambda name: fake_cupy if name == "cupy" else (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    assert benchmark_cv_models._package_version("cupy") == "13.6.0"


def test_profiler_observes_selector_and_full_data_refit() -> None:
    from dev.benchmarks.benchmark_cv_models import RegionProfiler

    namespace = {"__name__": "statgpu.synthetic_cv", "time": time}
    exec(
        """
def _select_synthetic_cv():
    time.sleep(0.002)

class Refit:
    def fit(self, X, y):
        time.sleep(0.002)
        return self

class Outer:
    def fit(self, X, y):
        _select_synthetic_cv()
        Refit().fit(X, y)
        return self
""",
        namespace,
    )

    class Array:
        shape = (20, 3)

    outer = namespace["Outer"]()
    with RegionProfiler(outer, n_samples=20, synchronize=lambda: None) as profiler:
        outer.fit(Array(), object())

    assert profiler.selector_ms > 0
    assert profiler.refit_ms > 0
