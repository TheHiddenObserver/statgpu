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
