"""Temporary exact-string patcher for the final PR80 review cycle."""

from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"expected block not found in {path}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/survival/_cox_score.py",
    "from statgpu.backends._utils import _require_real_array\n\n\ndef score(\n",
    '''from statgpu.backends._utils import _require_real_array


_MAX_CONCORDANCE_PAIR_ENTRIES = 2_000_000


def _concordance_batch_size(n_events: int, n_samples: int) -> int:
    """Bound pairwise concordance temporaries to a small fixed workspace."""
    return max(
        1,
        min(
            int(n_events),
            _MAX_CONCORDANCE_PAIR_ENTRIES // max(int(n_samples), 1),
        ),
    )


def score(
''',
)
replace_once(
    "statgpu/survival/_cox_score.py",
    "    chunk_size = max(1, min(n_events, int(128e6 / max(n_samples, 1))))\n",
    "    chunk_size = _concordance_batch_size(n_events, n_samples)\n",
)

replace_once(
    "statgpu/survival/_risk_sets.py",
    '''def _validate_counting_process_inputs(
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
) -> None:
''',
    '''def _validate_counting_process_inputs(
    X: Any,
    stop: Any,
    event: Any,
    start: Any,
    strata: Any,
    *,
    require_event: bool = True,
) -> None:
''',
)
replace_once(
    "statgpu/survival/_risk_sets.py",
    '    if _scalar_int(_sum(event, backend, xp)) == 0:\n        raise ValueError("at least one observed event is required")\n',
    '    if require_event and _scalar_int(_sum(event, backend, xp)) == 0:\n        raise ValueError("at least one observed event is required")\n',
)
replace_once(
    "statgpu/survival/_risk_sets.py",
    '''def prepare_counting_process_inputs(
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
''',
    '''def prepare_counting_process_inputs(
    X: Any,
    stop: Any,
    event: Any,
    *,
    start: Optional[Any] = None,
    strata: Optional[Any] = None,
    require_event: bool = True,
) -> Tuple[Any, Any, Any, Any, Any]:
''',
)
replace_once(
    "statgpu/survival/_risk_sets.py",
    "    _validate_counting_process_inputs(X, stop, event, start, strata)\n",
    '''    _validate_counting_process_inputs(
        X, stop, event, start, strata, require_event=bool(require_event)
    )
''',
)
risk_path = Path("statgpu/survival/_risk_sets.py")
risk_text = risk_path.read_text(encoding="utf-8")
marker = "def counting_process_concordance("
before, tail = risk_text.split(marker, 1)
old_call = '''    X, stop, event, start, strata = prepare_counting_process_inputs(
        X, stop, event, start=start, strata=strata
    )
'''
new_call = '''    X, stop, event, start, strata = prepare_counting_process_inputs(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        require_event=False,
    )
'''
if new_call not in tail:
    if old_call not in tail:
        raise SystemExit("concordance prepare call not found")
    tail = tail.replace(old_call, new_call, 1)
    risk_path.write_text(before + marker + tail, encoding="utf-8")

replace_once(
    "statgpu/survival/_cox_cv.py",
    '''            compute_inference=bool(self.compute_inference),
            cov_type=cov_type_name,
''',
    '''            compute_inference=bool(self.compute_inference),
            compute_cindex=False,
            cov_type=cov_type_name,
''',
)

replace_once(
    "statgpu/linear_model/penalized/_penalized_cox.py",
    "from ._base import PenalizedGeneralizedLinearModel\n\n\nclass PenalizedCoxPHModel",
    '''from ._base import PenalizedGeneralizedLinearModel


def _validate_boolean_control(value, name):
    """Accept booleans or integer 0/1 without interpreting truthy strings."""
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return
    raise ValueError(f"{name} must be a boolean or integer 0/1")


class PenalizedCoxPHModel''',
)
replace_once(
    "statgpu/linear_model/penalized/_penalized_cox.py",
    '''    ):
        if fit_intercept:
            raise ValueError(
''',
    '''    ):
        for name, value in (
            ("fit_intercept", fit_intercept),
            ("gpu_memory_cleanup", gpu_memory_cleanup),
            ("compute_inference", compute_inference),
            ("lla", lla),
        ):
            _validate_boolean_control(value, name)
        if bool(fit_intercept):
            raise ValueError(
''',
)
replace_once(
    "statgpu/linear_model/penalized/_penalized_cox.py",
    '''    def set_params(self, **params):
        """Set estimator parameters while preserving the no-intercept contract."""
        if params.get("fit_intercept", False):
''',
    '''    def set_params(self, **params):
        """Set estimator parameters while preserving the no-intercept contract."""
        for name in (
            "fit_intercept",
            "gpu_memory_cleanup",
            "compute_inference",
            "lla",
        ):
            if name in params:
                _validate_boolean_control(params[name], name)
        if bool(params.get("fit_intercept", False)):
''',
)

Path("dev/tests/test_pr80_complete_review_cycle.py").write_text(
    r'''"""Regression gates from the final complete PR80 review cycle."""

import numpy as np
import pytest

from statgpu.linear_model import PenalizedCoxPHModel
from statgpu.survival import CoxPH, CoxPHCV
from statgpu.survival._cox_score import (
    _MAX_CONCORDANCE_PAIR_ENTRIES,
    _concordance_batch_size,
)
from statgpu.survival._risk_sets import counting_process_concordance


def _fit_sample(seed=2401, n=36, p=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    stop = np.arange(1, n + 1, dtype=np.float64)
    event = np.ones(n, dtype=np.float64)
    event[::5] = 0.0
    event[0] = 1.0
    return X, stop, event


def test_ordinary_concordance_batch_is_bounded():
    batch = _concordance_batch_size(100_000, 1_000)
    assert batch == 2_000
    assert batch * 1_000 <= _MAX_CONCORDANCE_PAIR_ENTRIES
    assert _concordance_batch_size(0, 1_000) == 1


def test_all_censored_concordance_is_neutral_across_public_paths():
    X, stop, event = _fit_sample(p=1)
    fitted = CoxPH(
        compute_inference=False,
        compute_cindex=False,
        max_iter=80,
        tol=1e-7,
    ).fit(X, stop, event)
    X_score = X[:6]
    stop_score = np.arange(1, 7, dtype=np.float64)
    censored = np.zeros(6, dtype=np.float64)

    assert fitted.score(X_score, stop_score, censored) == 0.5
    assert fitted.score(
        X_score,
        stop_score,
        censored,
        start=np.zeros(6),
        strata=np.array([0, 0, 0, 1, 1, 1]),
    ) == 0.5
    assert float(
        counting_process_concordance(
            fitted.coef_,
            X_score,
            stop_score,
            censored,
            start=np.zeros(6),
            strata=np.array([0, 0, 0, 1, 1, 1]),
        )
    ) == 0.5


def test_penalized_cox_all_censored_score_is_neutral():
    X, stop, event = _fit_sample(seed=2402, p=1)
    model = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.2,
        max_iter=80,
        tol=1e-6,
        compute_inference=False,
    ).fit(X, np.column_stack((stop, event)))
    target = np.column_stack((stop[:5], np.zeros(5)))
    assert model.score(X[:5], target) == 0.5


def test_coxphcv_final_refit_skips_hidden_training_concordance():
    X, stop, event = _fit_sample(seed=2403)
    model = CoxPHCV(
        penalties=np.array([1.0]),
        cv=2,
        random_state=0,
        compute_inference=False,
        max_iter=100,
        tol=1e-6,
        device="cpu",
    ).fit(X, stop, event)
    assert model.estimator_.compute_cindex is False
    assert model.estimator_.concordance_ is None
    assert np.isfinite(model.score(X, stop, event))


@pytest.mark.parametrize(
    "name",
    ["fit_intercept", "gpu_memory_cleanup", "compute_inference", "lla"],
)
def test_penalized_cox_rejects_truthy_string_boolean_controls(name):
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        PenalizedCoxPHModel(**{name: "False"})

    model = PenalizedCoxPHModel()
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        model.set_params(**{name: "False"})


def test_penalized_cox_accepts_integer_boolean_controls_and_clones():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedCoxPHModel(
        fit_intercept=0,
        gpu_memory_cleanup=0,
        compute_inference=0,
        lla=1,
    )
    cloned = clone(model)
    assert cloned.fit_intercept == 0
    assert cloned.gpu_memory_cleanup == 0
    assert cloned.compute_inference == 0
    assert cloned.lla == 1
''',
    encoding="utf-8",
)
