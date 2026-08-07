"""Aligned CPU reference measurements for canonical CV benchmarks."""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Callable

SUPPORTED_SKLEARN_MODELS = {
    "RidgeCV",
    "LassoCV",
    "ElasticNetCV",
    "LogisticRegressionCV",
}


def _candidate_values(spec: Any) -> list[dict[str, Any]]:
    """Return candidates on statgpu's public parameter scale."""
    grid = spec.grid_parameters
    if spec.model_id == "RidgeCV":
        return [{"alpha": float(value)} for value in grid["alphas"]]
    if spec.model_id == "LassoCV":
        return [{"alpha": float(value)} for value in grid["alphas"]]
    if spec.model_id == "ElasticNetCV":
        ratios = grid["l1_ratio"]
        if not isinstance(ratios, (list, tuple)):
            ratios = [ratios]
        return [
            {"alpha": float(alpha), "l1_ratio": float(ratio)}
            for ratio in ratios
            for alpha in grid["alphas"]
        ]
    if spec.model_id == "LogisticRegressionCV":
        return [{"C": float(value)} for value in grid["Cs"]]
    raise ValueError(f"no aligned sklearn reference for {spec.model_id}")


def _make_estimator(
    spec: Any,
    candidate: dict[str, Any],
    seed: int,
    n_fit_samples: int,
):
    """Construct an aligned reference on the current fit-subset scale."""
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge

    if spec.model_id == "RidgeCV":
        # statgpu uses average squared loss while sklearn Ridge uses an
        # unnormalised residual sum of squares. The mapping therefore depends
        # on the number of rows in each fold and differs again for final refit.
        return Ridge(
            alpha=float(n_fit_samples) * candidate["alpha"],
            fit_intercept=True,
        )
    if spec.model_id == "LassoCV":
        return Lasso(
            alpha=candidate["alpha"],
            fit_intercept=True,
            max_iter=5000,
            tol=1e-7,
            selection="cyclic",
        )
    if spec.model_id == "ElasticNetCV":
        return ElasticNet(
            alpha=candidate["alpha"],
            l1_ratio=candidate["l1_ratio"],
            fit_intercept=True,
            max_iter=5000,
            tol=1e-7,
            selection="cyclic",
        )
    if spec.model_id == "LogisticRegressionCV":
        return LogisticRegression(
            C=candidate["C"],
            penalty="l2",
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
            tol=1e-7,
            random_state=seed,
        )
    raise ValueError(f"unsupported sklearn reference model: {spec.model_id}")


def _folds(spec: Any, X, y, seed: int):
    """Reuse the exact fold generator exercised by statgpu CV estimators."""
    from statgpu.cross_validation._base import kfold_indices

    return kfold_indices(
        n_samples=len(X),
        n_splits=3,
        random_state=seed,
        shuffle=True,
    )


def _loss(spec: Any, estimator, X, y) -> float:
    import numpy as np

    if spec.task == "classification":
        from sklearn.metrics import log_loss

        probability = estimator.predict_proba(X)
        return float(log_loss(y, probability, labels=estimator.classes_))
    residual = np.asarray(estimator.predict(X)) - np.asarray(y)
    return float(np.mean(residual * residual))


def _run_once(
    spec: Any,
    seed: int,
    n_samples: int,
    n_features: int,
    data_factory: Callable[[int, int, int], tuple[Any, Any, Any, Any]],
) -> dict[str, Any]:
    import numpy as np

    X, y, X_test, y_test = data_factory(seed, n_samples, n_features)
    candidates = _candidate_values(spec)
    folds = _folds(spec, X, y, seed)

    total_start = time.perf_counter()
    cv_start = time.perf_counter()
    mean_losses: list[float] = []
    for candidate in candidates:
        fold_losses: list[float] = []
        for train_index, validation_index in folds:
            estimator = _make_estimator(
                spec,
                candidate,
                seed,
                n_fit_samples=len(train_index),
            )
            estimator.fit(X[train_index], y[train_index])
            fold_losses.append(
                _loss(spec, estimator, X[validation_index], y[validation_index])
            )
        mean_losses.append(float(np.mean(fold_losses)))
    cv_ms = (time.perf_counter() - cv_start) * 1000.0

    best_index = int(np.argmin(mean_losses))
    selected = candidates[best_index]
    final = _make_estimator(
        spec,
        selected,
        seed,
        n_fit_samples=n_samples,
    )
    refit_start = time.perf_counter()
    final.fit(X, y)
    refit_ms = (time.perf_counter() - refit_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    n_iter_raw = getattr(final, "n_iter_", None)
    if n_iter_raw is None:
        n_iter = None
    else:
        n_iter = int(np.max(np.asarray(n_iter_raw)))

    return {
        "seed": seed,
        "cv_evaluation_ms": cv_ms,
        "final_refit_ms": refit_ms,
        "total_fit_ms": total_ms,
        "selected_parameters": selected,
        "validation_score": mean_losses[best_index],
        "final_score": _loss(spec, final, X_test, y_test),
        "n_iter": n_iter,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "framework": "sklearn",
        "backend": None,
        "device": "cpu",
        "status": "unavailable",
        "reason": reason,
        "timing": None,
        "selected_parameters": None,
        "scores": None,
        "convergence": None,
        "repeat_samples": [],
    }


def build_sklearn_reference(
    spec: Any,
    seeds: list[int],
    n_samples: int,
    n_features: int,
    warmup: int,
    data_factory: Callable[[int, int, int], tuple[Any, Any, Any, Any]],
) -> dict[str, Any] | None:
    """Build an aligned sklearn row, or ``None`` when no equivalent is declared."""
    if spec.model_id not in SUPPORTED_SKLEARN_MODELS:
        return None
    if importlib.util.find_spec("sklearn") is None:
        return _unavailable("scikit-learn is not installed in this environment")

    import numpy as np

    try:
        for warmup_index in range(warmup):
            _run_once(
                spec,
                seeds[0] + 2_000_000 + warmup_index,
                n_samples,
                n_features,
                data_factory,
            )
        samples = [
            _run_once(spec, seed, n_samples, n_features, data_factory)
            for seed in seeds
        ]
        selected = samples[0]["selected_parameters"]
        if any(sample["selected_parameters"] != selected for sample in samples[1:]):
            raise RuntimeError("sklearn selected parameters changed across repeats")
        return {
            "framework": "sklearn",
            "backend": None,
            "device": "cpu",
            "status": "success",
            "reason": None,
            "timing": {
                "cv_evaluation_ms": float(
                    np.median([sample["cv_evaluation_ms"] for sample in samples])
                ),
                "final_refit_ms": float(
                    np.median([sample["final_refit_ms"] for sample in samples])
                ),
                "total_fit_ms": float(
                    np.median([sample["total_fit_ms"] for sample in samples])
                ),
                "peak_memory_bytes": None,
            },
            "selected_parameters": selected,
            "scores": {
                "validation_score": float(
                    np.mean([sample["validation_score"] for sample in samples])
                ),
                "final_score": float(
                    np.mean([sample["final_score"] for sample in samples])
                ),
            },
            "convergence": {
                "candidate_count": len(_candidate_values(spec)),
                "fold_count": 3,
                "failed_candidates": 0,
                "failed_folds": 0,
                "final_refit_converged": True,
                "n_iter": max((sample["n_iter"] or 0) for sample in samples),
            },
            "repeat_samples": [
                {
                    "seed": sample["seed"],
                    "cv_evaluation_ms": sample["cv_evaluation_ms"],
                    "final_refit_ms": sample["final_refit_ms"],
                    "total_fit_ms": sample["total_fit_ms"],
                }
                for sample in samples
            ],
        }
    except Exception as exc:
        return {
            **_unavailable(
                f"aligned sklearn reference failed: {type(exc).__name__}: {exc}"
            ),
            "status": "failed",
        }
