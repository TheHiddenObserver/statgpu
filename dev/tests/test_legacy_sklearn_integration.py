"""scikit-learn <=1.2 integration contracts for public statgpu estimators."""

from __future__ import annotations

import numpy as np


def _regression_sample(seed: int = 20260804):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(48, 4))
    beta = np.array([1.5, -0.75, 0.4, 0.0])
    y = X @ beta + 0.05 * rng.normal(size=X.shape[0])
    return X, y


def test_clone_of_fitted_ridge_is_unfitted_and_preserves_parameters():
    from sklearn.base import clone
    from statgpu.linear_model import Ridge

    X, y = _regression_sample()
    fitted = Ridge(
        alpha=0.2,
        fit_intercept=True,
        compute_inference=False,
        device="cpu",
    ).fit(X, y)

    cloned = clone(fitted)

    assert cloned is not fitted
    assert type(cloned) is type(fitted)
    assert cloned._fitted is False
    assert cloned.get_params(deep=False)["alpha"] == 0.2
    assert cloned.get_params(deep=False)["device"] == "cpu"


def test_pipeline_nested_set_params_and_grid_search_work_on_legacy_sklearn():
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from statgpu.linear_model import Ridge

    X, y = _regression_sample(seed=20260805)
    pipeline = Pipeline(
        [
            (
                "ridge",
                Ridge(
                    alpha=1.0,
                    compute_inference=False,
                    device="cpu",
                ),
            )
        ]
    )

    pipeline.set_params(ridge__alpha=0.25)
    assert pipeline.get_params(deep=True)["ridge__alpha"] == 0.25
    assert pipeline.named_steps["ridge"].get_params(deep=False)["alpha"] == 0.25

    search = GridSearchCV(
        pipeline,
        param_grid={"ridge__alpha": [0.01, 0.1, 1.0]},
        scoring="neg_mean_squared_error",
        cv=3,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)

    prediction = np.asarray(search.predict(X))
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert search.best_params_["ridge__alpha"] in {0.01, 0.1, 1.0}
