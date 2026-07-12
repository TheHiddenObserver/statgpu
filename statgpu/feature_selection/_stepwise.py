"""Stepwise model selection for regression models.

The selector supports forward selection, backward elimination, and a
bidirectional search while keeping candidate feature order deterministic.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Literal, Optional

import numpy as np
from joblib import Parallel, delayed

from statgpu.backends import _to_float_scalar
from statgpu.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

__all__ = ["StepwiseSelector", "stepwise_selection"]


class StepwiseSelector:
    """Stepwise model selection using AIC or BIC.

    Parameters
    ----------
    model_class : class
        Estimator class to fit for every candidate subset. The class must expose
        ``fit`` and either finite ``aic``/``bic`` attributes or ``rsquared``.
    criterion : {'aic', 'bic'}, default='aic'
        Information criterion minimized during selection.
    direction : {'forward', 'backward', 'both'}, default='both'
        Search direction.
    max_features : int or None, default=None
        Maximum number of selected features. For backward selection, a value
        smaller than the input width is treated as a hard cap: features are
        removed until the cap is met, then elimination continues only while the
        criterion improves.
    n_jobs : int or None, default=None
        Number of joblib workers used to score candidates. Threads are used so
        device arrays are not copied into worker processes.
    verbose : bool, default=False
        Print accepted selection steps.
    **model_kwargs
        Arguments passed to ``model_class``.

    Notes
    -----
    Candidate subsets are always sorted before fitting. This is important: the
    final fitted coefficient order and the order used by ``predict`` must be
    identical.
    """

    _VALID_CRITERIA = {"aic", "bic"}
    _VALID_DIRECTIONS = {"forward", "backward", "both"}

    def __init__(
        self,
        model_class,
        criterion: str = "aic",
        direction: Literal["forward", "backward", "both"] = "both",
        max_features: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        **model_kwargs,
    ):
        self.model_class = model_class
        self.criterion = str(criterion).lower()
        self.direction = str(direction).lower()
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.verbose = bool(verbose)
        self.model_kwargs = dict(model_kwargs)
        self._validate_constructor_params()
        self._reset_fit_state()

    def _validate_constructor_params(self) -> None:
        if not callable(self.model_class):
            raise TypeError("model_class must be an estimator class or callable")
        if self.criterion not in self._VALID_CRITERIA:
            raise ValueError("criterion must be 'aic' or 'bic'")
        if self.direction not in self._VALID_DIRECTIONS:
            raise ValueError("direction must be 'forward', 'backward', or 'both'")
        if self.max_features is not None:
            if isinstance(self.max_features, bool) or not isinstance(
                self.max_features, Integral
            ):
                raise TypeError("max_features must be a non-negative integer or None")
            if int(self.max_features) < 0:
                raise ValueError("max_features must be non-negative")
        if self.n_jobs is not None:
            if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, Integral):
                raise TypeError("n_jobs must be an integer or None")
            if int(self.n_jobs) == 0:
                raise ValueError("n_jobs cannot be zero")

    def _reset_fit_state(self) -> None:
        self.selected_features_ = None
        self.best_model_ = None
        self.aic_history_ = []
        self.bic_history_ = []
        self.selection_history_ = []
        self._score_cache = {}
        self._fitted = False

    @staticmethod
    def _prepare_X(X):
        if not hasattr(X, "shape") or not hasattr(X, "ndim"):
            X = np.asarray(X)
        if int(X.ndim) != 2:
            raise ValueError(f"X must be 2D, got shape {getattr(X, 'shape', None)}")
        return X

    @staticmethod
    def _prepare_y(y):
        if not hasattr(y, "shape") or not hasattr(y, "ndim"):
            y = np.asarray(y)
        if int(y.ndim) == 2 and int(y.shape[1]) == 1:
            y = y.reshape(-1)
        if int(y.ndim) != 1:
            raise ValueError(f"y must be 1D, got shape {getattr(y, 'shape', None)}")
        return y

    def fit(self, X, y):
        """Run stepwise selection and fit the final estimator."""
        self._validate_constructor_params()
        self._reset_fit_state()
        X = self._prepare_X(X)
        y = self._prepare_y(y)
        n_samples, n_features = map(int, X.shape)
        if int(y.shape[0]) != n_samples:
            raise ValueError(
                f"X and y have inconsistent sample counts: {n_samples} and {y.shape[0]}"
            )
        if n_samples == 0:
            raise ValueError("X and y must contain at least one sample")

        feature_cap = n_features if self.max_features is None else int(self.max_features)
        if feature_cap > n_features:
            raise ValueError(
                f"max_features={feature_cap} exceeds n_features={n_features}"
            )

        if self.direction == "backward":
            selected = list(range(n_features))
        else:
            selected = []

        selected = sorted(selected)
        best_score = self._fit_and_score(X, y, selected)
        self._record_state(selected, best_score, action="initial", feature=None)

        iteration = 0
        while True:
            iteration += 1
            proposals = []

            # A backward search with a hard cap must remove features even if the
            # information criterion temporarily gets worse.
            mandatory_backward = (
                self.direction == "backward" and len(selected) > feature_cap
            )

            if not mandatory_backward and self.direction in ("forward", "both"):
                if len(selected) < feature_cap:
                    remaining = [j for j in range(n_features) if j not in selected]
                    proposals.extend(
                        ("add", feature, sorted(selected + [feature]))
                        for feature in remaining
                    )

            if self.direction in ("backward", "both") and selected:
                proposals.extend(
                    (
                        "remove",
                        feature,
                        [candidate for candidate in selected if candidate != feature],
                    )
                    for feature in selected
                )

            if not proposals:
                break

            evaluated = self._evaluate_candidates(X, y, proposals)
            finite = [
                item
                for item in evaluated
                if np.isfinite(item[3][self.criterion])
            ]
            if not finite:
                break

            action, feature, candidate_features, candidate_score = min(
                finite,
                key=lambda item: (
                    item[3][self.criterion],
                    0 if item[0] == "remove" else 1,
                    item[1],
                ),
            )

            current = float(best_score[self.criterion])
            candidate = float(candidate_score[self.criterion])
            tolerance = 1e-12 * max(1.0, abs(current))
            improves = candidate < current - tolerance
            if not (mandatory_backward or improves):
                break

            selected = list(candidate_features)
            best_score = candidate_score
            self._record_state(selected, best_score, action=action, feature=feature)
            if self.verbose:
                print(
                    f"Step {iteration}: {action} feature {feature}, "
                    f"{self.criterion.upper()}={candidate:.6g}"
                )

        # Fit in exactly the same deterministic order stored for prediction.
        self.selected_features_ = sorted(selected)
        self.best_model_ = self.model_class(**self.model_kwargs)
        self.best_model_.fit(X[:, self.selected_features_], y)
        self._fitted = True
        return self

    def _record_state(self, selected, score, *, action, feature) -> None:
        self.aic_history_.append(float(score["aic"]))
        self.bic_history_.append(float(score["bic"]))
        self.selection_history_.append(
            {
                "action": action,
                "feature": feature,
                "features": tuple(sorted(selected)),
                "aic": float(score["aic"]),
                "bic": float(score["bic"]),
            }
        )

    def _evaluate_candidates(self, X, y, proposals):
        """Evaluate candidate subsets, optionally with thread parallelism."""
        unique_keys = {tuple(features) for _, _, features in proposals}
        missing = [key for key in unique_keys if key not in self._score_cache]

        def evaluate_key(key):
            return key, self._fit_and_score_uncached(X, y, list(key))

        if missing:
            if self.n_jobs in (None, 1) or len(missing) == 1:
                results = [evaluate_key(key) for key in missing]
            else:
                results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(evaluate_key)(key) for key in missing
                )
            self._score_cache.update(results)

        return [
            (action, feature, tuple(features), self._score_cache[tuple(features)])
            for action, feature, features in proposals
        ]

    def _fit_and_score(self, X, y, feature_indices):
        key = tuple(sorted(feature_indices))
        if key not in self._score_cache:
            self._score_cache[key] = self._fit_and_score_uncached(X, y, list(key))
        return self._score_cache[key]

    def _fit_and_score_uncached(self, X, y, feature_indices):
        """Fit one candidate subset and return finite AIC/BIC scores."""
        model = self.model_class(**self.model_kwargs)
        try:
            model.fit(X[:, feature_indices], y)
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            if self.verbose:
                print(f"Candidate {feature_indices} failed numerically: {exc}")
            return {"aic": float("inf"), "bic": float("inf")}
        except RuntimeError as exc:
            message = str(exc).lower()
            if any(token in message for token in ("converg", "singular", "positive definite")):
                if self.verbose:
                    print(f"Candidate {feature_indices} failed numerically: {exc}")
                return {"aic": float("inf"), "bic": float("inf")}
            raise

        aic = getattr(model, "aic", None)
        bic = getattr(model, "bic", None)
        if aic is not None and bic is not None:
            aic_value = _to_float_scalar(aic)
            bic_value = _to_float_scalar(bic)
            if np.isfinite(aic_value) and np.isfinite(bic_value):
                return {"aic": aic_value, "bic": bic_value}

        r2 = getattr(model, "rsquared", None)
        if r2 is None:
            return {"aic": float("inf"), "bic": float("inf")}
        r2_value = _to_float_scalar(r2)
        if not np.isfinite(r2_value):
            return {"aic": float("inf"), "bic": float("inf")}

        n = int(y.shape[0])
        fit_intercept = bool(getattr(model, "fit_intercept", True))
        k = len(feature_indices) + int(fit_intercept)
        unexplained = max(1.0 - r2_value, np.finfo(float).tiny)
        return {
            "aic": float(n * np.log(unexplained) + 2.0 * k),
            "bic": float(n * np.log(unexplained) + k * np.log(n)),
        }

    def _check_is_fitted(self) -> None:
        if not self._fitted or self.best_model_ is None:
            raise RuntimeError("StepwiseSelector has not been fitted yet")

    def predict(self, X):
        """Predict with the selected feature subset."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        return self.best_model_.predict(X[:, self.selected_features_])

    def score(self, X, y):
        """Return the wrapped estimator's score."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        y = self._prepare_y(y)
        return self.best_model_.score(X[:, self.selected_features_], y)

    def summary(self):
        """Print a concise selection summary."""
        self._check_is_fitted()
        print("=" * 60)
        print("Stepwise Model Selection Summary")
        print("=" * 60)
        print(f"Criterion: {self.criterion.upper()}")
        print(f"Direction: {self.direction}")
        print(f"Selected features: {self.selected_features_}")
        print(f"Number of features: {len(self.selected_features_)}")
        print(f"Final AIC: {self.aic_history_[-1]:.6g}")
        print(f"Final BIC: {self.bic_history_[-1]:.6g}")
        print("=" * 60)

    def __sklearn_clone__(self):
        """Return an unfitted sklearn clone without copied selection state."""
        from copy import deepcopy

        return type(self)(**deepcopy(self.get_params(deep=False)))

    def get_params(self, deep=True):
        """Return constructor parameters using sklearn-style names."""
        params = {
            "model_class": self.model_class,
            "criterion": self.criterion,
            "direction": self.direction,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
        }
        params.update(self.model_kwargs)
        return params

    def set_params(self, **params):
        """Set selector or wrapped-model constructor parameters."""
        selector_names = {
            "model_class",
            "criterion",
            "direction",
            "max_features",
            "n_jobs",
            "verbose",
        }
        for name, value in params.items():
            if name in selector_names:
                setattr(self, name, value)
            else:
                self.model_kwargs[name] = value
        self.criterion = str(self.criterion).lower()
        self.direction = str(self.direction).lower()
        self.verbose = bool(self.verbose)
        self._validate_constructor_params()
        return self


def stepwise_selection(
    X,
    y,
    model_class=LinearRegression,
    criterion: str = "aic",
    direction: str = "both",
    **model_kwargs,
):
    """Fit and return a :class:`StepwiseSelector`."""
    selector = StepwiseSelector(
        model_class=model_class,
        criterion=criterion,
        direction=direction,
        **model_kwargs,
    )
    return selector.fit(X, y)
