"""
Stepwise model selection for regression models.
Supports forward, backward, and bidirectional selection.
"""

from typing import Optional, Union, List, Literal
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

from ..linear_model import LinearRegression, Ridge, Lasso, LogisticRegression


class StepwiseSelector:
    """
    Stepwise model selection using AIC or BIC criterion.
    
    Supports forward selection, backward elimination, and bidirectional search.
    
    Parameters
    ----------
    model_class : class
        Model class to use (LinearRegression, Ridge, Lasso, LogisticRegression).
    criterion : str, default='aic'
        Criterion for model selection: 'aic' or 'bic'.
    direction : str, default='both'
        Direction of search: 'forward', 'backward', or 'both'.
    max_features : int, optional
        Maximum number of features to select.
    **model_kwargs
        Additional arguments passed to the model.
    
    Attributes
    ----------
    selected_features_ : list
        Indices of selected features.
    best_model_ : object
        Fitted model with selected features.
    aic_history_ : list
        AIC values at each step.
    """
    
    def __init__(
        self,
        model_class,
        criterion: str = 'aic',
        direction: Literal['forward', 'backward', 'both'] = 'both',
        max_features: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **model_kwargs
    ):
        self.model_class = model_class
        self.criterion = criterion.lower()
        self.direction = direction
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.model_kwargs = model_kwargs
        
        if self.criterion not in ('aic', 'bic'):
            raise ValueError("criterion must be 'aic' or 'bic'")
        
        self.selected_features_ = None
        self.best_model_ = None
        self.aic_history_ = []
        self.bic_history_ = []
        self._score_cache = {}
    
    def fit(self, X, y):
        """
        Fit stepwise model selection.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        if self.max_features is None:
            self.max_features = n_features
        
        # Initialize
        self._score_cache = {}
        if self.direction == 'forward':
            selected = []
            remaining = list(range(n_features))
        elif self.direction == 'backward':
            selected = list(range(n_features))
            remaining = []
        else:  # both
            selected = []
            remaining = list(range(n_features))
        
        # Fit initial model
        best_score = self._fit_and_score(X, y, selected)
        self.aic_history_.append(best_score['aic'])
        self.bic_history_.append(best_score['bic'])
        
        improved = True
        iteration = 0
        
        while improved and len(selected) < self.max_features:
            improved = False
            iteration += 1
            
            if self.direction in ('forward', 'both'):
                # Try adding each remaining feature
                candidates = [(feature, selected + [feature]) for feature in remaining[:]]
                scores = self._evaluate_candidates(X, y, candidates)
                for feature, score in scores:
                    current_score = score[self.criterion]
                    if current_score < best_score[self.criterion]:
                        best_score = score
                        best_feature = feature
                        best_action = 'add'
                        improved = True
            
            if self.direction in ('backward', 'both') and len(selected) > 0:
                # Try removing each selected feature
                candidates = [(feature, [f for f in selected if f != feature]) for feature in selected[:]]
                scores = self._evaluate_candidates(X, y, candidates)
                for feature, score in scores:
                    current_score = score[self.criterion]
                    if current_score < best_score[self.criterion]:
                        best_score = score
                        best_feature = feature
                        best_action = 'remove'
                        improved = True
            
            if improved:
                if best_action == 'add':
                    selected.append(best_feature)
                    remaining.remove(best_feature)
                else:
                    selected.remove(best_feature)
                    remaining.append(best_feature)
                
                self.aic_history_.append(best_score['aic'])
                self.bic_history_.append(best_score['bic'])
                
                print(f"Step {iteration}: {best_action} feature {best_feature}, "
                      f"{self.criterion.upper()}={best_score[self.criterion]:.2f}")
        
        # Fit final model
        self.selected_features_ = sorted(selected)
        if len(selected) > 0:
            self.best_model_ = self.model_class(**self.model_kwargs)
            self.best_model_.fit(X[:, selected], y)
        
        return self
    
    def _evaluate_candidates(self, X, y, candidates):
        """Evaluate feature candidates in parallel with memoized scores."""
        feature_to_cache_key = {
            feature: tuple(sorted(feature_indices)) for feature, feature_indices in candidates
        }

        def _score_for_indices(feature_indices):
            key = tuple(sorted(feature_indices))
            if key in self._score_cache:
                return key, self._score_cache[key]

            score = self._fit_and_score(X, y, feature_indices)
            return key, score

        def eval_one(feature, feature_indices):
            key, score = _score_for_indices(feature_indices)
            self._score_cache[key] = score
            return feature, score

        def eval_one_parallel(feature, feature_indices):
            key, score = _score_for_indices(feature_indices)
            return feature, key, score

        if self.n_jobs == 1 or self.n_jobs is None or len(candidates) <= 1:
            return [eval_one(feature, feature_indices) for feature, feature_indices in candidates]

        out = Parallel(n_jobs=self.n_jobs)(
            delayed(eval_one_parallel)(feature, feature_indices) for feature, feature_indices in candidates
        )
        for feature, key, score in out:
            expected_key = feature_to_cache_key.get(feature)
            if expected_key is not None and expected_key == key and key not in self._score_cache:
                self._score_cache[key] = score
        return [(feature, score) for feature, _, score in out]
    
    def _fit_and_score(self, X, y, feature_indices):
        """Fit model and return AIC/BIC scores."""
        key = tuple(sorted(feature_indices))
        if key in self._score_cache:
            return self._score_cache[key]

        if len(feature_indices) == 0:
            # Null model
            score = {'aic': np.inf, 'bic': np.inf}
            self._score_cache[key] = score
            return score
        
        model = self.model_class(**self.model_kwargs)
        try:
            model.fit(X[:, feature_indices], y)
            
            if hasattr(model, 'aic') and model.aic is not None:
                score = {'aic': model.aic, 'bic': model.bic}
                self._score_cache[key] = score
                return score
            else:
                # Fallback: use R²-based approximation
                n = len(y)
                k = len(feature_indices) + 1  # +1 for intercept
                if hasattr(model, 'rsquared'):
                    r2 = model.rsquared
                    # Approximate AIC
                    aic = n * np.log(1 - r2 + 1e-10) + 2 * k
                    bic = n * np.log(1 - r2 + 1e-10) + k * np.log(n)
                    score = {'aic': aic, 'bic': bic}
                    self._score_cache[key] = score
                    return score
                else:
                    score = {'aic': np.inf, 'bic': np.inf}
                    self._score_cache[key] = score
                    return score
        except Exception:
            score = {'aic': np.inf, 'bic': np.inf}
            self._score_cache[key] = score
            return score
    
    def predict(self, X):
        """Predict using the best model."""
        if self.best_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X)
        return self.best_model_.predict(X[:, self.selected_features_])
    
    def score(self, X, y):
        """Return R² score of the best model."""
        if self.best_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = np.asarray(X)
        return self.best_model_.score(X[:, self.selected_features_], y)
    
    def summary(self):
        """Print summary of stepwise selection."""
        print("=" * 60)
        print("Stepwise Model Selection Summary")
        print("=" * 60)
        print(f"Criterion: {self.criterion.upper()}")
        print(f"Direction: {self.direction}")
        print(f"Selected features: {self.selected_features_}")
        print(f"Number of features: {len(self.selected_features_)}")
        if self.aic_history_:
            print(f"Final AIC: {self.aic_history_[-1]:.2f}")
            print(f"Final BIC: {self.bic_history_[-1]:.2f}")
        print("=" * 60)


def stepwise_selection(
    X, y,
    model_class=LinearRegression,
    criterion: str = 'aic',
    direction: str = 'both',
    **model_kwargs
):
    """
    Convenience function for stepwise selection.
    
    Parameters
    ----------
    X, y : array-like
        Training data.
    model_class : class
        Model class to use.
    criterion : str, default='aic'
        Selection criterion.
    direction : str, default='both'
        Search direction.
    **model_kwargs
        Model parameters.
    
    Returns
    -------
    selector : StepwiseSelector
        Fitted selector.
    """
    selector = StepwiseSelector(
        model_class=model_class,
        criterion=criterion,
        direction=direction,
        **model_kwargs
    )
    selector.fit(X, y)
    return selector
