"""
Stepwise model selection for regression models.
Supports forward, backward, and bidirectional selection.
"""

from typing import Optional, Union, List, Literal
import numpy as np
from copy import deepcopy

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
        **model_kwargs
    ):
        self.model_class = model_class
        self.criterion = criterion.lower()
        self.direction = direction
        self.max_features = max_features
        self.model_kwargs = model_kwargs
        
        if self.criterion not in ('aic', 'bic'):
            raise ValueError("criterion must be 'aic' or 'bic'")
        
        self.selected_features_ = None
        self.best_model_ = None
        self.aic_history_ = []
        self.bic_history_ = []
    
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
                for feature in remaining[:]:
                    candidate = selected + [feature]
                    score = self._fit_and_score(X, y, candidate)
                    current_score = score[self.criterion]
                    
                    if current_score < best_score[self.criterion]:
                        best_score = score
                        best_feature = feature
                        best_action = 'add'
                        improved = True
            
            if self.direction in ('backward', 'both') and len(selected) > 0:
                # Try removing each selected feature
                for feature in selected[:]:
                    candidate = [f for f in selected if f != feature]
                    score = self._fit_and_score(X, y, candidate)
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
    
    def _fit_and_score(self, X, y, feature_indices):
        """Fit model and return AIC/BIC scores."""
        if len(feature_indices) == 0:
            # Null model
            return {'aic': np.inf, 'bic': np.inf}
        
        model = self.model_class(**self.model_kwargs)
        try:
            model.fit(X[:, feature_indices], y)
            
            if hasattr(model, 'aic') and model.aic is not None:
                return {'aic': model.aic, 'bic': model.bic}
            else:
                # Fallback: use R²-based approximation
                n = len(y)
                k = len(feature_indices) + 1  # +1 for intercept
                if hasattr(model, 'rsquared'):
                    r2 = model.rsquared
                    # Approximate AIC
                    aic = n * np.log(1 - r2 + 1e-10) + 2 * k
                    bic = n * np.log(1 - r2 + 1e-10) + k * np.log(n)
                    return {'aic': aic, 'bic': bic}
                else:
                    return {'aic': np.inf, 'bic': np.inf}
        except Exception:
            return {'aic': np.inf, 'bic': np.inf}
    
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