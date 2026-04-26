"""
FormulaParser – R-style formula parser wrapping patsy.

Provides the ``FormulaParser`` class that converts R-style formulas like
``"y ~ x1 + x2 + C(sex)"`` into design matrices, using `patsy` internally.
"""

from typing import Optional, Tuple, List, Any

import numpy as np
import pandas as pd


class FormulaParser:
    """R-style formula parser that builds design matrices via patsy.

    Parameters
    ----------
    formula : str
        R-style formula string, e.g. ``"y ~ x1 + x2 + C(sex)"``.

    Attributes
    ----------
    formula : str
        The original formula string.
    design_info : patsy.DesignInfo or None
        Design matrix metadata (column names, term definitions).
        Set after :meth:`eval` is called.
    column_names : list[str] or None
        Names of the predictor columns (excluding the response).
        Set after :meth:`eval` is called.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     "y": np.random.randn(100),
    ...     "x1": np.random.randn(100),
    ...     "x2": np.random.randn(100),
    ... })
    >>> parser = FormulaParser("y ~ x1 + x2")
    >>> y, X, info = parser.eval(df)
    >>> parser.column_names
    ['x1', 'x2']
    """

    def __init__(self, formula: str):
        self.formula = formula
        self._design_info = None
        self._y_names: Optional[List[str]] = None

    @property
    def design_info(self):
        """Design matrix metadata, available after :meth:`eval`."""
        return self._design_info

    @property
    def column_names(self) -> Optional[List[str]]:
        """Predictor column names, available after :meth:`eval`."""
        if self._design_info is None:
            return None
        return list(self._design_info.column_names)

    def _require_patsy(self):
        """Return patsy module or raise ImportError with guidance."""
        try:
            import patsy
        except ImportError:
            raise ImportError(
                "The 'patsy' package is required for formula-based model fitting. "
                "Install it with: pip install statgpu[formula] "
                "or: pip install patsy"
            )
        return patsy

    def eval(
        self,
        data: pd.DataFrame,
        eval_env: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Parse formula and build design matrices from a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the columns referenced in the formula.
        eval_env : int, default=0
            Evaluation frame depth for patsy name resolution.

        Returns
        -------
        y : ndarray of shape (n_obs,) or (n_obs, n_responses)
            Response variable(s).
        X : ndarray of shape (n_obs, n_predictors)
            Predictor design matrix.
        design_info : patsy.DesignInfo
            Metadata for the predictor design (column names, term info).
        """
        patsy = self._require_patsy()
        data = data.copy()

        y, X = patsy.dmatrices(
            self.formula,
            data,
            eval_env=eval_env + 1,
            return_type="matrix",
        )

        self._y_names = list(y.design_info.column_names)
        self._design_info = X.design_info

        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        X_arr = np.asarray(X)

        return y_arr, X_arr, X.design_info

    def transform(
        self,
        new_data: pd.DataFrame,
        eval_env: int = 0,
    ) -> np.ndarray:
        """Build a design matrix for new data using the stored design_info.

        Used during :meth:`predict` to ensure new data is encoded
        with the same column structure (including categorical coding)
        as the training data.

        Parameters
        ----------
        new_data : pd.DataFrame
            DataFrame with the same columns as the training data.
        eval_env : int, default=0
            Evaluation frame depth for patsy name resolution.

        Returns
        -------
        X_new : ndarray of shape (n_new_obs, n_predictors)
            Design matrix aligned with the training design.

        Raises
        ------
        RuntimeError
            If :meth:`eval` has not been called yet (no design_info available).
        ValueError
            If new_data has columns that don't match the training structure.
        """
        if self._design_info is None:
            raise RuntimeError(
                "Cannot transform: no design_info available. "
                "Call eval() first on training data."
            )

        patsy = self._require_patsy()

        X_new = patsy.build_design_matrices(
            [self._design_info],
            new_data,
            return_type="matrix",
        )[0]

        return np.asarray(X_new)

    def summary(self) -> str:
        """Return a human-readable summary of the formula parsing.

        Shows the formula string, response variables, predictor names,
        and term definitions (useful for debugging categorical encoding).
        """
        lines = [f"Formula: {self.formula}"]

        if self._design_info is None:
            lines.append("(Not yet evaluated. Call eval() to parse.)")
            return "\n".join(lines)

        lines.append(f"Response: {self._y_names}")
        lines.append(f"Predictors ({len(self.column_names)}):")
        for name in self.column_names:
            lines.append(f"  - {name}")

        lines.append("\nTerms:")
        for term in self._design_info.term_name_slices.keys():
            lines.append(f"  {term}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        evaluated = "evaluated" if self._design_info is not None else "pending"
        return f"FormulaParser({self.formula!r}, {evaluated})"
