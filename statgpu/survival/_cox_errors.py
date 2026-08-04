"""Public numerical exception boundaries shared by Cox estimators."""


class CoxFitNumericalError(FloatingPointError):
    """A finite-input Cox fit produced an unrepresentable public result.

    This covers non-finite coefficients/likelihoods and finite coefficients
    whose hazard ratios are outside the finite positive float64 range.
    CoxPHCV may exclude this candidate while continuing the penalty path.
    Input, programming, allocator, driver, and other backend failures must use
    their original exception types and remain immediately visible.
    """


__all__ = ["CoxFitNumericalError"]
