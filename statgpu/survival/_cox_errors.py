"""Internal exception boundaries shared by Cox estimators."""


class CoxCandidateNumericalError(FloatingPointError):
    """A finite-input Cox candidate produced a non-finite fitted result.

    CoxPHCV may exclude this candidate while continuing the penalty path.
    Input, programming, allocator, driver, and other backend failures must use
    their original exception types and remain immediately visible.
    """


__all__ = ["CoxCandidateNumericalError"]
