"""Function-style distribution compatibility API.

This module keeps backward compatibility while the canonical API is object-
based (for example ``norm.cdf`` and ``t.ppf``).

Policy:
- R-style compatibility names (for example ``pnorm_gpu`` / ``qnorm_gpu``) are
    retained as compatibility aliases.
- Non-R historical names (for example ``norm_cdf_gpu`` / ``t_ppf_gpu``) are
    legacy and emit ``DeprecationWarning``.
"""

import warnings


def _warn_deprecated(name: str, replacement: str):
    warnings.warn(
        (
            f"`{name}` is deprecated and will be removed in a future release; "
            f"use `{replacement}` instead."
        ),
        category=DeprecationWarning,
        stacklevel=2,
    )


def t_cdf_gpu(x, df):
    _warn_deprecated("t_cdf_gpu", "t.cdf")
    from ._distributions_gpu import t

    return t.cdf(x, df=df)


def t_sf_gpu(x, df):
    _warn_deprecated("t_sf_gpu", "t.sf")
    from ._distributions_gpu import t

    return t.sf(x, df=df)


def t_ppf_gpu(q, df, *, max_bisect_steps=60):
    _warn_deprecated("t_ppf_gpu", "t.ppf")
    from ._distributions_gpu import t

    return t.ppf(q, df=df, max_bisect_steps=max_bisect_steps)


def t_two_sided_pvalue_gpu(stat_abs, df):
    _warn_deprecated("t_two_sided_pvalue_gpu", "t.two_sided_pvalue")
    from ._distributions_gpu import t

    return t.two_sided_pvalue(stat_abs, df=df)


def t_two_sided_critical_value_gpu(alpha, df, *, max_bisect_steps=60):
    _warn_deprecated("t_two_sided_critical_value_gpu", "t.two_sided_critical_value")
    from ._distributions_gpu import t

    return t.two_sided_critical_value(alpha, df=df, max_bisect_steps=max_bisect_steps)


def norm_cdf_gpu(x):
    _warn_deprecated("norm_cdf_gpu", "norm.cdf")
    from ._distributions_gpu import norm

    return norm.cdf(x)


def norm_sf_gpu(x):
    _warn_deprecated("norm_sf_gpu", "norm.sf")
    from ._distributions_gpu import norm

    return norm.sf(x)


def norm_ppf_gpu(q):
    _warn_deprecated("norm_ppf_gpu", "norm.ppf")
    from ._distributions_gpu import norm

    return norm.ppf(q)


def norm_isf_gpu(q):
    _warn_deprecated("norm_isf_gpu", "norm.isf")
    from ._distributions_gpu import norm

    return norm.isf(q)


def norm_two_sided_pvalue_gpu(stat_abs):
    _warn_deprecated("norm_two_sided_pvalue_gpu", "norm.two_sided_pvalue")
    from ._distributions_gpu import norm

    return norm.two_sided_pvalue(stat_abs)


def norm_two_sided_critical_value_gpu(alpha):
    _warn_deprecated("norm_two_sided_critical_value_gpu", "norm.two_sided_critical_value")
    from ._distributions_gpu import norm

    return norm.two_sided_critical_value(alpha)


def rnorm_gpu(size=None, loc=0.0, scale=1.0, dtype=None):
    from ._distributions_gpu import norm

    return norm.rvs(size=size, loc=loc, scale=scale, dtype=dtype)


def dnorm_gpu(x, *, loc=0.0, scale=1.0):
    from ._distributions_gpu import norm

    return norm.pdf(x, loc=loc, scale=scale)


def dt_gpu(x, df, *, loc=0.0, scale=1.0):
    from ._distributions_gpu import t

    return t.pdf(x, df=df, loc=loc, scale=scale)


def rt_gpu(df, size=None, dtype=None):
    from ._distributions_gpu import t

    return t.rvs(df=df, size=size, dtype=dtype)


def pnorm_gpu(x):
    from ._distributions_gpu import norm

    return norm.cdf(x)


def qnorm_gpu(q):
    from ._distributions_gpu import norm

    return norm.ppf(q)


def pt_gpu(x, df):
    from ._distributions_gpu import t

    return t.cdf(x, df=df)


def qt_gpu(q, df, *, max_bisect_steps=60):
    from ._distributions_gpu import t

    return t.ppf(q, df=df, max_bisect_steps=max_bisect_steps)


def dchisq_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import chi2

    return chi2.pdf(x, *shape_args, **kwargs)


def pchisq_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import chi2

    return chi2.cdf(x, *shape_args, **kwargs)


def qchisq_gpu(q, *shape_args, **kwargs):
    from ._distributions_gpu import chi2

    return chi2.ppf(q, *shape_args, **kwargs)


def rchisq_gpu(*shape_args, **kwargs):
    from ._distributions_gpu import chi2

    return chi2.rvs(*shape_args, **kwargs)


def dgamma_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import gamma

    return gamma.pdf(x, *shape_args, **kwargs)


def pgamma_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import gamma

    return gamma.cdf(x, *shape_args, **kwargs)


def qgamma_gpu(q, *shape_args, **kwargs):
    from ._distributions_gpu import gamma

    return gamma.ppf(q, *shape_args, **kwargs)


def rgamma_gpu(*shape_args, **kwargs):
    from ._distributions_gpu import gamma

    return gamma.rvs(*shape_args, **kwargs)


def dbeta_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import beta

    return beta.pdf(x, *shape_args, **kwargs)


def pbeta_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import beta

    return beta.cdf(x, *shape_args, **kwargs)


def qbeta_gpu(q, *shape_args, **kwargs):
    from ._distributions_gpu import beta

    return beta.ppf(q, *shape_args, **kwargs)


def rbeta_gpu(*shape_args, **kwargs):
    from ._distributions_gpu import beta

    return beta.rvs(*shape_args, **kwargs)


def df_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import f

    return f.pdf(x, *shape_args, **kwargs)


def pf_gpu(x, *shape_args, **kwargs):
    from ._distributions_gpu import f

    return f.cdf(x, *shape_args, **kwargs)


def qf_gpu(q, *shape_args, **kwargs):
    from ._distributions_gpu import f

    return f.ppf(q, *shape_args, **kwargs)


def rf_gpu(*shape_args, **kwargs):
    from ._distributions_gpu import f

    return f.rvs(*shape_args, **kwargs)


def dpois_gpu(k, *shape_args, **kwargs):
    from ._distributions_gpu import poisson

    return poisson.pmf(k, *shape_args, **kwargs)


def ppois_gpu(k, *shape_args, **kwargs):
    from ._distributions_gpu import poisson

    return poisson.cdf(k, *shape_args, **kwargs)


def qpois_gpu(q, *shape_args, **kwargs):
    from ._distributions_gpu import poisson

    return poisson.ppf(q, *shape_args, **kwargs)


def rpois_gpu(*shape_args, **kwargs):
    from ._distributions_gpu import poisson

    return poisson.rvs(*shape_args, **kwargs)


def dbinom_gpu(k, *shape_args, **kwargs):
    from ._distributions_gpu import binom

    return binom.pmf(k, *shape_args, **kwargs)


def pbinom_gpu(k, *shape_args, **kwargs):
    from ._distributions_gpu import binom

    return binom.cdf(k, *shape_args, **kwargs)


def qbinom_gpu(q, *shape_args, **kwargs):
    from ._distributions_gpu import binom

    return binom.ppf(q, *shape_args, **kwargs)


def rbinom_gpu(*shape_args, **kwargs):
    from ._distributions_gpu import binom

    return binom.rvs(*shape_args, **kwargs)


__all__ = [
    "t_cdf_gpu",
    "t_sf_gpu",
    "t_ppf_gpu",
    "t_two_sided_pvalue_gpu",
    "t_two_sided_critical_value_gpu",
    "norm_cdf_gpu",
    "norm_sf_gpu",
    "norm_ppf_gpu",
    "norm_isf_gpu",
    "norm_two_sided_pvalue_gpu",
    "norm_two_sided_critical_value_gpu",
    "rnorm_gpu",
    "dnorm_gpu",
    "dt_gpu",
    "rt_gpu",
    "pnorm_gpu",
    "qnorm_gpu",
    "pt_gpu",
    "qt_gpu",
    "dchisq_gpu",
    "pchisq_gpu",
    "qchisq_gpu",
    "rchisq_gpu",
    "dgamma_gpu",
    "pgamma_gpu",
    "qgamma_gpu",
    "rgamma_gpu",
    "dbeta_gpu",
    "pbeta_gpu",
    "qbeta_gpu",
    "rbeta_gpu",
    "df_gpu",
    "pf_gpu",
    "qf_gpu",
    "rf_gpu",
    "dpois_gpu",
    "ppois_gpu",
    "qpois_gpu",
    "rpois_gpu",
    "dbinom_gpu",
    "pbinom_gpu",
    "qbinom_gpu",
    "rbinom_gpu",
]
