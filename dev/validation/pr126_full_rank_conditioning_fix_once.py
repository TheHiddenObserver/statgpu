"""One-shot helper for the PR126 full-rank conditioning review fix."""

from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "statgpu/panel/_fixed_effects.py",
    """from statgpu.backends import (
    _LINALG_ERRORS,
    _to_float_scalar,
    _to_numpy,
    xp_cholesky_solve,
    xp_maximum,
    xp_asarray,
)
""",
    """from statgpu.backends import (
    _to_float_scalar,
    _to_numpy,
    xp_maximum,
    xp_asarray,
)
""",
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    """        fit_rank = panel_matrix_rank(X_d, xp)
        if fit_rank < int(X_d.shape[1]):
            coef, _ = panel_lstsq(X_d, y_d, xp)
        else:
            XtX = X_d.T @ X_d
            Xty = X_d.T @ y_d
            try:
                coef = xp_cholesky_solve(XtX, Xty, xp)
            except _LINALG_ERRORS:
                try:
                    coef = xp.linalg.solve(XtX, Xty)
                except _LINALG_ERRORS:
                    coef, _ = panel_lstsq(X_d, y_d, xp)
""",
    """        # Solve and rank from the same SVD policy. Full numerical rank does
        # not certify normal equations: forming X'X squares the condition number
        # and can silently corrupt coefficients well before the SVD rank cutoff.
        coef, fit_rank = panel_lstsq(X_d, y_d, xp)
""",
)

replace_once(
    "statgpu/panel/_between.py",
    "from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray\n",
    "from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray\n",
)
replace_once(
    "statgpu/panel/_between.py",
    """        rank_mean = panel_matrix_rank(X_mean, xp)
        if rank_mean < int(X_mean.shape[1]):
            params, _ = panel_lstsq(X_mean, y_mean, xp)
        else:
            XtX = X_mean.T @ X_mean
            Xty = X_mean.T @ y_mean
            try:
                params = xp.linalg.solve(XtX, Xty)
            except _LINALG_ERRORS:
                params, _ = panel_lstsq(X_mean, y_mean, xp)
""",
    """        params, rank_mean = panel_lstsq(X_mean, y_mean, xp)
""",
)

replace_once(
    "statgpu/panel/_first_diff.py",
    "from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray\n",
    "from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray\n",
)
replace_once(
    "statgpu/panel/_first_diff.py",
    """        rank_diff = panel_matrix_rank(X_diff, xp)
        if rank_diff < int(X_diff.shape[1]):
            params, _ = panel_lstsq(X_diff, y_diff, xp)
        else:
            XtX = X_diff.T @ X_diff
            Xty = X_diff.T @ y_diff
            try:
                params = xp.linalg.solve(XtX, Xty)
            except _LINALG_ERRORS:
                params, _ = panel_lstsq(X_diff, y_diff, xp)
""",
    """        params, rank_diff = panel_lstsq(X_diff, y_diff, xp)
""",
)

replace_once(
    "statgpu/panel/_random_effects.py",
    """from statgpu.backends import (
    _LINALG_ERRORS,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_cholesky_solve,
    xp_zeros,
)
""",
    """from statgpu.backends import (
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_zeros,
)
""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    "from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank\n",
    "from statgpu.panel._linalg import panel_lstsq\n",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """        rank_between = panel_matrix_rank(X_bar_unique, xp)
        if rank_between < int(X_bar_unique.shape[1]):
            beta_between, _ = panel_lstsq(X_bar_unique, y_bar_unique, xp)
        else:
            XtX_b = X_bar_unique.T @ X_bar_unique
            Xty_b = X_bar_unique.T @ y_bar_unique
            try:
                beta_between = xp.linalg.solve(XtX_b, Xty_b)
            except _LINALG_ERRORS:
                beta_between, _ = panel_lstsq(X_bar_unique, y_bar_unique, xp)
""",
    """        beta_between, rank_between = panel_lstsq(
            X_bar_unique, y_bar_unique, xp
        )
""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """                rank_within = panel_matrix_rank(X_within_fit, xp)
                if rank_within < int(X_within_fit.shape[1]):
                    beta_within, _ = panel_lstsq(X_within_fit, y_within, xp)
                else:
                    XtX_w = X_within_fit.T @ X_within_fit
                    Xty_w = X_within_fit.T @ y_within
                    beta_within = xp.linalg.pinv(XtX_w) @ Xty_w
                resid_within = y_within - X_within_fit @ beta_within
""",
    """                beta_within, rank_within = panel_lstsq(
                    X_within_fit, y_within, xp
                )
                resid_within = y_within - X_within_fit @ beta_within
""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """        else:
            rank_within = panel_matrix_rank(X_within, xp)
            if rank_within < int(X_within.shape[1]):
                beta_within, _ = panel_lstsq(X_within, y_within, xp)
            else:
                XtX_w = X_within.T @ X_within
                Xty_w = X_within.T @ y_within
                try:
                    beta_within = xp.linalg.solve(XtX_w, Xty_w)
                except _LINALG_ERRORS:
                    beta_within, _ = panel_lstsq(X_within, y_within, xp)
            resid_within = y_within - X_within @ beta_within
""",
    """        else:
            beta_within, rank_within = panel_lstsq(X_within, y_within, xp)
            resid_within = y_within - X_within @ beta_within
""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """        rank_star = panel_matrix_rank(X_star, xp)
        if rank_star < int(X_star.shape[1]):
            beta_gls, _ = panel_lstsq(X_star, y_star, xp)
        else:
            XtX_s = X_star.T @ X_star
            Xty_s = X_star.T @ y_star
            try:
                beta_gls = xp_cholesky_solve(XtX_s, Xty_s, xp)
            except _LINALG_ERRORS:
                try:
                    beta_gls = xp.linalg.solve(XtX_s, Xty_s)
                except _LINALG_ERRORS:
                    beta_gls, _ = panel_lstsq(X_star, y_star, xp)
""",
    """        beta_gls, rank_star = panel_lstsq(X_star, y_star, xp)
""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """        if has_constant:
            restricted_X = X_star[:, constant_index : constant_index + 1]
            restricted_rank = panel_matrix_rank(restricted_X, xp)
            if restricted_rank < int(restricted_X.shape[1]):
                restricted_params, _ = panel_lstsq(restricted_X, y_star, xp)
            else:
                restricted_params = xp.linalg.pinv(restricted_X) @ y_star
            restricted_resid = y_star - restricted_X @ restricted_params
""",
    """        if has_constant:
            restricted_X = X_star[:, constant_index : constant_index + 1]
            restricted_params, restricted_rank = panel_lstsq(
                restricted_X, y_star, xp
            )
            restricted_resid = y_star - restricted_X @ restricted_params
""",
)


test_path = Path("dev/tests/test_panel_stage_c_full_rank_conditioning.py")
if test_path.exists():
    raise RuntimeError(f"{test_path}: test file already exists")
test_path.write_text(
    '''"""Regression coverage for numerically full-rank ill-conditioned panel fits."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, RandomEffects


def _shared_rcond(X):
    return max(X.shape) * np.finfo(np.float64).eps


def _assert_numerically_full_rank(X):
    singular = np.linalg.svd(X, compute_uv=False)
    cutoff = _shared_rcond(X) * singular.max()
    assert int(np.count_nonzero(singular > cutoff)) == X.shape[1]
    assert singular.max() / singular.min() > 1.0e7


def _near_collinear_columns(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    X = np.column_stack([x, x + 1.0e-7 * z])
    _assert_numerically_full_rank(X)
    return rng, X


def test_panelols_full_rank_near_collinearity_matches_shared_svd_solution():
    rng, X = _near_collinear_columns(200, 2026081701)
    beta = np.array([1.0, -1.0])
    y = X @ beta + 1.0e-9 * rng.normal(size=X.shape[0])
    expected = np.linalg.lstsq(X, y, rcond=_shared_rcond(X))[0]
    model = PanelOLS(cov_type="hc0").fit(X, y)
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=2e-8, atol=2e-8)
    assert_allclose(X @ model.coef_, X @ expected, rtol=0, atol=2e-12)


def test_between_full_rank_near_collinearity_matches_shared_svd_solution():
    rng, X = _near_collinear_columns(180, 2026081702)
    entity = np.arange(X.shape[0], dtype=np.int64)
    beta = np.array([1.0, -1.0])
    y = 0.35 + X @ beta + 1.0e-9 * rng.normal(size=X.shape[0])
    design = np.column_stack([np.ones(X.shape[0]), X])
    _assert_numerically_full_rank(design)
    expected = np.linalg.lstsq(design, y, rcond=_shared_rcond(design))[0]
    model = BetweenOLS(cov_type="hc0").fit(X, y, entity_ids=entity)
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=2e-8, atol=2e-8)
    assert_allclose(design @ model.coef_, design @ expected, rtol=0, atol=2e-12)


def test_first_difference_full_rank_near_collinearity_matches_shared_svd_solution():
    rng, differences = _near_collinear_columns(140, 2026081703)
    n_entities = differences.shape[0]
    entity = np.repeat(np.arange(n_entities), 2)
    time = np.tile(np.arange(2), n_entities)
    X = np.zeros((2 * n_entities, 2), dtype=np.float64)
    X[1::2] = differences
    alpha = rng.normal(scale=0.2, size=n_entities)
    beta = np.array([1.0, -1.0])
    y = np.repeat(alpha, 2)
    y[1::2] += differences @ beta + 1.0e-9 * rng.normal(size=n_entities)
    expected = np.linalg.lstsq(
        differences, y[1::2] - y[0::2], rcond=_shared_rcond(differences)
    )[0]
    model = FirstDifferenceOLS(cov_type="hc0").fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=2e-8, atol=2e-8)
    assert_allclose(
        differences @ model.coef_, differences @ expected, rtol=0, atol=2e-12
    )


def test_random_effects_full_rank_near_collinearity_matches_quasi_demeaned_svd():
    rng = np.random.default_rng(2026081704)
    n_entities, n_times = 60, 4
    entity = np.repeat(np.arange(n_entities), n_times)
    x = rng.normal(size=entity.size)
    z = rng.normal(size=entity.size)
    X = np.column_stack([np.ones(entity.size), x, x + 1.0e-7 * z])
    _assert_numerically_full_rank(X)
    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)
    beta = np.array([0.4, 1.0, -1.0])
    y = X @ beta + alpha + 1.0e-9 * rng.normal(size=entity.size)
    model = RandomEffects(cov_type="hc0").fit(X, y, entity_ids=entity)
    theta = float(model.theta_)
    X_mean = np.repeat(
        X.reshape(n_entities, n_times, -1).mean(axis=1), n_times, axis=0
    )
    y_mean = np.repeat(y.reshape(n_entities, n_times).mean(axis=1), n_times)
    X_star = X - theta * X_mean
    y_star = y - theta * y_mean
    _assert_numerically_full_rank(X_star)
    expected = np.linalg.lstsq(X_star, y_star, rcond=_shared_rcond(X_star))[0]
    assert model._coefficient_inference_available is True
    assert_allclose(model.coef_, expected, rtol=3e-8, atol=3e-8)
    assert_allclose(X_star @ model.coef_, X_star @ expected, rtol=0, atol=3e-12)
'''
)
