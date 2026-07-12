"""Apply the focused ANOVA and kernel-method review patch.

This script is consumed by a temporary GitHub Actions workflow and deleted in the
same commit as the resulting source changes.
"""

from pathlib import Path


def replace_once(text: str, old: str, new: str, path: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


def replace_block(text: str, start: str, end: str, new: str, path: str) -> str:
    start_pos = text.index(start)
    end_pos = text.index(end, start_pos)
    return text[:start_pos] + new + text[end_pos:]


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------
path = Path("statgpu/anova/_oneway.py")
text = path.read_text()
text = replace_once(text, "    df_within : int\n", "    df_within : int or float\n", str(path))
text = replace_once(text, "    df_within: int\n", "    df_within: float\n", str(path))
path.write_text(text)

path = Path("statgpu/anova/_twoway.py")
text = path.read_text()
new_twoway = '''def f_twoway(
    data: Any,
    interaction: bool = True,
    backend: str = "auto",
    dtype: Any = None,
) -> TwoWayAnovaResult:
    """Perform a balanced two-way ANOVA.

    Each cell must contain the same number of observations.  Unbalanced
    designs require an explicit sums-of-squares convention (Type I/II/III),
    which this API does not expose, so they are rejected rather than silently
    applying the orthogonal balanced-design decomposition.
    """
    resolved = _resolve_backend(backend)
    xp = _get_xp(resolved)
    float_dtype = dtype if dtype is not None else xp.float64

    _, n_a, n_b, cell_arrays, cell_sizes_arr, _, _ = _parse_cells_vectorized(
        data, xp, float_dtype
    )
    if n_a < 2 or n_b < 2:
        raise ValueError("two-way ANOVA requires at least 2 levels for each factor")

    cell_sizes = np.asarray(_to_numpy(cell_sizes_arr), dtype=np.int64)
    if cell_sizes.size != n_a * n_b or np.any(cell_sizes != cell_sizes[0]):
        raise ValueError(
            "f_twoway currently requires a balanced design with equal cell sizes; "
            "unbalanced designs need an explicit Type I/II/III sums-of-squares choice"
        )
    n_cell = int(cell_sizes[0])
    if n_cell < 1:
        raise ValueError("each factor cell must contain at least one observation")

    cube = xp.stack(cell_arrays, axis=0).reshape(n_a, n_b, n_cell)
    cell_means = xp.mean(cube, axis=2)
    row_means = xp.mean(cell_means, axis=1)
    col_means = xp.mean(cell_means, axis=0)
    grand_mean = xp.mean(cell_means)

    ss_a = _to_float_scalar(
        float(n_b * n_cell) * xp.sum((row_means - grand_mean) ** 2)
    )
    ss_b = _to_float_scalar(
        float(n_a * n_cell) * xp.sum((col_means - grand_mean) ** 2)
    )
    interaction_effect = (
        cell_means - row_means[:, None] - col_means[None, :] + grand_mean
    )
    ss_ab_full = _to_float_scalar(
        float(n_cell) * xp.sum(interaction_effect ** 2)
    )
    ss_within_cells = _to_float_scalar(
        xp.sum((cube - cell_means[:, :, None]) ** 2)
    )

    df_a = n_a - 1
    df_b = n_b - 1
    df_ab_full = df_a * df_b
    n_total = n_a * n_b * n_cell

    if interaction:
        ss_ab = ss_ab_full
        df_ab = df_ab_full
        ss_error = ss_within_cells
        df_error = n_total - n_a * n_b
    else:
        ss_ab = 0.0
        df_ab = 0
        # Omitting the interaction makes its variation part of the additive
        # model residual.  Keeping only within-cell SSE inflates both main
        # effect F statistics.
        ss_error = ss_within_cells + ss_ab_full
        df_error = n_total - (1 + df_a + df_b)

    if df_error <= 0:
        raise ValueError(
            f"Not enough observations for the requested model: N={n_total}, "
            f"df_within={df_error}"
        )

    from statgpu.inference._distributions_backend import get_distribution

    f_dist = get_distribution("f", backend=resolved)
    ms_error = ss_error / df_error

    def _effect_test(ss_effect, df_effect):
        ms_effect = ss_effect / df_effect
        if ms_error == 0.0:
            if ms_effect == 0.0:
                return float("nan"), float("nan")
            return float("inf"), 0.0
        statistic = ms_effect / ms_error
        return statistic, _to_float_scalar(f_dist.sf(statistic, df_effect, df_error))

    f_a, p_a = _effect_test(ss_a, df_a)
    f_b, p_b = _effect_test(ss_b, df_b)
    if interaction:
        f_ab, p_ab = _effect_test(ss_ab, df_ab)
    else:
        f_ab = p_ab = None

    total_ss = ss_a + ss_b + ss_ab_full + ss_within_cells
    eta_a = ss_a / total_ss if total_ss > 0 else float("nan")
    eta_b = ss_b / total_ss if total_ss > 0 else float("nan")
    eta_ab = ss_ab_full / total_ss if total_ss > 0 and interaction else None

    return TwoWayAnovaResult(
        factor_a_statistic=f_a,
        factor_a_pvalue=p_a,
        factor_a_df=df_a,
        factor_a_eta_squared=eta_a,
        factor_b_statistic=f_b,
        factor_b_pvalue=p_b,
        factor_b_df=df_b,
        factor_b_eta_squared=eta_b,
        interaction_statistic=f_ab,
        interaction_pvalue=p_ab,
        interaction_df=df_ab if interaction else None,
        interaction_eta_squared=eta_ab,
        df_within=df_error,
        ss_within=ss_error,
    )


'''
text = replace_block(
    text,
    "def f_twoway(\n",
    "# ---------------------------------------------------------------------------\n# Helpers\n",
    new_twoway,
    str(path),
)
path.write_text(text)

path = Path("statgpu/anova/_posthoc.py")
text = path.read_text()
text = replace_once(
    text,
    "from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar\n",
    "from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy\n",
    str(path),
)
text = replace_once(
    text,
    '    if len(groups) < 2:\n        raise ValueError("tukey_hsd requires at least 2 groups")\n\n    resolved = _resolve_backend(backend, *groups)\n    xp = _get_xp(resolved)\n    float_dtype = dtype if dtype is not None else xp.float64\n\n    # Convert to numpy for statistics\n    flat_groups = [np.asarray(g, dtype=np.float64).ravel() for g in groups]\n',
    '    if len(groups) < 2:\n        raise ValueError("tukey_hsd requires at least 2 groups")\n    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:\n        raise ValueError("alpha must be finite and strictly between 0 and 1")\n\n    resolved = _resolve_backend(backend, *groups)\n\n    # The studentized-range calculation is CPU based.  Convert through the\n    # backend boundary so CuPy arrays and CUDA tensors are supported.\n    flat_groups = [np.asarray(_to_numpy(g), dtype=np.float64).ravel() for g in groups]\n',
    str(path),
)
text = replace_once(
    text,
    '    for i, g in enumerate(flat_groups):\n        if g.size < 2:\n            raise ValueError(f"Group {i} must have at least 2 observations for Tukey HSD")\n',
    '    for i, g in enumerate(flat_groups):\n        if g.size < 2:\n            raise ValueError(f"Group {i} must have at least 2 observations for Tukey HSD")\n        if not np.all(np.isfinite(g)):\n            raise ValueError(f"Group {i} contains NaN or infinite values")\n',
    str(path),
)
text = replace_once(
    text,
    '            q_stat = abs(mean_diff) / se if se > 0 else float("inf")\n',
    '            if se > 0:\n                q_stat = abs(mean_diff) / se\n            else:\n                q_stat = 0.0 if mean_diff == 0.0 else float("inf")\n',
    str(path),
)
text = replace_once(
    text,
    '    if len(groups) < 2:\n        raise ValueError("bonferroni requires at least 2 groups")\n\n    resolved = _resolve_backend(backend, *groups)\n    xp = _get_xp(resolved)\n\n    # Convert to numpy for statistics\n    flat_groups = [np.asarray(g, dtype=np.float64).ravel() for g in groups]\n',
    '    if len(groups) < 2:\n        raise ValueError("bonferroni requires at least 2 groups")\n    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:\n        raise ValueError("alpha must be finite and strictly between 0 and 1")\n\n    resolved = _resolve_backend(backend, *groups)\n\n    # Pairwise Welch tests are CPU based; use the explicit backend boundary.\n    flat_groups = [np.asarray(_to_numpy(g), dtype=np.float64).ravel() for g in groups]\n',
    str(path),
)
text = replace_once(
    text,
    '    for i, g in enumerate(flat_groups):\n        if g.size < 2:\n            raise ValueError(f"Group {i} must have at least 2 observations for t-test")\n',
    '    for i, g in enumerate(flat_groups):\n        if g.size < 2:\n            raise ValueError(f"Group {i} must have at least 2 observations for t-test")\n        if not np.all(np.isfinite(g)):\n            raise ValueError(f"Group {i} contains NaN or infinite values")\n',
    str(path),
)
text = replace_once(
    text,
    '            t_stat = mean_diff / se if se > 0 else float("inf")\n',
    '            if se > 0:\n                t_stat = mean_diff / se\n            else:\n                t_stat = 0.0 if mean_diff == 0.0 else np.copysign(float("inf"), mean_diff)\n',
    str(path),
)
path.write_text(text)

path = Path("statgpu/anova/_welch.py")
text = path.read_text()
text = replace_once(
    text,
    "from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar\n",
    "from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy\n",
    str(path),
)
text = replace_once(
    text,
    '        arr = np.asarray(g, dtype=np.float64).ravel()\n        if arr.size < 2:\n            raise ValueError("Welch ANOVA requires at least 2 observations per group")\n        flat_groups.append(arr)\n',
    '        arr = np.asarray(_to_numpy(g), dtype=np.float64).ravel()\n        if arr.size < 2:\n            raise ValueError("Welch ANOVA requires at least 2 observations per group")\n        if not np.all(np.isfinite(arr)):\n            raise ValueError("Welch ANOVA groups must contain only finite values")\n        flat_groups.append(arr)\n',
    str(path),
)
text = replace_once(
    text,
    '        # Some but not all zero: filter to non-zero variance groups\n        mask = s2_k > 0\n        flat_groups = [g for g, m in zip(flat_groups, mask) if m]\n        n_k = n_k[mask]\n        xbar_k = xbar_k[mask]\n        s2_k = s2_k[mask]\n        k = len(flat_groups)\n        if k < 2:\n            raise ValueError("After filtering zero-variance groups, fewer than 2 groups remain")\n',
    '        # Dropping only the zero-variance groups changes the null hypothesis.\n        # Require the caller to handle this degenerate mixed case explicitly.\n        raise ValueError(\n            "Welch ANOVA is undefined when only some groups have zero variance"\n        )\n',
    str(path),
)
text = replace_once(text, "        df_within=int(round(df2)),\n", "        df_within=float(df2),\n", str(path))
path.write_text(text)


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------
path = Path("statgpu/nonparametric/kernel_methods/_kernels.py")
text = path.read_text()
text = replace_once(
    text,
    "from statgpu.backends import xp_maximum\n",
    "from statgpu.backends import _to_float_scalar, xp_maximum\n",
    str(path),
)
helper = '''def _chi2_kernel_numpy_fallback(X, Y, gamma=1.0, max_elements=2_000_000):
    """Chunked NumPy chi-squared kernel used when sklearn is unavailable."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    n, p = X.shape
    m = Y.shape[0]
    chunk = min(p, max(1, int(max_elements) // max(n * m, 1)))
    chi2_dist = np.zeros((n, m), dtype=np.result_type(X.dtype, Y.dtype, np.float64))
    for start in range(0, p, chunk):
        end = min(start + chunk, p)
        Xc = X[:, None, start:end]
        Yc = Y[None, :, start:end]
        numerator = (Xc - Yc) ** 2
        denominator = Xc + Yc
        contribution = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=chi2_dist.dtype),
            where=denominator > 0,
        )
        chi2_dist += np.sum(contribution, axis=2)
    return np.exp(-float(gamma) * chi2_dist)


'''
text = replace_once(text, "def chi2_kernel(X, Y=None, gamma=1.0, xp=None):\n", helper + "def chi2_kernel(X, Y=None, gamma=1.0, xp=None):\n", str(path))
old_chi2_body = '''    if xp is None:
        xp = np
    if Y is None:
        Y = X

    # Ensure non-negative
    if xp is np:
        X = np.maximum(np.asarray(X), 0)
        Y = np.maximum(np.asarray(Y), 0)
    else:
        X = xp_maximum(X, 0, xp)
        Y = xp_maximum(Y, 0, xp)

    # chi-squared distance: sum_i (x_i - y_i)^2 / (x_i + y_i)
    if xp is np:
        # Use sklearn's Cython-optimized implementation for numpy
        try:
            from sklearn.metrics.pairwise import chi2_kernel as _sk_chi2
            return _sk_chi2(np.asarray(X), np.asarray(Y), gamma=gamma)
        except ImportError:
            pass
        # Fallback: chunked broadcasting
        n, p = X.shape
        m = Y.shape[0]
        chunk = min(p, max(1, 2000000 // max(n * m, 1)))
        chi2_dist = np.zeros((n, m), dtype=X.dtype)
        for start in range(0, p, chunk):
            end = min(start + chunk, p)
            Xc = X[:, start:end, None]
            Yc = Y[None, :, start:end]
            s = Xc + Yc
            np.maximum(s, 1e-10, out=s)
            chi2_dist += np.sum((Xc - Yc) ** 2 / s, axis=2)
        return np.exp(-gamma * chi2_dist)
    else:
        # GPU: use broadcasting
        X_exp = X[:, None, :]
        Y_exp = Y[None, :, :]
        numerator = (X_exp - Y_exp) ** 2
        denominator = X_exp + Y_exp
        denom_safe = xp_maximum(denominator, 1e-10, xp)
        chi2_dist = xp.sum(numerator / denom_safe, axis=2)

    return xp.exp(-gamma * chi2_dist, out=chi2_dist)
'''
new_chi2_body = '''    if xp is None:
        xp = np
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")

    if xp is np:
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
    elif Y is None:
        Y = X

    if getattr(X, "ndim", None) != 2 or getattr(Y, "ndim", None) != 2:
        raise ValueError("X and Y must be two-dimensional arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if _to_float_scalar(xp.min(X)) < 0 or _to_float_scalar(xp.min(Y)) < 0:
        raise ValueError("chi2_kernel requires non-negative input features")

    if xp is np:
        try:
            from sklearn.metrics.pairwise import chi2_kernel as _sk_chi2
            return _sk_chi2(X, Y, gamma=gamma)
        except ImportError:
            return _chi2_kernel_numpy_fallback(X, Y, gamma=gamma)

    X_exp = X[:, None, :]
    Y_exp = Y[None, :, :]
    numerator = (X_exp - Y_exp) ** 2
    denominator = X_exp + Y_exp
    denom_safe = xp_maximum(denominator, 1e-10, xp)
    chi2_dist = xp.sum(numerator / denom_safe, axis=2)
    return xp.exp(-gamma * chi2_dist)
'''
text = replace_once(text, old_chi2_body, new_chi2_body, str(path))
path.write_text(text)


# ---------------------------------------------------------------------------
# Kernel Ridge
# ---------------------------------------------------------------------------
path = Path("statgpu/nonparametric/kernel_methods/_krr.py")
text = path.read_text()
text = replace_once(
    text,
    "from statgpu.backends import _LINALG_ERRORS, _to_numpy, _torch_dev, xp_eye, xp_astype\n",
    "from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, _torch_dev, xp_eye, xp_astype\n",
    str(path),
)
new_fit = '''    def fit(self, X, y, sample_weight=None):
        """Fit Kernel Ridge Regression model."""
        self._backend = self._get_backend()
        xp = self._backend.xp
        self._xp = xp

        X_arr = xp_astype(self._to_array(X), xp.float64, xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")

        y_arr = xp_astype(self._to_array(y), xp.float64, xp)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2 or y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("y must be one- or two-dimensional with one row per X row")

        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if not bool(_to_float_scalar(xp.all(xp.isfinite(X_arr)))):
            raise ValueError("X contains NaN or infinite values")
        if not bool(_to_float_scalar(xp.all(xp.isfinite(y_arr)))):
            raise ValueError("y contains NaN or infinite values")

        n_samples = X_arr.shape[0]
        kernel_params = self._get_kernel_params()
        K = pairwise_kernels(X_arr, X_arr, metric=self.kernel, xp=xp, **kernel_params)
        eye = xp_eye(n_samples, K.dtype, xp, K)
        K_reg = K + alpha * eye

        try:
            self.dual_coef_ = xp.linalg.solve(K_reg, y_arr)
        except _LINALG_ERRORS:
            diagonal_scale = _to_float_scalar(xp.max(xp.abs(xp.diag(K))))
            jitter = max(diagonal_scale, 1.0) * 1e-10
            for _ in range(6):
                try:
                    self.dual_coef_ = xp.linalg.solve(K_reg + jitter * eye, y_arr)
                    break
                except _LINALG_ERRORS:
                    jitter *= 10.0
            else:
                raise ValueError(
                    "KernelRidge: regularized kernel matrix is singular even "
                    "after jitter escalation. Try increasing alpha."
                )

        self.X_fit_ = X_arr
        self.n_features_in_ = int(X_arr.shape[1])
        self._fitted = True
        return self

'''
text = replace_block(text, "    def fit(self, X, y, sample_weight=None):\n", "    def predict(self, X):\n", new_fit, str(path))
text = replace_once(
    text,
    '        X_arr = self._to_array(X)\n        kernel_params = self._get_kernel_params()\n',
    '        X_arr = xp_astype(self._to_array(X), xp.float64, xp)\n        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(-1, 1)\n        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:\n            raise ValueError(\n                f"X must have {self.n_features_in_} features; got "\n                f"{X_arr.shape[1] if X_arr.ndim == 2 else \'invalid shape\'}"\n            )\n        kernel_params = self._get_kernel_params()\n',
    str(path),
)
new_score = '''    def score(self, X, y):
        """Return uniform-average multi-output R-squared."""
        self._check_is_fitted()
        xp = self._xp

        y_pred = self.predict(X)
        y_arr = xp_astype(self._to_array(y), xp.float64, xp)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_arr.shape != y_pred.shape:
            raise ValueError(
                f"y has shape {tuple(y_arr.shape)} but predictions have shape "
                f"{tuple(y_pred.shape)}"
            )

        ss_res = xp.sum((y_arr - y_pred) ** 2, axis=0)
        ss_tot = xp.sum((y_arr - xp.mean(y_arr, axis=0)) ** 2, axis=0)
        ss_res_np = np.asarray(_to_numpy(ss_res), dtype=np.float64)
        ss_tot_np = np.asarray(_to_numpy(ss_tot), dtype=np.float64)
        scores = np.empty_like(ss_res_np)
        nonconstant = ss_tot_np > 0.0
        scores[nonconstant] = 1.0 - ss_res_np[nonconstant] / ss_tot_np[nonconstant]
        scores[~nonconstant] = np.where(ss_res_np[~nonconstant] <= 1e-15, 1.0, 0.0)
        return float(np.mean(scores))

'''
text = replace_block(text, "    def score(self, X, y):\n", "    def get_params(self, deep=True):\n", new_score, str(path))
path.write_text(text)

path = Path("statgpu/nonparametric/kernel_methods/_krr_cv.py")
text = path.read_text()
text = replace_once(
    text,
    "        n_samples = X_arr.shape[0]\n        n_targets = y_arr.shape[1]\n\n        # Compute full kernel matrix once\n",
    "        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(-1, 1)\n        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:\n            raise ValueError(\"X must be a non-empty two-dimensional array\")\n        if y_arr.ndim != 2 or y_arr.shape[0] != X_arr.shape[0]:\n            raise ValueError(\"y must have one row per X row\")\n\n        n_samples = X_arr.shape[0]\n        n_targets = y_arr.shape[1]\n        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)):\n            raise ValueError(\"cv must be an integer\")\n        n_folds = int(self.cv)\n        if n_folds < 2 or n_folds > n_samples:\n            raise ValueError(\"cv must satisfy 2 <= cv <= n_samples\")\n\n        # Compute full kernel matrix once\n",
    str(path),
)
text = replace_once(
    text,
    "        # Eigendecompose: K = Q @ diag(eigvals) @ Q.T\n        eigvals, Q = xp.linalg.eigh(K)\n\n        # Generate alpha grid if not provided\n        alphas_np = self.alphas\n        if alphas_np is None:\n            alphas_np = self._generate_alpha_grid(eigvals)\n        else:\n            alphas_np = np.asarray(alphas_np, dtype=np.float64).ravel()\n        n_alphas = alphas_np.shape[0]\n\n        # Project y into eigenbasis once: Q_T @ y\n        Q_T = Q.T  # eigh returns real eigenvectors for symmetric K\n        Qt_y = Q_T @ y_arr  # (n_samples, n_targets)\n\n        # K-fold CV\n        n_folds = int(self.cv)\n",
    "        # Generate alpha grid if not provided.  Only eigenvalues are needed;\n        # avoid materializing a full eigenvector matrix that the CV loop never uses.\n        alphas_np = self.alphas\n        if alphas_np is None:\n            eigvals = xp.linalg.eigvalsh(K)\n            alphas_np = self._generate_alpha_grid(eigvals)\n        else:\n            alphas_np = np.asarray(alphas_np, dtype=np.float64).ravel()\n        if alphas_np.size == 0 or not np.all(np.isfinite(alphas_np)) or np.any(alphas_np < 0):\n            raise ValueError(\"alphas must be a non-empty finite non-negative array\")\n        n_alphas = alphas_np.shape[0]\n\n        # K-fold CV\n",
    str(path),
)
text = replace_once(
    text,
    "        mse_table = xp_zeros((n_alphas, n_folds, n_targets), xp.float64, xp, X_arr)\n",
    "        mse_table = xp_zeros((n_alphas, n_folds, n_targets), xp.float64, xp, X_arr)\n        r2_table = xp_zeros((n_alphas, n_folds, n_targets), xp.float64, xp, X_arr)\n",
    str(path),
)
text = replace_once(
    text,
    "                mse_table[:, fi, :] = mse_vals\n            else:\n",
    "                mse_table[:, fi, :] = mse_vals\n                y_var = torch.mean((y_test - torch.mean(y_test, dim=0)) ** 2, dim=0)\n                r2_table[:, fi, :] = torch.where(\n                    y_var[None, :] > 0,\n                    1.0 - mse_vals / y_var[None, :],\n                    torch.where(mse_vals <= 1e-15, 1.0, 0.0),\n                )\n            else:\n",
    str(path),
)
text = replace_once(
    text,
    "                mse_table[:, fi, :] = mse_vals\n\n        # Mean MSE across folds: (n_alphas, n_targets)\n        mean_mse = xp.mean(mse_table, axis=1)\n",
    "                mse_table[:, fi, :] = mse_vals\n                y_var = xp.mean((y_test - xp.mean(y_test, axis=0)) ** 2, axis=0)\n                r2_table[:, fi, :] = xp.where(\n                    y_var[None, :] > 0,\n                    1.0 - mse_vals / y_var[None, :],\n                    xp.where(mse_vals <= 1e-15, 1.0, 0.0),\n                )\n\n        # Mean metrics across folds: (n_alphas, n_targets)\n        mean_mse = xp.mean(mse_table, axis=1)\n        mean_r2 = xp.mean(r2_table, axis=1)\n",
    str(path),
)
text = replace_once(
    text,
    "        # Compute mean R^2 across folds for best alpha\n        mean_mse_best = float(mean_mse[best_idx, 0].item()) if n_targets == 1 else float(xp.mean(mean_mse[best_idx]).item())\n        y_var = float(xp.var(y_arr).item())\n        self.best_score_ = 1.0 - mean_mse_best / y_var if y_var > 0 else 0.0\n",
    "        # Actual mean fold R^2, uniformly averaged across targets.\n        self.best_score_ = float(xp.mean(mean_r2[best_idx]).item())\n",
    str(path),
)
text = replace_once(
    text,
    '            "mse_table": _to_numpy(mse_table),\n            "best_alpha": self.alpha_,\n',
    '            "mse_table": _to_numpy(mse_table),\n            "mean_r2": _to_numpy(mean_r2),\n            "r2_table": _to_numpy(r2_table),\n            "best_alpha": self.alpha_,\n',
    str(path),
)
path.write_text(text)


# ---------------------------------------------------------------------------
# Kernel PCA and Nystroem
# ---------------------------------------------------------------------------
path = Path("statgpu/nonparametric/kernel_methods/_kpca.py")
text = path.read_text()
text = replace_once(
    text,
    "        n_samples = int(X_arr.shape[0])\n        n_features = int(X_arr.shape[1])\n",
    "        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:\n            raise ValueError(\"X must be a non-empty two-dimensional array\")\n        if isinstance(self.n_components, bool) or int(self.n_components) < 1:\n            raise ValueError(\"n_components must be a positive integer\")\n        if not np.isfinite(self.alpha) or self.alpha < 0:\n            raise ValueError(\"alpha must be finite and non-negative\")\n        if self.eigen_solver not in (\"auto\", \"dense\"):\n            raise ValueError(\"eigen_solver must be 'auto' or 'dense'\")\n\n        n_samples = int(X_arr.shape[0])\n        n_features = int(X_arr.shape[1])\n",
    str(path),
)
text = replace_once(
    text,
    "            eigenvalues = xp.asarray(eigvals_np, dtype=xp.float64)\n            eigenvectors = xp.asarray(eigvecs_np, dtype=xp.float64)\n\n        # Sort by descending eigenvalue\n",
    "            eigenvalues = xp_asarray(eigvals_np, dtype=xp.float64, xp=xp, ref_arr=K)\n            eigenvectors = xp_asarray(eigvecs_np, dtype=xp.float64, xp=xp, ref_arr=K)\n\n        # Adding alpha*I shifts eigenvalues but not eigenvectors.  Remove that\n        # shift before defining the KPCA embedding so training transform and\n        # out-of-sample transform use the same unregularized centered kernel.\n        eigenvalues = eigenvalues - float(self.alpha)\n\n        # Sort by descending eigenvalue\n",
    str(path),
)
text = replace_once(
    text,
    "        # Keep top n_components\n        eigenvalues = eigenvalues[:n_comp]\n        eigenvectors = eigenvectors[:, :n_comp]\n\n        # Normalize eigenvectors: alpha_k = v_k / sqrt(lambda_k)\n        # (only for positive eigenvalues)\n        norms = xp.sqrt(xp.maximum(eigenvalues, 1e-12))\n",
    "        # Keep positive eigenvalues only; centered kernels can have exact\n        # zero directions and indefinite user kernels can have negatives.\n        positive = eigenvalues > 1e-12\n        eigenvalues = eigenvalues[positive][:n_comp]\n        eigenvectors = eigenvectors[:, positive][:, :n_comp]\n        if int(eigenvalues.shape[0]) == 0:\n            raise ValueError(\"centered kernel matrix has no positive eigenvalues\")\n\n        norms = xp.sqrt(eigenvalues)\n",
    str(path),
)
text = replace_once(
    text,
    "        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)\n        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(-1, 1)\n\n        X_fit_arr = xp.asarray(self.X_fit_, dtype=xp.float64)\n",
    "        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)\n        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(-1, 1)\n        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:\n            raise ValueError(f\"X must have {self.n_features_in_} features\")\n\n        X_fit_arr = xp.asarray(self.X_fit_, dtype=xp.float64)\n",
    str(path),
)
text = replace_block(
    text,
    "    def fit_transform(self, X, y=None):\n",
    "    def predict(self, X):\n",
    '''    def fit_transform(self, X, y=None):
        """Fit and transform using the same out-of-sample centering path."""
        return self.fit(X, y).transform(X)

''',
    str(path),
)
path.write_text(text)

path = Path("statgpu/nonparametric/kernel_methods/_nystroem.py")
text = path.read_text()
text = replace_once(
    text,
    "        n_samples = int(X_arr.shape[0])\n        n_features = int(X_arr.shape[1])\n",
    "        if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:\n            raise ValueError(\"X must be a non-empty two-dimensional array\")\n        if isinstance(self.n_components, bool) or int(self.n_components) < 1:\n            raise ValueError(\"n_components must be a positive integer\")\n\n        n_samples = int(X_arr.shape[0])\n        n_features = int(X_arr.shape[1])\n",
    str(path),
)
text = replace_once(
    text,
    "        eigvals, eigvecs = np.linalg.eigh(K_mm_np)\n        eigvals = np.maximum(eigvals, 1e-12)\n\n        # Normalization: K_mm^{-1/2} = V @ diag(1/sqrt(λ)) @ V^T\n        self.normalization_ = (eigvecs * (1.0 / np.sqrt(eigvals))[None, :]) @ eigvecs.T\n        self.eigenvalues_ = eigvals\n",
    "        # SVD is stable for both PSD and indefinite kernels (for example,\n        # sigmoid).  Clipping negative eigenvalues from eigh would otherwise\n        # create enormous artificial features.\n        U, singular_values, Vt = np.linalg.svd(K_mm_np, full_matrices=False)\n        singular_values = np.maximum(singular_values, 1e-12)\n        self.normalization_ = (U / np.sqrt(singular_values)[None, :]) @ Vt\n        self.eigenvalues_ = singular_values\n",
    str(path),
)
text = replace_once(
    text,
    "        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)\n        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(-1, 1)\n\n        # Compute K_nm on the same device as X\n",
    "        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)\n        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(-1, 1)\n        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:\n            raise ValueError(f\"X must have {self.n_features_in_} features\")\n\n        # Compute K_nm on the same device as X\n",
    str(path),
)
path.write_text(text)

# Ensure the constant-target score test uses an interpolating full-rank kernel.
path = Path("dev/tests/test_module_review_anova_kernel.py")
text = path.read_text()
text = replace_once(
    text,
    'KernelRidge(alpha=0.0, kernel="linear").fit(X, np.ones(12))',
    'KernelRidge(alpha=0.0, kernel="rbf", gamma=0.2).fit(X, np.ones(12))',
    str(path),
)
path.write_text(text)

print("ANOVA and kernel-method review patch applied successfully")
