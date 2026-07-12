"""Apply focused covariance and panel review fixes."""

from pathlib import Path


def replace_once(text, old, new, path):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def replace_block(text, start, end, new, path):
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + new + text[j:]


# ---------------------------------------------------------------------------
# Empirical covariance
# ---------------------------------------------------------------------------
path = Path("statgpu/covariance/_empirical.py")
text = path.read_text()
text = replace_once(
    text,
    "        n_samples = int(X_arr.shape[0])\n        p = int(X_arr.shape[1])\n\n        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)\n",
    "        n_samples = int(X_arr.shape[0])\n        p = int(X_arr.shape[1])\n        if p != self.n_features_:\n            raise ValueError(f\"X must have {self.n_features_} features, got {p}\")\n        if n_samples == 0:\n            raise ValueError(\"X must contain at least one sample\")\n\n        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)\n",
    str(path),
)
text = replace_once(
    text,
    "        sign, logdet = xp.linalg.slogdet(cov)\n        logdet_val = _to_float_scalar(logdet)\n\n        # Average log-likelihood:\n",
    "        sign, logdet = xp.linalg.slogdet(cov)\n        sign_val = _to_float_scalar(sign)\n        if sign_val <= 0:\n            return float(\"-inf\")\n        logdet_val = _to_float_scalar(logdet)\n\n        # Average log-likelihood:\n",
    str(path),
)
text = replace_once(
    text,
    "        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(1, -1)\n\n        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)\n",
    "        if X_arr.ndim == 1:\n            X_arr = X_arr.reshape(1, -1)\n        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_:\n            got = X_arr.shape[1] if X_arr.ndim == 2 else \"invalid\"\n            raise ValueError(f\"X must have {self.n_features_} features, got {got}\")\n\n        loc = xp_asarray(self.location_, dtype=xp.float64, xp=xp, ref_arr=X_arr)\n",
    str(path),
)
old_inv = '''    jitter = base
    for _ in range(12):
        try:
            if jitter > 0:
                S_work = S + jitter * eye
            else:
                S_work = S

            inv_S = xp.linalg.inv(S_work)
            test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
            if np.isfinite(test_val):
                return inv_S
        except _LINALG_ERRORS + (ValueError,):
            pass
        jitter *= 10.0
'''
new_inv = '''    # Preserve the exact estimator whenever the covariance is invertible.
    # Jitter is a fallback, not part of the empirical covariance definition.
    try:
        inv_S = xp.linalg.inv(S)
        test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
        if np.isfinite(test_val):
            return inv_S
    except _LINALG_ERRORS + (ValueError,):
        pass

    jitter = base
    for _ in range(12):
        try:
            inv_S = xp.linalg.inv(S + jitter * eye)
            test_val = _to_float_scalar(xp.max(xp.abs(inv_S)))
            if np.isfinite(test_val):
                return inv_S
        except _LINALG_ERRORS + (ValueError,):
            pass
        jitter *= 10.0
'''
text = replace_once(text, old_inv, new_inv, str(path))
path.write_text(text)


# ---------------------------------------------------------------------------
# Graphical Lasso
# ---------------------------------------------------------------------------
path = Path("statgpu/covariance/_graphical_lasso.py")
text = path.read_text()
text = replace_once(
    text,
    "from statgpu.backends import _get_xp\n",
    "from statgpu.backends import _get_xp, _to_numpy\n",
    str(path),
)
new_fit = '''    def fit(self, X, y=None):
        """Fit graphical lasso by covariance block coordinate descent."""
        alpha = float(self.alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if isinstance(self.max_iter, bool) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.tol) or float(self.tol) <= 0:
            raise ValueError("tol must be finite and positive")

        X_np = np.asarray(_to_numpy(X), dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if X_np.ndim != 2 or X_np.shape[0] < 2 or X_np.shape[1] < 1:
            raise ValueError("X must be a non-empty 2D array with at least 2 samples")
        if not np.all(np.isfinite(X_np)):
            raise ValueError("X contains NaN or infinite values")

        n, p = X_np.shape
        if self.assume_centered:
            location_np = np.zeros(p, dtype=np.float64)
            X_centered = X_np
        else:
            location_np = X_np.mean(axis=0)
            X_centered = X_np - location_np
        empirical = X_centered.T @ X_centered / float(n)

        if alpha == 0.0 or p == 1:
            covariance = empirical.copy()
            precision = np.linalg.pinv(covariance)
            self.n_iter_ = 1
        else:
            covariance = empirical.copy()
            # The graphical-lasso penalty excludes precision diagonal terms;
            # consequently the dual covariance diagonal stays empirical.
            np.fill_diagonal(covariance, np.diag(empirical))
            inner_tol = min(1e-8, float(self.tol) * 0.1)
            beta_cache = [np.zeros(p - 1, dtype=np.float64) for _ in range(p)]
            self.n_iter_ = 0

            for outer in range(int(self.max_iter)):
                previous = covariance.copy()
                self.n_iter_ = outer + 1
                for j in range(p):
                    mask = np.arange(p) != j
                    W11 = covariance[np.ix_(mask, mask)]
                    s12 = empirical[mask, j]
                    beta = beta_cache[j].copy()

                    for _ in range(1000):
                        beta_old = beta.copy()
                        for coordinate in range(p - 1):
                            diagonal = W11[coordinate, coordinate]
                            if diagonal <= 0:
                                raise ValueError("GraphicalLasso encountered a non-positive covariance diagonal")
                            partial = (
                                s12[coordinate]
                                - W11[coordinate] @ beta
                                + diagonal * beta[coordinate]
                            )
                            beta[coordinate] = _soft_threshold(partial, alpha) / diagonal
                        if np.max(np.abs(beta - beta_old)) <= inner_tol:
                            break

                    beta_cache[j] = beta
                    w12 = W11 @ beta
                    covariance[mask, j] = w12
                    covariance[j, mask] = w12
                    covariance[j, j] = empirical[j, j]

                if np.max(np.abs(covariance - previous)) <= float(self.tol):
                    break

            covariance = 0.5 * (covariance + covariance.T)
            precision = np.linalg.pinv(covariance)
            precision = 0.5 * (precision + precision.T)

        backend_name = _detect_backend(X, self._get_compute_device())
        xp = _get_xp(backend_name)
        _ref = None
        if backend_name == "torch":
            import torch
            device = self._get_compute_device()
            target = "cuda" if device.value in ("torch", "cuda") else "cpu"
            _ref = torch.empty(0, dtype=torch.float64, device=target)
        kwargs = {"device": _ref.device} if _ref is not None else {}

        self.covariance_ = xp.asarray(covariance, dtype=xp.float64, **kwargs)
        self.precision_ = xp.asarray(precision, dtype=xp.float64, **kwargs)
        self.location_ = xp.asarray(location_np, dtype=xp.float64, **kwargs)
        self.n_samples_ = n
        self.n_features_ = p
        self._backend_name = backend_name
        self._fitted = True
        return self

'''
text = replace_block(text, "    def fit(self, X, y=None):\n", "    def get_params(self, deep=True):\n", new_fit, str(path))
text = replace_once(
    text,
    "        X_np = np.asarray(X, dtype=np.float64)\n",
    "        X_np = np.asarray(_to_numpy(X), dtype=np.float64)\n",
    str(path),
)
text = replace_once(
    text,
    "        n, p = X_np.shape\n\n        # Build alpha grid\n        if isinstance(self.alphas, int):\n            alpha_grid = np.logspace(-2, 0, self.alphas)\n        else:\n            alpha_grid = np.asarray(self.alphas, dtype=np.float64)\n\n        # K-fold CV\n",
    "        n, p = X_np.shape\n        if n < 2 or p < 1 or not np.all(np.isfinite(X_np)):\n            raise ValueError(\"X must be a finite 2D array with at least 2 samples\")\n        if isinstance(self.cv, bool) or not isinstance(self.cv, (int, np.integer)):\n            raise ValueError(\"cv must be an integer\")\n        if int(self.cv) < 2 or int(self.cv) > n:\n            raise ValueError(\"cv must satisfy 2 <= cv <= n_samples\")\n\n        # Build alpha grid\n        if isinstance(self.alphas, (int, np.integer)) and not isinstance(self.alphas, bool):\n            if int(self.alphas) < 1:\n                raise ValueError(\"alphas must be a positive integer or a non-empty array\")\n            alpha_grid = np.logspace(-2, 0, int(self.alphas))\n        else:\n            alpha_grid = np.asarray(self.alphas, dtype=np.float64).ravel()\n        if alpha_grid.size == 0 or not np.all(np.isfinite(alpha_grid)) or np.any(alpha_grid < 0):\n            raise ValueError(\"alphas must be finite, non-negative, and non-empty\")\n\n        # K-fold CV\n",
    str(path),
)
path.write_text(text)


# ---------------------------------------------------------------------------
# Minimum Covariance Determinant
# ---------------------------------------------------------------------------
path = Path("statgpu/covariance/_robust.py")
text = path.read_text()
text = replace_once(
    text,
    "        X_np = np.asarray(X, dtype=np.float64)\n",
    "        X_np = np.asarray(_to_numpy(X), dtype=np.float64)\n",
    str(path),
)
text = replace_once(
    text,
    "        # Determine h (support size) -- use ceil like sklearn\n        if self.support_fraction is not None:\n            h = int(np.ceil(self.support_fraction * n))\n",
    "        if self.support_fraction is not None:\n            fraction = float(self.support_fraction)\n            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:\n                raise ValueError(\"support_fraction must be finite and in (0, 1]\")\n\n        # Determine h (support size) -- use ceil like sklearn\n        if self.support_fraction is not None:\n            h = int(np.ceil(float(self.support_fraction) * n))\n",
    str(path),
)
text = replace_once(
    text,
    "        raw_location = X_sub.mean(axis=0)\n        raw_cov = (X_sub - raw_location).T @ (X_sub - raw_location) / float(h)\n",
    "        raw_location = np.zeros(p) if self.assume_centered else X_sub.mean(axis=0)\n        raw_centered = X_sub if self.assume_centered else X_sub - raw_location\n        raw_cov = raw_centered.T @ raw_centered / float(h)\n",
    str(path),
)
text = replace_once(
    text,
    "            final_location = X_support.mean(axis=0)\n            final_cov_emp = (X_support - final_location).T @ (X_support - final_location) / float(n_support)\n",
    "            final_location = np.zeros(p) if self.assume_centered else X_support.mean(axis=0)\n            final_centered = X_support if self.assume_centered else X_support - final_location\n            final_cov_emp = final_centered.T @ final_centered / float(n_support)\n",
    str(path),
)
text = replace_once(
    text,
    "    @staticmethod\n    def _c_step(X, subset, h, max_iter=30):\n",
    "    def _c_step(self, X, subset, h, max_iter=30):\n",
    str(path),
)
text = replace_once(
    text,
    "            loc = X_sub.mean(axis=0)\n            cov = (X_sub - loc).T @ (X_sub - loc) / float(h)\n",
    "            loc = np.zeros(X.shape[1]) if self.assume_centered else X_sub.mean(axis=0)\n            centered = X_sub if self.assume_centered else X_sub - loc\n            cov = centered.T @ centered / float(h)\n",
    str(path),
)
path.write_text(text)


# ---------------------------------------------------------------------------
# Panel covariance utilities
# ---------------------------------------------------------------------------
path = Path("statgpu/panel/_covariance.py")
text = path.read_text()
text = replace_once(
    text,
    "    X = xp_asarray(X, dtype=xp.float64, xp=xp)\n    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n    clusters = xp_asarray(clusters, xp=xp, ref_arr=X).ravel()\n\n    n, k = X.shape\n",
    "    # Factorize labels before moving them to a GPU backend, since CuPy and\n    # Torch cannot represent arbitrary string/categorical labels.\n    clusters_np = np.asarray(_to_numpy(clusters)).ravel()\n    X = xp_asarray(X, dtype=xp.float64, xp=xp)\n    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n\n    if X.ndim != 2:\n        raise ValueError(\"X must be two-dimensional\")\n    n, k = X.shape\n    if resid.shape[0] != n or clusters_np.shape[0] != n:\n        raise ValueError(\"X, resid, and clusters must have the same number of observations\")\n",
    str(path),
)
text = replace_once(
    text,
    "    # Factorize cluster labels to contiguous indices\n    clusters_np = _to_numpy(clusters)\n    unique_labels, cluster_idx = np.unique(clusters_np, return_inverse=True)\n",
    "    # Factorize cluster labels to contiguous indices\n    unique_labels, cluster_idx = np.unique(clusters_np, return_inverse=True)\n",
    str(path),
)
text = replace_once(
    text,
    "    elif hasattr(S, 'device') and not hasattr(S, 'get'):\n        # cupy — fall back to numpy loop\n        S_np = np.zeros((n_clusters, k), dtype=np.float64)\n        np.add.at(S_np, cluster_idx, _to_numpy(scores))\n        S = xp_asarray(S_np, dtype=xp.float64, xp=xp, ref_arr=X)\n    else:\n        # numpy\n        np.add.at(S, cluster_idx, scores)\n",
    "    elif type(S).__module__.startswith('cupy'):\n        xp.add.at(S, cluster_idx_xp, scores)\n    else:\n        np.add.at(S, cluster_idx, scores)\n",
    str(path),
)
text = replace_once(
    text,
    "    c1_raw = _to_numpy(xp_asarray(cluster1, xp=xp, ref_arr=V1).ravel())\n    c2_raw = _to_numpy(xp_asarray(cluster2, xp=xp, ref_arr=V1).ravel())\n",
    "    c1_raw = np.asarray(_to_numpy(cluster1)).ravel()\n    c2_raw = np.asarray(_to_numpy(cluster2)).ravel()\n    n = int(np.asarray(_to_numpy(X)).shape[0])\n    if c1_raw.shape[0] != n or c2_raw.shape[0] != n:\n        raise ValueError(\"cluster arrays must match the number of observations\")\n",
    str(path),
)
text = replace_once(
    text,
    "    xp = _ensure_xp(xp)\n\n    X = xp_asarray(X, dtype=xp.float64, xp=xp)\n",
    "    xp = _ensure_xp(xp)\n    if str(kernel).lower() != \"bartlett\":\n        raise ValueError(\"kernel must be 'bartlett'\")\n    if bandwidth is not None:\n        if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, np.integer)):\n            raise ValueError(\"bandwidth must be a non-negative integer or None\")\n        if int(bandwidth) < 0:\n            raise ValueError(\"bandwidth must be a non-negative integer or None\")\n\n    X = xp_asarray(X, dtype=xp.float64, xp=xp)\n",
    str(path),
)
path.write_text(text)


# ---------------------------------------------------------------------------
# Panel formula alignment helper
# ---------------------------------------------------------------------------
path = Path("statgpu/panel/_formula.py")
text = path.read_text()
text = replace_once(
    text,
    "    y_arr, X_arr, design_info = parser.eval(data)\n\n    formula_column_names = list(design_info.column_names)\n",
    "    y_arr, X_arr, design_info = parser.eval(data)\n    setattr(design_info, \"_statgpu_row_positions\", np.asarray(parser._row_positions, dtype=np.int64))\n\n    formula_column_names = list(design_info.column_names)\n",
    str(path),
)
text = replace_once(
    text,
    "    parser = FormulaParser(formula)\n    return parser.eval(data)\n",
    "    parser = FormulaParser(formula)\n    y_arr, X_arr, design_info = parser.eval(data)\n    setattr(design_info, \"_statgpu_row_positions\", np.asarray(parser._row_positions, dtype=np.int64))\n    return y_arr, X_arr, design_info\n",
    str(path),
)
helper = '''def _align_formula_side_array(values, design_info, expected_n=None, name="array"):
    """Align an observation-level side array with rows retained by Patsy."""
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be observation-level")
    positions = getattr(design_info, "_statgpu_row_positions", None)
    if positions is None:
        if expected_n is not None and arr.shape[0] != expected_n:
            raise ValueError(f"{name} must have {expected_n} observations")
        return arr
    positions = np.asarray(positions, dtype=np.int64)
    if arr.shape[0] == positions.shape[0]:
        return arr
    if positions.size and arr.shape[0] > int(positions.max()):
        return arr[positions]
    if positions.size == 0 and arr.shape[0] == 0:
        return arr
    raise ValueError(
        f"{name} has {arr.shape[0]} observations and cannot be aligned to "
        f"the {positions.shape[0]} rows retained by the formula"
    )


'''
text = replace_once(text, "def _formula_predict(X, design_info, formula_has_intercept, model_has_intercept):\n", helper + "def _formula_predict(X, design_info, formula_has_intercept, model_has_intercept):\n", str(path))
# Align IDs extracted from pipe/token formulas.
text = replace_once(
    text,
    "            if time_effects and time_ids is None and hasattr(data, 'columns'):\n                if 'time' in data.columns:\n                    time_ids = data['time'].values\n",
    "            if time_effects and time_ids is None and hasattr(data, 'columns'):\n                if 'time' in data.columns:\n                    time_ids = data['time'].values\n            entity_ids = _align_formula_side_array(entity_ids, design_info, len(y_arr), \"entity_ids\")\n            time_ids = _align_formula_side_array(time_ids, design_info, len(y_arr), \"time_ids\")\n",
    str(path),
)
path.write_text(text)


# ---------------------------------------------------------------------------
# Panel model call sites and stable rank-deficient fallbacks
# ---------------------------------------------------------------------------
# PooledOLS
path = Path("statgpu/panel/_pooled.py")
text = path.read_text()
text = replace_once(
    text,
    "        from statgpu.panel._formula import _prepare_formula_fit, _get_feature_names\n",
    "        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit\n",
    str(path),
)
text = replace_once(
    text,
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)\n\n        backend = self._get_backend(backend=\"auto\")\n",
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)\n        if formula is not None:\n            cluster = _align_formula_side_array(cluster, self._design_info, len(y_arr), \"cluster\")\n            time_index = _align_formula_side_array(time_index, self._design_info, len(y_arr), \"time_index\")\n\n        backend = self._get_backend(backend=\"auto\")\n",
    str(path),
)
text = replace_once(
    text,
    "        except _LINALG_ERRORS:\n            params = xp.linalg.lstsq(XtX, Xty)[0]\n\n        resid = y_arr - X_arr @ params\n        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)\n",
    "        except _LINALG_ERRORS:\n            params = xp.linalg.pinv(X_arr) @ y_arr\n\n        if n <= k:\n            raise ValueError(f\"positive residual degrees of freedom required; n={n}, k={k}\")\n        resid = y_arr - X_arr @ params\n        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)\n",
    str(path),
)
path.write_text(text)

# BetweenOLS
path = Path("statgpu/panel/_between.py")
text = path.read_text()
text = replace_once(
    text,
    "        from statgpu.panel._formula import _prepare_formula_fit\n",
    "        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit\n",
    str(path),
)
text = replace_once(
    text,
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)\n\n        backend = self._get_backend(backend=\"auto\")\n",
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)\n        if formula is not None:\n            entity_ids = _align_formula_side_array(entity_ids, self._design_info, len(y_arr), \"entity_ids\")\n\n        backend = self._get_backend(backend=\"auto\")\n",
    str(path),
)
text = replace_once(
    text,
    "        except _LINALG_ERRORS:\n            params = xp.linalg.lstsq(XtX, Xty)[0]\n\n        resid = y_mean - X_mean @ params\n        n = n_groups\n        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)\n",
    "        except _LINALG_ERRORS:\n            params = xp.linalg.pinv(X_mean) @ y_mean\n\n        resid = y_mean - X_mean @ params\n        n = n_groups\n        if n <= k:\n            raise ValueError(f\"positive residual degrees of freedom required; groups={n}, parameters={k}\")\n        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)\n",
    str(path),
)
path.write_text(text)

# FirstDifferenceOLS
path = Path("statgpu/panel/_first_diff.py")
text = path.read_text()
text = replace_once(
    text,
    "        from statgpu.panel._formula import _prepare_formula_fit\n",
    "        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit\n",
    str(path),
)
text = replace_once(
    text,
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=False)\n\n        backend = self._get_backend(backend=\"auto\")\n",
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=False)\n        if formula is not None:\n            entity_ids = _align_formula_side_array(entity_ids, self._design_info, len(y_arr), \"entity_ids\")\n            time_ids = _align_formula_side_array(time_ids, self._design_info, len(y_arr), \"time_ids\")\n\n        backend = self._get_backend(backend=\"auto\")\n",
    str(path),
)
text = replace_once(
    text,
    "        except _LINALG_ERRORS:\n            params = xp.linalg.lstsq(XtX, Xty)[0]\n\n        resid = y_diff - X_diff @ params\n        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)\n",
    "        except _LINALG_ERRORS:\n            params = xp.linalg.pinv(X_diff) @ y_diff\n\n        if n <= k:\n            raise ValueError(f\"positive residual degrees of freedom required; n={n}, k={k}\")\n        resid = y_diff - X_diff @ params\n        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)\n",
    str(path),
)
text = replace_once(
    text,
    "        xp.asarray(X_diff_np, dtype=xp.float64),\n        xp.asarray(y_diff_np, dtype=xp.float64),\n",
    "        xp_asarray(X_diff_np, dtype=xp.float64, xp=xp, ref_arr=X),\n        xp_asarray(y_diff_np, dtype=xp.float64, xp=xp, ref_arr=X),\n",
    str(path),
)
path.write_text(text)

# FamaMacBeth
path = Path("statgpu/panel/_fama_macbeth.py")
text = path.read_text()
text = replace_once(
    text,
    "        from statgpu.panel._formula import _prepare_formula_fit\n",
    "        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit\n",
    str(path),
)
text = replace_once(
    text,
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)\n\n        backend = self._get_backend(backend=\"auto\")\n        y_np = np.asarray(y_np, dtype=np.float64).ravel()\n        tids_np = np.asarray(time_ids).ravel()\n",
    "            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)\n        if formula is not None:\n            time_ids = _align_formula_side_array(time_ids, self._design_info, len(y_np), \"time_ids\")\n\n        backend = self._get_backend(backend=\"auto\")\n        X_np = np.asarray(_to_numpy(X_np), dtype=np.float64)\n        y_np = np.asarray(_to_numpy(y_np), dtype=np.float64).ravel()\n        tids_np = np.asarray(_to_numpy(time_ids)).ravel()\n",
    str(path),
)
text = replace_once(
    text,
    "            except np.linalg.LinAlgError:\n                beta_t = np.linalg.lstsq(X_t.T @ X_t, X_t.T @ y_t, rcond=None)[0]\n",
    "            except np.linalg.LinAlgError:\n                beta_t = np.linalg.pinv(X_t) @ y_t\n",
    str(path),
)
text = replace_once(
    text,
    "        T = betas.shape[0]\n\n        # Step 2: Time-series averages and SEs\n",
    "        T = betas.shape[0]\n        if T < 2:\n            raise ValueError(\"FamaMacBeth requires at least 2 time periods after filtering\")\n\n        # Step 2: Time-series averages and SEs\n",
    str(path),
)
path.write_text(text)

# PanelOLS fixed-effects side-array alignment and stable fallback.
path = Path("statgpu/panel/_fixed_effects.py")
text = path.read_text()
text = replace_once(
    text,
    "            from statgpu.panel._formula import _prepare_formula_fit\n",
    "            from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit\n",
    str(path),
)
text = replace_once(
    text,
    "            X = X_raw\n            y = y_raw\n",
    "            X = X_raw\n            y = y_raw\n            entity_ids = _align_formula_side_array(entity_ids, self._design_info, len(y_raw), \"entity_ids\")\n            time_ids = _align_formula_side_array(time_ids, self._design_info, len(y_raw), \"time_ids\")\n            cluster = _align_formula_side_array(cluster, self._design_info, len(y_raw), \"cluster\")\n",
    str(path),
)
# Replace only the final normal-equation fallback if present.
text = text.replace(
    "        except _LINALG_ERRORS:\n            params = xp.linalg.solve(XtX, Xty)\n",
    "        except _LINALG_ERRORS:\n            params = xp.linalg.pinv(X_tilde) @ y_tilde\n",
    1,
)
path.write_text(text)

print("Covariance and panel review patch applied")
