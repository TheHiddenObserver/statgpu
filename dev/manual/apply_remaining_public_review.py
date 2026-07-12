"""Apply remaining public API review fixes for smoothing, splines, GAM, metrics."""

from pathlib import Path


def replace_once(text, old, new, path):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


# B-spline validation -------------------------------------------------------
path = Path("statgpu/nonparametric/splines/_bspline_basis.py")
text = path.read_text()
text = replace_once(
    text,
    "    x = xp.asarray(x, dtype=xp.float64).ravel()\n    knots = xp.asarray(knots, dtype=xp.float64).ravel()\n    n = x.shape[0]\n    m = knots.shape[0]\n\n    if m == 0:\n        raise ValueError(\"At least one interior knot is required\")\n",
    "    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):\n        raise ValueError(\"degree must be an integer\")\n    if int(degree) < 0:\n        raise ValueError(\"degree must be non-negative\")\n    degree = int(degree)\n\n    x = xp_asarray(x, dtype=xp.float64, xp=xp).ravel()\n    knots = xp_asarray(knots, dtype=xp.float64, xp=xp, ref_arr=x).ravel()\n    n = x.shape[0]\n    m = knots.shape[0]\n\n    if n == 0:\n        raise ValueError(\"x must contain at least one value\")\n    if m == 0:\n        raise ValueError(\"At least one interior knot is required\")\n    if not bool(np.asarray(_to_numpy(xp.all(xp.isfinite(x)))).item()):\n        raise ValueError(\"x and knots must contain only finite values\")\n    if not bool(np.asarray(_to_numpy(xp.all(xp.isfinite(knots)))).item()):\n        raise ValueError(\"x and knots must contain only finite values\")\n    knots_np = np.asarray(_to_numpy(knots), dtype=np.float64)\n    if np.any(np.diff(knots_np) <= 0):\n        raise ValueError(\"interior knots must be strictly increasing\")\n",
    str(path),
)
text = replace_once(
    text,
    "    # Ensure interior knots are strictly within boundary\n    if knot_min <= boundary_lo or knot_max >= boundary_hi:\n",
    "    if not np.isfinite(boundary_lo) or not np.isfinite(boundary_hi) or boundary_lo >= boundary_hi:\n        raise ValueError(\"boundary_lo and boundary_hi must be finite with boundary_lo < boundary_hi\")\n\n    # Ensure interior knots are strictly within boundary\n    if knot_min <= boundary_lo or knot_max >= boundary_hi:\n",
    str(path),
)
path.write_text(text)


# Shared KDE/kernel-regression input contracts -----------------------------
path = Path("statgpu/nonparametric/kernel_smoothing/_kernel_common.py")
text = path.read_text()
text = replace_once(
    text,
    "    n_samples = int(arr.shape[0])\n    if n_samples < 2:\n        raise ValueError(\"samples must contain at least 2 observations\")\n    return arr\n",
    "    n_samples = int(arr.shape[0])\n    if n_samples < 2:\n        raise ValueError(\"samples must contain at least 2 observations\")\n    if not bool(_to_float_scalar(xp.all(xp.isfinite(arr)))):\n        raise ValueError(\"samples must contain only finite values\")\n    return arr\n",
    str(path),
)
text = replace_once(
    text,
    "    if int(arr.shape[1]) != int(n_features):\n        raise ValueError(\"points feature dimension does not match samples\")\n    return arr\n",
    "    if int(arr.shape[1]) != int(n_features):\n        raise ValueError(\"points feature dimension does not match samples\")\n    if not bool(_to_float_scalar(xp.all(xp.isfinite(arr)))):\n        raise ValueError(\"points must contain only finite values\")\n    return arr\n",
    str(path),
)
text = replace_once(
    text,
    "    if _to_float_scalar(xp.min(w)) < 0.0:\n        raise ValueError(\"weights must be non-negative\")\n\n    w_sum = xp.sum(w)\n",
    "    if not bool(_to_float_scalar(xp.all(xp.isfinite(w)))):\n        raise ValueError(\"weights must contain only finite values\")\n    if _to_float_scalar(xp.min(w)) < 0.0:\n        raise ValueError(\"weights must be non-negative\")\n\n    w_sum = xp.sum(w)\n",
    str(path),
)
path.write_text(text)


# GAM validation and 1D prediction -----------------------------------------
path = Path("statgpu/semiparametric/_gam.py")
text = path.read_text()
text = replace_once(
    text,
    "        xp = self._get_xp()\n\n        # Convert to arrays on the correct device\n",
    "        if isinstance(self.n_splines, bool) or not isinstance(self.n_splines, (int, np.integer)):\n            raise ValueError(\"n_splines must be an integer\")\n        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)):\n            raise ValueError(\"degree must be an integer\")\n        if int(self.degree) < 0:\n            raise ValueError(\"degree must be non-negative\")\n        if int(self.n_splines) <= int(self.degree) + 1:\n            raise ValueError(\"n_splines must be greater than degree + 1\")\n        if isinstance(self.penalty_order, bool) or not isinstance(self.penalty_order, (int, np.integer)) or int(self.penalty_order) < 1:\n            raise ValueError(\"penalty_order must be a positive integer\")\n        if self.lam is not None and (not np.isfinite(float(self.lam)) or float(self.lam) < 0):\n            raise ValueError(\"lam must be finite and non-negative or None\")\n        if str(self.knot_method).lower() not in {\"uniform\", \"quantile\"}:\n            raise ValueError(\"knot_method must be 'uniform' or 'quantile'\")\n        if not np.isfinite(float(self.gamma)) or float(self.gamma) <= 0:\n            raise ValueError(\"gamma must be finite and positive\")\n\n        xp = self._get_xp()\n\n        # Convert to arrays on the correct device\n",
    str(path),
)
text = replace_once(
    text,
    "        X = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=_ref)\n        y = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n\n        n, p = X.shape\n",
    "        X = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=_ref)\n        if X.ndim == 1:\n            X = X.reshape(-1, 1)\n        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:\n            raise ValueError(\"X must be a non-empty one- or two-dimensional array\")\n        y = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n        if int(y.shape[0]) != int(X.shape[0]):\n            raise ValueError(\"X and y must have the same number of observations\")\n        if not bool(float(xp.all(xp.isfinite(X)))) or not bool(float(xp.all(xp.isfinite(y)))):\n            raise ValueError(\"X and y must contain only finite values\")\n\n        n, p = X.shape\n        for j in range(p):\n            if float(xp.max(X[:, j]) - xp.min(X[:, j])) <= 0.0:\n                raise ValueError(f\"feature {j} is constant and cannot define a smooth term\")\n",
    str(path),
)
text = replace_once(
    text,
    "        X = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=self._xp_asarray_ref_)\n\n        n, p = X.shape\n",
    "        X = xp_asarray(X, dtype=xp.float64, xp=xp, ref_arr=self._xp_asarray_ref_)\n        if X.ndim == 1:\n            if self.n_features_ == 1:\n                X = X.reshape(-1, 1)\n            elif int(X.size) == self.n_features_:\n                X = X.reshape(1, -1)\n            else:\n                raise ValueError(\"X shape is incompatible with fitted feature count\")\n        if X.ndim != 2:\n            raise ValueError(\"X must be one- or two-dimensional\")\n        if not bool(float(xp.all(xp.isfinite(X)))):\n            raise ValueError(\"X must contain only finite values\")\n\n        n, p = X.shape\n",
    str(path),
)
path.write_text(text)


# Binary metrics threshold validation --------------------------------------
path = Path("statgpu/metrics/_classification.py")
text = path.read_text()
text = replace_once(
    text,
    "    if threshold < 0.0 or threshold > 1.0:\n        raise ValueError(\"threshold must be in [0, 1]\")\n",
    "    if not np.isfinite(float(threshold)) or threshold < 0.0 or threshold > 1.0:\n        raise ValueError(\"threshold must be finite and in [0, 1]\")\n",
    str(path),
)
path.write_text(text)

print("Remaining public module review patch applied")
