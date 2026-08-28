from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one anchor, found {count}")
    p.write_text(text.replace(old, new, 1))


# CuPy documents that cuSOLVER-backed linalg routines may return invalid
# results unless cupyx linalg error handling is enabled.  Make the existing
# rank/definiteness recovery paths observe those solver-status failures.
replace_once(
    "statgpu/backends/_gpu_inference_cupy.py",
    "    import cupy as cp\n\n    device_id = int(X_design.device.id)",
    "    import cupy as cp\n    import cupyx\n\n    device_id = int(X_design.device.id)",
)
replace_once(
    "statgpu/backends/_gpu_inference_cupy.py",
    '''    try:\n        # Use Cholesky for inversion\n        L = cp.linalg.cholesky(XtX)\n        with cp.cuda.Device(device_id):\n            identity = cp.eye(XtX.shape[0], dtype=XtX.dtype)\n        XtX_inv = cp.linalg.solve(L.T, cp.linalg.solve(L, identity))\n    except Exception as exc:\n        if not _linalg_exception_is_rank_failure(exc):\n            raise\n        # Pseudoinverse recovery is reserved for rank/definiteness failures.\n        XtX_inv = cp.linalg.pinv(XtX)\n''',
    '''    try:\n        # CuPy's cuSOLVER-backed Cholesky/solve can otherwise return NaNs\n        # for singular inputs instead of raising; request solver-status errors\n        # so the existing rank/definiteness recovery remains effective.\n        with cupyx.errstate(linalg="raise"):\n            L = cp.linalg.cholesky(XtX)\n            with cp.cuda.Device(device_id):\n                identity = cp.eye(XtX.shape[0], dtype=XtX.dtype)\n            XtX_inv = cp.linalg.solve(L.T, cp.linalg.solve(L, identity))\n    except Exception as exc:\n        if not _linalg_exception_is_rank_failure(exc):\n            raise\n        # Pseudoinverse recovery is reserved for rank/definiteness failures.\n        XtX_inv = cp.linalg.pinv(XtX)\n''',
)

replace_once(
    "statgpu/linear_model/wrappers/_linear.py",
    "        import cupy as cp\n        from cupyx.scipy.linalg import solve_triangular",
    "        import cupy as cp\n        import cupyx\n        from cupyx.scipy.linalg import solve_triangular",
)
replace_once(
    "statgpu/linear_model/wrappers/_linear.py",
    '''        try:\n            L = cp.linalg.cholesky(XtX)\n            tmp = solve_triangular(L, Xty, lower=True)\n            coef = solve_triangular(L.T, tmp, lower=False)\n            self.rank_ = n_design_cols\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n            lstsq_result = cp.linalg.lstsq(X_design, y, rcond=None)\n            coef = lstsq_result[0]\n            self.rank_ = int(lstsq_result[2]) if len(lstsq_result) > 2 else n_design_cols\n''',
    '''        try:\n            with cupyx.errstate(linalg="raise"):\n                L = cp.linalg.cholesky(XtX)\n                tmp = solve_triangular(L, Xty, lower=True)\n                coef = solve_triangular(L.T, tmp, lower=False)\n            self.rank_ = n_design_cols\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n            # Rank-deficient OLS is intentionally recovered with least squares.\n            with cupyx.errstate(linalg="raise"):\n                lstsq_result = cp.linalg.lstsq(X_design, y, rcond=None)\n            coef = lstsq_result[0]\n            self.rank_ = int(lstsq_result[2]) if len(lstsq_result) > 2 else n_design_cols\n''',
)

replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "        import cupy as cp\n        from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular",
    "        import cupy as cp\n        import cupyx\n        from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular",
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        try:\n            # Cholesky + triangular solve is faster than general solve\n            # for positive-definite matrices (Ridge penalty guarantees PD).\n            L = cp.linalg.cholesky(A)\n            tmp = cp_solve_triangular(L, Xty, lower=True)\n            return cp_solve_triangular(L.T, tmp, lower=False)\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n        try:\n            return cp.linalg.solve(A, Xty)\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n            return cp.linalg.pinv(A) @ Xty\n''',
    '''        try:\n            # Cholesky + triangular solve is faster than general solve\n            # for positive-definite matrices (Ridge penalty guarantees PD).\n            with cupyx.errstate(linalg="raise"):\n                L = cp.linalg.cholesky(A)\n                tmp = cp_solve_triangular(L, Xty, lower=True)\n                return cp_solve_triangular(L.T, tmp, lower=False)\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n        try:\n            with cupyx.errstate(linalg="raise"):\n                return cp.linalg.solve(A, Xty)\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n            return cp.linalg.pinv(A) @ Xty\n''',
)

replace_once(
    "statgpu/linear_model/penalized/_inference_mixin.py",
    "        import cupy as cp\n        from statgpu.inference._distributions_backend import t",
    "        import cupy as cp\n        import cupyx\n        from statgpu.inference._distributions_backend import t",
)
replace_once(
    "statgpu/linear_model/penalized/_inference_mixin.py",
    '''        try:\n            chol = cp.linalg.cholesky(bread)\n            with cp.cuda.Device(device_id):\n                identity = cp.eye(bread.shape[0], dtype=bread.dtype)\n            bread_inv = cp.linalg.solve(chol.T, cp.linalg.solve(chol, identity))\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n            bread_inv = cp.linalg.pinv(bread)\n''',
    '''        try:\n            with cupyx.errstate(linalg="raise"):\n                chol = cp.linalg.cholesky(bread)\n                with cp.cuda.Device(device_id):\n                    identity = cp.eye(bread.shape[0], dtype=bread.dtype)\n                bread_inv = cp.linalg.solve(chol.T, cp.linalg.solve(chol, identity))\n        except Exception as exc:\n            if not _linalg_exception_is_rank_failure(exc):\n                raise\n            bread_inv = cp.linalg.pinv(bread)\n''',
)

# Strengthen the existing physical functional case without changing the case
# count/schema: it now covers the public rank-deficient LinearRegression fit
# path as well as the shared inference primitive.
validator = "dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py"
replace_once(
    validator,
    '''    rank_errors = {\n        "bse": _max_error(result.bse, rank_ref.bse),\n        "statistic": _max_error(result.tvalues, rank_ref.tvalues),\n        "pvalue": _max_error(result.pvalues, rank_ref.pvalues),\n        "ci": _max_error(result.conf_int, rank_ref.conf_int),\n    }\n\n    # Multi-target numerical state stays native until the one reporting snapshot.\n''',
    '''    rank_errors = {\n        "bse": _max_error(result.bse, rank_ref.bse),\n        "statistic": _max_error(result.tvalues, rank_ref.tvalues),\n        "pvalue": _max_error(result.pvalues, rank_ref.pvalues),\n        "ci": _max_error(result.conf_int, rank_ref.conf_int),\n    }\n\n    # Exercise the public OLS Cholesky -> lstsq rank-recovery path too.\n    rank_X_np = np.column_stack([x, x])\n    rank_y_np = 0.5 + 0.5 * x + resid_np\n    rank_cpu = LinearRegression(\n        device="cpu", compute_inference=True, cov_type="nonrobust"\n    ).fit(rank_X_np, rank_y_np)\n    rank_X = _as_backend(rank_X_np, backend)\n    rank_y = _as_backend(rank_y_np, backend)\n    rank_gpu = LinearRegression(\n        device="cuda" if backend == "cupy" else "torch",\n        compute_inference=True,\n        cov_type="nonrobust",\n    ).fit(rank_X, rank_y)\n    if rank_gpu.rank_ >= rank_X_np.shape[1] + 1:\n        raise AssertionError(\n            f"{backend}: rank-deficient LinearRegression did not report reduced rank"\n        )\n    rank_fit_result = rank_gpu._inference_result\n    if rank_fit_result is None:\n        raise AssertionError("rank-deficient LinearRegression inference is missing")\n    if rank_fit_result.metadata.get("numerical_backend") != backend:\n        raise AssertionError(\n            f"rank-deficient LinearRegression backend mismatch: {rank_fit_result.metadata}"\n        )\n    if str(rank_fit_result.metadata.get("numerical_device", "")) != concrete_device:\n        raise AssertionError(\n            f"rank-deficient LinearRegression device mismatch: {rank_fit_result.metadata}"\n        )\n    rank_fit_errors = {\n        "prediction": _max_error(\n            _to_numpy(rank_gpu.predict(rank_X)), rank_cpu.predict(rank_X_np)\n        ),\n        "bse": _max_error(rank_gpu._bse, rank_cpu._bse),\n    }\n\n    # Multi-target numerical state stays native until the one reporting snapshot.\n''',
)
replace_once(
    validator,
    '''    for family, values in (("rank", rank_errors), ("multi_target", multi_errors)):\n''',
    '''    for family, values in (\n        ("rank", rank_errors),\n        ("rank_fit", rank_fit_errors),\n        ("multi_target", multi_errors),\n    ):\n''',
)
replace_once(
    validator,
    '''        "rank_errors": rank_errors,\n        "multi_target_errors": multi_errors,\n''',
    '''        "rank_errors": rank_errors,\n        "rank_fit_errors": rank_fit_errors,\n        "multi_target_errors": multi_errors,\n''',
)

# Hosted source-contract lock for all maintained CuPy linalg sites repaired by
# this round.  This deliberately does not claim physical behavior on CPU CI.
test_path = Path("dev/tests/test_gaussian_inference_gpu_runner_contract.py")
tests = test_path.read_text()
if "def test_cupy_cusolver_status_failures_are_enabled_for_rank_recovery" in tests:
    raise SystemExit("CuPy linalg regression test already exists")
tests += '''\n\n\ndef test_cupy_cusolver_status_failures_are_enabled_for_rank_recovery():\n    root = Path(__file__).parents[2]\n    sources = {\n        "gpu_inference": (root / "statgpu" / "backends" / "_gpu_inference_cupy.py").read_text(),\n        "linear": (root / "statgpu" / "linear_model" / "wrappers" / "_linear.py").read_text(),\n        "fit_mixin": (root / "statgpu" / "linear_model" / "penalized" / "_fit_mixin.py").read_text(),\n        "inference_mixin": (root / "statgpu" / "linear_model" / "penalized" / "_inference_mixin.py").read_text(),\n    }\n\n    for source in sources.values():\n        assert "import cupyx" in source\n        assert 'with cupyx.errstate(linalg="raise"):' in source\n    assert sources["fit_mixin"].count('with cupyx.errstate(linalg="raise"):') >= 2\n    runner_source = RUNNER.read_text()\n    assert '"rank_fit_errors": rank_fit_errors' in runner_source\n    assert "rank-deficient LinearRegression did not report reduced rank" in runner_source\n'''
test_path.write_text(tests)

print("PR129 CuPy linalg repair applied")
