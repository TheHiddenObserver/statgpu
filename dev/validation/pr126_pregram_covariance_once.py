from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:120]!r}")
    text = text.replace(old, new, 1)
    p.write_text(text)


cov = "statgpu/panel/_covariance.py"
anchor = '''def _symmetrize(matrix):
'''
helpers = '''def _column_working_values(values, xp):
    """Return per-coordinate unit values with restore scales bounded below by one.

    Only coordinates whose absolute magnitude exceeds one are normalized.  This
    keeps ordinary/tiny coordinates on their original scale while ensuring that
    Gram and lag products are formed from values with absolute magnitude at most
    one.  The restore scales are therefore all >= 1, so restoring a finite final
    covariance cannot overflow before that final value is reached.
    """
    if _is_torch(xp):
        max_abs = xp.max(xp.abs(values), dim=0).values
    else:
        max_abs = xp.max(xp.abs(values), axis=0)
    scale = xp.maximum(max_abs, xp.ones_like(max_abs))
    return values / scale, scale


def _restore_coordinate_covariance(covariance, scale, xp):
    """Restore a per-coordinate covariance scale without forming scale_i*scale_j."""
    row = scale[:, None]
    col = scale[None, :]
    large = xp.maximum(row, col)
    small = xp.minimum(row, col)
    return _symmetrize((covariance * large) * small)


def _cluster_component_from_scores(
    scores, codes, *, n_groups: int, nobs: int, group_debias: bool, xp
):
    """Return one cluster meat on an already-safe common score scale."""
    if int(n_groups) < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(scores, codes, n_groups=int(n_groups), xp=xp)
    correction = (
        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0
    )
    meat = _symmetrize(grouped.T @ grouped * float(correction))
    return meat, float(correction)


'''
replace_once(cov, anchor, helpers + anchor)

old_cluster = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    n_clusters = int(len(labels))
    if n_clusters < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(
        influence, cluster_idx, n_groups=n_clusters, xp=xp
    )
    correction = 1.0
    if group_debias:
        correction = _group_debias_factor(n_clusters, int(n))
    cov = _symmetrize(grouped.T @ grouped * float(correction))
'''
new_cluster = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    n_clusters = int(len(labels))
    cov_work, correction = _cluster_component_from_scores(
        influence_work,
        cluster_idx,
        n_groups=n_clusters,
        nobs=int(n),
        group_debias=group_debias,
        xp=xp,
    )
    cov = _restore_coordinate_covariance(cov_work, influence_scale, xp)
'''
replace_once(cov, old_cluster, new_cluster)

old_two = '''    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)
    n = int(X.shape[0])
    labels1, c1 = _factorize_1d_labels(cluster1, nobs=n, name="cluster1")
    labels2, c2 = _factorize_1d_labels(cluster2, nobs=n, name="cluster2")
    c12 = _paired_codes(c1, c2)
    n12 = int(np.max(c12)) + 1 if c12.size else 0

    meta1: dict = {}
    meta2: dict = {}
    meta12: dict = {}
    V1 = clustered_covariance(
        X,
        resid,
        c1,
        xp,
        group_debias=group_debias,
        metadata=meta1,
    )
    V2 = clustered_covariance(
        X,
        resid,
        c2,
        xp,
        group_debias=group_debias,
        metadata=meta2,
    )
    V12 = clustered_covariance(
        X,
        resid,
        c12,
        xp,
        group_debias=group_debias,
        metadata=meta12,
    )
    if metadata is not None:
        metadata.update(
            {
                "cluster_dimensions": 2,
                "cluster_group_counts": [
                    int(len(labels1)),
                    int(len(labels2)),
                    n12,
                ],
                "group_debias": bool(group_debias),
                "group_debias_factors": [
                    float(meta1["group_debias_factors"][0]),
                    float(meta2["group_debias_factors"][0]),
                    float(meta12["group_debias_factors"][0]),
                ],
            }
        )
    return _symmetrize(_stable_inclusion_exclusion(V1, V2, V12, xp))
'''
new_two = '''    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)
    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])
    labels1, c1 = _factorize_1d_labels(cluster1, nobs=n, name="cluster1")
    labels2, c2 = _factorize_1d_labels(cluster2, nobs=n, name="cluster2")
    c12 = _paired_codes(c1, c2)
    n12 = int(np.max(c12)) + 1 if c12.size else 0

    # All three Cameron-Gelbach-Miller components must be combined on one
    # common finite score scale. Restoring each component first can produce
    # Inf - Inf even when the inclusion-exclusion result itself is finite.
    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    V1_work, correction1 = _cluster_component_from_scores(
        influence_work,
        c1,
        n_groups=int(len(labels1)),
        nobs=n,
        group_debias=group_debias,
        xp=xp,
    )
    V2_work, correction2 = _cluster_component_from_scores(
        influence_work,
        c2,
        n_groups=int(len(labels2)),
        nobs=n,
        group_debias=group_debias,
        xp=xp,
    )
    V12_work, correction12 = _cluster_component_from_scores(
        influence_work,
        c12,
        n_groups=n12,
        nobs=n,
        group_debias=group_debias,
        xp=xp,
    )
    cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)
    cov = _restore_coordinate_covariance(cov_work, influence_scale, xp)
    if metadata is not None:
        metadata.update(
            {
                "cluster_dimensions": 2,
                "cluster_group_counts": [
                    int(len(labels1)),
                    int(len(labels2)),
                    n12,
                ],
                "group_debias": bool(group_debias),
                "group_debias_factors": [
                    correction1,
                    correction2,
                    correction12,
                ],
            }
        )
    return cov
'''
replace_once(cov, old_two, new_two)

old_hac_start = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    max_terms = int(bandwidth) + 1
    cov, scaled = _covariance_accumulator_start(
        _symmetrize(influence.T @ influence), max_terms=max_terms, xp=xp
    )
    for h in range(1, bandwidth + 1):
        w = 1.0 - h / (bandwidth + 1.0)
        gamma_h = influence[h:].T @ influence[: n - h]
'''
new_hac_start = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    max_terms = int(bandwidth) + 1
    cov, scaled = _covariance_accumulator_start(
        _symmetrize(influence_work.T @ influence_work), max_terms=max_terms, xp=xp
    )
    for h in range(1, bandwidth + 1):
        w = 1.0 - h / (bandwidth + 1.0)
        gamma_h = influence_work[h:].T @ influence_work[: n - h]
'''
replace_once(cov, old_hac_start, new_hac_start)
old_hac_return = '''    cov = _covariance_accumulator_finish(
        cov, scaled, max_terms=max_terms, xp=xp
    )
    return _symmetrize(cov)
'''
new_hac_return = '''    cov = _covariance_accumulator_finish(
        cov, scaled, max_terms=max_terms, xp=xp
    )
    return _restore_coordinate_covariance(cov, influence_scale, xp)
'''
replace_once(cov, old_hac_return, new_hac_return)

old_dk_influence = '''    influence, _X_pinv, _bread, rank = _influence_rows(X, resid, xp)
    k_columns = int(X.shape[1])
'''
new_dk_influence = '''    influence, _X_pinv, _bread, rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    k_columns = int(X.shape[1])
'''
replace_once(cov, old_dk_influence, new_dk_influence)
old_dk_group = '''    grouped = _grouped_score_sums(
        influence, time_codes, n_groups=n_periods, xp=xp
    )
'''
new_dk_group = '''    grouped = _grouped_score_sums(
        influence_work, time_codes, n_groups=n_periods, xp=xp
    )
    grouped_work, grouped_scale = _column_working_values(grouped, xp)
'''
replace_once(cov, old_dk_group, new_dk_group)
old_dk_ref = '''        ref_arr=grouped,
    )

    max_terms = 1 + int(np.count_nonzero(weights_np[1:]))
    cov, scaled = _covariance_accumulator_start(
        _symmetrize(grouped.T @ grouped), max_terms=max_terms, xp=xp
    )
'''
new_dk_ref = '''        ref_arr=grouped_work,
    )

    max_terms = 1 + int(np.count_nonzero(weights_np[1:]))
    cov, scaled = _covariance_accumulator_start(
        _symmetrize(grouped_work.T @ grouped_work), max_terms=max_terms, xp=xp
    )
'''
replace_once(cov, old_dk_ref, new_dk_ref)
old_dk_gamma = '''        gamma = grouped[lag:].T @ grouped[: n_periods - lag]
'''
new_dk_gamma = '''        gamma = grouped_work[lag:].T @ grouped_work[: n_periods - lag]
'''
replace_once(cov, old_dk_gamma, new_dk_gamma)
old_dk_finish = '''    scale = float(n) / float(denom)
    cov = _symmetrize(scale * cov)
'''
new_dk_finish = '''    cov = _restore_coordinate_covariance(cov, grouped_scale, xp)
    cov = _restore_coordinate_covariance(cov, influence_scale, xp)
    scale = float(n) / float(denom)
    cov = _symmetrize(scale * cov)
'''
replace_once(cov, old_dk_finish, new_dk_finish)

# NumPy regressions.
test_cov = Path("dev/tests/test_panel_stage_c_covariance.py")
text = test_cov.read_text()
append = r'''


def test_hac_and_dk_normalize_scores_before_zero_lag_gram_overflow():
    n = 10
    influence_sq = 1.0e308
    influence_amp = float(np.sqrt(influence_sq))
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    X = np.ones((n, 1), dtype=np.float64)
    resid = n * influence_amp * signs
    time = np.arange(n, dtype=np.int64)

    # The zero-lag meat is n*a^2 and overflows, but Bartlett bandwidth=1
    # subtracts (n-1)*a^2 through the lag pair, leaving exactly a^2.
    expected_hac = np.asarray([[influence_sq]], dtype=np.float64)
    expected_dk = expected_hac * (n / (n - 1.0))
    hac = hac_covariance(X, resid, bandwidth=1)
    dk = driscoll_kraay_covariance(X, resid, time, bandwidth=1)
    assert np.all(np.isfinite(hac))
    assert np.all(np.isfinite(dk))
    np.testing.assert_allclose(hac, expected_hac, rtol=2.0e-14, atol=0.0)
    np.testing.assert_allclose(dk, expected_dk, rtol=2.5e-14, atol=0.0)


def test_two_way_cluster_combines_components_before_overflowing_restore():
    n = 4
    influence_sq = 5.0e307
    influence_amp = float(np.sqrt(influence_sq))
    X = np.ones((n, 1), dtype=np.float64)
    resid = n * influence_amp * np.asarray([1.0, -1.0, 1.0, -1.0])
    unique = np.arange(n, dtype=np.int64)
    pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)

    # The unique-cluster component and its intersection component are each
    # 4*a^2 > DBL_MAX, but they are algebraically identical and cancel.  The
    # final two-way covariance is therefore the finite coarse-cluster component.
    reference = clustered_covariance(X, resid, pairs)
    actual = two_way_clustered_covariance(X, resid, unique, pairs)
    assert np.all(np.isfinite(reference))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, reference, rtol=3.0e-14, atol=0.0)
'''
if "test_hac_and_dk_normalize_scores_before_zero_lag_gram_overflow" not in text:
    test_cov.write_text(text + append)

# Torch CPU regressions.
torch_test = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = torch_test.read_text()
append = r'''


def test_stage_c_torch_cpu_pregram_and_two_way_component_cancellation():
    n = 10
    influence_sq = 1.0e308
    influence_amp = float(np.sqrt(influence_sq))
    signs_np = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    X = torch.ones((n, 1), dtype=torch.float64)
    resid = torch.as_tensor(n * influence_amp * signs_np, dtype=torch.float64)
    time = np.arange(n, dtype=np.int64)
    expected_hac = np.asarray([[influence_sq]], dtype=np.float64)
    assert_allclose(
        hac_covariance(X, resid, bandwidth=1, xp=torch),
        expected_hac,
        rtol=2.0e-13,
        atol=0.0,
    )
    assert_allclose(
        driscoll_kraay_covariance(X, resid, time, bandwidth=1, xp=torch),
        expected_hac * (n / (n - 1.0)),
        rtol=2.5e-13,
        atol=0.0,
    )

    n2 = 4
    component_sq = 5.0e307
    component_amp = float(np.sqrt(component_sq))
    X2 = torch.ones((n2, 1), dtype=torch.float64)
    resid2 = torch.as_tensor(
        n2 * component_amp * np.asarray([1.0, -1.0, 1.0, -1.0]),
        dtype=torch.float64,
    )
    unique = np.arange(n2, dtype=np.int64)
    pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)
    reference = clustered_covariance(X2, resid2, pairs, xp=torch)
    actual = two_way_clustered_covariance(X2, resid2, unique, pairs, xp=torch)
    assert_allclose(actual, reference, rtol=3.0e-13, atol=0.0)
'''
if "test_stage_c_torch_cpu_pregram_and_two_way_component_cancellation" not in text:
    torch_test.write_text(text + append)

# Physical CuPy/Torch validator: add the same counterexamples to the existing
# extreme-scale audit so actual GPU acceptance cannot omit these paths.
bench = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = bench.read_text()
old = '''    lag_time = np.arange(lag_n, dtype=np.int64)\n\n    if backend == "numpy":\n'''
new = '''    lag_time = np.arange(lag_n, dtype=np.int64)\n\n    pregram_n = 10\n    pregram_sq = 1.0e308\n    pregram_amp = float(np.sqrt(pregram_sq))\n    pregram_signs = np.where(np.arange(pregram_n) % 2 == 0, 1.0, -1.0)\n    X_pregram_np = np.ones((pregram_n, 1), dtype=np.float64)\n    resid_pregram_np = pregram_n * pregram_amp * pregram_signs\n    pregram_time = np.arange(pregram_n, dtype=np.int64)\n\n    component_n = 4\n    component_sq = 5.0e307\n    component_amp = float(np.sqrt(component_sq))\n    X_component_np = np.ones((component_n, 1), dtype=np.float64)\n    resid_component_np = component_n * component_amp * np.asarray([1.0, -1.0, 1.0, -1.0])\n    component_unique = np.arange(component_n, dtype=np.int64)\n    component_pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
if old not in text:
    raise RuntimeError("physical validator pregram anchor missing")
text = text.replace(old, new, 1)
old = '''        X_lag, resid_lag = X_lag_np, resid_lag_np\n    elif backend == "cupy":\n'''
new = '''        X_lag, resid_lag = X_lag_np, resid_lag_np\n        X_pregram, resid_pregram = X_pregram_np, resid_pregram_np\n        X_component, resid_component = X_component_np, resid_component_np\n    elif backend == "cupy":\n'''
text = text.replace(old, new, 1)
old = '''        X_lag, resid_lag = cp.asarray(X_lag_np), cp.asarray(resid_lag_np)\n    elif backend == "torch":\n'''
new = '''        X_lag, resid_lag = cp.asarray(X_lag_np), cp.asarray(resid_lag_np)\n        X_pregram, resid_pregram = cp.asarray(X_pregram_np), cp.asarray(resid_pregram_np)\n        X_component, resid_component = cp.asarray(X_component_np), cp.asarray(resid_component_np)\n    elif backend == "torch":\n'''
text = text.replace(old, new, 1)
old = '''        X_lag = torch.as_tensor(X_lag_np, dtype=torch.float64, device="cuda")\n        resid_lag = torch.as_tensor(resid_lag_np, dtype=torch.float64, device="cuda")\n    else:\n'''
new = '''        X_lag = torch.as_tensor(X_lag_np, dtype=torch.float64, device="cuda")\n        resid_lag = torch.as_tensor(resid_lag_np, dtype=torch.float64, device="cuda")\n        X_pregram = torch.as_tensor(X_pregram_np, dtype=torch.float64, device="cuda")\n        resid_pregram = torch.as_tensor(resid_pregram_np, dtype=torch.float64, device="cuda")\n        X_component = torch.as_tensor(X_component_np, dtype=torch.float64, device="cuda")\n        resid_component = torch.as_tensor(resid_component_np, dtype=torch.float64, device="cuda")\n    else:\n'''
text = text.replace(old, new, 1)
old = '''    lag_dk = _array(\n        driscoll_kraay_covariance(\n            X_lag, resid_lag, lag_time, bandwidth=lag_bandwidth, kernel="bartlett", xp=xp\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk)):\n'''
new = '''    lag_dk = _array(\n        driscoll_kraay_covariance(\n            X_lag, resid_lag, lag_time, bandwidth=lag_bandwidth, kernel="bartlett", xp=xp\n        )\n    )\n    pregram_hac = _array(hac_covariance(X_pregram, resid_pregram, bandwidth=1, xp=xp))\n    pregram_dk = _array(\n        driscoll_kraay_covariance(\n            X_pregram, resid_pregram, pregram_time, bandwidth=1, xp=xp\n        )\n    )\n    component_reference = _array(\n        clustered_covariance(X_component, resid_component, component_pairs, xp=xp)\n    )\n    component_two_way = _array(\n        two_way_clustered_covariance(\n            X_component,\n            resid_component,\n            component_unique,\n            component_pairs,\n            xp=xp,\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way)):\n'''
if old not in text:
    raise RuntimeError("physical validator computation anchor missing")
text = text.replace(old, new, 1)
old = '''    np.testing.assert_allclose(\n        lag_dk,\n        expected_lag_hac * (lag_n / (lag_n - 1.0)),\n        rtol=8e-13,\n        atol=0.0,\n    )\n    return {\n'''
new = '''    np.testing.assert_allclose(\n        lag_dk,\n        expected_lag_hac * (lag_n / (lag_n - 1.0)),\n        rtol=8e-13,\n        atol=0.0,\n    )\n    expected_pregram_hac = np.asarray([[pregram_sq]], dtype=np.float64)\n    np.testing.assert_allclose(pregram_hac, expected_pregram_hac, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(\n        pregram_dk,\n        expected_pregram_hac * (pregram_n / (pregram_n - 1.0)),\n        rtol=8e-13,\n        atol=0.0,\n    )\n    np.testing.assert_allclose(\n        component_two_way, component_reference, rtol=8e-13, atol=0.0\n    )\n    return {\n'''
text = text.replace(old, new, 1)
old = '''        "lag_accumulator_driscoll_kraay": lag_dk.tolist(),\n    }\n'''
new = '''        "lag_accumulator_driscoll_kraay": lag_dk.tolist(),\n        "pregram_hac": pregram_hac.tolist(),\n        "pregram_driscoll_kraay": pregram_dk.tolist(),\n        "two_way_component_cancellation": component_two_way.tolist(),\n    }\n'''
text = text.replace(old, new, 1)
bench.write_text(text)
