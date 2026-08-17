from pathlib import Path

p = Path("dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    "This runner verifies correctness/backend provenance for chronology, formula,\nrank, no-intercept behavior, both covariance modes, and the standard inference\n",
    "This runner verifies correctness/backend provenance for chronology, formula,\nrank, exactly identified periods, no-intercept behavior, both covariance modes, and the standard inference\n",
    1,
)

anchor = '''def _rank_rejection(backend: str):
    X, y, time_ids = _rank_fixture()
    Xb, yb = _arrays(X, y, backend)
    try:
        FamaMacBeth(device=_device(backend)).fit(Xb, yb, time_ids=time_ids)
    except ValueError as exc:
        if "rank deficient" not in str(exc):
            raise
        return True
    raise AssertionError("rank-deficient retained period was not rejected")


'''
if anchor not in text:
    raise RuntimeError("rank rejection anchor not found")
addition = anchor + '''def _exact_period_fixture():
    # Automatic intercept plus two slopes gives k=3. The first period is a
    # full-rank 3x3 solve, while later periods are overidentified. This directly
    # exercises the n_t == k eligibility boundary on the requested backend.
    blocks = [
        np.asarray([[-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]),
        np.asarray([[-2.0, 0.5], [-0.5, -1.0], [0.4, 1.5], [1.3, -0.2], [2.2, 0.8]]),
        np.asarray([[-1.5, -0.7], [-0.3, 0.9], [0.6, -1.2], [1.4, 0.4], [2.5, 1.1]]),
    ]
    period_params = np.asarray(
        [[0.2, 1.1, -0.4], [1.0, -0.6, 0.8], [-0.7, 0.3, 1.2]],
        dtype=np.float64,
    )
    ys = []
    for X_t, beta_t in zip(blocks, period_params):
        design = np.column_stack([np.ones(X_t.shape[0]), X_t])
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise AssertionError("exact-period physical fixture lost full rank")
        ys.append(design @ beta_t)
    X = np.vstack(blocks).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    time_ids = np.concatenate(
        [np.full(block.shape[0], i, dtype=np.int64) for i, block in enumerate(blocks)]
    )
    return X, y, time_ids, period_params


def _exact_period_case(backend: str):
    X, y, time_ids, expected_betas = _exact_period_fixture()
    reference = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device="cpu"
    ).fit(X, y, time_ids=time_ids)
    Xb, yb = _arrays(X, y, backend)
    actual = FamaMacBeth(
        cov_type="newey-west", bandwidth=1, device=_device(backend)
    ).fit(Xb, yb, time_ids=time_ids)
    if reference.n_periods != 3 or actual.n_periods != 3:
        raise AssertionError(
            f"exactly identified period was filtered: reference={reference.n_periods}, "
            f"actual={actual.n_periods}"
        )
    np.testing.assert_allclose(
        _public_array(reference.betas_), expected_betas, rtol=2e-12, atol=2e-13
    )
    _assert_backend_native_outputs(actual)
    prediction_X = X[:3]
    inference_result = _assert_inference_descriptors(
        _inference_descriptor(reference), _inference_descriptor(actual)
    )
    return {
        "status": "success",
        "executed_backend": actual._backend_name,
        "inference_backend": actual._inference_backend_name,
        "period_observation_counts": [3, 5, 5],
        "design_columns_including_intercept": 3,
        "inference_result": inference_result,
        "max_abs_differences": _assert_snapshot(
            _snapshot(reference, prediction_X), _snapshot(actual, prediction_X)
        ),
    }


def _square_rank_rejection(backend: str):
    X, y, time_ids, _expected_betas = _exact_period_fixture()
    mask = time_ids == 0
    X[mask, 1] = 2.0 * X[mask, 0]
    Xb, yb = _arrays(X, y, backend)
    try:
        FamaMacBeth(device=_device(backend)).fit(Xb, yb, time_ids=time_ids)
    except ValueError as exc:
        if "rank deficient" not in str(exc):
            raise
        return True
    raise AssertionError(
        "square rank-deficient n_t == k period was filtered or accepted instead of rejected"
    )


'''
text = text.replace(anchor, addition, 1)

old = '''            "rank_deficient_retained_period_rejected": _rank_rejection(backend),
            "no_intercept_formula_rejections": _no_intercept_rejections(backend),
'''
new = '''            "rank_deficient_retained_period_rejected": _rank_rejection(backend),
            "exactly_identified_full_rank_period": _exact_period_case(backend),
            "square_rank_deficient_retained_period_rejected": _square_rank_rejection(backend),
            "no_intercept_formula_rejections": _no_intercept_rejections(backend),
'''
if old not in text:
    raise RuntimeError("result payload anchor not found")
text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")
