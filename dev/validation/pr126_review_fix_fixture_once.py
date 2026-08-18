from pathlib import Path


def replace_between(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"fixture anchors not found in {path}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


test_path = "dev/tests/test_panel_stage_c_covariance.py"
replacement = r'''def test_grouped_score_multiscale_cancellation_survives_three_levels():
    scores = np.asarray(
        [[1.0e154], [-1.0e154], [1.0e138], [1.0], [-1.0e138], [-1.0]],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.int64)
    grouped = _grouped_score_sums(
        scores,
        cluster1,
        n_groups=2,
        xp=np,
    )
    np.testing.assert_array_equal(grouped, np.asarray([[-1.0], [1.0]]))

    # Use a binary-exact constant design: with n=8 and X=0.5, X'X=2,
    # bread=0.5 and every influence multiplier is exactly 0.25.  This keeps
    # the regression focused on grouped-score cancellation rather than SVD
    # representation error from a non-binary 1/6 design.
    deep_scores = np.asarray(
        [1.0e154, -1.0e154, 1.0e138, 1.0, -1.0e138, -1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    deep_cluster1 = np.asarray([0, 0, 0, 1, 0, 0, 2, 3], dtype=np.int64)
    deep_cluster2 = np.asarray([0, 0, 0, 1, 0, 1, 2, 3], dtype=np.int64)
    X = np.full((8, 1), 0.5, dtype=np.float64)
    actual = two_way_clustered_covariance(
        X,
        4.0 * deep_scores,
        deep_cluster1,
        deep_cluster2,
    )
    np.testing.assert_allclose(actual, np.zeros((1, 1)), rtol=0.0, atol=0.0)


'''
replace_between(
    test_path,
    "def test_grouped_score_multiscale_cancellation_survives_three_levels():",
    "def test_one_way_and_dk_preserve_small_score_after_same_sign_swallow():",
    replacement,
)

runner = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = runner.read_text(encoding="utf-8")
old = '''    deep_scores_np = np.asarray(
        [1.0e154, -1.0e154, 1.0e138, 1.0, -1.0e138, -1.0],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.int64)
    cluster2 = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64)
    X_deep_np = np.full((6, 1), 1.0 / 6.0, dtype=np.float64)
    deep_dummy = np.arange(6, dtype=np.int64)
    X_deep, deep_scores, _entity2, _time2 = _to_backend(
        X_deep_np,
        deep_scores_np,
        deep_dummy,
        deep_dummy,
        backend,
    )
'''
new = '''    deep_scores_np = np.asarray(
        [1.0e154, -1.0e154, 1.0e138, 1.0, -1.0e138, -1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0, 2, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 0, 0, 1, 0, 1, 2, 3], dtype=np.int64)
    X_deep_np = np.full((8, 1), 0.5, dtype=np.float64)
    deep_dummy = np.arange(8, dtype=np.int64)
    X_deep, deep_scores, _entity2, _time2 = _to_backend(
        X_deep_np,
        4.0 * deep_scores_np,
        deep_dummy,
        deep_dummy,
        backend,
    )
'''
if old not in text:
    raise RuntimeError("physical multiscale fixture anchor not found")
runner.write_text(text.replace(old, new, 1), encoding="utf-8")

Path("dev/validation/pr126_review_fix_fixture_once.py").unlink(missing_ok=True)
