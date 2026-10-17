"""Independent weighted closed-form checks for caller-owned RidgeCV splits."""
import os
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statgpu import RidgeCV
from statgpu.cross_validation._base import folds_are_complete
from statgpu.linear_model.cv._ridge_cv import _validated_ridge_folds, _ridge_folds_have_complementary_training


@pytest.fixture(params=['cpu', 'cuda', 'torch'])
def native(request):
    device = request.param
    try:
        if device == 'cuda':
            import cupy as cp
            cp.zeros(1)
            return device, cp.asarray
        if device == 'torch':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Torch CUDA unavailable')
            return device, lambda x: torch.as_tensor(x, dtype=torch.float64, device='cuda')
        return device, np.asarray
    except (ImportError, RuntimeError) as exc:
        if os.environ.get('STATGPU_REQUIRE_PHYSICAL_GPU') == '1':
            pytest.fail(str(exc))
        pytest.skip(str(exc))


def _problem(kind):
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(12, 2)), rng.normal(size=12)
    folds = [(np.arange(4, 6), np.arange(0, 4)),
             (np.arange(8, 10), np.arange(4, 8)),
             (np.arange(0, 2), np.arange(8, 12))]
    if kind == 'complete':
        folds = [(np.setdiff1d(np.arange(12), v)[::-1], v) for _, v in folds]
    if kind == 'partial':
        folds = folds[:2]
    return X, y, folds


def _solve(X, y, alpha, weight, intercept):
    w = np.ones(len(y)) if weight is None else weight
    xm = np.average(X, weights=w, axis=0) if intercept else np.zeros(X.shape[1])
    ym = np.average(y, weights=w) if intercept else 0.
    centered = X - xm
    coef = np.linalg.solve(centered.T @ (centered * w[:, None]) +
                           w.sum() * alpha * np.eye(X.shape[1]), centered.T @ (w * (y - ym)))
    return coef, ym - xm @ coef


@pytest.mark.parametrize('kind', ['complete', 'subset', 'partial'])
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('weight_kind', ['none', 'unit', 'varying'])
def test_custom_training_rows_match_closed_form(native, kind, intercept, weight_kind):
    device, array = native
    X, y, folds = _problem(kind)
    weight = None if weight_kind == 'none' else (np.ones(len(y)) if weight_kind == 'unit' else np.linspace(.2, 2., len(y)))
    alphas = np.array([.01, .1, 1., 10.])
    expected = []
    for alpha in alphas:
        losses = []
        for train, valid in folds:
            coef, offset = _solve(X[train], y[train], alpha, None if weight is None else weight[train], intercept)
            losses.append(np.average((X[valid] @ coef + offset - y[valid]) ** 2,
                                     weights=None if weight is None else weight[valid]))
        expected.append(losses)
    expected = np.array(expected)
    selected = alphas[np.argmin(expected.mean(axis=1))]
    model = RidgeCV(alphas=alphas, cv_splits=folds, fit_intercept=intercept, device=device,
                    compute_inference=False, gpu_cv_mixed_precision=False).fit(
        array(X), array(y), sample_weight=None if weight is None else array(weight))
    assert_allclose(model.cv_results_['mse_path'], expected, rtol=2e-10, atol=2e-10)
    assert_allclose(model.mean_mse_, expected.mean(axis=1), rtol=2e-10, atol=2e-10)
    assert model.alpha_ == selected
    assert_allclose(model.best_score_, -expected.mean(axis=1).min(), rtol=2e-10)
    coef, offset = _solve(X, y, selected, weight, intercept)
    assert_allclose(model.coef_, coef, rtol=2e-10, atol=2e-10)
    assert_allclose(model.intercept_, offset, rtol=2e-10, atol=2e-10)
    if kind == 'subset' and not intercept and weight_kind in ('none', 'unit'):
        assert selected == 10.


def test_fast_statistics_require_training_complements():
    for kind, fast, coverage in [('complete', True, True), ('subset', False, True), ('partial', False, False)]:
        _, _, folds = _problem(kind)
        normalized = _validated_ridge_folds(folds, 12)
        assert _ridge_folds_have_complementary_training(normalized, 12) == fast
        assert folds_are_complete(normalized, 12) == coverage


@pytest.mark.parametrize('splits', [[], [([], [1])], [([0], [])], [([0, 0], [1])],
    [([0], [1, 1])], [([0, 1], [1, 2])], [([-1], [1])], [([0], [12])],
    [([False, 2], [1])], [([.5], [1])], [([[0]], [1])], [(['0'], [1])], [([np.nan], [1])], [([np.inf], [1])],
    [([2 ** 64], [1])], [([0], [1], [2])], 3])
@pytest.mark.parametrize('alphas', [[1.], [.1, 1.]])
def test_invalid_splits_rejected_even_before_degenerate_selection(splits, alphas):
    X, y, _ = _problem('subset')
    with pytest.raises(ValueError, match='cv_splits'):
        RidgeCV(alphas=alphas, cv_splits=splits, device='cpu', compute_inference=False).fit(X, y)


@pytest.mark.parametrize("n_samples", [2, 3])
@pytest.mark.parametrize("alphas", [[.001, 10.], [10.]])
@pytest.mark.parametrize("cv", [2, 5])
def test_explicit_folds_scored_for_small_data_single_alpha_and_irrelevant_cv(native, n_samples, alphas, cv):
    device, array = native
    X = np.arange(1., n_samples + 1).reshape(-1, 1)
    y = np.ones(n_samples)
    folds = [(np.array([i]), np.setdiff1d(np.arange(n_samples), [i])) for i in range(2)]
    expected = []
    for alpha in alphas:
        losses = []
        for train, valid in folds:
            coef, offset = _solve(X[train], y[train], alpha, None, False)
            losses.append(np.mean((X[valid] @ coef + offset - y[valid]) ** 2))
        expected.append(losses)
    expected = np.asarray(expected)
    model = RidgeCV(alphas=alphas, cv=cv, cv_splits=folds, fit_intercept=False,
                    compute_inference=False, device=device, gpu_cv_mixed_precision=False).fit(array(X), array(y))
    assert_allclose(model.cv_results_["mse_path"], expected, rtol=2e-10, atol=2e-10)
    assert_allclose(model.mean_mse_, expected.mean(axis=1), rtol=2e-10, atol=2e-10)
    assert model.alpha_ == alphas[np.argmin(expected.mean(axis=1))]
    if n_samples == 3:
        assert model.alpha_ == 10.


@pytest.mark.parametrize("zero_subset", ["training", "validation"])
@pytest.mark.parametrize("alphas", [[.01, 10.], [10.]])
def test_each_fold_requires_positive_training_and_validation_weight_mass(native, zero_subset, alphas):
    device, array = native
    X = np.arange(12.).reshape(6, 2)
    y = np.arange(6.)
    folds = [(np.arange(3), np.arange(3, 6))]
    weight = np.ones(6)
    weight[folds[0][0 if zero_subset == "training" else 1]] = 0.
    with pytest.raises(ValueError, match=zero_subset + " sample_weight must have a positive sum"):
        RidgeCV(alphas=alphas, cv_splits=folds, compute_inference=False, device=device).fit(
            array(X), array(y), sample_weight=array(weight))


def test_zero_individual_weights_remain_valid_when_each_fold_has_positive_mass():
    X, y, folds = _problem("subset")
    weight = np.ones(12)
    weight[[0, 4, 8]] = 0.
    model = RidgeCV(alphas=[.01, 10.], cv_splits=folds, fit_intercept=False,
                    device="cpu", compute_inference=False).fit(X, y, sample_weight=weight)
    expected = []
    for alpha in [.01, 10.]:
        losses = []
        for train, valid in folds:
            coef, offset = _solve(X[train], y[train], alpha, weight[train], False)
            losses.append(np.average((X[valid] @ coef + offset - y[valid]) ** 2, weights=weight[valid]))
        expected.append(losses)
    assert_allclose(model.cv_results_["mse_path"], expected, rtol=2e-10, atol=2e-10)


def _negative_stride_folds(kind, reverse):
    if kind == "singleton":
        X = np.array([[1., 2.], [3., 4.]])
        y = np.array([2., 5.])
        folds = [(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)),
                 (np.array([1], dtype=np.int64), np.array([0], dtype=np.int64))]
    else:
        X, y, folds = _problem(kind)
    # _problem's complete folds already reverse train: start from owned, positive
    # strides so train-only / validation-only exercise distinct conversion sites.
    folds = [(train.copy(), valid.copy()) for train, valid in folds]
    return X, y, [
        (train[::-1] if reverse in ("train", "both") else train,
         valid[::-1] if reverse in ("validation", "both") else valid)
        for train, valid in folds
    ]


@pytest.mark.parametrize("kind,reverse,intercept,weight_kind", [
    ("complete", "train", False, "none"),
    ("complete", "validation", True, "unit"),
    ("complete", "both", False, "varying"),
    ("subset", "train", False, "unit"),
    ("subset", "validation", False, "varying"),
    ("subset", "both", True, "none"),
    ("partial", "train", True, "varying"),
    ("partial", "validation", False, "none"),
    ("partial", "both", True, "unit"),
    ("singleton", "train", True, "none"),
    ("singleton", "validation", False, "unit"),
    ("singleton", "both", True, "varying"),
])
def test_negative_stride_splits_match_closed_form(native, kind, reverse, intercept, weight_kind):
    device, array = native
    X, y, folds = _negative_stride_folds(kind, reverse)
    original_X, original_y = X.copy(), y.copy()
    originals = [(indices, indices.copy(), indices.strides)
                 for pair in folds for indices in pair]
    alphas = np.array([.01, .1, 1., 10.])
    weight = (None if weight_kind == "none" else
              np.ones(len(y)) if weight_kind == "unit" else np.linspace(.25, 2., len(y)))
    original_weight = None if weight is None else weight.copy()
    expected = []
    for alpha in alphas:
        losses = []
        for train, valid in folds:
            coef, offset = _solve(
                X[train], y[train], alpha,
                None if weight is None else weight[train], intercept,
            )
            losses.append(np.average(
                (X[valid] @ coef + offset - y[valid]) ** 2,
                weights=None if weight is None else weight[valid],
            ))
        expected.append(losses)
    expected = np.asarray(expected)
    selected = alphas[np.argmin(expected.mean(axis=1))]
    coef, offset = _solve(X, y, selected, weight, intercept)
    placements = [("native", array, array)]
    if device != "cpu":
        placements.append(("host", np.asarray, np.asarray))
        if weight is not None:
            placements.append(("host_weights", array, np.asarray))
    for placement, data_array, weight_array in placements:
        model = RidgeCV(
            alphas=alphas, cv_splits=folds, fit_intercept=intercept,
            device=device, compute_inference=False,
            gpu_cv_mixed_precision=False,
        ).fit(data_array(X), data_array(y), sample_weight=(
            None if weight is None else weight_array(weight)
        ))
        assert_allclose(model.alphas_, alphas, rtol=0, atol=0, err_msg=placement)
        assert_allclose(model.cv_results_["mse_path"], expected,
                        rtol=2e-10, atol=2e-10, err_msg=placement)
        assert_allclose(model.mean_mse_, expected.mean(axis=1),
                        rtol=2e-10, atol=2e-10, err_msg=placement)
        assert model.alpha_ == selected, placement
        assert_allclose(model.best_score_, -expected.mean(axis=1).min(),
                        rtol=2e-10, err_msg=placement)
        assert_allclose(model.coef_, coef, rtol=2e-10, atol=2e-10, err_msg=placement)
        assert_allclose(model.intercept_, offset, rtol=2e-10, atol=2e-10, err_msg=placement)
        assert model.cv_selected_device_.value == device, placement
        assert model.estimator_._get_compute_device().value == device, placement
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)
    if weight is not None:
        assert_array_equal(weight, original_weight)
    for indices, original, strides in originals:
        assert_array_equal(indices, original)
        assert indices.strides == strides


@pytest.mark.parametrize("size", [1, 3])
@pytest.mark.parametrize("reverse", ["train", "validation", "both"])
def test_negative_stride_normalization_preserves_indices_and_inputs(size, reverse):
    train = np.arange(size, dtype=np.int64)
    valid = np.arange(size, 2 * size, dtype=np.int64)
    train = train[::-1] if reverse in ("train", "both") else train
    valid = valid[::-1] if reverse in ("validation", "both") else valid
    for indices in (train, valid):
        indices.flags.writeable = False
    snapshots = [(indices.copy(), indices.strides) for indices in (train, valid)]
    normalized = _validated_ridge_folds([(train, valid)], 2 * size)[0]
    for original, actual, (values, strides) in zip((train, valid), normalized, snapshots):
        assert_array_equal(actual, values)
        assert actual.dtype == np.dtype(np.int64)
        assert actual.flags.c_contiguous
        assert all(stride >= 0 for stride in actual.strides)
        assert_array_equal(original, values)
        assert original.strides == strides
        assert not original.flags.writeable
        if any(stride < 0 for stride in strides):
            assert not np.shares_memory(actual, original)
            if size == 1:
                # ascontiguousarray alone leaves this negative stride untouched.
                assert original.flags.c_contiguous


@pytest.mark.parametrize("size", [1, 3])
def test_normalized_negative_stride_indices_convert_on_torch_cpu(size):
    # Exercise the actual weight-index conversion without making device='torch'
    # use CPU or simulating CUDA availability in the public estimator.
    torch = pytest.importorskip("torch")
    from statgpu.backends import xp_asarray

    folds = [(np.arange(size, dtype=np.int64)[::-1],
              np.arange(size, 2 * size, dtype=np.int64)[::-1])]
    normalized = _validated_ridge_folds(folds, 2 * size)
    weights = torch.arange(1, 2 * size + 1, dtype=torch.float64, device="cpu")
    for original, indices in zip(folds[0], normalized[0]):
        assert original.strides[0] < 0
        converted = xp_asarray(indices, dtype=torch.int64, xp=torch, ref_arr=weights)
        assert converted.device.type == "cpu"
        assert converted.dtype == torch.int64
        assert_array_equal(converted.numpy(), original)
        assert weights[converted].sum().item() == np.sum(original + 1)
