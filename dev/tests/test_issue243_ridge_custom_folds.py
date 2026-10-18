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


def _small_weight_scale_problem(kind):
    X = np.array([[1.], [2.], [3.]])
    y = np.array([3., 4., 5.])
    if kind == "complete":
        folds = [(np.delete(np.arange(3), i), np.array([i])) for i in range(3)]
    else:
        # The second training subset omits row 1 in addition to validation.
        folds = [(np.array([0, 1]), np.array([2])),
                 (np.array([2]), np.array([0]))]
    return X, y, folds


@pytest.mark.parametrize("kind", ["complete", "gapped"])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("weight_kind", ["uniform", "varying"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_small_explicit_folds_preserve_weight_scale(
    native, kind, intercept, weight_kind, mixed_precision,
):
    device, array = native
    X, y, folds = _small_weight_scale_problem(kind)
    weight = (np.ones(3) if weight_kind == "uniform" else
              np.array([.5, 1., 2.]))
    alphas = np.array([.01, .1, 1., 10.])
    expected = []
    for alpha in alphas:
        losses = []
        for train, valid in folds:
            coef, offset = _solve(
                X[train], y[train], alpha, weight[train], intercept,
            )
            losses.append(np.average(
                (X[valid] @ coef + offset - y[valid]) ** 2,
                weights=weight[valid],
            ))
        expected.append(losses)
    expected = np.asarray(expected)
    selected = alphas[np.argmin(expected.mean(axis=1))]
    coef, offset = _solve(X, y, selected, weight, intercept)
    tolerance = 3e-5 if device != "cpu" and mixed_precision else 2e-10
    for scale in (1e-200, 1e-18, 1., 1e200):
        scaled_weight = scale * weight
        original_weight = scaled_weight.copy()
        model = RidgeCV(
            alphas=alphas, cv_splits=folds, fit_intercept=intercept,
            device=device, compute_inference=False,
            gpu_cv_mixed_precision=mixed_precision,
        ).fit(array(X), array(y), sample_weight=array(scaled_weight))
        assert_allclose(model.cv_results_["mse_path"], expected,
                        rtol=tolerance, atol=tolerance)
        assert_allclose(model.mean_mse_, expected.mean(axis=1),
                        rtol=tolerance, atol=tolerance)
        assert model.alpha_ == selected
        assert_allclose(model.best_score_, -expected.mean(axis=1).min(),
                        rtol=tolerance, atol=tolerance)
        # Final refit uses all rows and original weights, with float64 fitting
        # independently of the selection phase's mixed-precision setting.
        assert_allclose(model.coef_, coef, rtol=2e-10, atol=2e-10)
        assert_allclose(model.intercept_, offset, rtol=2e-10, atol=2e-10)
        assert_allclose(model.predict(array(X)), X @ coef + offset,
                        rtol=2e-10, atol=2e-10)
        assert_array_equal(scaled_weight, original_weight)


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("kind", ["complete", "gapped"])
def test_weight_scale_reference_matches_objective_aligned_sklearn(intercept, kind):
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    X, y, folds = _small_weight_scale_problem(kind)
    for scale in (1e-200, 1., 1e200):
        weight = scale * np.array([.5, 1., 2.])
        for train, _ in folds:
            for alpha in (.01, .1, 1., 10.):
                coef, offset = _solve(X[train], y[train], alpha,
                                      weight[train], intercept)
                reference = sklearn_linear.Ridge(
                    alpha=alpha * weight[train].sum(),
                    fit_intercept=intercept, solver="cholesky",
                ).fit(X[train], y[train], sample_weight=weight[train])
                assert_allclose(coef, reference.coef_, rtol=2e-10, atol=2e-10)
                assert_allclose(offset, reference.intercept_, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("zero_subset", ["training", "validation"])
@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_weight_rescaling_does_not_accept_zero_mass_folds(native, zero_subset, scale):
    device, array = native
    X, y, _ = _small_weight_scale_problem("gapped")
    folds = [(np.array([0, 1]), np.array([2]))]
    weight = np.full(3, scale)
    weight[folds[0][0 if zero_subset == "training" else 1]] = 0.
    with pytest.raises(ValueError, match=zero_subset + " sample_weight must have a positive sum"):
        RidgeCV(alphas=[.01, 10.], cv_splits=folds, device=device,
                compute_inference=False).fit(array(X), array(y), sample_weight=array(weight))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("feature_scale", [1., 1e-8, 1e-15])
@pytest.mark.parametrize("rank_deficient", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_torch_cpu_ridge_eigen_path_has_no_absolute_positive_floor(
    dtype_name, feature_scale, rank_deficient, normalized,
):
    # Actual Torch CPU kernels, not a substitute for public CUDA execution.
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram

    X = feature_scale * np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    y = 2. * X[:, 0]
    if rank_deficient:
        X = np.column_stack([X[:, 0], 2. * X[:, 0]])
    weight = np.ones(6)
    if normalized:
        weight = weight / weight.sum()
    gram = X.T @ (weight[:, None] * X)
    cross = X.T @ (weight * y)
    alphas = feature_scale ** 2 * np.array([.01, .1, 1.])
    expected = np.array([
        np.linalg.solve(gram + alpha * weight.sum() * np.eye(X.shape[1]), cross)
        for alpha in alphas
    ])[:, None, :]
    dtype = getattr(torch, dtype_name)
    native_gram = torch.tensor(gram[None], dtype=dtype, requires_grad=True)
    native_cross = torch.tensor(cross[None], dtype=dtype, requires_grad=True)
    original_gram, original_cross = native_gram.detach().clone(), native_cross.detach().clone()
    actual = _solve_ridge_path_gpu_from_gram(
        native_gram, native_cross, np.array([weight.sum()]), alphas,
        TorchBackend(device="cpu"), fit_intercept=False,
    )
    assert actual.device.type == "cpu"
    assert actual.dtype == dtype
    assert native_gram.requires_grad and native_cross.requires_grad
    assert torch.equal(native_gram.detach(), original_gram)
    assert torch.equal(native_cross.detach(), original_cross)
    tolerance = 1e-4 if dtype_name == "float32" else 2e-10
    assert_allclose(actual.detach().numpy(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_small_feature_ridge_cv_matches_positive_alpha_objective(native, mixed_precision):
    device, array = native
    X = 1e-8 * np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.], [4.]])
    y = 2. * X[:, 0]
    folds = [(np.arange(6), np.array([6]))]
    alphas = np.array([1e-17, 1e-16, 1e-15])
    train, valid = folds[0]
    expected = []
    for alpha in alphas:
        coef, offset = _solve(X[train], y[train], alpha, np.ones(len(train)), False)
        expected.append(np.mean((X[valid] @ coef + offset - y[valid]) ** 2))
    for scale in (1e-18, 1.):
        model = RidgeCV(
            alphas=alphas, cv_splits=folds, fit_intercept=False,
            device=device, compute_inference=False,
            gpu_cv_mixed_precision=mixed_precision,
        ).fit(array(X), array(y), sample_weight=array(np.full(len(y), scale)))
        tolerance = 1e-4 if device != "cpu" and mixed_precision else 2e-10
        assert_allclose(model.cv_results_["mse_path"][:, 0], expected,
                        rtol=tolerance, atol=1e-28)
        assert model.alpha_ == alphas[np.argmin(expected)]
        coef, offset = _solve(X, y, model.alpha_, np.ones(len(y)), False)
        assert_allclose(model.coef_, coef, rtol=2e-10, atol=2e-10)
        assert_allclose(model.intercept_, offset, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("alpha", [1e-20, 1e-40])
def test_torch_cpu_rank_deficient_path_retains_tiny_representable_alpha(alpha):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram

    gram = torch.zeros((1, 2, 2), dtype=torch.float32)
    cross = torch.zeros((1, 2), dtype=torch.float32)
    result = _solve_ridge_path_gpu_from_gram(
        gram, cross, np.array([1.]), np.array([alpha]), TorchBackend(device="cpu"),
    )
    assert result.device.type == "cpu" and result.dtype == torch.float32
    assert torch.all(torch.isfinite(result))
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.equal(gram, torch.zeros_like(gram))
    assert torch.equal(cross, torch.zeros_like(cross))


@pytest.mark.parametrize("alpha", [1e-100, 1e100])
def test_torch_cpu_unrepresentable_alpha_reports_cv_precision(alpha):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram

    backend = TorchBackend(device="cpu")
    gram = torch.zeros((1, 1, 1), dtype=torch.float32)
    cross = torch.zeros((1, 1), dtype=torch.float32)
    with pytest.raises(FloatingPointError, match="gpu_cv_mixed_precision=False"):
        _solve_ridge_path_gpu_from_gram(
            gram, cross, np.array([1.]), np.array([alpha]), backend,
        )
    result = _solve_ridge_path_gpu_from_gram(
        gram.to(torch.float64), cross.to(torch.float64), np.array([1.]),
        np.array([alpha]), backend,
    )
    assert torch.equal(result, torch.zeros_like(result))


def test_torch_cpu_overflowing_regularized_diagonal_reports_cv_precision():
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram

    with pytest.raises(FloatingPointError, match="denominators.*finite and positive"):
        _solve_ridge_path_gpu_from_gram(
            torch.tensor([[[3e38]]], dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32), np.array([1.]),
            np.array([3e38]), TorchBackend(device="cpu"),
        )


def test_public_ridge_cv_reports_unrepresentable_mixed_precision_alpha(native):
    device, array = native
    X, y = np.zeros((3, 1)), np.arange(1., 4.)
    kwargs = dict(alphas=[1e-100, 1e100], cv_splits=[([0, 1], [2])],
                  fit_intercept=False, device=device, compute_inference=False)
    if device != "cpu":
        with pytest.raises(FloatingPointError, match="gpu_cv_mixed_precision=False"):
            RidgeCV(**kwargs, gpu_cv_mixed_precision=True).fit(array(X), array(y))
    model = RidgeCV(**kwargs, gpu_cv_mixed_precision=False).fit(array(X), array(y))
    assert_allclose(model.cv_results_["mse_path"], [[9.], [9.]], rtol=0, atol=0)
    assert_allclose(model.coef_, [0.], rtol=0, atol=0)


@pytest.mark.parametrize("constant", [.1, 1., 3.])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_normalized_weights_do_not_create_rank_in_constant_training_columns(
    native, constant, mixed_precision,
):
    device, array = native
    X = np.array([constant] * 7 + [constant + 1., constant + 2.])[:, None]
    y = np.arange(9., dtype=np.float64)
    weights = np.ones(9)
    alphas = np.array([1e-40, 1e-20, 1e-12, .1])
    model = RidgeCV(
        alphas=alphas, cv_splits=[(np.arange(7), np.arange(7, 9))],
        device=device, compute_inference=False,
        gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    # Every positive penalty has the unique constant-column slope zero.
    # The training intercept is 3; the two held-out losses average to 20.5.
    assert_allclose(model.cv_results_["mse_path"], np.full((4, 1), 20.5),
                    rtol=2e-12, atol=2e-12)
    assert model.alpha_ == alphas[0]


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("weight_kind", ["uniform", "varying_with_zero"])
def test_torch_cpu_anchored_weighted_moments_keep_exact_constant_nullspace(
    mixed_precision, weight_kind,
):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _weighted_ridge_centered_gram

    backend = TorchBackend(device="cpu")
    dtype = torch.float32 if mixed_precision else torch.float64
    X = torch.full((7, 2), .1, dtype=torch.float64)
    y = torch.arange(7, dtype=torch.float64)
    weights = torch.ones(7, dtype=torch.float64)
    if weight_kind == "varying_with_zero":
        weights = torch.tensor([0., .3, 2., .7, 4., 1., .5], dtype=torch.float64)
        X[0] = 1e300  # A zero-weight row cannot create centered rank.
    weights = weights / weights.sum()
    before = (X.clone(), y.clone(), weights.clone())
    gram, cross, _, _ = _weighted_ridge_centered_gram(
        X, y, weights, torch, backend=backend, dtype=dtype,
    )
    assert gram.dtype == cross.dtype == dtype
    assert gram.device.type == cross.device.type == "cpu"
    assert torch.equal(gram, torch.zeros_like(gram))
    assert torch.equal(cross, torch.zeros_like(cross))
    for actual, original in zip((X, y, weights), before):
        assert torch.equal(actual, original)


def _augmented_weighted_ridge(X, y, weights, alpha):
    """Independent QR least-squares reference with an unpenalized intercept."""
    weights = weights / weights.sum()
    design = np.column_stack([np.ones(len(y)), X])
    penalty_rows = np.column_stack([
        np.zeros(X.shape[1]), np.sqrt(alpha) * np.eye(X.shape[1]),
    ])
    lhs = np.vstack([np.sqrt(weights)[:, None] * design, penalty_rows])
    rhs = np.concatenate([np.sqrt(weights) * y, np.zeros(X.shape[1])])
    q, r = np.linalg.qr(lhs, mode="reduced")
    return np.linalg.solve(r, q.T @ rhs)


@pytest.mark.parametrize("shift", [0., .1, 32.])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_weighted_translated_fold_paths_match_augmented_qr(native, shift, mixed_precision):
    device, array = native
    rng = np.random.default_rng(751)
    X = rng.normal(size=(18, 3)) + shift
    y = rng.normal(size=18) + 4.
    weights = np.linspace(.2, 2., 18)
    weights[[1, 8]] = 0.
    folds = [(np.arange(0, 9)[::-1], np.arange(10, 14)),
             (np.arange(6, 15), np.arange(15, 18)[::-1])]
    alphas = np.array([.003, .4, 4.])
    expected = np.empty((len(alphas), len(folds)))
    for i, alpha in enumerate(alphas):
        for j, (train, valid) in enumerate(folds):
            params = _augmented_weighted_ridge(X[train], y[train], weights[train], alpha)
            errors = params[0] + X[valid] @ params[1:] - y[valid]
            expected[i, j] = np.average(errors ** 2, weights=weights[valid])
    model = RidgeCV(
        alphas=alphas, cv_splits=folds, device=device, compute_inference=False,
        gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    tolerance = 3e-6 if device != "cpu" and mixed_precision else 2e-10
    assert_allclose(model.cv_results_["mse_path"], expected, rtol=tolerance, atol=tolerance)
    assert_allclose(model.mean_mse_, expected.mean(axis=1), rtol=tolerance, atol=tolerance)
    selected = alphas[np.argmin(expected.mean(axis=1))]
    assert model.alpha_ == selected
    params = _augmented_weighted_ridge(X, y, weights, selected)
    assert_allclose(model.predict(array(X)), params[0] + X @ params[1:], rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_relative_weight_is_applied_before_training_precision_cast(native, mixed_precision):
    device, array = native
    X = np.array([[0.], [0.], [1e20], [1.]])
    y = np.array([0., 0., 1e20, .2])
    weights = np.array([1e20, 1e20, 1e-30, 1e20])
    alphas = np.array([5e-11, 1e-10])
    expected = np.array([.09, (1. / 3. - .2) ** 2])
    model = RidgeCV(
        alphas=alphas, cv_splits=[(np.arange(3), np.array([3]))],
        fit_intercept=False, device=device, compute_inference=False,
        gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    tolerance = 2e-6 if device != "cpu" and mixed_precision else 2e-12
    assert_allclose(model.cv_results_["mse_path"][:, 0], expected, rtol=tolerance, atol=0.)
    assert model.alpha_ == alphas[1]


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("response,weight,expected", [
    (1e15, 1e-50, 1e-20),
    (1e200, 1e-300, 1e100),
])
def test_weighted_validation_scales_before_squaring(native, mixed_precision, response, weight, expected):
    device, array = native
    X = np.zeros((4, 1))
    y = np.array([0., 0., 0., response])
    weights = np.array([1., 1., 1., weight])
    model = RidgeCV(
        alphas=[.1, 1.], cv_splits=[(np.array([0, 1]), np.array([2, 3]))],
        fit_intercept=False, device=device, compute_inference=False,
        gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    assert_allclose(model.cv_results_["mse_path"], expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_cpu_weighted_factors_retain_representable_small_contributions(dtype):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _weighted_ridge_gram, _batch_mse_all_folds

    backend = TorchBackend(device="cpu")
    solve_dtype = getattr(torch, dtype)
    X = torch.tensor([[0.], [0.], [1e20]], dtype=torch.float64)
    y = X[:, 0].clone()
    weights = torch.tensor([1e20, 1e20, 1e-30], dtype=torch.float64)
    weights /= weights.sum()
    before = (X.clone(), y.clone(), weights.clone())
    gram, cross = _weighted_ridge_gram(X, y, weights, torch, backend, solve_dtype)
    tolerance = 2e-6 if dtype == "float32" else 2e-12
    assert gram.dtype == cross.dtype == solve_dtype
    assert_allclose(gram.numpy(), [[5e-11]], rtol=tolerance, atol=0.)
    assert_allclose(cross.numpy(), [5e-11], rtol=tolerance, atol=0.)
    for actual, original in zip((X, y, weights), before):
        assert torch.equal(actual, original)
    result = _batch_mse_all_folds(
        torch.zeros((1, 2, 1), dtype=torch.float64),
        torch.tensor([[0., 1e200]], dtype=torch.float64),
        torch.zeros((1, 1, 1), dtype=solve_dtype),
        torch.zeros((1, 1), dtype=torch.float64), backend,
        torch.tensor([[1., 1e-300]], dtype=torch.float64), n_val_folds=[2],
    )
    assert result.device.type == "cpu"
    assert_allclose(result.numpy(), [[1e100]], rtol=2e-12, atol=0.)


@pytest.mark.parametrize("alpha,n_train", [(1e-46, 10), (1e-46, 100), (1e-47, 1000)])
def test_torch_cpu_effective_alpha_is_formed_before_precision_cast(alpha, n_train):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram

    effective = torch.tensor(alpha * n_train, dtype=torch.float32)
    assert effective > 0
    result = _solve_ridge_path_gpu_from_gram(
        torch.zeros((1, 1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32), np.array([n_train]),
        np.array([alpha]), TorchBackend(device="cpu"),
    )
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert torch.equal(result, torch.zeros_like(result))


def test_public_cv_accepts_representable_effective_alpha(native):
    device, array = native
    X, y = np.zeros((101, 1)), np.ones(101)
    model = RidgeCV(
        alphas=[1e-46, 1e-47], cv_splits=[(np.arange(100), np.array([100]))],
        fit_intercept=False, device=device, compute_inference=False,
        gpu_cv_mixed_precision=True,
    ).fit(array(X), array(y))
    assert_allclose(model.cv_results_["mse_path"], [[1.], [1.]], rtol=0, atol=0)
    assert_allclose(model.coef_, [0.], rtol=0, atol=0)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_relative_weight_below_float64_range_preserves_finite_training_products(native, mixed_precision):
    device, array = native
    X = np.array([0., 1e200, 1., 0., 1e200, 1.])[:, None]
    y = np.array([0., 1e200, .5, 0., 1e200, .5])
    weights = np.array([1e100, 1e-300, 1., 1e100, 1e-300, 1.])
    folds = [(np.array([0, 1]), np.array([2])),
             (np.array([3, 4]), np.array([5]))]
    # Each training fold has mean weighted Gram/cross 1, despite w_small /
    # sum(w) underflowing to zero. The exact slope is 1 / (1 + alpha).
    expected = np.repeat(np.array([[(1. / 1.1 - .5) ** 2], [0.]]), 2, axis=1)
    model = RidgeCV(
        alphas=[.1, 1.], cv_splits=folds, fit_intercept=False,
        device=device, compute_inference=False,
        gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    tolerance = 2e-6 if device != "cpu" and mixed_precision else 2e-12
    assert_allclose(model.cv_results_["mse_path"], expected, rtol=tolerance, atol=1e-28)
    assert model.alpha_ == 1.
    assert_allclose(model.coef_, [.5], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_relative_validation_weight_below_float64_range_is_applied_before_square(native, mixed_precision):
    device, array = native
    X, y = np.zeros((4, 1)), np.array([0., 0., 0., 1e200])
    weights = np.array([1., 1., 1e100, 1e-300])
    model = RidgeCV(
        alphas=[.1, 1.], cv_splits=[(np.array([0, 1]), np.array([2, 3]))],
        fit_intercept=False, device=device, compute_inference=False,
        gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    assert_allclose(model.cv_results_["mse_path"], [[1.], [1.]], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_cpu_extreme_relative_weights_survive_before_factor_cast(dtype):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _weighted_ridge_gram, _batch_mse_all_folds

    backend = TorchBackend(device="cpu")
    solve_dtype = getattr(torch, dtype)
    weights = torch.tensor([1e100, 1e-300], dtype=torch.float64)
    X = torch.tensor([[0.], [1e200]], dtype=torch.float64)
    gram, cross = _weighted_ridge_gram(X, X[:, 0], weights, torch, backend, solve_dtype)
    assert_allclose(gram.numpy(), [[1.]], rtol=2e-12, atol=2e-12)
    assert_allclose(cross.numpy(), [1.], rtol=2e-12, atol=2e-12)
    result = _batch_mse_all_folds(
        torch.zeros((1, 2, 1), dtype=torch.float64), X[:, 0][None],
        torch.zeros((1, 1, 1), dtype=solve_dtype),
        torch.zeros((1, 1), dtype=torch.float64), backend, weights[None], n_val_folds=[2],
    )
    assert_allclose(result.numpy(), [[1.]], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_finite_weighted_residuals_are_subtracted_before_scaling(native, mixed_precision):
    device, array = native
    X = np.zeros((4, 1))
    y = np.array([1e16, 1e16, 1e16 + 2., 1e16 + 2.])
    weights = np.array([1., 1., .5, 1.])
    model = RidgeCV(
        alphas=[.1, 1.], cv_splits=[(np.array([0, 1]), np.array([2, 3]))],
        device=device, compute_inference=False, gpu_cv_mixed_precision=mixed_precision,
    ).fit(array(X), array(y), sample_weight=array(weights))
    assert_allclose(model.cv_results_["mse_path"], [[4.], [4.]], rtol=2e-15, atol=0.)


def test_torch_cpu_finite_weighted_residual_cancellation_and_overflow_rescue():
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _batch_mse_all_folds
    backend = TorchBackend(device="cpu")
    result = _batch_mse_all_folds(
        torch.zeros((1, 2, 1), dtype=torch.float64),
        torch.tensor([[1e16 + 2., 1e16 + 2.]], dtype=torch.float64),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        torch.tensor([[1e16]], dtype=torch.float64), backend,
        torch.tensor([[.5, 1.]], dtype=torch.float64), n_val_folds=[2],
    )
    assert_allclose(result.numpy(), [[4.]], rtol=2e-15, atol=0.)
    rescued = _batch_mse_all_folds(
        torch.tensor([[[1e308], [0.]]], dtype=torch.float64),
        torch.zeros((1, 2), dtype=torch.float64),
        torch.tensor([[[10.]]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64), backend,
        torch.tensor([[1e-320, 1.]], dtype=torch.float64), n_val_folds=[2],
    )
    # Form the finite square-root weighted prediction independently.
    expected = (np.sqrt(1e-320) * 1e308 * 10.) ** 2
    assert_allclose(rescued.numpy(), [[expected]], rtol=2e-12, atol=0.)


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_rank_one_cv_predictions_use_stable_original_factors(native, mixed_precision, weighted):
    device, array = native
    v = np.array([1., 3., 7.])
    X, y = np.tile(v, (4, 1)), np.full(4, 3.)
    weights = np.array([.5, 2., 1., 1.]) if weighted else None
    kwargs = dict(alphas=[1e-40, 1.], cv_splits=[(np.arange(3), np.array([3]))],
                  fit_intercept=False, device=device, compute_inference=False,
                  gpu_cv_mixed_precision=mixed_precision)
    if weighted or device != "cpu":
        with pytest.warns(RuntimeWarning, match="same-device float64 factor-SVD recovery"):
            model = RidgeCV(**kwargs).fit(array(X), array(y), sample_weight=None if weights is None else array(weights))
    else:
        # The unchanged unweighted NumPy solve already handles this exact case.
        model = RidgeCV(**kwargs).fit(array(X), array(y))
    assert_allclose(model.cv_results_["mse_path"], [[0.], [.0025]], rtol=2e-10, atol=1e-26)
    assert model.alpha_ == 1e-40


@pytest.fixture(params=[
    pytest.param(("numpy", "cpu"), id="numpy"),
    pytest.param(("torch", "cpu"), id="torch"),
    pytest.param(("cupy", "cuda"), id="cupy"),
    pytest.param(("torch", "cuda"), id="torch-cuda"),
])
def factor_backend(request):
    from statgpu.backends import get_backend
    backend_name, device = request.param
    backend = get_backend(backend=backend_name, device=device)
    if not backend.is_available():
        message = f"{backend_name} {device} unavailable"
        if os.environ.get("STATGPU_REQUIRE_PHYSICAL_GPU") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return backend, device


def _assert_native_factor_array(actual, backend, device, reference=None):
    assert actual.dtype == backend.float64
    if backend.name == "torch":
        import torch
        assert isinstance(actual, torch.Tensor)
        assert actual.device.type == device
        if reference is not None:
            assert actual.device == reference.device
    elif backend.name == "cupy":
        assert isinstance(actual, backend.xp.ndarray)
        assert actual.device.id == backend.xp.cuda.Device().id
        if reference is not None:
            assert actual.device.id == reference.device.id
    else:
        assert isinstance(actual, np.ndarray)
        assert device == "cpu"


def test_factor_recovery_keeps_tiny_independent_and_resolved_collinear_directions(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    # A global relative singular cutoff would discard the second true column.
    X = np.diag([1., 1e-30])
    y = np.ones(2)
    alpha = np.array([1e-62])
    X_native, y_native = backend.asarray(X), backend.asarray(y)
    _assert_native_factor_array(X_native, backend, device)
    _assert_native_factor_array(y_native, backend, device)
    actual = _solve_ridge_path_from_factors(X_native, y_native, alpha, backend)
    _assert_native_factor_array(actual, backend, device, X_native)
    expected = np.diag(X) / (np.diag(X) ** 2 + alpha[0])
    assert_allclose(backend.to_numpy(actual)[:, 0], expected, rtol=2e-12, atol=0.)
    # The cancellation direction is small, but safely above float64 error.
    u, z = np.array([1., -1., 1., -1.]), np.array([1., 1., -1., -1.])
    delta = 2. ** -20
    X = np.column_stack([u, u + delta * z])
    alpha = delta ** 2 / 100.
    X_native, y_native = backend.asarray(X), backend.asarray(z)
    actual = _solve_ridge_path_from_factors(X_native, y_native, np.array([4. * alpha]), backend)
    _assert_native_factor_array(actual, backend, device, X_native)
    # Solve the 2x2 normal equations symbolically without subtracting nearly
    # equal float64 entries. Here X'X/4 = [[1,1],[1,1+delta^2]].
    determinant = delta ** 2 + 2. * alpha + alpha * delta ** 2 + alpha ** 2
    expected = np.array([-delta, (1. + alpha) * delta]) / determinant
    assert_allclose(backend.to_numpy(actual)[:, 0], expected, rtol=3e-10, atol=0.)


@pytest.mark.parametrize("delta", [2. ** -50, 2. ** -51, 2. ** -52])
def test_response_bearing_unresolved_directions_raise_precision_error(factor_backend, delta):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    u, z = np.array([1., -1., 1., -1.]), np.array([1., 1., -1., -1.])
    X = np.column_stack([u, u + delta * z])
    X_native, y_native = backend.asarray(X), backend.asarray(z)
    _assert_native_factor_array(X_native, backend, device)
    _assert_native_factor_array(y_native, backend, device)
    with pytest.raises(FloatingPointError, match="response-bearing.*finite ridge solution"):
        _solve_ridge_path_from_factors(X_native, y_native, np.array([4. * delta ** 2 / 100.]), backend)


def test_torch_cpu_gram_recovery_is_lazy_and_uses_original_precision():
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram
    backend = TorchBackend(device="cpu")
    calls = []
    def unexpected(_):
        calls.append(True)
        raise AssertionError("Well-resolved Gram must not request factor recovery")
    _solve_ridge_path_gpu_from_gram(
        torch.diag(torch.tensor([1e-20, 2e-20], dtype=torch.float64))[None],
        torch.tensor([[1e-20, 2e-20]], dtype=torch.float64), np.array([1.]),
        np.array([1e-40]), backend, factor_factory=unexpected,
    )
    assert not calls
    v = torch.tensor([1., 3., 7.], dtype=torch.float64)
    original_X = v.repeat(3, 1)
    original_y = torch.full((3,), 3., dtype=torch.float64)
    def factors(fold):
        assert fold == 0
        calls.append(True)
        return original_X, original_y
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        coef = _solve_ridge_path_gpu_from_gram(
            (original_X.T @ original_X).to(torch.float32)[None],
            (original_X.T @ original_y).to(torch.float32)[None],
            np.array([3.]), np.array([1e-40, 1.]), backend, factor_factory=factors,
        )
    assert calls == [True]
    assert coef.dtype == torch.float64 and coef.device.type == "cpu"
    expected = np.stack([3. * v.numpy() / (59. + alpha) for alpha in [1e-40, 1.]])
    assert_allclose(coef[:, 0].numpy(), expected, rtol=2e-12, atol=2e-12)


def test_weighted_cpu_off_span_predictions_match_rank_one_closed_form():
    v = np.array([1., 3., 7.])
    X = np.vstack([np.tile(v, (3, 1)), np.eye(3)])
    y = np.r_[np.full(3, 3.), np.zeros(3)]
    alphas = np.array([1e-40, .01, 1.])
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        model = RidgeCV(
            alphas=alphas, cv_splits=[(np.arange(3), np.arange(3, 6))],
            fit_intercept=False, device="cpu", compute_inference=False,
        ).fit(X, y, sample_weight=np.ones(6))
    expected = [np.mean((3. * v / (59. + alpha)) ** 2) for alpha in alphas]
    assert_allclose(model.cv_results_["mse_path"][:, 0], expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_unrepresentable_weighted_validation_loss_raises(native, mixed_precision):
    device, array = native
    with pytest.raises(FloatingPointError, match="weighted validation MSE.*not finite"):
        RidgeCV(
            alphas=[.1, 1.], cv_splits=[(np.array([0, 1]), np.array([2]))],
            fit_intercept=False, device=device, compute_inference=False,
            gpu_cv_mixed_precision=mixed_precision,
        ).fit(array(np.zeros((3, 1))), array(np.array([0., 0., 1e200])),
              sample_weight=array(np.ones(3)))


def test_torch_cpu_weighted_scoring_rejects_infinite_loss_and_masks_unused_rows():
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._ridge_cv import _batch_mse_all_folds
    backend = TorchBackend(device="cpu")
    with pytest.raises(FloatingPointError, match="weighted validation MSE.*not finite"):
        _batch_mse_all_folds(
            torch.zeros((1, 1, 1), dtype=torch.float64),
            torch.tensor([[1e200]], dtype=torch.float64),
            torch.zeros((1, 1, 1), dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64), backend,
            torch.ones((1, 1), dtype=torch.float64), n_val_folds=[1],
        )
    # The second row has zero analytic weight; the third is padding. Their
    # raw 1e308*10 predictions overflow, but neither contributes to the loss.
    result = _batch_mse_all_folds(
        torch.tensor([[[1.], [1e308], [1e308]]], dtype=torch.float64),
        torch.tensor([[9., 0., 0.]], dtype=torch.float64),
        torch.tensor([[[10.]]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64), backend,
        torch.tensor([[1., 0., 0.]], dtype=torch.float64), n_val_folds=[2],
    )
    assert_allclose(result.numpy(), [[1.]], rtol=0., atol=0.)
