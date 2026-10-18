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


@pytest.mark.parametrize("kind", ["repeated_rows", "duplicate_columns"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_noisy_exact_duplicate_folds_match_rank_one_objective(native, kind, mixed_precision):
    device, array = native
    if kind == "repeated_rows":
        direction = np.array([1., 3., 7.])
        X = np.vstack([np.tile(direction, (3, 1)), np.eye(3)])
        y = np.array([1., 2., 3., 0., 0., 0.])
        weights = np.ones(6)
        # Response variation within identical factor rows contributes only a
        # coefficient-independent constant to the training loss.
        train_coefs = lambda alpha: 2. * direction / (59. + alpha)
    else:
        x = np.array([1., 2., 3., 0.])
        X = np.vstack([np.column_stack([x, x]), np.eye(2)])
        y = np.array([1., 0., 3., 7., 0., 0.])
        weights = np.array([.5, 2., 1., 0., 1., 1.])
        mass = weights[:4].sum()
        cross = np.dot(weights[:4] * x, y[:4]) / mass
        gram = np.dot(weights[:4] * x, x) / mass
        # The minimum-penalty representation assigns equal slopes to the two
        # identical columns; their total slope has penalty alpha/2.
        train_coefs = lambda alpha: np.full(2, cross / (2. * gram + alpha))
    n_train = len(y) - X.shape[1]
    train, valid = np.arange(n_train), np.arange(n_train, len(y))
    alphas = np.array([1e-40, .1])
    expected = np.array([
        np.average((X[valid] @ train_coefs(alpha) - y[valid]) ** 2,
                   weights=weights[valid]) for alpha in alphas
    ])
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        model = RidgeCV(
            alphas=alphas, cv_splits=[(train, valid)], fit_intercept=False,
            device=device, compute_inference=False,
            gpu_cv_mixed_precision=mixed_precision,
        ).fit(array(X), array(y), sample_weight=array(weights))
    assert_allclose(model.cv_results_["mse_path"][:, 0], expected, rtol=2e-12, atol=2e-12)
    assert model.alpha_ == alphas[np.argmin(expected)] == .1
    expected_coef, expected_intercept = _solve(X, y, model.alpha_, weights, False)
    assert_allclose(model.coef_, expected_coef, rtol=2e-10, atol=2e-10)
    assert_allclose(model.predict(array(X)), X @ expected_coef, rtol=2e-10, atol=2e-10)
    assert model.intercept_ == expected_intercept == 0.
    predicted = X @ expected_coef
    assert_allclose(model.score(array(X), y),
                    1. - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2),
                    rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("scale", [1e-100, 1., 1e100])
def test_exact_factor_duplicates_preserve_tiny_independent_directions(factor_backend, scale):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    X = scale * np.array([[1., 1., 0.], [0., -0., 1e-30],
                         [0., 0., 0.], [1., 1., -0.]])
    y = np.array([1., 1., 3., 2.])
    alpha = scale ** 2 * 1e-62
    X_native, y_native = backend.asarray(X), backend.asarray(y)
    actual = _solve_ridge_path_from_factors(X_native, y_native, np.array([alpha]), backend)
    _assert_native_factor_array(actual, backend, device, X_native)
    expected = np.array([3. / (4. + 1e-62), 3. / (4. + 1e-62),
                         1e-30 / (1e-60 + 1e-62)]) / scale
    assert_allclose(backend.to_numpy(actual)[:, 0], expected, rtol=2e-12, atol=0.)
    assert_array_equal(backend.to_numpy(X_native), X)
    assert_array_equal(backend.to_numpy(y_native), y)


def test_exact_factor_reduction_preserves_objective_and_penalty(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _compress_exact_ridge_factors
    backend, device = factor_backend
    X = np.array([[1., 1., 2.], [-1., -1., 3.], [1., 1., 2.],
                  [0., -0., 0.], [-1., -1., 3.]])
    y = np.array([1., 4., -2., 8., 3.])
    reduced, row_inverse, row_roots, col_inverse, col_roots = _compress_exact_ridge_factors(
        backend.asarray(X), backend,
    )
    _assert_native_factor_array(reduced, backend, device)
    reduced = backend.to_numpy(reduced)
    ri, rr = backend.to_numpy(row_inverse), backend.to_numpy(row_roots)
    ci, cr = backend.to_numpy(col_inverse), backend.to_numpy(col_roots)
    grouped_y = np.array([np.sum(y[ri == group]) / rr[group] for group in range(len(rr))])
    constant = np.sum(y ** 2) - np.sum(grouped_y ** 2)
    for theta in [np.array([.2, -.7]), np.array([-3., 2.]), np.zeros(2)]:
        beta = theta[ci] / cr[ci]
        assert_allclose(X @ beta, (reduced @ theta)[ri] / rr[ri], rtol=2e-15, atol=2e-15)
        assert_allclose(np.dot(beta, beta), np.dot(theta, theta), rtol=2e-15, atol=2e-15)
        for alpha in [1e-40, .1, 4.]:
            original_objective = np.sum((X @ beta - y) ** 2) + alpha * np.dot(beta, beta)
            reduced_objective = np.sum((reduced @ theta - grouped_y) ** 2) + alpha * np.dot(theta, theta)
            assert_allclose(original_objective, reduced_objective + constant, rtol=2e-15, atol=2e-14)


def test_exact_factor_compression_is_lazy(monkeypatch):
    import statgpu.linear_model.cv._ridge_cv as module
    from statgpu.backends import get_backend
    backend = get_backend(backend="numpy", device="cpu")
    def unexpected(*args):
        raise AssertionError("A resolved factor path must not inspect duplicates")
    monkeypatch.setattr(module, "_compress_exact_ridge_factors", unexpected)
    result = module._solve_ridge_path_from_factors(
        np.diag([1., 1e-30]), np.ones(2), np.array([1e-62]), backend,
    )
    assert_allclose(result[:, 0], [1., 1e-30 / (1e-60 + 1e-62)], rtol=2e-12, atol=0.)


@pytest.mark.parametrize("response", [
    [1e150, -1e150, 1.], [1e308, -1e308, 1.], [1e16 + 2., -1e16, 0.],
])
def test_exact_factor_recovery_rejects_unresolved_response_cancellation(factor_backend, response):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, _ = factor_backend
    X = backend.asarray(np.tile([1., 3., 7.], (3, 1)))
    y = backend.asarray(np.array(response))
    with pytest.raises(FloatingPointError, match="response projection.*not resolved"):
        _solve_ridge_path_from_factors(X, y, np.array([1e-40, .1]), backend)


def test_exact_factor_recovery_keeps_finite_large_response_projection(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    X = backend.asarray(np.tile([1., 3., 7.], (3, 1)))
    y = backend.asarray(np.array([1e308, 1e308, -1e308]))
    alpha = np.array([1e-40, .1])
    result = _solve_ridge_path_from_factors(X, y, alpha, backend)
    _assert_native_factor_array(result, backend, device, X)
    # The response sum would overflow if positives were aggregated first.
    # Expanding U projects the original response without that intermediate.
    expected = (np.array([1., 3., 7.])[:, None] / (177. + alpha)[None, :]) * 1e308
    assert_allclose(backend.to_numpy(result), expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("response", [np.zeros(3), np.full(3, 2.)])
def test_zero_and_noiseless_factor_responses_do_not_enter_duplicate_retry(monkeypatch, response):
    import statgpu.linear_model.cv._ridge_cv as module
    from statgpu.backends import get_backend
    def unexpected(*args):
        raise AssertionError("An already resolved response must not enter duplicate retry")
    monkeypatch.setattr(module, "_compress_exact_ridge_factors", unexpected)
    result = module._solve_ridge_path_from_factors(
        np.tile([1., 3., 7.], (3, 1)), response, np.array([1e-40, .1]),
        get_backend(backend="numpy", device="cpu"),
    )
    expected = (np.mean(response) * np.array([1., 3., 7.])[:, None]
                / (59. + np.array([1e-40, .1]) / 3.)[None, :])
    assert_allclose(result, expected, rtol=2e-12, atol=2e-15)


def test_exactly_cancelling_noisy_factor_response_keeps_precision_failure(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, _ = factor_backend
    # The original guard also rejected this noisy response. Exact factor
    # grouping must not reinterpret an unresolved response projection as zero.
    with pytest.raises(FloatingPointError, match="response projection.*not resolved"):
        _solve_ridge_path_from_factors(
            backend.asarray(np.tile([1., 3., 7.], (3, 1))),
            backend.asarray(np.array([1., -1., 0.])), np.array([1e-40]), backend,
        )


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("weight_kind", ["unit", "varying_with_zero"])
def test_generated_noisy_duplicate_column_folds_preserve_centering_and_selection(native, intercept, weight_kind):
    from statgpu.cross_validation._base import kfold_indices
    device, array = native
    x = np.linspace(-2., 2., 12)
    X = np.column_stack([x, x])
    y = np.tile([1., -1.], 6) + .1 * x + (2. if intercept else 0.)
    weights = np.ones(12) if weight_kind == "unit" else np.linspace(.2, 2., 12)
    if weight_kind == "varying_with_zero":
        weights[[0, 6]] = 0.
    folds = kfold_indices(12, 3, random_state=39)
    alphas = np.array([1e-40, .1, 10.])
    expected = []
    for alpha in alphas:
        losses = []
        for train, valid in folds:
            w = weights[train]
            xm = np.average(x[train], weights=w) if intercept else 0.
            ym = np.average(y[train], weights=w) if intercept else 0.
            gram = np.average((x[train] - xm) ** 2, weights=w)
            cross = np.average((x[train] - xm) * (y[train] - ym), weights=w)
            coefficient = cross / (2. * gram + alpha)
            predictions = 2. * coefficient * (x[valid] - xm) + ym
            losses.append(np.average((predictions - y[valid]) ** 2, weights=weights[valid]))
        expected.append(losses)
    expected = np.asarray(expected)
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        model = RidgeCV(
            alphas=alphas, cv=3, random_state=39, fit_intercept=intercept,
            device=device, compute_inference=False, gpu_cv_mixed_precision=False,
        ).fit(array(X), array(y), sample_weight=array(weights))
    assert_allclose(model.cv_results_["mse_path"], expected, rtol=2e-12, atol=2e-12)
    assert model.alpha_ == alphas[np.argmin(expected.mean(axis=1))] == 10.
    coef, offset = _solve(X, y, model.alpha_, weights, intercept)
    assert_allclose(model.coef_, coef, rtol=2e-10, atol=2e-10)
    assert_allclose(model.intercept_, offset, rtol=2e-10, atol=2e-10)
    assert_allclose(model.predict(array(X)), X @ coef + offset, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("exponent", [20, 49, 50, 51, 52])
def test_duplicate_retry_preserves_remaining_near_collinear_precision_boundary(factor_backend, exponent):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    delta = 2. ** -exponent
    core = np.array([[1., 1., 0.], [1., 1. + delta, delta]])
    X = np.repeat(core, 3, axis=0)
    X = np.column_stack([X, X[:, 0]])
    y = np.arange(len(X), dtype=float) % 3
    alpha = .01 * delta ** 2
    if exponent != 20:
        # The exact quotient still contains two distinct, nearly collinear
        # rows. Dropping its unresolved direction gives roughly [1/3,1/3,0,1/3]
        # instead of the true ridge slopes near [.40,.20,-.20,.40].
        with pytest.raises(FloatingPointError, match="factor-SVD precision.*singular direction"):
            _solve_ridge_path_from_factors(
                backend.asarray(X), backend.asarray(y), np.array([alpha, .1]), backend,
            )
        return
    actual = _solve_ridge_path_from_factors(
        backend.asarray(X), backend.asarray(y), np.array([alpha]), backend,
    )
    _assert_native_factor_array(actual, backend, device)
    # Each repeated row group has response mean 1. Solve the two-row dual
    # system symbolically, expanding its determinant without cancellation.
    t = alpha / 3.
    determinant = 5. * delta ** 2 + 6. * t + 2. * delta * t + 2. * delta ** 2 * t + t ** 2
    expected = np.array([2. * delta ** 2 + 2. * t,
                         delta ** 2 + 2. * t + delta * t,
                         delta * t - delta ** 2,
                         2. * delta ** 2 + 2. * t]) / determinant
    assert_allclose(backend.to_numpy(actual)[:, 0], expected, rtol=1e-8, atol=0.)


def test_exact_factor_inverse_maps_accept_numpy_200_axis_shape(monkeypatch):
    import statgpu.linear_model.cv._ridge_cv as module
    from statgpu.backends import get_backend
    original_unique = np.unique
    def numpy_200_unique(*args, **kwargs):
        values, inverse, counts = original_unique(*args, **kwargs)
        return values, inverse.reshape(-1, 1), counts
    monkeypatch.setattr(np, "unique", numpy_200_unique)
    direction = np.array([1., 1., 2.])
    alpha = np.array([1e-40, .1])
    result = module._solve_ridge_path_from_factors(
        np.tile(direction, (3, 1)), np.array([1., 2., 3.]), alpha,
        get_backend(backend="numpy", device="cpu"),
    )
    expected = 2. * direction[:, None] / (6. + alpha / 3.)[None, :]
    assert_allclose(result, expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("first,constant,counts", [
    pytest.param(2.209641945357916, 2., [2, 2], id="distinct-scaled-rows"),
    pytest.param(.8, .8, [2, 2], id="rounded-scaled-columns"),
])
def test_duplicate_retry_cannot_certify_original_nullspace_from_rounded_zero(
    factor_backend, first, constant, counts,
):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, _ = factor_backend
    second = np.nextafter(first, np.inf)
    X = np.repeat(np.array([[first, constant], [second, constant]]), counts, axis=0)
    y = np.concatenate([
        np.arange(count) - (count - 1.) / 2. + 1. for count in counts
    ])
    alpha = .01 * (second - first) ** 2
    assert first != second
    assert np.unique(X, axis=0).shape[0] == 2
    assert np.unique(X.T, axis=0).shape[0] == 2
    # Exact row grouping leaves two mathematically independent directions.
    # Native SVD can nevertheless return singular==0 and represented==0 after
    # sqrt(count) scaling. That is not proof of an original structural zero.
    with pytest.raises(FloatingPointError, match="factor-SVD precision.*singular direction"):
        _solve_ridge_path_from_factors(
            backend.asarray(X), backend.asarray(y), np.array([alpha]), backend,
        )


@pytest.mark.parametrize("response_scale,count,success", [
    pytest.param(1e-200, 3, True, id="normal-bound"),
    pytest.param(1e-300, 3, False, id="subnormal-bound"),
    pytest.param(1e-320, 3, False, id="zero-bound"),
    pytest.param(np.nextafter(0., 1.), 4, False, id="zero-absolute-dot"),
])
def test_duplicate_retry_certifies_nonzero_projection_bound(
    factor_backend, response_scale, count, success,
):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    direction = np.array([1., 3., 7.])
    X = backend.asarray(1e-150 * np.tile(direction, (count, 1)))
    response = np.zeros(count)
    response[0] = response_scale
    y = backend.asarray(response)
    penalty = np.array([np.nextafter(0., 1.)])
    if not success:
        with pytest.raises(FloatingPointError, match="response projection.*not resolved"):
            _solve_ridge_path_from_factors(X, y, penalty, backend)
        return
    actual = _solve_ridge_path_from_factors(X, y, penalty, backend)
    _assert_native_factor_array(actual, backend, device, X)
    expected = (response_scale / 1e-150) * direction / (count * 59.)
    assert_allclose(backend.to_numpy(actual)[:, 0], expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("large_penalty,success", [
    pytest.param(1e100, True, id="normal-gain"),
    pytest.param(1e123, False, id="subnormal-gain"),
    pytest.param(1e200, False, id="zero-gain"),
])
def test_duplicate_retry_certifies_mixed_grid_gains(factor_backend, large_penalty, success):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram_eig
    backend, device = factor_backend
    X = backend.asarray(np.array([[1., 1., 0.], [0., 0., 1e-200],
                                  [0., 0., 0.], [1., 1., 0.]]))
    y = backend.asarray(np.array([1., 1e200, 3., 2.]))
    penalty = np.array([1e-40, large_penalty])
    def solve():
        return _solve_ridge_path_gpu_from_gram_eig(
            (X.T @ X)[None], (X.T @ y)[None], penalty, backend,
            factor_factory=lambda fold: (X, y),
        )
    if not success:
        with pytest.raises(FloatingPointError, match="factor-SVD gains.*not resolved"):
            solve()
        return
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        actual = solve()
    _assert_native_factor_array(actual, backend, device, X)
    # The tiny independent coordinate has X'y=1 and negligible X'X.
    expected = np.column_stack([3. / (4. + penalty), 3. / (4. + penalty), 1. / penalty])
    assert_allclose(backend.to_numpy(actual)[:, 0], expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("stage", ["modal_product", "column_expansion"])
def test_factor_path_rejects_nonzero_product_or_expansion_underflow(factor_backend, stage):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, _ = factor_backend
    if stage == "modal_product":
        X = np.tile([1., 3., 7.], (3, 1))
        y = np.array([1., 2., 3.]) * 1e-200
        penalty = np.array([1e-40, 1e200])
        message = "coefficient signals.*not representable"
    else:
        X = np.ones((3, 16)) * 1e30
        y = np.array([1., 2., 3.]) * 1e-293
        penalty = np.array([1e-40])
        message = "coefficient expansion underflowed"
    with pytest.raises(FloatingPointError, match=message):
        _solve_ridge_path_from_factors(backend.asarray(X), backend.asarray(y), penalty, backend)


def test_duplicate_retry_retains_representable_subnormal_coefficient_signal(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    X = backend.asarray(np.array([[1., 1., 0.], [0., 0., 1e-100],
                                  [0., 0., 0.], [1., 1., 0.]]))
    y = backend.asarray(np.array([1e-150, 1e-250, 3e-150, 2e-150]))
    actual = _solve_ridge_path_from_factors(X, y, np.array([1e-40]), backend)
    _assert_native_factor_array(actual, backend, device, X)
    assert_allclose(backend.to_numpy(actual)[:, 0], [7.5e-151, 7.5e-151, 1e-310],
                    rtol=2e-12, atol=0.)


def test_factor_path_normalized_gain_preserves_finite_gram_overflow_boundary(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram_eig
    backend, device = factor_backend
    scale = 5.4737146662668906e153
    X = backend.asarray(np.tile([scale, scale], (3, 1)))
    y = backend.asarray(np.array([1., 2., 3.]))
    gram = X.T @ X
    assert np.all(np.isfinite(backend.to_numpy(gram)))
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        actual = _solve_ridge_path_gpu_from_gram_eig(
            gram[None], (X.T @ y)[None], np.array([1e-40]), backend,
            factor_factory=lambda fold: (X, y),
        )
    _assert_native_factor_array(actual, backend, device, X)
    # Exact rank-one ridge: each slope tends to mean(y)/(2*scale).
    assert_allclose(backend.to_numpy(actual)[0, 0], [1. / scale, 1. / scale],
                    rtol=2e-12, atol=0.)


def test_factor_path_penalty_dominant_gain_retains_subnormal_singular_bits(factor_backend):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_from_factors
    backend, device = factor_backend
    singular = 9.4e-323
    penalty = 3.196839587747213e-17
    X = backend.asarray(np.diag([1., singular]))
    actual = _solve_ridge_path_from_factors(
        X, backend.asarray(np.ones(2)), np.array([penalty]), backend,
    )
    _assert_native_factor_array(actual, backend, device, X)
    # The singular square is negligible. Dividing s by sqrt(penalty) first
    # would quantize a subnormal numerator although s/penalty is normal.
    assert_allclose(backend.to_numpy(actual)[:, 0], [1. / (1. + penalty), singular / penalty],
                    rtol=2e-12, atol=0.)


@pytest.mark.parametrize("case", ["projection", "subnormal_gain", "zero_gain"])
def test_public_duplicate_recovery_rejects_unreliable_arithmetic(native, case):
    device, array = native
    if case == "projection":
        tiny = np.nextafter(0., 1.)
        X = np.vstack([1e-150 * np.tile([1., 3., 7.], (3, 1)), [1e150, 0., 0.]])
        y = np.array([tiny, 0., 0., 0.])
        alphas = np.array([tiny])
        train, valid = np.arange(3), np.array([3])
        message = "response projection.*not resolved"
        # The true validation loss is ((tiny/1e-150)/177*1e150)**2.
        # Rounded subnormal projections previously inflated it by a factor 9.
    else:
        X = np.array([[1., 1., 0.], [0., 0., 1e-200], [0., 0., 0.],
                      [1., 1., 0.], [0., 0., 1e100]])
        y = np.array([1., 1e200, 3., 2., 0.])
        large = 1e123 if case == "subnormal_gain" else 1e200
        alphas = np.array([1e-40, large]) / 4.
        train, valid = np.arange(4), np.array([4])
        message = "factor-SVD gains.*not resolved"
        # The large-penalty validation loss is (1e100/large)**2, not zero.
    with pytest.raises(FloatingPointError, match=message):
        RidgeCV(
            alphas=alphas, cv_splits=[(train, valid)], fit_intercept=False,
            device=device, compute_inference=False, gpu_cv_mixed_precision=False,
        ).fit(array(X), array(y), sample_weight=array(np.ones(len(y))))


def _decimal_constant_ridge_path(feature, responses, alphas, n_features=1):
    """Evaluate the scalar rank-one objective without float64 products."""
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 100
        x = Decimal.from_float(float(feature))
        response_mean = sum(Decimal.from_float(float(y)) for y in responses) / len(responses)
        return np.array([
            float(x * response_mean / (n_features * x * x + Decimal.from_float(float(alpha))))
            for alpha in alphas
        ])


@pytest.mark.parametrize("feature,penalty", [(1e-162, np.nextafter(0., 1.)), (2e-161, 1e-322)])
@pytest.mark.parametrize("n_features", [1, 2])
def test_public_subnormal_weighted_gram_preserves_selection_and_refit(native, feature, penalty, n_features):
    device, array = native
    X = np.full((4, n_features), feature)
    y = np.array([1., 1., .14, .14])
    weights = np.full(4, 1e300)
    alphas = penalty * np.array([1., 2., 4.])
    expected_slopes = _decimal_constant_ridge_path(feature, y[:2], alphas, n_features)
    expected_predictions = n_features * feature * expected_slopes
    expected_mse = np.mean((expected_predictions[:, None] - y[None, 2:]) ** 2, axis=1)
    selected = alphas[np.argmin(expected_mse)]
    expected_final = _decimal_constant_ridge_path(feature, y, [selected], n_features)[0]
    # Raw weighted moments are normal and the final direct fit is computable.
    # Dividing the weights first used to make the CV Gram subnormal or zero,
    # silently changing its objective and (for the first case) selected alpha.
    assert np.all(np.isfinite((X * weights[:, None]).T @ X))
    with pytest.warns(RuntimeWarning, match="same-device float64 factor-SVD recovery"):
        model = RidgeCV(
            alphas=alphas, cv_splits=[([0, 1], [2, 3])], fit_intercept=False,
            device=device, compute_inference=False, gpu_cv_mixed_precision=False,
        ).fit(array(X), array(y), sample_weight=array(weights))
    assert_allclose(model.cv_results_["mse_path"][:, 0], expected_mse, rtol=2e-12, atol=0.)
    assert model.alpha_ == selected
    assert_allclose(model.coef_, np.full(n_features, expected_final), rtol=2e-12, atol=0.)
    assert model.intercept_ == 0.
    assert model._fitted and model.estimator_._fitted


@pytest.mark.parametrize("dtype_name,feature,penalty", [
    ("float32", 1e-23, float(np.nextafter(np.float32(0.), np.float32(1.)))),
    ("float32", 2e-22, 1e-44),
    ("float64", 1e-162, np.nextafter(0., 1.)),
    ("float64", 2e-161, 1e-322),
])
@pytest.mark.parametrize("n_features", [1, 2])
def test_native_subnormal_gram_uses_original_factor_precision(
    factor_backend, dtype_name, feature, penalty, n_features,
):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram_eig
    backend, device = factor_backend
    original_X = backend.asarray(np.full((2, n_features), feature / np.sqrt(2.)), dtype=backend.float64)
    original_y = backend.asarray(np.ones(2) / np.sqrt(2.), dtype=backend.float64)
    dtype = getattr(backend, dtype_name)
    narrowed_X = backend.asarray(original_X, dtype=dtype)
    narrowed_y = backend.asarray(original_y, dtype=dtype)
    gram = (narrowed_X.T @ narrowed_X)[None]
    cross = (narrowed_X.T @ narrowed_y)[None]
    # Use exact representable penalties at the Gram precision so this test
    # isolates lost Gram products rather than penalty-casting accuracy.
    alphas = backend.to_numpy(backend.asarray(penalty * np.array([1., 2., 4.]), dtype=dtype)).astype(np.float64)
    assert np.all(alphas > 0)
    gram_before, cross_before = backend.to_numpy(gram).copy(), backend.to_numpy(cross).copy()
    assert np.max(abs(gram_before)) < np.finfo(getattr(np, dtype_name)).tiny
    calls = []
    def factors(fold):
        calls.append(fold)
        return original_X, original_y
    with pytest.warns(RuntimeWarning, match="factor-SVD recovery"):
        actual = _solve_ridge_path_gpu_from_gram_eig(gram, cross, alphas, backend, factor_factory=factors)
    expected = _decimal_constant_ridge_path(feature, [1., 1.], alphas, n_features)
    _assert_native_factor_array(actual, backend, device, original_X)
    assert_allclose(backend.to_numpy(actual)[:, 0], np.repeat(expected[:, None], n_features, axis=1), rtol=2e-12, atol=0.)
    assert calls == [0]
    assert_array_equal(backend.to_numpy(gram), gram_before)
    assert_array_equal(backend.to_numpy(cross), cross_before)


@pytest.mark.parametrize("feature,penalty", [(1., .1), (1e-150, 1e-300), (1e-162, 1e-300)])
def test_resolved_gram_or_dominating_penalty_does_not_request_factor_recovery(factor_backend, feature, penalty):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram_eig
    backend, device = factor_backend
    X = backend.asarray(np.full((2, 1), feature / np.sqrt(2.)), dtype=backend.float64)
    y = backend.asarray(np.ones(2) / np.sqrt(2.), dtype=backend.float64)
    def unexpected(fold):
        raise AssertionError("Resolved system requested factor recovery")
    actual = _solve_ridge_path_gpu_from_gram_eig(
        (X.T @ X)[None], (X.T @ y)[None], np.array([penalty]), backend,
        factor_factory=unexpected,
    )
    expected = _decimal_constant_ridge_path(feature, [1., 1.], [penalty])
    _assert_native_factor_array(actual, backend, device, X)
    assert_allclose(backend.to_numpy(actual)[:, 0, 0], expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("dtype_name,feature,penalty", [
    ("float32", 1e-23, float(np.nextafter(np.float32(0.), np.float32(1.)))),
    ("float64", 1e-162, np.nextafter(0., 1.)),
])
def test_subnormal_recovery_is_fold_local_and_keeps_effective_penalty(factor_backend, dtype_name, feature, penalty):
    from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram_eig
    backend, device = factor_backend
    factors_X = [backend.asarray(np.ones((2, 1)), dtype=backend.float64),
                 backend.asarray(np.full((3, 1), feature), dtype=backend.float64)]
    factors_y = [backend.asarray(np.ones(2), dtype=backend.float64),
                 backend.asarray(np.ones(3), dtype=backend.float64)]
    dtype = getattr(backend, dtype_name)
    narrowed_X = [backend.asarray(x, dtype=dtype) for x in factors_X]
    narrowed_y = [backend.asarray(y, dtype=dtype) for y in factors_y]
    gram = backend.stack([x.T @ x for x in narrowed_X], axis=0)
    cross = backend.stack([x.T @ y for x, y in zip(narrowed_X, narrowed_y)], axis=0)
    alphas = np.array([penalty, 4 * penalty, 1.])
    calls = []
    def factors(fold):
        calls.append(fold)
        return factors_X[fold], factors_y[fold]
    with pytest.warns(RuntimeWarning, match=r"recovery for 1 fold\(s\)"):
        actual = _solve_ridge_path_gpu_from_gram_eig(
            gram, cross, alphas, backend, n_samples_vec=np.array([2., 3.]),
            factor_factory=factors,
        )
    expected = np.column_stack([
        _decimal_constant_ridge_path(1., [1., 1.], alphas),
        _decimal_constant_ridge_path(feature, [1., 1., 1.], alphas),
    ])
    assert calls == [1]
    _assert_native_factor_array(actual, backend, device, factors_X[0])
    assert_allclose(backend.to_numpy(actual)[:, :, 0], expected, rtol=2e-12, atol=0.)
