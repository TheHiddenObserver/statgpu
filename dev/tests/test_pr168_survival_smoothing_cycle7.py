"""Fresh cycle-7 KernelPCA lifecycle checks and user guidance, on NumPy CPU."""

import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statgpu.nonparametric.kernel_methods import KernelPCA

ROOT = Path(__file__).resolve().parents[2]


class _KernelPCAMixedRefitState(Exception):
    """Only the reproduced old-projection/new-centering state is expected."""


def _data():
    return np.arange(6.)[:, None], np.full((6, 1), 20.), np.array([[.5], [2.], [4.]])


def _model(alpha=1.):
    return KernelPCA(kernel="linear", n_components=2, alpha=alpha, device="cpu")


def _check_refit_contract(alpha, fit_method):
    X, rejected, query = _data()
    model = _model(alpha).fit(X)
    before = model.transform(query).copy()
    assert before.shape == (3, 1)
    assert np.isfinite(before).all()
    # Linear-kernel PCA has the ordinary centered one-dimensional Gram matrix.
    centered_query = query - X.mean(axis=0)
    assert_allclose(before @ before.T, centered_query @ centered_query.T, atol=1e-13)
    old_projection = model.alphas_.copy()
    old_eigenvalues = model.lambdas_.copy()
    old_column_means = model._K_train_col_means_.copy()
    old_mean = model._K_train_mean_
    assert_allclose(old_column_means, (X @ X.T).mean(axis=0))
    assert old_mean == 6.25

    with pytest.raises(ValueError, match="centered kernel matrix has no positive eigenvalues"):
        getattr(model, fit_method)(rejected)
    if not model._fitted:
        # Invalidation is an acceptable future repair, as is full rollback.
        with pytest.raises(RuntimeError, match="not fitted"):
            model.transform(query)
        return
    after = model.transform(query)
    assert after.shape == before.shape
    assert np.isfinite(after).all()
    if np.allclose(after, before, rtol=1e-12, atol=1e-12):
        assert_array_equal(model.X_fit_, X)
        assert_allclose(model.alphas_, old_projection)
        assert_allclose(model.lambdas_, old_eigenvalues)
        assert_allclose(model._K_train_col_means_, old_column_means)
        assert model._K_train_mean_ == old_mean
        return

    # Match the complete known corruption, not just any numerical disagreement.
    assert model._fitted is True
    assert_array_equal(model.X_fit_, X)
    assert_array_equal(model.alphas_, old_projection)
    assert_array_equal(model.lambdas_, old_eigenvalues)
    assert_array_equal(model._K_train_col_means_, np.full(6, 400.))
    assert model._K_train_mean_ == 400.
    orientation = (before[1, 0] - before[0, 0]) / (query[1, 0] - query[0, 0])
    assert_allclose(abs(orientation), 1., atol=1e-14)
    assert_allclose(after, orientation * query, atol=1e-13)
    assert_allclose(model.predict(query), after, atol=1e-13)
    raise _KernelPCAMixedRefitState("failed refit retained projection but replaced kernel centering")


@pytest.mark.xfail(strict=True, raises=_KernelPCAMixedRefitState,
                   reason="Issue #236: KernelPCA failed-refit mixed state")
@pytest.mark.parametrize("alpha", [0., 1.])
@pytest.mark.parametrize("fit_method", ["fit", "fit_transform"])
def test_failed_kernel_pca_refit_must_preserve_or_invalidate_complete_state(alpha, fit_method):
    _check_refit_contract(alpha, fit_method)


@pytest.mark.parametrize("alpha", [0., 1.])
def test_new_instance_rejects_degenerate_data_and_valid_fresh_fit_recovers(alpha):
    X, rejected, query = _data()
    fresh = _model(alpha)
    with pytest.raises(ValueError, match="centered kernel matrix has no positive eigenvalues"):
        fresh.fit(rejected)
    assert fresh._fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        fresh.transform(query)
    valid = _model(alpha).fit(X)
    transformed = valid.transform(query)
    centered = query - X.mean(axis=0)
    assert_allclose(transformed @ transformed.T, centered @ centered.T, atol=1e-13)
    assert_allclose(valid.predict(query), transformed)


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
def test_refit_expected_failure_does_not_mask_other_output_failures(monkeypatch, kind):
    original = KernelPCA.transform

    def altered(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if self._K_train_mean_ != 400.:
            return output
        if kind == "exception":
            raise RuntimeError("unrelated transform failure")
        if kind == "shape":
            return output[:-1]
        if kind == "nonfinite":
            output = output.copy()
            output[0] = np.nan
            return output
        return output + .25

    monkeypatch.setattr(KernelPCA, "transform", altered)
    expected = RuntimeError if kind == "exception" else AssertionError
    with pytest.raises(expected):
        _check_refit_contract(1., "fit")


def test_full_rollback_repair_would_remove_expected_failure(monkeypatch):
    original = KernelPCA.fit

    def transactional(self, *args, **kwargs):
        previous = deepcopy(self.__dict__)
        try:
            return original(self, *args, **kwargs)
        except ValueError:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

    monkeypatch.setattr(KernelPCA, "fit", transactional)
    # No sentinel is raised; a repaired production path would XPASS the strict
    # marker, requiring the obsolete expected-failure marker to be removed.
    _check_refit_contract(1., "fit")


@pytest.mark.parametrize("language", ["en", "cn"])
def test_kernel_pca_failed_refit_guidance_matches_installed_help(language):
    text = (ROOT / f"docs/{language}/models/kernel-methods.md").read_text()
    if language == "en":
        for phrase in ["After a failed refit", "finite but incorrect", "discard that instance",
                       "not a way", "fit_transform"]:
            assert phrase in text
    else:
        for phrase in ["重新拟合失败后", "有限但错误", "丢弃该实例", "零秩核", "fit_transform"]:
            assert phrase in text
    assert "failed refit" in inspect.getdoc(KernelPCA)
    assert "finite but incorrect" in inspect.getdoc(KernelPCA)
    assert "discard this instance" in inspect.getdoc(KernelPCA.fit)
