"""Whole-cluster bootstrap laws and backend ownership (GPU checks are guarded)."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from statgpu import LinearRegression
from statgpu.backends import get_backend
from statgpu.inference import bootstrap_statistic
from statgpu.inference import _resampling as resampling


def _reference_indices(labels, seed, n_resamples):
    """Independent fixed-G cluster sampling, retaining ordered complete groups."""
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    draws = np.random.default_rng(seed).integers(
        0, len(groups), size=(n_resamples, len(groups))
    )
    return [np.concatenate([groups[cid] for cid in draw]) for draw in draws]


@pytest.mark.parametrize("labels", [
    np.array([0, 0, 0, 1]),
    np.array([8, -2, 8, 8, 4, -2]),
    np.array(["b", "a", "b", "b"]),
    np.array([5, 5, 5]),
    np.array([4, 2, 4, 2]),
    np.array([9]),
])
def test_cluster_index_draws_match_exact_fixed_group_count(labels):
    state = resampling._prepare_bootstrap_state(
        len(labels), "cluster", None, labels, None, "numpy"
    )
    rng = np.random.default_rng(7)
    for expected in _reference_indices(labels, 7, 31):
        indices = resampling._build_bootstrap_indices(
            rng, len(labels), state, "numpy"
        )
        np.testing.assert_array_equal(indices, expected)
        assert indices.dtype == np.int64
        # Each original row appears as often as every row in its own cluster.
        multiplicities = np.bincount(indices, minlength=len(labels))
        cluster_draws = 0
        for label in np.unique(labels):
            counts = multiplicities[labels == label]
            assert np.all(counts == counts[0])
            cluster_draws += counts[0]
        assert cluster_draws == len(np.unique(labels))


def test_issue199_reproduction_has_no_partial_clusters():
    labels = np.array([0, 0, 0, 1])

    def statistic(rows):
        if rows.ndim != 1:
            raise ValueError("scalar-only callback")
        return float(np.sum(rows == 0))

    result = bootstrap_statistic(
        statistic, labels, strategy="cluster", clusters=labels,
        n_resamples=12, random_state=0, backend="numpy",
    )
    np.testing.assert_array_equal(result.samples, [0, 3, 6, 6, 3, 0, 0, 0, 0, 0, 3, 3])
    assert np.count_nonzero(result.samples % 3) == 0
    assert result.observed == 3
    assert result.metadata == {
        "n_samples": 4, "backend": "numpy", "resampling_unit": "cluster",
        "n_clusters": 2, "clusters_per_resample": 2, "variable_sample_size": True,
    }
    assert result.to_dict()["metadata"] == result.metadata


@pytest.mark.parametrize("use_estimator", [False, True])
def test_unequal_cluster_callbacks_receive_complete_aligned_arrays(use_estimator):
    labels = np.array([4, 8, 4, 4, 9, 8])
    x = np.column_stack([np.arange(6.0), np.arange(6.0) ** 2])
    y = 3 * x[:, 0] + 1
    calls = []

    def statistic(xs, ys):
        assert xs.ndim == 2 and ys.ndim == 1
        np.testing.assert_array_equal(ys, 3 * xs[:, 0] + 1)
        calls.append((xs.copy(), ys.copy()))
        return np.mean(xs[:, 1])

    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(
        statistic, x, y, clusters=labels, strategy="cluster",
        n_resamples=37, random_state=19, backend="numpy",
    )
    expected_indices = _reference_indices(labels, 19, 37)
    assert len(calls) == 38  # Original sample plus scalar calls, no ragged probes.
    assert len({len(indices) for indices in expected_indices}) > 1
    for (xs, ys), indices in zip(calls[1:], expected_indices):
        np.testing.assert_array_equal(xs, x[indices])
        np.testing.assert_array_equal(ys, y[indices])
    expected = np.array([x[indices, 1].mean() for indices in expected_indices])
    np.testing.assert_allclose(result.samples, expected)
    np.testing.assert_allclose(result.confidence_interval, np.quantile(expected, [0.025, 0.975]))


@pytest.mark.parametrize("force_vectorized", [False, True])
def test_unequal_cluster_mean_hint_uses_actual_rows_only(force_vectorized):
    labels = np.array([0, 0, 0, 1])
    x = np.array([2.0, 3.0, 7.0, 23.0])
    calls = []

    def observed_only(values):
        calls.append(values.shape)
        assert len(calls) == 1
        return np.mean(values)

    result = bootstrap_statistic(
        observed_only, x, clusters=labels, strategy="cluster", n_resamples=50,
        random_state=0, backend="numpy", statistic_hint="mean",
        force_vectorized=force_vectorized,
    )
    expected = [x[idx].mean() for idx in _reference_indices(labels, 0, 50)]
    np.testing.assert_allclose(result.samples, expected)
    assert calls == [(4,)]
    assert set(result.samples) == {4.0, 8.75, 23.0}


@pytest.mark.parametrize("shape", [(4, 1), (4, 2)])
def test_cluster_mean_hint_rejects_non_vector_inputs(shape):
    with pytest.raises(ValueError, match="single one-dimensional input array"):
        bootstrap_statistic(
            np.mean, np.ones(shape), clusters=[0, 0, 0, 1], strategy="cluster",
            n_resamples=5, backend="numpy", statistic_hint="mean",
        )


def test_ragged_force_vectorized_rejects_before_any_resample_callback():
    calls = []

    def statistic(values):
        calls.append(values.shape)
        return values.mean(axis=-1)

    with pytest.raises(ValueError, match="force_vectorized=True requires equal-size clusters"):
        bootstrap_statistic(
            statistic, np.arange(4.0), clusters=[0, 0, 0, 1], strategy="cluster",
            n_resamples=5, backend="numpy", force_vectorized=True,
        )
    assert calls == [(4,)]


def test_ragged_default_calls_vectorizable_statistic_without_padding_or_batch_axis():
    shapes = []

    def statistic(values):
        shapes.append(values.shape)
        return values.mean(axis=-1)

    result = bootstrap_statistic(
        statistic, np.arange(4.0), clusters=[0, 0, 0, 1], strategy="cluster",
        n_resamples=12, random_state=0, backend="numpy",
    )
    assert all(len(shape) == 1 for shape in shapes)
    assert {shape[0] for shape in shapes} == {2, 4, 6}
    assert np.isfinite(result.samples).all()


@pytest.mark.parametrize("mode", ["scalar", "vectorized", "mean_hint"])
def test_equal_size_seed_and_final_batch_contract_is_preserved(mode):
    labels = np.array([4, 2, 4, 2, 9, 9])
    x = np.array([1.0, 3.0, 2.0, 9.0, -1.0, 7.0])
    shapes = []

    def statistic(values):
        shapes.append(values.shape)
        if mode != "vectorized" and values.ndim != 1:
            raise ValueError("scalar-only callback")
        return values.mean(axis=-1)

    result = bootstrap_statistic(
        statistic, x, clusters=labels, strategy="cluster", n_resamples=513,
        random_state=21, backend="numpy", force_vectorized=mode == "vectorized",
        statistic_hint="mean" if mode == "mean_hint" else None,
    )
    expected = [x[idx].mean() for idx in _reference_indices(labels, 21, 513)]
    np.testing.assert_allclose(result.samples, expected)
    assert result.metadata["variable_sample_size"] is False
    assert result.metadata["clusters_per_resample"] == 3
    if mode == "vectorized":
        assert shapes == [(6,), (512, 6), (1, 6)]
    elif mode == "mean_hint":
        assert shapes == [(6,)]


def test_cluster_seed_is_local_reproducible_and_resample_count_prefix_stable():
    options = dict(strategy="cluster", clusters=[0, 0, 0, 1], random_state=42,
                   backend="numpy", statistic_hint="mean")
    x = np.arange(4.0)
    before = np.random.get_state()
    first = bootstrap_statistic(np.mean, x, n_resamples=25, **options)
    after = np.random.get_state()
    for lhs, rhs in zip(before, after):
        np.testing.assert_equal(lhs, rhs)
    second = bootstrap_statistic(np.mean, x, n_resamples=25, **options)
    longer = bootstrap_statistic(np.mean, x, n_resamples=50, **options)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.samples, longer.samples[:25])


@pytest.mark.parametrize("labels", [[0, 0, np.nan, 1], [np.nan] * 4])
def test_cluster_missing_labels_are_rejected_without_empty_group_draws(labels):
    with pytest.raises(ValueError, match="nonmissing label per row"):
        bootstrap_statistic(
            np.mean, np.arange(4.0), strategy="cluster", clusters=labels,
            n_resamples=5, backend="numpy",
        )


@pytest.mark.parametrize("missing", [
    None, np.nan, np.float32("nan"), np.datetime64("NaT"), np.timedelta64("NaT"),
])
@pytest.mark.parametrize("mixed", [False, True])
def test_missing_object_cluster_labels_raise_before_grouping(missing, mixed, monkeypatch):
    labels = np.array([0, 0, missing, 1] if mixed else [missing], dtype=object)

    def must_not_group(_labels):
        raise AssertionError("missing object labels must be rejected before grouping")

    monkeypatch.setattr(np, "unique", must_not_group)
    with pytest.raises(ValueError, match="nonmissing label per row"):
        bootstrap_statistic(
            np.mean, np.arange(float(len(labels))), strategy="cluster",
            clusters=labels, n_resamples=5, backend="numpy",
        )


@pytest.mark.parametrize("labels", [
    np.array([0, 0, 0, 1], dtype=object),
    np.array(["b", "a", "b", "b"], dtype=object),
    np.array(["None", "None", "None", "NaT"], dtype=object),
])
def test_nonmissing_object_cluster_labels_preserve_whole_group_sampling(labels):
    x = np.arange(4.0)
    result = bootstrap_statistic(
        np.mean, x, strategy="cluster", clusters=labels,
        n_resamples=12, random_state=0, backend="numpy",
    )
    expected = [x[indices].mean() for indices in _reference_indices(labels, 0, 12)]
    np.testing.assert_allclose(result.samples, expected)


@pytest.mark.parametrize("sentinel", ["NA", "NaT"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("from_series", [False, True])
def test_pandas_missing_cluster_labels_raise_value_error(sentinel, mixed, from_series):
    pd = pytest.importorskip("pandas")
    missing = getattr(pd, sentinel)
    labels = [0, 0, missing, 1] if mixed else [missing]
    if from_series:
        labels = pd.Series(labels, dtype=object).to_numpy()
    with pytest.raises(ValueError, match="nonmissing label per row"):
        bootstrap_statistic(
            np.mean, np.arange(float(len(labels))), strategy="cluster",
            clusters=labels, n_resamples=5, backend="numpy",
        )


@pytest.mark.parametrize("dtype, values", [
    ("Int64", [0, 0, 0, 1]),
    ("string", ["b", "a", "b", "b"]),
])
def test_nonmissing_pandas_nullable_label_arrays_preserve_cluster_sampling(dtype, values):
    pd = pytest.importorskip("pandas")
    labels = pd.Series(values, dtype=dtype).to_numpy()
    x = np.arange(4.0)
    result = bootstrap_statistic(
        np.mean, x, strategy="cluster", clusters=labels,
        n_resamples=12, random_state=0, backend="numpy",
    )
    expected = [x[indices].mean() for indices in _reference_indices(np.asarray(values), 0, 12)]
    np.testing.assert_allclose(result.samples, expected)


def test_unorderable_object_cluster_labels_raise_clear_value_error():
    labels = np.array([0, "a", 0, "b"], dtype=object)
    with pytest.raises(ValueError, match="labels must be mutually comparable"):
        bootstrap_statistic(
            np.mean, np.arange(4.0), strategy="cluster", clusters=labels,
            n_resamples=5, backend="numpy",
        )


def test_unequal_cluster_index_sampling_never_converts_draws_to_numpy(monkeypatch):
    def no_host_transfer(_value):
        raise AssertionError("cluster resampling must not transfer drawn indices to host")

    monkeypatch.setattr(resampling, "_to_numpy", no_host_transfer)
    result = bootstrap_statistic(
        np.mean, np.arange(4.0), strategy="cluster", clusters=[0, 0, 0, 1],
        n_resamples=10, random_state=0, backend="numpy",
    )
    assert result.samples.shape == (10,)


@pytest.mark.parametrize("backend_name", ["cupy", "torch"])
def test_explicit_backend_failure_is_not_retried_with_numpy(monkeypatch, backend_name):
    calls = []

    def unavailable_backend(name):
        calls.append(name)
        raise RuntimeError("requested backend unavailable")

    monkeypatch.setattr(resampling, "get_backend", unavailable_backend)
    with pytest.raises(RuntimeError, match="requested backend unavailable"):
        bootstrap_statistic(
            np.mean, np.arange(4.0), strategy="cluster", clusters=[0, 0, 0, 1],
            backend=backend_name,
        )
    assert calls == [backend_name]


@pytest.fixture(params=["numpy", "torch_cpu", "cupy_cuda", "torch_cuda"])
def cluster_backend(request, monkeypatch):
    name = request.param
    if name == "numpy":
        yield get_backend("numpy"), nullcontext(), "cpu"
        return
    if name == "cupy_cuda":
        request.getfixturevalue("cupy_available")
        import cupy as cp
        ordinal = cp.cuda.runtime.getDeviceCount() - 1
        with cp.cuda.Device(ordinal):
            yield get_backend("cupy"), nullcontext(), ordinal
        return

    if name == "torch_cuda":
        request.getfixturevalue("torch_cuda_available")
    torch = pytest.importorskip("torch")
    from statgpu.backends._torch import TorchBackend
    device = f"cuda:{torch.cuda.device_count() - 1}" if name == "torch_cuda" else "cpu"
    selected = TorchBackend(device=device)
    original_get_backend = resampling.get_backend
    # Explicit CPU Torch coverage is separate from physical CUDA verification.
    monkeypatch.setattr(
        resampling, "get_backend",
        lambda backend_name: selected if backend_name == "torch" else original_get_backend(backend_name),
    )
    yield selected, nullcontext(), torch.device(device)


@pytest.mark.parametrize("mean_hint", [False, True])
@pytest.mark.parametrize("uniform", [False, True])
def test_backend_native_clusters_and_seed_replay(cluster_backend, mean_hint, uniform, monkeypatch):
    backend, context, device = cluster_backend
    labels_np = np.array([0, 1, 0, 2 if uniform else 0, 2, 1])
    groups = [np.flatnonzero(labels_np == label) for label in np.unique(labels_np)]
    with context:
        labels = backend.asarray(labels_np)
        x = backend.asarray(np.arange(6.0), dtype=backend.float64)
        y = x * 3 + 1
        calls = []

        def assert_ownership(values):
            if backend.name == "torch":
                assert values.device == device
            elif backend.name == "cupy":
                assert values.device.id == device
            else:
                assert isinstance(values, np.ndarray)

        def statistic(values, aligned=None):
            assert_ownership(values)
            assert values.ndim == 1
            if aligned is not None:
                assert_ownership(aligned)
                assert not bool(backend.xp.any(aligned != values * 3 + 1))
            calls.append(int(values.shape[0]))
            return backend.xp.mean(values)

        def no_host_transfer(_value):
            raise AssertionError("no sampled-index host transfer is permitted")

        monkeypatch.setattr(resampling, "_to_numpy", no_host_transfer)
        options = dict(strategy="cluster", clusters=labels, n_resamples=23,
                       random_state=8, backend=backend.name)
        if mean_hint:
            options.update(statistic_hint="mean", force_vectorized=True)
        args = (x,) if mean_hint else (x, y)
        first = bootstrap_statistic(statistic, *args, **options)
        assert_ownership(first.samples)
        assert first.samples.dtype == backend.float64
        assert first.metadata["backend"] == backend.name
        assert first.metadata["clusters_per_resample"] == 3
        assert len(calls) == (1 if mean_hint else 24)
        rng_device = str(device) if backend.name == "torch" else "cuda"
        rng = resampling._rng_default(backend.name, 8, device=rng_device)
        expected = []
        if uniform:
            draws = resampling._rng_integers(rng, 0, 3, (23, 3), backend.name, device=rng_device)
        else:
            draws = [resampling._rng_integers(rng, 0, 3, (3,), backend.name, device=rng_device)
                     for _ in range(23)]
        for ids in draws:
            ids_np = backend.to_numpy(ids)
            indices = np.concatenate([groups[cid] for cid in ids_np])
            expected.append(np.arange(6.0)[indices].mean())
        np.testing.assert_allclose(backend.to_numpy(first.samples), expected)
        second = bootstrap_statistic(statistic, *args, **options)
        np.testing.assert_array_equal(backend.to_numpy(first.samples), backend.to_numpy(second.samples))


def test_all_exported_estimator_helpers_share_the_cluster_dispatch():
    import inspect
    import statgpu
    from statgpu._base import BaseEstimator

    consumers = [getattr(statgpu, name) for name in statgpu.__all__
                 if inspect.isclass(getattr(statgpu, name))
                 and issubclass(getattr(statgpu, name), BaseEstimator)]
    assert len(consumers) >= 60
    for cls in consumers:
        assert cls.bootstrap_statistic is BaseEstimator.bootstrap_statistic


def test_ragged_callback_runtime_failure_propagates_without_replacement():
    calls = []

    def statistic(values):
        calls.append(values.shape)
        if len(calls) > 1:
            raise RuntimeError("statistic execution failed")
        return np.mean(values)

    with pytest.raises(RuntimeError, match="statistic execution failed"):
        bootstrap_statistic(
            statistic, np.arange(4.0), strategy="cluster", clusters=[0, 0, 0, 1],
            n_resamples=10, random_state=0, backend="numpy",
        )
    assert len(calls) == 2


def test_ragged_nonfinite_callback_rejects_inference():
    calls = []

    def statistic(values):
        calls.append(values.shape)
        return np.mean(values) if len(calls) == 1 else np.nan

    with pytest.raises(ValueError, match="non-finite value for a bootstrap resample"):
        bootstrap_statistic(
            statistic, np.arange(4.0), strategy="cluster", clusters=[0, 0, 0, 1],
            n_resamples=10, random_state=0, backend="numpy",
        )
