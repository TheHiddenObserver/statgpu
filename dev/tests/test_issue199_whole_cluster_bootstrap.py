"""Whole-cluster bootstrap laws and backend ownership (GPU checks are guarded)."""

from __future__ import annotations

from array import array
from collections import UserList, UserString, deque
from collections.abc import Sequence
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


# Mixed host labels must be checked before any backend conversion can stringify
# a missing scalar. The estimator entry has a separate conversion boundary.
# Estimator runtime finite-input validation may reject a missing scalar even
# earlier than the resampling-specific guard; both errors have the same public
# ValueError contract. Do not require bypassing that existing validation.
class _ClusterLabelSequence(Sequence):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _LegacyClusterLabelSequence:
    # NumPy accepts the sequence protocol without Sequence ABC registration.
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


_LABEL_CONTAINERS = [
    pytest.param(list, id="list"),
    pytest.param(tuple, id="tuple"),
    pytest.param(deque, id="deque"),
    pytest.param(UserList, id="UserList"),
    pytest.param(_ClusterLabelSequence, id="Sequence"),
    pytest.param(_LegacyClusterLabelSequence, id="legacy-sequence"),
    pytest.param(lambda x: np.asarray(x, dtype=object), id="object-array"),
]


@pytest.mark.parametrize("container", _LABEL_CONTAINERS)
@pytest.mark.parametrize("missing", [
    pytest.param(None, id="None"),
    pytest.param(np.nan, id="nan"),
    pytest.param(np.float32("nan"), id="float32-nan"),
    pytest.param(np.datetime64("NaT"), id="datetime-NaT"),
    pytest.param(np.timedelta64("NaT"), id="timedelta-NaT"),
])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_mixed_string_missing_clusters_rejected_before_conversion(
    container, missing, use_estimator, backend,
):
    labels = container(["a", "a", missing, "b"])
    calls = []

    def statistic(values):
        calls.append(values.shape)
        return values.mean(axis=-1)

    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    with pytest.raises(ValueError, match=("nonmissing label per row|must contain only finite values"
                                          if use_estimator else "nonmissing label per row")):
        function(statistic, np.array([1., 2., 100., 5.]), strategy="cluster",
                 clusters=labels, n_resamples=9, random_state=4, backend=backend)
    assert calls == []


@pytest.mark.parametrize("container", _LABEL_CONTAINERS)
@pytest.mark.parametrize("sentinel", ["NA", "NaT"])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_mixed_string_pandas_missing_clusters_rejected(
    container, sentinel, use_estimator, backend,
):
    pd = pytest.importorskip("pandas")
    labels = container(["a", "a", getattr(pd, sentinel), "b"])
    calls = []

    def statistic(values):
        calls.append(values.shape)
        return values.mean(axis=-1)

    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    with pytest.raises(ValueError, match=("nonmissing label per row|must contain only finite values"
                                          if use_estimator else "nonmissing label per row")):
        function(statistic, np.arange(4.), strategy="cluster", clusters=labels,
                 n_resamples=9, random_state=4, backend=backend)
    assert calls == []


@pytest.mark.parametrize("container", _LABEL_CONTAINERS + [
    pytest.param(np.asarray, id="string-array"),
])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_literal_missing_names_remain_valid_cluster_ids(container, use_estimator, backend):
    original = ["nan", "nan", "None", "NaT"]
    labels = container(original)
    values = np.array([1., 2., 100., 5.])
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, values, strategy="cluster", clusters=labels,
                      n_resamples=29, random_state=4, backend=backend)
    expected = np.array([values[idx].mean() for idx in
                         _reference_indices(np.asarray(original), 4, 29)])
    np.testing.assert_array_equal(result.samples, expected)
    np.testing.assert_array_equal(labels, original)
    assert result.metadata["n_clusters"] == 3
    np.testing.assert_array_equal(result.confidence_interval,
                                  np.quantile(expected, [.025, .975]))


class _ObjectClusterArrayLike:
    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=object)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.labels, dtype=dtype)


@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("missing", [None, np.nan])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_generic_object_array_like_retains_missing_cluster_rejection(
    size, missing, use_estimator, backend,
):
    # Existing post-conversion validation remains necessary for generic
    # array-likes; the new preconversion guard does not inspect these objects.
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    with pytest.raises(ValueError, match=("nonmissing label per row|must contain only finite values"
                                          if use_estimator else "nonmissing label per row")):
        function(np.mean, np.arange(float(size)), strategy="cluster",
                 clusters=_ObjectClusterArrayLike([missing] * size),
                 n_resamples=9, random_state=4, backend=backend)


@pytest.mark.parametrize("make", [
    pytest.param(lambda: range(4), id="range"),
    pytest.param(lambda: array("q", [2, 2, 9, 9]), id="array-int64"),
    pytest.param(lambda: array("d", [2., 2., 9., 9.]), id="array-float64"),
    pytest.param(lambda: memoryview(array("q", [2, 2, 9, 9])), id="memoryview"),
])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_nonmissing_numeric_host_sequences_preserve_cluster_sampling(
    make, use_estimator, backend,
):
    labels = make()
    before = np.asarray(labels).copy()
    values = np.array([1., 2., 100., 5.])
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, values, strategy="cluster", clusters=labels,
                      n_resamples=29, random_state=4, backend=backend)
    expected = np.array([values[idx].mean() for idx in
                         _reference_indices(before, 4, 29)])
    np.testing.assert_array_equal(result.samples, expected)
    np.testing.assert_array_equal(labels, before)
    assert np.asarray(labels).dtype == before.dtype
    assert result.samples.dtype == np.float64


@pytest.mark.parametrize("container", [deque, UserList, _LegacyClusterLabelSequence])
@pytest.mark.parametrize("shape", ["vector", "column", "two-columns"])
def test_host_cluster_sequence_shape_contract_is_unchanged(container, shape):
    original = ["a", "a", "b", "b"]
    values = {"vector": original, "column": [[x] for x in original],
              "two-columns": [[x, x] for x in original]}[shape]
    labels = container(values)
    if shape == "two-columns":
        with pytest.raises(ValueError, match="clusters must have the same length"):
            bootstrap_statistic(np.mean, np.arange(4.), clusters=labels,
                                strategy="cluster", n_resamples=5, backend="numpy")
        return
    result = bootstrap_statistic(np.mean, np.arange(4.), clusters=labels,
                                 strategy="cluster", n_resamples=29,
                                 random_state=4, backend="numpy")
    expected = [np.arange(4.)[idx].mean()
                for idx in _reference_indices(np.asarray(original), 4, 29)]
    np.testing.assert_array_equal(result.samples, expected)
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(values))


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("shape", [(4,), (4, 1)])
def test_readonly_object_cluster_labels_are_not_modified(missing, shape):
    labels = np.asarray(["a", "a", None if missing else "b", "b"],
                        dtype=object).reshape(shape)
    labels.setflags(write=False)
    before = labels.copy()
    if missing:
        with pytest.raises(ValueError, match="nonmissing label per row"):
            bootstrap_statistic(np.mean, np.arange(4.), clusters=labels,
                                strategy="cluster", n_resamples=5, backend="numpy")
    else:
        result = bootstrap_statistic(np.mean, np.arange(4.), clusters=labels,
                                     strategy="cluster", n_resamples=29,
                                     random_state=4, backend="numpy")
        expected = [np.arange(4.)[idx].mean() for idx in
                    _reference_indices(before.reshape(-1), 4, 29)]
        np.testing.assert_array_equal(result.samples, expected)
    assert not labels.flags.writeable
    np.testing.assert_array_equal(labels, before)


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_user_string_does_not_gain_character_sequence_support(use_estimator, backend):
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    with pytest.raises((IndexError, TypeError, ValueError)):
        function(np.mean, np.arange(4.), clusters=UserString("aabb"),
                 strategy="cluster", n_resamples=5, backend=backend)


def test_cluster_prevalidation_does_not_consume_iterators_or_mappings():
    visited = []

    def labels():
        for label in ["a", "a", np.nan, "b"]:
            visited.append(label)
            yield label

    iterator = labels()
    resampling._validate_cluster_labels_before_cast(iterator)
    assert visited == []
    assert next(iterator) == "a"

    class MappingLabels(dict):
        def __getitem__(self, key):
            raise AssertionError("a mapping is not a host label sequence")

        def __iter__(self):
            raise AssertionError("a mapping must not be iterated")

    resampling._validate_cluster_labels_before_cast(MappingLabels({0: np.nan}))


@pytest.mark.parametrize("sequence_base", [Sequence, object])
@pytest.mark.parametrize("protocol", [
    "__array__", "__array_interface__", "__array_struct__",
    "__cuda_array_interface__", "__dlpack__", "__dlpack_device__",
    "__array_namespace__",
])
def test_cluster_prevalidation_does_not_execute_array_protocols(sequence_base, protocol):
    def forbidden(_self, *_args):
        raise AssertionError("prevalidation must not access array protocols or rows")

    cls = type("ArrayProtocolLabels", (sequence_base,), {
        "__len__": forbidden, "__getitem__": forbidden,
        protocol: property(forbidden),
    })
    resampling._validate_cluster_labels_before_cast(cls())


class _ProtocolClusterLabelSequence(Sequence):
    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=object)
        self.conversions = 0

    def __len__(self):
        raise AssertionError("array protocol owns label conversion")

    def __getitem__(self, index):
        raise AssertionError("array protocol owns label conversion")

    def __array__(self, dtype=None, copy=None):
        self.conversions += 1
        return np.asarray(self.labels, dtype=dtype)


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
@pytest.mark.parametrize("missing", [False, True])
def test_sequence_array_protocol_retains_normal_conversion_and_object_validation(
    use_estimator, backend, missing,
):
    labels = _ProtocolClusterLabelSequence(
        ["nan", "nan", None if missing else "None", "NaT"]
    )
    resampling._validate_cluster_labels_before_cast(labels)
    assert labels.conversions == 0
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    values = np.array([1., 2., 100., 5.])
    if missing:
        with pytest.raises(ValueError, match="nonmissing label per row"):
            function(np.mean, values, clusters=labels, strategy="cluster",
                     n_resamples=29, random_state=4, backend=backend)
    else:
        result = function(np.mean, values, clusters=labels, strategy="cluster",
                          n_resamples=29, random_state=4, backend=backend)
        expected = [values[idx].mean() for idx in
                    _reference_indices(labels.labels, 4, 29)]
        np.testing.assert_array_equal(result.samples, expected)
    # Existing entrypoint conversions only; the precheck adds no __array__ call.
    expected_conversions = (3 if backend == "auto" else 2) if use_estimator else 2
    assert labels.conversions == expected_conversions


def test_cluster_prevalidation_leaves_backend_native_arrays_on_device(cluster_backend, monkeypatch):
    backend, context, device = cluster_backend
    with context:
        labels = backend.asarray([1., 1., float("nan"), 2.])

        def forbidden(*_args, **_kwargs):
            raise AssertionError("native labels must not be copied to NumPy")

        monkeypatch.setattr(np, "asarray", forbidden)
        resampling._validate_cluster_labels_before_cast(labels)
        if backend.name == "torch":
            assert labels.device == device
        elif backend.name == "cupy":
            assert labels.device.id == device


@pytest.mark.parametrize("container", [
    pytest.param(lambda values: array("d", values), id="array"),
    pytest.param(lambda values: memoryview(array("d", values)), id="memoryview"),
])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_numeric_host_sequence_missing_labels_reject_before_callback(
    container, use_estimator, backend,
):
    labels = container([1., 1., np.nan, 2.])
    calls = []

    def statistic(values):
        calls.append(values.shape)
        return np.mean(values)

    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    with pytest.raises(ValueError, match=("nonmissing label per row|must contain only finite values"
                                          if use_estimator else "nonmissing label per row")):
        function(statistic, np.arange(4.), strategy="cluster", clusters=labels,
                 n_resamples=5, backend=backend)
    assert calls == []


def test_instance_array_protocol_is_not_invoked_by_cluster_prevalidation():
    labels = _LegacyClusterLabelSequence(["a", "a", np.nan, "b"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prevalidation must not invoke a custom __array__")

    labels.__array__ = forbidden
    resampling._validate_cluster_labels_before_cast(labels)


class _DynamicClusterArrayBase(_LegacyClusterLabelSequence):
    def __init__(self, values, reject_object_dtype):
        super().__init__(values)
        self.reject_object_dtype = reject_object_dtype
        self.conversions = []

    def _convert(self, dtype=None, copy=None):
        self.conversions.append(dtype)
        if self.reject_object_dtype and dtype is not None:
            raise TypeError("custom converter does not support object dtype")
        return np.asarray(self.values, dtype=object)


class _GetattrClusterArray(_DynamicClusterArrayBase):
    def __getattr__(self, name):
        if name == "__array__":
            return self._convert
        raise AttributeError(name)


class _GetattributeClusterArray(_DynamicClusterArrayBase):
    def __getattribute__(self, name):
        if name == "__array__":
            return object.__getattribute__(self, "_convert")
        return object.__getattribute__(self, name)


class _GetattrClusterSequence(_GetattrClusterArray, Sequence):
    pass


class _GetattributeClusterSequence(_GetattributeClusterArray, Sequence):
    pass


@pytest.mark.parametrize("container", [
    _GetattrClusterArray, _GetattributeClusterArray,
    _GetattrClusterSequence, _GetattributeClusterSequence,
])
@pytest.mark.parametrize("reject_object_dtype", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_dynamic_cluster_array_protocol_preserves_conversion_and_object_validation(
    container, reject_object_dtype, missing, use_estimator, backend,
):
    original = ["nan", "nan", None if missing else "None", "NaT"]
    labels = container(original, reject_object_dtype)
    resampling._validate_cluster_labels_before_cast(labels)
    assert labels.conversions == []
    values = np.array([1., 2., 100., 5.])
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    if missing:
        with pytest.raises(ValueError, match="nonmissing label per row"):
            function(np.mean, values, clusters=labels, strategy="cluster",
                     n_resamples=29, random_state=4, backend=backend)
    else:
        result = function(np.mean, values, clusters=labels, strategy="cluster",
                          n_resamples=29, random_state=4, backend=backend)
        expected = [values[idx].mean() for idx in
                    _reference_indices(np.asarray(original), 4, 29)]
        np.testing.assert_array_equal(result.samples, expected)
    expected_conversions = (3 if backend == "auto" else 2) if use_estimator else 2
    assert labels.conversions == [None] * expected_conversions


def test_cluster_prevalidation_does_not_probe_dynamic_metaclass_capabilities():
    class DynamicCapabilities(type):
        def __getattr__(cls, name):
            if name in {"__len__", "__getitem__"}:
                raise AssertionError("do not dynamically discover sequence capabilities")
            raise AttributeError(name)

    class OpaqueLabels(metaclass=DynamicCapabilities):
        pass

    resampling._validate_cluster_labels_before_cast(OpaqueLabels())


class _StatefulHostClusterSequence(Sequence):
    """A normal index sequence whose next materialization changes its state."""

    def __init__(self, values, max_reads, later_values=None):
        self.values = values
        self.max_reads = max_reads
        self.later_values = later_values
        self.reads = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if index == 0:
            self.reads += 1
        if self.reads > self.max_reads:
            if self.later_values is None:
                raise RuntimeError("original sequence was materialized again")
            return self.later_values[index]
        return self.values[index]


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
@pytest.mark.parametrize("change_labels", [False, True])
@pytest.mark.parametrize("column", [False, True])
def test_host_sequence_snapshot_is_reused_for_validation_and_conversion(
    use_estimator, backend, change_labels, column,
):
    # Estimator-wide finite-input validation already reads inputs once before
    # entering the helper. Resampling itself must read the host sequence once.
    expected_reads = 2 if use_estimator else 1
    original = [[0], [0], [1], [1]] if column else [0, 0, 1, 1]
    later = [[0], [1], [0], [1]] if column else [0, 1, 0, 1]
    labels = _StatefulHostClusterSequence(
        original, expected_reads, later if change_labels else None,
    )
    values = np.array([1., 3., 10., 30.])
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, values, clusters=labels, strategy="cluster",
                      n_resamples=29, random_state=4, backend=backend)
    expected = [values[idx].mean() for idx in
                _reference_indices(np.array([0, 0, 1, 1]), 4, 29)]
    np.testing.assert_array_equal(result.samples, expected)
    assert labels.reads == expected_reads


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_nested_host_sequences_are_snapshotted_without_repeated_iteration(use_estimator, backend):
    expected_reads = 2 if use_estimator else 1
    rows = [_StatefulHostClusterSequence([label], expected_reads)
            for label in [0, 0, 1, 1]]
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, np.arange(4.), clusters=rows, strategy="cluster",
                      n_resamples=11, random_state=4, backend=backend)
    expected = [np.arange(4.)[idx].mean() for idx in
                _reference_indices(np.array([0, 0, 1, 1]), 4, 11)]
    np.testing.assert_array_equal(result.samples, expected)
    assert [row.reads for row in rows] == [expected_reads] * 4


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_snapshot_preserves_nested_float32_and_string_label_equivalence(use_estimator, backend):
    # An object-array .tolist() round-trip widens float32(0.1) to a Python float,
    # turning these four equivalent labels into two different string groups.
    labels = [np.array([0.1], dtype=np.float32),
              np.array([0.1], dtype=np.float32), ["0.1"], ["0.1"]]
    assert len(np.unique(np.asarray(labels))) == 1
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, np.array([1., 3., 10., 30.]), clusters=labels,
                      strategy="cluster", n_resamples=11, random_state=4, backend=backend)
    assert result.metadata["n_clusters"] == 1
    np.testing.assert_array_equal(result.samples, np.full(11, 11.))


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_multidimensional_host_buffer_snapshot_retains_shape_and_dtype(use_estimator, backend):
    labels_array = np.array([[2.], [2.], [1.], [1.]], dtype=np.float32)
    labels_array.flags.writeable = False
    labels = memoryview(labels_array)
    validated = resampling._validate_cluster_labels_before_cast(labels)
    assert validated.shape == (4, 1)
    assert validated.dtype == labels_array.dtype
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, np.arange(4.), clusters=labels, strategy="cluster",
                      n_resamples=11, random_state=4, backend=backend)
    expected = [np.arange(4.)[idx].mean() for idx in
                _reference_indices(labels_array.reshape(-1), 4, 11)]
    np.testing.assert_array_equal(result.samples, expected)
    assert not labels_array.flags.writeable


@pytest.mark.parametrize("container", [_GetattrClusterArray, _GetattributeClusterArray])
@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_host_snapshot_does_not_add_dynamic_array_protocol_conversions(
    container, use_estimator, backend,
):
    labels = container([0, 0, 1, 1], reject_object_dtype=True)
    # Dynamic array-like objects are not ordinary host sequences and remain
    # untouched by the snapshot boundary, including their conversion count.
    assert resampling._validate_cluster_labels_before_cast(labels) is labels
    assert labels.conversions == []
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(np.mean, np.arange(4.), clusters=labels, strategy="cluster",
                      n_resamples=11, random_state=4, backend=backend)
    expected_count = (3 if backend == "auto" else 2) if use_estimator else 2
    assert labels.conversions == [None] * expected_count
    expected = [np.arange(4.)[idx].mean() for idx in
                _reference_indices(np.array([0, 0, 1, 1]), 4, 11)]
    np.testing.assert_array_equal(result.samples, expected)


@pytest.mark.parametrize("use_estimator", [False, True], ids=["module", "estimator"])
@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_virtual_sequence_registration_does_not_add_iterable_input_support(use_estimator, backend):
    class LabelsIterable:
        def __init__(self):
            self.iterations = 0

        def __len__(self):
            return 4

        def __iter__(self):
            self.iterations += 1
            return iter([0, 0, 1, 1])

    Sequence.register(LabelsIterable)
    labels = LabelsIterable()
    assert isinstance(labels, Sequence)
    assert resampling._validate_cluster_labels_before_cast(labels) is labels
    assert labels.iterations == 0
    function = (LinearRegression(device="cpu").bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    with pytest.raises((IndexError, TypeError, ValueError)):
        function(np.mean, np.arange(4.), clusters=labels, strategy="cluster",
                 n_resamples=5, backend=backend)
    assert labels.iterations == 0


def test_snapshot_does_not_accept_sequence_with_unusable_length():
    class LabelsWithoutUsableLength:
        def __len__(self):
            raise TypeError("length unavailable")

        def __getitem__(self, index):
            return [0, 0, 1, 1][index]

    with pytest.raises((IndexError, TypeError, ValueError)):
        bootstrap_statistic(np.mean, np.arange(4.), clusters=LabelsWithoutUsableLength(),
                            strategy="cluster", n_resamples=5, backend="numpy")
