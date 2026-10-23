"""Executable documentation for generic, explicitly selected resampling paths."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.inference import bootstrap_statistic, permutation_test

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_module_resampling_examples_execute(language):
    text = (ROOT / 'docs' / language / 'guides' / 'inference-api.md').read_text()
    examples = re.findall(
        r'<!-- api-example: module-([^>]+) -->\s*```python\n(.*?)```',
        text, flags=re.DOTALL,
    )
    assert {label for label, _ in examples} == {'bootstrap', 'permutation', 'matrix-mean', 'cluster'}
    for label, code in examples:
        scope = {'__name__': '__doc_example__'}
        exec(compile(code, f'{language}/inference-api:{label}', 'exec'), scope)  # noqa: S102
        result = scope['result']
        assert np.isfinite(result.observed)
        assert np.all(np.isfinite(result.samples))
        if label == 'cluster':
            assert result.samples.shape == (99,)
            assert result.metadata['n_samples'] == 6
            assert result.metadata['resampling_unit'] == 'cluster'
            assert result.metadata['n_clusters'] == 3
            assert result.metadata['clusters_per_resample'] == 3
            assert result.metadata['variable_sample_size'] is True
        if label == 'matrix-mean':
            assert result.observed == 5.5
            assert result.samples.shape == (99,)


@pytest.mark.parametrize('shape', [(6, 1), (6, 2)])
def test_matrix_mean_without_hint_matches_replayed_row_bootstrap(shape):
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    result = bootstrap_statistic(np.mean, data, n_resamples=17, random_state=7,
                                 backend='numpy')
    indices = np.random.default_rng(7).integers(0, shape[0], size=(17, shape[0]))
    expected = data[indices].mean(axis=(1, 2))
    np.testing.assert_allclose(result.samples, expected)
    assert result.observed == data.mean()


def test_equal_size_cluster_bootstrap_keeps_complete_groups():
    group = np.repeat(np.arange(3), 2)

    def count_first_cluster(rows):
        # A scalar-only callback deliberately declines automatic batch probing.
        if rows.ndim != 1:
            raise ValueError('scalar statistic')
        return float(np.sum(rows == 0))

    result = bootstrap_statistic(count_first_cluster, group, clusters=group,
                                 strategy='cluster', n_resamples=40,
                                 random_state=0, backend='numpy')
    assert np.all(result.samples % 2 == 0)
    assert np.all((result.samples >= 0) & (result.samples <= len(group)))


@pytest.mark.parametrize('use_estimator', [False, True])
def test_unequal_cluster_row_counts_match_complete_seeded_draws(use_estimator):
    from statgpu import LinearRegression

    sizes = np.array([1, 2, 3])
    labels = np.repeat(np.arange(3), sizes)
    calls = []

    def count_rows(values):
        assert values.ndim == 1
        calls.append(len(values))
        return float(len(values))

    function = (LinearRegression(device='cpu').bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(count_rows, np.arange(6.), clusters=labels,
                      strategy='cluster', n_resamples=17,
                      random_state=7, backend='numpy')
    selected = np.random.default_rng(7).integers(0, 3, size=(17, 3))
    expected_rows = sizes[selected].sum(axis=1)
    np.testing.assert_array_equal(result.samples, expected_rows)
    np.testing.assert_array_equal(calls, np.r_[6, expected_rows])
    assert len(np.unique(expected_rows)) > 1
    assert result.metadata['variable_sample_size'] is True


def test_unequal_cluster_forced_vectorization_requires_one_dimensional_mean_hint():
    values = np.arange(6.)
    labels = np.repeat(np.arange(3), [1, 2, 3])
    options = dict(clusters=labels, strategy='cluster', n_resamples=17,
                   random_state=7, backend='numpy', force_vectorized=True)
    with pytest.raises(ValueError, match='force_vectorized'):
        bootstrap_statistic(np.mean, values, **options)
    observed_calls = []

    def mean(values):
        observed_calls.append(values.shape)
        return values.mean()

    actual = bootstrap_statistic(mean, values, statistic_hint='mean', **options)
    expected = bootstrap_statistic(np.mean, values, clusters=labels,
                                   strategy='cluster', n_resamples=17,
                                   random_state=7, backend='numpy')
    np.testing.assert_allclose(actual.samples, expected.samples)
    assert observed_calls == [(6,)]
    with pytest.raises(ValueError, match='one-dimensional'):
        bootstrap_statistic(np.mean, values[:, None], statistic_hint='mean', **options)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_free_resampling_arguments_and_limitations_are_readable(language):
    guide = (ROOT / 'docs' / language / 'guides' / 'inference-api.md').read_text()
    reference = (ROOT / 'docs' / language / 'reference' / 'estimator-api.md').read_text()
    for function in (bootstrap_statistic, permutation_test):
        for name in inspect.signature(function).parameters:
            assert name in guide
    for text in (guide, reference):
        for field in ('resampling_unit', 'n_clusters', 'clusters_per_resample',
                      'variable_sample_size', 'force_vectorized=True', 'ValueError'):
            assert field in text
    if language == 'en':
        assert 'variable row counts' in guide
        assert 'one-dimensional array' in reference
        assert 'Clusters are never truncated or padded' in reference
        assert 'rejects missing cluster labels' in guide
        assert 'NaN stratum labels' in reference
        assert 'Unequal-size groups can be truncated' not in guide
    else:
        assert '不同的重采样行数' in guide
        assert '一维数组' in reference
        assert '不截断或填充群组' in reference
        assert '缺失群组标签会抛出' in guide
        assert 'NaN 分层标签' in reference
        assert '不等大小群组可能被截断' not in guide


def test_resampling_public_help_matches_backend_and_batch_surface():
    for function in (bootstrap_statistic, permutation_test):
        text = inspect.getdoc(function)
        assert "'torch'" in text
        assert 'stratified' in text
        assert 'equivalence is not checked' in ' '.join(text.split()) or (
            'equivalence is not checked' in text.replace('\n', ' ')
        )
    assert 'unequal' in bootstrap_statistic.__doc__.lower()
    assert 'one-dimensional' in bootstrap_statistic.__doc__


def test_estimator_bootstrap_help_describes_whole_cluster_sampling():
    from statgpu import LinearRegression

    help_text = ' '.join(inspect.getdoc(LinearRegression.bootstrap_statistic).split())
    assert 'draws the original number of clusters with replacement' in help_text
    assert 'variable row counts' in help_text
    assert 'scalar callbacks without batch probes' in help_text
    assert 'Missing cluster labels raise ValueError' in help_text
    for field in ('resampling_unit', 'n_clusters', 'clusters_per_resample',
                  'variable_sample_size'):
        assert field in help_text
    assert 'truncates the final group' not in help_text
    assert 'can split clusters' not in help_text


@pytest.mark.parametrize("kind", ["bootstrap", "permutation"])
def test_vectorized_callback_handles_final_single_row_batch(kind):
    data = np.arange(10.0)
    shapes = []

    def statistic(*arrays):
        values = arrays[-1]
        shapes.append(values.shape)
        return values.mean(axis=-1)

    if kind == "bootstrap":
        result = bootstrap_statistic(
            statistic, data, n_resamples=1025, random_state=1,
            force_vectorized=True, backend="numpy",
        )
    else:
        result = permutation_test(
            statistic, data, data, n_resamples=257, random_state=1,
            force_vectorized=True, backend="numpy",
        )
    assert shapes[0] == (10,)
    assert shapes[-1] == (1, 10)
    assert all(len(shape) == 2 for shape in shapes[1:])
    assert np.isfinite(result.samples).all()
    for function in (bootstrap_statistic, permutation_test):
        text = ' '.join(inspect.getdoc(function).split())
        assert 'later incompatible batch can still fall back to scalar calls' in text
