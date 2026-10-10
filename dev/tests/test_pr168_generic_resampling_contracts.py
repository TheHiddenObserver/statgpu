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
    assert {label for label, _ in examples} == {'bootstrap', 'permutation', 'matrix-mean'}
    for label, code in examples:
        scope = {'__name__': '__doc_example__'}
        exec(compile(code, f'{language}/inference-api:{label}', 'exec'), scope)  # noqa: S102
        result = scope['result']
        assert np.isfinite(result.observed)
        assert np.all(np.isfinite(result.samples))
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


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_free_resampling_arguments_and_limitations_are_readable(language):
    guide = (ROOT / 'docs' / language / 'guides' / 'inference-api.md').read_text()
    reference = (ROOT / 'docs' / language / 'reference' / 'estimator-api.md').read_text()
    for function in (bootstrap_statistic, permutation_test):
        for name in inspect.signature(function).parameters:
            assert name in guide
    if language == 'en':
        assert 'Unequal-size groups can be truncated' in guide
        assert 'one-dimensional array' in reference
        assert 'Do not use its intervals for that setting' in reference
    else:
        assert '不等大小群组可能被截断' in guide
        assert '一维数组' in reference
        assert '此时不要使用它给出的区间' in reference


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


def test_estimator_bootstrap_help_discloses_unequal_cluster_limit():
    from statgpu import LinearRegression

    help_text = ' '.join(inspect.getdoc(LinearRegression.bootstrap_statistic).split())
    assert 'equal-size whole clusters' in help_text
    assert 'truncates the final group' in help_text
    assert 'can split clusters' in help_text
    assert 'do not use its intervals for that setting' in help_text


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
