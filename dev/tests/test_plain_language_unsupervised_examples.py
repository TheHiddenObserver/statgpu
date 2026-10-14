"""Check the arithmetic and API meanings taught in the revised short examples.

These are CPU example checks, not a claim to measure prose comprehension or
GPU accuracy. Negative controls distinguish plausible but incorrect readings.
"""
from pathlib import Path
import ast
import operator
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def page(language, slug):
    return (ROOT / f'docs/{language}/unsupervised/{slug}.md').read_text()


def run_example(language, slug, marker='learner-example', text=None):
    path = ROOT / f'docs/{language}/unsupervised/{slug}.md'
    name = 'nonnegative-spectra' if marker == 'worked-example' else slug
    match = re.search(rf'<!-- {marker}: {name} -->\s*```python\n(.*?)```',
                      path.read_text() if text is None else text, re.S)
    assert match, f'Missing executable example in {path}'
    namespace = {}
    exec(compile(match[1], str(path), 'exec'), namespace)
    return namespace


def validate_spectrum_example(text, language, slug):
    """Tie the prose operands/result and printed comment to the executed block."""
    opening = text.split('<!-- worked-example: nonnegative-spectra -->')[0]
    vectors = [np.array(ast.literal_eval(v), dtype=float)
               for v in re.findall(r'`(\[[^`]+\])`', opening)]
    assert len(vectors) == 3  # A, B, then this sample's coefficients.
    A, B, coefficients = vectors
    equation = re.search(r'`([0-9.]+)\*A\s*\+\s*([0-9.]+)\*B\s*=\s*(\[[^`]+\])`', opening)
    assert equation
    multipliers = np.array([float(equation[1]), float(equation[2])])
    declared = np.array(ast.literal_eval(equation[3]), dtype=float)
    ns = run_example(language, slug, 'worked-example', text=text)
    H, w, result = ns['H'], ns['w'], ns['x_hat']
    assert H.shape == (2, 5) and w.shape == (2,)
    np.testing.assert_array_equal(H, np.vstack([A, B]))
    np.testing.assert_array_equal(w, coefficients)
    np.testing.assert_array_equal(w, multipliers)
    np.testing.assert_allclose(declared, coefficients @ np.vstack([A, B]))
    np.testing.assert_allclose(result, declared)
    printed = re.search(r'print\(x_hat\)\s*#\s*\[([^\]]+)\]', text)
    assert printed
    np.testing.assert_allclose(np.fromstring(printed[1], sep=' '), declared)
    return H, w, result


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('slug', ['nmf', 'minibatch-nmf'])
def test_spectrum_coefficients_reconstruct_intensities(language, slug):
    H, w, result = validate_spectrum_example(page(language, slug), language, slug)
    assert not np.isclose(w.sum(), 1)
    assert not np.all(w == np.floor(w))
    assert not np.allclose((w / w.sum()) @ H, result)
    doubled = H.copy()
    doubled[0] *= 2
    assert not np.allclose(w @ doubled, result)  # Old coefficients, new A.
    np.testing.assert_allclose((w / [2, 1]) @ doubled, result)
    # Scale and permutation ambiguity preclude identifying unique sources.
    np.testing.assert_allclose(w[::-1] @ H[::-1], result)


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('slug', ['nmf', 'minibatch-nmf'])
@pytest.mark.parametrize('old,new', [
    ('2*A + 0.5*B', '3*A + 0.5*B'),  # Prose multiplier disagrees with block.
    ('x_hat = w @ H', 'x_hat = (w / w.sum()) @ H'),
    ('x_hat = w @ H', 'x_hat = w @ H.T'),
    ('# [2.  1.  4.  1.  2.5]', '# [9.  1.  4.  1.  2.5]'),
    ('= [2, 1, 4, 1, 2.5]', '= [9, 1, 4, 1, 2.5]'),
])
def test_spectrum_validator_rejects_changed_explanations(language, slug, old, new):
    text = page(language, slug)
    assert old in text
    with pytest.raises((AssertionError, ValueError)):
        validate_spectrum_example(text.replace(old, new), language, slug)


def arithmetic(expression):
    """Evaluate just scalar arithmetic from the documented coordinate example."""
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](visit(node.left), visit(node.right))
        raise AssertionError('Expected only numeric coordinate arithmetic')
    return visit(ast.parse(expression.strip(), mode='eval').body)


def validate_coordinate_example(text):
    opening = text.split('## When to use it')[0].split('## 何时使用')[0]
    vectors = [np.array(ast.literal_eval(v), dtype=float)
               for v in re.findall(r'`(\[[^`]+\])`', opening)]
    assert len(vectors) == 2
    equation = re.search(r'`([0-9.*+\s-]+)\s*=\s*([0-9.-]+)`', opening)
    assert equation
    declared = float(equation[2])
    np.testing.assert_allclose(vectors[0] @ vectors[1], declared)
    np.testing.assert_allclose(arithmetic(equation[1]), declared)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_coordinate_arithmetic_is_read_from_prose(language):
    text = page(language, 'incremental-pca')
    validate_coordinate_example(text)
    for old, new in [('[2, -1]', '[2, 1]'), ('[0.6, 0.8]', '[0.6, 0.9]'),
                     ('2*0.6 - 1*0.8', '2*0.6 + 1*0.8'), ('= 0.4`', '= 0.5`')]:
        assert old in text
        with pytest.raises(AssertionError):
            validate_coordinate_example(text.replace(old, new))


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('slug', ['nmf', 'minibatch-nmf'])
def test_fitted_tables_keep_samples_and_wavelengths_in_order(language, slug):
    ns = run_example(language, slug)
    model, X, W = ns['model'], ns['X'], ns['W']
    assert X.shape == (60, 5)
    assert W.shape == (60, 2) and model.components_.shape == (2, 5)
    np.testing.assert_allclose(ns['X_hat'], W @ model.components_)
    np.testing.assert_allclose(ns['X_hat'][17], W[17] @ model.components_)
    assert not np.allclose(ns['X_hat'], W[::-1] @ model.components_)
    if slug == 'minibatch-nmf':
        # Rows from all four batches were solved under the final dictionary.
        np.testing.assert_allclose(W[:15], model.transform(X[:15]))


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_incremental_coordinates_use_final_means_and_weights(language):
    ns = run_example(language, 'incremental-pca')
    m, X, Z = ns['model'], ns['X'], ns['Z']
    assert Z.shape == (60, 2) and m.n_samples_seen_ == 60
    np.testing.assert_allclose(Z, (X - m.mean_) @ m.components_.T)
    np.testing.assert_allclose(ns['X_hat'], Z @ m.components_ + m.mean_)
    assert not np.allclose(Z, X @ m.components_.T)
    # Reusing coordinates after changing the conversion cannot preserve readings.
    assert not np.allclose(ns['X_hat'], Z @ m.components_ + m.mean_ + 10)


def validate_probability_example(text, language):
    term = 'responsibilities' if language == 'en' else '责任度'
    explanation = text[:text.index(term)]
    pattern = (r'([0-9.]+) for component A and ([0-9.]+) for B'
               if language == 'en' else r'成分 A 为 ([0-9.]+)、成分 B 为 ([0-9.]+)')
    example = re.search(pattern, explanation)
    assert example, 'Concrete probabilities must precede their technical name'
    probabilities = np.array([float(example[1]), float(example[2])])
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(), 1)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_probability_example_precedes_technical_name_and_normalizes(language):
    text = page(language, 'gaussian-mixture')
    validate_probability_example(text, language)
    assert '0.7' in text
    with pytest.raises(AssertionError):
        validate_probability_example(text.replace('0.7', '0.8'), language)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_mixture_probabilities_are_per_record_not_population_weights(language):
    ns = run_example(language, 'gaussian-mixture')
    m, proba = ns['model'], ns['proba']
    assert proba.shape == (80, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert np.all((proba >= 0) & (proba <= 1))
    assert not np.allclose(proba, np.broadcast_to(m.weights_, proba.shape))
    np.testing.assert_array_equal(m.predict(ns['X']), proba.argmax(axis=1))


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_svd_coordinates_do_not_center_or_add_back_means(language):
    ns = run_example(language, 'truncated-svd')
    m, X, Z = ns['model'], ns['X'], ns['Z']
    assert Z.shape == (60, 2)
    np.testing.assert_allclose(Z, X @ m.components_.T)
    np.testing.assert_allclose(ns['X_hat'], Z @ m.components_)
    assert not np.allclose(Z, (X - X.mean(axis=0)) @ m.components_.T)
    assert not np.allclose(ns['X_hat'], Z @ m.components_ + X.mean(axis=0))
    np.testing.assert_allclose(m.explained_variance_, Z.var(axis=0))
    energy_fraction = np.sum(m.singular_values_ ** 2) / np.sum(X ** 2)
    assert not np.isclose(m.explained_variance_ratio_.sum(), energy_fraction)
