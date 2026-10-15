"""Check the concrete quantities explained by the bilingual model bridges.

These small CPU checks exercise the published examples and distinguish common
misreadings; they do not evaluate prose quality or establish inference coverage.
"""
import ast
from pathlib import Path
import re

import numpy as np
import pytest

from doc_examples import run_example

ROOT = Path(__file__).resolve().parents[2]


def lesson(language, page, example):
    path = ROOT / 'docs' / language / 'models' / f'{page}.md'
    text = path.read_text(encoding='utf-8')
    return text, run_example(text, example, str(path))


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_logistic_odds_example_converts_back_to_probability(language):
    text, ns = lesson(language, 'regression/logistic-regression', 'logistic-unpenalized')
    # Read the actual number used in the new worked probability expression.
    match = re.search(r'\$([\d.]+)/\(1\+([\d.]+)\)\\approx([\d.]+)\$', text)
    assert match is not None
    ratio, denominator_ratio, stated_probability = map(float, match.groups())
    assert ratio == denominator_ratio
    fitted_ratio = float(np.exp(ns['model'].coef_[0]))
    assert fitted_ratio == pytest.approx(ratio, abs=0.0005)
    assert ratio / (1 + ratio) == pytest.approx(stated_probability, abs=0.005)
    # A one-unit change multiplies odds, not probability. The same odds ratio
    # gives a smaller increase from probability 0.1 than from probability 0.5.
    probabilities = np.array([0.1, 0.5])
    odds = probabilities / (1 - probabilities)
    new_odds = odds * ratio
    changed = new_odds / (1 + new_odds)
    assert changed[0] - probabilities[0] < changed[1] - probabilities[1]
    assert not np.allclose(changed, probabilities * ratio)
    assert probabilities[1] * ratio > 1


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_kernel_pca_coordinates_belong_to_individual_rows(language):
    _, ns = lesson(language, 'nonparametric/kernel-pca', 'kernel-pca-cpu')
    coordinates, X = ns['test_coordinates'], ns['X_test']
    assert coordinates.shape == (60, 3)
    assert X.shape == (60, 2)
    assert coordinates[0].shape == (3,)
    # Query ordering changes row ordering, not what an observation represents.
    order = np.array([9, 0, 4])
    np.testing.assert_allclose(
        ns['kpca'].transform(X[order]), coordinates[order], atol=1e-12,
    )
    np.testing.assert_allclose(
        ns['kpca'].transform(X[:1])[0], coordinates[0], atol=1e-12,
    )
    # These are three constructed coordinates, not the two original features.
    assert coordinates.shape[1] != X.shape[1]


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_spline_matrix_rows_are_basis_values_before_response_fit(language):
    _, ns = lesson(language, 'nonparametric/splines', 'spline-transformer-reuse-cpu')
    transformer = ns['transformer']
    assert ns['B_train'].shape == (41, 7)
    assert ns['B_query'].shape == (3, 7)
    for i, point in enumerate(ns['X_query']):
        np.testing.assert_allclose(
            transformer.transform(point[None, :])[0], ns['B_query'][i], atol=1e-12,
        )
    # Each prediction sums an intercept and weighted basis evaluations.
    expected = ns['coefficients'][0] + ns['B_query'] @ ns['coefficients'][1:]
    np.testing.assert_allclose(ns['prediction'], expected, atol=1e-12)
    np.testing.assert_allclose(expected, [-0.778, 0, 0.778], atol=0.001)
    assert ns['B_query'].shape != ns['prediction'].shape


def _arithmetic(expression, **variables):
    """Evaluate only the small arithmetic grammar used by the prose examples."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise AssertionError(f'Unexpected illustrative arithmetic: {expression}')
    return visit(ast.parse(expression.replace(r'\times', '*'), mode='eval').body)


def _vectors(text):
    matches = re.findall(r'`(\[[\d., +\-\[\]]+\])`', text)
    assert len(matches) >= 2
    vectors = tuple(np.asarray(ast.literal_eval(x)) for x in matches[:2])
    assert all(vector.ndim == 1 for vector in vectors), 'Expected matrix row vectors'
    return vectors


def _equation(text):
    match = re.search(r'\$([\d. +\-]+(?:\\times[\d. +\-]+)+)=([\d. +\-]+)\$', text)
    assert match is not None
    lhs, rhs = match.groups()
    result = _arithmetic(rhs)
    assert _arithmetic(lhs) == pytest.approx(result)
    # Retain the two factors separately to check their documented roles.
    products = re.findall(r'([+\-]?[\d.]+)\\times([+\-]?[\d.]+)', lhs)
    factors = np.array([(float(a), float(b)) for a, b in products])
    return lhs, result, factors


def _check_sparse_illustration(text):
    marker = 'For arithmetic illustration only' if 'For arithmetic illustration only' in text else '仅作算术示意'
    illustration = text.split(marker, 1)[1].split('\n## ', 1)[0]
    intercept_match = re.search(r'(?:intercept is|截距为)\s*([+\-]?[\d.]+)', illustration)
    assert intercept_match is not None
    intercept = float(intercept_match.group(1))
    coefficients, row = _vectors(illustration)
    assert coefficients.shape == row.shape
    lhs, stated_prediction, factors = _equation(illustration)
    # Factors are ordered coefficient × predictor value in the shown equation.
    np.testing.assert_allclose(factors[:, 0], coefficients)
    np.testing.assert_allclose(factors[:, 1], row)
    shown_intercept = re.match(r'([+\-]?[\d.]+)\+', lhs)
    assert shown_intercept is not None
    assert float(shown_intercept.group(1)) == intercept
    assert intercept + row @ coefficients == pytest.approx(stated_prediction)
    return intercept, coefficients, row, stated_prediction


@pytest.mark.parametrize('language', ('en', 'cn'))
@pytest.mark.parametrize('page', ('scad', 'mcp'))
def test_sparse_linear_prediction_arithmetic(language, page):
    text = (ROOT / 'docs' / language / 'models/regression' / f'{page}.md').read_text()
    intercept, coefficients, row, stated_prediction = _check_sparse_illustration(text)
    zero_columns = np.flatnonzero(coefficients == 0)
    assert len(zero_columns) > 0
    changed_row = row.copy()
    changed_row[zero_columns] += 693
    assert intercept + changed_row @ coefficients == stated_prediction
    assert np.count_nonzero(coefficients) == 2

    # These mutations must be rejected by the same checker as the real prose.
    mutations = (
        text.replace(r'0.5\times2=6$', r'0.5\times2=7$', 1),
        text.replace('[2, 0, -0.5]', '[3, 7, 2]', 1)
            .replace('row `[3, 7, 2]`', 'row `[2, 0, -0.5]`', 1)
            .replace('一行为 `[3, 7, 2]`', '一行为 `[2, 0, -0.5]`', 1),
        text.replace(r'+0\times7', r'+1\times7', 1),
        text.replace('[3, 7, 2]', '[[3], [7], [2]]', 1),
        text.replace('intercept is 1', 'intercept is 2', 1)
            .replace('截距为 1', '截距为 2', 1),
    )
    for mutated in mutations:
        assert mutated != text
        with pytest.raises(AssertionError):
            _check_sparse_illustration(mutated)


def _check_two_basis_illustration(text):
    definitions = [re.search(rf'\$B_{j}\(t\)=([^$]+)\$', text) for j in (1, 2)]
    assert all(definitions)
    t_match = re.search(r'\$t=([+\-]?[\d.]+)\$', text)
    assert t_match is not None
    t = float(t_match.group(1))
    # The first vectors after the evaluation point give basis values and weights.
    illustration = text[t_match.end():].split('\n\n', 1)[0]
    stated_values, weights = _vectors(illustration)
    values = np.array([_arithmetic(match.group(1), t=t) for match in definitions])
    np.testing.assert_allclose(values, stated_values)
    assert values.sum() == pytest.approx(1)
    _, stated_prediction, factors = _equation(illustration)
    np.testing.assert_allclose(factors[:, 0], weights)
    np.testing.assert_allclose(factors[:, 1], stated_values)
    assert values @ weights == pytest.approx(stated_prediction)
    return t, values


@pytest.mark.parametrize('language', ('en', 'cn'))
def test_two_linear_basis_values_and_weighted_prediction(language):
    text = (ROOT / 'docs' / language / 'models/nonparametric/splines.md').read_text()
    t, values = _check_two_basis_illustration(text)
    # The scalar input itself is not the two-column transformed row.
    assert not np.allclose(values, [t, t])
    mutations = (
        text.replace(r'0.25=3$', r'0.25=4$', 1),
        text.replace('$t=0.25$', '$t=0.5$', 1),
        text.replace('$B_1(t)=1-t$', '$B_1(t)=t$', 1),
        text.replace('[0.75, 0.25]', '[0.25, 0.75]', 1),
        text.replace('[0.75, 0.25]', '[[0.75], [0.25]]', 1),
        text.replace('[2, 6]', '[6, 2]', 1),
    )
    for mutated in mutations:
        assert mutated != text
        with pytest.raises(AssertionError):
            _check_two_basis_illustration(mutated)
