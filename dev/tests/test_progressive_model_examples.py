"""Run bounded EN/CN model tutorials step by step, with declared setup only.

CUDA-only examples are syntax/prerequisite checked, not executed on this CPU
runner. Unmarked API sketches are excluded from runtime execution deliberately;
substantive estimator, inference, solver and reference tests remain separate.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from doc_examples import parse_examples, run_example

ROOT = Path(__file__).resolve().parents[2]
FIRST_EXAMPLES = {
    'adaptive-lasso': 'adaptive-lasso-prediction',
    'anova': 'anova-basic',
    'covariance': 'covariance-basic',
    'coxph': 'coxph-cpu-walkthrough',
    'elastic-net': 'elasticnet-prediction',
    'feature-selection': 'feature-selection-basic',
    'generalized-linear-model': 'glm-poisson',
    'kernel-methods': 'kernel-methods-cpu',
    'knockoff': 'knockoff-selection',
    'lasso': 'lasso-prediction',
    'linear-regression': 'linear-prediction',
    'logistic-regression': 'logistic-unpenalized',
    'losses': 'losses-huber',
    'mcp': 'mcp-prediction',
    'multiple-testing': 'multiple-testing-learner',
    'nonparametric': 'kde-cpu',
    'ordered': 'ordered-basic',
    'poisson-regression': 'poisson-unpenalized',
    'quantile': 'quantile-basic',
    'ridge': 'ridge-weighted-prediction',
    'robust': 'robust-basic',
    'scad': 'scad-prediction',
    'semiparametric': 'gam-cpu',
    'splines': 'spline-transformer-reuse-cpu',
}
# This explicit inventory prevents deleting later wrapper tags from silently
# removing tutorials from runtime coverage. Dependencies are part of the contract.
EXPECTED_EXAMPLES = {
    'adaptive-lasso': {
        'adaptive-lasso-prediction': (),
    },
    'anova': {
        'anova-basic': (),
        'anova-welch': ('anova-basic',),
        'anova-tukey': ('anova-basic',),
    },
    'covariance': {
        'covariance-basic': (),
        'covariance-robust': ('covariance-basic',),
        'covariance-sparse': ('covariance-basic',),
    },
    'coxph': {
        'coxph-cpu-walkthrough': (),
        'coxph-survival-prediction': ('coxph-cpu-walkthrough',),
        'coxph-cpu-cv': ('coxph-cpu-walkthrough',),
        'coxph-penalized-family-cv': ('coxph-cpu-walkthrough',),
        'coxph-cupy-fit': ('coxph-cpu-walkthrough',),
        'coxph-cupy-cv': ('coxph-cupy-fit',),
        'coxph-torch-fit': ('coxph-cpu-walkthrough',),
        'coxph-torch-cv': ('coxph-torch-fit',),
        'coxph-summary': ('coxph-cpu-walkthrough',),
    },
    'elastic-net': {
        'elasticnet-prediction': (),
        'elasticnet-nodewise': ('elasticnet-prediction',),
        'elasticnet-weighted-score': (),
    },
    'feature-selection': {
        'feature-selection-basic': (),
    },
    'generalized-linear-model': {
        'glm-poisson': (),
        'glm-formula': (),
        'glm-cv-inference': ('glm-poisson',),
    },
    'kernel-methods': {
        'kernel-methods-cpu': (),
        'kernel-callable-cpu': (),
        'kernel-methods-cupy': ('kernel-methods-cpu',),
        'kernel-methods-torch': ('kernel-methods-cpu',),
    },
    'knockoff': {
        'knockoff-selection': (),
    },
    'lasso': {
        'lasso-prediction': (),
        'lasso-simultaneous': ('lasso-prediction',),
    },
    'linear-regression': {
        'linear-prediction': (),
        'linear-column-target': (),
    },
    'logistic-regression': {
        'logistic-unpenalized': (),
    },
    'losses': {
        'losses-huber': (),
        'losses-weighted': ('losses-huber',),
        'losses-cox': (),
    },
    'mcp': {
        'mcp-prediction': (),
    },
    'multiple-testing': {
        'multiple-testing-learner': (),
    },
    'nonparametric': {
        'kde-cpu': (),
        'kernel-regression-cpu': (),
        'kde-bootstrap-cpu': (),
        'kde-gpu': ('kde-cpu',),
    },
    'ordered': {
        'ordered-basic': (),
        'ordered-inference': ('ordered-basic',),
        'ordered-probit': ('ordered-basic',),
    },
    'poisson-regression': {
        'poisson-unpenalized': (),
        'poisson-formula': (),
    },
    'quantile': {
        'quantile-basic': (),
        'quantile-inference': ('quantile-basic',),
        'quantile-penalized': ('quantile-basic',),
        'quantile-scad': ('quantile-penalized',),
        'quantile-explicit-solvers': ('quantile-penalized',),
        'quantile-weighted': ('quantile-penalized',),
    },
    'ridge': {
        'ridge-weighted-prediction': (),
        'ridge-training-origin': (),
    },
    'robust': {
        'robust-basic': (),
        'robust-scad': ('robust-basic',),
        'robust-bisquare': ('robust-basic',),
        'robust-fair': ('robust-basic',),
        'robust-direct-solver': ('robust-basic',),
    },
    'scad': {
        'scad-prediction': (),
    },
    'semiparametric': {
        'gam-cpu': (),
        'gam-gpu': ('gam-cpu',),
    },
    'splines': {
        'spline-transformer-reuse-cpu': (),
        'spline-raw-bases-cpu': (),
        'spline-raw-cupy': ('spline-raw-bases-cpu',),
        'spline-raw-torch': ('spline-raw-bases-cpu',),
    },
}

GPU_EXAMPLES = {
    'coxph-cupy-fit', 'coxph-cupy-cv', 'coxph-torch-fit', 'coxph-torch-cv',
    'kernel-methods-cupy', 'kernel-methods-torch', 'kde-gpu', 'gam-gpu',
    'spline-raw-cupy', 'spline-raw-torch',
}
OPTIONAL_EXAMPLES = {'glm-formula', 'poisson-formula'}


def _checked_examples(text, page, source):
    examples = parse_examples(text, source)
    actual = {name: example.requires for name, example in examples.items()}
    assert actual == EXPECTED_EXAMPLES[page], f"{source}: named example inventory changed"
    return examples


def _page(language, page):
    path = ROOT / f'docs/{language}/models/{page}.md'
    return path, path.read_text(encoding='utf-8')


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page', FIRST_EXAMPLES)
def test_first_tutorial_has_bounded_readable_steps_and_explicit_dependencies(language, page):
    path, text = _page(language, page)
    examples = _checked_examples(text, page, str(path))
    first = examples[FIRST_EXAMPLES[page]]
    assert first is next(iter(examples.values()))
    assert not first.requires, 'The first tutorial must provide its own setup'
    assert len(first.blocks) >= 3, 'Separate preparation, fitting/calculation, and results'
    assert max(len(block.splitlines()) for block in first.blocks) <= 16
    assert first.start < text.index('```python') < first.end
    for example in examples.values():
        for dependency in example.requires:
            assert examples[dependency].end < example.start, 'Setup must appear before its consumer'
        for block in example.blocks:
            ast.parse(block, filename=str(path))
    # A removed named wrapper cannot silently leave a single-fence tutorial.
    assert all(example.explicit_end for example in examples.values())


@pytest.mark.parametrize('page', FIRST_EXAMPLES)
def test_bilingual_tutorial_boundaries_and_prerequisites_match(page):
    en_path, en_text = _page('en', page)
    cn_path, cn_text = _page('cn', page)
    en = _checked_examples(en_text, page, str(en_path))
    cn = _checked_examples(cn_text, page, str(cn_path))
    assert list(en) == list(cn)
    for name in en:
        assert en[name].requires == cn[name].requires
        assert len(en[name].blocks) == len(cn[name].blocks)


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page', FIRST_EXAMPLES)
def test_every_named_cpu_example_executes_all_steps_with_only_declared_setup(language, page):
    path, text = _page(language, page)
    examples = _checked_examples(text, page, str(path))
    assert FIRST_EXAMPLES[page] in examples
    for name, example in examples.items():
        if name in GPU_EXAMPLES:
            assert example.requires, f'{name}: GPU example must declare its CPU/data prerequisite'
            # Do not mutate device/backend strings or inject pretend CUDA inputs.
            tree = ast.parse(example.code)
            explicit_backend = any(
                isinstance(node, ast.keyword) and node.arg in {'device', 'backend'}
                and isinstance(node.value, ast.Constant)
                and node.value.value in {'cuda', 'cupy', 'torch'}
                for node in ast.walk(tree)
            )
            backend_options = any(
                isinstance(node, ast.Dict) and any(
                    isinstance(key, ast.Constant) and key.value in {'device', 'backend'}
                    and isinstance(value, ast.Constant) and value.value in {'cuda', 'cupy', 'torch'}
                    for key, value in zip(node.keys, node.values)
                ) for node in ast.walk(tree)
            )
            raw_cupy = 'import cupy as cp' in example.code and 'xp=cp' in example.code
            assert explicit_backend or backend_options or raw_cupy
            continue
        if name in OPTIONAL_EXAMPLES:
            pytest.importorskip('pandas', reason='Formula tutorial requires the optional formula extra')
            pytest.importorskip('patsy', reason='Formula tutorial requires the optional formula extra')
        namespace = run_example(text, name, str(path))
        assert namespace, f'{name}: no statements executed'
        if page == 'adaptive-lasso':
            assert namespace['prediction'].shape == (40,)
            assert np.isfinite(namespace['prediction']).all()
            assert namespace['model'].compute_inference is False
            assert namespace['model'].score(namespace['X_test'], namespace['y_test']) > .9
        elif name == 'covariance-basic':
            covariance = namespace['lw'].covariance_
            assert covariance.shape == (10, 10)
            np.testing.assert_allclose(covariance, covariance.T)
            assert np.linalg.eigvalsh(covariance).min() > 0
        elif name == 'anova-basic':
            from scipy.stats import f_oneway
            expected = f_oneway(namespace['g1'], namespace['g2'])
            assert namespace['result'].statistic == pytest.approx(expected.statistic)
            assert namespace['result'].pvalue == pytest.approx(expected.pvalue)
        elif name == 'quantile-basic':
            prediction = namespace['prediction']
            assert prediction.shape == (80,) and np.isfinite(prediction).all()
            residual = namespace['y'][240:] - prediction
            quantile = namespace['model'].quantile
            expected = np.mean(residual * (quantile - (residual < 0)))
            assert namespace['pinball_loss'] == pytest.approx(expected)
            assert -namespace['model'].score(namespace['X'][240:], namespace['y'][240:]) == pytest.approx(expected)
        elif name == 'robust-basic':
            prediction = namespace['prediction']
            assert prediction.shape == (80,) and np.isfinite(prediction).all()
            expected = np.mean(np.abs(namespace['y'][240:] - prediction))
            assert namespace['mae'] == pytest.approx(expected)
        elif name == 'ordered-inference':
            model = namespace['inference_model']
            for field in ('_bse', '_zvalues', '_pvalues'):
                values = getattr(model, field)
                assert values.shape == (4,) and np.isfinite(values).all()
            assert model._conf_int.shape == (4, 2)
            assert np.isfinite(model._conf_int).all()
            assert (model._conf_int[:, 0] < model._conf_int[:, 1]).all()
            assert (model._bse > 0).all()
            assert ((model._pvalues >= 0) & (model._pvalues <= 1)).all()
        elif name == 'ordered-probit':
            probability = namespace['probit_probability']
            assert probability.shape == (100, 3) and np.isfinite(probability).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1)
        elif name == 'ordered-basic':
            assert namespace['prediction'].shape == (100,)
            probability = namespace['probability']
            assert probability.shape == (100, 3)
            assert np.isfinite(probability).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1)
            np.testing.assert_array_equal(namespace['prediction'], probability.argmax(axis=1))


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_removing_later_tutorial_wrappers_cannot_silently_shrink_coverage(language):
    path, text = _page(language, 'linear-regression')
    text = text.replace('<!-- learner-example: linear-column-target -->', '')
    text = text.replace('<!-- example-end: linear-column-target -->', '')
    # It remains valid Markdown and the first tutorial still parses, but the
    # explicit inventory must reject losing the later single-column workflow.
    assert 'linear-prediction' in parse_examples(text)
    with pytest.raises(AssertionError, match='named example inventory changed'):
        _checked_examples(text, 'linear-regression', str(path))


# Remaining unmarked fences are explicitly limited to these optional/sketch
# surfaces. They are compiled and classified, never reported as CPU execution.
UNMARKED_EXAMPLES = {
    'anova': ('gpu', 'gpu'),
    'covariance': ('gpu', 'gpu', 'import-inventory'),
    'coxph': ('formula-sketch',),
    'generalized-linear-model': ('gpu', 'configuration', 'configuration'),
    'knockoff': ('gpu', 'gpu'),
    'logistic-regression': ('gpu',),
    'ordered': ('gpu',),
    'poisson-regression': ('gpu',),
    'quantile': ('gpu',),
    'robust': ('gpu',),
}


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page', FIRST_EXAMPLES)
def test_remaining_python_fences_have_explicit_optional_or_sketch_scope(language, page):
    import re

    path, text = _page(language, page)
    examples = _checked_examples(text, page, str(path))
    unmarked = [match.group(1) for match in re.finditer(r'```python\n(.*?)```', text, re.DOTALL)
                if not any(e.start < match.start() < e.end for e in examples.values())]
    kinds = UNMARKED_EXAMPLES.get(page, ())
    assert len(unmarked) == len(kinds), f'{page}: unexpected unclassified Python fence'
    for code, kind in zip(unmarked, kinds):
        tree = ast.parse(code, filename=str(path))
        if kind == 'gpu':
            assert any(
                isinstance(node, ast.keyword) and node.arg in {'device', 'backend'}
                and isinstance(node.value, ast.Constant)
                and node.value.value in {'cuda', 'cupy', 'torch'}
                for node in ast.walk(tree)
            ), 'Optional CUDA snippets must explicitly request their backend/device'
        elif kind == 'import-inventory':
            assert all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
        elif kind == 'configuration':
            assert 'PenalizedGLM_CV(' in code and '.fit(' not in code
        else:
            assert kind == 'formula-sketch' and page == 'coxph'
            assert 'data=df' in code and 'formula=' in code
