"""Mutation checks for progressive tutorial extraction and ordered execution."""

import pytest
from doc_examples import example_code, parse_examples, run_example

TUTORIAL = '''<!-- learner-example: basic -->
### Import a helper
```python
from math import sqrt
```
### Prepare data
Use a fresh list so later calculations can reuse it.
```python
values = [9, 16]
```
### Calculate a result
```python
result = [sqrt(value) for value in values]
```
<!-- example-end: basic -->
'''


def _dependent(name, requirement, code):
    return (f'<!-- example-requires: {requirement} -->\n'
            f'<!-- example: {name} -->\n```python\n{code}\n```\n'
            f'<!-- /example: {name} -->\n')


@pytest.mark.parametrize('kind', ['example', 'learner-example', 'api-example', 'inference-example'])
@pytest.mark.parametrize('end', ['example-end', '/example'])
def test_all_marker_forms_execute_every_step_in_order(kind, end):
    text = TUTORIAL.replace('learner-example:', kind + ':').replace('example-end:', end + ':')
    example = parse_examples(text)['basic']
    assert len(example.blocks) == len(example.lines) == 3
    assert example.requires == ()
    assert run_example(text, 'basic')['result'] == [3, 4]
    assert 'from math import sqrt' in example_code(text, 'basic')
    assert 'result =' in example_code(text, 'basic')


def test_only_explicit_dependencies_are_run_once_and_namespaces_are_isolated():
    text = TUTORIAL + _dependent('second', 'basic', 'values.append(25)')
    text += _dependent('third', 'basic, second', 'total = sum(values)')
    text += '<!-- example: unrelated -->\n```python\nraise AssertionError("not requested")\n```\n<!-- example-end: unrelated -->'
    assert run_example(text, 'third')['total'] == 50
    assert run_example(text, 'basic')['values'] == [9, 16]
    assert 'unrelated' in parse_examples(text)


@pytest.mark.parametrize('mutation', [
    lambda text: text.replace('from math import sqrt', ''),
    lambda text: text.replace('values = [9, 16]', 'unused = [9, 16]'),
    lambda text: text.replace('values = [9, 16]', 'values = [sqrt(value) for value in values]'),
])
def test_missing_or_reordered_setup_cannot_be_supplied_by_test_globals(mutation):
    # Test module globals and previous run namespaces must never be injected.
    with pytest.raises((AssertionError, NameError)):
        run_example(mutation(TUTORIAL), 'basic')


def test_failure_in_a_later_step_cannot_be_hidden_by_first_fence_extraction():
    text = TUTORIAL.replace('result = [sqrt(value) for value in values]', 'raise RuntimeError("later step")')
    with pytest.raises(RuntimeError, match='later step'):
        run_example(text, 'basic')


@pytest.mark.parametrize('text, reason', [
    (TUTORIAL.replace('<!-- example-end: basic -->', ''), 'missing end'),
    (TUTORIAL.replace('example-end: basic', 'example-end: other'), 'mismatched end'),
    (TUTORIAL + TUTORIAL, 'duplicate example'),
    (TUTORIAL + '<!-- example-end: basic -->', 'unexpected end'),
    (TUTORIAL.replace('### Prepare data', '<!-- example: nested -->'), 'missing end'),
    (TUTORIAL.replace('```python\nvalues', '```python\n```python\nvalues'), 'unclosed fence'),
    (TUTORIAL + '<!-- example-requires: basic -->', 'orphan'),
    (_dependent('second', 'unknown', 'result = 1'), 'unknown example dependency'),
    (_dependent('first', 'second', 'result = 1') + _dependent('second', 'first', 'result = 2'), 'cyclic'),
    (_dependent('first', 'first', 'result = 1'), 'cyclic'),
    (TUTORIAL + _dependent('second', 'basic, basic', 'result = 1'), 'duplicate dependency'),
    (TUTORIAL + _dependent('second', 'basic,', 'result = 1'), 'malformed example-requires'),
    (TUTORIAL + _dependent('second', 'basic', 'result = 1').replace('<!-- example: second', 'explanation\n<!-- example: second'), 'immediately precede'),
])
def test_ambiguous_or_incomplete_example_structure_fails_loudly(text, reason):
    with pytest.raises(AssertionError, match=reason):
        parse_examples(text)


def test_legacy_single_fence_requires_explicit_opt_in():
    text = '<!-- api-example: old -->\n```python\nresult = 7\n```\n'
    with pytest.raises(AssertionError, match='missing end'):
        parse_examples(text)
    assert run_example(text, 'old', allow_legacy=True)['result'] == 7
    assert not parse_examples(text, allow_legacy=True)['old'].explicit_end


def test_removing_a_dependency_does_not_inject_its_data():
    text = TUTORIAL + _dependent('second', 'basic', 'total = sum(values)')
    text = text.replace('<!-- example-requires: basic -->\n', '')
    with pytest.raises(NameError, match='values'):
        run_example(text, 'second')
