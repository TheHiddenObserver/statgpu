from pathlib import Path
import compileall


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
old = '''    def __sklearn_clone__(self):
        """Return an unfitted estimator clone for scikit-learn >= 1.3.

        Several statgpu constructors validate or canonicalize immutable strings
        and copy mutable dictionaries. scikit-learn's legacy identity check
        treats those defensive copies as constructor mutation. The explicit
        clone protocol preserves the public constructor values while discarding
        fitted state.
        """
        from copy import deepcopy

        return type(self)(**deepcopy(self.get_params(deep=False)))
'''
new = '''    def __sklearn_clone__(self):
        """Return an unfitted recursive clone for scikit-learn >= 1.3.

        Constructor values are preserved by the public raw-parameter contract,
        while estimator-valued parameters must be cloned recursively so fitted
        state is never copied into the new estimator.
        """
        from copy import deepcopy

        params = self.get_params(deep=False)
        try:
            from sklearn.base import clone as sklearn_clone
        except ImportError:
            cloned_params = deepcopy(params)
        else:
            cloned_params = {
                name: sklearn_clone(value, safe=False)
                for name, value in params.items()
            }
        return type(self)(**cloned_params)
'''
text = replace_once(text, old, new, "recursive sklearn clone")
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_core_contracts.py")
text = p.read_text(encoding="utf-8")
anchor = '''def test_torch_rng_none_uses_entropy(monkeypatch):
'''
insert = '''def test_sklearn_clone_recursively_clears_nested_fitted_state():
    from sklearn.base import clone

    child = DummyEstimator(value=3).fit(np.ones((2, 1)))
    parent = DummyEstimator(value=5, child=child)

    cloned = clone(parent)

    assert cloned is not parent
    assert cloned.child is not child
    assert cloned.child.value == 3
    assert cloned.child._fitted is False
    assert child._fitted is True


def test_torch_rng_none_uses_entropy(monkeypatch):
'''
text = replace_once(text, anchor, insert, "nested clone regression")
p.write_text(text, encoding="utf-8")

for path in ("statgpu/_base.py", "dev/tests/test_core_contracts.py"):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f"compile failed: {path}")
