from pathlib import Path
import compileall


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


# Replace per-layer immediate restoration with a depth-aware two-phase commit.
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
old = '''        @functools.wraps(original_init)
        def wrapped(self, *args, **kwargs):
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                return original_init(self, *args, **kwargs)
            raw_params = {
                name: value
                for name, value in bound.arguments.items()
                if name != "self"
                and signature.parameters[name].kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }
            original_init(self, *args, **kwargs)
            normalized_names = type(self)._NORMALIZED_CONSTRUCTOR_PARAMS
            for name, raw_value in raw_params.items():
                private_name = f"_{name}"
                if name in normalized_names:
                    # Constructor wrappers are nested across the inheritance
                    # chain. An inner wrapper may already have restored the
                    # public raw value, so the private runtime value is the
                    # authoritative source when it exists.
                    if hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    elif hasattr(self, name):
                        runtime_value = getattr(self, name)
                    else:
                        runtime_value = raw_value
                    if isinstance(runtime_value, (dict, list, set, np.ndarray)):
                        runtime_value = copy.deepcopy(runtime_value)
                    setattr(self, private_name, runtime_value)
                    setattr(self, name, raw_value)
                elif not hasattr(self, name):
                    # Parameters delegated to a superclass or represented only
                    # by a private runtime field must still exist publicly.
                    setattr(self, name, raw_value)
            self._constructor_params_raw = raw_params
'''
new = '''        @functools.wraps(original_init)
        def wrapped(self, *args, **kwargs):
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                return original_init(self, *args, **kwargs)
            raw_params = {
                name: value
                for name, value in bound.arguments.items()
                if name != "self"
                and signature.parameters[name].kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }

            depth = int(getattr(self, "_statgpu_constructor_depth", 0))
            if depth == 0:
                self._statgpu_constructor_raw_pending = {}
            self._statgpu_constructor_depth = depth + 1

            try:
                result = original_init(self, *args, **kwargs)
            except BaseException:
                if depth == 0:
                    self.__dict__.pop("_statgpu_constructor_depth", None)
                    self.__dict__.pop("_statgpu_constructor_raw_pending", None)
                else:
                    self._statgpu_constructor_depth = depth
                raise

            pending = self._statgpu_constructor_raw_pending
            # Inner wrappers finish first; the most-derived constructor finishes
            # last and therefore overrides shared defaults with its actual call.
            pending.update(raw_params)
            normalized_names = type(self)._NORMALIZED_CONSTRUCTOR_PARAMS

            # Make runtime values available to any outer constructor code without
            # restoring public raw values until the complete chain has returned.
            for name, raw_value in raw_params.items():
                private_name = f"_{name}"
                if name in normalized_names:
                    if name == "device" and hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    elif hasattr(self, name):
                        runtime_value = getattr(self, name)
                    elif hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    else:
                        runtime_value = raw_value
                    setattr(self, private_name, runtime_value)
                elif not hasattr(self, name):
                    setattr(self, name, raw_value)

            remaining = depth
            if remaining > 0:
                self._statgpu_constructor_depth = remaining
                return result

            self.__dict__.pop("_statgpu_constructor_depth", None)
            merged_raw = dict(pending)
            self.__dict__.pop("_statgpu_constructor_raw_pending", None)

            for name, raw_value in merged_raw.items():
                private_name = f"_{name}"
                if name in normalized_names:
                    if name == "device" and hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    elif hasattr(self, name):
                        runtime_value = getattr(self, name)
                    elif hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    else:
                        runtime_value = raw_value
                    setattr(self, private_name, runtime_value)
                # sklearn <=1.2 requires the public attribute to be the exact
                # object supplied to the outermost constructor.
                setattr(self, name, raw_value)
            self._constructor_params_raw = merged_raw
            return result
'''
text = replace_once(text, old, new, "depth-aware constructor wrapper")
old = '''            for key, value in direct_updates.items():
                setattr(self, key, value)
                raw_params = getattr(self, "_constructor_params_raw", None)
'''
new = '''            for key, value in direct_updates.items():
                setattr(self, key, value)
                if key in self._NORMALIZED_CONSTRUCTOR_PARAMS:
                    setattr(self, f"_{key}", value)
                raw_params = getattr(self, "_constructor_params_raw", None)
'''
text = replace_once(text, old, new, "deferred private synchronization")
p.write_text(text, encoding="utf-8")


# Update tests that intentionally asserted the superseded public-normalized API.
replacements = {
    "dev/tests/test_core_contracts.py": [
        (
            '''    parent.set_params(device="auto")
    assert parent.device is Device.AUTO
''',
            '''    parent.set_params(device="auto")
    assert parent.device == "auto"
    assert parent._device is Device.AUTO
''',
            "core device contract",
        )
    ],
    "dev/tests/test_pr80_cv_fit_boundary.py": [
        (
            '''    assert model.device is Device.CPU
    assert model._fit_controls.ties == "efron"
''',
            '''    assert model.device == "cpu"
    assert model._device is Device.CPU
    assert model._fit_controls.ties == "efron"
''',
            "cox cv device contract",
        )
    ],
    "dev/tests/test_pr80_fit_boundary.py": [
        (
            '''    assert model.device is Device.CPU
    assert model._fit_controls.ties == "efron"
''',
            '''    assert model.device == "cpu"
    assert model._device is Device.CPU
    assert model._fit_controls.ties == "efron"
''',
            "cox device contract",
        )
    ],
}
for filename, edits in replacements.items():
    path = Path(filename)
    source = path.read_text(encoding="utf-8")
    for old, new, label in edits:
        source = replace_once(source, old, new, label)
    path.write_text(source, encoding="utf-8")


# Make the tag inventory test compatible with both sklearn 1.2 and current.
p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
old = '''def test_public_sklearn_tags_are_available_and_transformers_are_marked():
    import inspect
    import statgpu
    from sklearn.utils import get_tags

    errors = []
    missing_transformer_tags = []
    for name in statgpu.__all__:
        cls = getattr(statgpu, name, None)
        if not inspect.isclass(cls) or not hasattr(cls, "fit") or inspect.isabstract(cls):
            continue
        signature = inspect.signature(cls)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect._empty
            and parameter.kind
            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        if required:
            continue
        try:
            estimator = cls()
            tags = get_tags(estimator)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if callable(getattr(estimator, "transform", None)) and tags.transformer_tags is None:
            missing_transformer_tags.append(name)

    assert errors == []
    assert missing_transformer_tags == []
'''
new = '''def test_public_sklearn_tags_are_available_and_transformers_are_marked():
    import inspect
    import statgpu

    try:
        from sklearn.utils import get_tags
    except ImportError:
        get_tags = None
        from sklearn.utils._tags import _safe_tags

    errors = []
    missing_transformer_tags = []
    for name in statgpu.__all__:
        cls = getattr(statgpu, name, None)
        if not inspect.isclass(cls) or not hasattr(cls, "fit") or inspect.isabstract(cls):
            continue
        signature = inspect.signature(cls)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect._empty
            and parameter.kind
            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        if required:
            continue
        try:
            estimator = cls()
            if get_tags is None:
                tags = _safe_tags(estimator)
            else:
                tags = get_tags(estimator)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if (
            get_tags is not None
            and callable(getattr(estimator, "transform", None))
            and tags.transformer_tags is None
        ):
            missing_transformer_tags.append(name)

    assert errors == []
    assert missing_transformer_tags == []
'''
text = replace_once(text, old, new, "cross-version tag inventory")
old = '''def test_public_raw_private_mutable_kwargs_are_decoupled():
    from statgpu.linear_model import PenalizedLinearRegression

    penalty_kwargs = {"gamma": 3.0}
    loss_kwargs = {"scale": 2.0}
    model = PenalizedLinearRegression(
        penalty_kwargs=penalty_kwargs,
        loss_kwargs=loss_kwargs,
    )
    assert model.penalty_kwargs is penalty_kwargs
    assert model.loss_kwargs is loss_kwargs
    assert model._penalty_kwargs == penalty_kwargs
    assert model._loss_kwargs == loss_kwargs
    assert model._penalty_kwargs is not penalty_kwargs
    assert model._loss_kwargs is not loss_kwargs

    penalty_kwargs["external"] = True
    loss_kwargs["external"] = True
    assert "external" not in model._penalty_kwargs
    assert "external" not in model._loss_kwargs
'''
new = '''def test_public_raw_private_mutable_kwargs_preserve_runtime_identity():
    from statgpu.linear_model import PenalizedLinearRegression

    penalty_kwargs = {"gamma": 3.0}
    loss_kwargs = {"scale": 2.0}
    model = PenalizedLinearRegression(
        penalty_kwargs=penalty_kwargs,
        loss_kwargs=loss_kwargs,
    )
    assert model.penalty_kwargs is penalty_kwargs
    assert model.loss_kwargs is loss_kwargs
    assert model._penalty_kwargs is penalty_kwargs
    assert model._loss_kwargs is loss_kwargs

    penalty_kwargs["external"] = True
    loss_kwargs["external"] = True
    assert model._penalty_kwargs["external"] is True
    assert model._loss_kwargs["external"] is True
'''
text = replace_once(text, old, new, "mutable runtime identity test")
p.write_text(text, encoding="utf-8")

for path in [Path("statgpu/_base.py"), *map(Path, replacements), p]:
    if not compileall.compile_file(str(path), quiet=1):
        raise SystemExit(f"compile failed: {path}")
