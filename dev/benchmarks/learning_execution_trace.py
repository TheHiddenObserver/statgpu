"""Scoped diagnostic observation of selected numerical operations, not a GPU profiler.

Only source-identified stage invocations and calls to non-allocation numerical
primitives count. Each invocation must meet its own requirement. Observations do
not prove that unobserved operators or the whole algorithm ran on the GPU.
"""
from contextlib import ExitStack
from functools import wraps
import sys
import inspect
import types
import ast
import dis
import textwrap
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from dev.benchmarks.benchmark_release_refresh import array_identity


class TraceIncomplete(RuntimeError):
    pass


def identities(value):
    if isinstance(value, dict):
        return [item for v in value.values() for item in identities(v)]
    if isinstance(value, (list, tuple)):
        return [item for v in value for item in identities(v)]
    import numpy as np
    if isinstance(value, np.generic):
        # A NumPy scalar can be a legitimate scalar argument to a native
        # primitive (e.g. CuPy slogdet's where fill values). Retain its host
        # identity, but never confuse it with a host ndarray or array evidence.
        return [{'backend': 'numpy', 'device': 'cpu', 'dtype': str(value.dtype),
                 'kind': 'numpy-scalar'}]
    try:
        return [dict(array_identity(value), kind='array', shape=list(value.shape))]
    except (RuntimeError, AttributeError, TypeError):
        return []


def stage(fn, label, operations=(), minimum_calls=1, placement='requested-backend',
          reject_host_numerics=False, host_controls=None, expected_dtype=None,
          expected_backend=None, intermediates=(), operation_counts=None, operator_sites=(),
          control_outputs=None, executed_assignments=None):
    # Estimator validation decorators share one wrapper code object. Resolve
    # their implementation, or unrelated decorated methods could satisfy a
    # method-specific stage. This never unwraps native primitive provenance.
    try:
        source = inspect.unwrap(fn)
        code = source.__code__
    except (ValueError, AttributeError, TypeError) as exc:
        raise TraceIncomplete('Cannot resolve source stage: ' + label) from exc
    return {'code': code, 'label': label, 'operations': tuple(operations),
            'minimum_calls': minimum_calls, 'placement': placement,
            'reject_host_numerics': reject_host_numerics,
            'host_controls': host_controls or {}, 'expected_dtype': expected_dtype,
            'expected_backend': expected_backend, 'intermediates': tuple(intermediates),
            'operation_counts': operation_counts or {},
            'operator_sites': _operator_sites(source, operator_sites),
            'control_outputs': control_outputs or {},
            'assignment_sites': _assignment_sites(source, executed_assignments or {})}


def _operator_sites(fn, contracts):
    """Resolve exact source @ sites without rewriting or re-executing math.

    Only simple named operands (and their transpose/fitted attributes) are
    supported. Bytecode positions must uniquely identify the source expression.
    Unsupported interpreters fail closed, rather than treating return ownership
    as arithmetic. The recorded native slots execute with original operands.
    """
    if not contracts:
        return {}
    source, start = inspect.getsourcelines(fn)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    sites = {}
    for expression, output in contracts:
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)
                   and isinstance(node.op, ast.MatMult) and ast.unparse(node) == expression]
        if len(matches) != 1:
            raise TraceIncomplete('Cannot resolve unique numerical operator: ' + expression)
        node = matches[0]
        instructions = []
        current_line = None
        for op in dis.get_instructions(fn):
            current_line = op.starts_line if op.starts_line is not None else current_line
            position = getattr(op, 'positions', None)
            line = position.lineno if position is not None else current_line
            if (op.opname == 'BINARY_MATRIX_MULTIPLY'
                or (op.opname == 'BINARY_OP' and op.argrepr == '@')) and line == start + node.lineno - 1:
                instructions.append(op)
        if len(instructions) != 1:
            raise TraceIncomplete('Cannot resolve executed operator bytecode: ' + expression)
        paths = []
        for operand in (node.left, node.right):
            # Transpose is a native array view. The underlying array has the
            # same device/dtype and we do not create an extra view while tracing.
            if isinstance(operand, ast.Attribute) and operand.attr == 'T':
                operand = operand.value
            if isinstance(operand, ast.Name):
                paths.append((operand.id,))
            elif isinstance(operand, ast.Attribute) and isinstance(operand.value, ast.Name):
                paths.append((operand.value.id, operand.attr))
            else:
                raise TraceIncomplete('Unsupported operator operand: ' + expression)
        sites[instructions[0].offset] = {'expression': expression, 'paths': paths,
                                         'output': output, 'line': start + node.lineno - 1}
    return sites


def _assignment_sites(fn, requirements):
    """Count one exact numerical loop site, not ancestor operation totals."""
    result = {}
    for local, expected in requirements.items():
        instructions = [op for op in dis.get_instructions(fn)
                        if op.opname == 'STORE_FAST' and op.argval == local]
        if len(instructions) != 1:
            raise TraceIncomplete('Cannot resolve unique numerical assignment: ' + local)
        result[instructions[0].offset] = {'local': local, 'expected_count': expected}
    return result


class NumericalTrace:
    """Process-local observer; not thread-safe and rejects an existing profiler.

    Native operations are called unchanged, with original arguments. No arrays
    are transferred, values read, or operations synchronized by the observer.
    """
    OPS = ('exp', 'sum', 'mean', 'abs', 'sqrt', 'quantile', 'std', 'var',
           'dot', 'matmul', 'correlate', 'bincount', 'cumsum', 'searchsorted', 'argsort', 'rand',
           'where', 'min', 'max', 'log', 'argmin', 'diff', 'einsum', 'tanh',
           'maximum', 'minimum', 'clip', 'argmax', 'nonzero', 'topk', 'randint',
           'sin', 'cos', 'floor', 'argpartition', 'divide')
    LINALG = ('inv', 'solve', 'eigh', 'eigvalsh', 'qr', 'svd', 'det',
              'solve_ex', 'cholesky', 'solve_triangular', 'norm')

    def __init__(self, runtime, stages):
        self.runtime = runtime
        self.stages = stages
        self.by_code = {s['code']: s for s in stages}
        self.active = {}
        self.frames = {}
        self.records = []
        self._constant_arrays = {}
        self._stack = ExitStack()
        self._needs_opcodes = any(s['operator_sites'] or s['assignment_sites'] for s in stages)

    def _profile(self, frame, event, arg):
        spec = self.by_code.get(frame.f_code)
        if spec is None:
            return
        if event == 'call':
            record = {'stage': spec['label'], 'source': frame.f_code.co_filename,
                      'line': frame.f_code.co_firstlineno, 'operations': [],
                      'completed': False, 'placement': spec['placement'],
                      'expected_floating_dtype': spec['expected_dtype'] or getattr(self.runtime, 'dtype', None),
                      'expected_backend': spec['expected_backend'] or self.runtime.name,
                      'intermediates': {}, 'operator_evidence': [],
                      'executed_assignments': {site['local']: 0 for site in spec['assignment_sites'].values()}}
            self.active[id(frame)] = record
            self.frames[id(frame)] = frame
            self.records.append(record)
        elif event == 'return':
            record = self.active.pop(id(frame), None)
            self.frames.pop(id(frame), None)
            if record is not None:
                record['completed'] = arg is not None
                for field in spec['intermediates']:
                    record['intermediates'][field] = identities(frame.f_locals.get(field))
                for event in record['operator_evidence']:
                    result = arg if event['output_local'] == '$return' else frame.f_locals.get(event['output_local'])
                    event['outputs'] = identities(result)
                    record['operations'].append(event)
                for key, constant in list(self._constant_arrays.items()):
                    if constant[1] == id(frame):
                        self._constant_arrays.pop(key)

    def _trace_operator(self, frame, event, arg):
        spec = self.by_code.get(frame.f_code)
        if spec is None or not (spec['operator_sites'] or spec['assignment_sites']):
            return None
        frame.f_trace_opcodes = True
        frame.f_trace_lines = False
        record = self.active.get(id(frame))
        if record and event == 'opcode' and frame.f_lasti in spec['assignment_sites']:
            local = spec['assignment_sites'][frame.f_lasti]['local']
            record['executed_assignments'][local] += 1
        if record and record['operator_evidence']:
            pending = record['operator_evidence'][-1]
            if pending['status'] == 'pending':
                if event == 'exception':
                    pending['status'] = 'failed'
                elif event == 'opcode':
                    pending['status'] = 'completed'
        if event == 'opcode' and frame.f_lasti in spec['operator_sites']:
            site = spec['operator_sites'][frame.f_lasti]
            record = self.active.get(id(frame))
            if record is None:
                return self._trace_operator
            arrays = []
            valid = True
            for path in site['paths']:
                value = frame.f_locals.get(path[0])
                for attr in path[1:]:
                    value = getattr(value, attr, None)
                ident = identities(value)
                arrays.extend(ident)
                # Reject subclasses and foreign Python slots, including a
                # tensor-looking CPU-and-recast replacement of __matmul__.
                backend = ident[0]['backend'] if len(ident) == 1 else None
                module = sys.modules.get(backend)
                native_type = getattr(module, 'Tensor' if backend == 'torch' else 'ndarray', None)
                descriptor = getattr(native_type, '__matmul__', None)
                valid &= (type(value) is native_type and isinstance(descriptor,
                    (types.WrapperDescriptorType, types.MethodDescriptorType)))
            record['operator_evidence'].append({
                'operation': 'operator.matmul', 'backend': arrays[0]['backend'] if arrays else 'unknown',
                'inputs': arrays, 'outputs': [], 'status': 'pending' if valid else 'unverifiable',
                'evidence_kind': 'executed-source-opcode-with-exact-native-operand-slots',
                'source_expression': site['expression'], 'line': site['line'],
                'output_local': site['output']})
        return self._trace_operator

    @staticmethod
    def _mutation_version(value):
        # Torch's host-side counter detects normal version-tracked in-place
        # mutation, not arbitrary .data bypasses. It reads no tensor values and
        # does not synchronize CUDA. A missing counter fails closed.
        try:
            return value._version
        except (AttributeError, RuntimeError):
            return None

    def _wrap(self, fn, backend, name, mutated_argument=None):
        @wraps(fn)
        def observed(*args, **kwargs):
            records = tuple(self.active.items())
            if not records:
                return fn(*args, **kwargs)
            inputs = identities((args, kwargs))
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                for _, record in records:
                    record['operations'].append({'operation': name, 'backend': backend,
                        'inputs': inputs, 'outputs': [], 'status': 'failed',
                        'exception': exception_chain(exc)})
                raise
            outputs = identities(args[mutated_argument] if mutated_argument is not None else result)
            # Resolve metadata allowances once against exact source-local
            # objects; propagate that narrow classification to ancestor stages.
            controls = []
            for frame_id, record in records:
                frame = self.frames[frame_id]
                spec = self.by_code[frame.f_code]
                for field in spec['host_controls'].get(name, ()):
                    if backend == 'numpy' and args and args[0] is frame.f_locals.get(field):
                        controls.append({'stage': spec['label'], 'field': field})
            dtype_controls = []
            dtype_input_controls = []
            for frame_id, record in records:
                spec = self.by_code[self.frames[frame_id].f_code]
                control = spec['control_outputs'].get(name)
                caller = sys._getframe(1)
                caller_matches = bool(control)
                for via in control.get('via', ()) if control else ():
                    if caller is None or caller.f_code is not via.__code__:
                        caller_matches = False
                        break
                    caller = caller.f_back
                caller_matches = caller_matches and caller is not None and caller.f_code is spec['code']
                if control and caller_matches and control.get('allow_control_inputs') and name == 'where':
                    for argument_index, argument in enumerate(args):
                        prior = self._constant_arrays.get(id(argument))
                        if (argument_index in (1, 2) and prior is not None
                            and prior[0] is argument and prior[1] == frame_id
                            and prior[3] is not None and self._mutation_version(argument) == prior[3]):
                            dtype_input_controls.append(dict(prior[2],
                                input_index=len(identities(args[:argument_index]))))
                if (control and caller_matches and backend == 'torch' and len(args) == 3
                    and type(args[1]) is float and type(args[2]) is float
                    and (args[1], args[2]) == tuple(control['values'])
                    and len(inputs) == 1 and inputs[0].get('dtype') == 'bool'
                    and len(outputs) == 1 and outputs[0].get('dtype') == control['dtype']):
                    dtype_controls.append({'stage': spec['label'], 'dtype': control['dtype'],
                        'values': list(control['values']), 'role': 'exact-branch-constants',
                        'source': caller.f_code.co_filename, 'line': caller.f_lineno})
                    if control.get('allow_control_inputs'):
                        # Keep the exact returned constant tensor alive only for
                        # this observation. A copied/recast/same-shaped tensor
                        # cannot inherit its narrowly scoped input permission.
                        self._constant_arrays[id(result)] = (result, frame_id, dtype_controls[-1],
                                                            self._mutation_version(result))
            # Allocation/copy and scalar-only calls never establish arithmetic.
            for _, record in records:
                event = {'operation': name, 'backend': backend,
                         'inputs': inputs, 'outputs': outputs, 'status': 'completed'}
                if dtype_controls:
                    event['floating_dtype_controls'] = dtype_controls
                if dtype_input_controls:
                    event['floating_dtype_input_controls'] = dtype_input_controls
                if controls:
                    event['host_control'] = controls[0]['field']
                    event['host_control_sources'] = controls
                record['operations'].append(event)
            return result
        # NumPy reductions use np.minimum.reduce / np.maximum.reduce.
        # Preserve the native ufunc protocol rather than replacing it with an
        # ordinary function lacking methods. These forwarded methods remain
        # outside observation unless their enclosing primitive is registered.
        for attribute in ('reduce', 'accumulate', 'reduceat', 'outer', 'at',
                          'types', 'nin', 'nout', 'nargs', 'signature', 'identity',
                          'ntypes', 'resolve_dtypes'):
            if hasattr(fn, attribute):
                setattr(observed, attribute, getattr(fn, attribute))
        return observed

    @staticmethod
    def _numpy_generated_wrapper(fn, module):
        # NumPy <=1.24 uses a generated Python __array_function__ wrapper.
        # Recognize the actual decorator protocol, not merely its filename or
        # functools __wrapped__ attribute, and verify closure implementations.
        code=getattr(fn, '__code__', None)
        if code is None or code.co_filename != '<__array_function__ internals>':
            return False
        from numpy.core import overrides
        if fn.__globals__ is not vars(overrides):
            return False
        def dispatcher(x):return (x,)
        def implementation(x):return x
        template=overrides.array_function_dispatch(dispatcher)(implementation)
        reference=getattr(template, '__code__', None)
        fields=('co_code','co_consts','co_names','co_freevars','co_argcount','co_kwonlyargcount','co_flags')
        if reference is None or any(getattr(code,k)!=getattr(reference,k) for k in fields):
            return False
        closure=dict(zip(code.co_freevars, [c.cell_contents for c in fn.__closure__ or ()]))
        if closure.get('public_api') is not fn or closure.get('implementation') is not getattr(fn,'_implementation',None):
            return False
        for name in ('implementation','dispatcher'):
            candidate=closure.get(name)
            if candidate is None:
                return False
            NumericalTrace._verify_native_callable(candidate,module,'numpy',name)
        return True

    @staticmethod
    def _verify_native_callable(fn, module, backend, name):
        # Do not label an arbitrary Python replacement as native merely because
        # it occupies torch.exp/cupy.sum. Python library implementations must
        # originate inside the installed library; C primitives identify their
        # library via callable/type module. No unwrap: wrappers are not native.
        if backend=='numpy' and NumericalTrace._numpy_generated_wrapper(fn,module):
            return
        code = getattr(fn, '__code__', None)
        if code is None:
            code = getattr(getattr(type(fn), '__call__', None), '__code__', None)
        if code is not None:
            root = Path(module.__file__).resolve().parent
            source = Path(code.co_filename).resolve()
            if root not in source.parents:
                raise TraceIncomplete('Untrusted replacement of native primitive: ' + backend + '.' + name)
        else:
            kind=type(fn)
            if kind in (types.MethodDescriptorType, types.WrapperDescriptorType):
                owner = getattr(fn, '__objclass__', None)
                if backend == 'torch' and getattr(owner, '__module__', '').startswith('torch'):
                    return
                raise TraceIncomplete('Unverifiable native method: ' + backend + '.' + name)
            recognized=(kind is types.BuiltinFunctionType or
                        (backend=='numpy' and kind.__module__=='numpy' and kind.__name__ in ('ufunc','_ArrayFunctionDispatcher')) or
                        (backend=='cupy' and kind.__module__.startswith('cupy.') and kind.__name__=='ufunc'))
            if not recognized:
                raise TraceIncomplete('Unverifiable native callable type: ' + backend + '.' + name)
            origin = getattr(fn, '__module__', '') or type(fn).__module__
            if not (origin == backend or origin.startswith(backend + '.')):
                raise TraceIncomplete('Unverifiable native primitive: ' + backend + '.' + name)

    def __enter__(self):
        if sys.getprofile() is not None:
            raise TraceIncomplete('Existing profiler prevents isolated numerical trace')
        if sys.gettrace() is not None and self._needs_opcodes:
            raise TraceIncomplete('Existing trace function prevents isolated operator observation')
        import numpy as np
        modules = [('numpy', np)]
        if self.runtime.name != 'numpy':
            modules.append((self.runtime.name, self.runtime.module))
        try:
            for backend, module in modules:
                for name in dict.fromkeys(self.OPS):
                    fn = getattr(module, name, None)
                    if callable(fn):
                        self._verify_native_callable(fn, module, backend, name)
                        self._stack.enter_context(patch.object(module, name, self._wrap(fn, backend, name)))
                for name in self.LINALG:
                    fn = getattr(module.linalg, name, None)
                    if callable(fn):
                        self._verify_native_callable(fn, module, backend, 'linalg.' + name)
                        self._stack.enter_context(patch.object(module.linalg, name, self._wrap(fn, backend, 'linalg.' + name)))
            if self.runtime.name == 'torch':
                module = self.runtime.module
                for method in ('__matmul__', 'sum', 'clamp_', 'scatter_reduce_', 'index_add'):
                    name = 'Tensor.' + method
                    if any(name in spec['operations'] for spec in self.stages):
                        fn = getattr(module.Tensor, method)
                        self._verify_native_callable(fn, module, 'torch', name)
                        self._stack.enter_context(patch.object(module.Tensor, method,
                            self._wrap(fn, 'torch', name)))
            if self.runtime.name == 'cupy' and any('add.at' in spec['operations'] for spec in self.stages):
                module = self.runtime.module
                add = module.add
                self._verify_native_callable(add, module, 'cupy', 'add')
                self._verify_native_callable(add.at, module, 'cupy', 'add.at')
                wrapper = self._wrap(add, 'cupy', 'add')
                wrapper.at = self._wrap(add.at, 'cupy', 'add.at', mutated_argument=0)
                self._stack.enter_context(patch.object(module, 'add', wrapper))
            if self.runtime.name == 'cupy' and any('RandomState.random_sample' in spec['operations'] for spec in self.stages):
                module = self.runtime.module
                fn = module.random.RandomState.random_sample
                self._verify_native_callable(fn, module, 'cupy', 'RandomState.random_sample')
                self._stack.enter_context(patch.object(module.random.RandomState, 'random_sample',
                    self._wrap(fn, 'cupy', 'RandomState.random_sample')))
            sys.setprofile(self._profile)
            if self._needs_opcodes:
                # CPython 3.12 enables opcode events globally only when a
                # current frame requests them at settrace time (3.13 reverted
                # this requirement). No user frame flags are modified.
                sys._getframe().f_trace_opcodes = True
                sys.settrace(self._trace_operator)
            return self
        except BaseException:
            sys.setprofile(None)
            if self._needs_opcodes:
                sys.settrace(None)
            self._stack.close()
            raise

    def __exit__(self, *exc):
        sys.setprofile(None)
        if self._needs_opcodes:
            sys.settrace(None)
        self._stack.close()
        self._constant_arrays.clear()
        return False

    def _validate(self):
        for spec in self.stages:
            required_backend = spec['expected_backend'] or self.runtime.name
            required_device = 'cpu' if required_backend == 'numpy' else self.runtime.device
            records = [r for r in self.records if r['stage'] == spec['label']]
            if len(records) < spec['minimum_calls'] or any(not r['completed'] for r in records):
                raise TraceIncomplete('Missing or incomplete numerical stage: ' + spec['label'])
            for record in records:
                observed = Counter()
                for site in spec['assignment_sites'].values():
                    if record.get('executed_assignments', {}).get(site['local']) != site['expected_count']:
                        raise TraceIncomplete('Incomplete numerical loop execution in stage '
                            + spec['label'] + ': expected ' + str(site['expected_count'])
                            + ' assignments to ' + site['local'] + ', observed '
                            + str(record.get('executed_assignments', {}).get(site['local'])))
                for field in spec['intermediates']:
                    values = record.get('intermediates', {}).get(field, [])
                    if not values or any(v.get('kind') != 'array'
                        or v.get('backend') != required_backend or v.get('device') != required_device
                        for v in values):
                        raise TraceIncomplete('Missing/wrong-device numerical intermediate in stage '
                                              + spec['label'] + ': ' + field)
                    expected = spec.get('expected_dtype')
                    allowed = expected if isinstance(expected, (tuple, list)) else (expected,)
                    for value in values:
                        dtype = value.get('dtype', '')
                        if expected and dtype not in allowed and dtype not in ('bool', 'int32', 'int64'):
                            raise TraceIncomplete('Wrong floating dtype in intermediate ' + field)

                for event in record['operations']:
                    if event.get('status') in ('failed', 'unverifiable', 'pending'):
                        raise TraceIncomplete('Failed numerical primitive in stage ' + spec['label']
                                              + ': ' + event['operation'])
                    if event['backend'] != required_backend:
                        if (spec['reject_host_numerics'] and not event.get('host_control')
                                and any(a.get('kind') == 'array' for a in event['inputs'])):
                            raise TraceIncomplete('Host/wrong-backend array arithmetic in stage '
                                                  + spec['label'] + ': ' + event['operation'])
                        continue
                    matches = lambda item: (item['backend'] == required_backend and item['device'] == required_device
                        and (item.get('kind') == 'array'
                            or (required_backend == 'numpy' and item.get('kind') == 'numpy-scalar'
                                and 'shape' not in item)))
                    scalar_control = lambda item: (item.get('kind') == 'numpy-scalar'
                        and item['backend'] == 'numpy' and item['device'] == 'cpu'
                        and 'shape' not in item)
                    array_inputs = [item for item in event['inputs'] if not scalar_control(item)]
                    # Only INPUT scalar controls are exempt from placement.
                    # In particular, NumPy scalar outputs and 0D host ndarrays
                    # must still reject a purported GPU-native primitive.
                    if any(not matches(a) for a in array_inputs + event['outputs']):
                        raise TraceIncomplete('Native numerical primitive used wrong backend/device in stage '
                            + spec['label'] + ': operation=' + event['operation']
                            + '; expected=' + required_backend + '/' + required_device
                            + '; inputs=' + repr(event['inputs']) + '; outputs=' + repr(event['outputs']))
                    expected_dtype = spec.get('expected_dtype') or getattr(self.runtime, 'dtype', None)
                    if expected_dtype is not None:
                        import numpy as np
                        expected_dtype = (tuple(str(dt).replace('torch.', '') for dt in expected_dtype)
                            if isinstance(expected_dtype, (tuple, list)) else
                            (str(expected_dtype).replace('torch.', ''),))
                        operands = [(a, 'input', i) for i, a in enumerate(event['inputs']) if not scalar_control(a)]
                        operands += [(a, 'output', i) for i, a in enumerate(event['outputs'])]
                        for item, role, index in operands:
                            dtype = str(item.get('dtype', '')).replace('torch.', '')
                            try:
                                control_dtype = np.dtype(dtype).kind in 'biu'
                            except TypeError:
                                control_dtype = False
                            exact_sign_output = (role == 'output' and event.get('operation') == 'where'
                                and any(control.get('role') == 'exact-branch-constants'
                                        and control.get('dtype') == dtype
                                        and control.get('values') in ([-1.0, 1.0], [1.0, 0.0])
                                        for control in event.get('floating_dtype_controls', [])))
                            exact_constant_input = (role == 'input' and event.get('operation') == 'where'
                                and any(control.get('input_index') == index and control.get('dtype') == dtype
                                        and control.get('values') == [1.0, 0.0]
                                        and control.get('role') == 'exact-branch-constants'
                                        for control in event.get('floating_dtype_input_controls', [])))
                            if not control_dtype and dtype not in expected_dtype and not (exact_sign_output or exact_constant_input):
                                raise TraceIncomplete('Wrong floating dtype in stage ' + spec['label']
                                    + ': operation=' + event['operation'] + '; expected=' + repr(expected_dtype)
                                    + '; observed=' + dtype)
                    if (array_inputs or event['operation'] in ('rand', 'RandomState.random_sample')) and event['outputs']:
                        observed[event['operation']] += 1
                if any(observed[name] < count for name, count in spec['operation_counts'].items()):
                    raise TraceIncomplete('Incomplete numerical primitive count in stage ' + spec['label'])
                if not set(spec['operations']).issubset(observed):
                    raise TraceIncomplete('No matching native arithmetic for stage ' + spec['label'] + ': expected ' + ','.join(spec['operations']))

    def snapshot(self, status='incomplete', reason=None):
        report = {'status': status, 'backend': self.runtime.name, 'device': self.runtime.device,
                'stages': self.records,
                'scope': 'selected native numerical primitives and source-bound operator slots within stage invocations; not whole-algorithm GPU residency',
                'instrumentation': 'Python stage profiler, native function/method wrappers and source-bound opcode observation; untimed preflight only',
                'scalar_controls': 'NumPy scalar inputs retain host identities but do not establish array execution; outputs and all ndarrays remain placement-checked',
                'unobserved': 'unregistered array operator overloads, fused kernels and library internals are not exhaustively traced'}
        if reason is not None:
            report['reason'] = str(reason)
        return report

    def report(self):
        try:
            self._validate()
        except TraceIncomplete as exc:
            exc.execution_trace = self.snapshot(reason=exc)
            raise
        return self.snapshot(status='observed')


def exception_chain(exc):
    """JSON-safe complete cause/context chain, without dropping partial evidence."""
    import traceback
    return {'type': type(exc).__name__, 'message': str(exc),
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}



def learning_stages(case, backend):
    from statgpu.nonparametric.kernel_smoothing import _kde as kde
    from statgpu.nonparametric.kernel_smoothing import _bandwidth_selection as bw
    from statgpu.nonparametric.kernel_smoothing import _kernel_common as common
    from statgpu.feature_selection import _knockoff_utils as ku
    specs = []
    if case in ('bspline_basis', 'natural_spline', 'cyclic_spline', 'thin_plate_spline',
                'smoothing_helpers', 'pairwise_kernels', 'chi2_kernel'):
        return helper_stages(case, backend)
    if case.startswith('knockoff'):
        fn = ku._build_fixed_x_knockoffs if case == 'knockoff_fixed' else ku._build_model_x_knockoffs
        specs = [stage(fn, 'knockoff-construction', ('linalg.solve', 'linalg.eigh')),
                 stage(ku._corr_diff_statistics, 'knockoff-statistic', ('mean', 'abs')),
                 stage(ku._knockoff_threshold_and_path, 'host-threshold', placement='host-report-and-selection')]
    elif case == 'bandwidth_helpers':
        specs = [stage(bw.select_bandwidth, 'bandwidth-input-validation-and-scalar-rule', ('sum',),
                       minimum_calls=2, expected_dtype='float64', reject_host_numerics=True,
                       placement='array-input-validation-with-host-scalar-rule')]
    elif case.startswith('bandwidth_'):
        method = case[len('bandwidth_'):]
        if method == 'cv':
            specs = [stage(bw._kernel_regression_cv_score, 'cv-score', ('exp', 'sum')),
                     stage(bw._golden_section_minimize, 'host-search', placement='host-control-with-native-objective')]
        else:
            specs = [stage(kde.KernelDensityEstimator.fit, 'kde-fit', ('sum',)),
                     stage(bw.select_bandwidth, 'bandwidth-selection', placement='host-selector')]
            if backend != 'numpy':
                from statgpu.nonparametric.kernel_smoothing import _native_bandwidth as native_bw
                specs.append(stage(common._weighted_covariance, 'native-covariance', ('sum',)))
                specs[1] = stage(native_bw.select_bandwidth_native, 'native-bandwidth-selection', ('sum',))
                if method in ('nrd', 'nrd0'):
                    specs.append(stage(native_bw._nrd_factor, 'native-nrd', ('std', 'quantile')))
                elif method not in ('scott', 'silverman'):
                    specs.extend([stage(native_bw._r_factor, 'native-r-selector', ('std', 'quantile', 'bincount')),
                                  stage(native_bw._binned_functional, 'native-selector-objective', ('exp', 'sum')),
                                  stage(native_bw._weighted_quantile_resample_1d, 'native-weighted-selector-resampling', ('argsort', 'cumsum', 'searchsorted'), minimum_calls=0)])
                specs.append(stage(native_bw._project_to_principal_axis, 'native-selector-projection', ('linalg.eigh', 'sum'), minimum_calls=0))
            elif method in ('nrd', 'nrd0'):
                specs.append(stage(common._bandwidth_factor_1d_nrd, 'host-nrd', placement='host-numerical-selector'))
            elif method not in ('scott', 'silverman'):
                specs.append(stage(bw._bandwidth_factor_1d_r_selectors, 'host-r-selector', placement='host-numerical-selector'))
    elif case in ('kde_bootstrap', 'kde_normal_ci'):
        count = 21 if case == 'kde_bootstrap' and backend != 'numpy' else 1
        specs = [stage(kde.KernelDensityEstimator.fit, 'kde-fit', ('sum',), minimum_calls=count),
                 stage(kde.KernelDensityEstimator._evaluate_density, 'kde-density', ('exp',), minimum_calls=count),
                 stage(kde.kde_confidence_interval, 'host-interval-assembly', placement='mixed-density-and-host-report')]
        if case == 'kde_bootstrap' and backend != 'numpy':
            from statgpu.nonparametric.kernel_smoothing import _native_bootstrap as native
            specs[-1] = stage(kde.kde_confidence_interval, 'public-interval-assembly', placement='host-api-report-and-scalar-control')
            specs.extend([
                stage(common._weighted_covariance, 'native-covariance', ('sum',), minimum_calls=count),
                stage(native._probability_cdf, 'native-probability-cdf', ('cumsum',)),
                stage(native._draw_indices, 'native-resampling', ('searchsorted', 'rand' if backend == 'torch' else 'RandomState.random_sample'), minimum_calls=20),
                stage(native._percentiles, 'native-interval-quantiles', ('quantile',)),
            ])
    else:
        raise ValueError('No stage contract for ' + case)
    return specs


def estimator_stages(case, backend, phase):
    """Algorithm-specific contracts for fitted estimators, never field ownership."""
    from statgpu.nonparametric.kernel_smoothing import _kde, _kernel_regression, _kernel_common
    from statgpu.nonparametric.kernel_methods import _kpca, _nystroem, _kernels
    from statgpu.nonparametric.splines import _transformer, _bspline_basis, _penalized
    from statgpu.semiparametric._gam import GAM
    from statgpu.unsupervised._agglomerative import AgglomerativeClustering
    from statgpu.unsupervised._utils import squared_euclidean_distances

    def native(fn, label, operations):
        return stage(fn, label, operations, reject_host_numerics=True, expected_dtype='float64')

    if case in ('kde', 'kernel_regression_nw', 'kernel_regression_local_linear'):
        cls = _kde.KernelDensityEstimator if case == 'kde' else _kernel_regression.KernelRegression
        if phase == 'fit':
            fit = cls._fit_samples if case == 'kde' else cls.fit
            return [native(fit, case + '-fit', ('sum',)),
                    native(_kernel_common._weighted_covariance, 'weighted-covariance', ('sum',))]
        fn = (cls._evaluate_density if case == 'kde' else
              cls._evaluate_nadaraya_watson if case.endswith('_nw') else cls._evaluate_local_linear)
        return [native(fn, case + '-evaluation', ('exp',) if case == 'kde' else ('exp', 'sum'))]
    if case == 'spline':
        return ([native(_transformer.SplineTransformer._build_knots, 'spline-knot-range', ('min', 'max'))]
                if phase == 'fit' else
                [native(_transformer.SplineTransformer._basis_degree, 'spline-de-boor-recurrence', ('where',))])
    if case == 'kernel_pca':
        return ([native(_kpca.KernelPCA.fit, 'kernel-pca-eigensystem', ('mean', 'linalg.eigh', 'sqrt')),
                 native(_kernels.rbf_kernel, 'kernel-pca-training-gram', ('sum',))]
                if phase == 'fit' else
                [native(_kpca.KernelPCA.transform, 'kernel-pca-query-centering', ('mean',)),
                 native(_kernels.rbf_kernel, 'kernel-pca-query-gram', ('sum',))])
    if case == 'nystroem':
        return ([native(_nystroem.Nystroem.fit, 'nystroem-landmark-svd', ('linalg.svd', 'sqrt')),
                 native(_kernels.rbf_kernel, 'nystroem-landmark-gram', ('sum',))]
                if phase == 'fit' else
                [native(_kernels.rbf_kernel, 'nystroem-query-gram', ('sum',))])
    if case == 'gam':
        basis = stage(_bspline_basis.bspline_basis, 'gam-de-boor-recurrence', ('where',),
                      placement='native-basis-with-host-knot-validation-and-scalar-loop-controls',
                      reject_host_numerics=True, expected_dtype='float64',
                      host_controls={'diff': ('knots_np',)})
        if phase == 'fit':
            return [native(GAM._create_knots, 'gam-knot-quantiles', ('quantile',)), basis,
                    native(_penalized.penalized_ls, 'gam-penalized-system', ('linalg.solve',))]
        return [basis]
    if case == 'agglomerative':
        if phase != 'fit':
            return []  # This operation accesses stored host labels, not numerics.
        if backend == 'numpy':
            return [stage(AgglomerativeClustering.fit, 'host-scipy-linkage',
                          placement='host-scipy-linkage-not-native-primitive-observed')]
        return [native(squared_euclidean_distances, 'agglomerative-distances', ('sum', 'matmul')),
                native(AgglomerativeClustering._fit_gpu, 'agglomerative-merge-selection', ('argmin',)),
                stage(AgglomerativeClustering._labels_from_children, 'agglomerative-host-label-assembly',
                      placement='host-index-bookkeeping-and-report')]
    return broad_estimator_stages(case, backend, phase)


def broad_estimator_stages(case, backend, phase):
    """Source-specific contracts for the remaining learning adapters.

    These describe the runner's actual configured branch, not all settings of
    the estimator. Stages that contain real host numerics are observed as such,
    never exempted by case-name placement whitelists.
    """
    from statgpu.nonparametric.kernel_methods import _krr, _krr_cv, _kernels
    from statgpu.unsupervised import (_pca, _incremental_pca, _truncated_svd,
        _kmeans, _minibatch_kmeans, _dbscan, _gmm, _nmf, _minibatch_nmf,
        _tsne, _umap, _nndescent, _utils)

    def native(fn, label, ops, **kwargs):
        return stage(fn, label, ops, reject_host_numerics=True,
                     expected_dtype='float64', **kwargs)

    from statgpu.backends import TorchBackend
    sign_controls = native(_utils.svd_flip_components, 'decomposition-sign-convention',
        ('abs', 'argmax', 'where'),
        control_outputs={'where': {'values': (-1.0, 1.0), 'dtype': 'float32',
                                   'via': (TorchBackend.where,)}})

    if case in ('kernel_ridge', 'kernel_ridge_cv'):
        gram = native(_kernels.rbf_kernel, 'kernel-ridge-gram', ('sum',), intermediates=('K',))
        if phase == 'operation':
            return [gram, native(_krr.KernelRidge.predict, 'kernel-ridge-dual-prediction',
                ('operator.matmul',), intermediates=('K_test', 'y_pred'),
                operator_sites=(('K_test @ self.dual_coef_', 'y_pred'),))]
        specs = [gram, native(_krr.KernelRidge.fit, 'kernel-ridge-regularized-solve',
                             ('linalg.solve',), intermediates=('K_reg',))]
        if case == 'kernel_ridge_cv':
            specs += [native(_krr_cv.KernelRidgeCV.fit, 'kernel-ridge-folds-score-selection-refit',
                ('linalg.eigh', 'mean', 'argmin', 'linalg.solve'),
                operation_counts={'linalg.eigh': 3},
                control_outputs={'where': {'values': (1.0, 0.0), 'dtype': 'float32',
                                           'allow_control_inputs': True}},
                intermediates=('mse_table', 'r2_table', 'dual_coefs', 'residuals'))]
        return specs
    if case in ('pca', 'incremental_pca', 'truncated_svd'):
        cls = {'pca': _pca.PCA, 'incremental_pca': _incremental_pca.IncrementalPCA,
               'truncated_svd': _truncated_svd.TruncatedSVD}[case]
        if phase == 'operation':
            return [native(cls.transform, case + '-projection', ('matmul',))]
        if case == 'pca':
            return [sign_controls, native(cls.fit, 'pca-centered-svd', ('mean', 'linalg.svd', 'sum'),
                           intermediates=('X_centered', 'singular_values_all', 'vh'))]
        if case == 'incremental_pca':
            return [sign_controls, native(cls.partial_fit, 'incremental-pca-batch-svd',
                ('mean', 'linalg.svd', 'sum'), intermediates=('matrix', 'batch_var', 'vh'))]
        return [sign_controls, native(_utils.randomized_svd, 'truncated-svd-randomized-subspace',
                       ('matmul', 'linalg.qr', 'linalg.svd'),
                       intermediates=('omega', 'Q', 'B', 'singular_values')),
                native(_utils.backend_random_normal, 'truncated-svd-projection-rng',
                       ('sin', 'floor', 'log', 'sqrt', 'cos'), intermediates=('u1', 'u2', 'z'))]
    if case in ('kmeans', 'minibatch_kmeans'):
        cls = _kmeans.KMeans if case == 'kmeans' else _minibatch_kmeans.MiniBatchKMeans
        specs = [native(_kmeans.KMeans._squared_distances_with_x_norm,
                        'kmeans-distance-gram', ('sum', 'matmul'), intermediates=('distances',))]
        if phase == 'operation':
            return specs + [native(cls.predict, 'kmeans-label-assignment', ('argmin',))]
        specs += [stage(_kmeans.KMeans._init_centers, 'kmeans-host-probability-initialization',
                        ('sum',), expected_backend='numpy', expected_dtype='float64',
                        placement='host-probability-choice-with-native-distance-work'),
                  stage(_kmeans.KMeans._compute_centers, 'kmeans-centroid-update', ('bincount',),
                        expected_dtype='float64', intermediates=('sums', 'new_centers'),
                        placement='native-centroid-update-with-host-empty-cluster-handling')]
        sums_op = {'numpy': 'operator.matmul', 'torch': 'Tensor.index_add', 'cupy': 'add.at'}[backend]
        sums_operator = (('indicator.T @ X', '$return'),) if backend == 'numpy' else ()
        specs += [native(_kmeans.KMeans._label_sums, 'kmeans-cluster-sufficient-statistics',
                         (sums_op,), operator_sites=sums_operator)]
        if case == 'minibatch_kmeans':
            specs += [native(cls._single_batch_step, 'minibatch-kmeans-online-update',
                ('bincount', 'where'), intermediates=('batch_sums', 'batch_means', 'updated'))]
        return specs
    if case == 'dbscan':
        if phase != 'fit':
            return []  # Stored labels, not numerical execution.
        if backend == 'numpy':
            return [stage(_dbscan.DBSCAN._fit_numpy, 'dbscan-host-neighbor-label-dispatch',
                          expected_backend='numpy', intermediates=('labels', 'core_indices'),
                          placement='host-external-neighbor-search-and-connected-components')]
        if backend == 'torch':
            # Observe native tensor distance, mask reduction and scatter-label
            # primitives as well as source-local graph intermediates. Sparse
            # library internals remain outside this observer's exhaustive scope.
            return [stage(_dbscan.DBSCAN._fit_gpu, 'dbscan-torch-neighbor-graph-propagation',
                          ('Tensor.__matmul__', 'Tensor.sum', 'Tensor.clamp_', 'Tensor.scatter_reduce_', 'nonzero', 'minimum'), expected_dtype='float32',
                          reject_host_numerics=True, intermediates=('X_f32', 'x_norm', 'counts', 'labels_core'),
                          placement='native-float32-neighbors-and-integer-propagation-with-host-report')]
        return [stage(_dbscan.DBSCAN._neighbor_graph_sparse, 'dbscan-cupy-distance-graph',
                      ('sum', 'matmul'), expected_dtype='float32', reject_host_numerics=True,
                      intermediates=('X_f32', 'distances'), placement='native-distances-host-edge-assembly'),
                # Primitive observations are transitive: this outer invocation
                # also contains the child's declared float32 distance arithmetic.
                # Counts and propagated labels are integer arrays, not float64
                # computations. Keep the same strict float32 geometry contract.
                stage(_dbscan.DBSCAN._fit_gpu_cupy, 'dbscan-cupy-counts-host-controlled-propagation',
                      ('bincount',), expected_dtype='float32',
                      intermediates=('counts', 'labels_core'), placement='host-scalar-edge-label-propagation')]
    if case == 'gaussian_mixture':
        specs = [native(_gmm.GaussianMixture._estimate_log_gaussian_prob,
                        'gmm-gaussian-log-density', ('log', 'sum', 'matmul'),
                        intermediates=('quad', 'log_det')),
                 native(_gmm.GaussianMixture._e_step, 'gmm-responsibilities', ('exp', 'mean'),
                        intermediates=('weighted_log_prob', 'log_resp'))]
        if phase == 'fit':
            specs += [native(_gmm.GaussianMixture._m_step, 'gmm-weighted-moments',
                ('sum', 'matmul'), intermediates=('nk', 'means', 'second_moment', 'diag_covariances'))]
        return specs
    if case in ('nmf', 'minibatch_nmf'):
        cls = _nmf.NMF if case == 'nmf' else _minibatch_nmf.MiniBatchNMF
        specs = [native(cls._update_w, case + '-coefficient-update', ('matmul',),
                        intermediates=('numerator', 'denominator'))]
        if phase == 'fit':
            if case == 'nmf':
                specs += [native(cls._update_h, 'nmf-dictionary-update', ('matmul',),
                                 intermediates=('numerator', 'denominator')),
                          native(cls._reconstruction_error, 'nmf-residual-objective', ('matmul', 'sum') if backend == 'numpy' else ('matmul', 'sum', 'sqrt'),
                                 intermediates=('residual',))]
            else:
                specs += [native(cls._batch_stats, 'minibatch-nmf-sufficient-statistics', ('matmul',)),
                          native(cls._update_h_from_stats, 'minibatch-nmf-dictionary-update', ('matmul',),
                                 intermediates=('denominator',)),
                          native(cls._reconstruction_error_from_stats, 'minibatch-nmf-statistics-objective',
                                 ('sum', 'matmul') if backend == 'numpy' else ('sum', 'matmul', 'sqrt'))]
        return specs
    if case == 'tsne':
        if phase != 'fit':
            return []
        return [sign_controls, native(_tsne.TSNE._joint_probabilities, 'tsne-entropy-affinities',
                       ('exp', 'sum', 'log', 'where'), intermediates=('distances', 'beta', 'P')),
                native(_tsne.TSNE.fit, 'tsne-gradient-optimization',
                       ('matmul', 'sum', 'mean', 'where', 'log'),
                       executed_assignments={'grad': 250}, intermediates=('grad', 'Q', 'forces', 'velocity'))]
    if case in ('umap', 'umap_nndescent'):
        if phase != 'fit':
            return []
        # NumPy's eye promotes the exact neighbor distance to float64, and
        # NNDescent explicitly stores float64 distances after float32 arithmetic.
        specs = [stage(_umap.UMAP._smooth_knn_membership, 'umap-neighbor-memberships',
                       ('mean', 'exp'), reject_host_numerics=True,
                       expected_dtype=('float32', 'float64'), intermediates=('adjusted', 'membership')),
                 stage(_umap.UMAP._fuzzy_graph, 'umap-host-fuzzy-graph-union', ('clip',),
                       expected_backend='numpy', expected_dtype=('float32', 'float64'),
                       placement='host-scipy-graph-union-with-native-neighbor-search'),
                 stage(_umap.UMAP._initial_embedding, 'umap-host-spectral-initialization',
                       expected_backend='numpy', expected_dtype='float64',
                       intermediates=('eigenvectors', 'embedding_np'), placement='host-scipy-spectral-eigensolver'),
                 stage(_umap.UMAP._attraction_curve_params, 'umap-host-attraction-curve-fit',
                       ('exp', 'where'), expected_backend='numpy', expected_dtype='float64',
                       intermediates=('xv', 'yv'), placement='host-scipy-nonlinear-curve-fit'),
                 stage(_umap.UMAP.fit, 'umap-layout-gradient', ('sum', 'mean'),
                       expected_dtype=('float32', 'float64'),
                       executed_assignments={'grad': 100},
                       intermediates=('grad_attr', 'grad_rep', 'grad', 'diff_attr', 'diff_rep'),
                       placement='native-layout-with-observed-host-graph-and-initialization')]
        if case == 'umap_nndescent':
            fn = getattr(_nndescent, 'nndescent_' + backend)
            specs += [stage(fn, 'umap-nndescent-neighbor-refinement', ('sum', 'argsort'),
                            reject_host_numerics=True, expected_dtype=('float32', 'float64'),
                            intermediates=('distances', 'dists', 'indices'))]
        else:
            specs += [stage(_utils.squared_euclidean_distances, 'umap-exact-neighbor-distances',
                            ('sum', 'matmul'), reject_host_numerics=True,
                            expected_dtype='float32', intermediates=('distances',))]
        return specs
    raise ValueError('No estimator stage contract for ' + case)


def helper_stages(case, backend):
    from statgpu.nonparametric.splines import _bspline_basis, _cyclic, _thin_plate
    from statgpu.nonparametric.kernel_methods import _kernels
    from statgpu.nonparametric.kernel_smoothing import _kde, _kernel_regression, _kernel_common
    def native(fn, label, ops, **kwargs):
        return stage(fn, label, ops, reject_host_numerics=True, expected_dtype='float64', **kwargs)
    if case == 'smoothing_helpers':
        return [native(_kde.KernelDensityEstimator._fit_samples, 'helper-kde-covariance-fit', ('sum',), minimum_calls=2),
                native(_kde.KernelDensityEstimator._evaluate_density, 'helper-kde-density', ('exp',), minimum_calls=2),
                native(_kernel_regression.KernelRegression.fit, 'helper-regression-fit', ('sum',), minimum_calls=2),
                native(_kernel_regression.KernelRegression._evaluate_nadaraya_watson, 'helper-regression-weights', ('exp', 'sum'), minimum_calls=2),
                native(_kernel_common._weighted_covariance, 'helper-weighted-covariance', ('sum',), minimum_calls=4, intermediates=('centered', 'cov'))]
    if case == 'pairwise_kernels':
        return [native(_kernels.linear_kernel, 'pairwise-linear-dot-products', ('operator.matmul',),
                       operator_sites=(('X @ Y.T', '$return'),)),
                native(_kernels.polynomial_kernel, 'pairwise-polynomial-dot-products', ('operator.matmul',),
                       operator_sites=(('X @ Y.T', '$return'),)),
                native(_kernels.rbf_kernel, 'pairwise-rbf-distances', ('sum',), intermediates=('K',)),
                native(_kernels.laplacian_kernel, 'pairwise-laplacian-distances', ('exp',), intermediates=('dist',)),
                native(_kernels.cosine_kernel, 'pairwise-cosine-normalization', ('sqrt',), intermediates=('X_norm', 'Y_norm')),
                native(_kernels.sigmoid_kernel, 'pairwise-sigmoid-transform', ('tanh',))]
    if case == 'chi2_kernel':
        return [native(_kernels.chi2_kernel, 'chi2-distance-exponentiation',
                       ('exp',) if backend == 'numpy' else ('sum', 'exp'))]
    if case == 'thin_plate_spline':
        return [native(_thin_plate.thin_plate_spline_basis, 'thin-plate-radial-distances',
                       ('sum', 'sqrt'), intermediates=('diff', 'r_sq', 'phi'))]
    basis = native(_bspline_basis.bspline_basis, 'basis-de-boor-recurrence', ('where',),
                   intermediates=('B_new', 'w1', 'w2'), host_controls={'diff': ('knots_np',)},
                   placement='native-basis-with-host-knot-validation-and-scalar-loop-controls')
    if case == 'bspline_basis':
        return [basis]
    if case == 'natural_spline':
        return [basis, native(_bspline_basis.natural_cubic_spline_basis, 'natural-boundary-nullspace',
            ('linalg.svd',), intermediates=('C', 'null_space', 'B_natural'), host_controls={'diff': ('knots_np',)})]
    if case == 'cyclic_spline':
        return [basis, stage(_cyclic.cyclic_cubic_spline_basis, 'cyclic-boundary-nullspace',
            ('linalg.svd',), expected_dtype='float64', intermediates=('constraints', 'vt', 'result'),
            placement='host-constraint-normalization-and-native-nullspace-projection')]
    raise ValueError('No helper stage contract for ' + case)


def model_stages(case, backend):
    """Current Cox and panel numerical dispatch, excluding public report copies."""
    from statgpu.survival import _cox_counting
    from statgpu.losses._cox_ph import CoxPartialLikelihoodLoss
    from statgpu.panel import _linalg, _reductions
    if case in ('cox:breslow', 'cox:efron'):
        return [stage(CoxPartialLikelihoodLoss._full_objective_from_eta_backend,
                      'cox-risk-set-likelihood-score-information', ('exp', 'sum', 'cumsum'),
                      reject_host_numerics=True, expected_dtype='float64',
                      placement='native-risk-moments-with-host-event-index-bookkeeping',
                      host_controls={'cumsum': ('counts',), 'searchsorted': ('first_indices',)}),
                stage(_cox_counting._solve, 'cox-newton-system', ('linalg.solve',),
                      reject_host_numerics=True, expected_dtype='float64')]
    if case == 'panel:fama_macbeth':
        solve = 'linalg.solve_ex' if backend == 'torch' else 'linalg.solve'
        return [stage(_reductions._validate_group_reduction_inputs, 'panel-host-group-counts',
                      minimum_calls=0, placement='host-index-bookkeeping',
                      host_controls={'bincount': ('codes_np',)}),
                stage(_linalg.panel_lstsq_gram_certified_batched, 'panel-certified-period-systems',
                      ('matmul', 'linalg.eigvalsh', solve), reject_host_numerics=True, expected_dtype='float64'),
                stage(_linalg._panel_svd, 'panel-svd-fallback', ('linalg.svd',),
                      minimum_calls=0, reject_host_numerics=True, expected_dtype='float64')]
    if case.startswith('panel:'):
        return [stage(_linalg._panel_svd, 'panel-least-squares-svd', ('linalg.svd',),
                      reject_host_numerics=True, expected_dtype='float64')]
    return []
