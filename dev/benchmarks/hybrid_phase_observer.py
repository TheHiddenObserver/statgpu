"""Diagnostic Python stage observer. Never install during performance samples."""
from __future__ import annotations
import sys
import time


class PhaseObserver:
    """Inclusive spans with boundary sync; overlapping spans are NOT additive.

    Transfer hooks observe selected Python conversion helpers, not CUDA memcpy
    events. Their durations include allocation, dispatch and synchronization;
    they are not a bandwidth measurement or an exhaustive transfer census.
    """
    def __init__(self, runtime, specs, transfer_codes=()):
        self.runtime = runtime
        self.specs = {s['code']: s for s in specs}
        self.transfer_codes = set(transfer_codes)
        self.active = {}
        self.records = []
        self.captured = []
        self.sync_seconds = []

    def synchronize(self):
        start = time.perf_counter()
        self.runtime.sync(self.runtime.device)
        self.sync_seconds.append(time.perf_counter() - start)

    def profile(self, frame, event, arg):
        code = frame.f_code
        spec = self.specs.get(code)
        transfer = code in self.transfer_codes
        if spec is None and not transfer:
            return
        if event == 'call':
            self.synchronize()
            self.active[id(frame)] = time.perf_counter()
            # Retain references only. Conversion for the oracle occurs afterwards.
            if code.co_name == '_corr_diff_statistics':
                self.captured.append(tuple(frame.f_locals[k] for k in ('X_std', 'X_knock', 'y')))
        elif event == 'return':
            start = self.active.pop(id(frame), None)
            if start is None:
                return
            self.synchronize()
            self.records.append({'stage': spec['label'] if spec else code.co_name,
                                 'placement': spec['placement'] if spec else 'conversion-helper',
                                 'inclusive_seconds': time.perf_counter() - start,
                                 'kind': 'transfer-helper' if transfer else 'numerical-stage',
                                 'completed': arg is not None})

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('existing profiler: refusing conflicting instrumentation')
        sys.setprofile(self.profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)

    def report(self):
        totals = {}
        for record in self.records:
            item = totals.setdefault(record['stage'], {'calls': 0, 'inclusive_seconds': 0.0})
            item['calls'] += 1
            item['inclusive_seconds'] += record['inclusive_seconds']
        return {'records': self.records, 'stage_totals_nonadditive': totals,
                'accounting': 'inclusive overlapping spans; never sum or subtract from end-to-end',
                'timing_class': 'instrumented diagnostic, with stage-boundary synchronization overhead',
                'transfer_coverage': 'selected _to_numpy/xp_asarray Python helpers plus explicit input/output spans; not exhaustive',
                'unattributed': 'host RNG, unhooked copies, dispatch and other work remain unattributed; no residual inferred',
                'synchronized_device': self.runtime.device,
                'boundary_sync_seconds': self.sync_seconds,
                'boundary_sync_note': 'observer-added waits; overlaps stage spans, not production sync overhead',
                'cpu_bootstrap_limit': 'NumPy inline fast density loop, RNG and quantiles not isolated; interval assembly is inclusive'}
