from pathlib import Path

path = Path('statgpu/panel/_covariance.py')
text = path.read_text(encoding='utf-8')

old_labels = '''def _factorize_1d_labels(values, *, nobs: int, name: str):
    raw = np.asarray(_to_numpy(values))
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or raw.shape[0] != int(nobs):
        raise ValueError(f"{name} must be one-dimensional with length n_samples")
    try:
        labels, codes = np.unique(raw, return_inverse=True)
    except TypeError as exc:
        raise ValueError(f"{name} values must have a deterministic sortable identity") from exc
    return labels, codes.astype(np.int64, copy=False)
'''
new_labels = '''def _factorize_1d_labels(values, *, nobs: int, name: str):
    raw = np.asarray(_to_numpy(values))
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or raw.shape[0] != int(nobs):
        raise ValueError(f"{name} must be one-dimensional with length n_samples")

    invalid = False
    if np.issubdtype(raw.dtype, np.number):
        invalid = bool(np.any(~np.isfinite(raw)))
    elif np.issubdtype(raw.dtype, np.datetime64) or np.issubdtype(raw.dtype, np.timedelta64):
        invalid = bool(np.any(np.isnat(raw)))
    elif raw.dtype.kind == "O":
        for value in raw:
            if value is None:
                invalid = True
                break
            if isinstance(value, (float, np.floating, complex, np.complexfloating)):
                if not bool(np.isfinite(value)):
                    invalid = True
                    break
            if isinstance(value, (np.datetime64, np.timedelta64)) and bool(np.isnat(value)):
                invalid = True
                break
    if invalid:
        raise ValueError(f"{name} must not contain missing or non-finite values")

    try:
        labels, codes = np.unique(raw, return_inverse=True)
    except TypeError as exc:
        raise ValueError(f"{name} values must have a deterministic sortable identity") from exc
    return labels, codes.astype(np.int64, copy=False)
'''
if old_labels not in text:
    raise SystemExit('expected label factorization block not found')
text = text.replace(old_labels, new_labels, 1)

old_bw = '''def _validate_dk_bandwidth(bandwidth, n_periods: int) -> int:
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, np.integer)):
        raise ValueError("Driscoll-Kraay bandwidth must be a non-negative integer or None")
    bandwidth = int(bandwidth)
    if bandwidth < 0:
        raise ValueError("Driscoll-Kraay bandwidth must be a non-negative integer or None")
    return min(bandwidth, max(int(n_periods) - 1, 0))
'''
new_bw = '''def _validate_dk_bandwidth(bandwidth, n_periods: int) -> int:
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, np.integer)):
        raise ValueError("Driscoll-Kraay bandwidth must be a non-negative integer or None")
    bandwidth = int(bandwidth)
    if bandwidth < 0:
        raise ValueError("Driscoll-Kraay bandwidth must be a non-negative integer or None")
    # Do not silently cap an explicit bandwidth at T-1. Bartlett/Parzen use the
    # requested bandwidth in their weight denominator even though only observed
    # lags 1,...,T-1 can contribute; QS uses bandwidth as a smoothing scale over
    # all observed lags. Silent capping changes the requested covariance.
    return bandwidth
'''
if old_bw not in text:
    raise SystemExit('expected DK bandwidth validation block not found')
text = text.replace(old_bw, new_bw, 1)

path.write_text(text, encoding='utf-8')
