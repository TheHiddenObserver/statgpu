from pathlib import Path

path = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = path.read_text(encoding="utf-8")
old = '''    # Metadata is part of the Stage-C physical contract. It should be purely
    # configuration/count information and therefore backend-identical.
    if candidate["covariance_metadata"] != reference["covariance_metadata"]:
        raise AssertionError(
            f"{label}.covariance_metadata mismatch: "
            f"{candidate['covariance_metadata']} != {reference['covariance_metadata']}"
        )
    return differences
'''
new = '''    ref_meta = reference["covariance_metadata"]
    cand_meta = candidate["covariance_metadata"]
    if set(cand_meta) != set(ref_meta):
        raise AssertionError(
            f"{label}.covariance_metadata keys mismatch: "
            f"{sorted(cand_meta)} != {sorted(ref_meta)}"
        )
    for key, expected in ref_meta.items():
        actual = cand_meta[key]
        metric = f"covariance_metadata.{key}"
        if isinstance(expected, float):
            differences[metric] = _scalar_diff(
                actual, expected, rtol=rtol, atol=atol, label=f"{label}.{metric}"
            )
        elif isinstance(expected, list) and any(isinstance(v, float) for v in expected):
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=f"{label}.{metric}")
            differences[metric] = _max_abs(
                np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
            )
        elif actual != expected:
            raise AssertionError(
                f"{label}.{metric}: {actual!r} != {expected!r}"
            )
        else:
            differences[metric] = 0.0
    return differences
'''
if old not in text:
    raise SystemExit("expected physical metadata comparison block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
